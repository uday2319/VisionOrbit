"""Tests for the tool registry and fallback chain (brief §17, §18, §9, §28).

The registry is where "what task" becomes "which tool", so the suite is in five parts:

* **chains** — every tool a task declares is reachable, optical/SAR analysis forms a two-tool chain,
  change VQA aliases change detection, and each chain is ordered by tier;
* **dispatch** — real analyses against the demo scenes, including the input-driven choice between the
  single-image SAR tool and the optical+SAR fusion tool for the one shared task;
* **refusals** — ``OBJECT_DETECTION`` and ``SEGMENTATION`` are recognised but declined with a redirect
  to the capabilities that exist, never a fabricated box or mask (§18), regardless of the inputs;
* **applicability errors** — when the inputs fit no tool in a chain, the primary tool's specific,
  recoverable error is surfaced rather than a generic one;
* **the tier ladder** — driven with stub tools: an unavailable rung is skipped, the most-capable
  available rung runs, a run-time model outage falls through, and an exhausted chain is honest.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.agents.registry import (
    _CAPABILITY_PHRASES,
    ToolRegistry,
    default_registry,
)
from app.agents.tools import ChangeTool, Prepared, Tool, ToolContext, ToolResult, ToolTier
from app.core.errors import (
    ErrorCode,
    ModelUnavailableError,
    UnsupportedTaskError,
    ValidationError,
)
from app.core.types import Modality, QueryTask
from app.geospatial.raster import load_raster
from app.services.confidence import ConfidenceFactor, FactorKind, aggregate


# ---------------------------------------------------------------------------
# Demo scene fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def sar(demo_root):
    return load_raster(
        demo_root / "sar" / "scene_sar_vv.tif", max_edge=None, modality=Modality.SAR
    )


@pytest.fixture(scope="module")
def temporal(demo_root):
    before = load_raster(demo_root / "temporal" / "scene_t1_optical.tif", max_edge=None)
    after = load_raster(demo_root / "temporal" / "scene_t2_optical.tif", max_edge=None)
    return before, after


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Stub tools for the tier-ladder mechanism (no real analysis)
# ---------------------------------------------------------------------------
def _factor(value: float = 0.9) -> ConfidenceFactor:
    return ConfidenceFactor(
        name="evidence", value=value, weight=1.0, reason="stub", kind=FactorKind.CONTRIBUTOR
    )


class _Stub(Tool):
    """A configurable tool for exercising availability, tier order and run-time outage."""

    task = QueryTask.LAND_COVER_ANALYSIS  # class default; each instance overrides as needed

    def __init__(
        self,
        name: str,
        tier: ToolTier,
        *,
        task: QueryTask = QueryTask.LAND_COVER_ANALYSIS,
        available: bool = True,
        outage: bool = False,
        rejects: ErrorCode | None = None,
    ) -> None:
        self.name = name
        self.tier = tier
        self.task = task
        self._available = available
        self._outage = outage
        self._rejects = rejects
        self.ran = False

    def available(self) -> bool:
        return self._available

    def validate_input(self, ctx: ToolContext) -> None:
        if self._rejects is not None:
            raise ValidationError(f"{self.name} rejects these inputs", code=self._rejects)

    def preprocess(self, ctx: ToolContext) -> Prepared:
        return Prepared(rasters=list(ctx.rasters))

    def predict(self, prepared: Prepared) -> dict[str, Any]:
        if self._outage:
            raise ModelUnavailableError(f"{self.name} weights failed to load")
        self.ran = True
        return {"tool": self.name}

    def explain(self, raw: dict[str, Any]) -> list[str]:
        return [f"ran {raw['tool']}"]

    def postprocess(self, raw: dict[str, Any], prepared: Prepared) -> ToolResult:
        return self.build_result(
            answer=f"stub {raw['tool']}",
            data=raw,
            confidence=aggregate([_factor()]),
            evidence=self.explain(raw),
        )


_CTX = ToolContext("q", [])


# ---------------------------------------------------------------------------
# Chains and introspection
# ---------------------------------------------------------------------------
class TestChains:
    def test_every_shipped_tool_task_is_dispatchable(self, registry):
        for task in (
            QueryTask.LAND_COVER_ANALYSIS,
            QueryTask.GROUNDING,
            QueryTask.VQA,
            QueryTask.CAPTIONING,
            QueryTask.CHANGE_DETECTION,
            QueryTask.CHANGE_VQA,
            QueryTask.OPTICAL_SAR_ANALYSIS,
        ):
            assert registry.is_dispatchable(task), task

    def test_single_tool_tasks_map_to_their_one_tool(self, registry):
        expected = {
            QueryTask.LAND_COVER_ANALYSIS: "landcover-spectral-cv",
            QueryTask.GROUNDING: "grounding-spectral-cv",
            QueryTask.VQA: "vqa-landcover-cv",
            QueryTask.CHANGE_DETECTION: "change-cva-cv",
        }
        for task, name in expected.items():
            chain = registry.tools_for(task)
            assert [t.name for t in chain] == [name]

    def test_captioning_is_a_two_rung_chain_learned_model_first(self, registry):
        """Captioning is the one task with a learned rung above the deterministic analyser (§18).

        The adapted EuroSAT ResNet-18 leads because its tier is PREFERRED; the classical
        ``captioning-landcover-cv`` remains registered behind it, which is what makes the learned
        model's absence a degradation rather than an outage.
        """
        chain = registry.tools_for(QueryTask.CAPTIONING)
        assert [t.name for t in chain] == ["satquery-rs-visual-v1", "captioning-landcover-cv"]
        assert chain[0].tier is ToolTier.PREFERRED
        assert chain[1].tier is ToolTier.CLASSICAL

    def test_optical_sar_is_a_two_tool_chain_single_image_tool_first(self, registry):
        chain = registry.tools_for(QueryTask.OPTICAL_SAR_ANALYSIS)
        # Both classical, told apart by inputs; the single-image SAR tool leads the pair fusion tool.
        assert [t.name for t in chain] == ["sar-backscatter-cv", "fusion-decision-cv"]

    def test_change_vqa_has_its_own_chain(self, registry):
        """Change VQA is its own tool, not an alias of change detection.

        The two share one measurement but answer different questions — a report of what differs
        versus an answer to what was asked — so each has to be dispatchable in its own right. While
        VQA was an alias, the registry could not tell the difference and credited the change report
        for answering a question it had not read.
        """
        vqa = registry.tools_for(QueryTask.CHANGE_VQA)
        detection = registry.tools_for(QueryTask.CHANGE_DETECTION)
        assert [t.name for t in vqa] == ["change-vqa-cv"]
        assert [t.name for t in detection] == ["change-cva-cv"]

    def test_change_vqa_falls_back_to_the_change_chain_in_a_partial_registry(self):
        """A registry built without the VQA tool still serves the task rather than dead-ending."""
        partial = ToolRegistry([ChangeTool()])
        assert [t.name for t in partial.tools_for(QueryTask.CHANGE_VQA)] == ["change-cva-cv"]

    def test_chains_are_ordered_by_ascending_tier(self, registry):
        for task in registry.supported_tasks():
            tiers = [t.tier for t in registry.tools_for(task)]
            assert tiers == sorted(tiers)

    def test_unimplemented_tasks_are_unsupported_not_dispatchable(self, registry):
        for task in (QueryTask.OBJECT_DETECTION, QueryTask.SEGMENTATION):
            assert registry.is_unsupported(task)
            assert not registry.is_dispatchable(task)
            assert registry.tools_for(task) == ()

    def test_report_and_unknown_are_neither_dispatchable_nor_unsupported(self, registry):
        # These are handled upstream (report path / router), not by the registry's refusal.
        for task in (QueryTask.REPORT_GENERATION, QueryTask.UNKNOWN):
            assert not registry.is_dispatchable(task)
            assert not registry.is_unsupported(task)

    def test_every_supported_task_has_a_capability_phrase(self, registry):
        """Sync guard: a dispatchable task with no phrase would vanish from every refusal's redirect."""
        for task in registry.supported_tasks():
            assert task in _CAPABILITY_PHRASES

    def test_describe_all_lists_each_tool_once(self, registry):
        described = registry.describe_all()
        names = [d["name"] for d in described]
        # Nine distinct tools, no duplicates: eight deterministic analysers plus the one learned
        # rung, the EuroSAT-adapted scene classifier registered above classical captioning.
        assert len(names) == len(set(names)) == 9
        assert "change-cva-cv" in names
        assert "change-vqa-cv" in names  # change VQA is a tool of its own, not an alias
        assert "satquery-rs-visual-v1" in names
        for d in described:
            assert set(d) == {"task", "name", "tier", "summary"}

    def test_describe_with_contracts_reports_the_learned_checkpoint_additively(self, registry):
        """``runtime`` carries the checkpoint; ``describe()``'s four-key shape stays untouched (§26).

        The learned entry is the only one with a ``runtime`` block, because it is the only tool with
        an artefact on disk to report. Whether it reads LIVE depends on the host, so this asserts the
        shape and the mode vocabulary rather than a fixed value — a checkpoint-free clone must pass.
        """
        entries = {e["name"]: e for e in registry.describe_all_with_contracts()}
        for name, entry in entries.items():
            if name == "satquery-rs-visual-v1":
                continue
            assert "runtime" not in entry, name

        runtime = entries["satquery-rs-visual-v1"]["runtime"]
        assert runtime["model_id"] == "satquery-rs-visual-v1"
        assert runtime["mode"] in {"LIVE", "UNAVAILABLE"}
        if runtime["mode"] == "LIVE":
            assert runtime["checkpoint"].endswith("model.pt")
            assert runtime["checkpoint_present"] is True
            assert runtime["reason"] is None
            assert runtime["architecture"] == "resnet18"
            assert entries["satquery-rs-visual-v1"]["available"] is True
        else:
            # An unavailable rung must say why, so a reviewer can tell a missing checkpoint from a
            # disabled flag rather than seeing an unexplained blank.
            assert runtime["reason"]
            assert entries["satquery-rs-visual-v1"]["available"] is False

    def test_default_registry_is_shared(self):
        assert default_registry() is default_registry()


# ---------------------------------------------------------------------------
# Dispatch against the real demo scenes
# ---------------------------------------------------------------------------
class TestDispatch:
    def test_land_cover_runs_the_classifier(self, registry, optical):
        result = registry.dispatch(
            QueryTask.LAND_COVER_ANALYSIS, ToolContext("classify the land cover", [optical])
        )
        assert isinstance(result, ToolResult)
        assert result.task is QueryTask.LAND_COVER_ANALYSIS
        assert result.tool_name == "landcover-spectral-cv"
        assert "%" in result.answer

    def test_vqa_runs_the_vqa_tool(self, registry, optical):
        result = registry.dispatch(
            QueryTask.VQA, ToolContext("what land cover is in this image", [optical])
        )
        assert result.tool_name == "vqa-landcover-cv"
        assert "%" in result.answer

    def test_single_sar_image_dispatches_to_the_sar_tool(self, registry, sar):
        """The two-tool chain is disambiguated by inputs: one radar image is a SAR analysis."""
        result = registry.dispatch(
            QueryTask.OPTICAL_SAR_ANALYSIS, ToolContext("what does the radar show", [sar])
        )
        assert result.tool_name == "sar-backscatter-cv"

    def test_optical_sar_pair_dispatches_to_the_fusion_tool(self, registry, optical, sar):
        """...and one optical + one SAR image of the same area is a fusion — same task, other tool."""
        result = registry.dispatch(
            QueryTask.OPTICAL_SAR_ANALYSIS,
            ToolContext("combine the optical and radar views", [optical, sar]),
        )
        assert result.tool_name == "fusion-decision-cv"

    def test_change_vqa_dispatches_to_the_change_vqa_tool(self, registry, temporal):
        before, after = temporal
        result = registry.dispatch(
            QueryTask.CHANGE_VQA, ToolContext("what changed between the two dates", [before, after])
        )
        assert result.task is QueryTask.CHANGE_VQA
        assert result.tool_name == "change-vqa-cv"
        # The answer addresses the question and is built from the same measurement, so the change
        # payload is present in full rather than replaced by a bare sentence.
        assert result.data["changed_pixels"] >= 0
        assert result.data["question"]["asked"] == "what changed between the two dates"

    def test_change_vqa_answers_about_the_class_it_was_asked_about(self, registry, temporal):
        before, after = temporal
        result = registry.dispatch(
            QueryTask.CHANGE_VQA,
            ToolContext("did the vegetation increase between the two dates?", [before, after]),
        )
        assert result.data["question"]["interpreted_as"] == "net change in vegetation"
        assert "vegetation" in result.answer.lower()


# ---------------------------------------------------------------------------
# Refusals for recognised-but-unimplemented tasks (§18)
# ---------------------------------------------------------------------------
class TestUnsupportedRefusal:
    def test_object_detection_is_refused_with_a_redirect(self, registry, optical):
        with pytest.raises(UnsupportedTaskError) as ei:
            registry.dispatch(QueryTask.OBJECT_DETECTION, ToolContext("count the ships", [optical]))
        assert ei.value.code == ErrorCode.UNSUPPORTED_TASK
        assert ei.value.recoverable is True
        assert "bounding box" in ei.value.message.lower()
        assert "fabrication" in ei.value.message.lower()  # names why, does not just say no
        assert "What this system can do instead" in ei.value.message

    def test_segmentation_is_refused_with_a_redirect(self, registry, optical):
        with pytest.raises(UnsupportedTaskError) as ei:
            registry.dispatch(QueryTask.SEGMENTATION, ToolContext("segment the image", [optical]))
        assert ei.value.code == ErrorCode.UNSUPPORTED_TASK
        assert "pixel mask" in ei.value.message.lower()
        assert "land-cover classification" in ei.value.message.lower()  # the closest real capability

    def test_refusal_lists_the_actual_capabilities(self, registry, optical):
        with pytest.raises(UnsupportedTaskError) as ei:
            registry.dispatch(QueryTask.OBJECT_DETECTION, ToolContext("detect objects", [optical]))
        # The redirect is built from the real chain, so every supported capability phrase appears.
        for task in registry.supported_tasks():
            assert _CAPABILITY_PHRASES[task] in ei.value.message
        assert ei.value.context["supported"] == [t.value for t in registry.supported_tasks()]

    def test_refusal_precedes_input_handling(self, registry):
        """An unsupported task is declined regardless of the images — it needs none to say no."""
        with pytest.raises(UnsupportedTaskError):
            registry.dispatch(QueryTask.OBJECT_DETECTION, ToolContext("detect objects", []))


# ---------------------------------------------------------------------------
# Non-dispatchable tasks are an internal error, not a user refusal
# ---------------------------------------------------------------------------
class TestNonDispatchable:
    @pytest.mark.parametrize("task", [QueryTask.UNKNOWN, QueryTask.REPORT_GENERATION])
    def test_dispatching_an_upstream_task_is_a_programming_error(self, registry, optical, task):
        with pytest.raises(ValueError, match="not dispatchable"):
            registry.dispatch(task, ToolContext("q", [optical]))


# ---------------------------------------------------------------------------
# When the inputs fit no tool in the chain, surface the primary tool's reason
# ---------------------------------------------------------------------------
class TestApplicabilityErrors:
    def test_single_optical_to_optical_sar_reports_the_sar_modality_reason(self, registry, optical):
        """Neither SAR (needs radar) nor fusion (needs a pair) fits one optical image; the primary
        tool is the SAR analyser, so its modality error — which redirects to land cover — is shown."""
        with pytest.raises(ValidationError) as ei:
            registry.dispatch(QueryTask.OPTICAL_SAR_ANALYSIS, ToolContext("analyse", [optical]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY
        assert "radar" in ei.value.message.lower()

    def test_no_image_to_optical_sar_reports_missing_image(self, registry):
        with pytest.raises(ValidationError) as ei:
            registry.dispatch(QueryTask.OPTICAL_SAR_ANALYSIS, ToolContext("analyse", []))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_land_cover_with_a_sar_image_is_wrong_modality(self, registry, sar):
        with pytest.raises(ValidationError) as ei:
            registry.dispatch(QueryTask.LAND_COVER_ANALYSIS, ToolContext("classify", [sar]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY

    def test_change_with_one_image_is_missing_second(self, registry, optical):
        with pytest.raises(ValidationError) as ei:
            registry.dispatch(QueryTask.CHANGE_DETECTION, ToolContext("what changed", [optical]))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE


# ---------------------------------------------------------------------------
# The tier ladder (§18), exercised with stub tools
# ---------------------------------------------------------------------------
class TestTierFallback:
    def test_most_capable_available_tool_runs(self):
        preferred = _Stub("preferred", ToolTier.PREFERRED)
        classical = _Stub("classical", ToolTier.CLASSICAL)
        result = ToolRegistry(tools=[classical, preferred]).dispatch(
            QueryTask.LAND_COVER_ANALYSIS, _CTX
        )
        assert result.tool_name == "preferred"  # tier order, not registration order
        assert preferred.ran and not classical.ran

    def test_unavailable_preferred_rung_is_skipped(self):
        preferred = _Stub("preferred", ToolTier.PREFERRED, available=False)
        classical = _Stub("classical", ToolTier.CLASSICAL)
        result = ToolRegistry(tools=[preferred, classical]).dispatch(
            QueryTask.LAND_COVER_ANALYSIS, _CTX
        )
        assert result.tool_name == "classical"
        assert classical.ran and not preferred.ran

    def test_run_time_model_outage_falls_through_to_the_next_tier(self):
        preferred = _Stub("preferred", ToolTier.PREFERRED, outage=True)  # loads then fails at predict
        classical = _Stub("classical", ToolTier.CLASSICAL)
        result = ToolRegistry(tools=[preferred, classical]).dispatch(
            QueryTask.LAND_COVER_ANALYSIS, _CTX
        )
        assert result.tool_name == "classical"
        assert classical.ran

    def test_all_rungs_unavailable_raises_model_unavailable(self):
        registry = ToolRegistry(
            tools=[
                _Stub("preferred", ToolTier.PREFERRED, available=False),
                _Stub("fallback", ToolTier.FALLBACK, available=False),
            ]
        )
        with pytest.raises(ModelUnavailableError) as ei:
            registry.dispatch(QueryTask.LAND_COVER_ANALYSIS, _CTX)
        assert ei.value.code == ErrorCode.MODEL_UNAVAILABLE
        assert ei.value.status_code == 503
        assert ei.value.recoverable is False

    def test_outage_on_the_last_rung_propagates(self):
        registry = ToolRegistry(tools=[_Stub("only", ToolTier.CLASSICAL, outage=True)])
        with pytest.raises(ModelUnavailableError):
            registry.dispatch(QueryTask.LAND_COVER_ANALYSIS, _CTX)

    def test_applicable_lower_rung_runs_when_a_higher_rung_rejects_the_inputs(self):
        """Applicability is independent of tier: a higher rung that rejects the inputs is passed over
        for the lower rung that accepts them, and its rejection is not surfaced."""
        rejecting = _Stub("preferred", ToolTier.PREFERRED, rejects=ErrorCode.WRONG_MODALITY)
        accepting = _Stub("classical", ToolTier.CLASSICAL)
        result = ToolRegistry(tools=[rejecting, accepting]).dispatch(
            QueryTask.LAND_COVER_ANALYSIS, _CTX
        )
        assert result.tool_name == "classical"

    def test_primary_reason_is_the_first_rungs_error_when_none_apply(self):
        """No tool accepts the inputs: the primary (highest-tier, first) tool's error is raised."""
        registry = ToolRegistry(
            tools=[
                _Stub("preferred", ToolTier.PREFERRED, rejects=ErrorCode.WRONG_MODALITY),
                _Stub("classical", ToolTier.CLASSICAL, rejects=ErrorCode.MISSING_BAND),
            ]
        )
        with pytest.raises(ValidationError) as ei:
            registry.dispatch(QueryTask.LAND_COVER_ANALYSIS, _CTX)
        assert ei.value.code == ErrorCode.WRONG_MODALITY  # the preferred rung's reason, not the other
