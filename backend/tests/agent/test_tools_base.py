"""Tests for the uniform tool interface and result envelope (brief §17, §18).

These exercise the *mechanics* of the base layer with a stub tool, independent of any real
analysis: the :meth:`Tool.run` template calls its five steps in the right order, image-count
guards raise the right codes, status is derived correctly from confidence and warnings, and the
result serialises to JSON without leaking the heavy ``raw`` object. The concrete land-cover tool
proves the contract against a real analysis in ``test_landcover_tool.py``.
"""
from __future__ import annotations

import json

import pytest

from app.agents.tools import Prepared, Tool, ToolContext, ToolResult, ToolTier
from app.agents.tools.base import _status_for
from app.core.errors import ErrorCode, ValidationError
from app.core.types import ConfidenceLevel, QueryTask, StepStatus
from app.services.confidence import ConfidenceFactor, FactorKind, aggregate


def _factor(name: str, value: float, weight: float = 1.0, *, gate: bool = False) -> ConfidenceFactor:
    return ConfidenceFactor(
        name=name,
        value=value,
        weight=weight,
        reason=f"{name}={value}",
        kind=FactorKind.GATE if gate else FactorKind.CONTRIBUTOR,
    )


class _StubTool(Tool):
    """Minimal concrete tool that records call order and lets tests drive confidence/warnings."""

    task = QueryTask.VQA
    name = "stub"
    tier = ToolTier.CLASSICAL
    summary = "stub tool for interface tests"

    def __init__(self, *, warnings: list[str] | None = None, factors: list[ConfidenceFactor] | None = None):
        self.calls: list[str] = []
        self._warnings = warnings or []
        self._factors = factors if factors is not None else [_factor("evidence", 0.9)]

    def validate_input(self, ctx: ToolContext) -> None:
        self.calls.append("validate")

    def preprocess(self, ctx: ToolContext) -> Prepared:
        self.calls.append("preprocess")
        return Prepared(rasters=list(ctx.rasters))

    def predict(self, prepared: Prepared) -> dict:
        self.calls.append("predict")
        return {"value": 42}

    def explain(self, raw: dict) -> list[str]:
        return [f"value={raw['value']}"]

    def postprocess(self, raw: dict, prepared: Prepared) -> ToolResult:
        self.calls.append("postprocess")
        return self.build_result(
            answer=f"value is {raw['value']}",
            data=raw,
            confidence=aggregate(self._factors),
            evidence=self.explain(raw),
            warnings=self._warnings,
            raw=raw,
        )


# ---------------------------------------------------------------------------
# ToolTier
# ---------------------------------------------------------------------------
class TestToolTier:
    def test_label_is_lowercased_name(self):
        assert ToolTier.PREFERRED.label == "preferred"
        assert ToolTier.FALLBACK.label == "fallback"
        assert ToolTier.CLASSICAL.label == "classical"

    def test_order_is_try_order(self):
        """Lower value = tried first, so a chain can be sorted by tier."""
        assert ToolTier.PREFERRED < ToolTier.FALLBACK < ToolTier.CLASSICAL


# ---------------------------------------------------------------------------
# ToolContext.expect_count — the image-count guard
# ---------------------------------------------------------------------------
class TestExpectCount:
    def test_exact_count_is_accepted(self):
        ToolContext("q", [object()]).expect_count(1, QueryTask.LAND_COVER_ANALYSIS)  # no raise

    def test_too_few_raises_missing_second_image(self):
        with pytest.raises(ValidationError) as ei:
            ToolContext("q", []).expect_count(1, QueryTask.LAND_COVER_ANALYSIS)
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE
        assert ei.value.context["expected"] == 1
        assert ei.value.context["received"] == 0

    def test_too_many_raises_too_many_images(self):
        with pytest.raises(ValidationError) as ei:
            ToolContext("q", [object(), object()]).expect_count(1, QueryTask.LAND_COVER_ANALYSIS)
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES

    def test_pair_task_needs_two(self):
        with pytest.raises(ValidationError) as ei:
            ToolContext("q", [object()]).expect_count(2, QueryTask.CHANGE_DETECTION)
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE


# ---------------------------------------------------------------------------
# _status_for — the OK/WARNING rule, in isolation
# ---------------------------------------------------------------------------
class TestStatusFor:
    def test_clean_sufficient_result_is_ok(self):
        assert _status_for(aggregate([_factor("a", 0.9)]), []) is StepStatus.OK

    def test_any_warning_downgrades_to_warning(self):
        assert _status_for(aggregate([_factor("a", 0.9)]), ["heads up"]) is StepStatus.WARNING

    def test_insufficient_confidence_is_warning_even_without_warnings(self):
        report = aggregate([])  # no evidence -> INSUFFICIENT
        assert report.level is ConfidenceLevel.INSUFFICIENT
        assert _status_for(report, []) is StepStatus.WARNING


# ---------------------------------------------------------------------------
# Tool.run — the template method
# ---------------------------------------------------------------------------
class TestRun:
    def test_steps_run_in_contract_order(self):
        tool = _StubTool()
        tool.run(ToolContext("q", []))
        assert tool.calls == ["validate", "preprocess", "predict", "postprocess"]

    def test_run_returns_postprocess_result(self):
        result = _StubTool().run(ToolContext("q", []))
        assert isinstance(result, ToolResult)
        assert result.answer == "value is 42"
        assert result.task is QueryTask.VQA
        assert result.tier is ToolTier.CLASSICAL

    def test_validation_error_propagates_unchanged(self):
        """run() never swallows a raise; the orchestrator's chain decides what to do next."""

        class _Rejecting(_StubTool):
            def validate_input(self, ctx: ToolContext) -> None:
                raise ValidationError("nope", code=ErrorCode.WRONG_MODALITY)

        tool = _Rejecting()
        with pytest.raises(ValidationError):
            tool.run(ToolContext("q", []))
        assert tool.calls == []  # failed before preprocess

    def test_clean_result_is_ok(self):
        assert _StubTool().run(ToolContext("q", [])).status is StepStatus.OK

    def test_warned_result_is_warning(self):
        tool = _StubTool(warnings=["approximate"])
        assert tool.run(ToolContext("q", [])).status is StepStatus.WARNING

    def test_insufficient_result_is_warning(self):
        tool = _StubTool(factors=[])  # aggregate([]) -> INSUFFICIENT
        result = tool.run(ToolContext("q", []))
        assert result.confidence.level is ConfidenceLevel.INSUFFICIENT
        assert result.status is StepStatus.WARNING


# ---------------------------------------------------------------------------
# ToolResult serialisation
# ---------------------------------------------------------------------------
class TestToolResult:
    def test_to_dict_has_the_expected_keys(self):
        result = _StubTool().run(ToolContext("q", []))
        payload = json.loads(json.dumps(result.to_dict()))  # raises on any non-JSON value
        assert set(payload) == {
            "task", "tool", "tier", "status", "answer", "confidence", "evidence", "warnings", "data"
        }
        assert payload["tool"] == "stub"
        assert payload["tier"] == "classical"
        assert payload["task"] == QueryTask.VQA.value

    def test_raw_is_kept_on_object_but_excluded_from_dict(self):
        result = _StubTool().run(ToolContext("q", []))
        assert result.raw == {"value": 42}
        assert "raw" not in result.to_dict()

    def test_is_sufficient_tracks_confidence(self):
        assert _StubTool().run(ToolContext("q", [])).is_sufficient is True
        assert _StubTool(factors=[]).run(ToolContext("q", [])).is_sufficient is False

    def test_confidence_and_evidence_are_carried(self):
        result = _StubTool().run(ToolContext("q", []))
        assert result.confidence.level is ConfidenceLevel.HIGH
        assert result.evidence == ["value=42"]


# ---------------------------------------------------------------------------
# describe() — registry / /api/models metadata
# ---------------------------------------------------------------------------
class TestDescribe:
    def test_describe_reports_class_metadata(self):
        d = _StubTool().describe()
        assert d == {
            "task": QueryTask.VQA.value,
            "name": "stub",
            "tier": "classical",
            "summary": "stub tool for interface tests",
        }
