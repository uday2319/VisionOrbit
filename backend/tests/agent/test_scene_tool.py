"""Tests for the adapted learned scene-captioning rung (brief §17, §18, §26, §49).

Covers the two things the integration has to get right and one it must not do:

* **Live integration.** With the real checkpoint on disk the PREFERRED rung runs, the answer names
  the learned model's class, and the payload is a strict superset of the classical captioning
  payload — so no existing consumer of ``data`` loses a field because the learned model ran.
* **Fallback.** With the learned rung unavailable — the flag off, the checkpoint absent, or torch
  missing — the registry serves classical ``captioning-landcover-cv`` instead. That path is exercised
  by construction (a redirected ``checkpoint_dir``), not by mocking the registry, because the point
  is that the ladder itself works.
* **No over-claiming.** The tool declares the same input contract as the classical tool and refuses
  SAR and the wrong image count, and its confidence is gated on the model's own top-1 probability.

Tests that need the checkpoint are skipped, not failed, when it is absent: a clone that has not run
``python -m ml.adaptation.train`` is a supported state, and the fallback tests below are exactly what
proves it still works.
"""
from __future__ import annotations

import json

import pytest

from app.agents.registry import ToolRegistry
from app.agents.tools import AdaptedSceneCaptioningTool, CaptioningTool, ToolContext
from app.agents.tools.base import ToolResult, ToolTier
from app.core.errors import ErrorCode, ModelUnavailableError, ValidationError
from app.core.types import Modality, QueryTask
from app.geospatial.raster import load_raster
from app.services import scene as scene_service

requires_checkpoint = pytest.mark.skipif(
    not scene_service.weights_path().is_file() or not scene_service.torch_available(),
    reason=(
        "the adapted checkpoint or torch is absent; run `python -m ml.adaptation.train` to "
        "exercise the learned rung (the fallback tests below cover this state)"
    ),
)


@pytest.fixture(scope="module")
def optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def sar(demo_root):
    return load_raster(
        demo_root / "sar" / "scene_sar_vv.tif", max_edge=None, modality=Modality.SAR
    )


class TestContract:
    """Declared metadata only — these pass with or without a checkpoint."""

    def test_it_is_the_preferred_rung_of_the_captioning_task(self):
        tool = AdaptedSceneCaptioningTool()
        assert tool.task is QueryTask.CAPTIONING
        assert tool.tier is ToolTier.PREFERRED
        assert tool.name == "satquery-rs-visual-v1"

    def test_input_contract_matches_the_classical_captioning_tool(self):
        """A different contract would make the fallback conditional on the inputs, not the model."""
        learned = AdaptedSceneCaptioningTool().describe_contract()
        classical = CaptioningTool().describe_contract()
        assert learned == classical

    def test_describe_keeps_the_fixed_four_key_shape(self):
        d = AdaptedSceneCaptioningTool().describe()
        assert set(d) == {"task", "name", "tier", "summary"}
        assert d["tier"] == "preferred"

    def test_summary_does_not_claim_to_be_a_vqa_model(self):
        """§49: the artefact is a scene classifier and the summary has to say so."""
        summary = AdaptedSceneCaptioningTool().summary.lower()
        assert "resnet-18" in summary
        assert "eurosat" in summary
        assert "not a vqa model" in summary

    def test_availability_tracks_the_service_gate(self):
        assert AdaptedSceneCaptioningTool().available() is scene_service.is_available()

    def test_runtime_block_is_present_and_names_the_artefact(self):
        runtime = AdaptedSceneCaptioningTool().describe_runtime()
        assert runtime["model_id"] == "satquery-rs-visual-v1"
        assert runtime["mode"] in {"LIVE", "UNAVAILABLE"}


class TestRejects:
    """Refusals are contract-level, so they hold whether or not the model can load."""

    def test_sar_input_is_wrong_modality(self, sar):
        with pytest.raises(ValidationError) as ei:
            AdaptedSceneCaptioningTool().run(ToolContext("describe this scene", [sar]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY

    def test_no_image_is_rejected(self):
        with pytest.raises(ValidationError) as ei:
            AdaptedSceneCaptioningTool().run(ToolContext("describe this scene", []))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_two_images_is_too_many(self, optical):
        with pytest.raises(ValidationError) as ei:
            AdaptedSceneCaptioningTool().run(
                ToolContext("describe this scene", [optical, optical])
            )
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES


@requires_checkpoint
class TestLiveLearnedRun:
    @pytest.fixture(scope="class")
    def result(self, optical) -> ToolResult:
        return AdaptedSceneCaptioningTool().run(ToolContext("describe this scene", [optical]))

    def test_it_reports_itself_as_the_learned_rung(self, result):
        assert result.tool_name == "satquery-rs-visual-v1"
        assert result.tier is ToolTier.PREFERRED
        assert result.task is QueryTask.CAPTIONING

    def test_answer_names_the_model_and_its_predicted_class(self, result):
        answer = result.answer
        predicted = result.data["scene_classification"]["predicted_class"]
        assert "adapted remote-sensing model" in answer
        assert predicted in answer, "the answer must state the class the model actually predicted"
        assert "%" in answer  # the probability is stated, not hidden

    def test_payload_is_a_superset_of_the_classical_captioning_payload(self, optical, result):
        """The frontend reads the classical keys; the learned rung may only add to them."""
        classical = CaptioningTool().run(ToolContext("describe this scene", [optical]))
        assert set(classical.data) <= set(result.data)
        assert result.data["caption"]
        assert result.data["figures"]
        assert result.data["landcover"]

    def test_prediction_is_a_real_distribution_over_the_ten_eurosat_classes(self, result):
        block = result.data["scene_classification"]
        ranked = block["ranked"]
        assert len(ranked) == 10
        probs = [p["probability"] for p in ranked]
        assert probs == sorted(probs, reverse=True)
        assert sum(probs) == pytest.approx(1.0, abs=1e-3)
        assert block["predicted_class"] == ranked[0]["class_name"]
        assert block["probability"] == pytest.approx(ranked[0]["probability"], abs=1e-6)

    def test_provenance_is_read_from_the_checkpoint_not_asserted(self, result):
        block = result.data["scene_classification"]
        assert block["architecture"] == "resnet18"
        assert block["pretrained_weights"] == "IMAGENET1K_V1"
        assert "linear probe" in block["adaptation_method"]
        assert block["checkpoint"].endswith("model.pt")
        assert block["dataset"]

    def test_evidence_leads_with_the_learned_model_and_disclaims_its_scope(self, result):
        joined = " ".join(result.evidence)
        assert "satquery-rs-visual-v1" in result.evidence[0]
        assert "no pixel masks and no object detections" in joined
        # The deterministic evidence is still carried, not displaced.
        assert "separability" in joined.lower()

    def test_confidence_is_gated_on_the_model_probability(self, result):
        names = {f.name for f in result.confidence.factors}
        assert "model_probability" in names
        assert "model_margin" in names
        probability = result.data["scene_classification"]["probability"]
        assert result.confidence.score <= probability + 1e-6, (
            "a gate means the score can never exceed the model's own top-1 probability"
        )

    def test_result_serialises_without_the_raw_object(self, result):
        payload = json.loads(json.dumps(result.to_dict()))
        assert "raw" not in payload
        assert payload["data"]["scene_classification"]["predicted_class"]


@requires_checkpoint
class TestLiveRegistryDispatch:
    """End-to-end through the registry: the learned rung is what actually serves captioning."""

    def test_dispatch_uses_the_learned_rung(self, optical):
        result = ToolRegistry().dispatch(
            QueryTask.CAPTIONING, ToolContext("describe this scene", [optical])
        )
        assert result.tool_name == "satquery-rs-visual-v1"
        assert result.tier is ToolTier.PREFERRED

    def test_runtime_reports_live_with_the_real_checkpoint(self):
        """The three facts the brief asks the live backend to be able to report.

        ``/api/models`` surfaces exactly this block (asserted end-to-end in
        ``tests/api/test_routes.py``); this checks the values it carries.
        """
        runtime = AdaptedSceneCaptioningTool().describe_runtime()
        assert runtime["model_id"] == "satquery-rs-visual-v1"
        assert runtime["mode"] == "LIVE"
        assert runtime["checkpoint"].endswith("ml/checkpoints/satquery-rs-visual-v1/model.pt")
        assert runtime["checkpoint_present"] is True
        assert runtime["reason"] is None
        assert runtime["num_classes"] == 10


class TestFallbackWhenTheModelCannotRun:
    """The ladder: learned rung unavailable → deterministic analyser, never a refusal."""

    @pytest.fixture
    def _no_checkpoint(self, tmp_path, monkeypatch):
        """Point the loader at an empty directory: present flag, absent weights."""
        from app import config

        monkeypatch.setenv("SATQUERY_CHECKPOINT_DIR", str(tmp_path / "empty"))
        monkeypatch.setenv("SATQUERY_ENABLE_LEARNED_MODELS", "true")
        config.reset_settings_cache()
        scene_service.reset_model_cache()
        yield
        config.reset_settings_cache()
        scene_service.reset_model_cache()

    @pytest.fixture
    def _flag_off(self, monkeypatch):
        from app import config

        monkeypatch.setenv("SATQUERY_ENABLE_LEARNED_MODELS", "false")
        config.reset_settings_cache()
        scene_service.reset_model_cache()
        yield
        config.reset_settings_cache()
        scene_service.reset_model_cache()

    def test_missing_checkpoint_makes_the_rung_unavailable_with_a_reason(self, _no_checkpoint):
        assert AdaptedSceneCaptioningTool().available() is False
        runtime = AdaptedSceneCaptioningTool().describe_runtime()
        assert runtime["mode"] == "UNAVAILABLE"
        assert "no trained checkpoint" in runtime["reason"]
        assert "ml.adaptation.train" in runtime["reason"]

    def test_disabled_flag_makes_the_rung_unavailable_with_a_reason(self, _flag_off):
        assert AdaptedSceneCaptioningTool().available() is False
        assert "disabled" in AdaptedSceneCaptioningTool().describe_runtime()["reason"]

    def test_predict_raises_a_model_outage_rather_than_guessing(self, _no_checkpoint, optical):
        """The registry falls through on ModelUnavailableError only, so it must be that error."""
        tool = AdaptedSceneCaptioningTool()
        ctx = ToolContext("describe this scene", [optical])
        with pytest.raises(ModelUnavailableError):
            tool.predict(tool.preprocess(ctx))

    def test_registry_serves_captioning_from_the_deterministic_tool(self, _no_checkpoint, optical):
        result = ToolRegistry().dispatch(
            QueryTask.CAPTIONING, ToolContext("describe this scene", [optical])
        )
        assert result.tool_name == "captioning-landcover-cv"
        assert result.tier is ToolTier.CLASSICAL
        assert result.answer
        assert "scene_classification" not in result.data

    def test_the_deterministic_chain_is_still_registered_either_way(self):
        """Whatever the host state, dropping the classical rung would break the ladder itself."""
        names = [t.name for t in ToolRegistry().tools_for(QueryTask.CAPTIONING)]
        assert names == ["satquery-rs-visual-v1", "captioning-landcover-cv"]
