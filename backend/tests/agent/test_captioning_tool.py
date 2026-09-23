"""Tests for the captioning tool — single-image scene description behind the Tool contract (§2A, §17).

The query-free counterpart to :class:`~app.agents.tools.vqa.VqaTool`, run against the real demo
optical scene (no mocks). It wraps :func:`app.services.captioning.describe_scene`, so the caption
must stay measurement-grounded and carry the classification's evidence-based confidence, and the
tool must refuse what it cannot honestly caption — a SAR image (no spectral land-cover signal) and
the wrong number of images — rather than describe a scene it did not measure.
"""
from __future__ import annotations

import json

import pytest

from app.agents.tools import CaptioningTool, ToolContext
from app.agents.tools.base import ToolResult, ToolTier
from app.core.errors import ErrorCode, ValidationError
from app.core.types import ConfidenceLevel, Modality, QueryTask, StepStatus
from app.geospatial.raster import load_raster


@pytest.fixture(scope="module")
def optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def sar(demo_root):
    return load_raster(
        demo_root / "sar" / "scene_sar_vv.tif", max_edge=None, modality=Modality.SAR
    )


class TestSuccess:
    @pytest.fixture(scope="class")
    def result(self, optical) -> ToolResult:
        return CaptioningTool().run(ToolContext("describe this scene", [optical]))

    def test_identifies_itself_honestly(self, result):
        assert isinstance(result, ToolResult)
        assert result.task is QueryTask.CAPTIONING
        assert result.tool_name == "captioning-landcover-cv"
        assert result.tier is ToolTier.CLASSICAL  # a templated caption, not a generated one

    def test_answer_is_measurement_grounded(self, result):
        assert result.answer
        assert "%" in result.answer  # the caption quotes measured class shares

    def test_clean_demo_scene_is_high_confidence(self, result):
        assert result.confidence.level is ConfidenceLevel.HIGH

    def test_status_is_consistent_with_its_inputs(self, result):
        if result.warnings or result.confidence.level is ConfidenceLevel.INSUFFICIENT:
            assert result.status is StepStatus.WARNING
        else:
            assert result.status is StepStatus.OK

    def test_evidence_discloses_provenance(self, result):
        joined = " ".join(result.evidence).lower()
        assert "separability" in joined
        assert "classification" in joined

    def test_raw_present_but_not_serialised(self, result):
        assert result.raw is not None  # the SceneCaption, for rendering the class map behind it
        payload = json.loads(json.dumps(result.to_dict()))
        assert "raw" not in payload
        assert payload["data"]["caption"]
        assert payload["data"]["figures"]  # the cited shares carried through
        assert "class_map" not in payload["data"]["landcover"]

    def test_describe(self):
        d = CaptioningTool().describe()
        assert d["name"] == "captioning-landcover-cv"
        assert d["tier"] == "classical"
        assert d["task"] == QueryTask.CAPTIONING.value
        assert d["summary"]


class TestRejects:
    def test_sar_input_is_wrong_modality(self, sar):
        """Captioning describes spectral land cover; a single SAR image carries none."""
        with pytest.raises(ValidationError) as ei:
            CaptioningTool().run(ToolContext("describe this scene", [sar]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY

    def test_no_image_is_rejected(self):
        with pytest.raises(ValidationError) as ei:
            CaptioningTool().run(ToolContext("describe this scene", []))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_two_images_is_too_many(self, optical):
        with pytest.raises(ValidationError) as ei:
            CaptioningTool().run(ToolContext("describe this scene", [optical, optical]))
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES
