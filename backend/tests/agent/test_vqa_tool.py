"""Tests for the VQA tool — single-image question answering behind the Tool contract (§2A, §17, §27).

Runs against the real demo optical scene (no mocks). The tool wraps
:func:`app.services.vqa.answer_question`, so the honesty properties proven there must survive the
wrapping: a measurable answer stays measurement-grounded and carries evidence-based confidence, a
*where* / count question is redirected rather than faked, and the inputs and queries the capability
cannot honestly handle — a SAR image, the wrong image count, an object with no detector, a spatial
relation, an empty query — are refused with specific, recoverable errors instead of a fabricated
sentence.
"""
from __future__ import annotations

import json

import pytest

from app.agents.tools import ToolContext, VqaTool
from app.agents.tools.base import ToolResult, ToolTier
from app.core.errors import ErrorCode, UnsupportedTaskError, ValidationError
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
        return VqaTool().run(ToolContext("what land cover is in this image", [optical]))

    def test_identifies_itself_honestly(self, result):
        assert isinstance(result, ToolResult)
        assert result.task is QueryTask.VQA
        assert result.tool_name == "vqa-landcover-cv"
        assert result.tier is ToolTier.CLASSICAL  # spectral classifier + parsing, not a VLM

    def test_answer_is_measurement_grounded(self, result):
        assert result.answer
        assert "%" in result.answer  # phrased from measured class fractions

    def test_clean_demo_scene_is_high_confidence(self, result):
        # The answer rests on the classification, so it inherits the classification's confidence.
        assert result.confidence.level is ConfidenceLevel.HIGH

    def test_status_is_consistent_with_its_inputs(self, result):
        if result.warnings or result.confidence.level is ConfidenceLevel.INSUFFICIENT:
            assert result.status is StepStatus.WARNING
        else:
            assert result.status is StepStatus.OK

    def test_evidence_discloses_provenance(self, result):
        joined = " ".join(result.evidence).lower()
        assert "separability" in joined  # classifier separability disclosed
        assert "composition query" in joined  # how the question was read is disclosed

    def test_raw_present_but_not_serialised(self, result):
        assert result.raw is not None  # the VqaAnswer, for rendering the class map behind it
        payload = json.loads(json.dumps(result.to_dict()))
        assert "raw" not in payload
        assert payload["data"]["intent"] == "composition"
        assert payload["data"]["figures"]  # the cited shares carried through
        assert "class_map" not in payload["data"]["landcover"]  # array kept out of the payload

    def test_describe(self):
        d = VqaTool().describe()
        assert d["name"] == "vqa-landcover-cv"
        assert d["tier"] == "classical"
        assert d["task"] == QueryTask.VQA.value
        assert d["summary"]


class TestRedirectsWhereToGrounding:
    """A *where* question is answered with coverage and pointed at grounding, not faked here."""

    @pytest.fixture(scope="class")
    def result(self, optical) -> ToolResult:
        return VqaTool().run(ToolContext("where is the water?", [optical]))

    def test_redirect_is_carried_through(self, result):
        assert result.data["redirect"] == "grounding"

    def test_answer_keeps_the_coverage_figure(self, result):
        assert "%" in result.answer  # the measurable part is still answered
        assert "grounding" in result.answer.lower()  # and the better capability is named


class TestRejects:
    def test_sar_input_is_wrong_modality(self, sar):
        """A single SAR image has no spectral land-cover signal to read; refuse, don't approximate."""
        with pytest.raises(ValidationError) as ei:
            VqaTool().run(ToolContext("how much water is there?", [sar]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY

    def test_no_image_is_rejected(self):
        with pytest.raises(ValidationError) as ei:
            VqaTool().run(ToolContext("how much water is there?", []))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_two_images_is_too_many(self, optical):
        with pytest.raises(ValidationError) as ei:
            VqaTool().run(ToolContext("how much water is there?", [optical, optical]))
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES

    def test_undetectable_object_is_unsupported(self, optical):
        """"Ships" has no detector; refuse by name rather than guess from land cover."""
        with pytest.raises(UnsupportedTaskError):
            VqaTool().run(ToolContext("how many ships are in the harbour?", [optical]))

    def test_spatial_relation_is_unsupported(self, optical):
        with pytest.raises(UnsupportedTaskError):
            VqaTool().run(ToolContext("are there buildings near the river?", [optical]))

    def test_empty_query_is_rejected(self, optical):
        with pytest.raises(ValidationError) as ei:
            VqaTool().run(ToolContext("   ", [optical]))
        assert ei.value.code == ErrorCode.EMPTY_QUERY
