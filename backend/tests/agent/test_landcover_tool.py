"""Tests for the land-cover tool — the reference implementation of the Tool contract (brief §17).

Runs the tool against the real demo optical scene (no mocks): a clean multispectral image must
classify with HIGH, measurement-grounded confidence, and the tool must refuse the inputs it cannot
honestly handle — a SAR image (wrong modality) and the wrong number of images — with specific,
recoverable errors rather than a fabricated map.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.agents.tools import LandCoverTool, ToolContext
from app.agents.tools.base import ToolResult, ToolTier
from app.core.errors import ErrorCode, ValidationError
from app.core.types import ConfidenceLevel, Modality, QueryTask, StepStatus
from app.geospatial.raster import load_raster
from app.services.landcover import LandCoverResult


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
        return LandCoverTool().run(ToolContext("classify the land cover", [optical]))

    def test_identifies_itself_honestly(self, result):
        assert isinstance(result, ToolResult)
        assert result.task is QueryTask.LAND_COVER_ANALYSIS
        assert result.tool_name == "landcover-spectral-cv"
        assert result.tier is ToolTier.CLASSICAL  # not claiming to be a trained model

    def test_answer_is_nonempty_and_measurement_grounded(self, result):
        assert result.answer
        assert "%" in result.answer  # phrased from measured class fractions

    def test_clean_demo_scene_is_high_confidence(self, result):
        assert result.confidence.level is ConfidenceLevel.HIGH

    def test_status_is_consistent_with_its_inputs(self, result):
        if result.warnings or result.confidence.level is ConfidenceLevel.INSUFFICIENT:
            assert result.status is StepStatus.WARNING
        else:
            assert result.status is StepStatus.OK

    def test_evidence_mentions_separability(self, result):
        assert any("separability" in line.lower() for line in result.evidence)

    def test_evidence_names_the_method(self, result):
        assert any("spectral" in line.lower() for line in result.evidence)

    def test_raw_present_but_not_serialised(self, result):
        assert result.raw is not None  # the LandCoverResult, for rendering the class map
        payload = json.loads(json.dumps(result.to_dict()))
        assert "raw" not in payload
        assert payload["data"]["method"] == "spectral-indices"
        assert payload["data"]["classes"]  # per-class stats carried through

    def test_describe(self):
        d = LandCoverTool().describe()
        assert d["name"] == "landcover-spectral-cv"
        assert d["tier"] == "classical"
        assert d["task"] == QueryTask.LAND_COVER_ANALYSIS.value
        assert d["summary"]


class TestRejects:
    def test_sar_input_is_wrong_modality(self, sar):
        """A SAR image cannot be land-cover classified spectrally; refuse, don't approximate."""
        with pytest.raises(ValidationError) as ei:
            LandCoverTool().run(ToolContext("classify the land cover", [sar]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY

    def test_no_image_is_rejected(self):
        with pytest.raises(ValidationError) as ei:
            LandCoverTool().run(ToolContext("classify the land cover", []))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_two_images_is_too_many(self, optical):
        with pytest.raises(ValidationError) as ei:
            LandCoverTool().run(ToolContext("classify", [optical, optical]))
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES


class TestDegenerateResult:
    """Phrasing when the classifier could delineate nothing (e.g. no usable bands).

    The tool must state that plainly rather than crash on a missing dominant class or invent a
    breakdown — the honest empty-result path.
    """

    @pytest.fixture
    def empty(self) -> LandCoverResult:
        return LandCoverResult(
            class_map=np.zeros((4, 4), dtype=np.uint8),
            stats=[],
            method="insufficient-bands",
            indices_used=[],
            separability=0.0,
            warnings=["No identifiable red, green and blue bands."],
        )

    def test_summary_declines_instead_of_inventing_classes(self, empty):
        assert LandCoverTool._summarize(empty).startswith("No land-cover classes")

    def test_explain_skips_absent_sections_without_crashing(self, empty):
        lines = LandCoverTool().explain(empty)
        assert any("insufficient-bands" in line for line in lines)
        assert any("separability" in line.lower() for line in lines)
        assert not any("dominant" in line.lower() for line in lines)  # no dominant class
        assert not any("indices used" in line.lower() for line in lines)  # none were used
        assert "No identifiable red, green and blue bands." in lines  # warning carried through
