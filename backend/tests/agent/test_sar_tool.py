"""Tests for the SAR tool — single-image radar backscatter analysis (brief §4, §17, §28).

Runs against the real demo SAR scene (no mocks): a clean single-polarisation image must segment
into scattering regimes with measured, non-insufficient confidence, and the tool must (a) refuse the
inputs it cannot honestly handle — an optical image (wrong modality) and the wrong number of images
— and (b) stay honest on a featureless image, where the regime boundaries are *refused* rather than
forced, reporting only the diffuse population it can actually measure.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.agents.tools import SarTool, ToolContext
from app.agents.tools.base import ToolResult, ToolTier
from app.core.errors import ErrorCode, ValidationError
from app.core.types import ConfidenceLevel, Modality, QueryTask, ScatteringRegime, StepStatus
from app.geospatial.raster import load_raster
from tests.conftest import write_test_raster


@pytest.fixture(scope="module")
def sar(demo_root):
    return load_raster(
        demo_root / "sar" / "scene_sar_vv.tif", max_edge=None, modality=Modality.SAR
    )


@pytest.fixture(scope="module")
def optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


class TestSuccess:
    @pytest.fixture(scope="class")
    def result(self, sar) -> ToolResult:
        return SarTool().run(ToolContext("what does the radar show?", [sar]))

    def test_identifies_itself_honestly(self, result):
        assert isinstance(result, ToolResult)
        assert result.task is QueryTask.OPTICAL_SAR_ANALYSIS
        assert result.tool_name == "sar-backscatter-cv"
        assert result.tier is ToolTier.CLASSICAL  # signal processing, not a trained model

    def test_answer_is_measurement_grounded(self, result):
        assert result.answer
        assert "dB" in result.answer  # phrased from measured backscatter, in decibels
        assert "scattering" in result.answer.lower()  # regimes, not land-cover classes

    def test_clean_scene_is_not_insufficient(self, result):
        # The demo SAR scene has distinct regimes; confidence must reflect measured evidence.
        assert result.confidence.level is not ConfidenceLevel.INSUFFICIENT

    def test_status_is_consistent_with_its_inputs(self, result):
        if result.warnings or result.confidence.level is ConfidenceLevel.INSUFFICIENT:
            assert result.status is StepStatus.WARNING
        else:
            assert result.status is StepStatus.OK

    def test_evidence_discloses_speckle_filtering(self, result):
        joined = " ".join(result.evidence).lower()
        assert "speckle" in joined  # the filter is disclosed
        assert "looks" in joined  # the estimated look count is disclosed

    def test_raw_present_but_not_serialised(self, result):
        assert result.raw is not None  # the SarResult, for rendering the regime map
        payload = json.loads(json.dumps(result.to_dict()))
        assert "raw" not in payload
        assert payload["data"]["regimes"]  # per-regime stats carried through
        assert payload["data"]["polarization"]

    def test_describe(self):
        d = SarTool().describe()
        assert d["name"] == "sar-backscatter-cv"
        assert d["tier"] == "classical"
        assert d["task"] == QueryTask.OPTICAL_SAR_ANALYSIS.value
        assert d["summary"]


class TestRejects:
    def test_optical_input_is_wrong_modality(self, optical):
        """An optical image is not radar; refuse rather than analyse reflectance as backscatter."""
        with pytest.raises(ValidationError) as ei:
            SarTool().run(ToolContext("analyse the backscatter", [optical]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY

    def test_no_image_is_rejected(self):
        with pytest.raises(ValidationError) as ei:
            SarTool().run(ToolContext("analyse the backscatter", []))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_two_images_is_too_many(self, sar):
        with pytest.raises(ValidationError) as ei:
            SarTool().run(ToolContext("analyse", [sar, sar]))
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES


class TestFeaturelessImageRefusesBoundaries:
    """A near-uniform backscatter field has no distinct smooth or bright mode. The tool must not
    force regime boundaries onto noise (§18, §28): both thresholds are refused, everything measured
    is reported as the diffuse middle, and the explanation omits the boundary lines it cannot state.
    """

    def test_boundaries_refused_but_result_still_honest(self, tmp_path):
        rng = np.random.default_rng(7)
        # Unimodal Gaussian backscatter around -12 dB: no separable dark or bright population.
        arr = rng.normal(-12.0, 1.0, (64, 64)).astype(np.float32)
        path = write_test_raster(tmp_path / "flat_sar.tif", arr, descriptions=("VV",))
        raster = load_raster(path, max_edge=None, modality=Modality.SAR)

        result = SarTool().run(ToolContext("what does the radar show?", [raster]))
        assert isinstance(result, ToolResult)  # analysed without crashing
        assert result.raw.smooth_threshold_db is None  # no separable smooth mode
        assert result.raw.bright_threshold_db is None  # no separable bright mode
        joined = " ".join(result.evidence).lower()
        assert "boundary at" not in joined  # the two threshold lines are omitted
        assert "speckle" in joined  # the speckle line is still present


class TestEmptyStatsPhrasing:
    """The defensive empty-result path in :meth:`SarTool._summarize`: if no regime could be measured
    at all, the tool states that plainly instead of indexing an empty stats list. Driven with a
    duck-typed stub, since :func:`analyze_sar` raises before returning an empty result on real data.
    """

    def test_summary_declines_instead_of_indexing_empty(self):
        stub = SimpleNamespace(stats=[])
        assert SarTool._summarize(stub).startswith("No scattering regimes")

    def test_summary_omits_surface_hint_when_regime_has_none(self):
        # UNCLASSIFIED is a real regime with no candidate surfaces; if it were dominant, the
        # "consistent with ..." clause must be omitted rather than trailing an empty list.
        dom = SimpleNamespace(regime=ScatteringRegime.UNCLASSIFIED, fraction=1.0, mean_db=-9.0)
        stub = SimpleNamespace(
            stats=[dom],
            separated_regimes=[ScatteringRegime.UNCLASSIFIED],
            polarization="VV",
        )
        sentence = SarTool._summarize(stub)
        assert "unclassified scattering" in sentence.lower()  # dominant regime still named
        assert "consistent with" not in sentence  # no surface hint to offer
