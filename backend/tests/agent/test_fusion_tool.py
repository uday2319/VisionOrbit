"""Tests for the fusion tool — cross-modal optical + SAR analysis (brief §4, §17, §18, §27).

The second *pair* tool. Runs against the real demo optical/SAR pair (no mocks): a co-registered
optical and radar acquisition of the same area must produce a measured, arbitrated land-cover
reading whose figures come from :class:`~app.services.fusion.FusionResult`, never asserted beyond
what the two sensors jointly support. Like :class:`~app.agents.tools.change.ChangeTool` it owns the
alignment the fusion *service* refuses to guess at, so the tool must (a) refuse inputs it cannot
honestly fuse — the wrong number of images, or a same-sensor pair that is a change/SAR question
rather than a cross-modal one — and (b) resample a partially-overlapping pair onto the optical grid
while surfacing the geometry warning rather than dropping it.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import Affine

from app.agents.tools import ToolContext
from app.agents.tools.base import ToolResult, ToolTier
from app.agents.tools.fusion import FusionTool
from app.core.errors import ErrorCode, ValidationError
from app.core.types import ConfidenceLevel, LandCoverClass, Modality, QueryTask, StepStatus
from app.geospatial.raster import load_raster
from tests.conftest import TEST_ORIGIN_X, TEST_ORIGIN_Y, TEST_PIXEL_M, write_test_raster

_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")


@pytest.fixture(scope="module")
def optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def sar(demo_root):
    return load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None, modality=Modality.SAR)


def _origin(x: float, y: float) -> Affine:
    return Affine(TEST_PIXEL_M, 0.0, x, 0.0, -TEST_PIXEL_M, y)


class TestSuccess:
    @pytest.fixture(scope="class")
    def result(self, optical, sar) -> ToolResult:
        return FusionTool().run(ToolContext("combine the optical and radar views", [optical, sar]))

    def test_identifies_itself_honestly(self, result):
        assert isinstance(result, ToolResult)
        assert result.task is QueryTask.OPTICAL_SAR_ANALYSIS
        assert result.tool_name == "fusion-decision-cv"
        assert result.tier is ToolTier.CLASSICAL  # decision-level rules, not a learned fusion model

    def test_answer_is_measurement_grounded(self, result):
        assert result.answer
        assert "%" in result.answer  # phrased from the measured class breakdown / corroboration
        assert "optical and radar" in result.answer.lower()  # both sensors named, neither invented

    def test_clean_pair_is_not_insufficient(self, result):
        # The demo pair is the canonical fusion case (both sensors agree on most of the scene);
        # confidence must reflect that measured evidence, not be forced low or high.
        assert result.confidence.level is not ConfidenceLevel.INSUFFICIENT

    def test_status_is_consistent_with_its_inputs(self, result):
        if result.warnings or result.confidence.level is ConfidenceLevel.INSUFFICIENT:
            assert result.status is StepStatus.WARNING
        else:
            assert result.status is StepStatus.OK

    def test_evidence_discloses_registration_and_corroboration(self, result):
        joined = " ".join(result.evidence).lower()
        assert "registration" in joined  # the alignment residual is disclosed
        assert "corroborated" in joined  # the fraction both sensors independently supported

    def test_raw_present_but_not_serialised(self, result):
        assert result.raw is not None  # the FusionResult, for rendering the fused map
        payload = json.loads(json.dumps(result.to_dict()))
        assert "raw" not in payload
        assert payload["data"]["classes"]  # per-class fused coverage carried through
        assert payload["data"]["dominant_class"]
        assert "corroborated_fraction" in payload["data"]
        assert "conflicts" in payload["data"]  # arbitrated disagreements carried through
        assert payload["data"]["method"].startswith("decision-level")

    def test_describe(self):
        d = FusionTool().describe()
        assert d["name"] == "fusion-decision-cv"
        assert d["tier"] == "classical"
        assert d["task"] == QueryTask.OPTICAL_SAR_ANALYSIS.value
        assert d["summary"]


class TestRejects:
    def test_one_image_is_missing_second(self, optical):
        with pytest.raises(ValidationError) as ei:
            FusionTool().run(ToolContext("fuse the two views", [optical]))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_three_images_is_too_many(self, optical, sar):
        with pytest.raises(ValidationError) as ei:
            FusionTool().run(ToolContext("fuse the views", [optical, sar, optical]))
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES

    def test_two_optical_is_wrong_modality(self, optical):
        """Two optical images are a change question, not a cross-modal one — redirect, don't fuse."""
        with pytest.raises(ValidationError) as ei:
            FusionTool().run(ToolContext("fuse the views", [optical, optical]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY

    def test_two_sar_is_wrong_modality(self, sar):
        """Two SAR images cannot be fused across modalities; one SAR image is a SAR analysis."""
        with pytest.raises(ValidationError) as ei:
            FusionTool().run(ToolContext("fuse the views", [sar, sar]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY


class TestAlignsAndSurfacesGeometry:
    """A partially-overlapping optical/SAR pair on the same CRS must be resampled onto the optical
    grid — the tool's own :meth:`FusionTool.preprocess` step — with the partial-overlap geometry
    warning surfaced to the user rather than silently dropped (§8, §32)."""

    def test_partial_overlap_is_aligned_and_warned(self, tmp_path):
        rng = np.random.default_rng(3)
        opt_arr = (rng.random((6, 64, 64)) * 10000).astype(np.uint16)
        sar_arr = rng.normal(-12.0, 3.0, (64, 64)).astype(np.float32)
        opt_path = write_test_raster(
            tmp_path / "optical.tif", opt_arr, transform=_origin(TEST_ORIGIN_X, TEST_ORIGIN_Y),
            descriptions=_BANDS,
        )
        # Shift the radar grid east by 20 px (200 m): ~69% overlap — above the compatibility floor,
        # so it is aligned rather than refused, but below 98%, so a partial-overlap warning fires.
        sar_path = write_test_raster(
            tmp_path / "sar.tif", sar_arr, transform=_origin(TEST_ORIGIN_X + 200, TEST_ORIGIN_Y),
            descriptions=("VV",),
        )
        optical = load_raster(opt_path, max_edge=None)
        sar = load_raster(sar_path, max_edge=None, modality=Modality.SAR)

        result = FusionTool().run(ToolContext("fuse the views", [optical, sar]))
        assert isinstance(result, ToolResult)  # aligned and fused, did not crash
        assert any("overlap" in w.lower() for w in result.warnings)  # geometry caveat surfaced
        assert result.status is StepStatus.WARNING


class TestSummaryPhrasing:
    """Phrasing of the branches in :meth:`FusionTool._summarize`, driven with duck-typed stubs so
    each is covered deterministically without fabricating a full ``FusionResult``: the empty-scene
    refusal, and the three arbiter cases (radar-decided, optical-decided, and unresolved), plus the
    no-conflict path where the disagreement clause must be omitted rather than invented."""

    def test_empty_scene_declines_instead_of_indexing_empty(self):
        stub = SimpleNamespace(stats=[])
        assert FusionTool._summarize(stub).startswith("Neither the optical")

    def test_radar_arbitrated_conflict_is_attributed_to_radar(self):
        stub = SimpleNamespace(
            stats=[SimpleNamespace(label=LandCoverClass.BUILT_UP, fraction=0.40)],
            corroborated_fraction=0.80,
            conflicts=[
                SimpleNamespace(arbiter="sar", fraction=0.052, resolved_as=LandCoverClass.BUILT_UP)
            ],
        )
        sentence = FusionTool._summarize(stub)
        assert "predominantly built up" in sentence.lower()  # dominant class named from stats
        assert "80%" in sentence  # corroboration reported, not invented
        assert "resolved by radar" in sentence.lower()  # "sar" -> "radar"
        assert "5.2%" in sentence  # the conflict's measured fraction

    def test_optically_arbitrated_conflict_is_attributed_to_optical(self):
        stub = SimpleNamespace(
            stats=[SimpleNamespace(label=LandCoverClass.VEGETATION, fraction=0.55)],
            corroborated_fraction=0.60,
            conflicts=[
                SimpleNamespace(arbiter="optical", fraction=0.03, resolved_as=LandCoverClass.WATER)
            ],
        )
        sentence = FusionTool._summarize(stub)
        assert "the optical bands" in sentence.lower()  # "optical" -> "the optical bands"
        assert "in favour of water" in sentence.lower()

    def test_unresolved_conflict_is_recorded_not_attributed(self):
        stub = SimpleNamespace(
            stats=[SimpleNamespace(label=LandCoverClass.WATER, fraction=0.60)],
            corroborated_fraction=0.50,
            conflicts=[
                SimpleNamespace(arbiter="none", fraction=0.03, resolved_as=LandCoverClass.WATER)
            ],
        )
        sentence = FusionTool._summarize(stub)
        assert "could not be resolved" in sentence.lower()  # arbiter with no mapped word
        assert "recorded as-is" in sentence.lower()

    def test_no_conflict_omits_the_disagreement_clause(self):
        stub = SimpleNamespace(
            stats=[SimpleNamespace(label=LandCoverClass.VEGETATION, fraction=0.90)],
            corroborated_fraction=1.0,
            conflicts=[],  # perfect agreement -> nothing to arbitrate
        )
        sentence = FusionTool._summarize(stub)
        assert "predominantly vegetation" in sentence.lower()
        assert "disagreement" not in sentence.lower()  # clause skipped, not invented
