"""Tests for the change tool — the first *pair* tool (brief §3, §17, §18).

Runs against the real demo bi-temporal pair (no mocks): two dates of the same area must produce a
measured, corroborated change report, and the tool must (a) refuse inputs it cannot honestly compare
— the wrong number of images, or an optical/SAR pair that is a cross-modal question rather than a
change one — and (b) own the alignment step the change *service* deliberately does not: reject a
non-overlapping pair, and resample a partially-overlapping one while surfacing the geometry warning.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import Affine

from app.agents.tools import ToolContext
from app.agents.tools.base import ToolResult, ToolTier
from app.agents.tools.change import ChangeTool
from app.core.errors import ErrorCode, GeospatialError, ValidationError
from app.core.types import ConfidenceLevel, Modality, QueryTask, StepStatus
from app.geospatial.raster import load_raster
from tests.conftest import TEST_ORIGIN_X, TEST_ORIGIN_Y, TEST_PIXEL_M, write_test_raster

_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")


@pytest.fixture(scope="module")
def temporal(demo_root):
    before = load_raster(demo_root / "temporal" / "scene_t1_optical.tif", max_edge=None)
    after = load_raster(demo_root / "temporal" / "scene_t2_optical.tif", max_edge=None)
    return before, after


@pytest.fixture(scope="module")
def sar(demo_root):
    return load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None, modality=Modality.SAR)


def _origin(x: float, y: float) -> Affine:
    return Affine(TEST_PIXEL_M, 0.0, x, 0.0, -TEST_PIXEL_M, y)


class TestSuccess:
    @pytest.fixture(scope="class")
    def result(self, temporal) -> ToolResult:
        before, after = temporal
        return ChangeTool().run(ToolContext("what changed between the two dates?", [before, after]))

    def test_identifies_itself_honestly(self, result):
        assert isinstance(result, ToolResult)
        assert result.task is QueryTask.CHANGE_DETECTION
        assert result.tool_name == "change-cva-cv"
        assert result.tier is ToolTier.CLASSICAL  # deterministic CV, not a trained model

    def test_answer_is_measurement_grounded(self, result):
        assert result.answer
        assert "%" in result.answer  # phrased from the measured changed fraction

    def test_real_change_is_not_insufficient(self, result):
        # The demo pair contains genuine change; confidence must reflect measured evidence.
        assert result.confidence.level is not ConfidenceLevel.INSUFFICIENT

    def test_status_is_consistent_with_its_inputs(self, result):
        if result.warnings or result.confidence.level is ConfidenceLevel.INSUFFICIENT:
            assert result.status is StepStatus.WARNING
        else:
            assert result.status is StepStatus.OK

    def test_evidence_names_detectors_and_registration(self, result):
        joined = " ".join(result.evidence).lower()
        assert "detector" in joined  # two-detector method is disclosed
        assert "registration" in joined  # alignment residual is disclosed

    def test_raw_present_but_not_serialised(self, result):
        assert result.raw is not None  # the ChangeResult, for rendering the change mask
        payload = json.loads(json.dumps(result.to_dict()))
        assert "raw" not in payload
        assert payload["data"]["detectors"]  # per-detector provenance carried through
        assert "changed_fraction" in payload["data"]

    def test_describe(self):
        d = ChangeTool().describe()
        assert d["name"] == "change-cva-cv"
        assert d["tier"] == "classical"
        assert d["task"] == QueryTask.CHANGE_DETECTION.value
        assert d["summary"]


class TestRejects:
    def test_one_image_is_missing_second(self, temporal):
        before, _ = temporal
        with pytest.raises(ValidationError) as ei:
            ChangeTool().run(ToolContext("what changed?", [before]))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_three_images_is_too_many(self, temporal):
        before, after = temporal
        with pytest.raises(ValidationError) as ei:
            ChangeTool().run(ToolContext("what changed?", [before, after, before]))
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES

    def test_optical_sar_pair_is_wrong_modality(self, temporal, sar):
        """An optical+SAR pair is a cross-modal question, not a change one — redirect, don't diff."""
        before, _ = temporal
        with pytest.raises(ValidationError) as ei:
            ChangeTool().run(ToolContext("what changed?", [before, sar]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY

    def test_non_overlapping_pair_is_refused(self, tmp_path):
        """Two georeferenced scenes with no common ground cannot be compared — refuse honestly."""
        rng = np.random.default_rng(1)
        arr = (rng.random((6, 48, 48)) * 10000).astype(np.uint16)
        a = write_test_raster(
            tmp_path / "a.tif", arr, transform=_origin(TEST_ORIGIN_X, TEST_ORIGIN_Y),
            descriptions=_BANDS,
        )
        # 100 km east/south — no overlap with A.
        b = write_test_raster(
            tmp_path / "b.tif", arr, transform=_origin(TEST_ORIGIN_X + 100_000, TEST_ORIGIN_Y - 100_000),
            descriptions=_BANDS,
        )
        before = load_raster(a, max_edge=None)
        after = load_raster(b, max_edge=None)
        with pytest.raises(GeospatialError) as ei:
            ChangeTool().run(ToolContext("what changed?", [before, after]))
        assert ei.value.code == ErrorCode.NO_SPATIAL_OVERLAP


class TestRejectsBeforeExecution:
    """A pair with no band in common is refused at the gate, naming the cause (§6, §27, §38).

    :func:`app.services.change.detect_change` already refuses this — but only after alignment and
    reprojection have run, and by reporting the symptom ("no bands in common") rather than the cause
    the user can act on. The commonest cause is a sensor that could not be determined from the file:
    a SAR scene exported with a neutral filename and an unlabelled band loads as ``UNKNOWN`` with a
    single ``gray`` band, pairs with an optical scene as a bi-temporal request, and lands here.
    """

    @pytest.fixture
    def unlabelled_single_band(self, tmp_path):
        """A one-band scene with no band description — modality and band role both undetermined."""
        rng = np.random.default_rng(4)
        arr = (rng.random((1, 48, 48)) * 400).astype(np.uint16)
        path = write_test_raster(
            tmp_path / "subset_1.tif", arr, transform=_origin(TEST_ORIGIN_X, TEST_ORIGIN_Y)
        )
        raster = load_raster(path, max_edge=None)
        assert raster.modality is Modality.UNKNOWN  # the premise of the case
        return raster

    def test_no_shared_band_is_refused_at_the_gate(self, temporal, unlabelled_single_band):
        before, _ = temporal
        with pytest.raises(ValidationError) as ei:
            ChangeTool().validate_input(
                ToolContext("what changed?", [before, unlabelled_single_band])
            )
        assert ei.value.code == ErrorCode.MISSING_BAND

    def test_refusal_names_the_undetermined_sensor_and_the_remedy(
        self, temporal, unlabelled_single_band
    ):
        before, _ = temporal
        with pytest.raises(ValidationError) as ei:
            ChangeTool().validate_input(
                ToolContext("what changed?", [before, unlabelled_single_band])
            )
        message = ei.value.message
        assert "image 2" in message  # which of the two, not just "an image"
        assert "could not be determined" in message
        assert "modality_hint" in message  # the actionable remedy
        # And the structured context carries the facts behind the refusal.
        assert ei.value.context["modalities"] == ["optical", "unknown"]
        assert ei.value.context["band_roles"][1] == ["gray"]

    def test_two_unlabelled_scenes_share_gray_and_are_accepted(self, tmp_path):
        """Two undetermined scenes are not refused: they share the ``gray`` band, so a comparison
        is possible. The refusal is about shared bands, not about modality being unknown."""
        rng = np.random.default_rng(5)
        a = load_raster(
            write_test_raster(
                tmp_path / "one.tif", (rng.random((1, 48, 48)) * 400).astype(np.uint16),
                transform=_origin(TEST_ORIGIN_X, TEST_ORIGIN_Y),
            ),
            max_edge=None,
        )
        b = load_raster(
            write_test_raster(
                tmp_path / "two.tif", (rng.random((1, 48, 48)) * 400).astype(np.uint16),
                transform=_origin(TEST_ORIGIN_X, TEST_ORIGIN_Y),
            ),
            max_edge=None,
        )
        ChangeTool().validate_input(ToolContext("what changed?", [a, b]))  # does not raise

    def test_two_sar_scenes_share_their_backscatter_band(self, sar):
        """Two radar scenes are a supported change comparison — the router now defaults to it."""
        ChangeTool().validate_input(ToolContext("what changed?", [sar, sar]))  # does not raise


class TestAlignsAndSurfacesGeometry:
    """A partially-overlapping pair on the same CRS must be resampled onto a common grid, with the
    partial-overlap geometry warning surfaced to the user rather than silently dropped (§8, §32)."""

    def test_partial_overlap_is_aligned_and_warned(self, tmp_path):
        rng = np.random.default_rng(2)
        arr = (rng.random((6, 64, 64)) * 10000).astype(np.uint16)
        a = write_test_raster(
            tmp_path / "a.tif", arr, transform=_origin(TEST_ORIGIN_X, TEST_ORIGIN_Y),
            descriptions=_BANDS,
        )
        # Shift east by 20 px (200 m): ~69% overlap — above the compatibility floor, so it is
        # aligned rather than refused, but below 98%, so a partial-overlap warning is raised.
        b = write_test_raster(
            tmp_path / "b.tif", arr, transform=_origin(TEST_ORIGIN_X + 200, TEST_ORIGIN_Y),
            descriptions=_BANDS,
        )
        before = load_raster(a, max_edge=None)
        after = load_raster(b, max_edge=None)
        result = ChangeTool().run(ToolContext("what changed?", [before, after]))
        assert isinstance(result, ToolResult)  # aligned and analysed, did not crash
        assert any("overlap" in w.lower() for w in result.warnings)  # geometry caveat surfaced
        assert result.status is StepStatus.WARNING


class TestNoChange:
    """Identical dates: the tool must state that no change was detected — not invent one — and its
    explanation must skip the corroboration/transition lines that have nothing to describe."""

    @pytest.fixture(scope="class")
    def result(self, temporal) -> ToolResult:
        before, _ = temporal
        return ChangeTool().run(ToolContext("what changed?", [before, before]))

    def test_declines_instead_of_inventing_change(self, result):
        assert result.answer.startswith("No change large enough")
        assert result.raw.changed_pixels == 0
        assert result.confidence.level is ConfidenceLevel.INSUFFICIENT

    def test_explain_skips_absent_sections(self, result):
        lines = result.evidence
        assert any("method" in line.lower() for line in lines)
        assert any("registration" in line.lower() for line in lines)
        assert not any("corroborated" in line.lower() for line in lines)  # nothing changed
        assert not any("dominant transition" in line.lower() for line in lines)  # no transition


class TestSummaryPhrasing:
    """Phrasing of the conditional clauses in :meth:`ChangeTool._summarize`, for the cases the
    demo pair does not exercise: change with no georeferenced area, spectral-only change with no
    land-cover conversion to name, and a conversion the evidence does not support asserting.
    Driven with a duck-typed stub so the branches are covered without fabricating a full
    ``ChangeResult``."""

    def test_change_without_area_or_named_transition(self):
        stub = SimpleNamespace(
            changed_pixels=100,
            changed_fraction=0.12,
            changed_area_m2=None,  # no georeference -> the km² clause must be omitted
            corroborated_fraction=0.5,
            dominant_transition=lambda: None,  # spectral-only change, no class conversion to name
            semantic_transition=lambda: None,
            semantic_withheld_reason=None,  # nothing was withheld: there was nothing to withhold
        )
        sentence = ChangeTool._summarize(stub)
        assert "12.0%" in sentence  # measured fraction still reported
        assert "km" not in sentence  # area clause skipped, not invented as 0
        assert "dominant transition" not in sentence.lower()  # transition clause skipped
        assert "corroborated" in sentence.lower()  # corroboration clause always present
        # No conversion existed, so the sentence must not imply one was suppressed either.
        assert "no specific land-cover conversion" not in sentence.lower()

    def test_withheld_conversion_is_stated_not_silently_dropped(self):
        """A measured transition the evidence cannot support must be *reported as withheld*.

        Silently omitting it would leave the reader of a transitions table with nine rows and no
        explanation of why none is named; asserting it is the fabrication the brief forbids. The
        only honest third option is to say the difference was measured and the interpretation
        withheld, with the reason.
        """
        stub = SimpleNamespace(
            changed_pixels=5000,
            changed_fraction=0.31,
            changed_area_m2=None,
            corroborated_fraction=0.11,
            dominant_transition=lambda: SimpleNamespace(),
            semantic_transition=lambda: None,  # gate or corroboration refused it
            semantic_withheld_reason="Registration quality 0.08 is below the 0.45 required.",
        )
        sentence = ChangeTool._summarize(stub)
        assert "31.0%" in sentence  # the measurement is still delivered
        assert "no specific land-cover conversion is claimed" in sentence.lower()
        assert "0.45" in sentence  # the reason travels with the refusal
