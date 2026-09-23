"""End-to-end tests for the two demo bi-temporal pairs (brief §3, §5, §6, §10, §12).

Two pairs, two required outcomes, no mocks anywhere:

* ``data/demo/temporal_reliable/`` — both dates are 6-band GeoTIFFs on one identical
  georeferenced grid with a known reservoir impoundment. This pair must produce a *complete*
  change analysis: sub-pixel registration, a passed registration gate, a named land-cover
  conversion corroborated by the independent spectral detector, real ground areas, and HIGH
  confidence with no warnings.
* ``data/demo/temporal_unreliable/`` — the same ground, but the later date is rotated 4° and
  neither file carries a CRS. The rotation is the point: co-registration here models translation
  only, so it cannot be removed, and the pipeline must still deliver the measured pixel
  difference while *withholding* any named conversion, warning, and refusing HIGH confidence.

Every number asserted here was measured from the generated data, and the reliable pair reaches
HIGH without any threshold in ``change.py``, ``confidence.py``, ``landcover.py``, ``indices.py``
or ``align.py`` having been relaxed for it. The ground-truth cross-check reads
``manifest.json``, which the generator writes from the label maps rather than from the
classifier, so the assertions compare the analysis against the scene's design and not against
its own output.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.agents.tools import ToolContext
from app.agents.tools.base import ToolResult
from app.agents.tools.change import ChangeTool
from app.core.types import ChangeType, ConfidenceLevel, LandCoverClass, StepStatus
from app.geospatial.align import coregister, measure_registration, registration_gate
from app.geospatial.raster import RasterData, load_raster


@pytest.fixture(scope="module")
def reliable(demo_root):
    """The preferred bi-temporal input: georeferenced, co-registered, known change."""
    root = demo_root / "temporal_reliable"
    return (
        load_raster(root / "reservoir_t1_optical.tif", max_edge=None),
        load_raster(root / "reservoir_t2_optical.tif", max_edge=None),
    )


@pytest.fixture(scope="module")
def unreliable(demo_root):
    """The safety case: no CRS, and a rotation translation-only alignment cannot remove."""
    root = demo_root / "temporal_unreliable"
    return (
        load_raster(root / "pair_t1.tif", max_edge=None),
        load_raster(root / "pair_t2.tif", max_edge=None),
    )


@pytest.fixture(scope="module")
def reliable_result(reliable) -> ToolResult:
    before, after = reliable
    return ChangeTool().run(ToolContext("what changed between the two dates?", [before, after]))


@pytest.fixture(scope="module")
def unreliable_result(unreliable) -> ToolResult:
    before, after = unreliable
    return ChangeTool().run(ToolContext("what changed between the two dates?", [before, after]))


# ---------------------------------------------------------------------------
# The reliable pair: a complete change analysis
# ---------------------------------------------------------------------------
class TestReliablePairInputIsFitForSemanticClaims:
    """Point 2 of the brief: proper GeoTIFF with CRS, transform and bounds is the demo input."""

    def test_both_dates_are_georeferenced_on_the_same_grid(self, reliable):
        before, after = reliable
        for raster in (before, after):
            assert raster.metadata.is_georeferenced
            assert raster.metadata.crs_epsg == 32643
            assert raster.metadata.transform is not None
            assert raster.metadata.bounds is not None
            assert raster.metadata.pixel_size == (10.0, 10.0)
            assert raster.metadata.is_north_up
        assert before.metadata.transform == after.metadata.transform
        assert before.shape == after.shape

    def test_both_dates_carry_the_six_bands_the_indices_need(self, reliable):
        """NDVI, MNDWI and NDBI all exist here, so no cue falls back to raw bands."""
        for raster in reliable:
            roles = {r.value for r in raster.band_roles}
            assert {"red", "nir", "green", "swir1"} <= roles


class TestReliablePairRegistrationQualityIsHigh:
    """Points 3 and 5: spatial compatibility is checked, and the gate is explicit."""

    def test_the_pair_is_aligned_to_a_fraction_of_a_pixel(self, reliable):
        before, after = reliable
        quality = measure_registration(after, before)
        assert quality.offset_magnitude < 0.25
        assert quality.ncc > 0.5
        assert quality.score > 0.75
        assert quality.warnings == []

    def test_the_gate_passes_and_says_so_with_no_reasons(self, reliable):
        before, after = reliable
        gate = registration_gate(measure_registration(after, before))
        assert gate.passed is True
        assert gate.reasons == []
        assert gate.measured["ncc"] >= gate.thresholds["min_ncc"]
        assert gate.measured["score"] >= gate.thresholds["min_score"]
        assert gate.measured["offset_magnitude_px"] <= gate.thresholds["max_offset_px"]

    def test_nothing_needed_correcting(self, reliable):
        """An already-aligned pair must be left alone rather than nudged."""
        before, after = reliable
        result = coregister(after, before)
        assert result.method == "none"
        assert result.applied is False
        assert (result.shift_x, result.shift_y) == (0.0, 0.0)


class TestReliablePairProducesACompleteAnalysis:
    def test_the_tool_succeeds_without_warnings(self, reliable_result):
        assert isinstance(reliable_result, ToolResult)
        assert reliable_result.tool_name == "change-cva-cv"
        assert reliable_result.warnings == []
        assert reliable_result.status is StepStatus.OK

    def test_confidence_is_high_and_names_its_limiting_factor(self, reliable_result):
        confidence = reliable_result.confidence
        assert confidence.level is ConfidenceLevel.HIGH
        assert confidence.score >= 0.75
        # The score is a real bound, not a label: registration is the binding constraint here,
        # and it is named so a reader can see what would have to improve to raise the score.
        assert confidence.limiting_factor == "registration"
        assert confidence.factors

    def test_the_measured_difference_is_reported_in_both_pixels_and_ground_area(
        self, reliable_result
    ):
        """Point 9's *measured* half: the pixel difference, with a real area behind it."""
        data = reliable_result.data
        assert data["changed_pixels"] > 0
        assert 0.0 < data["changed_fraction"] < 0.25
        assert data["changed_area_m2"] is not None
        # 10 m pixels, so 100 m² each — the area must be derived, not invented.
        assert data["changed_area_m2"] == pytest.approx(data["changed_pixels"] * 100.0, rel=1e-6)

    def test_the_change_is_corroborated_by_both_detectors(self, reliable_result):
        split = reliable_result.data["evidence_split"]
        assert split["corroborated_pixels"] > 0
        assert split["corroborated_fraction"] > 0.75
        # Class disagreement alone is the weak evidence class; it must be the small remainder.
        assert split["class_only_pixels"] < split["corroborated_pixels"]

    def test_a_named_land_cover_conversion_is_asserted_and_justified(self, reliable_result):
        """Point 8: the conversion is claimed only because the spectral detector agrees."""
        semantic = reliable_result.data["semantic"]
        assert semantic["supported"] is True
        assert semantic["withheld_reason"] is None

        transition = semantic["transition"]
        assert transition is not None
        assert transition["from"] == LandCoverClass.VEGETATION.value
        assert transition["to"] == LandCoverClass.WATER.value
        assert transition["change_type"] == ChangeType.WATER_GAIN.value
        assert transition["semantically_reportable"] is True
        assert transition["spectral_agreement"] >= semantic["min_spectral_agreement"], (
            "a conversion may only be named when the independent detector corroborates it"
        )
        assert transition["area_m2"] == pytest.approx(transition["pixel_count"] * 100.0, rel=1e-6)

    def test_every_transition_carries_whether_it_may_be_asserted(self, reliable_result):
        """Point 9: the renderer must never have to re-derive the rule."""
        transitions = reliable_result.data["transitions"]
        assert transitions
        for transition in transitions:
            assert isinstance(transition["semantically_reportable"], bool)

    def test_the_answer_states_the_measurement_and_the_conversion(self, reliable_result):
        answer = reliable_result.answer.lower()
        assert "%" in answer
        assert "vegetation to water" in answer
        assert "corroborated" in answer
        assert "insufficient evidence" not in answer


class TestReliablePairAgreesWithTheKnownGroundTruth:
    """The change is *known*, so the analysis is scored against the scene's design.

    ``manifest.json`` is written by the generator from its own label maps, before any
    classifier runs, so these are independent numbers rather than a restatement of the result.
    """

    def test_the_manifest_declares_the_change_this_pair_contains(self, demo_manifest):
        pair = demo_manifest["reliable_pair"]
        assert pair["transition_pixels"]
        dominant = max(pair["transition_pixels"].items(), key=lambda kv: kv[1])
        assert dominant[0] == "vegetation->water"
        assert pair["changed_pixels"] == sum(pair["transition_pixels"].values())

    def test_the_measured_extent_matches_the_designed_extent(self, reliable_result, demo_manifest):
        truth = demo_manifest["reliable_pair"]
        measured = reliable_result.data["changed_pixels"]
        # 20% tolerance: the detector works on reflectance, so it picks up the shoreline the
        # label map draws as a hard edge. A tighter bound would be fitting to this scene.
        assert measured == pytest.approx(truth["changed_pixels"], rel=0.2)
        assert reliable_result.data["changed_area_m2"] == pytest.approx(
            truth["changed_area_m2"], rel=0.2
        )

    def test_the_named_conversion_is_the_one_the_generator_drew(
        self, reliable_result, demo_manifest
    ):
        truth = demo_manifest["reliable_pair"]["transition_pixels"]
        transition = reliable_result.data["semantic"]["transition"]
        assert f"{transition['from']}->{transition['to']}" in truth
        assert transition["pixel_count"] == pytest.approx(truth["vegetation->water"], rel=0.2)

    def test_water_grew_and_vegetation_shrank_by_the_designed_amounts(
        self, reliable_result, demo_manifest
    ):
        """The class deltas are a second, independent view of the same impoundment."""
        pair = demo_manifest["reliable_pair"]
        expected_water = pair["t2_class_pixels"]["water"] - pair["t1_class_pixels"]["water"]
        deltas = {d["class"]: d for d in reliable_result.data["class_deltas"]}
        water = deltas[LandCoverClass.WATER.value]
        assert water["pixel_delta"] > 0
        assert water["pixel_delta"] == pytest.approx(expected_water, rel=0.25)
        assert deltas[LandCoverClass.VEGETATION.value]["pixel_delta"] < 0


# ---------------------------------------------------------------------------
# The unreliable pair: the safe warning must survive
# ---------------------------------------------------------------------------
class TestUnreliablePairIsRecognisedAsUnfitForSemanticClaims:
    def test_neither_date_is_a_map(self, unreliable):
        for raster in unreliable:
            assert raster.metadata.is_georeferenced is False
            assert raster.metadata.crs_epsg is None

    def test_the_gate_fails_and_states_a_measured_reason(self, unreliable):
        before, after = unreliable
        corrected = coregister(after, before)
        gate = registration_gate(corrected.final)
        assert gate.passed is False
        assert gate.reasons
        joined = " ".join(gate.reasons).lower()
        assert "structural agreement" in joined
        assert gate.measured["ncc"] < gate.thresholds["min_ncc"]

    def test_co_registration_still_tries_and_still_cannot_rescue_it(self, unreliable):
        """Point 4 does not become point 5's excuse.

        Translation-only alignment does improve the score here — the later date really is
        shifted as well as rotated — but the 4° rotation it cannot model leaves the structural
        agreement far below the gate. Doing the best available correction and *still* refusing is
        the behaviour under test; a rotation-capable aligner would be a different claim.
        """
        before, after = unreliable
        corrected = coregister(after, before)
        assert corrected.applied is True
        assert corrected.improvement > 0.0
        assert corrected.final.ncc < 0.35
        assert registration_gate(corrected.final).passed is False


class TestUnreliablePairStillDeliversTheMeasurementSafely:
    def test_the_tool_does_not_fail_and_does_not_pretend(self, unreliable_result):
        """Distinguishing (A) analysis ran, evidence weak from (B) the tool broke (§27)."""
        assert isinstance(unreliable_result, ToolResult)
        assert unreliable_result.tool_name == "change-cva-cv"
        assert unreliable_result.status is StepStatus.WARNING
        assert unreliable_result.warnings

    def test_confidence_is_not_high(self, unreliable_result):
        confidence = unreliable_result.confidence
        assert confidence.level is not ConfidenceLevel.HIGH
        assert confidence.score < 0.75
        assert confidence.limiting_factor == "registration"

    def test_the_pixel_difference_is_still_measured_and_reported(self, unreliable_result):
        """Point 9's two halves come apart here: the measurement survives, the meaning does not."""
        data = unreliable_result.data
        assert data["changed_pixels"] > 0
        assert data["changed_fraction"] > 0.0
        # No CRS and an identity transform, so there is no ground area to report. Inventing one
        # from a nominal pixel size would be fabricated geospatial data.
        assert data["changed_area_m2"] is None

    def test_no_land_cover_conversion_is_named_and_the_refusal_gives_its_reason(
        self, unreliable_result
    ):
        """Point 8, in its negative direction: the transitions exist but none may be asserted."""
        semantic = unreliable_result.data["semantic"]
        assert semantic["supported"] is False
        assert semantic["transition"] is None
        assert semantic["withheld_reason"]
        assert unreliable_result.data["registration_gate"]["passed"] is False
        for transition in unreliable_result.data["transitions"]:
            assert transition["semantically_reportable"] is False

    def test_the_answer_reports_the_difference_without_claiming_a_conversion(
        self, unreliable_result
    ):
        answer = unreliable_result.answer.lower()
        assert "%" in answer
        assert "to water" not in answer
        assert "to bare soil" not in answer


# ---------------------------------------------------------------------------
# Co-registration: the correction is real, and so is its refusal to guess
# ---------------------------------------------------------------------------
class TestCoregistrationDoesRealWorkAndReportsWhatItDid:
    """Point 4, measured on three inputs whose right answer is known in advance."""

    def test_an_aligned_pair_is_left_untouched(self, reliable):
        before, _ = reliable
        result = coregister(before, before)
        assert result.method == "none"
        assert result.applied is False
        assert result.improvement == pytest.approx(0.0, abs=1e-9)

    def test_a_deliberate_shift_is_recovered_to_a_tenth_of_a_pixel(self, reliable):
        """The only test here with a ground-truth shift: 12 px east, 7 px north, injected.

        Both candidate methods are given the same problem and the better-scoring one is kept, so
        this asserts the selection as well as the correction.
        """
        before, after = reliable
        shifted = RasterData(
            np.stack([np.roll(np.roll(b, -7, axis=0), 12, axis=1) for b in after.data]),
            after.metadata,
            after.band_roles,
            after.modality,
        )
        initial = measure_registration(shifted, before)
        assert initial.offset_magnitude == pytest.approx(13.9, abs=0.5)

        result = coregister(shifted, before)
        assert result.applied is True
        assert result.method == "orb-translation"
        assert result.shift_x == pytest.approx(-12.0, abs=0.5)
        assert result.shift_y == pytest.approx(7.0, abs=0.5)
        assert result.final.offset_magnitude < 0.5
        assert result.final.score > initial.score + 0.5

    def test_the_kept_method_is_the_best_scoring_candidate_and_the_others_are_recorded(
        self, reliable
    ):
        """§26/§49: the trace must show what was tried, not only what won."""
        before, after = reliable
        shifted = RasterData(
            np.stack([np.roll(np.roll(b, -7, axis=0), 12, axis=1) for b in after.data]),
            after.metadata,
            after.band_roles,
            after.modality,
        )
        result = coregister(shifted, before)
        methods = {c["method"]: c["score"] for c in result.candidates}
        assert {"none", "pyramid-phase-correlation", "orb-translation"} <= set(methods)
        assert methods[result.method] == max(methods.values())
        assert methods[result.method] > methods["none"]

    def test_a_correction_that_does_not_help_is_rejected_rather_than_applied(self, reliable):
        """Noise against the real scene: there is no shift to find, so none may be claimed."""
        before, _ = reliable
        rng = np.random.default_rng(0)
        noise = RasterData(
            rng.integers(0, 6000, size=before.data.shape).astype(before.data.dtype),
            before.metadata,
            before.band_roles,
            before.modality,
        )
        result = coregister(noise, before)
        assert result.improvement < 0.05
        assert result.final.score < 0.35
        assert registration_gate(result.final).passed is False
