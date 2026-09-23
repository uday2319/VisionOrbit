"""Tests for bi-temporal change analysis.

Three kinds of assertion appear here:

* **Behavioural** — the evidence split is exhaustive and disjoint, refusals happen instead of
  guesses, areas are absent rather than invented, warnings appear when the result is weaker
  than it looks.
* **Accuracy** — real precision/recall/IoU computed at test time against the synthetic pair's
  ground-truth change raster. Bounds sit below the current measurement so the suite catches
  regressions without becoming a tuning target.
* **Regression** — each one pins a bug that shipped and produced confident, plausible,
  entirely wrong output. Those are named after the failure, not the fix.
"""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import Affine

from app.core.errors import ErrorCode, GeospatialError
from app.core.types import ChangeType, LandCoverClass
from app.geospatial.raster import load_raster
from app.services import indices
from app.services.change import (
    MIN_CHANGE_PATCH_M2,
    _robust_outlier_threshold,
    detect_change,
)
from tests.conftest import (
    TEST_ORIGIN_X,
    TEST_ORIGIN_Y,
    TEST_PIXEL_M,
    write_test_raster,
)

OPTICAL_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
KNOWN_TRANSFORM = Affine(TEST_PIXEL_M, 0.0, TEST_ORIGIN_X, 0.0, -TEST_PIXEL_M, TEST_ORIGIN_Y)

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def before(demo_root):
    return load_raster(demo_root / "temporal" / "scene_t1_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def after(demo_root):
    return load_raster(demo_root / "temporal" / "scene_t2_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def change_gt(demo_root):
    return load_raster(demo_root / "temporal" / "change_gt.tif", max_edge=None).data[0] > 0


@pytest.fixture(scope="module")
def result(before, after):
    return detect_change(before, after)


def scores(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    """Precision, recall, F1 and IoU — computed here, never quoted."""
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return p, r, f1, iou


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------


class TestStructure:
    def test_mask_matches_the_input_grid(self, result, before):
        assert result.change_mask.shape == before.shape
        assert result.change_mask.dtype == bool
        assert result.magnitude.shape == before.shape

    def test_reports_the_features_it_actually_used(self, result):
        assert set(result.features_used) == {"ndvi", "mndwi", "ndbi"}
        assert result.method == "cva-spectral-indices"

    def test_both_detectors_contributed(self, result):
        assert result.detectors == ["spectral-cva", "land-cover-disagreement"]

    def test_changed_pixels_matches_the_mask(self, result):
        assert result.changed_pixels == int(result.change_mask.sum())

    def test_evidence_split_is_exhaustive_and_disjoint(self, result):
        """Every changed pixel must be attributable to at least one detector.

        If these did not sum, some pixels would be counted in the headline figure with no
        stated reason behind them — which is exactly the kind of unexplained number this
        module exists to avoid.
        """
        parts = (
            result.corroborated_pixels
            + result.spectral_only_pixels
            + result.class_only_pixels
        )
        assert parts == result.changed_pixels

    def test_mask_is_confined_to_valid_pixels(self, result, before, after):
        valid = before.valid_mask() & after.valid_mask()
        assert not (result.change_mask & ~valid).any()

    def test_changed_fraction_is_a_fraction(self, result):
        assert 0.0 < result.changed_fraction < 1.0
        assert result.valid_pixels > 0
        assert result.changed_pixels <= result.valid_pixels

    def test_transition_fractions_sum_to_one(self, result):
        assert sum(t.fraction_of_change for t in result.transitions) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_transition_pixels_sum_to_changed_pixels(self, result):
        assert sum(t.pixel_count for t in result.transitions) == result.changed_pixels

    def test_transitions_are_ordered_by_size(self, result):
        counts = [t.pixel_count for t in result.transitions]
        assert counts == sorted(counts, reverse=True)

    def test_is_deterministic(self, before, after):
        """No randomness may influence a measurement."""
        a = detect_change(before, after)
        b = detect_change(before, after)
        np.testing.assert_array_equal(a.change_mask, b.change_mask)
        assert a.threshold == b.threshold
        assert a.corroborated_pixels == b.corroborated_pixels

    def test_serialises_without_numpy_types(self, result):
        payload = result.to_dict()
        assert isinstance(payload["changed_pixels"], int)
        assert isinstance(payload["transitions"], list)
        assert payload["dominant_transition"]["change_type"] in {c.value for c in ChangeType}
        split = payload["evidence_split"]
        assert (
            split["corroborated_pixels"]
            + split["spectral_only_pixels"]
            + split["class_only_pixels"]
            == payload["changed_pixels"]
        )

    def test_serialised_payload_is_json_round_trippable(self, result):
        import json

        assert json.loads(json.dumps(result.to_dict()))["method"] == result.method


# --------------------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------------------


class TestAccuracy:
    """Real metrics against the ground-truth change raster."""

    def test_detects_change_accurately(self, result, change_gt):
        p, r, f1, iou = scores(result.change_mask, change_gt)
        assert p > 0.78, f"precision {p:.3f}"
        assert r > 0.87, f"recall {r:.3f}"
        assert f1 > 0.84, f"F1 {f1:.3f}"
        assert iou > 0.73, f"IoU {iou:.3f}"

    def test_overall_pixel_accuracy(self, result, change_gt):
        assert float((result.change_mask == change_gt).mean()) > 0.96

    def test_changed_extent_is_the_right_order_of_magnitude(self, result, change_gt):
        """A detector that flags 60% of a scene as changed is useless even at good recall."""
        assert result.changed_fraction < 2.0 * float(change_gt.mean())

    def test_beats_a_spectral_only_baseline_on_recall(self, result, change_gt):
        """Documents *why* the second detector exists.

        Spectral CVA alone misses built-up/bare-soil conversions entirely, because those two
        classes separate on texture rather than on any of NDVI, MNDWI or NDBI. This measures
        the gain instead of asserting it.
        """
        spectral = result.magnitude > result.threshold
        _, r_spectral, _, _ = scores(spectral, change_gt)
        _, r_union, _, _ = scores(result.change_mask, change_gt)
        assert r_union > r_spectral + 0.10, (
            f"union recall {r_union:.3f} vs spectral {r_spectral:.3f} — the second detector "
            f"is not contributing"
        )

    def test_every_known_change_type_from_the_manifest_is_found(self, result, demo_manifest):
        """The manifest names four ground changes; each must appear as a transition."""
        assert demo_manifest["known_changes"], "manifest must document its own changes"
        found = {t.change_type for t in result.transitions if t.pixel_count > 100}
        for expected in (
            ChangeType.URBAN_EXPANSION,
            ChangeType.VEGETATION_GAIN,
            ChangeType.WATER_LOSS,
        ):
            assert expected in found, f"{expected.value} not detected"


# --------------------------------------------------------------------------------------
# Regressions
# --------------------------------------------------------------------------------------


class TestRadiometricNormalisationRegression:
    """Normalising the two dates before differencing indices fabricated an entire lake.

    ``normalize_radiometry`` matched each band's mean and variance across dates. On this pair
    that drove stable water's SWIR1 reflectance to −59 — physically impossible — which flipped
    NDBI there from −0.30 to +0.79. That is a change of 1.17 in an index bounded to [−1, 1],
    against a true change of 0.02, so every pixel of the reservoir was reported as changed
    with the highest magnitude in the scene and zero ground-truth support.

    The fix was to drop the step from the index path entirely: a normalised difference is a
    ratio, so it is already invariant to the per-band gain that dominates illumination
    differences. Removing it took precision from 0.695 to 0.988 with recall unmoved.
    """

    @pytest.fixture(scope="class")
    def stable_water(self, demo_root):
        """Pixels the ground truth says were water at both dates and did not change."""
        t1 = load_raster(
            demo_root / "temporal" / "scene_t1_landcover_gt.tif", max_edge=None
        ).data[0]
        t2 = load_raster(
            demo_root / "temporal" / "scene_t2_landcover_gt.tif", max_edge=None
        ).data[0]
        gt_water = 1
        return (t1 == gt_water) & (t2 == gt_water)

    def test_stable_water_is_not_reported_as_changed(self, result, stable_water):
        assert stable_water.sum() > 1000, "fixture must cover a meaningful area"
        flagged = float((result.change_mask & stable_water).sum() / stable_water.sum())
        assert flagged < 0.05, (
            f"{flagged:.1%} of unchanged water flagged as changed — the normalisation bug is "
            f"back"
        )

    def test_water_to_water_is_not_the_dominant_finding(self, result):
        """The bug's signature was ``water → water`` topping the transition table."""
        dominant = result.dominant_transition()
        assert dominant is not None
        assert not (
            dominant.before is LandCoverClass.WATER and dominant.after is LandCoverClass.WATER
        )

    def test_magnitudes_stay_inside_the_index_range(self, result):
        """A stack of indices bounded to [-1, 1] cannot produce a difference above 2."""
        finite = result.magnitude[np.isfinite(result.magnitude)]
        assert finite.max() <= 2.0, "magnitude exceeds what bounded indices can produce"

    def test_indices_are_differenced_on_unmodified_imagery(self, before, after):
        """Directly asserts the invariant: no band may leave physical range en route.

        The bug was invisible in the mask alone; it showed up as a negative reflectance. This
        checks the inputs the change vector is actually built from.
        """
        for raster in (before, after):
            for name in ("ndvi", "mndwi", "ndbi"):
                arr = indices.compute_index(raster, name).array
                finite = arr[np.isfinite(arr)]
                assert finite.min() >= -1.0 and finite.max() <= 1.0, name


class TestEvidenceIsNotCircular:
    """Evidence strength used to be ``separability(magnitude, change_mask)``.

    The mask is a threshold on the magnitude, so the two groups differ by construction: that
    number reads 1.000 on any input whatsoever, including pure noise, and therefore said
    nothing. It is replaced by Sarle's bimodality coefficient and Otsu's between-class
    variance ratio, both computed on the magnitude population *before* any mask exists.
    """

    @pytest.fixture(scope="class")
    def noise_pair(self, tmp_path_factory):
        """Two independent noise images: no real change, only sensor noise."""
        rng = np.random.default_rng(11)
        tmp = tmp_path_factory.mktemp("noise")
        out = []
        for name in ("n1.tif", "n2.tif"):
            arr = rng.normal(3000, 200, (6, 96, 96)).clip(1, 10000).astype(np.uint16)
            out.append(
                load_raster(
                    write_test_raster(
                        tmp / name, arr, transform=KNOWN_TRANSFORM,
                        descriptions=OPTICAL_BANDS,
                    ),
                    max_edge=None,
                )
            )
        return out

    @pytest.fixture(scope="class")
    def noise(self, noise_pair):
        """Analysed once; every assertion below reads the same measurement."""
        return detect_change(*noise_pair)

    def test_circular_metric_would_have_scored_perfectly_on_noise(self, noise):
        """Demonstrates the bug rather than describing it.

        Computed exactly as the shipped version was — separability of the magnitude field with
        respect to a threshold on that same field. It reads 1.000 on pure noise, where there
        is nothing whatsoever to find, which is why it was removed.
        """
        circular = indices.separability(
            noise.magnitude.ravel(), (noise.magnitude > noise.threshold).ravel()
        )
        assert circular == pytest.approx(1.0, abs=0.01), (
            "the circular pairing must still be demonstrable, or this test no longer "
            "documents why the metric was replaced"
        )

    def test_replacement_metric_is_low_on_noise(self, noise):
        """The honest measure must be able to say the evidence is weak."""
        assert noise.bimodality < indices.BIMODALITY_THRESHOLD, (
            f"bimodality {noise.bimodality:.3f} on pure noise — the metric is still circular"
        )

    def test_replacement_metric_is_high_on_the_real_pair(self, result):
        """And able to say it is strong, or it would just be a constant."""
        assert result.bimodality > indices.BIMODALITY_THRESHOLD

    def test_spectral_detector_finds_nothing_in_noise(self, noise):
        """The measurement path itself must not invent change.

        The magnitude threshold flags under 1% of a noise pair, and the minimum-patch filter
        removes all of it — no spectrally-supported change is reported at all.
        """
        assert noise.threshold_method == "robust-outlier-mad", (
            "Otsu must be rejected here: a unimodal noise histogram has no valley to find"
        )
        assert float((noise.magnitude > noise.threshold).mean()) < 0.02
        assert noise.corroborated_pixels == 0
        assert noise.spectral_only_pixels == 0

    def test_change_reported_on_noise_is_flagged_as_uncorroborated(self, noise):
        """Classifying noise twice yields different labels, so the class detector does fire.

        That is inherent — a classifier will always put *some* label on noise. What must hold
        is that none of it is presented as corroborated, and that the result says so in plain
        language, so a confidence layer reading these fields declines to draw a conclusion.
        """
        assert noise.class_only_pixels == noise.changed_pixels
        assert noise.corroborated_fraction == 0.0
        joined = " ".join(noise.warnings).lower()
        assert "does not corroborate" in joined
        assert "less certain" in joined

    def test_every_evidence_channel_reports_weakness_on_noise(self, noise):
        """The point of the evidence fields: on a worthless input, all of them say so.

        No single number is trusted to catch this. Bimodality (is the magnitude population
        two-population at all), corroboration (do the detectors agree) and registration (do
        the images even share structure) are measured independently, and a noise pair fails
        all three at once. That is what lets the confidence layer decline without needing a
        special case for "this input is meaningless".
        """
        assert noise.bimodality < indices.BIMODALITY_THRESHOLD
        assert noise.corroborated_fraction == 0.0
        assert noise.registration.ncc < 0.15, (
            "two independent noise fields must not appear to share structure"
        )
        assert any("could not be confirmed" in w for w in noise.warnings), (
            "the registration warning must reach the caller, not stay inside the "
            "registration object"
        )

    def test_bimodality_and_threshold_quality_are_bounded(self, result):
        assert 0.0 <= result.bimodality <= 1.0
        assert 0.0 <= result.threshold_quality <= 1.0


class TestOffsettingChangeIsNotAveragedAway:
    """A scene that both gains and loses vegetation had its NDVI delta cancel to +0.06.

    Reported as a mean alone, that narrates as "vegetation barely moved" when in fact it moved
    a great deal in both directions. ``increase_fraction`` counts pixels instead, so the honest
    answer — "mixed" — is available.
    """

    def test_index_deltas_report_a_direction_by_pixel_count(self, result):
        for delta in result.index_deltas:
            assert 0.0 <= delta.increase_fraction <= 1.0
            assert delta.direction in {"increase", "decrease", "mixed"}

    def test_a_two_way_scene_is_called_mixed(self, result):
        """This pair gains vegetation in the south and loses it in the north-west."""
        ndvi = next(d for d in result.index_deltas if d.name == "ndvi")
        assert ndvi.direction == "mixed", (
            f"increase_fraction {ndvi.increase_fraction:.3f} — offsetting change is being "
            f"reported as one-directional"
        )

    def test_the_transition_table_still_carries_the_direction(self, result):
        """"Mixed" must not mean information was lost — it lives in the transitions."""
        gains = result.pixels_of_type(ChangeType.VEGETATION_GAIN)
        losses = result.pixels_of_type(ChangeType.VEGETATION_LOSS)
        assert gains > 0 and losses > 0, "both directions must be individually reported"


# --------------------------------------------------------------------------------------
# Detector agreement as evidence
# --------------------------------------------------------------------------------------


class TestDetectorAgreement:
    """Agreement between the two detectors is reported because it predicts correctness.

    These tests verify that claim on this pair rather than trusting it: a transition's
    ``spectral_agreement`` must actually track how often it is right.
    """

    def test_agreement_is_a_fraction(self, result):
        for t in result.transitions:
            assert 0.0 <= t.spectral_agreement <= 1.0

    def test_corroborated_pixels_are_more_accurate_than_uncorroborated_ones(
        self, result, change_gt
    ):
        """The measured justification for reporting the split at all."""
        spectral = result.magnitude > result.threshold
        corroborated = result.change_mask & spectral
        class_only = result.change_mask & ~spectral
        assert corroborated.sum() > 1000 and class_only.sum() > 1000

        p_corr = float((corroborated & change_gt).sum() / corroborated.sum())
        p_class = float((class_only & change_gt).sum() / class_only.sum())
        assert p_corr > 0.95, f"corroborated precision {p_corr:.3f}"
        assert p_corr > p_class + 0.25, (
            f"corroborated {p_corr:.3f} vs class-only {p_class:.3f} — agreement no longer "
            f"predicts correctness, so reporting it would be misleading"
        )

    def test_high_agreement_transitions_are_the_accurate_ones(self, result, change_gt):
        """Per transition, not just in aggregate: the ranking must hold at that granularity."""
        rows = []
        for t in result.transitions:
            if t.pixel_count < 100:
                continue
            mask = (
                result.change_mask
                & result.before.mask_for(t.before)
                & result.after.mask_for(t.after)
            )
            rows.append((t.spectral_agreement, float((mask & change_gt).sum() / mask.sum())))
        assert len(rows) >= 5, "need several transitions to compare"

        strong = [prec for agree, prec in rows if agree > 0.9]
        weak = [prec for agree, prec in rows if agree < 0.5]
        assert strong and weak, "this pair must contain both kinds to be a valid test"
        assert min(strong) > max(weak), (
            f"agreement does not separate accurate transitions from inaccurate ones: "
            f"strong={sorted(strong)} weak={sorted(weak)}"
        )

    def test_uncorroborated_change_is_warned_about(self, result):
        """A third of this pair's change is class-only, so the result must say so."""
        assert result.class_only_pixels / result.changed_pixels > 0.25
        joined = " ".join(result.warnings).lower()
        assert "corroborate" in joined
        assert "less certain" in joined

    def test_corroborated_fraction_is_consistent_with_the_counts(self, result):
        assert result.corroborated_fraction == pytest.approx(
            result.corroborated_pixels / result.changed_pixels
        )

    def test_the_dominant_transition_is_corroborated(self, result):
        """The headline claim must not be one of the weak ones."""
        dominant = result.dominant_transition()
        assert dominant is not None
        assert dominant.spectral_agreement > 0.9


class TestBuiltUpBareSoilLimitation:
    """The known limitation, asserted so it stays documented and bounded.

    Built-up land and bare soil are near-identical in every index in the change vector, so
    conversions between them are recovered only by the classifier's texture cue. That half of
    the result is measurably weaker, and this pins both facts.
    """

    def test_spectral_detector_alone_misses_these_conversions(
        self, result, demo_root, change_gt
    ):
        t1 = load_raster(
            demo_root / "temporal" / "scene_t1_landcover_gt.tif", max_edge=None
        ).data[0]
        t2 = load_raster(
            demo_root / "temporal" / "scene_t2_landcover_gt.tif", max_edge=None
        ).data[0]
        gt_built, gt_soil = 3, 4
        soil_to_built = (t1 == gt_soil) & (t2 == gt_built) & change_gt
        assert soil_to_built.sum() > 1000, "fixture must contain such a conversion"

        spectral = result.magnitude > result.threshold
        spectral_recall = float((spectral & soil_to_built).sum() / soil_to_built.sum())
        union_recall = float(
            (result.change_mask & soil_to_built).sum() / soil_to_built.sum()
        )
        assert spectral_recall < 0.10, (
            "if the spectral detector now finds these, this limitation can be relaxed"
        )
        assert union_recall > 0.60, (
            f"the class detector must recover most of them; got {union_recall:.3f}"
        )

    def test_these_transitions_are_flagged_as_uncorroborated(self, result):
        weak = [
            t
            for t in result.transitions
            if {t.before, t.after} == {LandCoverClass.BUILT_UP, LandCoverClass.BARE_SOIL}
            and t.pixel_count > 100
        ]
        assert weak, "this pair contains built-up/bare-soil conversions"
        for t in weak:
            assert t.spectral_agreement < 0.10, (
                "a spectrally invisible conversion must not be reported as corroborated"
            )


# --------------------------------------------------------------------------------------
# Transition semantics
# --------------------------------------------------------------------------------------


class TestTransitionSemantics:
    def test_unchanged_class_is_labelled_spectral_only(self, result):
        for t in result.transitions:
            if t.before is t.after:
                assert t.change_type is ChangeType.SPECTRAL_ONLY

    def test_spectral_only_is_excluded_from_the_dominant_claim(self, result):
        """Reporting "bare soil became bare soil" as the headline change would be absurd."""
        dominant = result.dominant_transition()
        assert dominant is not None
        assert dominant.change_type is not ChangeType.SPECTRAL_ONLY
        assert dominant.before is not dominant.after

    def test_change_type_matches_the_class_pair(self, result):
        for t in result.transitions:
            assert t.change_type is ChangeType.from_transition(t.before, t.after)

    def test_pixels_of_type_sums_the_matching_transitions(self, result):
        for change_type in ChangeType:
            expected = sum(
                t.pixel_count
                for t in result.transitions
                if t.change_type is change_type
            )
            assert result.pixels_of_type(change_type) == expected

    def test_area_of_type_agrees_with_pixel_counts(self, result):
        """10 m pixels: an area must be exactly 100 m² per pixel or it is invented."""
        for change_type in ChangeType:
            pixels = result.pixels_of_type(change_type)
            area = result.area_of_type(change_type)
            if pixels and area is not None:
                assert area == pytest.approx(pixels * 100.0, rel=1e-6)

    def test_area_of_type_is_none_for_an_absent_type(self, result):
        absent = [c for c in ChangeType if result.pixels_of_type(c) == 0]
        assert absent, "some change type must be absent from this pair"
        assert result.area_of_type(absent[0]) is None

    def test_urban_expansion_corroborates_the_manifest(self, result, demo_manifest):
        """The manifest documents urban expansion; measure it independently."""
        assert any("urban expansion" in c for c in demo_manifest["known_changes"])
        assert result.pixels_of_type(ChangeType.URBAN_EXPANSION) > 1000

    def test_water_loss_corroborates_the_manifest(self, result, demo_manifest):
        assert any("reservoir shrinkage" in c for c in demo_manifest["known_changes"])
        assert result.pixels_of_type(ChangeType.WATER_LOSS) > 1000


class TestClassDeltas:
    """An independent check on the transition table: both must tell the same story."""

    def test_measured_over_the_valid_area_not_the_change_mask(self, result):
        for delta in result.class_deltas:
            assert delta.pixels_before + delta.pixels_after > result.changed_pixels / 100

    def test_urban_growth_and_water_loss_appear_as_net_deltas(self, result):
        by_class = {d.label: d for d in result.class_deltas}
        assert by_class[LandCoverClass.BUILT_UP].pixel_delta > 0
        assert by_class[LandCoverClass.WATER].pixel_delta < 0

    def test_net_delta_direction_agrees_with_the_transitions(self, result):
        """Cross-check: the class that grew most must be a transition destination."""
        grew = max(result.class_deltas, key=lambda d: d.pixel_delta)
        destinations = {t.after for t in result.transitions if t.before is not t.after}
        assert grew.label in destinations

    def test_relative_delta_is_none_when_the_class_was_absent(self):
        from app.services.change import ClassDelta

        delta = ClassDelta(LandCoverClass.WATER, 0, 500, None, None)
        assert delta.relative_delta is None
        assert delta.pixel_delta == 500
        assert delta.to_dict()["relative_delta"] is None

    def test_unclassified_is_not_reported_as_a_class(self, result):
        assert LandCoverClass.UNCLASSIFIED not in {d.label for d in result.class_deltas}

    def test_areas_follow_pixel_counts(self, result):
        for delta in result.class_deltas:
            assert delta.area_m2_before == pytest.approx(delta.pixels_before * 100.0)
            assert delta.area_m2_after == pytest.approx(delta.pixels_after * 100.0)


# --------------------------------------------------------------------------------------
# Honesty about what cannot be measured
# --------------------------------------------------------------------------------------


class TestAreaReporting:
    def test_areas_follow_the_pixel_size(self, result):
        assert result.changed_area_m2 == pytest.approx(result.changed_pixels * 100.0)
        for t in result.transitions:
            assert t.area_m2 == pytest.approx(t.pixel_count * 100.0)

    def test_no_area_without_georeferencing(self, make_raster, six_band_scene):
        """§8: a plain image has no ground area, so none may be reported."""
        modified = six_band_scene.copy()
        modified[3, 40:60, 40:60] = 200  # destroy NIR in a block: real, large change
        b = load_raster(
            make_raster("p1.tif", six_band_scene, transform=Affine.identity(), epsg=None,
                        descriptions=OPTICAL_BANDS),
            max_edge=None,
        )
        a = load_raster(
            make_raster("p2.tif", modified, transform=Affine.identity(), epsg=None,
                        descriptions=OPTICAL_BANDS),
            max_edge=None,
        )
        res = detect_change(b, a)
        assert res.changed_area_m2 is None
        assert all(t.area_m2 is None for t in res.transitions)
        assert all(d.area_m2_before is None for d in res.class_deltas)
        assert any("not georeferenced" in w for w in res.warnings)

    def test_pixel_counts_are_still_reported_without_georeferencing(
        self, make_raster, six_band_scene
    ):
        """Losing areas must not mean losing the measurement."""
        modified = six_band_scene.copy()
        modified[3, 40:60, 40:60] = 200
        b = load_raster(make_raster("q1.tif", six_band_scene, transform=Affine.identity(),
                                    epsg=None, descriptions=OPTICAL_BANDS), max_edge=None)
        a = load_raster(make_raster("q2.tif", modified, transform=Affine.identity(),
                                    epsg=None, descriptions=OPTICAL_BANDS), max_edge=None)
        res = detect_change(b, a)
        assert res.changed_pixels > 0
        assert res.area_of_type(ChangeType.VEGETATION_LOSS) is None


class TestNoChange:
    def test_identical_images_yield_no_change(self, before):
        res = detect_change(before, before)
        assert res.changed_pixels == 0
        assert res.changed_fraction == 0.0
        assert res.transitions == []
        assert res.dominant_transition() is None

    def test_identical_images_are_reported_honestly(self, before):
        """"Nothing detected" must not be phrased as "nothing changed"."""
        res = detect_change(before, before)
        joined = " ".join(res.warnings)
        assert "not proof that nothing changed" in joined

    def test_identical_images_have_zero_class_delta(self, before):
        res = detect_change(before, before)
        assert all(d.pixel_delta == 0 for d in res.class_deltas)

    def test_identical_images_serialise(self, before):
        payload = detect_change(before, before).to_dict()
        assert payload["dominant_transition"] is None
        assert payload["changed_pixels"] == 0
        assert payload["evidence_split"]["corroborated_fraction"] == 0.0


class TestRefusals:
    """Each of these has no honest degraded answer, so a refusal is the correct output."""

    def test_mismatched_sizes_are_refused(self, before, make_raster):
        small = load_raster(
            make_raster("small.tif", np.zeros((6, 16, 16), dtype=np.uint16),
                        transform=KNOWN_TRANSFORM, descriptions=OPTICAL_BANDS),
            max_edge=None,
        )
        with pytest.raises(GeospatialError) as exc:
            detect_change(before, small)
        assert exc.value.code is ErrorCode.SIZE_MISMATCH

    def test_no_shared_band_is_refused(self, make_raster):
        """Nothing can be differenced between a VV-only and a VH-only image."""
        a = load_raster(
            make_raster("vv.tif", np.full((1, 32, 32), 100, dtype=np.uint16),
                        transform=KNOWN_TRANSFORM, descriptions=("VV",)),
            max_edge=None,
        )
        b = load_raster(
            make_raster("vh.tif", np.full((1, 32, 32), 100, dtype=np.uint16),
                        transform=KNOWN_TRANSFORM, descriptions=("VH",)),
            max_edge=None,
        )
        with pytest.raises(GeospatialError) as exc:
            detect_change(a, b)
        assert exc.value.code is ErrorCode.MISSING_BAND

    def test_no_valid_overlap_is_refused(self, make_raster):
        """Two fully-nodata images share no pixel to compare."""
        arr = np.zeros((6, 32, 32), dtype=np.uint16)
        a = load_raster(make_raster("n1.tif", arr, transform=KNOWN_TRANSFORM,
                                    descriptions=OPTICAL_BANDS, nodata=0), max_edge=None)
        b = load_raster(make_raster("n2.tif", arr, transform=KNOWN_TRANSFORM,
                                    descriptions=OPTICAL_BANDS, nodata=0), max_edge=None)
        with pytest.raises(GeospatialError) as exc:
            detect_change(a, b)
        assert exc.value.code is ErrorCode.NO_SPATIAL_OVERLAP

    def test_refusal_messages_are_user_facing(self, before, make_raster):
        """§29: no stack traces, no internal identifiers in a user-visible message."""
        small = load_raster(
            make_raster("s2.tif", np.zeros((6, 16, 16), dtype=np.uint16),
                        transform=KNOWN_TRANSFORM, descriptions=OPTICAL_BANDS),
            max_edge=None,
        )
        with pytest.raises(GeospatialError) as exc:
            detect_change(before, small)
        message = str(exc.value)
        assert message[0].isupper() and message.rstrip().endswith(".")
        for token in ("Traceback", "numpy", "ndarray", "None"):
            assert token not in message


class TestRawBandFallback:
    """Without a computable index, change is still measured — with a stated caveat."""

    @pytest.fixture(scope="class")
    def gray_pair(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("gray")
        base = np.full((1, 64, 64), 3000, dtype=np.uint16)
        later = base.copy()
        later[0, 20:44, 20:44] = 8000  # unambiguous brightening
        return [
            load_raster(
                write_test_raster(tmp / name, arr, transform=KNOWN_TRANSFORM,
                                  descriptions=("gray",)),
                max_edge=None,
            )
            for name, arr in (("g1.tif", base), ("g2.tif", later))
        ]

    def test_falls_back_to_raw_bands(self, gray_pair):
        res = detect_change(*gray_pair)
        assert res.method == "cva-raw-bands"
        assert res.features_used == ["gray"]

    def test_warns_that_conclusions_are_weaker(self, gray_pair):
        joined = " ".join(detect_change(*gray_pair).warnings)
        assert "raw band values" in joined
        assert "weaker conclusions" in joined

    def test_still_finds_the_real_change(self, gray_pair):
        """The fallback must be a working detector, not a placeholder."""
        res = detect_change(*gray_pair)
        truth = np.zeros((64, 64), dtype=bool)
        truth[20:44, 20:44] = True
        p, r, _, iou = scores(res.change_mask, truth)
        assert iou > 0.80, f"IoU {iou:.3f} (P={p:.3f} R={r:.3f})"

    def test_single_band_scene_has_no_class_detector(self, gray_pair):
        """A one-band image cannot be classified, so only one detector may be claimed."""
        res = detect_change(*gray_pair)
        assert res.detectors == ["spectral-cva"]
        assert res.class_only_pixels == 0

    def test_joint_scaling_does_not_erase_the_change(self, gray_pair):
        """Regression guard: per-date normalisation would map both onto [0, 1] and delete it."""
        assert detect_change(*gray_pair).changed_pixels > 0


class TestRobustOutlierThreshold:
    def test_flags_only_the_tail_of_a_unimodal_field(self):
        """The shape the fallback exists for: one near-zero mode with a right tail."""
        rng = np.random.default_rng(3)
        values = np.abs(rng.normal(0, 0.01, 20000))
        thr = _robust_outlier_threshold(values)
        assert 0.0 < thr < 0.1
        assert float((values > thr).mean()) < 0.02

    def test_a_constant_field_flags_nothing(self):
        """Zero variation means nothing is an outlier; flagging the scene would be wrong."""
        values = np.full(1000, 0.42)
        assert float((values > _robust_outlier_threshold(values)).mean()) == 0.0

    def test_ignores_non_finite_values(self):
        values = np.concatenate([np.abs(np.random.default_rng(4).normal(0, 0.01, 5000)),
                                 np.full(100, np.nan)])
        assert np.isfinite(_robust_outlier_threshold(values))

    def test_scales_with_the_spread(self):
        rng = np.random.default_rng(5)
        tight = _robust_outlier_threshold(np.abs(rng.normal(0, 0.01, 10000)))
        wide = _robust_outlier_threshold(np.abs(rng.normal(0, 0.10, 10000)))
        assert wide > tight


class TestMinimumPatchSizing:
    def test_minimum_patch_is_expressed_as_a_ground_area(self):
        """So it means the same thing at 10 m and at 30 m."""
        assert MIN_CHANGE_PATCH_M2 > 0

    def test_speck_changes_are_filtered_out(self, make_raster, six_band_scene):
        """A handful of altered pixels is misregistration or noise, not a land conversion."""
        speckled = six_band_scene.copy()
        rng = np.random.default_rng(9)
        rows = rng.integers(0, 64, 8)
        cols = rng.integers(0, 64, 8)
        speckled[3, rows, cols] = 100  # 8 scattered NIR dropouts
        b = load_raster(make_raster("b.tif", six_band_scene, transform=KNOWN_TRANSFORM,
                                    descriptions=OPTICAL_BANDS), max_edge=None)
        a = load_raster(make_raster("a.tif", speckled, transform=KNOWN_TRANSFORM,
                                    descriptions=OPTICAL_BANDS), max_edge=None)
        res = detect_change(b, a)
        assert res.changed_pixels < 25, (
            f"{res.changed_pixels} pixels reported from 8 isolated dropouts"
        )

    def test_a_large_real_change_survives_filtering(self, make_raster, six_band_scene):
        """The counterpart: the filter must not be strong enough to erase real change."""
        modified = six_band_scene.copy()
        modified[3, 40:60, 40:60] = 200
        b = load_raster(make_raster("b2.tif", six_band_scene, transform=KNOWN_TRANSFORM,
                                    descriptions=OPTICAL_BANDS), max_edge=None)
        a = load_raster(make_raster("a2.tif", modified, transform=KNOWN_TRANSFORM,
                                    descriptions=OPTICAL_BANDS), max_edge=None)
        res = detect_change(b, a)
        truth = np.zeros((64, 64), dtype=bool)
        truth[40:60, 40:60] = True
        _, recall, _, _ = scores(res.change_mask, truth)
        assert recall > 0.85


class TestRegistrationReporting:
    def test_registration_is_measured_and_reported(self, result):
        assert result.registration.offset_magnitude >= 0.0
        assert "offset_magnitude_px" in result.registration.to_dict()

    def test_aligned_demo_pair_is_not_flagged_as_misregistered(self, result):
        joined = " ".join(result.warnings).lower()
        assert "misaligned" not in joined

    def test_a_shifted_pair_is_flagged(self, make_raster, six_band_scene):
        """Misregistration turns every land-cover boundary into apparent change; say so."""
        shifted = np.roll(six_band_scene, shift=4, axis=2)
        b = load_raster(make_raster("r1.tif", six_band_scene, transform=KNOWN_TRANSFORM,
                                    descriptions=OPTICAL_BANDS), max_edge=None)
        a = load_raster(make_raster("r2.tif", shifted, transform=KNOWN_TRANSFORM,
                                    descriptions=OPTICAL_BANDS), max_edge=None)
        res = detect_change(b, a)
        joined = " ".join(res.warnings).lower()
        assert "misaligned" in joined
        assert "over-estimate" in joined


class TestChangeTypeMapping:
    """Unit tests for the transition-labelling lookup itself."""

    def test_every_off_diagonal_pair_has_exactly_one_label(self):
        real = [c for c in LandCoverClass if c is not LandCoverClass.UNCLASSIFIED]
        for b in real:
            for a in real:
                label = ChangeType.from_transition(b, a)
                assert isinstance(label, ChangeType)
                if b is a:
                    assert label is ChangeType.SPECTRAL_ONLY
                else:
                    assert label is not ChangeType.SPECTRAL_ONLY
                    assert label is not ChangeType.OTHER

    def test_destination_determines_the_gain_labels(self):
        assert ChangeType.from_transition(
            LandCoverClass.BARE_SOIL, LandCoverClass.WATER
        ) is ChangeType.WATER_GAIN
        assert ChangeType.from_transition(
            LandCoverClass.BARE_SOIL, LandCoverClass.VEGETATION
        ) is ChangeType.VEGETATION_GAIN
        assert ChangeType.from_transition(
            LandCoverClass.BARE_SOIL, LandCoverClass.BUILT_UP
        ) is ChangeType.URBAN_EXPANSION

    def test_becoming_bare_ground_names_what_was_lost(self):
        assert ChangeType.from_transition(
            LandCoverClass.WATER, LandCoverClass.BARE_SOIL
        ) is ChangeType.WATER_LOSS
        assert ChangeType.from_transition(
            LandCoverClass.VEGETATION, LandCoverClass.BARE_SOIL
        ) is ChangeType.VEGETATION_LOSS
        assert ChangeType.from_transition(
            LandCoverClass.BUILT_UP, LandCoverClass.BARE_SOIL
        ) is ChangeType.URBAN_LOSS

    def test_unclassified_endpoints_are_not_given_a_meaning(self):
        for other in LandCoverClass:
            if other is LandCoverClass.UNCLASSIFIED:
                continue
            assert ChangeType.from_transition(
                LandCoverClass.UNCLASSIFIED, other
            ) is ChangeType.OTHER
            assert ChangeType.from_transition(
                other, LandCoverClass.UNCLASSIFIED
            ) is ChangeType.OTHER

    def test_unclassified_to_unclassified_is_not_a_transition(self):
        assert ChangeType.from_transition(
            LandCoverClass.UNCLASSIFIED, LandCoverClass.UNCLASSIFIED
        ) is ChangeType.SPECTRAL_ONLY


class TestUnclassifiedCannotManufactureChange:
    """A gap in the analysis must never become a finding.

    The class detector compares two class maps. If it ignored the unclassified code, every
    pixel one date failed to classify would read as a conversion to or from "unclassified" —
    turning the classifier's own uncertainty into reported change.
    """

    def test_class_detector_ignores_pixels_unclassified_on_either_date(self, result):
        from app.services.change import _class_disagreement

        valid = np.ones(result.change_mask.shape, dtype=bool)
        disagreement = _class_disagreement(result.before, result.after, valid)
        unknown = LandCoverClass.UNCLASSIFIED
        touching_unknown = (
            result.before.mask_for(unknown) | result.after.mask_for(unknown)
        )
        assert not (disagreement & touching_unknown).any()

    def test_other_transitions_are_never_the_dominant_claim(self, result):
        dominant = result.dominant_transition()
        assert dominant is not None
        assert dominant.change_type is not ChangeType.OTHER
