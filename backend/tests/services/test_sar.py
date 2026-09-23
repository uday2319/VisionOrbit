"""Tests for SAR backscatter analysis.

The measurements quoted in ``app/services/sar.py`` docstrings are asserted here, so a
regression that changes the physics also fails the suite rather than quietly making the
documentation wrong. Accuracy is scored against the demo scene's known class map, and the
refusal paths get as much attention as the success paths — declining to report a regime that
is not in the image is a feature of this module, not a gap in it.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.errors import ErrorCode, ValidationError
from app.core.types import Modality, ScatteringRegime
from app.geospatial.raster import load_raster
from app.services import indices, sar

# Nominal backscatter of the demo scene's four classes, in dB, and its class codes.
# Duplicated from the generator deliberately: if the generator changes, these tests should
# fail rather than silently track it.
GT_WATER, GT_VEGETATION, GT_BUILT_UP, GT_BARE_SOIL = 1, 2, 3, 4
DEMO_WATER_DB = -20.0
DEMO_VEG_DB = -10.0
DEMO_BUILT_DB = -3.5
DEMO_SOIL_DB = -14.0
DEMO_LOOKS = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def f1_score(pred: np.ndarray, truth: np.ndarray) -> float:
    """F1 of a boolean mask against a boolean truth."""
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _erode(mask: np.ndarray, size: int) -> np.ndarray:
    """Interior of a mask, ``size``x``size`` structuring element."""
    return cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)


def _boundary_band(truth: np.ndarray, width: int) -> np.ndarray:
    """Pixels within ``width`` of the truth mask's edge."""
    kernel = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    solid = truth.astype(np.uint8)
    return cv2.dilate(solid, kernel).astype(bool) & ~cv2.erode(solid, kernel).astype(bool)


def speckled_scene(
    labels: np.ndarray, dark_db: float, light_db: float, seed: int, looks: int = DEMO_LOOKS
) -> np.ndarray:
    """A two-level dB scene with real speckle.

    Speckle is applied multiplicatively in linear power with a Gamma distribution, exactly as
    a real multi-look intensity image is distributed. Adding Gaussian noise in dB instead
    would let the Lee filter look better than it is.
    """
    rng = np.random.default_rng(seed)
    linear = np.where(labels, 10 ** (dark_db / 10), 10 ** (light_db / 10))
    linear = linear * rng.gamma(looks, 1.0 / looks, size=labels.shape)
    return 10 * np.log10(linear)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def demo_sar(demo_root):
    """The generated demo SAR scene at full resolution, loaded once."""
    return load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None)


@pytest.fixture(scope="module")
def demo_labels(demo_root):
    """Ground-truth class map for the demo scene."""
    return load_raster(
        demo_root / "optical" / "scene_landcover_gt.tif", max_edge=None
    ).data[0].astype(int)


@pytest.fixture(scope="module")
def analysed(demo_sar):
    """A single full analysis, shared across assertions (it is the expensive step)."""
    return sar.analyze_sar(demo_sar)


@pytest.fixture
def two_regime_scene():
    """A dark patch covering a quarter of a uniform diffuse field, with real speckle."""
    n = 192
    labels = np.zeros((n, n), dtype=bool)
    labels[20:116, 20:116] = True  # 25% of the frame
    return speckled_scene(labels, DEMO_WATER_DB, DEMO_VEG_DB, seed=4), labels


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
class TestUnitHandling:
    """dB and linear power must not be confused: the filter is only valid in one of them."""

    def test_db_and_linear_round_trip(self):
        db = np.array([-25.0, -20.0, -14.0, -10.0, -3.5, 0.0])
        assert sar.to_db(sar.to_linear_power(db)) == pytest.approx(db, abs=1e-9)

    def test_linear_power_of_known_values(self):
        """-10 dB is exactly 0.1 in power; -20 dB exactly 0.01."""
        assert sar.to_linear_power(np.array([-10.0, -20.0, 0.0])) == pytest.approx(
            [0.1, 0.01, 1.0]
        )

    def test_zero_power_becomes_a_floor_not_negative_infinity(self):
        """An exact zero must not poison every downstream histogram with -inf."""
        out = sar.to_db(np.array([0.0, 1e-30]))
        assert np.all(np.isfinite(out))
        assert np.all(out <= -99.0)

    def test_db_raster_is_recognised(self, demo_sar):
        assert sar.looks_like_db(demo_sar.data[0])

    def test_linear_power_raster_is_not_mistaken_for_db(self):
        """Strictly positive power values must not be read as decibels."""
        rng = np.random.default_rng(0)
        assert not sar.looks_like_db(rng.random(4096) * 0.2)

    def test_digital_numbers_are_not_mistaken_for_db(self):
        """Raw uint DN spanning thousands is not backscatter in dB."""
        rng = np.random.default_rng(1)
        assert not sar.looks_like_db(rng.integers(0, 8000, 4096).astype(float))

    def test_empty_input_is_not_claimed_to_be_db(self):
        assert not sar.looks_like_db(np.array([np.nan, np.nan]))

    def test_non_db_input_is_converted_and_the_caveat_is_surfaced(
        self, make_raster, known_transform
    ):
        """A linear-power raster must be converted *and* the user told the units were guessed."""
        rng = np.random.default_rng(2)
        linear = (rng.gamma(6, 1 / 6.0, size=(1, 96, 96)) * 0.05).astype(np.float32)
        raster = load_raster(
            make_raster("lin.tif", linear, transform=known_transform, descriptions=("vv",))
        )
        result = sar.analyze_sar(raster)
        assert any("not in decibels" in w for w in result.warnings)
        # Whatever thresholds survive must be quoted in dB, i.e. negative for this input.
        for threshold in (result.smooth_threshold_db, result.bright_threshold_db):
            assert threshold is None or threshold < 0


# ---------------------------------------------------------------------------
# ENL estimation
# ---------------------------------------------------------------------------
class TestEnlEstimation:
    """The look count must come from the pixels, not from a header or a constant."""

    @pytest.mark.parametrize("looks", [2, 6, 16])
    def test_estimate_recovers_a_known_look_count(self, looks):
        """On pure single-class speckle with no structure, the estimate must be accurate.

        This is the only situation where a ground-truth ENL exists unambiguously, so it is
        where the estimator's correctness is actually checkable.
        """
        rng = np.random.default_rng(5)
        linear = 10 ** (DEMO_VEG_DB / 10) * rng.gamma(looks, 1.0 / looks, size=(256, 256))
        estimate, warnings = sar.estimate_enl(linear, 7)
        assert estimate == pytest.approx(looks, rel=0.25), (
            f"ENL estimate {estimate:.2f} is not within 25% of the true {looks} looks"
        )
        assert not warnings

    def test_global_cv_would_be_physically_impossible_on_a_multi_class_scene(self, demo_sar):
        """Demonstrates why the estimator uses *local* statistics, rather than asserting it.

        A whole-scene coefficient of variation is inflated by the contrast between classes,
        not just speckle within them. On the demo scene it implies fewer than one look, which
        is impossible for an intensity image — so any estimator built on it is wrong by
        construction, however reasonable the arithmetic looks.
        """
        linear = sar.to_linear_power(demo_sar.data[0].astype(np.float64))
        finite = linear[np.isfinite(linear)]
        global_cv = float(finite.std() / finite.mean())
        global_enl = 1.0 / (global_cv * global_cv)
        assert global_enl < 1.0, (
            "the global-CV pathology must remain demonstrable, or this test no longer "
            "documents why local statistics are used"
        )
        assert global_enl == pytest.approx(0.73, abs=0.05)

        local, _ = sar.estimate_enl(linear, 7)
        assert local > 4.0 * global_enl
        assert local == pytest.approx(15.1, abs=1.0)

    def test_estimate_is_stable_across_window_sizes(self, demo_sar):
        """A quantity that swung with the window would not be a property of the image."""
        linear = sar.to_linear_power(demo_sar.data[0].astype(np.float64))
        estimates = [sar.estimate_enl(linear, w)[0] for w in (5, 7, 9, 11)]
        assert min(estimates) >= 13.5
        assert max(estimates) <= 17.0
        assert max(estimates) - min(estimates) < 2.5

    def test_estimate_exceeds_the_nominal_look_count(self, demo_sar):
        """The impulse response correlates neighbours, so effective looks exceed nominal.

        Pinned because using the nominal value from a product header instead would
        over-smooth, and the whole reason this function exists is that the header cannot be
        trusted.
        """
        linear = sar.to_linear_power(demo_sar.data[0].astype(np.float64))
        estimate, _ = sar.estimate_enl(linear, 7)
        assert estimate > DEMO_LOOKS

    def test_too_little_data_falls_back_to_single_look_with_a_warning(self):
        linear = np.full((8, 8), 0.1)
        estimate, warnings = sar.estimate_enl(linear, 7)
        assert estimate == 1.0
        assert any("Too little valid data" in w for w in warnings)

    def test_an_already_filtered_image_is_reported_rather_than_over_trusted(self):
        """Near-zero local variation is not 'infinite looks', it is a processed product."""
        linear = np.full((128, 128), 0.05)
        estimate, warnings = sar.estimate_enl(linear, 7)
        assert estimate == 64.0
        assert any("almost no local variation" in w for w in warnings)

    def test_nodata_does_not_drag_local_statistics(self):
        """Partly-invalid windows must be excluded, not treated as dark pixels.

        Filling NaN with zero and taking a plain box mean would depress the local mean near
        every nodata border, inflating the local CV there and biasing the whole estimate down.
        """
        rng = np.random.default_rng(6)
        linear = 0.1 * rng.gamma(6, 1 / 6.0, size=(200, 200))
        clean, _ = sar.estimate_enl(linear, 7)
        holed = linear.copy()
        holed[:, :60] = np.nan
        with_holes, _ = sar.estimate_enl(holed, 7)
        assert with_holes == pytest.approx(clean, rel=0.15)


# ---------------------------------------------------------------------------
# Speckle filtering
# ---------------------------------------------------------------------------
class TestLeeFilter:
    """The filter must reduce speckle without moving class means or smearing boundaries."""

    def test_homogeneous_region_converges_to_the_local_mean(self):
        """With no signal variance, the MMSE weight goes to zero and the mean is returned."""
        rng = np.random.default_rng(7)
        truth = 0.1
        linear = truth * rng.gamma(8, 1 / 8.0, size=(128, 128))
        filtered = sar.lee_filter(linear, 8.0, 7)
        assert float(np.nanmean(filtered)) == pytest.approx(truth, rel=0.02)
        assert float(np.nanstd(filtered)) < 0.4 * float(np.nanstd(linear))

    def test_filtering_does_not_shift_class_means(self, two_regime_scene):
        """A filter that biased the levels would move every threshold with it."""
        db, labels = two_regime_scene
        filtered, _ = sar.despeckle(db, 7)
        assert float(filtered[_erode(labels, 11)].mean()) == pytest.approx(
            DEMO_WATER_DB, abs=0.35
        )
        assert float(filtered[_erode(~labels, 11)].mean()) == pytest.approx(
            DEMO_VEG_DB, abs=0.35
        )

    def test_step_edge_is_narrower_than_a_box_mean_can_make_it(self, two_regime_scene):
        """The adaptive weight must leave a real boundary alone.

        A box mean of window 7 necessarily spreads a step over about 7 pixels; the Lee weight
        approaches 1 at an edge, so the transition stays narrower. Measured as the count of
        pixels sitting strictly between the two class levels along a row that crosses the
        patch edge, which is what "smearing" means in practice.
        """
        db, labels = two_regime_scene
        linear = sar.to_linear_power(db)
        enl, _ = sar.estimate_enl(linear, 7)
        low, high = DEMO_WATER_DB + 2.0, DEMO_VEG_DB - 2.0

        def intermediate_pixels(arr: np.ndarray) -> int:
            row = sar.to_db(arr)[68, 105:130]
            return int(((row > low) & (row < high)).sum())

        lee = intermediate_pixels(sar.lee_filter(linear, enl, 7))
        box = intermediate_pixels(sar._box_mean(linear, 7))
        assert lee < box, (
            f"the Lee filter spread the edge over {lee} pixels and a plain box mean over "
            f"{box}; the adaptive weight is not preserving boundaries"
        )
        # And the step must still be in the right place, not shifted away.
        filtered_db = sar.to_db(sar.lee_filter(linear, enl, 7))
        assert filtered_db[68, 100] < (DEMO_WATER_DB + DEMO_VEG_DB) / 2 < filtered_db[68, 125]

    def test_lee_preserves_more_edge_contrast_than_a_box_mean(self, demo_sar, demo_labels):
        """The measured justification for choosing Lee over the simpler filter.

        Scored as the ratio of mean gradient magnitude on true class boundaries to that in
        class interiors — high means structure survived while speckle did not. The box mean is
        given every window from 3 to 13 and its best result is still well short.
        """
        linear = sar.to_linear_power(demo_sar.data[0].astype(np.float64))
        enl, _ = sar.estimate_enl(linear, 7)

        labels = demo_labels
        edge = np.zeros(labels.shape, dtype=bool)
        for axis in (0, 1):
            edge |= (labels != np.roll(labels, 1, axis)) | (labels != np.roll(labels, -1, axis))
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

        def gradient_ratio(arr: np.ndarray) -> float:
            db = sar.to_db(arr).astype(np.float32)
            mag = cv2.magnitude(
                cv2.Sobel(db, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(db, cv2.CV_32F, 0, 1, ksize=3),
            )
            return float(mag[edge].mean() / mag[~edge].mean())

        lee = gradient_ratio(sar.lee_filter(linear, enl, 7))
        best_box = max(gradient_ratio(sar._box_mean(linear, w)) for w in (3, 5, 7, 9, 11, 13))
        assert lee == pytest.approx(9.0, abs=0.6)
        assert best_box == pytest.approx(6.4, abs=0.6)
        assert lee > 1.3 * best_box

    def test_lee_keeps_class_means_truer_than_a_wide_box_mean(self, demo_sar, demo_labels):
        """A box mean wide enough to smooth comparably starts moving the class levels.

        Radiometric bias matters more than it looks: every threshold in this module is a dB
        value, so a filter that shifts a class mean shifts the decision boundary with it.
        """
        linear = sar.to_linear_power(demo_sar.data[0].astype(np.float64))
        enl, _ = sar.estimate_enl(linear, 7)
        nominal = {
            GT_WATER: DEMO_WATER_DB,
            GT_VEGETATION: DEMO_VEG_DB,
            GT_BUILT_UP: DEMO_BUILT_DB,
            GT_BARE_SOIL: DEMO_SOIL_DB,
        }

        def worst_bias(arr: np.ndarray) -> float:
            db = sar.to_db(arr)
            out = []
            for code, level in nominal.items():
                interior = _erode(demo_labels == code, 9)
                if interior.sum() > 50:
                    out.append(abs(float(db[interior].mean()) - level))
            return max(out)

        lee = worst_bias(sar.lee_filter(linear, enl, 7))
        assert lee < 0.05
        # Bias grows monotonically once the box window starts spanning class boundaries.
        assert worst_bias(sar._box_mean(linear, 11)) > 2 * lee
        assert worst_bias(sar._box_mean(linear, 13)) > 4 * lee

    def test_speckle_reduction_is_reported_and_real(self, demo_sar):
        filtered, report = sar.despeckle(demo_sar.data[0].astype(np.float64), 7)
        assert report.filter_name == "lee-mmse"
        assert report.output_cv < report.input_cv
        assert report.speckle_reduction > 4.0
        assert report.estimated_enl > DEMO_LOOKS
        assert filtered.shape == demo_sar.data[0].shape

    def test_nan_is_preserved_not_filled(self):
        """Invalid pixels must stay invalid; a filter must not manufacture data."""
        rng = np.random.default_rng(8)
        linear = 0.1 * rng.gamma(6, 1 / 6.0, size=(64, 64))
        linear[10:20, 10:20] = np.nan
        out = sar.lee_filter(linear, 6.0, 7)
        assert np.all(np.isnan(out[10:20, 10:20]))
        assert np.isfinite(out[40:50, 40:50]).all()

    def test_all_invalid_input_returns_all_invalid_output(self):
        out = sar.lee_filter(np.full((16, 16), np.nan), 6.0, 7)
        assert np.all(np.isnan(out))

    def test_output_is_never_negative_power(self):
        """Linear extrapolation at a hard edge must not produce a non-physical value."""
        linear = np.zeros((32, 32)) + 1e-4
        linear[16:, :] = 10.0
        out = sar.lee_filter(linear, 2.0, 7)
        assert np.all(out[np.isfinite(out)] > 0)

    @pytest.mark.parametrize("window", [2, 4, 1, 0, -3])
    def test_invalid_window_is_rejected(self, window):
        """An even window has no centre pixel, so the result would be silently shifted."""
        with pytest.raises(ValueError, match="odd"):
            sar.lee_filter(np.full((32, 32), 0.1), 6.0, window)

    def test_window_constant_is_overridable_at_call_time(self, demo_sar, monkeypatch):
        """Guards against the window being frozen as a default argument.

        It was, once: ``despeckle(db, window=SPECKLE_WINDOW)`` binds the value at import time,
        so changing the constant had no effect and any window sweep silently measured the same
        thing four times over.
        """
        monkeypatch.setattr(sar, "SPECKLE_WINDOW", 3)
        _, report = sar.despeckle(demo_sar.data[0].astype(np.float64))
        assert report.window == 3
        assert sar.analyze_sar(demo_sar).speckle.window == 3


# ---------------------------------------------------------------------------
# Regime segmentation accuracy
# ---------------------------------------------------------------------------
class TestRegimeAccuracy:
    """Scored against the demo scene's known class map."""

    def test_smooth_regime_recovers_water(self, analysed, demo_labels):
        score = f1_score(analysed.mask_for(ScatteringRegime.SMOOTH), demo_labels == GT_WATER)
        assert score > 0.98, f"water F1 fell to {score:.4f}"

    def test_water_errors_sit_on_the_shoreline_not_in_the_interior(self, analysed, demo_labels):
        """Locates the residual water error, which is what makes the F1 interpretable.

        Of 153 disagreements against a 12,460-pixel water body, every one lies within 3 pixels
        of the shoreline and 92% within 1 pixel — the reservoir interior is exact. A threshold
        set slightly wrong would instead scatter errors through the interior, at the same F1.
        """
        predicted = analysed.mask_for(ScatteringRegime.SMOOTH)
        truth = demo_labels == GT_WATER
        errors = predicted != truth
        assert errors.sum() > 0, "an exact match would make this test vacuous"
        assert not (errors & ~_boundary_band(truth, 3)).any(), (
            "some water errors lie away from the shoreline, so this is not a mixed-pixel effect"
        )
        near = int((errors & _boundary_band(truth, 1)).sum())
        assert near / int(errors.sum()) > 0.85

    def test_diffuse_regime_recovers_vegetation_and_soil_together(self, analysed, demo_labels):
        """Both are diffuse scatterers, and the module claims only that."""
        rough = (demo_labels == GT_VEGETATION) | (demo_labels == GT_BARE_SOIL)
        score = f1_score(analysed.mask_for(ScatteringRegime.DIFFUSE), rough)
        assert score > 0.98, f"rough-surface F1 fell to {score:.4f}"

    def test_double_bounce_regime_recovers_built_up(self, analysed, demo_labels):
        """Corner-reflector returns are the easiest regime to isolate, and must be.

        Built-up backscatter sits about 10 dB clear of every other class in the scene, so a
        correct threshold should leave almost nothing on the wrong side. The absolute error
        counts are asserted as well as the F1, because a ratio near 1.0 hides whether the
        remaining handful of errors is 17 pixels or 1700.
        """
        predicted = analysed.mask_for(ScatteringRegime.DOUBLE_BOUNCE)
        truth = demo_labels == GT_BUILT_UP
        assert f1_score(predicted, truth) > 0.99

        false_positives = int((predicted & ~truth).sum())
        false_negatives = int((~predicted & truth).sum())
        assert false_positives + false_negatives < 0.01 * int(truth.sum())

    def test_the_road_grid_inside_the_urban_block_is_resolved(self, analysed, demo_labels):
        """The hardest structure in the scene, and a guard against a too-generous mask.

        The demo scene's urban blocks are cut by a grid of 5-pixel-wide (50 m) bare-soil roads.
        A mask produced by over-smoothing would swallow them and still score well against the
        block as a whole, so the roads are scored on their own: they must *not* be reported as
        double-bounce.

        A pre-compaction measurement put built-up F1 at 0.868 by scoring against a truth mask
        that counted these road pixels *as* buildings. This test pins the distinction that
        measurement blurred.
        """
        built = demo_labels == GT_BUILT_UP
        # The road grid is what a morphological closing of the built-up mask would swallow:
        # gaps narrow enough to be bridged, as opposed to the open soil outside the blocks.
        closed = cv2.morphologyEx(
            built.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        ).astype(bool)
        roads = closed & ~built
        assert roads.sum() > 2000, "the road grid must exist for this test to mean much"
        assert (roads & (demo_labels == GT_BARE_SOIL)).sum() == roads.sum(), (
            "the closed gaps must be exactly bare-soil roads, not some other class"
        )

        swallowed = analysed.mask_for(ScatteringRegime.DOUBLE_BOUNCE)[roads]
        assert swallowed.mean() < 0.02, (
            f"{100 * swallowed.mean():.1f}% of the road pixels inside the urban blocks were "
            f"reported as buildings; the filter is over-smoothing"
        )

    def test_thresholds_are_near_optimal_without_seeing_the_truth(self, analysed, demo_labels):
        """The strongest available check that the thresholds are not fitted.

        Both cuts are derived from the histogram alone. Sweeping the threshold against the
        known class map finds the best achievable value; the data-driven choice must land
        essentially on it. If it did not, the method would be relying on this scene's
        particulars rather than on a property of backscatter distributions.
        """
        db = analysed.backscatter_db
        water, built = demo_labels == GT_WATER, demo_labels == GT_BUILT_UP
        best_smooth = max(f1_score(db < t, water) for t in np.arange(-25.0, -8.0, 0.1))
        best_bright = max(f1_score(db > t, built) for t in np.arange(-12.0, 0.0, 0.1))

        chosen_smooth = f1_score(db < analysed.smooth_threshold_db, water)
        chosen_bright = f1_score(db > analysed.bright_threshold_db, built)

        assert best_smooth - chosen_smooth < 0.01
        assert best_bright - chosen_bright < 0.01

    def test_speckle_filtering_improves_water_detection(self, demo_sar, demo_labels):
        """The measured payoff for filtering at all."""
        water = demo_labels == GT_WATER
        raw_f1 = f1_score(
            sar.analyze_sar(demo_sar, filter_speckle=False).mask_for(ScatteringRegime.SMOOTH),
            water,
        )
        filtered_f1 = f1_score(
            sar.analyze_sar(demo_sar).mask_for(ScatteringRegime.SMOOTH), water
        )
        assert filtered_f1 > raw_f1
        assert filtered_f1 - raw_f1 > 0.02

    def test_accuracy_is_insensitive_to_the_window_choice(self, demo_sar, demo_labels):
        """Guards the honesty of the SPECKLE_WINDOW comment.

        The constant is documented as *not* selected by task accuracy. If accuracy did in fact
        vary materially with the window, that comment would be misleading and this says so.
        """
        scores = [
            f1_score(
                sar.analyze_sar(demo_sar, window=w).mask_for(ScatteringRegime.SMOOTH),
                demo_labels == GT_WATER,
            )
            for w in (3, 5, 7, 9, 11)
        ]
        assert max(scores) - min(scores) < 0.01


# ---------------------------------------------------------------------------
# Refusal behaviour
# ---------------------------------------------------------------------------
class TestEvidenceIsNotCircular:
    """Otsu's own quality figure cannot justify the split Otsu made."""

    def test_otsu_quality_would_have_accepted_pure_noise(self, demo_sar):
        """The circularity that :func:`indices.mode_prominence` exists to avoid.

        The three-class between-class variance ratio scores *higher* on Gaussian noise than on
        the genuine four-class scene, because partitioning any distribution by value explains
        most of its variance. A gate built on it would admit noise, and no choice of threshold
        would fix that, because the ordering itself is inverted.
        """
        real = demo_sar.data[0][np.isfinite(demo_sar.data[0])]
        _, real_quality = indices.multilevel_otsu_threshold(real)
        noise = np.random.default_rng(12).normal(-13.0, 3.0, real.size)
        _, noise_quality = indices.multilevel_otsu_threshold(noise)

        assert noise_quality > real_quality, (
            "the inversion must remain demonstrable, or this test no longer documents why "
            "Otsu's quality figure is not used as evidence"
        )
        assert real_quality == pytest.approx(0.748, abs=0.02)
        assert noise_quality == pytest.approx(0.810, abs=0.02)

    def test_prominence_separates_real_modes_from_noise_by_a_wide_margin(self, demo_sar):
        """The gate must sit in an empty band, so its exact value cannot matter much."""
        real = demo_sar.data[0][np.isfinite(demo_sar.data[0])]
        (lo, hi), _ = indices.multilevel_otsu_threshold(real)
        real_prominence = indices.mode_prominence(real, lo, upper=hi)

        noise = np.random.default_rng(13).normal(-13.0, 3.0, real.size)
        (nlo, nhi), _ = indices.multilevel_otsu_threshold(noise)
        noise_prominence = indices.mode_prominence(noise, nlo, upper=nhi)

        assert real_prominence == pytest.approx(0.72, abs=0.06)
        assert noise_prominence < 0.10
        # The decision boundary is nowhere near either measurement.
        assert real_prominence - indices.MIN_MODE_PROMINENCE > 0.4
        assert indices.MIN_MODE_PROMINENCE - noise_prominence > 0.1

    def test_uniform_noise_is_refused(self):
        """A featureless distribution has no valley anywhere in it."""
        rng = np.random.default_rng(10)
        values = rng.uniform(-25.0, -3.0, 40000)
        (lo, hi), quality = indices.multilevel_otsu_threshold(values)
        assert quality > 0.8, "the circular variance ratio must still look excellent here"
        assert indices.mode_prominence(values, lo, upper=hi) < indices.MIN_MODE_PROMINENCE
        assert indices.mode_prominence(values, hi, lower=lo) < indices.MIN_MODE_PROMINENCE

    def test_single_speckle_population_has_no_prominent_valley(self):
        """One population cut in three would fabricate two regimes out of speckle."""
        rng = np.random.default_rng(9)
        linear = 10 ** (DEMO_VEG_DB / 10) * rng.gamma(6, 1 / 6.0, size=(160, 160))
        values = (10 * np.log10(linear)).ravel()
        (lo, hi), _ = indices.multilevel_otsu_threshold(values)
        assert indices.mode_prominence(values, lo, upper=hi) < indices.MIN_MODE_PROMINENCE

    def test_refusal_boundary_tracks_where_detection_actually_fails(self):
        """The gate's threshold is not arbitrary: it refuses where the method stops working.

        A dark patch is shrunk until it is undetectable. At and below 1% coverage, forcing a
        detection yields a poor mask, so refusing is right. The gate is also shown to be
        conservative rather than merely correct — it is allowed to decline a case a forced
        threshold would have recovered, preferring a missed regime to a phantom one.
        """
        n = 256
        outcomes = {}
        for i, fraction in enumerate((0.005, 0.01, 0.05, 0.25)):
            labels = np.zeros((n, n), dtype=bool)
            side = max(1, int(np.sqrt(fraction * n * n)))
            labels[10 : 10 + side, 10 : 10 + side] = True
            db = speckled_scene(labels, DEMO_WATER_DB, DEMO_VEG_DB, seed=14 + i)
            values = db[np.isfinite(db)]
            (lo, hi), _ = indices.multilevel_otsu_threshold(values)
            prominence = indices.mode_prominence(values, lo, upper=hi)
            outcomes[fraction] = (
                prominence >= indices.MIN_MODE_PROMINENCE,
                f1_score(db < lo, labels),
            )

        # Refused where it genuinely does not work.
        assert outcomes[0.005][0] is False and outcomes[0.005][1] < 0.3
        assert outcomes[0.01][0] is False and outcomes[0.01][1] < 0.4
        # Accepted where it does, and accurate there.
        assert outcomes[0.05][0] is True and outcomes[0.05][1] > 0.9
        assert outcomes[0.25][0] is True and outcomes[0.25][1] > 0.95


class TestRefusesWhenRegimesAreAbsent:
    """A regime that is not in the image must not be reported as if it were."""

    def test_a_scene_with_no_bright_targets_reports_no_double_bounce(
        self, make_raster, known_transform
    ):
        """Water and vegetation only: a built-up regime must not appear from nowhere."""
        n = 160
        labels = np.zeros((n, n), dtype=bool)
        labels[30:120, 30:120] = True
        db = speckled_scene(labels, DEMO_WATER_DB, DEMO_VEG_DB, seed=15)
        raster = load_raster(
            make_raster(
                "nobright.tif", db[None, :, :].astype(np.float32),
                transform=known_transform, descriptions=("vv",),
            )
        )
        result = sar.analyze_sar(raster)

        assert result.bright_threshold_db is None
        assert result.stats_for(ScatteringRegime.DOUBLE_BOUNCE) is None
        assert ScatteringRegime.DOUBLE_BOUNCE not in result.separated_regimes
        assert any("no double-bounce area" in w for w in result.warnings)
        # The regime that *is* present must still be found.
        assert result.smooth_threshold_db is not None
        assert f1_score(result.mask_for(ScatteringRegime.SMOOTH), labels) > 0.9

    def test_a_featureless_scene_refuses_both_boundaries(self, make_raster, known_transform):
        """Neither regime is claimed, and the user is told why."""
        rng = np.random.default_rng(16)
        linear = 10 ** (DEMO_VEG_DB / 10) * rng.gamma(6, 1 / 6.0, size=(160, 160))
        db = (10 * np.log10(linear))[None, :, :].astype(np.float32)
        raster = load_raster(
            make_raster("flat.tif", db, transform=known_transform, descriptions=("vv",))
        )
        result = sar.analyze_sar(raster)

        assert result.smooth_threshold_db is None
        assert result.bright_threshold_db is None
        assert result.threshold_method == "refused"
        assert result.separated_regimes == [ScatteringRegime.DIFFUSE]
        assert any("No distinct dark-surface population" in w for w in result.warnings)
        assert any("No distinct bright-scattering population" in w for w in result.warnings)
        # Everything valid is diffuse; nothing is left unclassified by accident.
        assert result.mask_for(ScatteringRegime.DIFFUSE).sum() == int(raster.valid_mask().sum())

    def test_constant_image_uses_reference_thresholds_and_says_so(
        self, make_raster, known_transform
    ):
        """A degenerate histogram must not produce a data-driven-looking answer."""
        flat = np.full((1, 96, 96), -12.0, dtype=np.float32)
        raster = load_raster(
            make_raster("const.tif", flat, transform=known_transform, descriptions=("vv",))
        )
        result = sar.analyze_sar(raster)
        assert result.threshold_method == "literature"
        assert result.smooth_threshold_db == sar.LITERATURE_SMOOTH_DB
        assert result.bright_threshold_db == sar.LITERATURE_BRIGHT_DB
        assert any("too uniform" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Honesty of the reported result
# ---------------------------------------------------------------------------
class TestReportedResultIsHonest:
    """What the module says about its own output must match what it measured."""

    def test_regimes_are_not_labelled_as_land_cover(self, analysed):
        """The central honesty claim of this module.

        SAR measures a scattering mechanism. Naming a dark pixel 'water' would convert a
        physical measurement into a land-cover claim single-channel data cannot support.
        """
        for stats in analysed.stats:
            assert stats.regime.value not in {"water", "vegetation", "built_up", "bare_soil"}

    def test_every_regime_offers_several_candidate_surfaces(self, analysed):
        """A single candidate would be a land-cover claim by another name."""
        for stats in analysed.stats:
            candidates = stats.regime.candidate_surfaces
            assert len(candidates) > 1, f"{stats.regime} names only {candidates}"

    def test_smooth_candidates_include_non_water_explanations(self):
        """Dark is not the same as wet: shadow and dry tarmac are equally consistent."""
        candidates = " ".join(ScatteringRegime.SMOOTH.candidate_surfaces).lower()
        assert "water" in candidates
        assert "shadow" in candidates
        assert "tarmac" in candidates or "sand" in candidates

    def test_the_diffuse_middle_is_never_split(self, analysed):
        """Single-pol VV cannot separate vegetation from bare soil, so it must not try."""
        assert any(
            "vegetation and bare soil cannot be separated" in w for w in analysed.warnings
        )
        assert len([s for s in analysed.stats if s.regime is ScatteringRegime.DIFFUSE]) == 1

    def test_mechanism_not_land_cover_is_stated_in_every_result(self, analysed):
        assert any("not land cover" in w for w in analysed.warnings)

    def test_regime_means_are_ordered_as_the_physics_requires(self, analysed):
        """Smooth must be darker than diffuse, which must be darker than double-bounce.

        A violation would mean regimes had been assigned to the wrong side of a threshold — a
        labelling bug that a single class's accuracy metric could miss.
        """
        by_regime = {s.regime: s.mean_db for s in analysed.stats}
        assert by_regime[ScatteringRegime.SMOOTH] < by_regime[ScatteringRegime.DIFFUSE]
        assert by_regime[ScatteringRegime.DIFFUSE] < by_regime[ScatteringRegime.DOUBLE_BOUNCE]

    def test_regime_means_land_near_the_nominal_backscatter(self, analysed):
        """Ties the segmentation back to the physical levels the scene was built from."""
        by_regime = {s.regime: s.mean_db for s in analysed.stats}
        assert by_regime[ScatteringRegime.SMOOTH] == pytest.approx(DEMO_WATER_DB, abs=1.5)
        assert by_regime[ScatteringRegime.DOUBLE_BOUNCE] == pytest.approx(
            DEMO_BUILT_DB, abs=1.5
        )

    def test_fractions_sum_to_one_over_valid_pixels(self, analysed):
        assert sum(s.fraction for s in analysed.stats) == pytest.approx(1.0, abs=1e-6)

    def test_pixel_counts_sum_to_valid_pixels(self, analysed, demo_sar):
        assert sum(s.pixel_count for s in analysed.stats) == int(demo_sar.valid_mask().sum())

    def test_areas_follow_from_pixel_counts_on_a_10m_grid(self, analysed):
        for stats in analysed.stats:
            assert stats.area_m2 == pytest.approx(stats.pixel_count * 100.0)

    def test_areas_are_none_without_georeferencing(self, make_raster):
        """Pixel counts yes, square metres no — coordinates must never be invented."""
        n = 160
        labels = np.zeros((n, n), dtype=bool)
        labels[20:110, 20:110] = True
        db = speckled_scene(labels, DEMO_WATER_DB, DEMO_VEG_DB, seed=17)
        raster = load_raster(
            make_raster(
                "nogeo.tif", db[None, :, :].astype(np.float32),
                transform=None, epsg=None, descriptions=("vv",),
            )
        )
        result = sar.analyze_sar(raster)
        assert all(s.area_m2 is None for s in result.stats)
        assert all(s.pixel_count > 0 for s in result.stats)

    def test_serialised_output_carries_the_evidence(self, analysed):
        """The API layer must be able to show *why*, not just *what*."""
        payload = analysed.to_dict()
        assert payload["mode_prominence"]["threshold"] == indices.MIN_MODE_PROMINENCE
        assert payload["mode_prominence"]["smooth"] >= indices.MIN_MODE_PROMINENCE
        assert payload["speckle"]["estimated_enl"] > DEMO_LOOKS
        assert payload["speckle"]["filter"] == "lee-mmse"
        assert payload["thresholds_db"]["method"] == "otsu-3class"
        assert payload["warnings"]
        assert payload["histogram"]
        assert {r["regime"] for r in payload["regimes"]} == {
            s.regime.value for s in analysed.stats
        }
        for entry in payload["regimes"]:
            assert len(entry["candidate_surfaces"]) > 1

    def test_unfiltered_analysis_reports_that_no_filter_ran(self, demo_sar):
        result = sar.analyze_sar(demo_sar, filter_speckle=False)
        assert result.speckle.filter_name == "none"
        assert np.isnan(result.speckle.estimated_enl)
        assert result.speckle.speckle_reduction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
class TestInputHandling:
    """Wrong, degenerate and mislabelled inputs must fail clearly or degrade honestly."""

    def test_optical_input_is_flagged_rather_than_silently_analysed(
        self, make_raster, six_band_scene, known_transform
    ):
        """Running backscatter analysis on reflectance is meaningless; say so."""
        raster = load_raster(
            make_raster(
                "optical.tif", six_band_scene, transform=known_transform,
                descriptions=("blue", "green", "red", "nir", "swir1", "swir2"),
            )
        )
        assert raster.modality is Modality.OPTICAL
        result = sar.analyze_sar(raster)
        assert any("identified as optical" in w for w in result.warnings)

    def test_empty_raster_is_rejected_with_a_usable_error(self, make_raster, known_transform):
        arr = np.full((1, 8, 8), np.nan, dtype=np.float32)
        raster = load_raster(
            make_raster(
                "empty.tif", arr, transform=known_transform, descriptions=("vv",),
                nodata=float("nan"),
            )
        )
        with pytest.raises(ValidationError) as excinfo:
            sar.analyze_sar(raster)
        assert excinfo.value.code == ErrorCode.NO_VALID_PIXELS
        # The message must be for a person, and must not leak internals.
        assert "enough valid data" in excinfo.value.message
        assert "Traceback" not in excinfo.value.message
        assert excinfo.value.recoverable

    def test_polarization_is_reported_when_known(self, analysed):
        assert analysed.polarization == "VV"

    def test_unknown_polarization_is_not_guessed(self, make_raster, known_transform):
        """Thresholds differ between polarisations, so an assumed label would mislead."""
        rng = np.random.default_rng(18)
        arr = (10 * np.log10(0.1 * rng.gamma(6, 1 / 6.0, size=(1, 96, 96)))).astype(np.float32)
        raster = load_raster(make_raster("nopol.tif", arr, transform=known_transform))
        result = sar.analyze_sar(raster)
        assert result.polarization == "unknown"
        assert any("polarisation of this image is not recorded" in w for w in result.warnings)

    def test_dual_pol_raster_reports_which_band_was_used(self, make_raster, known_transform):
        n = 128
        labels = np.zeros((n, n), dtype=bool)
        labels[20:90, 20:90] = True
        arr = np.stack(
            [
                speckled_scene(labels, DEMO_WATER_DB, DEMO_VEG_DB, seed=19),
                speckled_scene(labels, DEMO_WATER_DB - 3, DEMO_VEG_DB - 3, seed=20),
            ]
        ).astype(np.float32)
        raster = load_raster(
            make_raster("dual.tif", arr, transform=known_transform, descriptions=("vv", "vh"))
        )
        result = sar.analyze_sar(raster)
        assert result.polarization == "VV"
        assert any("2 bands" in w and "VV" in w for w in result.warnings)

    def test_uncalibrated_bright_values_are_flagged(self, make_raster, known_transform):
        """Backscatter far above 0 dB is not calibrated sigma-nought."""
        rng = np.random.default_rng(20)
        arr = rng.normal(-12.0, 3.0, size=(1, 96, 96)).astype(np.float32)
        arr[0, 40:56, 40:56] = 35.0
        raster = load_raster(
            make_raster("hot.tif", arr, transform=known_transform, descriptions=("vv",))
        )
        result = sar.analyze_sar(raster)
        assert any("may not be calibrated" in w for w in result.warnings)

    def test_partial_nodata_is_excluded_from_statistics(self, make_raster, known_transform):
        """Nodata must not be counted as a dark regime — that would invent water."""
        n = 160
        labels = np.zeros((n, n), dtype=bool)
        labels[20:110, 60:150] = True
        db = speckled_scene(labels, DEMO_WATER_DB, DEMO_VEG_DB, seed=21).astype(np.float32)
        db[:, :40] = -9999.0
        raster = load_raster(
            make_raster(
                "holed.tif", db[None, :, :], transform=known_transform,
                descriptions=("vv",), nodata=-9999.0,
            )
        )
        result = sar.analyze_sar(raster)
        total = sum(s.pixel_count for s in result.stats)
        assert total == int(raster.valid_mask().sum())
        assert total < n * n
        # The nodata column must not have been swept into the smooth regime.
        assert not result.mask_for(ScatteringRegime.SMOOTH)[:, :40].any()
