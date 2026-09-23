"""Tests for land-cover classification.

Two kinds of assertion appear here:

* **Behavioural** — the decision hierarchy is exclusive, areas follow from pixel counts,
  degraded inputs produce warnings instead of silent guesses.
* **Accuracy** — measured against the synthetic scene's ground-truth label map. These are
  real metrics computed at test time, not quoted numbers. Bounds are set below the current
  measurement so the suite catches regressions without being a tuning target.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.types import BandRole, LandCoverClass
from app.geospatial.raster import load_raster
from app.services import indices
from app.services.landcover import (
    MIN_BUILTUP_PATCH_M2,
    _min_patch_pixels,
    _region_threshold,
    classify_land_cover,
)

# Class codes as written by the demo-data generator's ground-truth raster.
GT_WATER, GT_VEGETATION, GT_BUILT_UP, GT_BARE_SOIL = 1, 2, 3, 4


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    union = int((pred | truth).sum())
    return int((pred & truth).sum()) / union if union else 0.0


def precision_recall(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    hit = int((pred & truth).sum())
    p = hit / int(pred.sum()) if pred.sum() else 0.0
    r = hit / int(truth.sum()) if truth.sum() else 0.0
    return p, r


@pytest.fixture(scope="module")
def optical_scene(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def ground_truth(demo_root):
    return load_raster(
        demo_root / "optical" / "scene_landcover_gt.tif", max_edge=None
    ).data[0].astype(int)


@pytest.fixture(scope="module")
def spectral_result(optical_scene):
    return classify_land_cover(optical_scene)


@pytest.fixture(scope="module")
def rgb_scene(demo_root):
    return load_raster(demo_root / "edge_cases" / "rgb_only.tif", max_edge=None)


class TestSpectralPathStructure:
    def test_selects_the_spectral_method_when_nir_is_present(self, spectral_result):
        assert spectral_result.method == "spectral-indices"

    def test_reports_the_indices_it_actually_used(self, spectral_result):
        assert set(spectral_result.indices_used) == {"mndwi", "ndvi", "ndbi"}

    def test_classes_are_mutually_exclusive(self, spectral_result):
        """Every pixel gets exactly one label; overlapping masks would double-count area."""
        masks = [spectral_result.mask_for(c) for c in LandCoverClass if c.value != "unclassified"]
        assert np.sum(np.stack(masks), axis=0).max() <= 1

    def test_fractions_sum_to_one(self, spectral_result):
        assert sum(s.fraction for s in spectral_result.stats) == pytest.approx(1.0, abs=1e-6)

    def test_pixel_counts_sum_to_valid_pixels(self, spectral_result, optical_scene):
        total = sum(s.pixel_count for s in spectral_result.stats)
        assert total == int(optical_scene.valid_mask().sum())

    def test_area_is_pixel_count_times_pixel_area(self, spectral_result):
        """10 m pixels means every class area must be an exact multiple of 100 m²."""
        for stat in spectral_result.stats:
            assert stat.area_m2 == pytest.approx(stat.pixel_count * 100.0)

    def test_dominant_class_is_the_largest_real_class(self, spectral_result):
        dominant = spectral_result.dominant()
        assert dominant is not None
        assert dominant.fraction == max(s.fraction for s in spectral_result.stats)

    def test_clean_scene_produces_no_warnings(self, spectral_result):
        assert spectral_result.warnings == []

    def test_separability_is_measured_and_bounded(self, spectral_result):
        assert 0.0 < spectral_result.separability <= 1.0

    def test_is_deterministic(self, optical_scene):
        """No randomness may influence a measurement."""
        a = classify_land_cover(optical_scene).class_map
        b = classify_land_cover(optical_scene).class_map
        np.testing.assert_array_equal(a, b)

    def test_serialises_without_numpy_types(self, spectral_result):
        payload = spectral_result.to_dict()
        assert isinstance(payload["classes"], list)
        assert all(isinstance(c["pixel_count"], int) for c in payload["classes"])
        assert payload["dominant"]["class"] in {c.value for c in LandCoverClass}


class TestSpectralAccuracy:
    """Real metrics, computed here against the ground-truth raster."""

    def test_water_is_detected_precisely(self, spectral_result, ground_truth):
        pred = spectral_result.mask_for(LandCoverClass.WATER)
        truth = ground_truth == GT_WATER
        p, r = precision_recall(pred, truth)
        assert iou(pred, truth) > 0.85
        assert p > 0.95, "water must not be over-reported"
        assert r > 0.85

    def test_vegetation_is_detected_reliably(self, spectral_result, ground_truth):
        pred = spectral_result.mask_for(LandCoverClass.VEGETATION)
        truth = ground_truth == GT_VEGETATION
        assert iou(pred, truth) > 0.90

    def test_bare_soil_is_detected_reliably(self, spectral_result, ground_truth):
        pred = spectral_result.mask_for(LandCoverClass.BARE_SOIL)
        truth = ground_truth == GT_BARE_SOIL
        assert iou(pred, truth) > 0.90

    def test_built_up_beats_the_spectral_only_baseline(self, spectral_result, ground_truth):
        """NDBI alone scores IoU ~0.07 here because bare soil is also SWIR-bright. Requiring
        edge density as well is what lifts it; this bound documents that gain."""
        pred = spectral_result.mask_for(LandCoverClass.BUILT_UP)
        truth = ground_truth == GT_BUILT_UP
        p, _ = precision_recall(pred, truth)
        assert iou(pred, truth) > 0.55
        assert p > 0.60

    def test_built_up_is_not_wildly_over_predicted(self, spectral_result, ground_truth):
        """The failure mode being guarded: flagging most of the scene as built-up."""
        pred = spectral_result.mask_for(LandCoverClass.BUILT_UP)
        truth = ground_truth == GT_BUILT_UP
        assert pred.sum() < 2.0 * truth.sum()

    def test_overall_accuracy_and_mean_iou(self, spectral_result, ground_truth):
        pred = spectral_result.class_map.astype(int)
        assert (pred == ground_truth).mean() > 0.93
        ious = [
            iou(pred == code, ground_truth == code)
            for code in (GT_WATER, GT_VEGETATION, GT_BUILT_UP, GT_BARE_SOIL)
        ]
        assert float(np.mean(ious)) > 0.80

    def test_every_class_is_found(self, spectral_result):
        found = {s.label for s in spectral_result.stats if s.pixel_count > 0}
        assert {
            LandCoverClass.WATER, LandCoverClass.VEGETATION,
            LandCoverClass.BUILT_UP, LandCoverClass.BARE_SOIL,
        } <= found


class TestRgbFallback:
    def test_selects_the_approximation_method_without_nir(self, rgb_scene):
        assert classify_land_cover(rgb_scene).method == "rgb-approximation"

    def test_warns_that_nir_is_missing(self, rgb_scene):
        warnings = " ".join(classify_land_cover(rgb_scene).warnings).lower()
        assert "near-infrared" in warnings

    def test_refuses_to_claim_built_up_without_swir(self, rgb_scene):
        """Built-up and bare soil overlap so heavily in visible bands that every threshold
        rule tested gave precision below 0.15. Emitting no built-up is the honest outcome."""
        result = classify_land_cover(rgb_scene)
        assert result.mask_for(LandCoverClass.BUILT_UP).sum() == 0
        assert any("short-wave infrared" in w for w in result.warnings)

    def test_detects_water_despite_no_nir(self, rgb_scene, ground_truth):
        """Regression: Excess Green cannot separate water from vegetation because both are
        green-dominant in the visible, so testing vegetation first silently absorbed all
        water. Water must be tested first."""
        pred = classify_land_cover(rgb_scene).mask_for(LandCoverClass.WATER)
        truth = ground_truth == GT_WATER
        assert pred.sum() > 0, "water must not be swallowed by the vegetation test"
        assert iou(pred, truth) > 0.85

    def test_detects_vegetation(self, rgb_scene, ground_truth):
        pred = classify_land_cover(rgb_scene).mask_for(LandCoverClass.VEGETATION)
        assert iou(pred, ground_truth == GT_VEGETATION) > 0.90

    def test_unvegetated_ground_covers_both_built_up_and_soil(self, rgb_scene, ground_truth):
        """Since the two are not separated, the combined class must capture both."""
        pred = classify_land_cover(rgb_scene).mask_for(LandCoverClass.BARE_SOIL)
        truth = np.isin(ground_truth, [GT_BUILT_UP, GT_BARE_SOIL])
        assert iou(pred, truth) > 0.90

    def test_is_measurably_weaker_than_the_spectral_path(self, rgb_scene, spectral_result):
        """The degraded path must not advertise itself as equally capable."""
        rgb = classify_land_cover(rgb_scene)
        assert len(rgb.warnings) > len(spectral_result.warnings)
        assert len(rgb.indices_used) < len(spectral_result.indices_used)

    def test_fractions_still_sum_to_one(self, rgb_scene):
        assert sum(s.fraction for s in classify_land_cover(rgb_scene).stats) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_is_deterministic(self, rgb_scene):
        np.testing.assert_array_equal(
            classify_land_cover(rgb_scene).class_map, classify_land_cover(rgb_scene).class_map
        )


class TestInsufficientBands:
    def test_single_band_scene_refuses_to_classify(self, make_raster, known_transform):
        arr = (np.random.default_rng(0).random((1, 32, 32)) * 10000).astype(np.uint16)
        raster = load_raster(make_raster("gray.tif", arr, transform=known_transform))
        result = classify_land_cover(raster)
        assert result.method == "insufficient-bands"
        assert result.separability == 0.0
        assert any("red, green and blue" in w for w in result.warnings)

    def test_refusal_reports_no_classes_rather_than_guessing(self, make_raster, known_transform):
        arr = (np.random.default_rng(0).random((1, 32, 32)) * 10000).astype(np.uint16)
        raster = load_raster(make_raster("gray.tif", arr, transform=known_transform))
        result = classify_land_cover(raster)
        assert result.dominant() is None
        assert result.stats == []


class TestAreaReporting:
    def test_no_area_without_georeferencing(self, demo_root):
        """§8 honesty rule: a plain image has no ground area to report."""
        raster = load_raster(demo_root / "edge_cases" / "no_georeference.tif", max_edge=None)
        result = classify_land_cover(raster)
        assert all(s.area_m2 is None for s in result.stats)
        assert all(s.pixel_count > 0 for s in result.stats)

    def test_area_scales_with_pixel_size(self, demo_root):
        """A 20 m pixel covers 4x the ground of a 10 m pixel."""
        raster = load_raster(demo_root / "edge_cases" / "different_res.tif", max_edge=None)
        result = classify_land_cover(raster)
        for stat in result.stats:
            if stat.area_m2 is not None:
                assert stat.area_m2 == pytest.approx(stat.pixel_count * 400.0)

    def test_geographic_crs_reports_areas_in_metres_not_degrees(self, demo_root):
        """``different_crs.tif`` is in EPSG:4326, where pixel size is in degrees.

        Multiplying a pixel count by a degrees-squared "area" would report a city as
        occupying a fraction of a square metre. The implied per-pixel area must land near
        the 100 m² of the same scene in UTM.
        """
        raster = load_raster(demo_root / "edge_cases" / "different_crs.tif", max_edge=None)
        result = classify_land_cover(raster)
        measured = [s for s in result.stats if s.area_m2 is not None]
        assert measured, "a georeferenced scene must yield areas"
        for stat in measured:
            assert 80.0 < stat.area_m2 / stat.pixel_count < 110.0


class TestGeographicCrsRegression:
    """A geographic CRS used to silently erase every built-up detection.

    ``_min_patch_pixels`` divided 2500 m² by a pixel area read straight from ``pixel_size``,
    which is in *degrees* here. The result was 3·10¹¹ pixels against a 262,144-pixel frame,
    so the minimum-patch filter removed the entire built-up class — no exception, no warning,
    just a plausible-looking map with no settlements in it.
    """

    @pytest.fixture(scope="class")
    def geographic_result(self, demo_root):
        raster = load_raster(demo_root / "edge_cases" / "different_crs.tif", max_edge=None)
        return classify_land_cover(raster)

    def test_min_patch_size_stays_inside_the_frame(self, demo_root):
        raster = load_raster(demo_root / "edge_cases" / "different_crs.tif", max_edge=None)
        h, w = raster.shape
        assert 1 <= _min_patch_pixels(raster) < h * w

    def test_built_up_is_still_detected(self, geographic_result):
        built = [
            s for s in geographic_result.stats
            if s.label is LandCoverClass.BUILT_UP
        ]
        assert built, "built-up vanished entirely — the min-patch filter erased it"
        assert built[0].pixel_count > 0

    def test_finds_the_same_classes_as_the_projected_scene(self, geographic_result, demo_root):
        """Reprojecting the same ground must not change *which* classes exist."""
        utm = classify_land_cover(
            load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)
        )
        assert {s.label for s in geographic_result.stats} == {s.label for s in utm.stats}

    def test_class_proportions_are_close_to_the_projected_scene(
        self, geographic_result, demo_root
    ):
        """Resampling shifts boundaries slightly, but not by tens of percent."""
        utm = classify_land_cover(
            load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)
        )
        utm_fracs = {s.label: s.fraction for s in utm.stats}
        for stat in geographic_result.stats:
            assert stat.fraction == pytest.approx(utm_fracs[stat.label], abs=0.02), stat.label


class TestMinPatchSizing:
    def test_converts_ground_area_to_pixels_at_ten_metres(self, optical_scene):
        assert _min_patch_pixels(optical_scene) == int(MIN_BUILTUP_PATCH_M2 / 100.0)

    def test_coarser_pixels_need_fewer_of_them(self, demo_root):
        """Expressing the floor as an area keeps it meaningful across resolutions."""
        fine = load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)
        coarse = load_raster(demo_root / "edge_cases" / "different_res.tif", max_edge=None)
        assert _min_patch_pixels(coarse) < _min_patch_pixels(fine)

    def test_falls_back_to_a_pixel_count_without_georeferencing(self, demo_root):
        raster = load_raster(demo_root / "edge_cases" / "no_georeference.tif", max_edge=None)
        assert _min_patch_pixels(raster) >= 1

    def test_never_returns_zero(self, demo_root):
        for name in ("optical/scene_optical.tif", "edge_cases/different_res.tif",
                     "edge_cases/no_georeference.tif"):
            assert _min_patch_pixels(load_raster(demo_root / name, max_edge=None)) >= 1


class TestRegionThreshold:
    def test_uses_only_pixels_inside_the_region(self):
        """Leaving other classes in would drag the cut away from the real boundary."""
        arr = np.concatenate([np.full(500, -5.0), np.full(500, 0.0), np.full(500, 1.0)])
        region = np.zeros(1500, dtype=bool)
        region[500:] = True  # exclude the far-off -5 population
        thr, _, _ = _region_threshold(arr, region, fallback=-2.0)
        assert thr > -1.0, "the excluded population must not influence the threshold"

    def test_raises_a_permissive_literature_fallback_to_the_region_median(self):
        """A whole-scene threshold is too permissive inside a pre-restricted subset."""
        arr = np.random.default_rng(0).normal(0.5, 0.1, 5000)
        region = np.ones(5000, dtype=bool)
        thr, method, _ = _region_threshold(arr, region, fallback=0.0)
        assert method == "literature"
        assert thr == pytest.approx(float(np.median(arr)), abs=1e-6)

    def test_keeps_a_stricter_literature_fallback_unchanged(self):
        arr = np.random.default_rng(1).normal(0.0, 0.1, 5000)
        region = np.ones(5000, dtype=bool)
        thr, method, _ = _region_threshold(arr, region, fallback=0.9)
        assert method == "literature"
        assert thr == pytest.approx(0.9)

    def test_prefers_otsu_on_a_genuinely_bimodal_region(self):
        rng = np.random.default_rng(2)
        arr = np.concatenate([rng.normal(-1, 0.05, 3000), rng.normal(1, 0.05, 3000)])
        region = np.ones(6000, dtype=bool)
        thr, method, quality = _region_threshold(arr, region, fallback=0.9)
        assert method == "otsu"
        assert quality > 0.9
        assert -0.5 < thr < 0.5

    def test_tiny_region_falls_back_without_error(self):
        arr = np.random.default_rng(3).normal(0, 1, 100)
        region = np.zeros(100, dtype=bool)
        region[:4] = True
        thr, method, quality = _region_threshold(arr, region, fallback=0.25)
        assert method == "literature" and thr == pytest.approx(0.25) and quality == 0.0

    def test_empty_region_falls_back_without_error(self):
        arr = np.random.default_rng(4).normal(0, 1, 100)
        thr, method, _ = _region_threshold(arr, np.zeros(100, dtype=bool), fallback=0.4)
        assert method == "literature" and thr == pytest.approx(0.4)


class TestEvidenceOfSeparability:
    def test_built_up_requires_both_cues_to_be_informative(self, optical_scene, ground_truth):
        """Documents *why* the classifier needs edge density: NDBI cannot separate built-up
        from bare soil on its own. Both are measured here, not assumed."""
        from app.services import features

        ndbi = indices.compute_index(optical_scene, "ndbi").array
        edges = features.edge_density(np.nanmean(optical_scene.data, axis=0), window=9)
        built = ground_truth == GT_BUILT_UP
        soil = ground_truth == GT_BARE_SOIL

        def cohens_d(values: np.ndarray) -> float:
            a, b = values[built], values[soil]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            pooled = float(np.sqrt(0.5 * (a.var() + b.var())))
            return abs(float(a.mean()) - float(b.mean())) / pooled if pooled > 1e-9 else 0.0

        assert cohens_d(edges) > cohens_d(ndbi), "edge density must be the stronger cue"
        assert cohens_d(edges) > 1.0, "and strong enough to act on"

    def test_ndbi_is_positive_for_both_built_up_and_bare_soil(self, optical_scene, ground_truth):
        """The root cause of the original over-prediction, asserted directly."""
        ndbi = indices.compute_index(optical_scene, "ndbi").array
        assert float(np.nanmean(ndbi[ground_truth == GT_BUILT_UP])) > 0
        assert float(np.nanmean(ndbi[ground_truth == GT_BARE_SOIL])) > 0


class TestMaskAccessors:
    def test_mask_for_matches_the_reported_pixel_count(self, spectral_result):
        for stat in spectral_result.stats:
            assert int(spectral_result.mask_for(stat.label).sum()) == stat.pixel_count

    def test_mask_for_an_absent_class_is_empty(self, rgb_scene):
        assert not classify_land_cover(rgb_scene).mask_for(LandCoverClass.BUILT_UP).any()

    def test_class_map_contains_only_known_codes(self, spectral_result):
        assert set(np.unique(spectral_result.class_map)) <= {0, 1, 2, 3, 4}


class TestBandRoleDependence:
    def test_spectral_path_requires_nir_role(self, optical_scene, rgb_scene):
        assert BandRole.NIR in set(optical_scene.band_roles)
        assert BandRole.NIR not in set(rgb_scene.band_roles)
