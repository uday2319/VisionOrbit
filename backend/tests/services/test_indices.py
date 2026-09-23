"""Tests for spectral indices and threshold selection.

Index values are checked against hand-computed arithmetic rather than against another
implementation, so a regression in the formula cannot hide behind a matching bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import AppError, ErrorCode
from app.geospatial.raster import load_raster
from app.services import indices


class TestNormalizedDifference:
    def test_matches_hand_computed_value(self):
        a = np.array([[0.5]], dtype=np.float32)
        b = np.array([[0.1]], dtype=np.float32)
        # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6666...
        assert indices.normalized_difference(a, b)[0, 0] == pytest.approx(2 / 3, abs=1e-6)

    def test_is_scale_invariant(self):
        """A ratio of two bands must be unchanged by a constant gain."""
        a = np.array([[0.4, 0.2]], dtype=np.float32)
        b = np.array([[0.1, 0.3]], dtype=np.float32)
        base = indices.normalized_difference(a, b)
        scaled = indices.normalized_difference(a * 10000, b * 10000)
        np.testing.assert_allclose(base, scaled, atol=1e-5)

    def test_zero_denominator_is_nan_not_infinity(self):
        """Both bands zero means nodata/deep shadow: the ratio is meaningless, not huge."""
        a = np.zeros((2, 2), dtype=np.float32)
        b = np.zeros((2, 2), dtype=np.float32)
        out = indices.normalized_difference(a, b)
        assert np.isnan(out).all()

    def test_opposite_signs_cancelling_denominator_is_nan(self):
        a = np.array([[1.0]], dtype=np.float32)
        b = np.array([[-1.0]], dtype=np.float32)
        assert np.isnan(indices.normalized_difference(a, b)[0, 0])

    def test_output_is_clipped_to_valid_range(self):
        a = np.array([[1.0, -1.0]], dtype=np.float32)
        b = np.array([[-0.5, 0.4]], dtype=np.float32)
        out = indices.normalized_difference(a, b)
        finite = out[np.isfinite(out)]
        assert finite.min() >= -1.0 and finite.max() <= 1.0


class TestAvailability:
    def test_six_band_scene_supports_full_index_set(self, six_band_scene, make_raster):
        path = make_raster(
            "s.tif", six_band_scene,
            descriptions=("blue", "green", "red", "nir", "swir1", "swir2"),
        )
        raster = load_raster(path)
        assert {"ndvi", "ndwi", "mndwi", "ndbi", "bsi", "exg"} <= indices.available_indices(raster)

    def test_rgb_only_scene_offers_no_nir_indices(self, make_raster):
        """The gate is band *availability*, so an index is never computed from a guess."""
        arr = (np.random.default_rng(0).random((3, 32, 32)) * 10000).astype(np.uint16)
        raster = load_raster(make_raster("rgb.tif", arr, descriptions=("red", "green", "blue")))
        available = indices.available_indices(raster)
        assert available == {"exg", "brightness"}
        assert "ndvi" not in available and "ndbi" not in available

    def test_missing_band_raises_rather_than_substituting(self, make_raster):
        arr = (np.random.default_rng(0).random((3, 32, 32)) * 10000).astype(np.uint16)
        raster = load_raster(make_raster("rgb.tif", arr, descriptions=("red", "green", "blue")))
        with pytest.raises(AppError) as excinfo:
            indices.compute_index(raster, "ndvi")
        assert excinfo.value.code == ErrorCode.MISSING_BAND

    def test_unknown_index_name_rejected(self, six_band_scene, make_raster):
        raster = load_raster(make_raster(
            "s.tif", six_band_scene,
            descriptions=("blue", "green", "red", "nir", "swir1", "swir2"),
        ))
        with pytest.raises(ValueError, match="unknown index"):
            indices.compute_index(raster, "definitely_not_an_index")

    def test_compute_all_available_skips_nothing_it_advertises(self, six_band_scene, make_raster):
        raster = load_raster(make_raster(
            "s.tif", six_band_scene,
            descriptions=("blue", "green", "red", "nir", "swir1", "swir2"),
        ))
        assert set(indices.compute_all_available(raster)) == indices.available_indices(raster)


class TestIndexSigns:
    """Signs must be unambiguous on a scene built with known physics."""

    @pytest.fixture
    def raster(self, six_band_scene, make_raster, known_transform):
        return load_raster(make_raster(
            "s.tif", six_band_scene, transform=known_transform,
            descriptions=("blue", "green", "red", "nir", "swir1", "swir2"),
        ))

    def test_ndvi_positive_on_vegetation_negative_on_water(self, raster, water_square_bounds):
        r0, r1, c0, c1 = water_square_bounds
        ndvi = indices.compute_index(raster, "ndvi").array
        assert ndvi[r0:r1, c0:c1].mean() < 0, "water has near-zero NIR so NDVI must be negative"
        assert ndvi[0:5, 0:5].mean() > 0.5, "high-NIR vegetation must give strongly positive NDVI"

    def test_ndwi_positive_on_water_negative_on_vegetation(self, raster, water_square_bounds):
        r0, r1, c0, c1 = water_square_bounds
        ndwi = indices.compute_index(raster, "ndwi").array
        assert ndwi[r0:r1, c0:c1].mean() > 0
        assert ndwi[0:5, 0:5].mean() < 0

    def test_mndwi_positive_on_water(self, raster, water_square_bounds):
        r0, r1, c0, c1 = water_square_bounds
        mndwi = indices.compute_index(raster, "mndwi").array
        assert mndwi[r0:r1, c0:c1].mean() > 0

    def test_ndvi_and_ndwi_disagree_by_construction(self, raster):
        """A pixel cannot be strongly vegetated and open water at once."""
        ndvi = indices.compute_index(raster, "ndvi").array
        ndwi = indices.compute_index(raster, "ndwi").array
        both = (ndvi > 0.3) & (ndwi > 0.0)
        assert both.sum() == 0

    def test_records_formula_and_bands_used(self, raster):
        result = indices.compute_index(raster, "ndvi")
        assert result.bands_used == ("nir", "red")
        assert "NIR" in result.formula and "Red" in result.formula
        assert result.valid_fraction == pytest.approx(1.0)

    def test_to_dict_reports_real_statistics(self, raster):
        payload = indices.compute_index(raster, "ndvi").to_dict()
        arr = indices.compute_index(raster, "ndvi").array
        assert payload["mean"] == pytest.approx(float(arr.mean()), abs=1e-4)
        assert payload["min"] == pytest.approx(float(arr.min()), abs=1e-4)


class TestOtsuThreshold:
    def test_separates_two_known_populations(self):
        """Any cut inside an empty gap is an equally valid Otsu result, so the property to
        assert is that the threshold actually separates the populations — not where in the
        gap it lands."""
        rng = np.random.default_rng(1)
        low = rng.normal(-1.0, 0.05, 5000)
        high = rng.normal(1.0, 0.05, 5000)
        thr, quality = indices.otsu_threshold(np.concatenate([low, high]))
        assert (low < thr).all(), "every low-mode sample must fall below the threshold"
        assert (high > thr).all(), "every high-mode sample must fall above the threshold"
        assert quality > 0.9, "a cleanly bimodal histogram must report high quality"

    def test_threshold_lies_within_the_data_range(self):
        rng = np.random.default_rng(12)
        values = np.concatenate([rng.normal(-0.6, 0.05, 5000), rng.normal(0.6, 0.05, 5000)])
        thr, _ = indices.otsu_threshold(values)
        assert values.min() <= thr <= values.max()

    def test_threshold_lands_in_the_middle_of_a_wide_empty_valley(self):
        """Regression: every cut in an empty gap scores identically, so a plain ``argmax``
        returns the gap's left edge and can land inside the lower mode's tail. The threshold
        must sit in the valley, symmetrically between two symmetric modes."""
        rng = np.random.default_rng(15)
        values = np.concatenate([rng.normal(-1.0, 0.05, 5000), rng.normal(1.0, 0.05, 5000)])
        thr, _ = indices.otsu_threshold(values)
        assert thr == pytest.approx(0.0, abs=0.05)

    def test_threshold_is_stable_across_bin_counts(self):
        """A threshold that moves with histogram resolution is an artefact, not a decision."""
        rng = np.random.default_rng(16)
        values = np.concatenate([rng.normal(-1.0, 0.05, 5000), rng.normal(1.0, 0.05, 5000)])
        found = [indices.otsu_threshold(values, bins=b)[0] for b in (64, 256, 1024)]
        assert max(found) - min(found) < 0.05

    def test_unimodal_histogram_reports_low_quality(self):
        """This is the evidence signal: one population means the cut is through noise."""
        values = np.random.default_rng(2).normal(0.0, 1.0, 10000)
        _, quality = indices.otsu_threshold(values)
        assert quality < 0.75

    def test_unimodal_gaussian_quality_approaches_two_over_pi(self):
        """Splitting a Gaussian at its mean explains exactly 2/pi of the variance.

        This is why the between-class variance ratio alone cannot certify bimodality, and
        why :func:`adaptive_threshold` also requires a bimodality test.
        """
        values = np.random.default_rng(13).normal(0.0, 1.0, 200_000)
        _, quality = indices.otsu_threshold(values)
        assert quality == pytest.approx(2 / np.pi, abs=0.02)

    def test_uniform_scores_higher_than_overlapping_bimodal(self):
        """Documents the failure mode the bimodality test exists to catch."""
        rng = np.random.default_rng(14)
        uniform = rng.uniform(-1, 1, 20000)
        overlapping = np.concatenate([rng.normal(-0.5, 0.5, 5000), rng.normal(0.5, 0.5, 5000)])
        assert indices.otsu_threshold(uniform)[1] > indices.otsu_threshold(overlapping)[1]

    def test_bimodal_quality_exceeds_unimodal_quality(self):
        rng = np.random.default_rng(3)
        bimodal = np.concatenate([rng.normal(-1, 0.1, 4000), rng.normal(1, 0.1, 4000)])
        unimodal = rng.normal(0, 1, 8000)
        assert indices.otsu_threshold(bimodal)[1] > indices.otsu_threshold(unimodal)[1]

    def test_too_few_samples_returns_nan(self):
        thr, quality = indices.otsu_threshold(np.array([0.1, 0.2, 0.3]))
        assert np.isnan(thr) and quality == 0.0

    def test_constant_input_returns_nan(self):
        thr, quality = indices.otsu_threshold(np.full(500, 0.42))
        assert np.isnan(thr) and quality == 0.0

    def test_ignores_nan_values(self):
        rng = np.random.default_rng(4)
        values = np.concatenate([rng.normal(-1, 0.1, 2000), rng.normal(1, 0.1, 2000)])
        with_nan = np.concatenate([values, np.full(500, np.nan)])
        assert indices.otsu_threshold(with_nan)[0] == pytest.approx(
            indices.otsu_threshold(values)[0], abs=0.05
        )

    def test_all_nan_returns_nan(self):
        thr, quality = indices.otsu_threshold(np.full(100, np.nan))
        assert np.isnan(thr) and quality == 0.0


class TestSeparability:
    def test_well_separated_populations_score_high(self):
        values = np.concatenate([np.full(500, 1.0), np.full(500, -1.0)])
        mask = np.zeros(1000, dtype=bool)
        mask[:500] = True
        assert indices.separability(values, mask) > 0.9

    def test_identical_populations_score_zero(self):
        values = np.random.default_rng(5).normal(0, 1, 1000)
        mask = np.zeros(1000, dtype=bool)
        mask[::2] = True
        assert indices.separability(values, mask) < 0.15

    def test_result_is_bounded(self):
        values = np.concatenate([np.full(500, 1000.0), np.full(500, -1000.0)])
        mask = np.zeros(1000, dtype=bool)
        mask[:500] = True
        assert 0.0 <= indices.separability(values, mask) <= 1.0

    def test_too_small_group_scores_zero(self):
        """Fewer than 8 pixels is not a population; refuse to claim separability."""
        values = np.random.default_rng(6).normal(0, 1, 1000)
        mask = np.zeros(1000, dtype=bool)
        mask[:3] = True
        assert indices.separability(values, mask) == 0.0

    def test_empty_mask_scores_zero(self):
        values = np.random.default_rng(7).normal(0, 1, 100)
        assert indices.separability(values, np.zeros(100, dtype=bool)) == 0.0


class TestBimodalityCoefficient:
    """Sarle's coefficient has exact known values, so these are true reference tests."""

    def test_normal_distribution_scores_one_third(self):
        values = np.random.default_rng(20).normal(0.0, 1.0, 200_000)
        assert indices.bimodality_coefficient(values) == pytest.approx(1 / 3, abs=0.02)

    def test_uniform_distribution_scores_five_ninths(self):
        values = np.random.default_rng(21).uniform(0.0, 1.0, 200_000)
        assert indices.bimodality_coefficient(values) == pytest.approx(5 / 9, abs=0.02)

    def test_two_point_distribution_scores_one(self):
        values = np.tile([0.0, 1.0], 50_000)
        assert indices.bimodality_coefficient(values) == pytest.approx(1.0, abs=0.01)

    def test_is_invariant_to_shift_and_scale(self):
        """The coefficient is built from standardised moments, so units cannot change it."""
        rng = np.random.default_rng(22)
        values = np.concatenate([rng.normal(-1, 0.1, 5000), rng.normal(1, 0.1, 5000)])
        base = indices.bimodality_coefficient(values)
        assert indices.bimodality_coefficient(values * 1000 + 500) == pytest.approx(base, abs=1e-6)

    def test_normal_falls_below_threshold_and_two_point_above(self):
        rng = np.random.default_rng(23)
        normal = rng.normal(0, 1, 50_000)
        split = np.concatenate([rng.normal(-1, 0.05, 5000), rng.normal(1, 0.05, 5000)])
        assert indices.bimodality_coefficient(normal) < indices.BIMODALITY_THRESHOLD
        assert indices.bimodality_coefficient(split) > indices.BIMODALITY_THRESHOLD

    def test_detects_bimodality_despite_unequal_group_sizes(self):
        """A 70/30 split is still two populations; the test must not require balance."""
        rng = np.random.default_rng(24)
        values = np.concatenate([rng.normal(-1, 0.08, 7000), rng.normal(1, 0.08, 3000)])
        assert indices.bimodality_coefficient(values) > indices.BIMODALITY_THRESHOLD

    def test_too_few_samples_scores_zero(self):
        assert indices.bimodality_coefficient(np.arange(10.0)) == 0.0

    def test_constant_input_scores_zero(self):
        assert indices.bimodality_coefficient(np.full(100, 0.5)) == 0.0

    def test_all_nan_scores_zero(self):
        assert indices.bimodality_coefficient(np.full(100, np.nan)) == 0.0

    def test_result_is_bounded(self):
        rng = np.random.default_rng(25)
        for values in (rng.normal(0, 1, 1000), rng.exponential(1, 1000), np.tile([0, 1], 500)):
            assert 0.0 <= indices.bimodality_coefficient(np.asarray(values, dtype=float)) <= 1.0


class TestAdaptiveThreshold:
    def test_prefers_otsu_when_histogram_is_bimodal(self):
        rng = np.random.default_rng(8)
        values = np.concatenate([rng.normal(-0.5, 0.03, 5000), rng.normal(0.5, 0.03, 5000)])
        thr, method, quality = indices.adaptive_threshold(values, fallback=0.9)
        assert method == "otsu"
        assert thr != pytest.approx(0.9)
        assert quality >= 0.55

    def test_falls_back_to_literature_when_unimodal(self):
        values = np.random.default_rng(9).normal(0.0, 1.0, 10000)
        thr, method, _ = indices.adaptive_threshold(values, fallback=0.3)
        assert method == "literature"
        assert thr == pytest.approx(0.3)

    def test_rejects_otsu_when_variance_is_high_but_shape_is_unimodal(self):
        """The regression this guards: a single Gaussian clears the variance-ratio bar
        (~2/pi = 0.64) yet is not two populations, so it must not be labelled ``otsu``."""
        values = np.random.default_rng(30).normal(0.0, 1.0, 50_000)
        _, quality = indices.otsu_threshold(values)
        assert quality >= 0.55, "precondition: the variance ratio alone would accept this"
        _, method, _ = indices.adaptive_threshold(values, fallback=0.1)
        assert method == "literature", "the bimodality test must veto it"

    def test_bimodality_gate_can_be_relaxed_explicitly(self):
        """Callers may opt out, but only deliberately — the default stays strict."""
        values = np.random.default_rng(31).normal(0.0, 1.0, 50_000)
        _, strict, _ = indices.adaptive_threshold(values, 0.1)
        _, relaxed, _ = indices.adaptive_threshold(values, 0.1, min_bimodality=0.0)
        assert strict == "literature" and relaxed == "otsu"

    def test_quality_is_reported_even_when_falling_back(self):
        """The confidence engine needs the measured quality regardless of which cut won."""
        values = np.random.default_rng(32).normal(0.0, 1.0, 50_000)
        _, method, quality = indices.adaptive_threshold(values, 0.1)
        assert method == "literature"
        assert quality == pytest.approx(indices.otsu_threshold(values)[1])

    def test_degenerate_input_falls_back(self):
        _, method, quality = indices.adaptive_threshold(np.full(500, 0.42), fallback=0.25)
        assert method == "literature" and quality == 0.0

    def test_reported_method_is_honest_about_provenance(self):
        """The confidence engine relies on this label, so it must never be guessed."""
        rng = np.random.default_rng(10)
        bimodal = np.concatenate([rng.normal(-1, 0.05, 3000), rng.normal(1, 0.05, 3000)])
        _, method, quality = indices.adaptive_threshold(bimodal, 0.0)
        assert method == "otsu" and quality >= 0.55
        _, method2, quality2 = indices.adaptive_threshold(bimodal, 0.0, min_otsu_quality=0.999)
        assert method2 == "literature" and quality2 == pytest.approx(quality)


class TestHistogramSummary:
    def test_reports_real_statistics(self):
        values = np.random.default_rng(11).normal(0.25, 0.1, 5000)
        summary = indices.histogram_summary(values, bins=16)
        assert len(summary["counts"]) == 16
        assert len(summary["bins"]) == 17
        assert summary["mean"] == pytest.approx(float(values.mean()), abs=1e-3)
        assert sum(summary["counts"]) == values.size

    def test_empty_input_yields_nulls_not_zeros(self):
        summary = indices.histogram_summary(np.array([]))
        assert summary["mean"] is None and summary["counts"] == []

    def test_all_nan_input_yields_nulls(self):
        summary = indices.histogram_summary(np.full(50, np.nan))
        assert summary["mean"] is None
