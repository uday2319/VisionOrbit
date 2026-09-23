"""Tests for spatial feature extraction.

Texture and morphology operators are checked against synthetic patterns whose correct answer
is known by construction — a flat field has zero variance, a checkerboard is maximally
edge-dense, a disc is compact — rather than against remembered output values.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services import features


class TestNormalize01:
    def test_maps_range_to_unit_interval(self):
        arr = np.linspace(10.0, 20.0, 1000).astype(np.float32)
        out = features.normalize01(arr, lo_pct=0.0, hi_pct=100.0)
        assert out.min() == pytest.approx(0.0, abs=1e-5)
        assert out.max() == pytest.approx(1.0, abs=1e-5)

    def test_is_monotonic(self):
        """Stretching may rescale but must never reorder pixels."""
        arr = np.array([1.0, 5.0, 3.0, 9.0, 7.0], dtype=np.float32)
        out = features.normalize01(arr, lo_pct=0.0, hi_pct=100.0)
        assert list(np.argsort(out)) == list(np.argsort(arr))

    def test_percentile_clipping_saturates_outliers(self):
        """Extreme values must not compress the useful range into a sliver."""
        body = np.linspace(4.0, 6.0, 998, dtype=np.float32)
        arr = np.concatenate([body, np.array([-1000.0, 1000.0], dtype=np.float32)])
        out = features.normalize01(arr, lo_pct=1.0, hi_pct=99.0)
        assert out[-2] == 0.0 and out[-1] == 1.0, "outliers clip to the ends"
        # The bulk of the data must still span most of [0, 1] rather than collapsing.
        assert out[:998].max() - out[:998].min() > 0.9

    def test_constant_input_with_outliers_is_treated_as_degenerate(self):
        """When percentiles coincide there is no range to stretch; return zeros, not NaN."""
        arr = np.concatenate([np.full(998, 5.0), [-1000.0, 1000.0]]).astype(np.float32)
        out = features.normalize01(arr, lo_pct=1.0, hi_pct=99.0)
        assert np.all(out == 0.0)

    def test_constant_input_returns_zeros_not_nan(self):
        out = features.normalize01(np.full((8, 8), 3.0, dtype=np.float32))
        assert np.all(out == 0.0)

    def test_all_nan_input_returns_zeros(self):
        out = features.normalize01(np.full((8, 8), np.nan, dtype=np.float32))
        assert np.all(out == 0.0)

    def test_output_is_bounded_even_with_nan_present(self):
        arr = np.array([[1.0, np.nan], [5.0, 9.0]], dtype=np.float32)
        out = features.normalize01(arr)
        finite = out[np.isfinite(out)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0


class TestLocalVariance:
    def test_flat_region_has_zero_texture(self):
        """The defining property: uniform ground is not textured."""
        out = features.local_variance(np.full((64, 64), 0.5, dtype=np.float32))
        assert np.all(out == 0.0)

    def test_noisy_region_is_more_textured_than_flat_region(self):
        rng = np.random.default_rng(0)
        arr = np.full((64, 128), 0.5, dtype=np.float32)
        arr[:, 64:] += rng.normal(0, 0.2, (64, 64)).astype(np.float32)
        out = features.local_variance(arr, window=7)
        assert out[16:48, 80:112].mean() > out[16:48, 16:48].mean()

    def test_output_is_bounded(self):
        rng = np.random.default_rng(1)
        out = features.local_variance(rng.normal(0, 100, (64, 64)).astype(np.float32))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_handles_nan_without_propagating(self):
        arr = np.full((32, 32), 0.5, dtype=np.float32)
        arr[10, 10] = np.nan
        out = features.local_variance(arr)
        assert np.isfinite(out).all(), "NaN must be filled, not spread across the kernel"

    def test_is_deterministic(self):
        rng = np.random.default_rng(2)
        arr = rng.normal(0.5, 0.1, (48, 48)).astype(np.float32)
        np.testing.assert_array_equal(
            features.local_variance(arr), features.local_variance(arr)
        )


class TestEdgeDensity:
    def test_flat_region_has_no_edges(self):
        out = features.edge_density(np.full((64, 64), 0.5, dtype=np.float32))
        assert out.max() == pytest.approx(0.0, abs=1e-6)

    def test_checkerboard_is_more_edge_dense_than_smooth_gradient(self):
        """A structured pattern must outscore a smooth ramp; this is the built-up cue."""
        checker = np.indices((64, 64)).sum(axis=0) % 2
        checker = (checker * 255).astype(np.float32)
        gradient = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (64, 1))
        assert features.edge_density(checker).mean() > features.edge_density(gradient).mean()

    def test_output_is_bounded(self):
        rng = np.random.default_rng(3)
        out = features.edge_density(rng.normal(0, 50, (64, 64)).astype(np.float32))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_handles_nan_without_propagating(self):
        arr = np.full((32, 32), 0.5, dtype=np.float32)
        arr[5, 5] = np.nan
        assert np.isfinite(features.edge_density(arr)).all()

    def test_is_deterministic(self):
        rng = np.random.default_rng(4)
        arr = rng.normal(0.5, 0.2, (48, 48)).astype(np.float32)
        np.testing.assert_array_equal(features.edge_density(arr), features.edge_density(arr))


class TestLinearity:
    def test_responds_to_a_straight_line(self):
        arr = np.zeros((64, 64), dtype=np.float32)
        arr[32, :] = 1.0
        out = features.linearity(arr, min_length=15)
        assert out[32, 20:44].mean() > out[10, 20:44].mean()

    def test_output_is_bounded(self):
        rng = np.random.default_rng(5)
        out = features.linearity(rng.normal(0.5, 0.2, (64, 64)).astype(np.float32), min_length=15)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_line_kernel_is_square_and_odd_sized(self):
        kernel = features._line_kernel(20, 45)
        assert kernel.shape[0] == kernel.shape[1]
        assert kernel.shape[0] % 2 == 1

    @pytest.mark.parametrize("angle", [0, 30, 60, 90, 120, 150])
    def test_line_kernel_is_non_empty_at_every_orientation(self, angle):
        assert features._line_kernel(21, angle).sum() > 0


class TestCleanMask:
    def test_removes_isolated_speckle(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10, 10] = True  # single-pixel noise
        assert not features.clean_mask(mask, open_radius=1).any()

    def test_preserves_a_solid_block(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:40, 20:40] = True
        out = features.clean_mask(mask, open_radius=1, close_radius=1)
        # Opening then closing a large square is near-identity on its interior.
        assert out[25:35, 25:35].all()
        assert out.sum() == pytest.approx(mask.sum(), rel=0.15)

    def test_closing_fills_small_interior_holes(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:40, 20:40] = True
        mask[29:31, 29:31] = False
        out = features.clean_mask(mask, open_radius=0, close_radius=2)
        assert out[29:31, 29:31].all()

    def test_min_area_drops_small_components(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[5:8, 5:8] = True     # 9 px
        mask[20:40, 20:40] = True  # 400 px
        out = features.clean_mask(mask, open_radius=0, close_radius=0, min_area=100)
        assert not out[5:8, 5:8].any()
        assert out[20:40, 20:40].all()

    def test_empty_mask_stays_empty(self):
        assert not features.clean_mask(np.zeros((32, 32), dtype=bool)).any()

    def test_returns_boolean_dtype(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True
        assert features.clean_mask(mask).dtype == bool


class TestRemoveSmallComponents:
    def test_keeps_only_components_meeting_the_threshold(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[2:4, 2:4] = True       # 4 px
        mask[10:20, 10:20] = True   # 100 px
        out = features.remove_small_components(mask, min_area=50)
        assert out.sum() == 100

    def test_threshold_is_inclusive(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[5:10, 5:10] = True  # exactly 25 px
        assert features.remove_small_components(mask, min_area=25).sum() == 25

    def test_diagonal_pixels_count_as_one_component(self):
        """8-connectivity: a diagonal chain is one object, not several."""
        mask = np.zeros((32, 32), dtype=bool)
        for i in range(10):
            mask[i, i] = True
        assert features.remove_small_components(mask, min_area=10).sum() == 10

    def test_empty_mask_returns_empty(self):
        assert not features.remove_small_components(np.zeros((16, 16), dtype=bool), 5).any()


class TestShapeDescriptors:
    def test_disc_is_more_compact_than_a_thin_bar(self):
        """Isoperimetric compactness peaks for a circle; this is the water-vs-river cue."""
        disc = np.zeros((128, 128), dtype=bool)
        yy, xx = np.ogrid[:128, :128]
        disc[(yy - 64) ** 2 + (xx - 64) ** 2 <= 30**2] = True
        bar = np.zeros((128, 128), dtype=bool)
        bar[62:66, 10:118] = True
        assert features.compactness(disc) > features.compactness(bar)

    def test_disc_compactness_approaches_one(self):
        disc = np.zeros((256, 256), dtype=bool)
        yy, xx = np.ogrid[:256, :256]
        disc[(yy - 128) ** 2 + (xx - 128) ** 2 <= 80**2] = True
        assert features.compactness(disc) > 0.85

    def test_rectangle_rectangularity_approaches_one(self):
        mask = np.zeros((128, 128), dtype=bool)
        mask[20:100, 30:90] = True
        assert features.rectangularity(mask) > 0.95

    def test_disc_is_less_rectangular_than_a_rectangle(self):
        rect = np.zeros((128, 128), dtype=bool)
        rect[20:100, 30:90] = True
        disc = np.zeros((128, 128), dtype=bool)
        yy, xx = np.ogrid[:128, :128]
        disc[(yy - 64) ** 2 + (xx - 64) ** 2 <= 40**2] = True
        assert features.rectangularity(disc) < features.rectangularity(rect)

    def test_descriptors_return_zero_for_empty_mask(self):
        empty = np.zeros((32, 32), dtype=bool)
        assert features.compactness(empty) == 0.0
        assert features.rectangularity(empty) == 0.0

    def test_descriptors_ignore_sub_pixel_noise(self):
        """Components too small to have a meaningful shape must not contribute."""
        mask = np.zeros((32, 32), dtype=bool)
        mask[5, 5] = True
        assert features.compactness(mask) == 0.0
        assert features.rectangularity(mask) == 0.0

    def test_descriptors_are_bounded(self):
        rng = np.random.default_rng(6)
        mask = rng.random((64, 64)) > 0.5
        assert 0.0 <= features.compactness(mask) <= 1.0
        assert 0.0 <= features.rectangularity(mask) <= 1.0
