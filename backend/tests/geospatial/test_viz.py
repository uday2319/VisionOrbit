"""Tests for raster/overlay rendering.

Rendering is where a plausible-looking lie is easiest to tell, so these tests check that the
pixels shown are derived from the array given: a heatmap colour tracks the value, an overlay
only tints inside the mask, invalid data renders as invalid rather than as a colour.
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from app.core.types import BandRole
from app.geospatial import viz
from app.geospatial.raster import load_raster
from app.services.landcover import CODE_TO_NAME


@pytest.fixture
def rgb_raster(six_band_scene, make_raster, known_transform):
    return load_raster(make_raster(
        "s.tif", six_band_scene, transform=known_transform,
        descriptions=("blue", "green", "red", "nir", "swir1", "swir2"),
    ))


@pytest.fixture
def gray_raster(make_raster, known_transform):
    arr = np.linspace(0, 10000, 64 * 64, dtype=np.uint16).reshape(1, 64, 64)
    return load_raster(make_raster("g.tif", arr, transform=known_transform,
                                   descriptions=("vv",)))


class TestPercentileStretch:
    def test_maps_to_full_uint8_range(self):
        arr = np.linspace(0.0, 1.0, 10000, dtype=np.float32)
        out = viz._percentile_stretch(arr)
        assert out.dtype == np.uint8
        assert out.min() == 0 and out.max() == 255

    def test_is_monotonic(self):
        arr = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        out = viz._percentile_stretch(arr, 0.0, 100.0)
        assert list(np.argsort(out)) == list(np.argsort(arr))

    def test_outliers_do_not_flatten_the_image(self):
        """A single hot pixel must not push everything else to black."""
        arr = np.full(1000, 0.5, dtype=np.float32)
        arr[:500] = 0.2
        arr[999] = 1e6
        out = viz._percentile_stretch(arr)
        assert out[:500].mean() < out[500:999].mean()

    def test_all_nan_renders_black_rather_than_raising(self):
        out = viz._percentile_stretch(np.full((8, 8), np.nan, dtype=np.float32))
        assert out.shape == (8, 8) and np.all(out == 0)

    def test_constant_input_does_not_divide_by_zero(self):
        out = viz._percentile_stretch(np.full((8, 8), 0.5, dtype=np.float32))
        assert np.isfinite(out).all()

    def test_nan_pixels_render_as_zero(self):
        arr = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        arr[0, 0] = np.nan
        assert viz._percentile_stretch(arr)[0, 0] == 0


class TestRenderRgb:
    def test_shape_and_dtype(self, rgb_raster):
        out = viz.render_rgb(rgb_raster)
        assert out.shape == (*rgb_raster.shape, 3)
        assert out.dtype == np.uint8

    def test_uses_resolved_band_roles_not_file_order(self, rgb_raster):
        """The scene is stored blue-green-red, so a naive ``data[:3]`` would swap red and
        blue. Each output channel must be exactly the stretch of its *role's* band."""
        out = viz.render_rgb(rgb_raster)
        for i, role in enumerate((BandRole.RED, BandRole.GREEN, BandRole.BLUE)):
            np.testing.assert_array_equal(
                out[..., i], viz._percentile_stretch(rgb_raster.band(role))
            )

    def test_the_naive_file_order_would_have_been_wrong(self, rgb_raster):
        """Gives the test above its teeth: file order and role order genuinely differ here."""
        out = viz.render_rgb(rgb_raster)
        assert not np.array_equal(out[..., 0], viz._percentile_stretch(rgb_raster.data[0]))

    def test_water_and_vegetation_render_differently(self, rgb_raster, water_square_bounds):
        r0, r1, c0, c1 = water_square_bounds
        out = viz.render_rgb(rgb_raster)
        assert not np.allclose(out[r0 + 2:r1 - 2, c0 + 2:c1 - 2].mean(axis=(0, 1)),
                               out[0:5, 0:5].mean(axis=(0, 1)), atol=5)

    def test_single_band_renders_as_grayscale(self, gray_raster):
        out = viz.render_rgb(gray_raster)
        assert out.shape == (*gray_raster.shape, 3)
        np.testing.assert_array_equal(out[..., 0], out[..., 1])
        np.testing.assert_array_equal(out[..., 1], out[..., 2])

    def test_is_deterministic(self, rgb_raster):
        np.testing.assert_array_equal(viz.render_rgb(rgb_raster), viz.render_rgb(rgb_raster))


class TestRenderIndexHeatmap:
    def test_shape_and_dtype(self):
        out = viz.render_index_heatmap(np.zeros((16, 16), dtype=np.float32))
        assert out.shape == (16, 16, 3) and out.dtype == np.uint8

    def test_colour_tracks_the_value(self):
        """The core honesty property: equal values render identically, different ones don't."""
        low = viz.render_index_heatmap(np.full((4, 4), -0.8, dtype=np.float32))
        low2 = viz.render_index_heatmap(np.full((4, 4), -0.8, dtype=np.float32))
        high = viz.render_index_heatmap(np.full((4, 4), 0.8, dtype=np.float32))
        np.testing.assert_array_equal(low, low2)
        assert not np.array_equal(low, high)

    def test_a_gradient_renders_as_a_gradient(self):
        arr = np.tile(np.linspace(-1, 1, 64, dtype=np.float32), (8, 1))
        out = viz.render_index_heatmap(arr)
        assert len({tuple(px) for px in out[0]}) > 32, "distinct values must get distinct colours"

    def test_invalid_pixels_render_black_not_coloured(self):
        """A NaN must never be shown as if it were a measurement."""
        arr = np.zeros((8, 8), dtype=np.float32)
        arr[3, 3] = np.nan
        out = viz.render_index_heatmap(arr)
        assert tuple(out[3, 3]) == (0, 0, 0)
        assert tuple(out[0, 0]) != (0, 0, 0)

    def test_values_outside_the_range_are_clamped_not_wrapped(self):
        at_max = viz.render_index_heatmap(np.full((4, 4), 1.0, dtype=np.float32))
        beyond = viz.render_index_heatmap(np.full((4, 4), 5.0, dtype=np.float32))
        np.testing.assert_array_equal(at_max, beyond)

    def test_custom_range_rescales(self):
        arr = np.full((4, 4), 0.5, dtype=np.float32)
        assert not np.array_equal(
            viz.render_index_heatmap(arr, vmin=-1.0, vmax=1.0),
            viz.render_index_heatmap(arr, vmin=0.0, vmax=1.0),
        )


class TestOverlayMask:
    @pytest.fixture
    def base(self):
        return np.full((32, 32, 3), 100, dtype=np.uint8)

    def test_pixels_outside_the_mask_are_untouched(self, base):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:16, 8:16] = True
        out = viz.overlay_mask(base, mask, (255, 0, 0), outline=False)
        np.testing.assert_array_equal(out[24:, 24:], base[24:, 24:])

    def test_pixels_inside_the_mask_are_tinted(self, base):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:16, 8:16] = True
        out = viz.overlay_mask(base, mask, (255, 0, 0), outline=False)
        assert out[12, 12, 0] > base[12, 12, 0]

    def test_alpha_controls_blend_strength(self, base):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:16, 8:16] = True
        light = viz.overlay_mask(base, mask, (255, 0, 0), alpha=0.2, outline=False)
        heavy = viz.overlay_mask(base, mask, (255, 0, 0), alpha=0.9, outline=False)
        assert heavy[12, 12, 0] > light[12, 12, 0]

    def test_does_not_mutate_the_input(self, base):
        original = base.copy()
        mask = np.ones((32, 32), dtype=bool)
        viz.overlay_mask(base, mask, (255, 0, 0))
        np.testing.assert_array_equal(base, original)

    def test_empty_mask_is_a_no_op(self, base):
        out = viz.overlay_mask(base, np.zeros((32, 32), dtype=bool), (255, 0, 0))
        np.testing.assert_array_equal(out, base)

    def test_outline_draws_on_the_boundary(self, base):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        outlined = viz.overlay_mask(base, mask, (255, 0, 0), alpha=0.0, outline=True)
        assert outlined[8, 12, 0] > base[8, 12, 0], "boundary is drawn"

    def test_output_stays_in_uint8_range(self, base):
        out = viz.overlay_mask(np.full((8, 8, 3), 250, dtype=np.uint8),
                               np.ones((8, 8), dtype=bool), (255, 255, 255), alpha=1.0)
        assert out.dtype == np.uint8 and out.max() <= 255


class TestDrawBboxes:
    @pytest.fixture
    def base(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def test_draws_a_box(self, base):
        out = viz.draw_bboxes(base, [(10, 10, 40, 40)], (0, 255, 0))
        assert out[10, 25, 1] > 0, "top edge drawn"
        assert out[50, 50, 1] == 0, "outside the box untouched"

    def test_does_not_mutate_the_input(self, base):
        original = base.copy()
        viz.draw_bboxes(base, [(5, 5, 20, 20)], (255, 0, 0))
        np.testing.assert_array_equal(base, original)

    def test_empty_list_is_a_no_op(self, base):
        np.testing.assert_array_equal(viz.draw_bboxes(base, [], (255, 0, 0)), base)

    def test_draws_every_box(self, base):
        out = viz.draw_bboxes(base, [(2, 2, 12, 12), (40, 40, 60, 60)], (255, 0, 0))
        assert out[2, 7, 0] > 0 and out[40, 50, 0] > 0

    def test_labels_are_rendered(self, base):
        plain = viz.draw_bboxes(base, [(10, 20, 40, 50)], (255, 0, 0))
        labelled = viz.draw_bboxes(base, [(10, 20, 40, 50)], (255, 0, 0), labels=["water"])
        assert labelled.sum() > plain.sum()

    def test_label_at_the_top_edge_does_not_go_out_of_bounds(self, base):
        out = viz.draw_bboxes(base, [(0, 0, 20, 20)], (255, 0, 0), labels=["x"])
        assert out.shape == base.shape

    def test_fewer_labels_than_boxes_is_tolerated(self, base):
        out = viz.draw_bboxes(base, [(2, 2, 12, 12), (30, 30, 50, 50)], (255, 0, 0),
                              labels=["only-one"])
        assert out.shape == base.shape


class TestRenderClassMap:
    def test_each_class_gets_its_own_colour(self):
        class_map = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
        out = viz.render_class_map(class_map, CODE_TO_NAME)
        colours = {tuple(out[r, c]) for r in range(2) for c in range(3)}
        assert len(colours) == 5

    def test_colours_come_from_the_shared_palette(self):
        out = viz.render_class_map(np.array([[1]], dtype=np.uint8), CODE_TO_NAME)
        assert tuple(out[0, 0]) == viz.CLASS_COLORS["water"]

    def test_unknown_codes_fall_back_to_the_unclassified_colour(self):
        out = viz.render_class_map(np.array([[99]], dtype=np.uint8), {99: "not_a_real_class"})
        assert tuple(out[0, 0]) == viz.CLASS_COLORS["unclassified"]

    def test_shape_and_dtype(self):
        out = viz.render_class_map(np.zeros((7, 5), dtype=np.uint8), CODE_TO_NAME)
        assert out.shape == (7, 5, 3) and out.dtype == np.uint8

    def test_every_land_cover_class_has_a_colour(self):
        """A missing entry would silently render two classes identically."""
        for name in CODE_TO_NAME.values():
            assert name in viz.CLASS_COLORS, name

    def test_palette_colours_are_distinct(self):
        colours = list(viz.CLASS_COLORS.values())
        assert len(set(colours)) == len(colours)


class TestSavePng:
    def test_writes_a_readable_png(self, tmp_path):
        import cv2

        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[..., 0] = 200  # red
        out = viz.save_png(tmp_path / "x.png", rgb)
        assert out.exists()
        read = cv2.imread(str(out))  # BGR
        assert read is not None
        assert read[8, 8, 2] == 200, "red must survive the RGB->BGR conversion"

    def test_creates_missing_parent_directories(self, tmp_path):
        out = viz.save_png(tmp_path / "a" / "b" / "c.png", np.zeros((4, 4, 3), dtype=np.uint8))
        assert out.exists()

    def test_writes_grayscale_arrays(self, tmp_path):
        out = viz.save_png(tmp_path / "g.png", np.full((8, 8), 128, dtype=np.uint8))
        assert out.exists() and out.stat().st_size > 0

    def test_accepts_a_string_path(self, tmp_path):
        out = viz.save_png(str(tmp_path / "s.png"), np.zeros((4, 4, 3), dtype=np.uint8))
        assert out.exists()


class TestMakeLegend:
    def test_returns_hex_colours_matching_the_palette(self):
        legend = viz.make_legend([("Water", "water")])
        r, g, b = viz.CLASS_COLORS["water"]
        assert legend[0] == {"label": "Water", "color": f"#{r:02x}{g:02x}{b:02x}", "key": "water"}

    def test_hex_is_always_seven_characters(self):
        for entry in viz.make_legend([(k.title(), k) for k in viz.CLASS_COLORS]):
            assert len(entry["color"]) == 7 and entry["color"].startswith("#")

    def test_unknown_key_falls_back_without_raising(self):
        entry = viz.make_legend([("Mystery", "nope")])[0]
        r, g, b = viz.CLASS_COLORS["unclassified"]
        assert entry["color"] == f"#{r:02x}{g:02x}{b:02x}"

    def test_preserves_order(self):
        keys = ["water", "vegetation", "built_up"]
        assert [e["key"] for e in viz.make_legend([(k, k) for k in keys])] == keys

    def test_empty_input_yields_empty_legend(self):
        assert viz.make_legend([]) == []


class TestThumbnail:
    def test_downscales_a_large_image(self):
        out = viz.thumbnail(np.zeros((1024, 512, 3), dtype=np.uint8), max_edge=256)
        assert max(out.shape[:2]) == 256

    def test_preserves_aspect_ratio(self):
        out = viz.thumbnail(np.zeros((800, 400, 3), dtype=np.uint8), max_edge=200)
        assert out.shape[0] / out.shape[1] == pytest.approx(2.0, abs=0.05)

    def test_small_image_is_returned_unchanged(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        assert viz.thumbnail(arr, max_edge=256) is arr

    def test_image_exactly_at_the_limit_is_unchanged(self):
        arr = np.zeros((256, 128, 3), dtype=np.uint8)
        assert viz.thumbnail(arr, max_edge=256) is arr

    def test_wide_image_is_bounded_on_its_long_edge(self):
        out = viz.thumbnail(np.zeros((100, 900, 3), dtype=np.uint8), max_edge=300)
        assert out.shape[1] == 300


class TestDataUri:
    def test_produces_a_decodable_png_uri(self):
        uri = viz.to_data_uri(np.zeros((8, 8, 3), dtype=np.uint8))
        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"

    def test_round_trips_pixel_content(self):
        import cv2

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[..., 0] = 123
        raw = base64.b64decode(viz.to_data_uri(rgb).split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert decoded[4, 4, 2] == 123

    def test_jpeg_format_is_labelled_correctly(self):
        uri = viz.to_data_uri(np.zeros((8, 8, 3), dtype=np.uint8), fmt=".jpg")
        assert uri.startswith("data:image/jpeg;base64,")

    def test_accepts_grayscale(self):
        assert viz.to_data_uri(np.zeros((8, 8), dtype=np.uint8)).startswith("data:image/png")


class TestSummaryDict:
    def test_drops_none_values(self):
        assert viz.summary_dict(a=1, b=None, c="x") == {"a": 1, "c": "x"}

    def test_keeps_falsy_but_present_values(self):
        """0 and "" are measurements; None is an absence. They must not be conflated."""
        assert viz.summary_dict(zero=0, empty="", false=False) == {
            "zero": 0, "empty": "", "false": False
        }

    def test_empty_input_yields_empty_dict(self):
        assert viz.summary_dict() == {}
