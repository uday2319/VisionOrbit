"""Tests for CRS-aware ground measurement.

Every assertion here is checked against an independently-derived number — geodesic distance,
a unit conversion factor, or plain arithmetic — rather than against what the implementation
happens to return. A pixel area that is wrong by a constant factor is invisible in the output
but corrupts every area and every minimum-size filter downstream.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from pyproj import Geod
from rasterio.transform import Affine

from app.geospatial.measure import (
    PixelArea,
    pixel_area_m2,
    pixels_for_ground_area,
)
from app.geospatial.raster import read_metadata

# A degree step chosen so the north-south extent is very close to 10 m, matching the
# projected fixtures and making the two paths directly comparable.
DEG_10M = 10.0 / 111_320.0


@pytest.fixture
def utm_meta(make_raster, known_transform):
    """512-band-agnostic raster in EPSG:32643 with exactly 10 m pixels."""
    path = make_raster("utm.tif", np.zeros((1, 32, 32), dtype=np.uint16),
                       transform=known_transform, epsg=32643)
    return read_metadata(path)


@pytest.fixture
def geographic_meta(make_raster):
    """Same grid expressed in EPSG:4326, so pixel size is in degrees."""
    path = make_raster("geo.tif", np.zeros((1, 32, 32), dtype=np.uint16),
                       transform=Affine(DEG_10M, 0, 77.5, 0, -DEG_10M, 18.9), epsg=4326)
    return read_metadata(path)


class TestProjectedCrs:
    def test_area_is_the_product_of_pixel_dimensions(self, utm_meta):
        assert pixel_area_m2(utm_meta).area_m2 == pytest.approx(100.0)

    def test_no_caveat_for_a_metric_projected_crs(self, utm_meta):
        """The value is exact here, so attaching a caveat would be noise."""
        assert pixel_area_m2(utm_meta).caveat is None

    def test_scales_with_pixel_size(self, make_raster):
        path = make_raster("coarse.tif", np.zeros((1, 8, 8), dtype=np.uint16),
                           transform=Affine(30.0, 0, 700000, 0, -30.0, 2100000), epsg=32643)
        assert pixel_area_m2(read_metadata(path)).area_m2 == pytest.approx(900.0)

    def test_non_square_pixels_multiply_both_axes(self, make_raster):
        path = make_raster("rect.tif", np.zeros((1, 8, 8), dtype=np.uint16),
                           transform=Affine(20.0, 0, 700000, 0, -5.0, 2100000), epsg=32643)
        assert pixel_area_m2(read_metadata(path)).area_m2 == pytest.approx(100.0)

    def test_survey_feet_are_converted_to_metres(self, make_raster):
        """EPSG:2264 is in US survey feet. A "10" pixel covers 9.29 m², not 100 m².

        Assuming metres here would overstate every reported area by a factor of 10.76 while
        looking entirely reasonable.
        """
        path = make_raster("feet.tif", np.zeros((1, 8, 8), dtype=np.uint16),
                           transform=Affine(10.0, 0, 2_000_000, 0, -10.0, 700_000), epsg=2264)
        ft_to_m = 0.3048006096012192
        assert pixel_area_m2(read_metadata(path)).area_m2 == pytest.approx(
            100.0 * ft_to_m * ft_to_m, rel=1e-6
        )

    def test_a_metric_crs_has_a_unit_factor_of_one(self, utm_meta):
        """Guards the test above: the conversion must not perturb an already-metric CRS."""
        assert pixel_area_m2(utm_meta).area_m2 == pytest.approx(100.0, rel=1e-12)


class TestGeographicCrs:
    def test_returns_an_area_rather_than_degrees_squared(self, geographic_meta):
        """Regression: reading ``pixel_size`` as metres gives ~8e-9, which turned a 2500 m²
        minimum patch into 3e11 pixels and silently erased every detection."""
        area = pixel_area_m2(geographic_meta).area_m2
        assert area is not None
        assert 1.0 < area < 1000.0, f"got {area}, which is degrees-squared, not m²"

    def test_matches_an_independent_geodesic_computation(self, geographic_meta):
        """Cross-checked against pyproj's own geodesic distances, computed a different way:
        the north-south and east-west edge lengths measured separately, then multiplied."""
        geod = Geod(ellps="WGS84")
        lon0, lat0 = 77.5 + 16 * DEG_10M, 18.9 - 16 * DEG_10M
        _, _, ns = geod.inv(lon0, lat0, lon0, lat0 + DEG_10M)
        _, _, ew = geod.inv(lon0, lat0, lon0 + DEG_10M, lat0)
        assert pixel_area_m2(geographic_meta).area_m2 == pytest.approx(ns * ew, rel=0.01)

    def test_area_shrinks_with_latitude(self, make_raster):
        """A degree of longitude covers less ground further from the equator; if the
        implementation ignored latitude these two would be identical."""
        areas = []
        for lat in (5.0, 60.0):
            path = make_raster(f"lat{lat}.tif", np.zeros((1, 8, 8), dtype=np.uint16),
                               transform=Affine(DEG_10M, 0, 10.0, 0, -DEG_10M, lat), epsg=4326)
            areas.append(pixel_area_m2(read_metadata(path)).area_m2)
        assert areas[0] is not None and areas[1] is not None
        assert areas[1] < areas[0]
        # cos(60°)/cos(5°) ≈ 0.502 — the ratio is predicted, not just ordered.
        assert areas[1] / areas[0] == pytest.approx(
            math.cos(math.radians(60)) / math.cos(math.radians(5)), rel=0.02
        )

    def test_small_extent_gets_no_caveat(self, geographic_meta):
        """A 32-pixel-tall scene has negligible variation, so a caveat would be misleading."""
        assert pixel_area_m2(geographic_meta).caveat is None

    def test_tall_scene_at_high_latitude_is_flagged_as_approximate(self, make_raster):
        """Spanning 20° of latitude at 55°N, a single centre-pixel area is a poor summary and
        the result must say so rather than present it as measured."""
        path = make_raster("tall.tif", np.zeros((1, 200, 8), dtype=np.uint16),
                           transform=Affine(0.1, 0, 10.0, 0, -0.1, 65.0), epsg=4326)
        result = pixel_area_m2(read_metadata(path))
        assert result.area_m2 is not None
        assert result.caveat is not None
        assert "latitude" in result.caveat.lower()

    def test_caveat_is_a_user_facing_sentence(self, make_raster):
        path = make_raster("tall2.tif", np.zeros((1, 200, 8), dtype=np.uint16),
                           transform=Affine(0.1, 0, 10.0, 0, -0.1, 65.0), epsg=4326)
        caveat = pixel_area_m2(read_metadata(path)).caveat
        assert caveat is not None
        assert caveat[0].isupper() and caveat.rstrip().endswith(".")
        for token in ("pyproj", "geod", "None", "traceback"):
            assert token not in caveat


class TestUngeoreferenced:
    def test_identity_transform_yields_no_area(self, make_raster):
        """§8: an ordinary image carries no coordinates, so it has no ground area."""
        path = make_raster("plain.tif", np.zeros((1, 8, 8), dtype=np.uint16),
                           transform=Affine.identity(), epsg=None)
        assert pixel_area_m2(read_metadata(path)) == PixelArea(None, None)

    def test_transform_without_crs_yields_no_area(self, make_raster, known_transform):
        """A geotransform alone does not say what units it is in."""
        path = make_raster("nocrs.tif", np.zeros((1, 8, 8), dtype=np.uint16),
                           transform=known_transform, epsg=None)
        assert pixel_area_m2(read_metadata(path)).area_m2 is None

    def test_crs_with_identity_transform_yields_no_area(self, make_raster):
        """An identity transform is the sentinel for "no real geotransform"."""
        path = make_raster("idcrs.tif", np.zeros((1, 8, 8), dtype=np.uint16),
                           transform=Affine.identity(), epsg=32643)
        assert pixel_area_m2(read_metadata(path)).area_m2 is None

    def test_degenerate_pixel_size_yields_no_area(self, utm_meta):
        """A corrupt geotransform with zero scale would make every area zero.

        Reporting "0 m² of water" reads as a measurement; reporting nothing does not.
        """
        broken = replace(utm_meta, transform=(0.0, 0.0, 700000.0, 0.0, 0.0, 2100000.0))
        assert broken.is_georeferenced, "must reach the pixel-size check, not exit before it"
        assert pixel_area_m2(broken).area_m2 is None


class TestPixelsForGroundArea:
    def test_converts_area_to_pixel_count(self, utm_meta):
        """2500 m² at 10 m pixels is exactly 25 pixels."""
        assert pixels_for_ground_area(utm_meta, 2500.0, default_pixels=99) == 25

    def test_scales_inversely_with_resolution(self, make_raster):
        """The same ground area is fewer pixels on a coarser grid — the whole point of
        expressing minimum sizes physically rather than as a pixel count."""
        path = make_raster("c.tif", np.zeros((1, 64, 64), dtype=np.uint16),
                           transform=Affine(50.0, 0, 700000, 0, -50.0, 2100000), epsg=32643)
        assert pixels_for_ground_area(read_metadata(path), 2500.0, default_pixels=99) == 1

    def test_falls_back_when_pixel_area_is_unknowable(self, make_raster):
        """An ungeoreferenced image still needs denoising, so the default applies."""
        path = make_raster("plain.tif", np.zeros((1, 8, 8), dtype=np.uint16),
                           transform=Affine.identity(), epsg=None)
        assert pixels_for_ground_area(read_metadata(path), 2500.0, default_pixels=25) == 25

    def test_never_returns_zero(self, make_raster):
        """A floor of 0 would disable the filter entirely rather than relax it."""
        path = make_raster("huge_px.tif", np.zeros((1, 8, 8), dtype=np.uint16),
                           transform=Affine(1000.0, 0, 700000, 0, -1000.0, 2100000), epsg=32643)
        assert pixels_for_ground_area(read_metadata(path), 1.0, default_pixels=25) >= 1

    def test_caps_at_a_plausible_share_of_the_frame(self, make_raster):
        """A minimum-patch size larger than the image can only erase every detection.

        Sub-metre pixels make 2500 m² exceed a small frame, so the value is capped instead of
        emptying the result — the failure mode this whole module exists to prevent.
        """
        path = make_raster("drone.tif", np.zeros((1, 32, 32), dtype=np.uint16),
                           transform=Affine(0.05, 0, 700000, 0, -0.05, 2100000), epsg=32643)
        meta = read_metadata(path)
        got = pixels_for_ground_area(meta, 2500.0, default_pixels=25)
        uncapped = 2500.0 / (0.05 * 0.05)
        assert uncapped > meta.width * meta.height, "fixture must actually exceed the frame"
        assert got == int(meta.width * meta.height * 0.25)

    def test_cap_is_configurable(self, make_raster):
        path = make_raster("drone2.tif", np.zeros((1, 32, 32), dtype=np.uint16),
                           transform=Affine(0.05, 0, 700000, 0, -0.05, 2100000), epsg=32643)
        meta = read_metadata(path)
        assert pixels_for_ground_area(
            meta, 2500.0, default_pixels=25, max_frame_fraction=0.5
        ) == int(meta.width * meta.height * 0.5)

    def test_geographic_crs_does_not_erase_everything(self, geographic_meta):
        """Regression for the original bug: this returned 3·10¹¹ before."""
        got = pixels_for_ground_area(geographic_meta, 2500.0, default_pixels=25)
        assert 1 <= got <= geographic_meta.width * geographic_meta.height
        # ~94 m² per pixel at this latitude, so ~27 pixels.
        assert got == pytest.approx(27, abs=2)


class TestDemoDataConsistency:
    def test_projected_demo_scene_matches_the_manifest(self, demo_root, demo_manifest):
        """The manifest records the intended pixel area; measuring it independently confirms
        the fixtures and the measurement code agree."""
        meta = read_metadata(demo_root / "optical" / "scene_optical.tif")
        assert pixel_area_m2(meta).area_m2 == pytest.approx(
            demo_manifest["grid"]["pixel_area_m2"]
        )

    def test_reprojected_demo_scene_measures_close_to_the_original(self, demo_root):
        """``different_crs.tif`` is the same ground in EPSG:4326. Its pixel area must land
        near the 100 m² of the projected version, not orders of magnitude away."""
        geo = pixel_area_m2(read_metadata(demo_root / "edge_cases" / "different_crs.tif"))
        assert geo.area_m2 is not None
        assert 80.0 < geo.area_m2 < 110.0
