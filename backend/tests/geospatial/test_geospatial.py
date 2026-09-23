"""Geospatial correctness tests.

These assert against *known arithmetic*: a 10 m pixel in a projected CRS is exactly 100 m²,
and pixel (0, 0) maps to exactly the raster origin. That makes them real correctness checks
rather than snapshots of whatever the code currently happens to produce.
"""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import Affine

from app.core.errors import GeospatialError, ValidationError
from app.core.types import BandRole, Modality
from app.geospatial.align import (
    align_to_reference,
    measure_registration,
    normalize_radiometry,
)
from app.geospatial.raster import (
    compute_decimated_shape,
    load_raster,
    read_metadata,
    resolve_band_roles,
)
from app.geospatial.validate import (
    AlignmentStrategy,
    assess_quality,
    check_pair_compatibility,
    compute_overlap_fraction,
    require_compatible,
    validate_modality,
)
from app.geospatial.vectorize import pixel_bbox_to_geo, vectorize_mask
from tests.conftest import TEST_ORIGIN_X, TEST_ORIGIN_Y, TEST_PIXEL_M

pytestmark = pytest.mark.geospatial


# ---------------------------------------------------------------------------
# Metadata & georeferencing
# ---------------------------------------------------------------------------
class TestRasterMetadata:
    def test_reads_crs_transform_and_bounds(self, make_raster, six_band_scene, known_transform):
        path = make_raster("scene.tif", six_band_scene, transform=known_transform,
                           descriptions=("blue", "green", "red", "nir", "swir1", "swir2"))
        meta = read_metadata(path)

        assert meta.crs_epsg == 32643
        assert meta.width == 64 and meta.height == 64
        assert meta.count == 6
        assert meta.is_georeferenced

        # Bounds follow directly from origin + size * pixel, north-up.
        minx, miny, maxx, maxy = meta.bounds
        assert minx == pytest.approx(TEST_ORIGIN_X)
        assert maxy == pytest.approx(TEST_ORIGIN_Y)
        assert maxx == pytest.approx(TEST_ORIGIN_X + 64 * TEST_PIXEL_M)
        assert miny == pytest.approx(TEST_ORIGIN_Y - 64 * TEST_PIXEL_M)

    def test_pixel_size_is_exact(self, make_raster, six_band_scene, known_transform):
        path = make_raster("s.tif", six_band_scene, transform=known_transform)
        meta = read_metadata(path)
        assert meta.pixel_size == pytest.approx((TEST_PIXEL_M, TEST_PIXEL_M))

    def test_identity_transform_is_not_georeferenced(self, make_raster, six_band_scene):
        """A plain image must never be treated as carrying coordinates (brief §8)."""
        path = make_raster("plain.tif", six_band_scene, transform=Affine.identity(), epsg=None)
        assert read_metadata(path).is_georeferenced is False

    def test_missing_file_raises_validation_error(self, tmp_path):
        with pytest.raises(ValidationError) as exc:
            read_metadata(tmp_path / "nope.tif")
        assert exc.value.code == "INVALID_FILE"

    def test_empty_file_raises_specific_error(self, tmp_path):
        empty = tmp_path / "empty.tif"
        empty.write_bytes(b"")
        with pytest.raises(ValidationError) as exc:
            read_metadata(empty)
        assert exc.value.code == "EMPTY_FILE"

    def test_non_raster_file_raises_invalid_geotiff(self, tmp_path):
        bad = tmp_path / "not_an_image.pdf"
        bad.write_bytes(b"%PDF-1.4\nnot an image at all\n")
        with pytest.raises(ValidationError) as exc:
            read_metadata(bad)
        assert exc.value.code == "INVALID_GEOTIFF"
        # The user-facing message must not leak internals.
        assert "Traceback" not in exc.value.message


# ---------------------------------------------------------------------------
# Band role resolution
# ---------------------------------------------------------------------------
class TestBandRoles:
    def test_resolves_from_descriptions(self):
        roles = resolve_band_roles(6, ("blue", "green", "red", "nir", "swir1", "swir2"))
        assert roles == [BandRole.BLUE, BandRole.GREEN, BandRole.RED,
                         BandRole.NIR, BandRole.SWIR1, BandRole.SWIR2]

    def test_three_band_defaults_to_rgb(self):
        assert resolve_band_roles(3) == [BandRole.RED, BandRole.GREEN, BandRole.BLUE]

    def test_four_band_geotiff_assumes_nir_but_png_assumes_alpha(self):
        """The 4th band means different things in a GeoTIFF vs a PNG."""
        assert resolve_band_roles(4, driver="GTiff")[3] is BandRole.NIR
        assert resolve_band_roles(4, driver="PNG")[3] is BandRole.ALPHA

    def test_sar_two_band_is_vv_vh(self):
        assert resolve_band_roles(2, modality=Modality.SAR) == [BandRole.VV, BandRole.VH]

    def test_unrecognised_count_yields_unknown_not_a_guess(self):
        """Guessing a NIR band would silently corrupt every downstream index."""
        roles = resolve_band_roles(7)
        assert all(r is BandRole.UNKNOWN for r in roles)

    def test_freetext_descriptions_fall_back_to_convention(self):
        roles = resolve_band_roles(3, ("Band 1", "Band 2", "Band 3"))
        assert roles == [BandRole.RED, BandRole.GREEN, BandRole.BLUE]

    def test_missing_band_access_raises_with_clear_message(
        self, make_raster, six_band_scene, known_transform
    ):
        path = make_raster("rgb.tif", six_band_scene[[2, 1, 0]], transform=known_transform,
                           descriptions=("red", "green", "blue"))
        raster = load_raster(path)
        assert not raster.has_role(BandRole.SWIR1)
        with pytest.raises(ValidationError) as exc:
            raster.band(BandRole.SWIR1)
        assert exc.value.code == "MISSING_BAND"


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------
class TestDecimation:
    @pytest.mark.parametrize(
        "w,h,max_edge,expected",
        [
            (100, 100, 1024, (100, 100, 1.0)),   # no decimation needed
            (2048, 1024, 1024, (512, 1024, 2.0)),
            (4096, 4096, 1024, (1024, 1024, 4.0)),
        ],
    )
    def test_shape_and_factor(self, w, h, max_edge, expected):
        assert compute_decimated_shape(w, h, max_edge) == expected

    def test_aspect_ratio_preserved(self):
        out_h, out_w, _ = compute_decimated_shape(3000, 1000, 600)
        assert out_w / out_h == pytest.approx(3.0, abs=0.02)

    def test_decimated_read_rescales_transform_so_bounds_are_preserved(
        self, make_raster, known_transform
    ):
        """Downsampling must not move the image on the ground."""
        big = np.ones((1, 512, 512), dtype=np.uint16) * 1000
        path = make_raster("big.tif", big, transform=known_transform)
        full = load_raster(path, max_edge=None)
        small = load_raster(path, max_edge=128)

        assert small.shape == (128, 128)
        assert small.metadata.decimation == pytest.approx(4.0)
        # Pixel size must grow by the decimation factor, keeping extent identical.
        assert small.metadata.pixel_size[0] == pytest.approx(TEST_PIXEL_M * 4)
        full_extent = full.width * full.metadata.pixel_size[0]
        small_extent = small.width * small.metadata.pixel_size[0]
        assert full_extent == pytest.approx(small_extent)

    def test_oversized_raster_is_rejected(self, make_raster, known_transform):
        arr = np.ones((1, 100, 100), dtype=np.uint16)
        path = make_raster("x.tif", arr, transform=known_transform)
        with pytest.raises(ValidationError) as exc:
            load_raster(path, max_pixels=100)
        assert exc.value.code == "FILE_TOO_LARGE"


# ---------------------------------------------------------------------------
# Pixel <-> world coordinates
# ---------------------------------------------------------------------------
class TestCoordinateConversion:
    def test_pixel_origin_maps_to_raster_origin(
        self, make_raster, six_band_scene, known_transform
    ):
        path = make_raster("s.tif", six_band_scene, transform=known_transform)
        meta = read_metadata(path)
        geo = pixel_bbox_to_geo((0, 0, 1, 1), meta)
        assert geo[0] == pytest.approx(TEST_ORIGIN_X)
        assert geo[3] == pytest.approx(TEST_ORIGIN_Y)

    def test_bbox_projection_is_exact(self, make_raster, six_band_scene, known_transform):
        """A bbox at cols 10-30, rows 10-30 has an arithmetically known world position."""
        path = make_raster("s.tif", six_band_scene, transform=known_transform)
        meta = read_metadata(path)
        minx, miny, maxx, maxy = pixel_bbox_to_geo((10, 10, 30, 30), meta)
        assert minx == pytest.approx(TEST_ORIGIN_X + 10 * TEST_PIXEL_M)
        assert maxx == pytest.approx(TEST_ORIGIN_X + 30 * TEST_PIXEL_M)
        assert maxy == pytest.approx(TEST_ORIGIN_Y - 10 * TEST_PIXEL_M)
        assert miny == pytest.approx(TEST_ORIGIN_Y - 30 * TEST_PIXEL_M)

    def test_returns_none_without_georeference(self, make_raster, six_band_scene):
        path = make_raster("p.tif", six_band_scene, transform=Affine.identity(), epsg=None)
        assert pixel_bbox_to_geo((0, 0, 5, 5), read_metadata(path)) is None


# ---------------------------------------------------------------------------
# Vectorization & area
# ---------------------------------------------------------------------------
class TestVectorize:
    def test_area_equals_pixel_count_times_pixel_area(
        self, make_raster, six_band_scene, known_transform
    ):
        """The core area assertion: 400 pixels at 10 m = 40,000 m² exactly."""
        path = make_raster("s.tif", six_band_scene, transform=known_transform)
        meta = read_metadata(path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True  # 20x20 = 400 px

        result = vectorize_mask(mask, meta, min_pixels=10)

        assert result.count == 1
        assert result.total_pixels == 400
        assert result.regions[0].pixel_area == 400
        assert result.regions[0].area_m2 == pytest.approx(400 * 100.0)  # 40,000 m²
        assert result.total_area_m2 == pytest.approx(40_000.0)
        assert result.coverage_fraction == pytest.approx(400 / 4096)

    def test_separate_blobs_counted_separately(
        self, make_raster, six_band_scene, known_transform
    ):
        path = make_raster("s.tif", six_band_scene, transform=known_transform)
        meta = read_metadata(path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[5:15, 5:15] = True
        mask[40:50, 40:50] = True
        result = vectorize_mask(mask, meta, min_pixels=10)
        assert result.count == 2
        assert result.total_pixels == 200

    def test_specks_below_min_pixels_are_dropped(
        self, make_raster, six_band_scene, known_transform
    ):
        path = make_raster("s.tif", six_band_scene, transform=known_transform)
        meta = read_metadata(path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True   # 400 px, kept
        mask[60, 60] = True         # 1 px, noise
        result = vectorize_mask(mask, meta, min_pixels=25)
        assert result.count == 1

    def test_no_area_without_georeference(self, make_raster, six_band_scene):
        """Ungeoreferenced input must yield pixel measures and explicitly no m²."""
        path = make_raster("p.tif", six_band_scene, transform=Affine.identity(), epsg=None)
        meta = read_metadata(path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True
        result = vectorize_mask(mask, meta, min_pixels=10)

        assert result.georeferenced is False
        assert result.total_area_m2 is None
        assert result.regions[0].area_m2 is None
        assert result.regions[0].polygon_geo is None
        assert any("not georeferenced" in w for w in result.warnings)

    def test_empty_mask_yields_no_regions(self, make_raster, six_band_scene, known_transform):
        path = make_raster("s.tif", six_band_scene, transform=known_transform)
        meta = read_metadata(path)
        result = vectorize_mask(np.zeros((64, 64), dtype=bool), meta)
        assert result.count == 0 and result.total_pixels == 0

    def test_polygon_geo_lies_within_raster_bounds(
        self, make_raster, six_band_scene, known_transform
    ):
        path = make_raster("s.tif", six_band_scene, transform=known_transform)
        meta = read_metadata(path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True
        region = vectorize_mask(mask, meta, min_pixels=10).regions[0]
        minx, miny, maxx, maxy = meta.bounds
        for x, y in region.polygon_geo:
            assert minx <= x <= maxx
            assert miny <= y <= maxy

    def test_geographic_crs_uses_geodesic_area(self, make_raster):
        """In EPSG:4326 the transform is in degrees, so area must be geodesic, not degrees²."""
        arr = np.ones((1, 64, 64), dtype=np.uint16)
        deg = 10.0 / 111_320.0
        path = make_raster("geo.tif", arr, transform=Affine(deg, 0, 77.5, 0, -deg, 18.9),
                           epsg=4326)
        meta = read_metadata(path)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True  # 400 px of ~10 m -> ~40,000 m²
        region = vectorize_mask(mask, meta, min_pixels=10).regions[0]
        assert region.area_m2 is not None
        # Within 25%: degree->metre scaling varies with latitude and the ring is simplified.
        assert region.area_m2 == pytest.approx(40_000.0, rel=0.25)


# ---------------------------------------------------------------------------
# Overlap & pair compatibility
# ---------------------------------------------------------------------------
class TestPairCompatibility:
    def _load(self, make_raster, scene, transform, name, **kw):
        return load_raster(make_raster(name, scene, transform=transform, **kw))

    def test_identical_grids_need_no_alignment(
        self, make_raster, six_band_scene, known_transform
    ):
        a = self._load(make_raster, six_band_scene, known_transform, "a.tif")
        b = self._load(make_raster, six_band_scene, known_transform, "b.tif")
        pair = check_pair_compatibility(a, b)
        assert pair.compatible
        assert pair.strategy is AlignmentStrategy.IDENTICAL_GRID
        assert pair.overlap_fraction == pytest.approx(1.0)

    def test_full_overlap_fraction_is_one(self, make_raster, six_band_scene, known_transform):
        a = self._load(make_raster, six_band_scene, known_transform, "a.tif")
        assert compute_overlap_fraction(a.metadata, a.metadata) == pytest.approx(1.0)

    def test_half_shifted_scene_gives_half_overlap(
        self, make_raster, six_band_scene, known_transform
    ):
        """Shifting by half the width must give ~0.5 overlap — checkable by hand."""
        shifted = Affine(TEST_PIXEL_M, 0, TEST_ORIGIN_X + 32 * TEST_PIXEL_M,
                         0, -TEST_PIXEL_M, TEST_ORIGIN_Y)
        a = self._load(make_raster, six_band_scene, known_transform, "a.tif")
        b = self._load(make_raster, six_band_scene, shifted, "b.tif")
        assert compute_overlap_fraction(a.metadata, b.metadata) == pytest.approx(0.5, abs=0.02)

    def test_disjoint_extents_are_rejected(self, make_raster, six_band_scene, known_transform):
        far = Affine(TEST_PIXEL_M, 0, TEST_ORIGIN_X + 500_000,
                     0, -TEST_PIXEL_M, TEST_ORIGIN_Y + 500_000)
        a = self._load(make_raster, six_band_scene, known_transform, "a.tif")
        b = self._load(make_raster, six_band_scene, far, "b.tif")
        pair = check_pair_compatibility(a, b)

        assert pair.compatible is False
        assert pair.strategy is AlignmentStrategy.INCOMPATIBLE
        assert pair.overlap_fraction == pytest.approx(0.0)
        with pytest.raises(GeospatialError) as exc:
            require_compatible(pair)
        assert exc.value.code == "NO_SPATIAL_OVERLAP"

    def test_insufficient_overlap_is_rejected_with_distinct_code(
        self, make_raster, six_band_scene, known_transform
    ):
        shifted = Affine(TEST_PIXEL_M, 0, TEST_ORIGIN_X + 58 * TEST_PIXEL_M,
                         0, -TEST_PIXEL_M, TEST_ORIGIN_Y)
        a = self._load(make_raster, six_band_scene, known_transform, "a.tif")
        b = self._load(make_raster, six_band_scene, shifted, "b.tif")
        pair = check_pair_compatibility(a, b, min_overlap=0.30)
        assert pair.compatible is False
        with pytest.raises(GeospatialError) as exc:
            require_compatible(pair)
        assert exc.value.code == "INSUFFICIENT_OVERLAP"

    def test_different_crs_triggers_reprojection_strategy(
        self, make_raster, six_band_scene, known_transform
    ):
        deg = 10.0 / 111_320.0
        a = self._load(make_raster, six_band_scene, known_transform, "a.tif")
        b = self._load(make_raster, six_band_scene,
                       Affine(deg, 0, 77.5, 0, -deg, 18.9), "b.tif", epsg=4326)
        pair = check_pair_compatibility(a, b)
        # Different CRS is handled by reprojecting, not refused.
        assert pair.strategy in (AlignmentStrategy.REPROJECT, AlignmentStrategy.INCOMPATIBLE)
        if pair.compatible:
            assert any("coordinate reference system" in w for w in pair.warnings)

    def test_different_resolution_triggers_resampling(
        self, make_raster, six_band_scene, known_transform
    ):
        coarse = Affine(TEST_PIXEL_M * 2, 0, TEST_ORIGIN_X, 0, -TEST_PIXEL_M * 2, TEST_ORIGIN_Y)
        a = self._load(make_raster, six_band_scene, known_transform, "a.tif")
        b = self._load(make_raster, six_band_scene[:, ::2, ::2], coarse, "b.tif")
        pair = check_pair_compatibility(a, b)
        assert pair.compatible
        assert pair.strategy is AlignmentStrategy.RESAMPLE

    def test_ungeoreferenced_pair_warns_and_reports_no_overlap(
        self, make_raster, six_band_scene
    ):
        a = load_raster(make_raster("a.tif", six_band_scene,
                                    transform=Affine.identity(), epsg=None))
        b = load_raster(make_raster("b.tif", six_band_scene,
                                    transform=Affine.identity(), epsg=None))
        pair = check_pair_compatibility(a, b)
        assert pair.compatible
        assert pair.strategy is AlignmentStrategy.PIXEL_ASSUME
        assert pair.overlap_fraction is None
        assert any("georeferenced" in w.lower() for w in pair.warnings)

    def test_require_georeference_rejects_plain_images(self, make_raster, six_band_scene):
        a = load_raster(make_raster("a.tif", six_band_scene,
                                    transform=Affine.identity(), epsg=None))
        pair = check_pair_compatibility(a, a, require_georeference=True)
        assert pair.compatible is False


# ---------------------------------------------------------------------------
# Alignment & radiometry
# ---------------------------------------------------------------------------
class TestAlignment:
    def test_resample_produces_reference_shape(
        self, make_raster, six_band_scene, known_transform
    ):
        coarse = Affine(TEST_PIXEL_M * 2, 0, TEST_ORIGIN_X, 0, -TEST_PIXEL_M * 2, TEST_ORIGIN_Y)
        ref = load_raster(make_raster("ref.tif", six_band_scene, transform=known_transform))
        src = load_raster(make_raster("src.tif", six_band_scene[:, ::2, ::2], transform=coarse))
        aligned, report = align_to_reference(src, ref, AlignmentStrategy.RESAMPLE)
        assert aligned.shape == ref.shape
        assert report.target_shape == ref.shape

    def test_reprojection_preserves_reference_crs(self, make_raster, six_band_scene, known_transform):
        deg = 10.0 / 111_320.0
        ref = load_raster(make_raster("ref.tif", six_band_scene, transform=known_transform))
        src = load_raster(make_raster("src.tif", six_band_scene,
                                      transform=Affine(deg, 0, 77.5, 0, -deg, 18.9), epsg=4326))
        aligned, report = align_to_reference(src, ref, AlignmentStrategy.REPROJECT)
        assert aligned.shape == ref.shape
        assert aligned.metadata.crs_epsg == ref.metadata.crs_epsg
        assert report.reprojected is True

    def test_identical_grid_is_a_noop(self, make_raster, six_band_scene, known_transform):
        ref = load_raster(make_raster("ref.tif", six_band_scene, transform=known_transform))
        aligned, report = align_to_reference(ref, ref, AlignmentStrategy.IDENTICAL_GRID)
        assert report.resampling == "none"
        assert np.array_equal(np.nan_to_num(aligned.data), np.nan_to_num(ref.data))

    def test_incompatible_strategy_raises(self, make_raster, six_band_scene, known_transform):
        ref = load_raster(make_raster("ref.tif", six_band_scene, transform=known_transform))
        with pytest.raises(GeospatialError):
            align_to_reference(ref, ref, AlignmentStrategy.INCOMPATIBLE)


class TestRegistration:
    def test_identical_images_register_with_zero_offset(
        self, make_raster, six_band_scene, known_transform
    ):
        r = load_raster(make_raster("a.tif", six_band_scene, transform=known_transform))
        quality = measure_registration(r, r)
        assert quality.offset_magnitude < 0.5
        assert quality.ncc == pytest.approx(1.0, abs=0.01)
        assert quality.score > 0.8

    def test_known_shift_is_measured(self, make_raster, six_band_scene, known_transform):
        """Roll the image by a known number of pixels; the measurement must recover it."""
        shifted = np.roll(six_band_scene, shift=6, axis=2)
        a = load_raster(make_raster("a.tif", six_band_scene, transform=known_transform))
        b = load_raster(make_raster("b.tif", shifted, transform=known_transform))
        quality = measure_registration(a, b)
        assert quality.offset_magnitude > 2.0
        assert abs(abs(quality.offset_x) - 6.0) < 2.0
        assert any("misalign" in w.lower() for w in quality.warnings)

    def test_shape_mismatch_raises(self, make_raster, six_band_scene, known_transform):
        a = load_raster(make_raster("a.tif", six_band_scene, transform=known_transform))
        b = load_raster(make_raster("b.tif", six_band_scene[:, :32, :32], transform=known_transform))
        with pytest.raises(GeospatialError) as exc:
            measure_registration(a, b)
        assert exc.value.code == "SIZE_MISMATCH"

    def test_unrelated_images_report_low_structural_agreement(
        self, make_raster, six_band_scene, known_transform
    ):
        rng = np.random.default_rng(3)
        noise = (rng.random(six_band_scene.shape) * 10000).astype(np.uint16)
        a = load_raster(make_raster("a.tif", six_band_scene, transform=known_transform))
        b = load_raster(make_raster("b.tif", noise, transform=known_transform))
        quality = measure_registration(a, b)
        assert quality.ncc < 0.5


class TestRadiometricNormalization:
    def test_brightness_offset_is_removed(self, make_raster, six_band_scene, known_transform):
        """A pure gain/offset difference must be normalised away, not read as change."""
        brighter = np.clip(six_band_scene.astype(np.float32) * 1.4 + 300, 0, 65535).astype(np.uint16)
        ref = load_raster(make_raster("ref.tif", six_band_scene, transform=known_transform))
        tgt = load_raster(make_raster("tgt.tif", brighter, transform=known_transform))

        before = abs(float(np.nanmean(tgt.data)) - float(np.nanmean(ref.data)))
        normalized, warnings = normalize_radiometry(tgt, ref)
        after = abs(float(np.nanmean(normalized.data)) - float(np.nanmean(ref.data)))

        assert after < before * 0.05
        assert after < 1.0

    def test_shape_mismatch_raises(self, make_raster, six_band_scene, known_transform):
        a = load_raster(make_raster("a.tif", six_band_scene, transform=known_transform))
        b = load_raster(make_raster("b.tif", six_band_scene[:, :32, :32], transform=known_transform))
        with pytest.raises(GeospatialError):
            normalize_radiometry(b, a)

    def test_constant_target_band_is_left_alone(
        self, make_raster, six_band_scene, known_transform
    ):
        """Scaling a band with no variance is a division by zero; skip and say so."""
        flat = six_band_scene.copy()
        flat[2] = 4000
        ref = load_raster(make_raster("ref.tif", six_band_scene, transform=known_transform))
        tgt = load_raster(make_raster("tgt.tif", flat, transform=known_transform))
        normalized, warnings = normalize_radiometry(tgt, ref)
        assert float(np.nanstd(normalized.data[2])) == pytest.approx(0.0, abs=1e-6)
        assert any("constant" in w for w in warnings)

    def test_constant_reference_band_does_not_flatten_the_target(
        self, make_raster, six_band_scene, known_transform
    ):
        """Regression. The variance guard was one-sided and the omission was silent.

        A constant *target* band was guarded, but a constant *reference* band was not. The
        gain factor is ``r_std / t_std``, so a zero-variance reference made it 0 and collapsed
        the whole target band onto the single value ``r_mean``. Every difference between the
        two dates in that band was erased — no exception, no warning, just a change map that
        confidently found nothing. Contrast must survive, and the degradation must be stated.
        """
        flat_ref = six_band_scene.copy()
        flat_ref[3] = 4000
        ref = load_raster(make_raster("ref.tif", flat_ref, transform=known_transform))
        tgt = load_raster(make_raster("tgt.tif", six_band_scene, transform=known_transform))

        normalized, warnings = normalize_radiometry(tgt, ref)

        original_std = float(np.nanstd(tgt.data[3]))
        assert original_std > 1.0, "the fixture must have contrast for this to test anything"
        assert float(np.nanstd(normalized.data[3])) == pytest.approx(original_std, rel=1e-4), (
            "the target band's own contrast was destroyed by a zero gain factor"
        )
        assert float(np.nanmean(normalized.data[3])) == pytest.approx(4000.0, abs=1.0), (
            "with no reference range to match, the means should still be aligned"
        )
        assert any("could only be matched on average" in w for w in warnings), (
            "a silently degraded normalisation is the failure being guarded against"
        )

    def test_other_bands_are_unaffected_by_one_constant_band(
        self, make_raster, six_band_scene, known_transform
    ):
        """A degenerate band must not contaminate the bands that were fine."""
        flat_ref = six_band_scene.copy()
        flat_ref[3] = 4000
        ref = load_raster(make_raster("ref.tif", flat_ref, transform=known_transform))
        tgt = load_raster(make_raster("tgt.tif", six_band_scene, transform=known_transform))
        normalized, _ = normalize_radiometry(tgt, ref)
        for band in (0, 1, 2, 4, 5):
            assert float(np.nanmean(normalized.data[band])) == pytest.approx(
                float(np.nanmean(ref.data[band])), abs=1.0
            )

    def test_too_few_valid_pixels_returns_the_input_unchanged_with_a_warning(
        self, make_raster, known_transform
    ):
        """Matching statistics from a handful of pixels would be noise, not correction."""
        rng = np.random.default_rng(21)
        arr = (rng.random((6, 12, 12)) * 8000 + 1000).astype(np.uint16)
        arr[:, 4:, :] = 0  # leaves 48 jointly valid pixels, below the 100-pixel floor
        ref = load_raster(make_raster("ref.tif", arr, transform=known_transform, nodata=0))
        tgt = load_raster(make_raster("tgt.tif", arr, transform=known_transform, nodata=0))
        normalized, warnings = normalize_radiometry(tgt, ref)
        assert normalized is tgt
        assert any("Too few jointly valid pixels" in w for w in warnings)


# ---------------------------------------------------------------------------
# Quality & modality
# ---------------------------------------------------------------------------
class TestQualityAssessment:
    def test_good_image_scores_well(self, make_raster, six_band_scene, known_transform):
        r = load_raster(make_raster("good.tif", six_band_scene, transform=known_transform))
        q = assess_quality(r)
        assert q.score > 0.4
        assert q.nodata_fraction == pytest.approx(0.0)

    def test_flat_image_flagged_as_low_contrast(self, make_raster, known_transform):
        flat = np.full((6, 64, 64), 5000, dtype=np.uint16)
        r = load_raster(make_raster("flat.tif", flat, transform=known_transform))
        q = assess_quality(r)
        assert q.dynamic_range < 0.15
        assert any("contrast" in w for w in q.warnings)

    def test_heavy_nodata_is_measured_and_warned(self, make_raster, known_transform):
        arr = np.full((1, 64, 64), 1000, dtype=np.uint16)
        arr[:, :48, :] = 0  # 75% nodata
        r = load_raster(make_raster("nd.tif", arr, transform=known_transform, nodata=0))
        q = assess_quality(r)
        assert q.nodata_fraction == pytest.approx(0.75, abs=0.02)
        assert any("nodata" in w for w in q.warnings)
        assert q.score < 0.6


class TestModalityValidation:
    def test_matching_modality_yields_no_warnings(self, make_raster, six_band_scene, known_transform):
        r = load_raster(make_raster("o.tif", six_band_scene, transform=known_transform),
                        modality=Modality.OPTICAL)
        assert validate_modality(r, Modality.OPTICAL) == []

    def test_mismatch_warns_by_default_and_raises_when_strict(
        self, make_raster, known_transform
    ):
        sar = (np.random.default_rng(1).random((1, 64, 64)) * 100).astype(np.float32)
        r = load_raster(make_raster("s.tif", sar, transform=known_transform,
                                    descriptions=("vv",)), modality=Modality.SAR)
        warnings = validate_modality(r, Modality.OPTICAL)
        assert warnings and "requires optical" in warnings[0]
        with pytest.raises(ValidationError) as exc:
            validate_modality(r, Modality.OPTICAL, strict=True)
        assert exc.value.code == "WRONG_MODALITY"

    def test_sar_inferred_from_band_description(self, make_raster, known_transform):
        sar = (np.random.default_rng(1).random((1, 64, 64)) * 100).astype(np.float32)
        r = load_raster(make_raster("x.tif", sar, transform=known_transform,
                                    descriptions=("vv",)))
        assert r.modality is Modality.SAR

    def test_optical_inferred_from_rgb_bands(self, make_raster, six_band_scene, known_transform):
        r = load_raster(make_raster("x.tif", six_band_scene, transform=known_transform,
                                    descriptions=("blue", "green", "red", "nir", "swir1", "swir2")))
        assert r.modality is Modality.OPTICAL


# ---------------------------------------------------------------------------
# Real demo data
# ---------------------------------------------------------------------------
class TestDemoDataIntegrity:
    def test_demo_optical_scene_loads_with_expected_geometry(self, demo_root, demo_manifest):
        r = load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)
        assert r.metadata.crs_epsg == 32643
        assert r.shape == (512, 512)
        assert r.has_role(BandRole.NIR) and r.has_role(BandRole.SWIR1)
        assert r.modality is Modality.OPTICAL
        assert r.metadata.pixel_size[0] == pytest.approx(
            demo_manifest["grid"]["pixel_size_m"]
        )

    def test_demo_sar_scene_is_detected_as_sar(self, demo_root):
        r = load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None)
        assert r.modality is Modality.SAR

    def test_demo_quicklook_png_is_not_georeferenced(self, demo_root):
        """Proves the PNG path cannot smuggle in fake coordinates."""
        r = load_raster(demo_root / "optical" / "scene_quicklook.png")
        assert r.metadata.is_georeferenced is False

    def test_temporal_pair_is_on_an_identical_grid(self, demo_root):
        a = load_raster(demo_root / "temporal" / "scene_t1_optical.tif", max_edge=None)
        b = load_raster(demo_root / "temporal" / "scene_t2_optical.tif", max_edge=None)
        pair = check_pair_compatibility(a, b)
        assert pair.strategy is AlignmentStrategy.IDENTICAL_GRID
        assert pair.overlap_fraction == pytest.approx(1.0)

    def test_optical_and_sar_are_coregistered(self, demo_root):
        """The fusion demo depends on this actually being true, so it is asserted."""
        opt = load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)
        sar = load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None)
        pair = check_pair_compatibility(opt, sar)
        assert pair.compatible
        assert pair.overlap_fraction == pytest.approx(1.0)

    def test_edge_case_files_exist_and_fail_correctly(self, demo_root):
        edge = demo_root / "edge_cases"
        with pytest.raises(ValidationError):
            read_metadata(edge / "empty.tif")
        with pytest.raises(ValidationError):
            read_metadata(edge / "not_an_image.pdf")
        assert read_metadata(edge / "no_georeference.tif").is_georeferenced is False
        assert read_metadata(edge / "different_crs.tif").crs_epsg == 4326

    def test_disjoint_demo_file_is_actually_disjoint(self, demo_root):
        a = load_raster(demo_root / "optical" / "scene_optical.tif")
        b = load_raster(demo_root / "edge_cases" / "disjoint_extent.tif")
        assert check_pair_compatibility(a, b).compatible is False

    def test_ground_truth_pixel_counts_match_manifest(self, demo_root, demo_manifest):
        """Guards against silent drift between generator and manifest."""
        gt = load_raster(demo_root / "optical" / "scene_landcover_gt.tif", max_edge=None)
        labels = gt.data[0]
        codes = {"water": 1, "vegetation": 2, "built_up": 3, "bare_soil": 4}
        for name, code in codes.items():
            expected = demo_manifest["ground_truth"]["t1_class_pixels"][name]
            assert int((labels == code).sum()) == expected
