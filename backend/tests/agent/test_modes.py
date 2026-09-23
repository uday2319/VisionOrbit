"""Tests for canonical input-mode normalisation — the step the brief requires *before* routing (§5).

The module had no tests, and it is the keystone of the pipeline: the mode it settles is what the
router sees, what the tool contract is checked against, and what is persisted and reported. Three
properties are pinned here:

* **the mode comes from the rasters, never the query text** — the same sentence normalises differently
  for one image, two dates and an optical+SAR pair;
* **there is no default** — zero inputs and three inputs are errors, not a guessed ``single_optical``;
* **an assumption is stated, not implied** — a pair containing an image whose sensor could not be
  determined is still treated as bi-temporal, but says so in ``pair_warnings`` (§49).

Rasters are written to disk and loaded through :func:`app.geospatial.raster.load_raster`, so the pair
compatibility check runs against real geometry rather than a stand-in.
"""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import Affine

from app.agents.modes import MAX_INPUTS, normalize_request
from app.core.errors import ErrorCode, PairIncompatibleError, ValidationError
from app.core.types import InputMode, Modality
from app.geospatial.raster import load_raster
from tests.conftest import TEST_ORIGIN_X, TEST_ORIGIN_Y, TEST_PIXEL_M, write_test_raster

_BANDS = ("blue", "green", "red", "nir")
_QUERY = "what is in these images"


def _origin(x: float, y: float) -> Affine:
    return Affine(TEST_PIXEL_M, 0.0, x, 0.0, -TEST_PIXEL_M, y)


@pytest.fixture
def scene(tmp_path):
    """Factory for a loaded scene whose modality really is the one asked for.

    ``modality`` is the modality the *loaded* raster must end up with, and the fixture asserts it —
    passing ``Modality.UNKNOWN`` to :func:`load_raster` requests *inference*, not an unknown result,
    and a four-band raster with no descriptions is positionally read as RGB+NIR and inferred optical.
    An undetermined raster is therefore built the way a real one arrives: a single unlabelled band,
    which is exactly the shape of a SAR scene exported without band metadata.
    """
    counter = {"n": 0}

    def _make(
        *,
        modality: Modality = Modality.OPTICAL,
        origin_x: float = TEST_ORIGIN_X,
        origin_y: float = TEST_ORIGIN_Y,
        epsg: int | None = 32643,
    ):
        counter["n"] += 1
        rng = np.random.default_rng(counter["n"])
        bands = 4 if modality is Modality.OPTICAL else 1
        arr = (rng.random((bands, 48, 48)) * 10000).astype(np.uint16)
        path = write_test_raster(
            tmp_path / f"scene_{counter['n']}.tif",
            arr,
            transform=_origin(origin_x, origin_y),
            epsg=epsg,
            descriptions=_BANDS[:bands] if modality is Modality.OPTICAL else None,
        )
        # UNKNOWN means "infer it", so it is not passed through as a declaration.
        raster = load_raster(
            path,
            max_edge=None,
            modality=Modality.UNKNOWN if modality is Modality.UNKNOWN else modality,
        )
        assert raster.modality is modality, "fixture did not produce the modality it claims"
        return raster

    return _make


class TestSingle:
    def test_one_optical_is_single_optical(self, scene):
        req = normalize_request(_QUERY, [scene(modality=Modality.OPTICAL)])
        assert req.mode is InputMode.SINGLE_OPTICAL
        assert req.input_count == 1
        assert not req.is_pair

    def test_one_sar_is_single_sar(self, scene):
        req = normalize_request(_QUERY, [scene(modality=Modality.SAR)])
        assert req.mode is InputMode.SINGLE_SAR

    def test_one_undetermined_is_single_optical(self, scene):
        # SAR is reliably detected when its metadata says so, so an image that evades detection is
        # far more likely optical; the analysis then verifies the bands and refuses if they are absent.
        req = normalize_request(_QUERY, [scene(modality=Modality.UNKNOWN)])
        assert req.mode is InputMode.SINGLE_OPTICAL

    def test_no_pair_warning_for_a_single_image(self, scene):
        req = normalize_request(_QUERY, [scene(modality=Modality.UNKNOWN)])
        assert req.pair_warnings == []


class TestPairs:
    def test_two_optical_is_bitemporal(self, scene):
        req = normalize_request(
            _QUERY, [scene(modality=Modality.OPTICAL), scene(modality=Modality.OPTICAL)]
        )
        assert req.mode is InputMode.BITEMPORAL_PAIR
        assert req.is_pair
        assert req.modality_set() == {Modality.OPTICAL}

    def test_optical_and_sar_is_cross_modal(self, scene):
        req = normalize_request(
            _QUERY,
            [
                scene(modality=Modality.OPTICAL),
                scene(modality=Modality.SAR),
            ],
        )
        assert req.mode is InputMode.OPTICAL_SAR_PAIR

    def test_order_does_not_change_the_cross_modal_verdict(self, scene):
        req = normalize_request(
            _QUERY,
            [
                scene(modality=Modality.SAR),
                scene(modality=Modality.OPTICAL),
            ],
        )
        assert req.mode is InputMode.OPTICAL_SAR_PAIR

    def test_two_sar_is_bitemporal(self, scene):
        req = normalize_request(
            _QUERY,
            [
                scene(modality=Modality.SAR),
                scene(modality=Modality.SAR),
            ],
        )
        assert req.mode is InputMode.BITEMPORAL_PAIR

    def test_details_record_what_was_normalised(self, scene):
        req = normalize_request(
            _QUERY, [scene(modality=Modality.OPTICAL), scene(modality=Modality.OPTICAL)]
        )
        assert req.details["input_count"] == 2
        assert req.details["modalities"] == ["optical", "optical"]
        assert req.details["shapes"] == ["48x48", "48x48"]
        assert req.details["pair_compatibility"]["compatible"] is True

    def test_non_overlapping_pair_is_refused(self, scene):
        with pytest.raises(PairIncompatibleError) as ei:
            normalize_request(
                _QUERY,
                [
                    scene(modality=Modality.OPTICAL),
                    scene(
                        modality=Modality.OPTICAL,
                        origin_x=TEST_ORIGIN_X + 100_000,
                        origin_y=TEST_ORIGIN_Y - 100_000,
                    ),
                ],
            )
        assert ei.value.code == ErrorCode.PAIR_INCOMPATIBLE
        assert ei.value.context["details"]["pair_compatibility"]["compatible"] is False


class TestUndeterminedModalityIsDisclosed:
    """A pair mode assumed because a sensor could not be determined must say so (§49).

    ``cross_modal`` needs a *recognised* optical scene and a *recognised* SAR one, so any pair
    holding an ``UNKNOWN`` falls to ``BITEMPORAL_PAIR`` — which asserts the two are one sensor at two
    dates. Nothing established that; the modality is undetermined precisely because the file carried
    no sensor evidence. The mode is still assumed (an unlabelled optical pair really is bi-temporal),
    but the assumption is recorded rather than left for the user to discover in the answer.
    """

    def test_optical_beside_undetermined_warns(self, scene):
        req = normalize_request(
            _QUERY, [scene(modality=Modality.OPTICAL), scene(modality=Modality.UNKNOWN)]
        )
        assert req.mode is InputMode.BITEMPORAL_PAIR
        warning = "\n".join(req.pair_warnings)
        assert "image 2" in warning
        assert "could not be determined" in warning
        assert "optical" in warning  # names what the other image reads as
        assert "declare the sensor on upload" in warning

    def test_warning_names_the_undetermined_side(self, scene):
        req = normalize_request(
            _QUERY, [scene(modality=Modality.UNKNOWN), scene(modality=Modality.OPTICAL)]
        )
        assert "image 1" in "\n".join(req.pair_warnings)

    def test_two_undetermined_images_warn_about_both(self, scene):
        req = normalize_request(
            _QUERY, [scene(modality=Modality.UNKNOWN), scene(modality=Modality.UNKNOWN)]
        )
        warning = "\n".join(req.pair_warnings)
        assert "Neither image's sensor type" in warning

    def test_a_recognised_pair_carries_no_such_warning(self, scene):
        req = normalize_request(
            _QUERY, [scene(modality=Modality.OPTICAL), scene(modality=Modality.OPTICAL)]
        )
        assert not any("sensor type" in w for w in req.pair_warnings)

    def test_a_cross_modal_pair_carries_no_such_warning(self, scene):
        # Both sensors are recognised, so there is no assumption to disclose.
        req = normalize_request(
            _QUERY,
            [
                scene(modality=Modality.OPTICAL),
                scene(modality=Modality.SAR),
            ],
        )
        assert not any("sensor type" in w for w in req.pair_warnings)


class TestNoDefaultedMode:
    def test_zero_inputs_is_an_error_not_a_mode(self, scene):
        with pytest.raises(ValidationError) as ei:
            normalize_request(_QUERY, [])
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_more_than_two_inputs_is_refused(self, scene):
        images = [scene(modality=Modality.OPTICAL) for _ in range(MAX_INPUTS + 1)]
        with pytest.raises(ValidationError) as ei:
            normalize_request(_QUERY, images)
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES
        assert ei.value.context["received"] == MAX_INPUTS + 1

    def test_query_text_does_not_influence_the_mode(self, scene):
        # The same pair normalises identically however the question is phrased.
        pair = [scene(modality=Modality.OPTICAL), scene(modality=Modality.OPTICAL)]
        assert (
            normalize_request("what changed between the dates", pair).mode
            is normalize_request("fuse the optical and radar images", pair).mode
        )


class TestRequestProperties:
    def test_image_ids_are_carried_through(self, scene):
        req = normalize_request(_QUERY, [scene(modality=Modality.OPTICAL)], image_ids=["img-1"])
        assert req.image_ids == ["img-1"]

    def test_georeferenced_is_all_or_nothing(self, scene):
        mixed = [scene(modality=Modality.OPTICAL), scene(modality=Modality.OPTICAL, epsg=None)]
        assert normalize_request(_QUERY, mixed).georeferenced is False

    def test_rasters_are_copied_not_aliased(self, scene):
        supplied = [scene(modality=Modality.OPTICAL)]
        req = normalize_request(_QUERY, supplied)
        supplied.clear()
        assert req.input_count == 1
