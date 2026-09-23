"""Tests for text-guided region grounding (brief §2B).

Every figure quoted in ``app/services/grounding.py`` is asserted here, so a change that alters
the answer also fails the suite rather than quietly making the documentation wrong. Three claims
carry the module and each has a test that would fail if it were untrue:

* grounding on a raw index instead of the classifier cascade is catastrophically worse;
* pixel accuracy does not imply countability, and radar is what supplies it;
* morphological cleaning removes objects rather than noise.

The rest of the suite covers the parser — including every refusal, which is behaviour rather
than failure — and the geometry gates that stop a compass answer being given for an image with
no north.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace

import cv2
import numpy as np
import pytest
from rasterio.transform import Affine

from app.core.errors import ErrorCode, GeospatialError, UnsupportedTaskError, ValidationError
from app.core.types import LandCoverClass, ScatteringRegime, SpatialSector
from app.geospatial.raster import load_raster
from app.geospatial.vectorize import vectorize_mask
from app.services import features, indices
from app.services.fusion import fuse_optical_sar
from app.services.grounding import (
    MIN_REGION_AREA_M2,
    GroundingQuery,
    _check_geometry,
    ground_query,
    normalise_query,
    parse_grounding_query,
    sector_mask,
)
from app.services.landcover import classify_land_cover

GT_WATER, GT_VEGETATION, GT_BUILT_UP, GT_BARE_SOIL = 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def components(mask: np.ndarray, min_px: int = 25) -> list[np.ndarray]:
    """Connected components of a mask, above the same floor grounding applies."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]


def region_match(
    pred: np.ndarray, truth: np.ndarray, min_iou: float = 0.3
) -> dict[str, float]:
    """Match predicted components to truth components by IoU, greedily and one-to-one.

    Region-level scoring, not pixel-level: it answers "does each predicted object correspond to
    a real one", which is the only question relevant to counting.
    """
    preds, truths = components(pred), components(truth)
    used: set[int] = set()
    tp, ious = 0, []
    for p in preds:
        best, best_iou = -1, 0.0
        for j, t in enumerate(truths):
            if j in used or not (p & t).any():
                continue
            overlap = int((p & t).sum()) / int((p | t).sum())
            if overlap > best_iou:
                best, best_iou = j, overlap
        if best >= 0 and best_iou >= min_iou:
            used.add(best)
            tp += 1
            ious.append(best_iou)
    fp, fn = len(preds) - tp, len(truths) - tp
    return {
        "pred": len(preds), "true": len(truths), "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def demo_optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def demo_sar(demo_root):
    return load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None)


@pytest.fixture(scope="module")
def demo_rgb(demo_root):
    """The same scene with visible bands only — no NIR, no SWIR."""
    return load_raster(demo_root / "edge_cases" / "rgb_only.tif", max_edge=None)


@pytest.fixture(scope="module")
def demo_truth(demo_root):
    return load_raster(
        demo_root / "optical" / "scene_landcover_gt.tif", max_edge=None
    ).data[0].astype(int)


@pytest.fixture(scope="module")
def optical_lc(demo_optical):
    return classify_land_cover(demo_optical)


@pytest.fixture(scope="module")
def fused(demo_optical, demo_sar, optical_lc):
    return fuse_optical_sar(demo_optical, demo_sar, optical_result=optical_lc)


@pytest.fixture(scope="module")
def flat_sar(demo_sar):
    """A radar image with a single diffuse population and no separable bright class.

    A physically real failure mode — a uniformly rough or flooded scene, or a mis-calibrated
    product — and the case where radar cannot arbitrate the built-up boundary at all.
    """
    rng = np.random.default_rng(4)
    linear = 10 ** (-10.0 / 10) * rng.gamma(6.0, 1 / 6.0, size=demo_sar.data.shape)
    return replace(demo_sar, data=(10 * np.log10(linear)).astype(np.float32))


@pytest.fixture
def plain_optical(make_raster, six_band_scene):
    """A six-band scene with real pixels, no CRS, and therefore no north."""
    return load_raster(make_raster("plain.tif", six_band_scene, epsg=None), max_edge=None)


@pytest.fixture
def rotated_optical(make_raster, six_band_scene):
    """Georeferenced but with shear terms, so rows do not run north-to-south."""
    return load_raster(
        make_raster(
            "rotated.tif",
            six_band_scene,
            transform=Affine(10.0, 1.0, 700000.0, 1.0, -10.0, 2100000.0),
        ),
        max_edge=None,
    )


# ---------------------------------------------------------------------------
# Query parsing: targets
# ---------------------------------------------------------------------------
class TestTargetParsing:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("where is the water", LandCoverClass.WATER),
            ("show me the lakes", LandCoverClass.WATER),
            ("find the flooded areas", LandCoverClass.WATER),
            ("how many buildings are there", LandCoverClass.BUILT_UP),
            ("locate built-up land", LandCoverClass.BUILT_UP),
            ("urban areas", LandCoverClass.BUILT_UP),
            ("where is the vegetation", LandCoverClass.VEGETATION),
            ("show the forests", LandCoverClass.VEGETATION),
            ("find cropland", LandCoverClass.VEGETATION),
            ("where is the bare soil", LandCoverClass.BARE_SOIL),
            ("show barren land", LandCoverClass.BARE_SOIL),
            ("locate the sand", LandCoverClass.BARE_SOIL),
        ],
    )
    def test_each_supported_target_parses(self, query, expected):
        assert parse_grounding_query(query).target is expected

    def test_bare_soil_vocabulary_wins_over_vegetation_for_a_fallow_field(self):
        """A fallow field is soil, and the longer phrase is what says so.

        Both "fallow field" and "field" are in the vocabulary. Resolving nested matches
        longest-first is what keeps a ploughed field out of the vegetation class, where the
        spectral classifier would never have put it either.
        """
        assert parse_grounding_query("fallow fields").target is LandCoverClass.BARE_SOIL
        assert parse_grounding_query("ploughed field").target is LandCoverClass.BARE_SOIL
        assert parse_grounding_query("the fields").target is LandCoverClass.VEGETATION

    def test_target_phrase_is_reported_so_the_answer_can_be_explained(self):
        parsed = parse_grounding_query("how many settlements are there")
        assert parsed.target_phrase == "settlements"
        assert parsed.to_dict()["target_phrase"] == "settlements"

    def test_normalisation_folds_hyphens_and_superscripts(self):
        assert normalise_query("Built-Up  Areas!") == "built up areas"
        assert "km2" in normalise_query("larger than 2 km²")
        assert normalise_query("north-east") == "north east"


class TestRefusals:
    """Every refusal here is correct behaviour, not a failure (brief §9, §28)."""

    def test_empty_query_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            parse_grounding_query("   ")
        assert exc.value.code == ErrorCode.EMPTY_QUERY
        assert exc.value.recoverable

    def test_overlong_query_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            parse_grounding_query("water " * 400)
        assert exc.value.code == ErrorCode.QUERY_TOO_LONG

    @pytest.mark.parametrize(
        "query",
        [
            "buildings near the river",
            "vegetation adjacent to the water",
            "the water body closest to the town",
            "bare soil between the fields",
            "buildings along the coast",
        ],
    )
    def test_spatial_relations_are_refused_not_ignored(self, query):
        """Dropping the relation would answer a different question from the one asked."""
        with pytest.raises(UnsupportedTaskError) as exc:
            parse_grounding_query(query)
        assert exc.value.code == ErrorCode.UNSUPPORTED_TASK
        assert "spatial relations" in exc.value.message

    @pytest.mark.parametrize(
        ("query", "named"),
        [
            ("how many ships are there", "ships"),
            ("count the cars", "cars"),
            ("find the roads", "road"),
            ("where is the airport", "airport"),
            ("how many solar panels", "solar panels"),
            ("show the parking lot", "parking lot"),
            ("are there any clouds", "clouds"),
        ],
    )
    def test_undetectable_object_classes_are_refused_by_name(self, query, named):
        """A car park is not built-up land in any sense the user meant.

        Naming the missing detector is the point: "unsupported query" leaves the user guessing,
        while "there is no detector for ships" tells them exactly what this system does not do.
        """
        with pytest.raises(UnsupportedTaskError) as exc:
            parse_grounding_query(query)
        assert named in exc.value.message
        assert "water, vegetation, built-up land and bare soil" in exc.value.message

    def test_unrecognised_target_is_refused_with_the_supported_list(self):
        with pytest.raises(UnsupportedTaskError) as exc:
            parse_grounding_query("show me the elephants")
        assert "water" in exc.value.message and "bare soil" in exc.value.message

    def test_two_targets_at_once_are_refused(self):
        with pytest.raises(UnsupportedTaskError) as exc:
            parse_grounding_query("show water and buildings")
        assert "one land-cover type at a time" in exc.value.message

    def test_no_supported_target_is_shadowed_by_an_unsupported_phrase(self):
        """The two vocabularies must stay disjoint, since the unsupported check runs first.

        A future entry like "dockland" in the target list would be permanently unreachable
        behind "dock" in the unsupported list, and the failure would look like a parser bug
        rather than a vocabulary collision. This asserts the invariant instead of relying on
        review to catch it.
        """
        from app.services.vocabulary import (
            LAND_COVER_TARGETS as _TARGETS,
        )
        from app.services.vocabulary import (
            UNSUPPORTED_OBJECTS as _UNSUPPORTED_TARGETS,
        )

        shadowed = [
            (phrase, bad)
            for phrase, _ in _TARGETS
            for bad in _UNSUPPORTED_TARGETS
            if re.search(rf"\b{re.escape(bad)}\b", phrase)
        ]
        assert shadowed == []


class TestSectorParsing:
    @pytest.mark.parametrize(
        ("query", "sector", "frame_relative"),
        [
            ("built-up land in the north east", SpatialSector.NORTH_EAST, False),
            ("water in the northern half", SpatialSector.NORTH, False),
            ("vegetation in the southwest", SpatialSector.SOUTH_WEST, False),
            ("water in the top left", SpatialSector.NORTH_WEST, True),
            ("buildings on the right", SpatialSector.EAST, True),
            ("vegetation in the middle", SpatialSector.CENTRE, True),
            ("water in the bottom half", SpatialSector.SOUTH, True),
        ],
    )
    def test_compass_and_frame_wording_are_distinguished(
        self, query, sector, frame_relative
    ):
        parsed = parse_grounding_query(query)
        assert parsed.sector is sector
        assert parsed.frame_relative is frame_relative

    def test_a_compass_word_governs_even_when_frame_wording_appears_first(self):
        """The stronger claim wins, so it can be validated rather than silently softened.

        "The top of the northern half" mentions the frame first. Reading it as frame-relative
        would let a rotated image answer without complaint; reading it as a compass claim makes
        the geometry gate check the grid, which is the whole point of having the gate.
        """
        parsed = parse_grounding_query("water in the top of the northern half")
        assert parsed.sector is SpatialSector.NORTH
        assert parsed.frame_relative is False

    def test_no_sector_leaves_the_field_unset(self):
        assert parse_grounding_query("where is the water").sector is None


class TestSizeAndOrderParsing:
    @pytest.mark.parametrize(
        ("query", "field", "expected"),
        [
            ("water larger than 20 hectares", "min_area_m2", 200_000.0),
            ("water bodies over 1 ha", "min_area_m2", 10_000.0),
            ("buildings above 5000 m2", "min_area_m2", 5000.0),
            ("vegetation larger than 2 km²", "min_area_m2", 2_000_000.0),
            ("water smaller than 30 hectares", "max_area_m2", 300_000.0),
            ("buildings under 2 acres", "max_area_m2", 8093.7128448),
            ("vegetation above 500 pixels", "min_pixels", 500),
            ("buildings under 100 px", "max_pixels", 100),
        ],
    )
    def test_units_convert_to_square_metres_or_pixels(self, query, field, expected):
        assert getattr(parse_grounding_query(query), field) == pytest.approx(expected)

    def test_the_acre_is_the_exact_defined_value(self):
        """66 x 660 international feet, not a rounded 4047.

        Rounding a defined constant inside a measurement pipeline is a small dishonesty that
        compounds through every area comparison downstream.
        """
        assert parse_grounding_query("water over 1 acre").min_area_m2 == 4046.8564224

    def test_nested_comparator_reads_as_the_outer_one(self):
        """"No more than 5 ha" is an upper bound, though "more than" sits inside it.

        Resolving by the comparator that *ends* closest to the number, longest first, is what
        gets this right; a naive "last comparator found" scan inverts the filter.
        """
        parsed = parse_grounding_query("buildings no more than 5 ha")
        assert parsed.max_area_m2 == pytest.approx(50_000.0)
        assert parsed.min_area_m2 is None

    @pytest.mark.parametrize(
        ("phrase", "field"),
        [
            ("at least", "min_area_m2"),
            ("minimum", "min_area_m2"),
            ("exceeding", "min_area_m2"),
            ("wider than", "min_area_m2"),
            ("at most", "max_area_m2"),
            ("up to", "max_area_m2"),
            ("maximum", "max_area_m2"),
            ("not more than", "max_area_m2"),
            ("no larger than", "max_area_m2"),
            ("no bigger than", "max_area_m2"),
        ],
    )
    def test_the_whole_comparator_vocabulary_picks_the_right_side(self, phrase, field):
        """A comparator that parses to the wrong side silently inverts the user's filter.

        "Minimum" and "maximum" also cover the case where a shorter comparator ("min", "max")
        nests inside a longer one and must not win: both readings agree on the side here, but
        the same resolution rule is what keeps "no larger than" from reading as "larger than".
        """
        parsed = parse_grounding_query(f"water {phrase} 5 hectares")
        other = "max_area_m2" if field == "min_area_m2" else "min_area_m2"
        assert getattr(parsed, field) == pytest.approx(50_000.0)
        assert getattr(parsed, other) is None

    def test_a_measurement_with_no_comparator_is_ignored(self):
        """"5 ha" alone could mean at least, at most or exactly, so no bound is set."""
        parsed = parse_grounding_query("water bodies of 5 ha")
        assert parsed.min_area_m2 is None
        assert parsed.max_area_m2 is None

    def test_superlatives_and_limits(self):
        assert parse_grounding_query("the largest lake").superlative == "largest"
        assert parse_grounding_query("the largest lake").limit == 1
        assert parse_grounding_query("the 3 largest water bodies").limit == 3
        assert parse_grounding_query("two smallest buildings").limit == 2
        assert parse_grounding_query("two smallest buildings").superlative == "smallest"

    def test_contradictory_superlatives_set_neither(self):
        parsed = parse_grounding_query("the largest and smallest water bodies")
        assert parsed.superlative is None
        assert parsed.limit is None

    def test_count_and_area_intent_are_separate_questions(self):
        counting = parse_grounding_query("how many water bodies are there")
        assert counting.wants_count and not counting.wants_area
        measuring = parse_grounding_query("how much water is in the image")
        assert measuring.wants_area and not measuring.wants_count


# ---------------------------------------------------------------------------
# Sector geometry
# ---------------------------------------------------------------------------
class TestSpatialSectorGeometry:
    def test_opposite_halves_partition_the_frame(self):
        north = sector_mask((64, 64), SpatialSector.NORTH)
        south = sector_mask((64, 64), SpatialSector.SOUTH)
        assert not (north & south).any()
        assert (north | south).all()

    def test_the_four_quadrants_partition_the_frame(self):
        quads = [
            sector_mask((64, 64), s)
            for s in (
                SpatialSector.NORTH_EAST, SpatialSector.NORTH_WEST,
                SpatialSector.SOUTH_EAST, SpatialSector.SOUTH_WEST,
            )
        ]
        stacked = np.stack(quads)
        assert (stacked.sum(axis=0) == 1).all()
        for q in quads:
            assert q.mean() == pytest.approx(0.25)

    def test_centre_is_the_middle_quarter_by_area(self):
        assert sector_mask((64, 64), SpatialSector.CENTRE).mean() == pytest.approx(0.25)

    def test_north_is_the_upper_half_because_rows_increase_downward(self):
        north = sector_mask((64, 64), SpatialSector.NORTH)
        assert north[:32].all() and not north[32:].any()

    def test_only_the_centre_survives_without_a_north_up_grid(self):
        """Naming the middle of a frame involves no orientation assumption; naming north does."""
        for sector in SpatialSector:
            assert sector.needs_north_up is (sector is not SpatialSector.CENTRE)

    def test_every_sector_has_frame_relative_wording(self):
        assert {s.frame_name for s in SpatialSector} == {
            "top", "bottom", "left", "right",
            "top left", "top right", "bottom left", "bottom right", "centre",
        }

    def test_sector_masks_work_on_a_non_square_odd_frame(self):
        """Rounding must leave neither a gap nor an overlap between opposite halves."""
        east = sector_mask((37, 51), SpatialSector.EAST)
        west = sector_mask((37, 51), SpatialSector.WEST)
        assert not (east & west).any()
        assert (east | west).all()


class TestNorthUpDetection:
    def test_the_demo_scene_is_north_up(self, demo_optical):
        assert demo_optical.metadata.is_north_up

    def test_a_rotated_grid_is_not_north_up(self, rotated_optical):
        assert rotated_optical.metadata.is_georeferenced
        assert not rotated_optical.metadata.is_north_up

    def test_an_ungeoreferenced_image_is_never_north_up(self, plain_optical):
        assert not plain_optical.metadata.is_georeferenced
        assert not plain_optical.metadata.is_north_up

    def test_a_south_up_grid_is_not_north_up(self, make_raster, six_band_scene):
        """A positive row step means rows run *northward*, so the top of the array is the south."""
        flipped = load_raster(
            make_raster(
                "south_up.tif",
                six_band_scene,
                transform=Affine(10.0, 0.0, 700000.0, 0.0, 10.0, 2100000.0),
            ),
            max_edge=None,
        )
        assert flipped.metadata.is_georeferenced
        assert not flipped.metadata.is_north_up


# ---------------------------------------------------------------------------
# Geometry gates
# ---------------------------------------------------------------------------
class TestGeometryGates:
    def test_a_compass_sector_is_refused_without_a_north_up_grid(self, plain_optical):
        with pytest.raises(GeospatialError) as exc:
            ground_query("water in the north", plain_optical)
        assert exc.value.code == ErrorCode.MISSING_GEOREFERENCE
        assert "not georeferenced" in exc.value.message
        assert "top" in exc.value.message  # the wording that does work is offered

    def test_the_frame_relative_wording_works_on_the_same_image(self, plain_optical):
        result = ground_query("water in the top half", plain_optical)
        assert result.query.sector is SpatialSector.NORTH
        assert result.query.frame_relative
        assert not result.georeferenced
        assert result.count == 1

    def test_a_compass_sector_is_refused_on_a_rotated_grid(self, rotated_optical):
        with pytest.raises(GeospatialError) as exc:
            ground_query("water in the north", rotated_optical)
        assert "rotated relative to north" in exc.value.message

    def test_the_centre_needs_no_orientation_even_read_as_a_compass_sector(
        self, plain_optical
    ):
        """The middle of the pixel grid is the middle of the footprint under any rotation."""
        query = GroundingQuery(
            raw="centre", normalised="centre", target=LandCoverClass.WATER,
            target_phrase="water", sector=SpatialSector.CENTRE, sector_phrase="centre",
            frame_relative=False,
        )
        _check_geometry(query, plain_optical.metadata)  # must not raise

    def test_a_real_world_size_filter_is_refused_without_georeferencing(self, plain_optical):
        with pytest.raises(GeospatialError) as exc:
            ground_query("water larger than 2 hectares", plain_optical)
        assert "georeferenced image" in exc.value.message
        assert "pixels" in exc.value.message

    def test_a_pixel_size_filter_works_without_georeferencing(self, plain_optical):
        result = ground_query("water larger than 100 pixels", plain_optical)
        assert result.query.min_pixels == 100
        assert result.count == 1
        assert result.regions[0].area_m2 is None
        assert result.selected_area_m2 is None
        assert any("not georeferenced" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# The measurements the design rests on
# ---------------------------------------------------------------------------
class TestSourceMaskChoice:
    def test_a_raw_index_threshold_would_be_a_catastrophic_grounding_source(
        self, demo_optical, demo_truth, optical_lc
    ):
        """NDBI alone finds nearly every building and is nearly always wrong.

        Recall 0.9997 at precision 0.0406 is the signature of a fabricated detection: it looks
        like a detector because it never misses, and it is useless because bare soil is
        short-wave-infrared-bright too. The cascade, which additionally requires edge density,
        scores an order of magnitude better on IoU. This is why grounding never thresholds an
        index directly.
        """
        truth = demo_truth == GT_BUILT_UP
        ndbi = indices.compute_all_available(demo_optical)["ndbi"].array
        finite = np.isfinite(ndbi)
        threshold, _, _ = indices.adaptive_threshold(
            ndbi[finite], indices.DEFAULT_THRESHOLDS["ndbi_builtup"]
        )
        index_mask = finite & (ndbi > threshold)

        index_iou = iou(index_mask, truth)
        cascade_iou = iou(optical_lc.mask_for(LandCoverClass.BUILT_UP), truth)

        assert index_iou == pytest.approx(0.0406, abs=0.005)
        assert cascade_iou == pytest.approx(0.6475, abs=0.01)
        assert cascade_iou > 10 * index_iou

        recall = int((index_mask & truth).sum()) / int(truth.sum())
        assert recall > 0.99, "the index misses almost nothing, which is why it looks credible"

    def test_morphological_cleaning_destroys_regions_instead_of_noise(
        self, optical_lc, demo_truth
    ):
        """Opening and closing the built-up mask leaves 19 regions at radius 1 and 10 at radius 2.

        No true positive is gained at either radius. This is why grounding applies no default
        morphology: the mask looks tidier and the objects in it are gone.
        """
        mask = optical_lc.mask_for(LandCoverClass.BUILT_UP)
        truth = demo_truth == GT_BUILT_UP
        assert len(components(mask)) == 23

        for radius, expected in ((1, 19), (2, 10)):
            cleaned = features.clean_mask(
                mask, open_radius=radius, close_radius=radius, min_area=25
            )
            assert len(components(cleaned)) == expected
            assert region_match(cleaned, truth)["tp"] == 0


class TestCountability:
    """Pixel accuracy does not imply countability, and radar is what supplies it."""

    def test_optical_alone_cannot_count_built_up_areas(self, demo_optical, demo_truth):
        """Region precision and recall are 0.000 despite a respectable pixel-level F1.

        The false-positive soil pixels between the blocks bridge them into one component, so
        every predicted region overlaps several truth blocks and none matches one. A grounding
        module that reported "23 built-up areas" here would be confidently wrong, which is why
        the count is withheld rather than the extent.
        """
        result = ground_query("how many buildings are there", demo_optical)
        truth = demo_truth == GT_BUILT_UP

        assert result.count == 23
        assert result.count_reportable is False
        assert result.count_caveat is not None
        assert "co-registered SAR image" in result.count_caveat
        assert any("cannot be stated reliably" in w for w in result.warnings)

        measured = region_match(result.mask, truth)
        assert measured["precision"] == 0.0
        assert measured["recall"] == 0.0

        # The extent is still a real answer, and is still returned.
        assert result.selected_pixels > 0
        assert result.selected_area_m2 is not None

    @pytest.mark.parametrize("min_iou", [0.1, 0.2, 0.3, 0.5, 0.7])
    def test_the_optical_only_counting_failure_is_not_a_strict_threshold_artefact(
        self, optical_lc, demo_truth, min_iou
    ):
        measured = region_match(
            optical_lc.mask_for(LandCoverClass.BUILT_UP),
            demo_truth == GT_BUILT_UP,
            min_iou=min_iou,
        )
        assert measured["precision"] == 0.0
        assert measured["recall"] == 0.0

    def test_one_optical_component_swallows_all_twenty_of_the_blocks(
        self, optical_lc, demo_truth
    ):
        """The mechanism behind the 0.000, stated as a measurement rather than an explanation.

        A single bridged component touches every one of the twenty truth blocks, and the other
        22 components touch none of them: the count is not "roughly right with some merging",
        it is one blob plus noise.
        """
        blocks = components(demo_truth == GT_BUILT_UP)
        assert len(blocks) == 20
        touched = [
            sum(1 for b in blocks if (c & b).any())
            for c in components(optical_lc.mask_for(LandCoverClass.BUILT_UP))
        ]
        assert max(touched) == 20
        assert sorted(touched, reverse=True)[1] == 0

    def test_radar_makes_the_built_up_count_correct_and_reportable(
        self, demo_optical, demo_sar, demo_truth
    ):
        result = ground_query("how many buildings are there", demo_optical, sar=demo_sar)
        measured = region_match(result.mask, demo_truth == GT_BUILT_UP)

        assert result.count == 20
        assert result.count_reportable is True
        assert result.count_caveat is None
        assert measured == pytest.approx(
            {"pred": 20, "true": 20, "tp": 20, "fp": 0, "fn": 0,
             "precision": 1.0, "recall": 1.0, "mean_iou": 0.9930},
            abs=0.005,
        )
        assert "optical-sar-fusion" in result.source

    def test_no_predicted_component_spans_two_blocks_once_radar_arbitrates(
        self, fused, demo_truth
    ):
        blocks = components(demo_truth == GT_BUILT_UP)
        touched = [
            sum(1 for b in blocks if (c & b).any())
            for c in components(fused.mask_for(LandCoverClass.BUILT_UP))
        ]
        assert max(touched) == 1
        assert min(touched) == 1

    def test_counting_works_from_radar_plus_visible_bands_alone(
        self, demo_rgb, demo_sar, demo_truth
    ):
        """The improvement is radar's, not the infrared bands'.

        With no NIR and no SWIR the optical classifier cannot even name built-up land, and the
        fused result still recovers all twenty blocks at the same mean IoU. That rules out the
        alternative explanation that the extra spectral bands were doing the work.
        """
        result = ground_query("how many buildings are there", demo_rgb, sar=demo_sar)
        measured = region_match(result.mask, demo_truth == GT_BUILT_UP)
        assert result.count == 20
        assert result.count_reportable is True
        assert measured["precision"] == 1.0
        assert measured["recall"] == 1.0
        assert measured["mean_iou"] == pytest.approx(0.9930, abs=0.005)

    def test_bare_soil_moves_with_built_up_because_it_is_the_same_boundary(
        self, demo_optical, demo_sar, demo_truth
    ):
        """Twelve regions for one truth region from optical, one from the fused map.

        Bare soil and built-up sit on opposite sides of the boundary radar arbitrates, so a
        countability rule scoped to built-up alone would have left this case broken.
        """
        truth = demo_truth == GT_BARE_SOIL
        alone = ground_query("where is the bare soil", demo_optical)
        with_radar = ground_query("where is the bare soil", demo_optical, sar=demo_sar)

        assert alone.count == 12
        assert alone.count_reportable is False
        assert region_match(alone.mask, truth)["precision"] == pytest.approx(0.083, abs=0.01)

        assert with_radar.count == 1
        assert with_radar.count_reportable is True
        assert region_match(with_radar.mask, truth)["mean_iou"] == pytest.approx(
            0.9890, abs=0.005
        )

    @pytest.mark.parametrize(
        ("query", "target", "gt_code", "expected"),
        [
            ("how many water bodies are there", LandCoverClass.WATER, GT_WATER, 3),
            ("how many forests are there", LandCoverClass.VEGETATION, GT_VEGETATION, 4),
        ],
    )
    def test_water_and_vegetation_need_no_radar_to_be_countable(
        self, demo_optical, demo_sar, demo_truth, query, target, gt_code, expected
    ):
        """Only the built-up axis moves, so only it depends on radar.

        Both classes return the right number of regions with perfect region-level precision and
        recall from optical alone, and adding radar changes neither. A blanket "counting needs
        SAR" rule would have withheld two answers that are already correct.
        """
        truth = demo_truth == gt_code
        alone = ground_query(query, demo_optical)
        with_radar = ground_query(query, demo_optical, sar=demo_sar)

        for result in (alone, with_radar):
            measured = region_match(result.mask, truth)
            assert result.query.target is target
            assert result.count == expected
            assert result.count_reportable is True
            assert measured["precision"] == 1.0
            assert measured["recall"] == 1.0

    def test_a_veto_still_requires_a_measurement(self, demo_optical, flat_sar):
        """Radar that separated no bright population cannot make the count reportable.

        The countability rule keys on whether a double-bounce population was actually separated,
        not on whether a SAR file was supplied. Supplying radar that could not measure the
        boundary must leave the count withheld, and the caveat must say which of the two
        failures happened.
        """
        result = ground_query("how many buildings are there", demo_optical, sar=flat_sar)
        assert ScatteringRegime.DOUBLE_BOUNCE not in result.evidence["sar_regimes_separated"]
        assert result.count_reportable is False
        assert result.count_caveat is not None
        assert "no separable bright" in result.count_caveat
        assert "partial" in result.source


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
class TestSectorSelection:
    def test_selection_is_by_majority_overlap_not_by_touching(
        self, demo_optical, optical_lc
    ):
        """Three water bodies touch the centre of the demo scene; one lies mainly in it.

        Counting every region that overlaps would triple the answer. Majority overlap is the
        honest rule, and the per-region overlap fraction is reported so a region at 0.62 is not
        presented as if it were wholly inside.
        """
        centre = sector_mask(demo_optical.shape, SpatialSector.CENTRE)
        water = components(optical_lc.mask_for(LandCoverClass.WATER))
        touching = sum(1 for c in water if (c & centre).any())
        majority = sum(1 for c in water if (c & centre).sum() / c.sum() > 0.5)
        assert (majority, touching) == (1, 3)

        result = ground_query("water in the middle", demo_optical)
        assert result.count == majority
        assert result.excluded["outside_sector"] == 2
        for region in result.regions:
            assert region.sector_overlap is not None
            assert region.sector_overlap > 0.5

    def test_a_compass_sector_selects_the_measured_number_of_regions(
        self, demo_optical, optical_lc
    ):
        north_east = sector_mask(demo_optical.shape, SpatialSector.NORTH_EAST)
        expected = sum(
            1
            for c in components(optical_lc.mask_for(LandCoverClass.BUILT_UP))
            if (c & north_east).sum() / c.sum() > 0.5
        )
        assert expected == 11

        result = ground_query("built-up areas in the north east", demo_optical)
        assert result.count == expected
        assert result.evidence["sector_selection"] == (
            "majority overlap (more than half the region)"
        )

    def test_no_sector_reports_no_overlap_fraction(self, demo_optical):
        result = ground_query("where is the water", demo_optical)
        assert all(r.sector_overlap is None for r in result.regions)
        assert result.evidence["sector_selection"] is None


class TestSizeFilters:
    def test_a_ground_area_lower_bound_selects_the_one_large_water_body(self, demo_optical):
        """The demo scene's water bodies are 19.40, 20.67 and 75.65 hectares."""
        result = ground_query("water bodies larger than 50 hectares", demo_optical)
        assert result.count == 1
        assert result.excluded["below_min_area"] == 2
        assert result.selected_area_m2 == pytest.approx(756_500, rel=0.01)

    def test_a_lower_bound_below_every_region_selects_all_of_them(self, demo_optical):
        result = ground_query("water bodies larger than 1 hectare", demo_optical)
        assert result.count == 3
        assert not any(result.excluded.values())

    def test_an_upper_bound_excludes_the_large_one(self, demo_optical):
        result = ground_query("water bodies smaller than 50 hectares", demo_optical)
        assert result.count == 2
        assert result.excluded["above_max_area"] == 1

    def test_a_filter_that_matches_nothing_explains_which_filter_emptied_it(
        self, demo_optical
    ):
        result = ground_query("water bodies larger than 1000 hectares", demo_optical)
        assert result.count == 0
        assert result.excluded["below_min_area"] == 3
        assert any("size limit in the question" in w for w in result.warnings)
        assert result.class_pixels > 0, "the class is present; only the filter emptied it"

    def test_regions_below_the_noise_floor_are_never_reported(self, demo_optical):
        result = ground_query("where is the water", demo_optical)
        floor = result.evidence["min_region_pixels"]
        assert floor == 25  # 2500 m² at 10 m pixels
        assert result.evidence["min_region_area_m2"] == MIN_REGION_AREA_M2
        assert all(r.pixel_area >= floor for r in result.regions)


class TestEmptyResultExplanations:
    """An empty answer must name the filter that emptied it (brief §28, §29).

    "No regions found" is true of four different situations that call for four different next
    actions from the user, and the only way to keep these sentences accurate is to assert them.
    """

    @pytest.fixture
    def one_lake_north_west(self, make_raster, six_band_scene, known_transform):
        """The stock scene, whose single 20x20 water square sits in the top-left quadrant."""
        return load_raster(
            make_raster("nw_lake.tif", six_band_scene, transform=known_transform),
            max_edge=None,
        )

    def test_a_sector_that_excludes_everything_names_the_sector(self, one_lake_north_west):
        result = ground_query("water in the south east", one_lake_north_west)
        assert result.count == 0
        assert result.excluded["outside_sector"] == 1
        assert any(
            "none of them lies mainly in the south east" in w for w in result.warnings
        )

    def test_the_frame_wording_is_echoed_back_when_that_is_what_was_asked(
        self, one_lake_north_west
    ):
        """Answering a "bottom right" question in compass terms would change the question."""
        result = ground_query("water in the bottom right", one_lake_north_west)
        assert result.count == 0
        assert any(
            "none of them lies mainly in the bottom right" in w for w in result.warnings
        )

    def test_a_pixel_count_limit_is_explained_in_pixels(self, one_lake_north_west):
        result = ground_query("water larger than 1000 pixels", one_lake_north_west)
        assert result.count == 0
        assert result.excluded["below_min_pixels"] == 1
        assert any("pixel-count limit in the question" in w for w in result.warnings)

    def test_an_upper_pixel_bound_is_explained_the_same_way(self, one_lake_north_west):
        result = ground_query("water smaller than 30 pixels", one_lake_north_west)
        assert result.count == 0
        assert result.excluded["above_max_pixels"] == 1
        assert any("pixel-count limit in the question" in w for w in result.warnings)

    @pytest.mark.parametrize(
        ("georeferenced", "unit"),
        [(True, "no patch of at least 2500 m2"), (False, "no patch of at least 25 pixels")],
    )
    def test_class_pixels_below_the_patch_floor_are_reported_as_such(
        self, make_raster, six_band_scene, known_transform, georeferenced, unit
    ):
        """Sixteen water pixels are a real measurement and not a distinct area.

        Reporting zero regions without saying that water *was* detected would read as "there is
        no water here", which is a different and false claim. The floor is quoted in whichever
        unit the image can actually support, so an ungeoreferenced picture is never told its
        limit in square metres.
        """
        scene = six_band_scene.copy()
        water = scene[:, 15, 15].copy()
        scene[:, 10:30, 10:30] = scene[:, 0, 0][:, None, None]  # erase the lake
        scene[:, 40:44, 40:44] = water[:, None, None]  # 16 px, under the 25 px floor
        raster = load_raster(
            make_raster(
                f"speck_{georeferenced}.tif",
                scene,
                transform=known_transform if georeferenced else None,
            ),
            max_edge=None,
        )
        assert raster.metadata.is_georeferenced is georeferenced

        result = ground_query("where is the water", raster)
        assert result.count == 0
        assert result.class_pixels == 16
        assert any("Water pixels are present" in w for w in result.warnings)
        assert any(unit in w for w in result.warnings)

    def test_a_georeferenced_image_whose_area_cannot_be_measured_says_so(
        self, make_raster, six_band_scene
    ):
        """The georeference gate and the area measurement are two different questions.

        A degenerate transform — zero ground height per row — carries a CRS, so the gate lets a
        hectares filter through, and then no region has an area to test. Silently returning zero
        matches would present an unmeasurable image as an empty one.
        """
        raster = load_raster(
            make_raster(
                "degenerate.tif",
                six_band_scene,
                transform=Affine(10.0, 0.0, 700000.0, 0.0, 0.0, 2100000.0),
            ),
            max_edge=None,
        )
        assert raster.metadata.is_georeferenced

        result = ground_query("water larger than 1 hectare", raster)
        assert result.count == 0
        assert result.excluded["area_unknown"] == 1
        assert any("could not be established" in w for w in result.warnings)


class TestOrdering:
    def test_largest_first_is_the_default_order(self, demo_optical):
        result = ground_query("where is the water", demo_optical)
        areas = [r.area_m2 for r in result.regions]
        assert areas == sorted(areas, reverse=True)

    def test_the_largest_returns_one_region_and_says_what_it_dropped(self, demo_optical):
        result = ground_query("the largest water body", demo_optical)
        assert result.count == 1
        assert result.excluded["beyond_limit"] == 2
        assert result.regions[0].area_m2 == pytest.approx(756_500, rel=0.01)

    def test_the_smallest_reverses_the_order(self, demo_optical):
        largest = ground_query("the largest water body", demo_optical)
        smallest = ground_query("the smallest water body", demo_optical)
        assert smallest.count == 1
        assert smallest.regions[0].area_m2 == pytest.approx(194_000, rel=0.02)
        assert smallest.regions[0].area_m2 < largest.regions[0].area_m2

    def test_an_explicit_count_limits_and_keeps_the_order(self, demo_optical):
        result = ground_query("the 2 largest water bodies", demo_optical)
        areas = [r.area_m2 for r in result.regions]
        assert result.count == 2
        assert areas == sorted(areas, reverse=True)
        assert result.excluded["beyond_limit"] == 1


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------
class TestResultContract:
    def test_the_returned_mask_is_exactly_the_returned_regions(
        self, demo_optical, optical_lc
    ):
        """An overlay must show what was reported, not the whole class behind it."""
        result = ground_query("the largest water body", demo_optical)
        assert result.selected_pixels == int(result.mask.sum())
        assert result.selected_pixels == sum(r.pixel_area for r in result.regions)
        assert result.selected_pixels < result.class_pixels
        assert not (result.mask & ~optical_lc.mask_for(LandCoverClass.WATER)).any()

    def test_the_class_totals_describe_the_whole_class_not_the_selection(self, demo_optical):
        """A filtered answer must not shrink the scene it was measured against."""
        whole = ground_query("where is the water", demo_optical)
        filtered = ground_query("the largest water body", demo_optical)
        assert filtered.class_pixels == whole.class_pixels
        assert filtered.class_coverage_fraction == whole.class_coverage_fraction

    def test_a_region_count_is_never_presented_as_a_building_count(self, demo_optical):
        semantics = ground_query("where are the buildings", demo_optical).region_semantics
        assert "not an individual building" in semantics
        assert "10 m per pixel" in semantics

    def test_evidence_carries_the_measurements_confidence_will_need(
        self, demo_optical, demo_sar
    ):
        result = ground_query("where are the buildings", demo_optical, sar=demo_sar)
        evidence = result.evidence
        assert evidence["classification_method"] == "spectral-indices"
        assert "ndbi" in evidence["indices_used"]
        assert 0.0 <= evidence["class_separability"] <= 1.0
        assert evidence["region_score_index"] == "ndbi"
        assert 0.0 <= evidence["fusion_agreement_fraction"] <= 1.0
        assert 0.0 <= evidence["fusion_corroborated_fraction"] <= 1.0
        assert "double_bounce" in evidence["sar_regimes_separated"]

    def test_optical_only_evidence_omits_the_fusion_fields_rather_than_faking_them(
        self, demo_optical
    ):
        evidence = ground_query("where is the water", demo_optical).evidence
        assert "fusion_method" not in evidence
        assert "fusion_agreement_fraction" not in evidence

    def test_per_region_scores_are_evidence_and_stay_inside_the_unit_interval(
        self, demo_optical
    ):
        result = ground_query("where is the water", demo_optical)
        assert result.evidence["region_score_index"] == "mndwi"
        for region in result.regions:
            assert region.region.score is not None
            assert 0.0 <= region.region.score <= 1.0

    def test_bare_soil_gets_no_invented_score(self, demo_optical):
        """It is the cascade's remainder class, so no index magnitude means "more bare"."""
        result = ground_query("where is the bare soil", demo_optical)
        assert result.evidence["region_score_index"] is None
        assert all(r.region.score is None for r in result.regions)

    def test_the_result_serialises_for_the_api(self, demo_optical, demo_sar):
        result = ground_query(
            "the 2 largest built-up areas in the north", demo_optical, sar=demo_sar
        )
        payload = json.loads(json.dumps(result.to_dict()))  # raises on any numpy scalar
        assert payload["count"] == result.count == 2
        assert payload["query"]["sector"] == "north"
        assert payload["query"]["sector_interpretation"] == "compass"
        assert len(payload["regions"]) == result.count
        assert payload["excluded"], "a limited answer must say what it left out"
        assert payload["region_semantics"]

    def test_geojson_carries_one_feature_per_returned_region(self, demo_optical):
        result = ground_query("where is the water", demo_optical)
        collection = result.to_geojson()
        assert collection["type"] == "FeatureCollection"
        assert len(collection["features"]) == result.count

    def test_geojson_is_empty_rather_than_invented_without_coordinates(self, plain_optical):
        """A region with no CRS has no lon/lat, so there is no feature to emit."""
        result = ground_query("where is the water", plain_optical)
        assert result.count == 1
        assert result.to_geojson()["features"] == []

    def test_a_class_that_is_absent_says_so_instead_of_returning_nothing(
        self, make_raster, six_band_scene
    ):
        """Zero regions with no explanation is indistinguishable from a failure."""
        no_water = six_band_scene.copy()
        no_water[:, 10:30, 10:30] = no_water[:, 0, 0][:, None, None]
        raster = load_raster(make_raster("dry.tif", no_water), max_edge=None)
        result = ground_query("where is the water", raster)
        assert result.count == 0
        assert result.class_pixels == 0
        assert any("nothing to locate" in w for w in result.warnings)

    def test_source_warnings_are_carried_through_to_the_regions(self, demo_rgb):
        """A caveat about the classification is equally a caveat about regions taken from it."""
        result = ground_query("where is the water", demo_rgb)
        assert any("short-wave infrared" in w for w in result.warnings)

    def test_a_precomputed_classification_is_reused_rather_than_recomputed(
        self, demo_optical, optical_lc
    ):
        result = ground_query("where is the water", demo_optical, optical_result=optical_lc)
        assert result.source == "optical-classification (spectral-indices)"
        assert result.count == 3

    def test_a_precomputed_fusion_takes_precedence_over_a_raster(
        self, demo_optical, demo_sar, fused
    ):
        result = ground_query(
            "how many buildings are there", demo_optical, sar=demo_sar, fusion_result=fused
        )
        assert result.count == 20
        assert result.source == f"optical-sar-fusion ({fused.method})"


# ---------------------------------------------------------------------------
# The vectorize support grounding needed
# ---------------------------------------------------------------------------
class TestRegionPixelRecovery:
    """``pixels_of`` exists so per-region statistics need no second labelling pass.

    Grounding measures sector overlap and index scores against each region's exact pixels. The
    alternative — approximating a region by its simplified polygon — would make the overlap
    fraction that decides sector membership disagree with the mask that gets returned.
    """

    def test_a_region_reports_its_exact_pixels(self, make_raster, six_band_scene):
        raster = load_raster(make_raster("px.tif", six_band_scene), max_edge=None)
        mask = np.zeros(raster.shape, dtype=bool)
        mask[5:15, 5:15] = True
        mask[40:50, 40:50] = True
        result = vectorize_mask(mask, raster.metadata, min_pixels=25)

        assert result.count == 2
        recovered = [result.pixels_of(r) for r in result.regions]
        for pixels, region in zip(recovered, result.regions, strict=True):
            assert int(pixels.sum()) == region.pixel_area
            assert region.label > 0
        assert not (recovered[0] & recovered[1]).any()
        assert ((recovered[0] | recovered[1]) == mask).all()

    def test_a_label_less_result_refuses_instead_of_returning_an_empty_mask(
        self, make_raster, six_band_scene
    ):
        """An empty mask would read as "this region has no pixels", a different claim entirely."""
        raster = load_raster(make_raster("empty.tif", six_band_scene), max_edge=None)
        populated = vectorize_mask(
            np.ones(raster.shape, dtype=bool), raster.metadata, min_pixels=1
        )
        empty = vectorize_mask(np.zeros(raster.shape, dtype=bool), raster.metadata)

        assert empty.count == 0
        assert empty.labels is None
        with pytest.raises(ValueError, match="no component labels"):
            empty.pixels_of(populated.regions[0])

    def test_the_label_is_internal_and_stays_out_of_the_api_payload(
        self, make_raster, six_band_scene
    ):
        """It indexes an array the client never receives, so exposing it would only mislead."""
        raster = load_raster(make_raster("lbl.tif", six_band_scene), max_edge=None)
        mask = np.zeros(raster.shape, dtype=bool)
        mask[5:15, 5:15] = True
        region = vectorize_mask(mask, raster.metadata, min_pixels=25).regions[0]
        assert region.label > 0
        assert "label" not in region.to_dict()
