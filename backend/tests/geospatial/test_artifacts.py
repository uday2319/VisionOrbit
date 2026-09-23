"""Tests for the shared artifact renderer — the imagery a result page shows.

This module is the single place both the live API and the recorded-fixture generator get their
pictures from, so the things worth pinning are the ones a plausible-looking rendering would get
wrong: every overlay must be a view of the array it was handed, ``bounds_wgs84`` must stay absent
when the input carries no coordinates (§8), and a rendering failure must be *reported* rather than
turning into a blank panel that looks like an analysis with nothing to show (§27).

Carriers are duck-typed on purpose (``getattr`` for ``class_map`` / ``regime_map`` / ``change_mask``
/ ``mask``), so the stubs below stand in for the real domain results; the API tests exercise the
genuine tool outputs end to end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pytest

from app.core.types import LandCoverClass, Modality, ScatteringRegime
from app.geospatial import artifacts, viz
from app.geospatial.raster import load_raster
from app.schemas import ArtifactSchema, InputMetaSchema

PREFIX = "/storage/artifacts/SAT-2026-000001"


@pytest.fixture
def optical(six_band_scene, make_raster, known_transform):
    """A georeferenced 6-band optical scene at a known UTM origin."""
    return load_raster(
        make_raster(
            "scene.tif", six_band_scene, transform=known_transform,
            descriptions=("blue", "green", "red", "nir", "swir1", "swir2"),
        ),
        modality=Modality.OPTICAL,
    )


@pytest.fixture
def plain(six_band_scene, make_raster):
    """The same pixels with no CRS and an identity transform — a plain image export."""
    return load_raster(make_raster("plain.tif", six_band_scene, epsg=None),
                       modality=Modality.OPTICAL)


def _codes(shape: tuple[int, int]) -> np.ndarray:
    """A code map with two land-cover classes, half the scene each."""
    out = np.full(shape, LandCoverClass.VEGETATION.code, dtype=np.uint8)
    out[: shape[0] // 2] = LandCoverClass.WATER.code
    return out


def _mask(shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[10:20, 10:20] = True
    return out


@dataclass
class HasClassMap:
    class_map: np.ndarray


@dataclass
class HasRegimeMap:
    regime_map: np.ndarray


@dataclass
class HasChangeMask:
    change_mask: np.ndarray


@dataclass
class HasMask:
    mask: np.ndarray


@dataclass
class Caption:
    """A phrased answer that carries the analysis it was phrased from, as the real ones do."""

    landcover: Any


@dataclass
class SceneCaption:
    """The adapted scene tool's result: a caption around a caption around the analysis."""

    caption: Any


def _render(raw: Any, rasters, out_dir) -> tuple[list[dict[str, Any]], list[str]]:
    return artifacts.render_artifacts(raw, rasters, out_dir=out_dir, url_prefix=PREFIX)


def _read_png(path) -> np.ndarray:
    """The written PNG as RGB. Lossless, so it compares against the array that was drawn."""
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class TestWgs84Bounds:
    def test_transforms_the_footprint_into_lat_lon(self, optical):
        """The box is the raster's own footprint in WGS84, not a copy of its UTM numbers."""
        box = artifacts.wgs84_bounds(optical)
        assert box is not None
        assert box["south"] < box["north"]
        assert box["west"] < box["east"]
        # The conftest origin is in UTM zone 43N (EPSG:32643), so the scene lands in western India.
        assert 70.0 < box["west"] < 80.0
        assert 15.0 < box["south"] < 25.0

    def test_is_absent_without_coordinates(self, plain):
        """No CRS means no coordinates to report — never a plausible box (§8)."""
        assert artifacts.wgs84_bounds(plain) is None


class TestInputMeta:
    def test_reports_what_loading_resolved(self, optical):
        meta = artifacts.input_meta(optical)
        assert meta["filename"] == "scene.tif"
        assert meta["modality"] == "optical"
        assert meta["band_roles"] == ["blue", "green", "red", "nir", "swir1", "swir2"]
        assert meta["georeferenced"] is True
        assert meta["bounds_wgs84"] is not None

    def test_an_unnamed_band_keeps_its_position(self, plain):
        """A file that names no band reports nulls, one per band, rather than a shorter list.

        Compacting them would slide band 3's name onto band 2. This also guards the wire schema:
        ``band_descriptions`` was ``list[str]``, so the first unnamed band made the response fail
        validation — a plain PNG upload is enough to hit it.
        """
        meta = artifacts.input_meta(plain)
        assert len(meta["band_descriptions"]) == plain.metadata.count
        assert meta["band_descriptions"] == [None] * plain.metadata.count
        assert InputMetaSchema.model_validate(meta).band_descriptions == [None] * 6

    def test_matches_the_published_schema(self, optical):
        assert InputMetaSchema.model_validate(artifacts.input_meta(optical)).filename == "scene.tif"


class TestBaseComposite:
    def test_base_comes_first_and_is_written_to_disk(self, optical, tmp_path):
        out = tmp_path / "render"
        descriptors, warnings = _render(None, [optical], out)
        assert warnings == []
        assert [d["kind"] for d in descriptors] == ["base"]
        assert descriptors[0]["url"] == f"{PREFIX}/{artifacts.BASE_FILENAME}"
        assert (out / artifacts.BASE_FILENAME).is_file()
        assert descriptors[0]["legend"] == []

    def test_is_rendered_from_the_reference_grid(self, optical, plain, tmp_path):
        """``rasters[0]`` is the grid every map is produced on, so it is the base.

        Alignment resamples the second raster onto the first, so basing the composite on anything
        else would put the overlay on a grid the picture underneath does not share. PNG is lossless,
        so the file read back is the composite of the *first* raster exactly.
        """
        descriptors, _ = _render(None, [optical, plain], tmp_path / "a")
        assert np.array_equal(
            _read_png(tmp_path / "a" / artifacts.BASE_FILENAME), viz.render_rgb(optical)
        )
        assert descriptors[0]["bounds_wgs84"] == artifacts.wgs84_bounds(optical)

    def test_no_rasters_renders_nothing_and_complains_about_nothing(self, tmp_path):
        assert _render(HasClassMap(_codes((8, 8))), [], tmp_path / "empty") == ([], [])

    def test_a_base_failure_yields_no_artifacts_and_one_warning(self, optical, tmp_path, monkeypatch):
        def boom(_raster):
            raise RuntimeError("no bands to composite")

        monkeypatch.setattr(viz, "render_rgb", boom)
        descriptors, warnings = _render(HasClassMap(_codes((64, 64))), [optical], tmp_path / "b")
        assert descriptors == []
        assert len(warnings) == 1
        # The failure is named, and it is clear the analysis itself still stands.
        assert "RuntimeError" in warnings[0]
        assert "analysis itself is unaffected" in warnings[0]


class TestOverlaySelection:
    """Which map gets drawn, and under which kind the UI keys its layer toggle on."""

    def test_a_class_map_becomes_the_land_cover_overlay(self, optical, tmp_path):
        out = tmp_path / "lc"
        descriptors, warnings = _render(HasClassMap(_codes((64, 64))), [optical], out)
        assert warnings == []
        assert [d["kind"] for d in descriptors] == ["base", "class_map"]
        overlay = descriptors[1]
        assert overlay["label"] == "Land-cover classification"
        assert overlay["url"] == f"{PREFIX}/{artifacts.OVERLAY_FILENAME}"
        assert (out / artifacts.OVERLAY_FILENAME).is_file()
        assert {e["key"] for e in overlay["legend"]} == {
            c.value for c in LandCoverClass if c.code != 0
        }

    def test_the_class_overlay_draws_the_codes_it_was_given(self, optical, tmp_path):
        """Two classes, split across the scene — so the drawn halves must differ, and match the key."""
        codes = _codes((64, 64))
        _render(HasClassMap(codes), [optical], tmp_path / "lc2")
        drawn = _read_png(tmp_path / "lc2" / artifacts.OVERLAY_FILENAME)
        water = np.array(viz.CLASS_COLORS["water"], dtype=np.uint8)
        vegetation = np.array(viz.CLASS_COLORS["vegetation"], dtype=np.uint8)
        assert np.array_equal(drawn[codes == LandCoverClass.WATER.code], np.tile(water, (32 * 64, 1)))
        assert np.array_equal(
            drawn[codes == LandCoverClass.VEGETATION.code], np.tile(vegetation, (32 * 64, 1))
        )

    def test_a_regime_map_becomes_the_sar_overlay(self, optical, tmp_path):
        regimes = np.full((64, 64), ScatteringRegime.DIFFUSE.code, dtype=np.uint8)
        regimes[:10] = ScatteringRegime.DOUBLE_BOUNCE.code
        descriptors, warnings = _render(HasRegimeMap(regimes), [optical], tmp_path / "sar")
        assert warnings == []
        assert descriptors[1]["kind"] == "regime_map"
        assert descriptors[1]["label"] == "SAR scattering regimes"
        legend = {e["label"]: e["color"] for e in descriptors[1]["legend"]}
        assert set(legend) == {"Smooth", "Diffuse", "Double bounce"}
        # The swatch is the colour the pixels were drawn with, read from the same table.
        drawn = _read_png(tmp_path / "sar" / artifacts.OVERLAY_FILENAME)
        expected = np.array(
            artifacts._REGIME_COLORS[ScatteringRegime.DOUBLE_BOUNCE.code], dtype=np.uint8
        )
        assert np.array_equal(drawn[0, 0], expected)
        assert legend["Double bounce"] == artifacts._hex(tuple(int(v) for v in expected))

    def test_a_change_mask_tints_only_inside_the_mask(self, optical, tmp_path):
        mask = _mask((64, 64))
        descriptors, warnings = _render(HasChangeMask(mask), [optical], tmp_path / "chg")
        assert warnings == []
        assert descriptors[1]["kind"] == "change_mask"
        assert descriptors[1]["label"] == "Detected change"
        drawn = _read_png(tmp_path / "chg" / artifacts.OVERLAY_FILENAME)
        base = viz.render_rgb(optical)
        # The regions are also outlined, so the two-pixel border around the mask is legitimately
        # painted. Away from that border the base shows through untouched; inside, it is tinted.
        far_outside = np.ones_like(mask)
        far_outside[7:23, 7:23] = False
        assert np.array_equal(drawn[far_outside], base[far_outside])
        assert not np.array_equal(drawn[mask], base[mask])

    def test_a_grounding_mask_becomes_the_grounded_regions_overlay(self, optical, tmp_path):
        descriptors, warnings = _render(HasMask(_mask((64, 64))), [optical], tmp_path / "gnd")
        assert warnings == []
        assert descriptors[1]["kind"] == "grounding"
        assert descriptors[1]["label"] == "Grounded regions"

    def test_a_phrased_answer_shows_the_analysis_it_rests_on(self, optical, tmp_path):
        """A caption or VQA answer carries the land-cover result behind its sentence.

        Without walking to it, those tasks showed a bare input image while the classification they
        were phrased from went undrawn.
        """
        raw = SceneCaption(Caption(HasClassMap(_codes((64, 64)))))
        descriptors, warnings = _render(raw, [optical], tmp_path / "cap")
        assert warnings == []
        assert [d["kind"] for d in descriptors] == ["base", "class_map"]

    def test_a_result_with_no_map_shows_the_input_alone(self, optical, tmp_path):
        out = tmp_path / "none"
        descriptors, warnings = _render(object(), [optical], out)
        assert [d["kind"] for d in descriptors] == ["base"]
        assert warnings == []
        assert not (out / artifacts.OVERLAY_FILENAME).exists()

    def test_a_mask_off_the_base_grid_still_renders(self, optical, tmp_path):
        """Alignment makes this a guard, not a normal path — but it must not lose the overlay."""
        descriptors, warnings = _render(HasChangeMask(_mask((32, 32))), [optical], tmp_path / "odd")
        assert warnings == []
        assert descriptors[1]["kind"] == "change_mask"
        assert _read_png(tmp_path / "odd" / artifacts.OVERLAY_FILENAME).shape[:2] == (32, 32)


class TestFailuresAreReported:
    def test_an_overlay_failure_keeps_the_base_and_states_what_happened(
        self, optical, tmp_path, monkeypatch
    ):
        """§27: a rendering fault is a rendering fault, not "nothing was found"."""

        def boom(*_args, **_kwargs):
            raise ValueError("bad code map")

        monkeypatch.setattr(viz, "render_class_map", boom)
        descriptors, warnings = _render(HasClassMap(_codes((64, 64))), [optical], tmp_path / "f")
        assert [d["kind"] for d in descriptors] == ["base"]
        assert len(warnings) == 1
        assert "ValueError" in warnings[0]
        assert "measurements it would have drawn are unaffected" in warnings[0]


class TestGeoreferencing:
    def test_descriptors_carry_the_scene_footprint(self, optical, tmp_path):
        descriptors, _ = _render(HasClassMap(_codes((64, 64))), [optical], tmp_path / "geo")
        box = artifacts.wgs84_bounds(optical)
        assert [d["bounds_wgs84"] for d in descriptors] == [box, box]

    def test_an_ungeoreferenced_input_reports_no_bounds(self, plain, tmp_path):
        """Null is what tells the UI to use a plain pixel viewer; a plausible box would lie (§8)."""
        descriptors, warnings = _render(HasClassMap(_codes((64, 64))), [plain], tmp_path / "px")
        assert warnings == []
        assert [d["kind"] for d in descriptors] == ["base", "class_map"]
        assert all(d["bounds_wgs84"] is None for d in descriptors)

    def test_descriptors_match_the_published_schema(self, optical, tmp_path):
        descriptors, _ = _render(HasRegimeMap(np.ones((64, 64), np.uint8)), [optical], tmp_path / "s")
        parsed = [ArtifactSchema.model_validate(d) for d in descriptors]
        assert [p.kind for p in parsed] == ["base", "regime_map"]
        assert parsed[1].bounds_wgs84 is not None
        assert all(e.color.startswith("#") for e in parsed[1].legend)
