"""Tests for single-image scene captioning over land cover (brief §2A).

Captioning is the query-free companion to :mod:`app.services.vqa`: it turns one optical scene into
a short descriptive sentence built entirely from the classifier's measured fractions — no language
model, no free-text generation, so it can never describe something the analysis did not find. The
suite covers the phrasing branches with duck-typed stubs (the empty scene, the single-class scene,
the mixed scene, and the georeferenced/plain-image area wording), the evidence lines, and a handful
of real captions of the demo scene where every measurable statement must quote a percentage.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core.types import LandCoverClass
from app.geospatial.raster import load_raster
from app.services.captioning import _evidence, _phrase_caption, describe_scene
from app.services.landcover import classify_land_cover

WATER = LandCoverClass.WATER
VEG = LandCoverClass.VEGETATION
BUILT = LandCoverClass.BUILT_UP
BARE = LandCoverClass.BARE_SOIL


@pytest.fixture(scope="module")
def optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


def _stat(label: LandCoverClass, fraction: float, area_m2: float | None = None):
    return SimpleNamespace(label=label, fraction=fraction, area_m2=area_m2)


def _lc(*stats, warnings=None, indices_used=("mndwi", "ndvi"), method="spectral-indices"):
    dominant = max(stats, key=lambda s: s.fraction) if stats else None
    return SimpleNamespace(
        stats=list(stats),
        dominant=lambda: dominant,
        method=method,
        separability=0.9,
        indices_used=list(indices_used),
        warnings=list(warnings or []),
    )


class TestCaptionPhrasing:
    def test_mixed_scene_names_dominant_then_remaining_then_area(self):
        caption, figures = _phrase_caption(
            _lc(
                _stat(BARE, 0.69, 18_000_000.0),
                _stat(VEG, 0.23, 6_000_000.0),
                _stat(WATER, 0.04, 1_000_000.0),
                _stat(BUILT, 0.04, 1_000_000.0),
            )
        )
        assert caption.startswith("A predominantly bare soil scene (69% of classified pixels).")
        assert "Remaining land cover: vegetation 23%, water 4%, built up 4%." in caption
        assert "km²" in caption  # every class is georeferenced, so a total extent is reported
        assert figures == {"bare_soil": 69.0, "vegetation": 23.0, "water": 4.0, "built_up": 4.0}

    def test_single_class_scene_omits_the_remaining_clause(self):
        caption, figures = _phrase_caption(_lc(_stat(VEG, 1.0, 10_000_000.0)))
        assert caption.startswith("A predominantly vegetation scene (100% of classified pixels).")
        assert "Remaining land cover" not in caption  # nothing else to list, so nothing invented
        assert figures == {"vegetation": 100.0}

    def test_no_area_when_the_scene_is_not_georeferenced(self):
        """A plain image has no ground extent to report — never an estimate dressed as measured."""
        caption, _ = _phrase_caption(_lc(_stat(BARE, 0.6), _stat(VEG, 0.4)))
        assert "predominantly bare soil" in caption
        assert "km²" not in caption

    def test_area_is_withheld_if_any_class_lacks_it(self):
        """Mixed provenance can't be summed into one honest total, so the extent is skipped."""
        caption, _ = _phrase_caption(_lc(_stat(BARE, 0.6, 6_000_000.0), _stat(VEG, 0.4, None)))
        assert "km²" not in caption

    def test_empty_scene_declines_to_describe(self):
        caption, figures = _phrase_caption(_lc())
        assert "cannot be described" in caption
        assert figures == {}


class TestEvidence:
    def test_names_method_separability_dominant_and_indices(self):
        lines = _evidence(_lc(_stat(BARE, 0.7), _stat(VEG, 0.3)))
        joined = " ".join(lines).lower()
        assert "separability" in joined
        assert "spectral-indices" in joined
        assert "dominant class is bare soil" in joined
        assert "mndwi" in joined

    def test_empty_scene_evidence_skips_absent_sections_without_crashing(self):
        lines = _evidence(_lc(warnings=["No identifiable bands."], indices_used=()))
        joined = " ".join(lines).lower()
        assert "separability" in joined
        assert "dominant" not in joined  # no dominant class to name
        assert "indices used" not in joined  # none were used
        assert "No identifiable bands." in lines  # the warning is carried up


class TestDescribesTheDemoScene:
    @pytest.fixture(scope="class")
    def caption(self, optical):
        return describe_scene(optical)

    def test_caption_is_measurement_grounded(self, caption):
        assert caption.caption
        assert "%" in caption.caption
        assert caption.figures  # the shares behind the sentence
        assert caption.warnings == []  # the demo scene is clean

    def test_caption_names_the_dominant_class(self, caption, optical):
        dominant = classify_land_cover(optical).dominant()
        assert dominant is not None
        assert dominant.label.value.replace("_", " ") in caption.caption

    def test_evidence_discloses_provenance(self, caption):
        joined = " ".join(caption.evidence).lower()
        assert "separability" in joined
        assert "classification" in joined

    def test_to_dict_is_json_safe_and_omits_the_class_map_array(self, caption):
        payload = json.loads(json.dumps(caption.to_dict()))  # raises if a numpy array leaked
        assert payload["caption"] == caption.caption
        assert "class_map" not in payload["landcover"]

    def test_reuses_a_supplied_classification(self, optical):
        lc = classify_land_cover(optical)
        cap = describe_scene(optical, landcover_result=lc)
        assert cap.landcover is lc  # the passed-in result is used, not a fresh one

    def test_is_deterministic(self, optical):
        assert describe_scene(optical).caption == describe_scene(optical).caption
