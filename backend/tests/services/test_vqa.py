"""Tests for single-image visual question answering over land cover (brief §2A, §9, §28).

VQA has no model of its own: every answer is a sentence phrased from the land-cover fractions
:mod:`app.services.landcover` measured, and every refusal follows from the *absence* of a detector
in the shared vocabulary (:mod:`app.services.vocabulary`) — the same vocabulary region grounding
reads, so the two must refuse the same things for the same reasons. The suite is therefore in four
parts:

* **interpretation** — a normalised query is reduced to the right intent and target(s), including
  the precedence rules (a *where* or count-of-regions question is redirected, not answered as
  coverage) and the target-blanking that stops a class name being read as a sector;
* **phrasing** — each branch of the answer template, driven with duck-typed stubs so the empty,
  absent-class and area/no-area wordings are all covered without fabricating a full result;
* **end-to-end** — real answers against the demo scene, every measurable one quoting a percentage;
* **refusals** — behaviour, not failure: objects with no detector, spatial relations, empty and
  overlong queries, and the invariant that VQA refuses exactly what grounding refuses.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core.errors import ErrorCode, UnsupportedTaskError, ValidationError
from app.core.types import LandCoverClass
from app.geospatial.raster import load_raster
from app.services.grounding import parse_grounding_query
from app.services.vocabulary import normalise_query
from app.services.vqa import (
    VqaIntent,
    _extent_phrase,
    _interpret,
    _phrase,
    _phrase_comparison,
    _phrase_composition,
    answer_question,
)

WATER = LandCoverClass.WATER
VEG = LandCoverClass.VEGETATION
BUILT = LandCoverClass.BUILT_UP
BARE = LandCoverClass.BARE_SOIL


@pytest.fixture(scope="module")
def optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


# ---------------------------------------------------------------------------
# Stub builders for the pure phrasing functions
# ---------------------------------------------------------------------------
def _stat(label: LandCoverClass, fraction: float, area_m2: float | None = None):
    return SimpleNamespace(label=label, fraction=fraction, area_m2=area_m2)


def _lc(*stats):
    """A duck-typed :class:`LandCoverResult`: enough surface for the phrasing helpers."""
    dominant = max(stats, key=lambda s: s.fraction) if stats else None
    return SimpleNamespace(
        stats=list(stats),
        dominant=lambda: dominant,
        method="spectral-indices",
        separability=0.9,
        indices_used=["mndwi", "ndvi"],
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
class TestInterpretation:
    """Reducing a normalised query to (intent, targets). The parser, not the classifier."""

    def _read(self, query: str):
        return _interpret(normalise_query(query))

    def test_bare_target_is_a_presence_question(self):
        assert self._read("water") == (VqaIntent.PRESENCE, [WATER])

    def test_is_there_is_a_presence_question(self):
        assert self._read("is there any water in this image") == (VqaIntent.PRESENCE, [WATER])

    def test_how_much_is_a_quantity_question(self):
        assert self._read("how much vegetation is there") == (VqaIntent.QUANTITY, [VEG])

    def test_what_percentage_is_a_quantity_question(self):
        assert self._read("what percentage of the scene is built up") == (
            VqaIntent.QUANTITY,
            [BUILT],
        )

    def test_how_many_is_a_count_question(self):
        """"How many" asks for a count of regions — grounding's job — not for coverage."""
        assert self._read("how many buildings are there") == (VqaIntent.COUNT, [BUILT])

    def test_where_is_a_location_question(self):
        assert self._read("where is the water") == (VqaIntent.LOCATION, [WATER])

    def test_describe_without_a_target_is_a_composition_question(self):
        assert self._read("describe the scene") == (VqaIntent.COMPOSITION, [])

    def test_what_land_cover_is_a_composition_question(self):
        assert self._read("what land cover is in this image") == (VqaIntent.COMPOSITION, [])

    def test_describe_a_single_class_is_that_class_quantity(self):
        """"Describe the vegetation" is answerable as how much of that class there is."""
        assert self._read("describe the vegetation") == (VqaIntent.QUANTITY, [VEG])

    def test_two_distinct_classes_is_a_comparison_in_first_appearance_order(self):
        assert self._read("is there more water or vegetation") == (
            VqaIntent.COMPARISON,
            [WATER, VEG],
        )

    def test_a_sector_restricted_quantity_is_redirected_not_answered(self):
        """"How much water in the north" restricts to a sector VQA cannot honour; the honest
        move is to redirect to grounding (LOCATION), not to quote a whole-scene percentage."""
        assert self._read("how much water is in the north") == (VqaIntent.LOCATION, [WATER])

    def test_a_class_name_embedding_a_direction_is_not_read_as_a_sector(self):
        """"rooftop" contains "top"; blanking the target span keeps it a quantity question
        rather than a location one (regression guard on :func:`_blank_targets`)."""
        assert self._read("how much rooftop is there") == (VqaIntent.QUANTITY, [BUILT])

    def test_no_target_and_no_description_is_uninterpretable(self):
        assert _interpret(normalise_query("what is the weather like")) == (None, [])


# ---------------------------------------------------------------------------
# Phrasing branches
# ---------------------------------------------------------------------------
class TestPresenceAndQuantityPhrasing:
    def test_presence_of_a_measured_class_quotes_the_coverage(self):
        answer, figures, redirect = _phrase(
            VqaIntent.PRESENCE, [WATER], _lc(_stat(WATER, 0.25, 1_000_000.0), _stat(VEG, 0.75))
        )
        assert answer.lower().startswith("yes")
        assert "25%" in answer and "km²" in answer  # measured share and georeferenced area
        assert figures == {"water": 25.0}
        assert redirect is None

    def test_presence_of_an_absent_class_says_so_without_a_number(self):
        answer, figures, redirect = _phrase(VqaIntent.PRESENCE, [WATER], _lc(_stat(VEG, 1.0)))
        assert answer.lower().startswith("no")
        assert "%" not in answer  # nothing measured -> no fabricated figure
        assert figures == {} and redirect is None

    def test_quantity_of_a_measured_class_leads_with_the_percentage(self):
        answer, figures, _ = _phrase(VqaIntent.QUANTITY, [VEG], _lc(_stat(VEG, 0.4, 500_000.0)))
        assert answer.startswith("Vegetation covers 40%")
        assert figures == {"vegetation": 40.0}

    def test_quantity_of_an_absent_class_reports_zero_coverage(self):
        answer, figures, _ = _phrase(VqaIntent.QUANTITY, [WATER], _lc(_stat(VEG, 1.0)))
        assert "0% coverage" in answer
        assert figures == {}


class TestRedirectPhrasing:
    """Count and location questions keep the coverage figure but hand off to grounding."""

    def test_count_answers_how_much_and_redirects_for_how_many(self):
        answer, figures, redirect = _phrase(
            VqaIntent.COUNT, [BUILT], _lc(_stat(BUILT, 0.04, 950_000.0))
        )
        assert "4%" in answer
        assert "how much, not how many" in answer
        assert redirect == "grounding"
        assert figures == {"built_up": 4.0}

    def test_count_of_an_absent_class_has_nothing_to_count(self):
        answer, figures, redirect = _phrase(VqaIntent.COUNT, [WATER], _lc(_stat(VEG, 1.0)))
        assert "none to count" in answer
        assert figures == {} and redirect is None

    def test_location_answers_coverage_and_redirects_for_where(self):
        answer, _, redirect = _phrase(VqaIntent.LOCATION, [WATER], _lc(_stat(WATER, 0.04)))
        assert "4%" in answer
        assert "region grounding" in answer.lower()
        assert redirect == "grounding"

    def test_location_of_an_absent_class_has_no_location(self):
        answer, figures, redirect = _phrase(VqaIntent.LOCATION, [WATER], _lc(_stat(VEG, 1.0)))
        assert "no location to report" in answer
        assert figures == {} and redirect is None


class TestCompositionPhrasing:
    def test_composition_names_the_dominant_class_and_the_full_breakdown(self):
        answer, figures, redirect = _phrase_composition(
            _lc(_stat(BARE, 0.69), _stat(VEG, 0.23), _stat(WATER, 0.04), _stat(BUILT, 0.04))
        )
        assert "predominantly bare soil" in answer
        assert "69%" in answer
        assert "breakdown" in answer.lower()
        assert figures == {"bare_soil": 69.0, "vegetation": 23.0, "water": 4.0, "built_up": 4.0}
        assert redirect is None

    def test_composition_of_an_empty_scene_declines_instead_of_indexing_nothing(self):
        answer, figures, redirect = _phrase_composition(_lc())
        assert "cannot be described" in answer
        assert figures == {} and redirect is None


class TestComparisonPhrasing:
    def test_comparison_reports_each_share_and_which_is_larger(self):
        answer, figures, _ = _phrase_comparison([WATER, VEG], _lc(_stat(WATER, 0.10), _stat(VEG, 0.40)))
        assert "covers more of the scene than" in answer
        assert answer.lower().startswith("vegetation")  # the larger class leads the verdict
        assert figures == {"vegetation": 40.0, "water": 10.0}

    def test_equal_shares_are_called_equal_not_ranked(self):
        answer, _, _ = _phrase_comparison([WATER, VEG], _lc(_stat(WATER, 0.3), _stat(VEG, 0.3)))
        assert "about equally" in answer


class TestExtentPhrase:
    def test_a_georeferenced_class_gets_a_measured_area(self):
        assert _extent_phrase(37.0, _stat(WATER, 0.37, 2_000_000.0)) == (
            "37% of the classified scene (about 2.00 km²)"
        )

    def test_an_ungeoreferenced_class_reports_percent_only(self):
        phrase = _extent_phrase(37.0, _stat(WATER, 0.37, None))
        assert "37% of the classified scene" in phrase
        assert "km²" not in phrase  # no ground area to claim

    def test_missing_stats_reports_percent_only(self):
        assert _extent_phrase(0.0, None) == "0% of the classified scene"


# ---------------------------------------------------------------------------
# End-to-end against the real demo scene
# ---------------------------------------------------------------------------
class TestAnswersTheDemoScene:
    def test_composition_answer_is_measurement_grounded(self, optical):
        ans = answer_question("what land cover is in this image", optical)
        assert ans.intent is VqaIntent.COMPOSITION
        assert ans.targets == []
        assert "%" in ans.answer
        assert ans.figures  # per-class shares carried through
        assert ans.warnings == []  # the demo scene is clean
        assert ans.redirect is None

    def test_quantity_answer_quotes_a_percentage(self, optical):
        ans = answer_question("how much bare soil is there", optical)
        assert ans.intent is VqaIntent.QUANTITY
        assert "%" in ans.answer
        assert ans.figures.get("bare_soil") is not None

    def test_presence_answer_confirms_and_measures(self, optical):
        ans = answer_question("is there any water", optical)
        assert ans.intent is VqaIntent.PRESENCE
        assert ans.answer.lower().startswith("yes")  # water is present in the demo scene
        assert "%" in ans.answer

    def test_comparison_answer_names_both_classes(self, optical):
        ans = answer_question("is there more water or vegetation", optical)
        assert ans.intent is VqaIntent.COMPARISON
        assert "water" in ans.answer.lower() and "vegetation" in ans.answer.lower()
        assert "%" in ans.answer

    def test_count_question_is_redirected_to_grounding(self, optical):
        ans = answer_question("how many buildings are there", optical)
        assert ans.intent is VqaIntent.COUNT
        assert ans.redirect == "grounding"
        assert "how much, not how many" in ans.answer

    def test_where_question_is_redirected_to_grounding(self, optical):
        ans = answer_question("where is the water", optical)
        assert ans.intent is VqaIntent.LOCATION
        assert ans.redirect == "grounding"

    def test_evidence_discloses_how_the_question_was_read(self, optical):
        ans = answer_question("how much vegetation is there", optical)
        joined = " ".join(ans.evidence).lower()
        assert "quantity query" in joined  # the intent is disclosed, not hidden
        assert "separability" in joined  # the classifier evidence is disclosed

    def test_to_dict_is_json_safe_and_omits_the_class_map_array(self, optical):
        ans = answer_question("what land cover is in this image", optical)
        payload = json.loads(json.dumps(ans.to_dict()))  # raises if a numpy array leaked
        assert payload["intent"] == "composition"
        assert "class_map" not in payload["landcover"]  # the array is kept out of the payload
        assert payload["answer"] == ans.answer

    def test_reuses_a_supplied_classification_rather_than_reclassifying(self, optical):
        from app.services.landcover import classify_land_cover

        lc = classify_land_cover(optical)
        ans = answer_question("how much water is there", optical, landcover_result=lc)
        assert ans.landcover is lc  # the passed-in result is used, not a fresh one


# ---------------------------------------------------------------------------
# Refusals (correct behaviour, brief §9, §28)
# ---------------------------------------------------------------------------
class TestRefusals:
    def test_empty_query_is_rejected(self, optical):
        with pytest.raises(ValidationError) as ei:
            answer_question("   ", optical)
        assert ei.value.code == ErrorCode.EMPTY_QUERY

    def test_overlong_query_is_rejected(self, optical):
        with pytest.raises(ValidationError) as ei:
            answer_question("water " * 400, optical)
        assert ei.value.code == ErrorCode.QUERY_TOO_LONG

    def test_spatial_relation_is_refused(self, optical):
        with pytest.raises(UnsupportedTaskError) as ei:
            answer_question("buildings near the river", optical)
        assert "spatial relations" in ei.value.message

    def test_undetectable_object_is_refused_by_name(self, optical):
        with pytest.raises(UnsupportedTaskError) as ei:
            answer_question("how many ships are in the harbour", optical)
        assert "ships" in ei.value.message
        assert "water, vegetation, built-up land and bare soil" in ei.value.message

    def test_a_question_naming_no_land_cover_type_is_refused(self, optical):
        with pytest.raises(UnsupportedTaskError) as ei:
            answer_question("how many elephants are there", optical)
        assert "water, vegetation, built-up land and bare soil" in ei.value.message

    @pytest.mark.parametrize(
        "query",
        ["buildings near the river", "how many ships are in the harbour", "show me the elephants"],
    )
    def test_vqa_refuses_exactly_what_grounding_refuses(self, optical, query):
        """The payoff of a shared vocabulary: a query grounding declines, VQA declines too."""
        with pytest.raises(UnsupportedTaskError):
            parse_grounding_query(query)
        with pytest.raises(UnsupportedTaskError):
            answer_question(query, optical)
