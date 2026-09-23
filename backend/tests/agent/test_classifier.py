"""Tests for the query classifier (brief §5, §9).

The classifier is deterministic keyword/phrase matching, so these tests are exhaustive about
behaviour rather than statistical: a fixed query has one correct task. Three things are held —
that realistic phrasings land on the right task, that a query with no real signal stays
``UNKNOWN`` instead of being forced into a bucket (§9, §28), and that the evidence the trace will
show (matched terms, ranked alternatives) is actually populated.
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from app.agents import classifier
from app.agents.classifier import Classification, classify
from app.core.types import QueryTask


# ---------------------------------------------------------------------------
# Lexicon integrity — cheap invariants that prevent whole classes of silent bug
# ---------------------------------------------------------------------------
class TestLexiconIntegrity:
    def test_every_phrase_is_already_normalised(self):
        """A lexicon phrase that isn't in normalised form compiles to a regex that never matches."""
        for task, phrases in classifier._LEXICON.items():
            for phrase, _weight in phrases:
                assert phrase == classifier._normalize(phrase), (task, phrase)

    def test_tie_priority_covers_exactly_the_scored_tasks(self):
        """Every lexicon key must appear in the tie-break order, and nothing else may."""
        assert set(classifier._TIE_PRIORITY) == set(classifier._LEXICON)
        assert len(classifier._TIE_PRIORITY) == len(set(classifier._TIE_PRIORITY))

    def test_change_vqa_is_never_scored_directly(self):
        """CHANGE_VQA is derived by mood, so it must not be a lexicon or tie-break entry."""
        assert QueryTask.CHANGE_VQA not in classifier._LEXICON
        assert QueryTask.CHANGE_VQA not in classifier._TIE_PRIORITY

    def test_all_weights_are_positive(self):
        for phrases in classifier._LEXICON.values():
            for _phrase, weight in phrases:
                assert weight >= 1


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Built-up?", "built up"),
            ("BUILT_UP", "built up"),
            ("Sentinel-1!", "sentinel 1"),
            ("  where   is  the water  ", "where is the water"),
            ("optical & SAR", "optical sar"),
            ("", ""),
            ("   ", ""),
            ("!!!", ""),
        ],
    )
    def test_normalize(self, raw, expected):
        assert classifier._normalize(raw) == expected


# ---------------------------------------------------------------------------
# Task recognition — one representative set per task
# ---------------------------------------------------------------------------
class TestTaskRecognition:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            # VQA — questions about content answered in text
            ("What can you see in this image?", QueryTask.VQA),
            ("Is there water present?", QueryTask.VQA),
            ("What kind of surface is shown?", QueryTask.VQA),
            # Captioning — describe / summarise the whole scene
            ("Describe the image.", QueryTask.CAPTIONING),
            ("Generate a caption for the scene.", QueryTask.CAPTIONING),
            ("Give an overview of the scene.", QueryTask.CAPTIONING),
            ("Summarise the image.", QueryTask.CAPTIONING),
            # Grounding — locate / show / highlight a region
            ("Where is the water?", QueryTask.GROUNDING),
            ("Highlight the built-up areas.", QueryTask.GROUNDING),
            ("Show me where the vegetation is.", QueryTask.GROUNDING),
            ("Locate the buildings in the north-east.", QueryTask.GROUNDING),
            ("Outline the forest extent.", QueryTask.GROUNDING),
            # Land cover — classification of the surface
            ("Classify the land cover.", QueryTask.LAND_COVER_ANALYSIS),
            ("What land cover is present?", QueryTask.LAND_COVER_ANALYSIS),
            ("Map the land use here.", QueryTask.LAND_COVER_ANALYSIS),
            ("Perform land-use classification.", QueryTask.LAND_COVER_ANALYSIS),
            # Optical + SAR — anything invoking radar / fusion
            ("Combine the optical and SAR images.", QueryTask.OPTICAL_SAR_ANALYSIS),
            ("Analyse the radar backscatter.", QueryTask.OPTICAL_SAR_ANALYSIS),
            ("Fuse optical and SAR data.", QueryTask.OPTICAL_SAR_ANALYSIS),
            ("What does the SAR show?", QueryTask.OPTICAL_SAR_ANALYSIS),
            # Change detection — imperative, produce a map
            ("Detect the change between the two images.", QueryTask.CHANGE_DETECTION),
            ("Show me the change map.", QueryTask.CHANGE_DETECTION),
            ("Change detection between these two dates.", QueryTask.CHANGE_DETECTION),
            ("Detect deforestation over time.", QueryTask.CHANGE_DETECTION),
            # Change VQA — interrogative, answer a question
            ("What changed between the two images?", QueryTask.CHANGE_VQA),
            ("What is different here?", QueryTask.CHANGE_VQA),
            ("How much has the built-up area grown?", QueryTask.CHANGE_VQA),
            ("Has the water body shrunk?", QueryTask.CHANGE_VQA),
            # Object detection — recognised though unimplemented (honest refusal downstream)
            ("Detect all objects.", QueryTask.OBJECT_DETECTION),
            ("Count the ships in the harbour.", QueryTask.OBJECT_DETECTION),
            ("Draw bounding boxes around vehicles.", QueryTask.OBJECT_DETECTION),
            # Segmentation — recognised though unimplemented
            ("Segment the image into regions.", QueryTask.SEGMENTATION),
            ("Perform semantic segmentation.", QueryTask.SEGMENTATION),
            # Report
            ("Generate a full analysis report.", QueryTask.REPORT_GENERATION),
            ("Create a PDF report.", QueryTask.REPORT_GENERATION),
        ],
    )
    def test_query_classifies_as(self, query, expected):
        result = classify(query)
        assert result.task is expected, (query, result.ranked)
        assert result.is_recognized
        assert result.score >= classifier._MIN_SCORE


# ---------------------------------------------------------------------------
# UNKNOWN — the honest no-signal outcome
# ---------------------------------------------------------------------------
class TestUnknown:
    @pytest.mark.parametrize(
        "query",
        ["", "   ", "!!!", "asdfjkl qwerty", "Please help me with this."],
    )
    def test_no_signal_is_unknown_with_no_evidence(self, query):
        result = classify(query)
        assert result.task is QueryTask.UNKNOWN
        assert not result.is_recognized
        assert result.matched_terms == ()

    @pytest.mark.parametrize("query", ["difference", "region", "different"])
    def test_a_single_weak_token_stays_below_threshold(self, query):
        """One ambiguous word is a signal but not a claim — it must not name a task on its own."""
        result = classify(query)
        assert result.task is QueryTask.UNKNOWN
        # The signal is still reported in ranked, so the router can see what was almost matched.
        assert result.ranked
        assert result.score < classifier._MIN_SCORE


# ---------------------------------------------------------------------------
# Word boundaries — substrings must not trigger a task
# ---------------------------------------------------------------------------
class TestWordBoundaries:
    def test_exchange_does_not_match_change(self):
        assert classify("the exchange rate today").task is QueryTask.UNKNOWN

    def test_sardines_does_not_match_sar(self):
        assert classify("a can of sardines").task is QueryTask.UNKNOWN

    def test_a_weak_word_does_not_outrank_a_real_phrase(self):
        """"different" (weak change token) must not hijack a land-cover query."""
        result = classify("different types of terrain")
        assert result.task is QueryTask.LAND_COVER_ANALYSIS


# ---------------------------------------------------------------------------
# Change mood — same detector, question vs command
# ---------------------------------------------------------------------------
class TestChangeMood:
    def test_imperative_change_is_detection(self):
        assert classify("show the change").task is QueryTask.CHANGE_DETECTION

    def test_interrogative_change_is_vqa(self):
        result = classify("what has changed?")
        assert result.task is QueryTask.CHANGE_VQA
        assert result.matched_terms  # evidence survives the relabel
        assert result.score >= classifier._MIN_SCORE

    def test_a_trailing_question_mark_flips_mood(self):
        assert classify("show the change").task is QueryTask.CHANGE_DETECTION
        assert classify("show the change?").task is QueryTask.CHANGE_VQA

    def test_ranked_reports_the_pre_mood_task(self):
        """`ranked` shows the raw match (CHANGE_DETECTION); `task` shows the mood-resolved label."""
        result = classify("what changed between the two images?")
        assert result.task is QueryTask.CHANGE_VQA
        assert result.ranked[0][0] is QueryTask.CHANGE_DETECTION

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("what?", True),
            ("change?", True),
            ("how much has it grown", True),
            ("show the change", False),
            ("detect change", False),
            ("", False),
        ],
    )
    def test_is_interrogative(self, query, expected):
        assert classifier._is_interrogative(query, classifier._normalize(query)) is expected


# ---------------------------------------------------------------------------
# Tie-breaking by priority
# ---------------------------------------------------------------------------
class TestTieBreak:
    def test_counting_prefers_grounding_over_vqa(self):
        """"how many" (grounding=2) ties "are there" (vqa=2); grounding has priority."""
        result = classify("how many water bodies are there?")
        assert result.task is QueryTask.GROUNDING

    def test_sar_prefers_optical_sar_over_land_cover(self):
        """"classify" (land-cover=2) ties "sar" (optical-sar=2); SAR family has priority."""
        result = classify("classify the sar")
        assert result.task is QueryTask.OPTICAL_SAR_ANALYSIS


# ---------------------------------------------------------------------------
# Evidence and serialisation
# ---------------------------------------------------------------------------
class TestEvidenceAndContract:
    def test_matched_terms_are_the_triggering_phrases(self):
        result = classify("where is the water")
        assert "where is" in result.matched_terms

    def test_ranked_is_sorted_by_descending_score(self):
        result = classify("classify the sar land cover")
        scores = [s for _t, s in result.ranked]
        assert scores == sorted(scores, reverse=True)

    def test_to_dict_is_json_serialisable(self):
        result = classify("detect the change between the two images")
        payload = json.loads(json.dumps(result.to_dict()))
        assert set(payload) == {
            "task", "score", "recognized", "matched_terms", "alternatives"
        }
        assert payload["task"] == result.task.value
        assert payload["recognized"] is True
        assert len(payload["alternatives"]) <= 3

    def test_alternatives_exclude_the_winner(self):
        result = classify("classify the sar land cover")
        payload = result.to_dict()
        winner = payload["task"]
        assert all(alt["task"] != winner for alt in payload["alternatives"])

    def test_frozen_dataclass(self):
        result = classify("where is the water")
        assert isinstance(result, Classification)
        with pytest.raises(FrozenInstanceError):
            result.task = QueryTask.VQA  # type: ignore[misc]
