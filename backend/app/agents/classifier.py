"""Query classifier: natural-language query -> :class:`QueryTask` (brief §5, §9).

This is the first stage of the agentic pipeline (§5): classifier -> router -> tools ->
evidence validator -> result integrator. Its single job is to read the *text* of a user's
question and decide *what kind of analysis* is being asked for. It deliberately does **not**
look at the images — whether a change query is answerable depends on there being two inputs,
and that reconciliation is the router's job (:mod:`app.agents.router`), not the classifier's.

Three commitments shape the design:

1. **Deterministic and explainable, never a language model.** Classification is weighted
   keyword/phrase matching over a small closed set of tasks. Every decision points at the exact
   terms that produced it (:attr:`Classification.matched_terms`), which is what the execution
   trace shows the user — no opaque "the model thought so". The ``score`` is a *routing signal*
   (total matched-term weight), not an analysis confidence; genuine evidence-based confidence is
   computed later and separately by :mod:`app.services.confidence` (§27), and the two must not be
   conflated.

2. **``UNKNOWN`` is a first-class, honest outcome (§9, §28).** A query with no clear task signal
   resolves to :attr:`QueryTask.UNKNOWN` rather than being forced into the nearest bucket. The
   router then either applies an input-aware default (one optical image + no task word -> a
   land-cover overview) or asks the user to clarify. Guessing a workflow for a vague query is the
   failure mode this avoids.

3. **Recognising a task is not the same as being able to do it.** ``OBJECT_DETECTION`` and
   ``SEGMENTATION`` are recognised here because a user can legitimately ask for them, but no tool
   implements them; the registry (§18) turns that into an explicit refusal with a helpful
   redirect rather than a fabricated bounding box. Classifying them honestly is what lets the
   refusal be specific.

Change carries one extra step. ``CHANGE_DETECTION`` and ``CHANGE_VQA`` are the same underlying
analysis asked in two grammatical moods — "show the change" (produce a map) versus "what changed?"
(answer a question) — so they share a single lexicon and are told apart by whether the query is
interrogative, rather than by two overlapping keyword sets that would fight each other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..core.logging import get_logger
from ..core.types import QueryTask

logger = get_logger(__name__)

# A winning task must clear this total weight to be recognised. A single ambiguous token
# (weight 1, e.g. a stray "difference" in "spot the difference") is deliberately *not* enough — it
# leaves the result UNKNOWN so the router can default from the inputs instead of over-claiming.
# A domain task word (weight >= 2) or two independent weak signals clear the bar.
_MIN_SCORE = 2

# Task lexicons. Each entry is ``(phrase, weight)`` where the phrase is already normalised
# (lowercase, punctuation collapsed to single spaces — see :func:`_normalize`). Weights encode
# specificity, not frequency: a multi-word intent phrase ("detect change") is worth more than the
# bare noun ("change") it contains, so the specific phrasing wins when both match. Overlap between
# tasks is expected and fine — the argmax resolves it, and genuine ties fall back to
# :data:`_TIE_PRIORITY`.
#
# CHANGE_VQA is intentionally absent: change queries all score under CHANGE_DETECTION and are
# relabelled to CHANGE_VQA in :func:`classify` when the phrasing is a question.
_LEXICON: dict[QueryTask, tuple[tuple[str, int], ...]] = {
    QueryTask.CHANGE_DETECTION: (
        ("detect change", 3),
        ("detect changes", 3),
        ("change detection", 3),
        ("change map", 3),
        ("show the change", 3),
        ("show the changes", 3),
        ("show me the change", 3),
        ("show changes", 3),
        ("what changed", 3),
        ("what has changed", 3),
        ("what is different", 3),
        ("what are the differences", 3),
        ("what changes", 3),
        ("what changes occurred", 3),
        ("how much changed", 3),
        ("how much has changed", 3),
        ("find change", 2),
        ("find the change", 2),
        ("identify change", 2),
        ("map the change", 2),
        ("before and after", 3),
        ("over time", 2),
        ("between the two dates", 3),
        ("between the two", 2),
        ("between these two", 2),
        ("bitemporal", 3),
        ("bi temporal", 3),
        ("time series", 2),
        ("deforestation", 2),
        ("urban growth", 2),
        ("urban expansion", 2),
        ("compare", 2),
        ("change", 2),
        ("changed", 2),
        ("changes", 2),
        ("difference", 1),
        ("different", 1),
        ("increased", 2),
        ("decreased", 2),
        ("increase", 2),
        ("decrease", 2),
        ("grown", 2),
        ("shrunk", 2),
    ),
    QueryTask.GROUNDING: (
        ("where is", 3),
        ("where are", 3),
        ("where can i find", 3),
        ("where can i see", 3),
        ("locate", 3),
        ("show me the", 2),
        ("show me where", 3),
        ("highlight", 3),
        ("point to", 3),
        ("point out", 3),
        ("mark the", 2),
        ("find the", 1),
        ("which part", 2),
        ("what part", 2),
        ("outline", 2),
        ("delineate", 2),
        ("extent of", 2),
        ("region", 1),
        ("regions", 1),
        ("how many", 2),
        ("in the north", 1),
        ("in the south", 1),
        ("in the east", 1),
        ("in the west", 1),
        ("north east", 1),
        ("north west", 1),
        ("south east", 1),
        ("south west", 1),
        ("top left", 1),
        ("top right", 1),
        ("bottom left", 1),
        ("bottom right", 1),
    ),
    QueryTask.LAND_COVER_ANALYSIS: (
        ("land cover", 3),
        ("landcover", 3),
        ("land use", 3),
        ("classify the land", 3),
        ("what land cover", 3),
        ("classify", 2),
        ("classification", 2),
        ("what covers", 2),
        ("surface types", 2),
        ("types of terrain", 2),
        ("map the land", 2),
        ("what is the terrain", 2),
    ),
    QueryTask.OPTICAL_SAR_ANALYSIS: (
        ("optical and sar", 3),
        ("sar and optical", 3),
        ("optical sar", 3),
        ("combine optical", 3),
        ("cross modal", 3),
        ("fuse", 3),
        ("fusion", 3),
        ("radar", 2),
        ("sar", 2),
        ("backscatter", 2),
        ("scattering", 2),
        ("sentinel 1", 2),
        ("polarization", 2),
        ("polarisation", 2),
        ("microwave", 2),
    ),
    QueryTask.VQA: (
        ("what can you see", 3),
        ("what do you see", 3),
        ("what does the image show", 3),
        ("what is in the image", 3),
        ("what is in this image", 3),
        ("is there", 2),
        ("are there", 2),
        ("does the image", 2),
        ("does this image", 2),
        ("what type of", 2),
        ("what kind of", 2),
        ("can you see", 2),
        ("what is this", 2),
        ("tell me what", 1),
        ("what is the", 1),
        ("how much", 1),
    ),
    QueryTask.CAPTIONING: (
        ("caption", 3),
        ("generate a caption", 3),
        ("describe the image", 3),
        ("describe this image", 3),
        ("describe the scene", 3),
        ("tell me about this image", 3),
        ("what does this image show", 2),
        ("summarize the image", 3),
        ("summarise the image", 3),
        ("summarize", 2),
        ("summarise", 2),
        ("overview", 2),
        ("describe what", 2),
        ("describe", 1),
    ),
    QueryTask.OBJECT_DETECTION: (
        ("detect objects", 3),
        ("object detection", 3),
        ("detect all", 2),
        ("bounding box", 3),
        ("bounding boxes", 3),
        ("count the ships", 3),
        ("count the cars", 3),
        ("count the vehicles", 3),
        ("count the vessels", 3),
        ("count the planes", 3),
        ("count the aircraft", 3),
        ("vehicles", 2),
        ("cars", 2),
        ("ships", 2),
        ("boats", 2),
        ("vessels", 2),
        ("aircraft", 2),
        ("airplanes", 2),
        ("airplane", 2),
        ("planes", 2),
    ),
    QueryTask.SEGMENTATION: (
        ("segment", 3),
        ("segmentation", 3),
        ("semantic segmentation", 3),
        ("pixel mask", 2),
        ("pixel level mask", 3),
        ("mask of", 2),
    ),
    QueryTask.REPORT_GENERATION: (
        ("generate a report", 3),
        ("generate report", 3),
        ("full report", 3),
        ("analysis report", 3),
        ("produce a report", 3),
        ("create a report", 3),
        ("pdf report", 3),
        ("report", 2),
    ),
}

# Tie-break order for equal scores: more specific / less ambiguous tasks first, so an exact
# score tie resolves toward the more concrete interpretation. Must contain exactly the lexicon
# keys (enforced by test); CHANGE_VQA and UNKNOWN are never scored directly and so are absent.
_TIE_PRIORITY: tuple[QueryTask, ...] = (
    QueryTask.CHANGE_DETECTION,
    QueryTask.OPTICAL_SAR_ANALYSIS,
    QueryTask.GROUNDING,
    QueryTask.OBJECT_DETECTION,
    QueryTask.SEGMENTATION,
    QueryTask.LAND_COVER_ANALYSIS,
    QueryTask.CAPTIONING,
    QueryTask.REPORT_GENERATION,
    QueryTask.VQA,
)

# Words that, leading a query, make it a question. Used only to split a change query into the
# "answer a question" (CHANGE_VQA) versus "produce a map" (CHANGE_DETECTION) mood.
_QUESTION_WORDS = frozenset(
    {
        "what", "how", "did", "does", "do", "is", "are", "has", "have", "was",
        "were", "which", "why", "where", "when", "can", "could", "will",
        "would", "should",
    }
)

_NON_TOKEN = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    """Lowercase and collapse every run of non-alphanumeric characters to a single space.

    This is what makes ``"Built-up?"``, ``"built up"`` and ``"BUILT_UP"`` match the same phrase,
    and it is applied identically to the query and (at import time) to every lexicon phrase so the
    two can never drift.
    """
    return _NON_TOKEN.sub(" ", text.lower()).strip()


def _is_interrogative(query: str, norm: str) -> bool:
    """Whether the query reads as a question.

    True if it contains a question mark or opens with an interrogative word. This decides change
    mood only; it is never used to invent a task where none was matched.
    """
    if "?" in query:
        return True
    if not norm:
        return False
    return norm.split(" ", 1)[0] in _QUESTION_WORDS


def _compile() -> dict[QueryTask, tuple[tuple[re.Pattern[str], str, int], ...]]:
    """Pre-compile each lexicon phrase into a word-boundaried regex.

    Word boundaries matter: without them ``"sar"`` would match "sardine" and ``"change"`` would
    match "exchange". Phrases are normalised first so a stray hyphen in the lexicon cannot produce
    a pattern that never matches normalised input.
    """
    compiled: dict[QueryTask, tuple[tuple[re.Pattern[str], str, int], ...]] = {}
    for task, phrases in _LEXICON.items():
        entries: list[tuple[re.Pattern[str], str, int]] = []
        for phrase, weight in phrases:
            norm = _normalize(phrase)
            pattern = re.compile(rf"\b{re.escape(norm)}\b")
            entries.append((pattern, norm, weight))
        compiled[task] = tuple(entries)
    return compiled


_COMPILED = _compile()


@dataclass(frozen=True)
class Classification:
    """The classifier's verdict for one query.

    Attributes:
        task: The chosen :class:`QueryTask`, or :attr:`QueryTask.UNKNOWN` when no task cleared
            :data:`_MIN_SCORE`. May be :attr:`QueryTask.CHANGE_VQA`, which is derived by mood from
            a :attr:`QueryTask.CHANGE_DETECTION` match rather than scored directly.
        score: Total matched-term weight for the winning task. A routing signal only — **not** an
            analysis confidence (that is computed by :mod:`app.services.confidence`).
        matched_terms: The exact normalised phrases that fired for the winning task, in lexicon
            order. This is the evidence surfaced in the execution trace.
        ranked: Every task that scored above zero, best first, as ``(task, score)`` — the raw
            match before mood resolution, so a caller can see the runners-up. The router consults
            these to disambiguate against the available inputs (e.g. prefer
            ``OPTICAL_SAR_ANALYSIS`` when a SAR image is actually present).
    """

    task: QueryTask
    score: float
    matched_terms: tuple[str, ...] = ()
    ranked: tuple[tuple[QueryTask, float], ...] = ()

    @property
    def is_recognized(self) -> bool:
        """Whether a concrete task was identified from the text alone."""
        return self.task is not QueryTask.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "score": self.score,
            "recognized": self.is_recognized,
            "matched_terms": list(self.matched_terms),
            "alternatives": [
                {"task": t.value, "score": s} for t, s in self.ranked[1:4]
            ],
        }


def classify(query: str) -> Classification:
    """Classify a natural-language query into a :class:`QueryTask`.

    Returns :attr:`QueryTask.UNKNOWN` (with an empty ``matched_terms``) when the text carries no
    task signal that clears :data:`_MIN_SCORE` — including for an empty or whitespace-only query.
    Deciding *what* to do with UNKNOWN belongs to the router, which can see the inputs; the
    classifier's contract is only to report honestly what the words alone support.
    """
    norm = _normalize(query)
    if not norm:
        return Classification(QueryTask.UNKNOWN, 0.0)

    scores: dict[QueryTask, int] = {}
    matched: dict[QueryTask, list[str]] = {}
    for task, entries in _COMPILED.items():
        total = 0
        terms: list[str] = []
        for pattern, phrase, weight in entries:
            if pattern.search(norm):
                total += weight
                terms.append(phrase)
        if total > 0:
            scores[task] = total
            matched[task] = terms

    if not scores:
        return Classification(QueryTask.UNKNOWN, 0.0)

    ranked = sorted(
        scores.items(), key=lambda kv: (-kv[1], _TIE_PRIORITY.index(kv[0]))
    )
    best_task, best_score = ranked[0]
    ranked_tuple = tuple((t, float(s)) for t, s in ranked)

    if best_score < _MIN_SCORE:
        # A signal exists but is too weak to claim a task; stay honest and let the router decide.
        logger.debug(
            "query classified UNKNOWN (below threshold)",
            extra={"top_task": best_task.value, "score": best_score},
        )
        return Classification(QueryTask.UNKNOWN, float(best_score), (), ranked_tuple)

    winner_terms = tuple(matched[best_task])

    # Change mood: the same detector, asked as a question, is a CHANGE_VQA rather than a request
    # for a change map. The evidence (matched terms, score) is unchanged — only the label.
    if best_task is QueryTask.CHANGE_DETECTION and _is_interrogative(query, norm):
        best_task = QueryTask.CHANGE_VQA

    return Classification(
        task=best_task,
        score=float(best_score),
        matched_terms=winner_terms,
        ranked=ranked_tuple,
    )
