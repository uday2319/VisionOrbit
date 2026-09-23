"""Single-image visual question answering over land cover (brief §2A).

VQA here answers a natural-language question about *one* optical scene using the only evidence this
repo can actually produce: the deterministic land-cover classification in
:mod:`app.services.landcover`. A question is reduced — by keyword and vocabulary matching, never by
a language model — to one of a small set of intents that the class fractions can answer honestly:

* **presence**   — "is there water in this image?"        → yes/no with measured coverage
* **quantity**   — "how much of the scene is vegetation?"  → the class's coverage fraction
* **comparison** — "is there more water or built-up land?" → the two fractions, and which is larger
* **composition**— "what land cover is in this image?"     → the full measured breakdown

Everything outside that scope is declined rather than guessed at, and the refusals are the *same*
ones region grounding gives because both read the same vocabulary (:mod:`app.services.vocabulary`):

* an object class with no detector ("how many ships") is refused by name;
* a spatial relation ("buildings near the river") is refused as unevaluable;
* a *where* question, or one restricted to a named sector, is answered with the coverage figure and
  pointed at region grounding — the capability that actually returns regions — rather than
  re-implemented badly here;
* a *count of regions* ("how many lakes") returns how-much, not how-many, and says so: a distinct
  count is grounding's job, and grounding withholds it when it is not physically reliable.

Every measurable answer quotes a percentage that came from the classifier. No number in a VQA answer
is asserted that :class:`~app.services.landcover.LandCoverResult` does not contain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..config import get_settings
from ..core.errors import ErrorCode, UnsupportedTaskError, ValidationError
from ..core.types import LandCoverClass
from ..geospatial.raster import RasterData
from .landcover import ClassStats, LandCoverResult, classify_land_cover
from .vocabulary import (
    SPATIAL_RELATIONS,
    UNSUPPORTED_OBJECTS,
    find_phrase,
    normalise_query,
    targets_in,
)


class VqaIntent(StrEnum):
    """What kind of question the parser reduced the query to.

    Every member maps to something the land-cover fractions can answer or honestly redirect;
    there is deliberately no "object detection" or "spatial" intent, because there is no evidence
    behind those and inventing one is exactly what the brief forbids (§28, §44).
    """

    PRESENCE = "presence"
    QUANTITY = "quantity"
    COMPARISON = "comparison"
    COMPOSITION = "composition"
    COUNT = "count"
    """A count-of-regions question about a supported class: answered as coverage, redirected for
    the count itself."""
    LOCATION = "location"
    """A *where* / sector question: answered as coverage, redirected to grounding for the regions."""


# Interrogative cues, matched whole-word against the normalised query. Kept separate from the
# grounding count/area phrases because VQA draws a line grounding does not need: "how much"
# (coverage, answerable here) versus "how many" (a count of regions, grounding's job). A bare
# target with none of these cues ("is there water", or just "water") is treated as a presence
# question — the natural default — so there is no separate presence-cue list to keep in step.
_QUANTITY_CUES: tuple[str, ...] = (
    "how much", "what percentage", "what fraction", "what proportion", "what share",
    "percentage", "percent", "proportion", "fraction", "coverage", "how large", "how big",
    "area of", "extent of", "extent", "how dominant",
)
_COUNT_CUES: tuple[str, ...] = (
    "how many", "number of", "count of", "count the", "how many are", "how numerous",
)
_COMPOSITION_CUES: tuple[str, ...] = (
    "describe", "description", "what is in", "what's in", "whats in", "what land cover",
    "land cover", "landcover", "what types", "what kind", "what kinds", "overview",
    "summarise", "summarize", "summary", "composition", "dominant", "predominant", "main",
    "majority", "what does this image show", "what do you see", "what can you see", "breakdown",
    "characterise", "characterize",
)
# *Where* / sector cues. The sector words overlap ordinary language, so they are matched
# whole-word (via :func:`find_phrase`) against a copy of the query with the recognised target
# phrases blanked out — otherwise "rooftop" would read as the "top" sector.
_LOCATION_CUES: tuple[str, ...] = (
    "where", "whereabouts", "which part", "what part", "which side", "which region",
    "which area", "located", "location", "location of", "position", "position of",
    "positioned", "situated", "north", "south", "east", "west", "northern", "southern",
    "eastern", "western", "top", "bottom", "left", "right", "centre", "center", "central",
    "middle", "corner", "upper", "lower", "quadrant",
)


def _pretty(label: LandCoverClass) -> str:
    """Human-readable class name, e.g. ``BUILT_UP`` -> ``built up``."""
    return label.value.replace("_", " ")


@dataclass
class VqaAnswer:
    """The answer to one land-cover question, with the measurements it rests on.

    Attributes:
        question / normalised: The raw query and its normalised form, so the answer can always be
            traced back to the words that produced it.
        intent: The :class:`VqaIntent` the parser reduced the query to.
        targets: The land-cover class(es) the question is about; empty for a whole-scene
            composition question.
        answer: A concise, deterministic sentence phrased entirely from measured class fractions.
        figures: The percentages actually cited, keyed by class value — the numbers behind the
            sentence, for a caller that wants them structured.
        landcover: The full :class:`LandCoverResult` the answer was computed from. Carried so the
            tool layer can attach evidence-based confidence (§27); kept off :meth:`to_dict` because
            it holds the class-map array.
        redirect: The capability better suited to the question (``"grounding"``) for the two
            redirect intents, else ``None``.
        evidence / warnings: The user-facing "why", and any classification caveats carried up.
    """

    question: str
    normalised: str
    intent: VqaIntent
    targets: list[LandCoverClass]
    answer: str
    figures: dict[str, float]
    landcover: LandCoverResult
    redirect: str | None = None
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent.value,
            "targets": [t.value for t in self.targets],
            "answer": self.answer,
            "figures": self.figures,
            "redirect": self.redirect,
            "landcover": self.landcover.to_dict(),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
        }


def answer_question(
    query: str, raster: RasterData, *, landcover_result: LandCoverResult | None = None
) -> VqaAnswer:
    """Answer a land-cover question about a single optical scene.

    Args:
        query: The natural-language question.
        raster: The optical scene to analyse.
        landcover_result: A pre-computed classification to reuse; when ``None`` the scene is
            classified here. Supplied by callers that have already run the classifier so it is
            not run twice.

    Raises:
        ValidationError: The query is empty or longer than the configured limit.
        UnsupportedTaskError: The query asks for an object class with no detector, a spatial
            relation, or names no land-cover type and is not a whole-scene description. These are
            refusals by design — the same ones :func:`app.services.grounding.parse_grounding_query`
            gives, because both read the same vocabulary.
    """
    text = _normalise_or_refuse(query)

    relation = find_phrase(text, SPATIAL_RELATIONS)
    if relation is not None:
        raise UnsupportedTaskError(
            "This assistant answers questions about the land cover in a single image; it cannot "
            f'evaluate spatial relations between objects, so "{relation[0]}" cannot be answered. '
            'Try asking about a land-cover type directly — for example "how much built-up land '
            'is in this image".',
            context={"relation": relation[0]},
        )

    unsupported = find_phrase(text, UNSUPPORTED_OBJECTS)
    if unsupported is not None:
        raise UnsupportedTaskError(
            f'There is no detector for "{unsupported[0]}" in this system, so it will not be '
            "guessed at from land cover. This assistant can answer questions about water, "
            "vegetation, built-up land and bare soil.",
            context={"requested": unsupported[0]},
        )

    intent, targets = _interpret(text)
    if intent is None:
        raise UnsupportedTaskError(
            "That question does not name a land-cover type this assistant can measure. It can "
            "answer about water, vegetation, built-up land and bare soil — for example \"how "
            "much vegetation is in this image\" or \"is there any water\" — or describe the "
            "whole scene.",
            context={"normalised": text},
        )

    lc = landcover_result if landcover_result is not None else classify_land_cover(raster)
    answer, figures, redirect = _phrase(intent, targets, lc)
    return VqaAnswer(
        question=query,
        normalised=text,
        intent=intent,
        targets=targets,
        answer=answer,
        figures=figures,
        landcover=lc,
        redirect=redirect,
        evidence=_evidence(intent, targets, lc),
        warnings=list(lc.warnings),
    )


def _normalise_or_refuse(query: str) -> str:
    """Validate query length and return its normalised form, or raise like grounding does."""
    if not query or not query.strip():
        raise ValidationError(
            "Please enter a question about the image.", code=ErrorCode.EMPTY_QUERY
        )
    limit = get_settings().max_query_length
    if len(query) > limit:
        raise ValidationError(
            f"That question is too long. Please keep it under {limit} characters.",
            code=ErrorCode.QUERY_TOO_LONG,
            context={"length": len(query)},
        )
    return normalise_query(query)


def _interpret(text: str) -> tuple[VqaIntent | None, list[LandCoverClass]]:
    """Reduce a normalised query to an intent and the ordered, de-duplicated target classes.

    Returns ``(None, [])`` when the query names no target and is not a whole-scene description —
    the caller turns that into an :class:`UnsupportedTaskError`. Precedence is deliberate:
    a count-of-regions or *where* question about a real class is redirected before it can be
    mistaken for a plain coverage query, and a comparison needs two distinct classes.
    """
    matches = targets_in(text)
    # Preserve first-appearance order while de-duplicating to distinct classes.
    ordered: list[LandCoverClass] = []
    for _, label, _ in sorted(matches, key=lambda m: m[2]):
        if label not in ordered:
            ordered.append(label)

    blanked = _blank_targets(text, matches)
    has_count = find_phrase(text, _COUNT_CUES) is not None
    has_quantity = find_phrase(text, _QUANTITY_CUES) is not None
    has_location = find_phrase(blanked, _LOCATION_CUES) is not None
    has_composition = find_phrase(text, _COMPOSITION_CUES) is not None

    if not ordered:
        # No land-cover type named. A whole-scene description is still answerable; anything else
        # is outside scope and refused by the caller.
        return (VqaIntent.COMPOSITION, []) if has_composition else (None, [])

    if len(ordered) == 1:
        target = ordered[0]
        if has_count and not has_quantity:
            return VqaIntent.COUNT, [target]
        if has_location:
            return VqaIntent.LOCATION, [target]
        if has_quantity:
            return VqaIntent.QUANTITY, [target]
        if has_composition:
            # "describe the vegetation" — how much of that class there is.
            return VqaIntent.QUANTITY, [target]
        return VqaIntent.PRESENCE, [target]

    # Two or more distinct classes named. A count/where question over several classes has no
    # single subject, and a bare "water and vegetation" is still most usefully answered by
    # reporting each one's coverage, so every multi-class query resolves to a comparison.
    return VqaIntent.COMPARISON, ordered


def _blank_targets(text: str, matches: list[tuple[str, LandCoverClass, int]]) -> str:
    """Blank the recognised target phrases so sector cues are not read out of a class name.

    "rooftop" contains "top" and "eastern farmland" contains "east"; whole-word matching already
    stops the former, but blanking the target spans first makes the location scan robust to any
    class phrase that embeds a direction word.
    """
    chars = list(text)
    for phrase, _, start in matches:
        for i in range(start, min(start + len(phrase), len(chars))):
            chars[i] = " "
    return "".join(chars)


def _coverage(lc: LandCoverResult, label: LandCoverClass) -> tuple[float, ClassStats | None]:
    """The coverage fraction and stats for ``label``; ``(0.0, None)`` when the class is absent."""
    for stat in lc.stats:
        if stat.label is label:
            return stat.fraction, stat
    return 0.0, None


def _extent_phrase(pct: float, stat: ClassStats | None) -> str:
    """``"37% of the classified scene"``, adding a measured area only when one is available."""
    extent = f"{pct:.0f}% of the classified scene"
    if stat is not None and stat.area_m2 is not None:
        extent += f" (about {stat.area_m2 / 1e6:.2f} km²)"
    return extent


def _phrase(
    intent: VqaIntent, targets: list[LandCoverClass], lc: LandCoverResult
) -> tuple[str, dict[str, float], str | None]:
    """Compose the answer sentence, the cited figures, and any redirect, from measured fractions.

    Pure over ``lc`` so each branch is unit-testable with a stub result. Every branch that has a
    measurement to report puts a percentage in the sentence; a branch with nothing to report
    (a class that is absent) says so plainly rather than quoting a fabricated figure.
    """
    if intent is VqaIntent.COMPOSITION:
        return _phrase_composition(lc)
    if intent is VqaIntent.COMPARISON:
        return _phrase_comparison(targets, lc)

    # The remaining intents are all single-target.
    target = targets[0]
    frac, stat = _coverage(lc, target)
    pct = frac * 100
    name = _pretty(target)

    if intent is VqaIntent.PRESENCE:
        if frac == 0:
            return f"No, no {name} was detected in this image.", {}, None
        return (
            f"Yes, {name} is present, covering {_extent_phrase(pct, stat)}.",
            {target.value: round(pct, 2)},
            None,
        )

    if intent is VqaIntent.QUANTITY:
        if frac == 0:
            return f"No {name} was detected in this image (0% coverage).", {}, None
        return (
            f"{name.capitalize()} covers {_extent_phrase(pct, stat)}.",
            {target.value: round(pct, 2)},
            None,
        )

    if intent is VqaIntent.COUNT:
        if frac == 0:
            return (
                f"No {name} was detected in this image, so there are none to count.", {}, None
            )
        return (
            f"{name.capitalize()} covers {_extent_phrase(pct, stat)}. This answers how much, not "
            f"how many: counting distinct {name} regions is handled by the region-grounding "
            "capability, which reports a count only when it is physically reliable.",
            {target.value: round(pct, 2)},
            "grounding",
        )

    # VqaIntent.LOCATION
    if frac == 0:
        return (
            f"No {name} was detected in this image, so there is no location to report.", {}, None
        )
    return (
        f"{name.capitalize()} covers {_extent_phrase(pct, stat)}. To see *where* it lies — as "
        f'regions, sectors or areas — use region grounding (for example "where is the {name}").',
        {target.value: round(pct, 2)},
        "grounding",
    )


def _phrase_composition(
    lc: LandCoverResult,
) -> tuple[str, dict[str, float], str | None]:
    """Whole-scene breakdown, naming the dominant class and the measured share of each."""
    dominant = lc.dominant()
    if dominant is None:
        return (
            "No land-cover classes could be delineated from this image, so its composition "
            "cannot be described.",
            {},
            None,
        )
    # `stats` is already sorted by descending fraction (see landcover._class_stats).
    breakdown = ", ".join(f"{_pretty(s.label)} {s.fraction * 100:.0f}%" for s in lc.stats)
    figures = {s.label.value: round(s.fraction * 100, 2) for s in lc.stats}
    return (
        f"The scene is predominantly {_pretty(dominant.label)} "
        f"({dominant.fraction * 100:.0f}% of classified pixels). "
        f"Full land-cover breakdown: {breakdown}.",
        figures,
        None,
    )


def _phrase_comparison(
    targets: list[LandCoverClass], lc: LandCoverResult
) -> tuple[str, dict[str, float], str | None]:
    """Report each named class's coverage and, when they differ, which covers more."""
    measured = sorted(
        ((t, _coverage(lc, t)[0]) for t in targets), key=lambda p: p[1], reverse=True
    )
    listing = ", ".join(f"{_pretty(t)} {f * 100:.0f}%" for t, f in measured)
    figures = {t.value: round(f * 100, 2) for t, f in measured}
    top, top_frac = measured[0]
    second, second_frac = measured[1]
    if top_frac == second_frac:
        verdict = (
            f"{_pretty(top).capitalize()} and {_pretty(second)} cover the scene about equally"
        )
    else:
        verdict = f"{_pretty(top).capitalize()} covers more of the scene than {_pretty(second)}"
    return f"{verdict} ({listing}).", figures, None


def _evidence(
    intent: VqaIntent, targets: list[LandCoverClass], lc: LandCoverResult
) -> list[str]:
    """The user-facing "why": how the question was read and what the classifier measured."""
    subject = ", ".join(_pretty(t) for t in targets) if targets else "the whole scene"
    lines = [
        f"Question read as a {intent.value} query about {subject}.",
        f"Answered from land-cover classification (method: {lc.method}; mean class separability "
        f"{lc.separability:.2f}).",
    ]
    dominant = lc.dominant()
    if dominant is not None:
        lines.append(
            f"Dominant class is {_pretty(dominant.label)}, covering "
            f"{dominant.fraction * 100:.1f}% of valid pixels."
        )
    if lc.indices_used:
        lines.append(f"Spectral indices used: {', '.join(lc.indices_used)}.")
    lines.extend(lc.warnings)
    return lines
