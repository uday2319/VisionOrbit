"""Change question-answering, separated from change *detection* (brief §3, §17, §18; task point 7).

Both tasks rest on the same measurement, and until now the registry answered them with the same
tool by aliasing :attr:`QueryTask.CHANGE_VQA` onto the change-detection chain. That conflated three
different things the brief asks to be kept apart:

* **Raw change detection** — "how much of this scene differs, and where?" A per-pixel measurement.
  :class:`~app.agents.tools.change.ChangeTool` answers this, and its answer is a report.
* **Semantic land-cover transition** — "what became what?" An *interpretation* of that measurement,
  which :class:`~app.services.change.ChangeResult` now gates on registration quality and per-
  transition spectral corroboration rather than asserting whenever a transition exists.
* **Change VQA** — "did the water shrink?", "how much changed?" A *question*, whose answer is a
  direct response to what was asked and not a full report the user has to read to find their answer.

Sharing the measurement while separating the answer is why this subclasses the change tool: the
validation, alignment, co-registration and differencing are literally the same code, so there is no
second implementation to drift. What differs is only :meth:`postprocess` — which reads the question
and answers *it*, from the same measured quantities, drawing on the same semantic gate so a question
about a land-cover conversion cannot get a confident answer the change report would have withheld.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from ...core.types import LandCoverClass, QueryTask
from ...services import confidence
from .base import Prepared, ToolContext, ToolResult
from .change import ChangeTool, _pretty

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ...services.change import ChangeResult

# Words that name a land-cover class in a question. Only the four classes the classifier actually
# has a vocabulary for appear here: a question about a class the system cannot measure must fall
# through to the general answer rather than be answered about the nearest available class.
_CLASS_WORDS: tuple[tuple[LandCoverClass, tuple[str, ...]], ...] = (
    (LandCoverClass.WATER, ("water", "lake", "river", "reservoir", "flood", "sea", "pond")),
    (LandCoverClass.VEGETATION, ("vegetation", "vegetated", "forest", "tree", "crop", "green",
                                 "farm", "plant")),
    (LandCoverClass.BUILT_UP, ("built", "built-up", "urban", "city", "settlement", "construction",
                               "development", "building")),
    (LandCoverClass.BARE_SOIL, ("bare", "soil", "barren", "sand", "cleared", "dirt")),
)

# Questions about *how much*, which are answered from the changed extent rather than from any class.
_EXTENT_WORDS: tuple[str, ...] = (
    "how much", "how many", "how large", "what area", "extent", "percentage", "percent",
    "what fraction", "how big",
)


def _mentioned_class(query: str) -> LandCoverClass | None:
    """The land-cover class a question is about, or ``None`` when it names none of them.

    Word-boundary matching, because ``"bare"`` must not fire on ``"barely"`` and ``"green"`` must
    not fire on ``"greenhouse gas"``. First match in :data:`_CLASS_WORDS` order wins; a question
    naming two classes is answered about the first, and the full transition table is in the data
    either way.
    """
    lowered = query.lower()
    for cls, words in _CLASS_WORDS:
        for word in words:
            if re.search(rf"\b{re.escape(word)}", lowered):
                return cls
    return None


def _asks_extent(query: str) -> bool:
    lowered = query.lower()
    return any(phrase in lowered for phrase in _EXTENT_WORDS)


class ChangeVqaTool(ChangeTool):
    """Answer a question about what changed between two dates, from the measured change.

    Deliberately not a language model. Every clause of the answer is a formatted measurement from
    the :class:`~app.services.change.ChangeResult`, so there is nothing in the sentence that the
    analysis did not produce.
    """

    task: ClassVar[QueryTask] = QueryTask.CHANGE_VQA
    name: ClassVar[str] = "change-vqa-cv"
    summary: ClassVar[str] = (
        "Answers a question about change between two dates from the measured bi-temporal "
        "analysis — the changed extent, a specific land-cover class's net gain or loss, or the "
        "transitions the evidence supports naming."
    )

    def preprocess(self, ctx: ToolContext) -> Prepared:
        prepared = super().preprocess(ctx)
        # The question is needed in postprocess, which the base template does not hand the context.
        prepared.context["query"] = ctx.query
        return prepared

    def postprocess(self, raw: ChangeResult, prepared: Prepared) -> ToolResult:
        report = confidence.for_change(raw, prepared.quality)
        extra = prepared.context.get("warnings", [])
        warnings = list(dict.fromkeys([*raw.warnings, *extra]))
        data = raw.to_dict()
        coreg = prepared.context.get("coregistration")
        if coreg is not None:
            data["coregistration"] = coreg
        query = str(prepared.context.get("query", ""))
        answer, kind = self._answer_question(raw, query)
        # What the question was read as, so a misread question is visible in the result rather
        # than only in an answer that quietly addresses something else.
        data["question"] = {"asked": query, "interpreted_as": kind}
        return self.build_result(
            answer=answer,
            data=data,
            confidence=report,
            evidence=self.explain(raw),
            warnings=warnings,
            raw=raw,
        )

    @staticmethod
    def _answer_question(raw: ChangeResult, query: str) -> tuple[str, str]:
        """The answer, and which reading of the question produced it."""
        if raw.changed_pixels == 0:
            return (
                "No change large enough to be distinguished from image noise was detected "
                "between the two dates, so there is nothing measured to report for this "
                "question. This measures no detected change, not proof that nothing changed."
            ), "no-change"

        target = _mentioned_class(query)
        if target is not None:
            answer = ChangeVqaTool._answer_about_class(raw, target)
            if answer is not None:
                return answer, f"net change in {target.value}"

        if _asks_extent(query):
            return ChangeVqaTool._answer_extent(raw), "changed extent"

        return ChangeTool._summarize(raw), "general change summary"

    @staticmethod
    def _answer_about_class(raw: ChangeResult, target: LandCoverClass) -> str | None:
        """Net gain or loss for one class, or ``None`` when that class was not classified.

        Returning ``None`` matters: a question about water in a scene with no water class measured
        must fall through to the general answer, not be answered with a fabricated zero.
        """
        delta = next((d for d in raw.class_deltas if d.label is target), None)
        if delta is None:
            return None

        area_delta = (
            delta.area_m2_after - delta.area_m2_before
            if delta.area_m2_after is not None and delta.area_m2_before is not None
            else None
        )
        name = _pretty(target)
        if delta.pixel_delta == 0:
            headline = f"{name.capitalize()} shows no net change in extent between the two dates"
        else:
            direction = "increased" if delta.pixel_delta > 0 else "decreased"
            magnitude = (
                f"{abs(delta.pixel_delta)} px"
                if area_delta is None
                else f"{abs(area_delta) / 1e6:.2f} km²"
            )
            relative = (
                "" if delta.relative_delta is None
                else f" ({abs(delta.relative_delta) * 100:.0f}% of its earlier extent)"
            )
            headline = f"{name.capitalize()} {direction} by {magnitude}{relative}"

        parts = [
            f"{headline}, measured over the "
            f"{raw.changed_fraction * 100:.1f}% of the scene that changed."
        ]
        # Transitions involving this class, but only those the evidence supports naming — the same
        # gate the change report applies, so the two cannot disagree about what may be claimed.
        involved = [
            t for t in raw.transitions
            if (t.before is target or t.after is target) and raw.transition_is_reportable(t)
        ]
        if involved:
            top = max(involved, key=lambda t: t.pixel_count)
            parts.append(
                f" The largest supported conversion involving it is {_pretty(top.before)} to "
                f"{_pretty(top.after)}, covering {top.fraction_of_change * 100:.0f}% of the "
                f"changed area."
            )
        elif raw.semantic_withheld_reason:
            parts.append(f" No conversion is named for it: {raw.semantic_withheld_reason}")
        return "".join(parts)

    @staticmethod
    def _answer_extent(raw: ChangeResult) -> str:
        extent = (
            f"{raw.changed_pixels} pixels" if raw.changed_area_m2 is None
            else f"about {raw.changed_area_m2 / 1e6:.2f} km² ({raw.changed_pixels} pixels)"
        )
        return (
            f"{raw.changed_fraction * 100:.1f}% of the valid area changed between the two dates — "
            f"{extent}. {raw.corroborated_fraction * 100:.0f}% of that is corroborated by both "
            f"detectors."
        )
