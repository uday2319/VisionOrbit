"""Single-image visual question answering wrapped as a :class:`~app.agents.tools.base.Tool` (§2A, §17).

The other query-driven tool alongside :class:`~app.agents.tools.grounding.GroundingTool`, and its
complement: grounding answers *where* with regions, VQA answers *what / how much / is there* with a
sentence grounded in measured land-cover fractions. Because
:meth:`~app.agents.tools.base.Tool.predict` does not receive the
:class:`~app.agents.tools.base.ToolContext`, the question travels in :attr:`Prepared.context` from
:meth:`preprocess` to :meth:`predict`, the same convention grounding uses.

Every honesty property is inherited from :func:`app.services.vqa.answer_question` and must not be
weakened here: an object class with no detector or a spatial relation is refused (propagated as
:class:`~app.core.errors.UnsupportedTaskError`); a *where* or count-of-regions question is answered
with coverage and redirected rather than faked. Confidence is the evidence-based report from
:func:`app.services.confidence.for_landcover` (§27) — the VQA answer rests entirely on the land-cover
classification, so that is exactly the confidence in the answer.

Registers at :attr:`~app.agents.tools.base.ToolTier.CLASSICAL`: the answer comes from the spectral
classifier and deterministic query parsing, not a trained vision-language model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.errors import ErrorCode, ValidationError
from ...core.types import InputMode, Modality, QueryTask
from ...geospatial.validate import assess_quality
from ...services import confidence
from ...services.vqa import answer_question
from .base import Prepared, Tool, ToolContext, ToolResult, ToolTier

if TYPE_CHECKING:
    from ...services.vqa import VqaAnswer


class VqaTool(Tool):
    """Answer a natural-language question about one optical scene from its land cover."""

    task: ClassVar[QueryTask] = QueryTask.VQA
    name: ClassVar[str] = "vqa-landcover-cv"
    tier: ClassVar[ToolTier] = ToolTier.CLASSICAL
    summary: ClassVar[str] = (
        "Single-image question answering over land cover: parses the question against a fixed "
        "land-cover vocabulary (refusing objects it has no detector for and spatial relations it "
        "cannot evaluate), then answers presence, quantity, comparison and composition questions "
        "from measured class fractions — redirecting where/count questions to region grounding."
    )

    # Exactly one optical scene. Two images is the case the brief singles out: it must be
    # rejected before execution, not answered with zero confidence.
    supported_modes: ClassVar[tuple[InputMode, ...]] = (InputMode.SINGLE_OPTICAL,)
    min_images: ClassVar[int] = 1
    max_images: ClassVar[int] = 1
    required_modalities: ClassVar[frozenset[Modality]] = frozenset({Modality.OPTICAL})

    def validate_input(self, ctx: ToolContext) -> None:
        ctx.expect_count(1, self.task)
        raster = ctx.rasters[0]
        if raster.modality is Modality.SAR:
            raise ValidationError(
                "This question answering reads land cover, which needs an optical (multispectral) "
                "image — a single SAR image carries no spectral land-cover signal. Use SAR "
                "analysis for radar imagery.",
                code=ErrorCode.WRONG_MODALITY,
                context={"tool": self.name, "modality": raster.modality.value},
            )

    def preprocess(self, ctx: ToolContext) -> Prepared:
        raster = ctx.rasters[0]
        # predict() does not receive the context, so the question travels in Prepared.context.
        return Prepared(
            rasters=[raster],
            quality=assess_quality(raster),
            context={"query": ctx.query},
        )

    def predict(self, prepared: Prepared) -> VqaAnswer:
        return answer_question(prepared.context["query"], prepared.rasters[0])

    def explain(self, raw: VqaAnswer) -> list[str]:
        return list(raw.evidence)

    def postprocess(self, raw: VqaAnswer, prepared: Prepared) -> ToolResult:
        # The answer rests entirely on the land-cover classification, so its confidence is the
        # classification's confidence (§27) — not a separate, ungrounded VQA score.
        report = confidence.for_landcover(raw.landcover, prepared.quality)
        return self.build_result(
            answer=raw.answer,
            data=raw.to_dict(),
            confidence=report,
            evidence=self.explain(raw),
            warnings=raw.warnings,
            raw=raw,
        )
