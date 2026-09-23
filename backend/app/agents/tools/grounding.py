"""Text-guided region grounding wrapped as a :class:`~app.agents.tools.base.Tool` (brief §2B, §17).

This is the query-driven tool: it locates the land-cover type a phrase names ("where is the water",
"how many built-up areas in the north") and hands back *regions* — patches, areas, counts — rather
than prose. Because :meth:`~app.agents.tools.base.Tool.predict` does not receive the
:class:`~app.agents.tools.base.ToolContext`, the query text is stashed in
:attr:`Prepared.context` during :meth:`preprocess` and read back in :meth:`predict`.

Two honesty properties are inherited straight from :func:`app.services.grounding.ground_query` and
must not be papered over here:

* **A count is only stated when it is reportable.** Built-up land and bare soil cannot be counted
  reliably from optical data alone (their mutual boundary is not spectrally measurable), so the
  service returns ``count_reportable=False`` with a physical reason. The tool then describes the
  mapped *extent* and declines to give a number, rather than inventing one.
* **An unparseable or unsupported query is refused, not approximated.** ``ground_query`` raises
  :class:`~app.core.errors.UnsupportedTaskError` for object classes with no detector ("ships") and
  spatial relations ("near the river"); those propagate for the orchestrator to surface honestly.

Registers at :attr:`~app.agents.tools.base.ToolTier.CLASSICAL`: the regions come from the spectral
classifier cascade and connected-component vectorisation, not a trained grounding model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.errors import ErrorCode, ValidationError
from ...core.types import InputMode, Modality, QueryTask
from ...geospatial.validate import assess_quality
from ...services import confidence
from ...services.grounding import ground_query
from .base import Prepared, Tool, ToolContext, ToolResult, ToolTier

if TYPE_CHECKING:
    from ...services.grounding import GroundingResult


class GroundingTool(Tool):
    """Locate the land-cover type a text query names, as regions rather than prose."""

    task: ClassVar[QueryTask] = QueryTask.GROUNDING
    name: ClassVar[str] = "grounding-spectral-cv"
    tier: ClassVar[ToolTier] = ToolTier.CLASSICAL
    summary: ClassVar[str] = (
        "Text-guided region grounding: parses the query against a fixed land-cover vocabulary "
        "(refusing objects it has no detector for), then vectorises the classifier's mask into "
        "regions, filtering by named sector and size — stating a count only when it is reportable."
    )

    # Grounding localises land cover in one optical scene; it has no cross-image semantics.
    supported_modes: ClassVar[tuple[InputMode, ...]] = (InputMode.SINGLE_OPTICAL,)
    min_images: ClassVar[int] = 1
    max_images: ClassVar[int] = 1
    required_modalities: ClassVar[frozenset[Modality]] = frozenset({Modality.OPTICAL})

    def validate_input(self, ctx: ToolContext) -> None:
        ctx.expect_count(1, self.task)
        raster = ctx.rasters[0]
        if raster.modality is Modality.SAR:
            raise ValidationError(
                "Region grounding locates land-cover types, which needs an optical image — a "
                "single SAR image carries no spectral land-cover signal. Use SAR analysis for "
                "radar imagery.",
                code=ErrorCode.WRONG_MODALITY,
                context={"tool": self.name, "modality": raster.modality.value},
            )

    def preprocess(self, ctx: ToolContext) -> Prepared:
        raster = ctx.rasters[0]
        # predict() does not receive the context, so the query travels in Prepared.context.
        return Prepared(
            rasters=[raster],
            quality=assess_quality(raster),
            context={"query": ctx.query},
        )

    def predict(self, prepared: Prepared) -> GroundingResult:
        return ground_query(prepared.context["query"], prepared.rasters[0])

    def explain(self, raw: GroundingResult) -> list[str]:
        ev = raw.evidence
        lines = [
            f"Source: {raw.source}.",
            f"Classification method: {ev.get('classification_method')}; class separability "
            f"{ev.get('class_separability')}.",
            f"{ev.get('candidate_regions')} candidate patches above the "
            f"{ev.get('min_region_pixels')}-pixel floor "
            f"({ev.get('min_region_area_m2'):.0f} m² minimum region).",
        ]
        if ev.get("region_score_index"):
            lines.append(
                f"Region strength scored by {ev['region_score_index']} (reported as evidence "
                "only, never used to accept or reject a region)."
            )
        if ev.get("sector_selection"):
            lines.append(f"Sector selection: {ev['sector_selection']}.")
        lines.extend(raw.warnings)
        return lines

    def postprocess(self, raw: GroundingResult, prepared: Prepared) -> ToolResult:
        report = confidence.for_grounding(raw, prepared.quality)
        return self.build_result(
            answer=self._summarize(raw),
            data=raw.to_dict(),
            confidence=report,
            evidence=self.explain(raw),
            warnings=raw.warnings,
            raw=raw,
        )

    @staticmethod
    def _summarize(raw: GroundingResult) -> str:
        target = raw.query.target.value.replace("_", " ")
        if raw.class_pixels == 0:
            return (
                f"No {target} was detected anywhere in this image, so there is nothing to locate."
            )
        where = f" in the {raw.query.sector_phrase}" if raw.query.sector is not None else ""
        if not raw.regions:
            return (
                f"{target.capitalize()} is present in the image, but no area{where} met the "
                "query's criteria, so nothing is reported as a distinct region."
            )
        extent = f"{raw.selected_pixels} px"
        if raw.selected_area_m2 is not None:
            extent = f"about {raw.selected_area_m2 / 1e6:.2f} km² ({raw.selected_pixels} px)"
        if raw.count_reportable:
            plural = "s" if raw.count != 1 else ""
            lead = (
                f"Found {raw.count} distinct {target} region{plural}{where}, covering {extent}. "
            )
        else:
            lead = (
                f"The {target} extent{where} is mapped (covering {extent}), but the number of "
                "separate regions cannot be stated reliably from this image alone. "
            )
        return lead + raw.region_semantics
