"""Land-cover classification wrapped as a dispatchable :class:`~app.agents.tools.base.Tool` (§17).

This is the reference implementation of the uniform tool contract: it takes a single optical
image, runs the deterministic spectral/RGB classifier in :mod:`app.services.landcover`, and returns
a :class:`~app.agents.tools.base.ToolResult` whose ``answer`` is phrased entirely from measured
class fractions and whose confidence is the evidence-based report from
:func:`app.services.confidence.for_landcover` (§27) — never a guess.

It registers at :attr:`~app.agents.tools.base.ToolTier.CLASSICAL` because the classifier is a
transparent index-and-texture decision hierarchy, not a trained model (§6). If a learned land-cover
model is added later it slots in at a higher tier behind the same task, and this tool remains as the
honest fallback.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.errors import ErrorCode, ValidationError
from ...core.types import InputMode, LandCoverClass, Modality, QueryTask
from ...geospatial.validate import assess_quality
from ...services import confidence
from ...services.landcover import classify_land_cover
from .base import Prepared, Tool, ToolContext, ToolResult, ToolTier

if TYPE_CHECKING:
    from ...services.landcover import LandCoverResult


def _pretty(label: LandCoverClass) -> str:
    """Human-readable class name, e.g. ``BUILT_UP`` -> ``built up``."""
    return label.value.replace("_", " ")


class LandCoverTool(Tool):
    """Classify a single optical scene into water / vegetation / built-up / bare-soil classes."""

    task: ClassVar[QueryTask] = QueryTask.LAND_COVER_ANALYSIS
    name: ClassVar[str] = "landcover-spectral-cv"
    tier: ClassVar[ToolTier] = ToolTier.CLASSICAL
    summary: ClassVar[str] = (
        "Per-pixel land-cover classification from spectral indices (NDWI/NDVI/NDBI) and edge "
        "texture, with an RGB approximation when no near-infrared band is present."
    )

    # One optical scene. A second image makes this a change question, not a land-cover one.
    supported_modes: ClassVar[tuple[InputMode, ...]] = (InputMode.SINGLE_OPTICAL,)
    min_images: ClassVar[int] = 1
    max_images: ClassVar[int] = 1
    required_modalities: ClassVar[frozenset[Modality]] = frozenset({Modality.OPTICAL})

    def validate_input(self, ctx: ToolContext) -> None:
        ctx.expect_count(1, self.task)
        raster = ctx.rasters[0]
        if raster.modality is Modality.SAR:
            raise ValidationError(
                "Land-cover classification needs an optical (multispectral) image, but a SAR "
                "image was provided. Use the optical/SAR analysis for radar imagery.",
                code=ErrorCode.WRONG_MODALITY,
                context={"tool": self.name, "modality": raster.modality.value},
            )

    def preprocess(self, ctx: ToolContext) -> Prepared:
        raster = ctx.rasters[0]
        return Prepared(rasters=[raster], quality=assess_quality(raster))

    def predict(self, prepared: Prepared) -> LandCoverResult:
        return classify_land_cover(prepared.rasters[0])

    def explain(self, raw: LandCoverResult) -> list[str]:
        lines = [f"Classification method: {raw.method}."]
        dominant = raw.dominant()
        if dominant is not None:
            lines.append(
                f"Dominant class is {_pretty(dominant.label)}, covering "
                f"{dominant.fraction * 100:.1f}% of valid pixels."
            )
        lines.append(f"Measured class separability: {raw.separability:.2f}.")
        if raw.indices_used:
            lines.append(f"Spectral indices used: {', '.join(raw.indices_used)}.")
        lines.extend(raw.warnings)
        return lines

    def postprocess(self, raw: LandCoverResult, prepared: Prepared) -> ToolResult:
        report = confidence.for_landcover(raw, prepared.quality)
        return self.build_result(
            answer=self._summarize(raw),
            data=raw.to_dict(),
            confidence=report,
            evidence=self.explain(raw),
            warnings=raw.warnings,
            raw=raw,
        )

    @staticmethod
    def _summarize(raw: LandCoverResult) -> str:
        """A concise, measurement-grounded sentence naming the dominant class and top classes.

        Every number here is a class coverage fraction the classifier computed; nothing is
        asserted that the ``stats`` do not contain.
        """
        dominant = raw.dominant()
        if dominant is None:
            return "No land-cover classes could be delineated from this image."
        # `stats` is already sorted by descending fraction (see landcover._class_stats).
        breakdown = ", ".join(
            f"{_pretty(s.label)} {s.fraction * 100:.0f}%" for s in raw.stats[:3]
        )
        return (
            f"The scene is predominantly {_pretty(dominant.label)} "
            f"({dominant.fraction * 100:.0f}% of classified pixels). "
            f"Class breakdown: {breakdown}."
        )
