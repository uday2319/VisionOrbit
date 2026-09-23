"""Single-image SAR backscatter analysis wrapped as a :class:`~app.agents.tools.base.Tool` (§4, §17).

Takes one radar image and runs the deterministic speckle-filter-and-segment pipeline in
:mod:`app.services.sar`, reporting scattering regimes (smooth / diffuse / double-bounce) rather
than land-cover classes — because single-polarisation backscatter measures surface geometry, not
composition, and the service refuses to split the diffuse middle into vegetation vs bare soil. That
refusal is the honest position and is exactly what motivates optical/SAR fusion (:class:`FusionTool`).

Registers at :attr:`~app.agents.tools.base.ToolTier.CLASSICAL`: the Lee filter and Otsu regime
boundaries are transparent signal processing, not a trained model. Shares the
:attr:`QueryTask.OPTICAL_SAR_ANALYSIS` task with :class:`FusionTool`; the two are told apart by the
number and modality of the images, via :meth:`validate_input`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.errors import ErrorCode, ValidationError
from ...core.types import InputMode, Modality, QueryTask, ScatteringRegime
from ...geospatial.validate import assess_quality
from ...services import confidence
from ...services.sar import analyze_sar
from .base import Prepared, Tool, ToolContext, ToolResult, ToolTier

if TYPE_CHECKING:
    from ...services.sar import SarResult


def _regime_pretty(regime: ScatteringRegime) -> str:
    return regime.value.replace("_", " ")


class SarTool(Tool):
    """Segment a single SAR image into scattering regimes with speckle suppression."""

    task: ClassVar[QueryTask] = QueryTask.OPTICAL_SAR_ANALYSIS
    name: ClassVar[str] = "sar-backscatter-cv"
    tier: ClassVar[ToolTier] = ToolTier.CLASSICAL
    summary: ClassVar[str] = (
        "Single-polarisation SAR analysis: Lee speckle filtering with a look count estimated "
        "from the image, then Otsu scattering-regime segmentation (smooth / diffuse / "
        "double-bounce), gated on mode prominence so a regime is only reported when measured."
    )

    # This is the single-image rung of the OPTICAL_SAR_ANALYSIS chain: one radar scene, no
    # optical counterpart. When an optical+SAR pair arrives, FusionTool is the rung that matches.
    supported_modes: ClassVar[tuple[InputMode, ...]] = (InputMode.SINGLE_SAR,)
    min_images: ClassVar[int] = 1
    max_images: ClassVar[int] = 1
    required_modalities: ClassVar[frozenset[Modality]] = frozenset({Modality.SAR})

    def validate_input(self, ctx: ToolContext) -> None:
        ctx.expect_count(1, self.task)
        raster = ctx.rasters[0]
        if raster.modality is Modality.OPTICAL:
            raise ValidationError(
                "Backscatter analysis needs a radar (SAR) image, but an optical image was "
                "provided. Use land-cover analysis for optical imagery.",
                code=ErrorCode.WRONG_MODALITY,
                context={"tool": self.name, "modality": raster.modality.value},
            )

    def preprocess(self, ctx: ToolContext) -> Prepared:
        raster = ctx.rasters[0]
        return Prepared(rasters=[raster], quality=assess_quality(raster))

    def predict(self, prepared: Prepared) -> SarResult:
        return analyze_sar(prepared.rasters[0])

    def explain(self, raw: SarResult) -> list[str]:
        lines = [f"Polarization: {raw.polarization}; threshold method: {raw.threshold_method}."]
        if raw.smooth_threshold_db is not None:
            lines.append(
                f"Smooth/specular boundary at {raw.smooth_threshold_db:.1f} dB "
                f"(mode prominence {raw.smooth_prominence:.2f})."
            )
        if raw.bright_threshold_db is not None:
            lines.append(
                f"Double-bounce boundary at {raw.bright_threshold_db:.1f} dB "
                f"(mode prominence {raw.bright_prominence:.2f})."
            )
        sp = raw.speckle
        lines.append(
            f"Speckle filter: {sp.filter_name}, window {sp.window}, "
            f"estimated {sp.estimated_enl:.1f} looks, variability reduced "
            f"{sp.speckle_reduction:.1f}x."
        )
        lines.extend(raw.warnings)
        return lines

    def postprocess(self, raw: SarResult, prepared: Prepared) -> ToolResult:
        report = confidence.for_sar(raw, prepared.quality)
        return self.build_result(
            answer=self._summarize(raw),
            data=raw.to_dict(),
            confidence=report,
            evidence=self.explain(raw),
            warnings=raw.warnings,
            raw=raw,
        )

    @staticmethod
    def _summarize(raw: SarResult) -> str:
        if not raw.stats:
            return "No scattering regimes could be measured from this radar image."
        dom = raw.stats[0]  # sorted by fraction desc
        regimes = ", ".join(_regime_pretty(r) for r in raw.separated_regimes)
        lead = (
            f"The {raw.polarization} radar backscatter is predominantly "
            f"{_regime_pretty(dom.regime)} scattering ({dom.fraction * 100:.0f}% of valid "
            f"pixels, mean {dom.mean_db:.1f} dB). Regimes separated: {regimes}."
        )
        surfaces = ", ".join(dom.regime.candidate_surfaces)
        if surfaces:
            lead += (
                f" {_regime_pretty(dom.regime).capitalize()} scattering is consistent with "
                f"{surfaces} — radar cannot decide between these from backscatter alone."
            )
        return lead
