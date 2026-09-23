"""Cross-modal optical + SAR fusion wrapped as a :class:`~app.agents.tools.base.Tool` (§4, §17).

The second *pair* tool. Like :class:`~app.agents.tools.change.ChangeTool` it owns the alignment the
fusion service refuses to guess at: :func:`app.services.fusion.fuse_optical_sar` raises on a shape
mismatch rather than resampling silently, so this tool runs
:func:`~app.geospatial.validate.check_pair_compatibility` →
:func:`~app.geospatial.validate.require_compatible` →
:func:`~app.geospatial.align.align_to_reference` (bringing the SAR image onto the optical grid) and
only then fuses.

It shares :attr:`QueryTask.OPTICAL_SAR_ANALYSIS` with :class:`~app.agents.tools.sar.SarTool`; the
two are told apart by image count and modality in :meth:`validate_input` — one image is a SAR
analysis, one optical + one SAR is a fusion. The answer is phrased from the fused class breakdown,
the fraction both sensors corroborate, and the largest arbitrated disagreement — never asserting
more than the :class:`~app.services.fusion.FusionResult` measured.

Registers at :attr:`~app.agents.tools.base.ToolTier.CLASSICAL`: decision-level rules over classical
land-cover and scattering-regime maps, not a learned fusion model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.errors import ErrorCode, ValidationError
from ...core.types import InputMode, LandCoverClass, Modality, QueryTask, ScatteringRegime
from ...geospatial.align import align_to_reference
from ...geospatial.validate import (
    assess_quality,
    check_pair_compatibility,
    require_compatible,
)
from ...services import confidence
from ...services.fusion import fuse_optical_sar
from .base import Prepared, Tool, ToolContext, ToolResult, ToolTier

if TYPE_CHECKING:
    from ...services.fusion import FusionResult

_ARBITER_WORDS = {"sar": "radar", "optical": "the optical bands"}


def _pretty(label: LandCoverClass) -> str:
    return label.value.replace("_", " ")


def _regime(regime: ScatteringRegime) -> str:
    return regime.value.replace("_", " ")


class FusionTool(Tool):
    """Combine a co-registered optical and SAR image into one arbitrated land-cover reading."""

    task: ClassVar[QueryTask] = QueryTask.OPTICAL_SAR_ANALYSIS
    name: ClassVar[str] = "fusion-decision-cv"
    tier: ClassVar[ToolTier] = ToolTier.CLASSICAL
    summary: ClassVar[str] = (
        "Decision-level optical/SAR fusion: classify the optical image and segment the radar into "
        "scattering regimes on a common grid, then arbitrate their disagreements by rule (radar "
        "settles the built-up vs bare-soil boundary optical cannot), reporting every conflict."
    )

    # Fusion is only defined for one optical *and* one radar scene of the same ground.
    supported_modes: ClassVar[tuple[InputMode, ...]] = (InputMode.OPTICAL_SAR_PAIR,)
    min_images: ClassVar[int] = 2
    max_images: ClassVar[int] = 2
    required_modalities: ClassVar[frozenset[Modality]] = frozenset(
        {Modality.OPTICAL, Modality.SAR}
    )

    def validate_input(self, ctx: ToolContext) -> None:
        ctx.expect_count(2, self.task)
        if {r.modality for r in ctx.rasters} != {Modality.OPTICAL, Modality.SAR}:
            raise ValidationError(
                "Cross-modal fusion needs exactly one optical image and one SAR image of the same "
                "area. For two images of one sensor over time, use change detection; for a single "
                "SAR image, use SAR analysis.",
                code=ErrorCode.WRONG_MODALITY,
                context={"modalities": [r.modality.value for r in ctx.rasters]},
            )

    def preprocess(self, ctx: ToolContext) -> Prepared:
        optical = next(r for r in ctx.rasters if r.modality is Modality.OPTICAL)
        sar = next(r for r in ctx.rasters if r.modality is Modality.SAR)
        pair = check_pair_compatibility(optical, sar)
        require_compatible(pair)  # raises GeospatialError on no / insufficient overlap
        aligned_sar, report = align_to_reference(sar, optical, pair.strategy)
        return Prepared(
            rasters=[optical, aligned_sar],
            quality=assess_quality(optical),
            context={
                "warnings": [*pair.warnings, *report.warnings],
                "compatibility": pair.to_dict(),
                "alignment": report.to_dict(),
            },
        )

    def predict(self, prepared: Prepared) -> FusionResult:
        optical, sar = prepared.rasters
        return fuse_optical_sar(optical, sar)

    def explain(self, raw: FusionResult) -> list[str]:
        lines = [
            f"Method: {raw.method}.",
            f"Sensor agreement over jointly-classified pixels: {raw.agreement_fraction * 100:.1f}%"
            f"; corroborated {raw.corroborated_fraction * 100:.1f}% of the fused map.",
            f"Registration offset {raw.registration.offset_magnitude:.2f} px "
            f"(quality {raw.registration.score:.2f}).",
            "Radar separated regimes: "
            f"{', '.join(_regime(r) for r in raw.sar.separated_regimes)}.",
        ]
        for c in raw.conflicts[:2]:
            lines.append(
                f"Conflict: optical {_pretty(c.optical_class)} vs radar {_regime(c.sar_regime)} "
                f"over {c.pixel_count} px resolved to {_pretty(c.resolved_as)} (by {c.arbiter})."
            )
        lines.extend(raw.warnings)
        return lines

    def postprocess(self, raw: FusionResult, prepared: Prepared) -> ToolResult:
        report = confidence.for_fusion(raw, prepared.quality)
        extra = prepared.context.get("warnings", [])
        warnings = list(dict.fromkeys([*raw.warnings, *extra]))
        return self.build_result(
            answer=self._summarize(raw),
            data=raw.to_dict(),
            confidence=report,
            evidence=self.explain(raw),
            warnings=warnings,
            raw=raw,
        )

    @staticmethod
    def _summarize(raw: FusionResult) -> str:
        if not raw.stats:
            return "Neither the optical nor the radar image could classify this scene."
        dom = raw.stats[0]  # sorted by fraction desc
        lead = (
            f"Combining optical and radar evidence, the scene is predominantly "
            f"{_pretty(dom.label)} ({dom.fraction * 100:.0f}% of classified pixels). Both sensors "
            f"independently agree on {raw.corroborated_fraction * 100:.0f}% of the classified area."
        )
        if raw.conflicts:
            top = raw.conflicts[0]
            who = _ARBITER_WORDS.get(top.arbiter)
            if who is not None:
                lead += (
                    f" The largest sensor disagreement ({top.fraction * 100:.1f}% of the scene) "
                    f"was resolved by {who} in favour of {_pretty(top.resolved_as)}."
                )
            else:
                lead += (
                    f" The largest sensor disagreement ({top.fraction * 100:.1f}% of the scene) "
                    "could not be resolved and is recorded as-is."
                )
        return lead
