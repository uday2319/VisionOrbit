"""Bi-temporal change analysis wrapped as a :class:`~app.agents.tools.base.Tool` (brief §3, §17).

This is the first *pair* tool, and it owns the step the change service deliberately does not:
bringing the two dates onto a common grid. :func:`app.services.change.detect_change` refuses a
shape mismatch rather than guessing an alignment, so the tool runs
:func:`~app.geospatial.validate.check_pair_compatibility` first — which decides whether the scenes
may be compared at all and, if so, how — then :func:`~app.geospatial.validate.require_compatible`
(an honest refusal when they do not overlap) and :func:`~app.geospatial.align.align_to_reference`
(a no-op for an identical grid, a reprojection/resample otherwise). Only then does it difference the
dates.

It serves both :attr:`QueryTask.CHANGE_DETECTION` ("show the change") and
:attr:`QueryTask.CHANGE_VQA` ("what changed?"): the same measurement answers both moods, and the
registry maps both tasks here. The answer is phrased from the measured changed fraction, the
dominant land-cover transition and the two-detector corroboration — never asserted beyond what the
:class:`~app.services.change.ChangeResult` contains.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.errors import ErrorCode, ValidationError
from ...core.types import InputMode, LandCoverClass, Modality, QueryTask
from ...geospatial.align import align_to_reference, coregister
from ...geospatial.raster import BandRole
from ...geospatial.validate import (
    assess_quality,
    check_pair_compatibility,
    require_compatible,
)
from ...services import confidence
from ...services.change import detect_change
from .base import Prepared, Tool, ToolContext, ToolResult, ToolTier

if TYPE_CHECKING:
    from ...geospatial.raster import RasterData
    from ...services.change import ChangeResult


def _pretty(label: LandCoverClass) -> str:
    return label.value.replace("_", " ")


class ChangeTool(Tool):
    """Compare two co-registered dates of the same area and describe what changed."""

    task: ClassVar[QueryTask] = QueryTask.CHANGE_DETECTION
    name: ClassVar[str] = "change-cva-cv"
    tier: ClassVar[ToolTier] = ToolTier.CLASSICAL
    summary: ClassVar[str] = (
        "Bi-temporal change analysis combining spectral change-vector analysis with land-cover "
        "disagreement, after aligning the two dates to a common grid and measuring registration."
    )

    # Two images of the same sensor family. An optical+SAR pair normalises to
    # OPTICAL_SAR_PAIR, which this contract excludes — that is a cross-modal question.
    supported_modes: ClassVar[tuple[InputMode, ...]] = (InputMode.BITEMPORAL_PAIR,)
    min_images: ClassVar[int] = 2
    max_images: ClassVar[int] = 2
    required_modalities: ClassVar[frozenset[Modality]] = frozenset()

    def validate_input(self, ctx: ToolContext) -> None:
        ctx.expect_count(2, self.task)
        a, b = ctx.rasters
        # Bi-temporal change compares one sensor across time. An optical/SAR pair is a
        # cross-modal question, not a change question, so redirect rather than let the service
        # refuse later with a bare "no bands in common".
        if {a.modality, b.modality} == {Modality.OPTICAL, Modality.SAR}:
            raise ValidationError(
                "Change detection compares two images from the same sensor over time, but these "
                "appear to be different sensor types (optical and SAR). For a combined reading of "
                "the two, use the optical/SAR analysis instead.",
                code=ErrorCode.WRONG_MODALITY,
                context={"modalities": [a.modality.value, b.modality.value]},
            )
        self._require_shared_bands(a, b)

    @staticmethod
    def _require_shared_bands(a: RasterData, b: RasterData) -> None:
        """Refuse a pair with no identified band in common, *before* any analysis runs (§6, §38).

        :func:`app.services.change.detect_change` differences shared spectral indices, falling back
        to shared raw bands; both need at least one identified :class:`BandRole` present in both
        dates, so an empty intersection means the measurement is impossible. The service already
        raises ``MISSING_BAND`` for it — but only after alignment and reprojection have run, and with
        wording ("the two images have no bands in common") that names the symptom rather than the
        cause the user can act on.

        The cause is usually an undetermined sensor. A real SAR scene exported as ``subset_1.tif``
        with an unlabelled band cannot be recognised from its metadata, so it loads as ``UNKNOWN``
        with a single ``gray`` band; paired with an optical scene it normalises to a bi-temporal
        request and lands here. The remedy is to declare the sensor on upload, which the message
        says outright instead of leaving the user to infer it from a band list.
        """
        shared = {
            role for role in a.band_roles if role is not BandRole.UNKNOWN
        } & {role for role in b.band_roles if role is not BandRole.UNKNOWN}
        if shared:
            return

        undetermined = [
            i + 1 for i, r in enumerate((a, b)) if r.modality is Modality.UNKNOWN
        ]
        if undetermined:
            which = (
                f"image {undetermined[0]}"
                if len(undetermined) == 1
                else "neither image"
            )
            remedy = (
                f"The sensor type of {which} could not be determined from its metadata, so its "
                "bands could not be matched to the other image. Declare the sensor when uploading "
                "(the optical / SAR selection in the interface, or the 'modality_hint' upload "
                "field), or supply imagery whose band descriptions name the bands."
            )
        else:
            remedy = (
                "Supply two scenes from the same sensor, so that at least one band is present in "
                "both dates."
            )
        raise ValidationError(
            "These two images have no identified band in common, so there is nothing to compare "
            f"between the two dates. {remedy}",
            code=ErrorCode.MISSING_BAND,
            context={
                "modalities": [a.modality.value, b.modality.value],
                "band_roles": [
                    [role.value for role in a.band_roles],
                    [role.value for role in b.band_roles],
                ],
            },
        )

    def preprocess(self, ctx: ToolContext) -> Prepared:
        before, after = ctx.rasters
        pair = check_pair_compatibility(before, after)
        require_compatible(pair)  # raises GeospatialError on no / insufficient overlap
        aligned_after, report = align_to_reference(after, before, pair.strategy)
        # Being on a common grid is not being aligned. Correct the residual misalignment before
        # differencing, keeping the correction only if re-measuring shows it helped; `coreg.final`
        # is the registration of the pair that is actually differenced below.
        coreg = coregister(aligned_after, before)
        return Prepared(
            rasters=[before, coreg.raster],
            quality=assess_quality(before),
            context={
                "warnings": [*pair.warnings, *report.warnings, *coreg.warnings],
                "compatibility": pair.to_dict(),
                "alignment": report.to_dict(),
                "coregistration": coreg.to_dict(),
                "registration": coreg.final,
            },
        )

    def predict(self, prepared: Prepared) -> ChangeResult:
        before, after = prepared.rasters
        return detect_change(before, after, registration=prepared.context.get("registration"))

    def explain(self, raw: ChangeResult) -> list[str]:
        lines = [
            f"Method: {raw.method}; detectors: {', '.join(raw.detectors)}.",
            f"Change threshold {raw.threshold:.4f} ({raw.threshold_method}); "
            f"magnitude bimodality {raw.bimodality:.2f}.",
            f"Registration offset {raw.registration.offset_magnitude:.2f} px "
            f"(quality {raw.registration.score:.2f}).",
        ]
        gate = raw.gate
        lines.append(
            f"Semantic-claim gate: {'passed' if gate.passed else 'not passed'} "
            f"(needs offset <= {gate.thresholds['max_offset_px']:.1f} px, "
            f"structural agreement >= {gate.thresholds['min_ncc']:.2f}, "
            f"registration quality >= {gate.thresholds['min_score']:.2f})."
        )
        if raw.changed_pixels:
            lines.append(
                f"{raw.corroborated_fraction * 100:.0f}% of {raw.changed_pixels} changed pixels "
                f"are corroborated by both detectors."
            )
        # Named transitions are reported as *measurements* here regardless of the gate, and the
        # gate verdict is stated alongside — evidence should show what was measured and why a
        # conclusion was or was not drawn from it, not hide the measurement.
        dom = raw.dominant_transition()
        if dom is not None:
            lines.append(
                f"Largest measured transition {_pretty(dom.before)} to {_pretty(dom.after)} "
                f"({dom.spectral_agreement * 100:.0f}% spectral agreement, "
                f"{'reportable' if raw.transition_is_reportable(dom) else 'withheld'} as a "
                f"semantic finding)."
            )
        if raw.semantic_withheld_reason:
            lines.append(f"Semantic claim withheld: {raw.semantic_withheld_reason}")
        lines.extend(raw.warnings)
        return lines

    def postprocess(self, raw: ChangeResult, prepared: Prepared) -> ToolResult:
        report = confidence.for_change(raw, prepared.quality)
        # raw.warnings already carries registration + per-date caveats; the pair/alignment
        # warnings (partial overlap, reprojection) are added here. dict.fromkeys dedups.
        extra = prepared.context.get("warnings", [])
        warnings = list(dict.fromkeys([*raw.warnings, *extra]))
        data = raw.to_dict()
        coreg = prepared.context.get("coregistration")
        if coreg is not None:
            # What was done about the misalignment, not merely what it measured — so the result
            # page can show the correction and its measured effect rather than a bare final score.
            data["coregistration"] = coreg
        return self.build_result(
            answer=self._summarize(raw),
            data=data,
            confidence=report,
            evidence=self.explain(raw),
            warnings=warnings,
            raw=raw,
        )

    @staticmethod
    def _summarize(raw: ChangeResult) -> str:
        if raw.changed_pixels == 0:
            return (
                "No change large enough to be distinguished from image noise was detected "
                "between the two dates. This measures no detected change, not proof that "
                "nothing changed."
            )
        parts = [f"Change was detected across {raw.changed_fraction * 100:.1f}% of the valid area"]
        if raw.changed_area_m2 is not None:
            parts.append(f" (about {raw.changed_area_m2 / 1e6:.2f} km²)")
        parts.append(".")
        # The named transition is a *conclusion drawn from* the measured difference, and is
        # asserted only when the pair is registered well enough and the spectral detector
        # independently corroborates that specific transition. Otherwise the sentence says the
        # difference was measured and the interpretation withheld — it does not quietly drop the
        # subject, because a reader who can see nine transitions in the table deserves to know
        # why none of them is being named.
        semantic = raw.semantic_transition()
        if semantic is not None:
            parts.append(
                f" The dominant transition is {_pretty(semantic.before)} to "
                f"{_pretty(semantic.after)}, covering "
                f"{semantic.fraction_of_change * 100:.0f}% of the changed area."
            )
        elif raw.semantic_withheld_reason:
            parts.append(
                " No specific land-cover conversion is claimed for that change: "
                f"{raw.semantic_withheld_reason}"
            )
        parts.append(
            f" {raw.corroborated_fraction * 100:.0f}% of the detected change is corroborated by "
            f"both detectors."
        )
        return "".join(parts)
