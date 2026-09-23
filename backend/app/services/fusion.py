"""Cross-modal optical + SAR analysis (brief §4).

Fusion here is at the **decision level**, not the pixel level. Reflectance and backscatter are
physically unrelated quantities — one is a fraction of incident sunlight, the other a
normalised radar cross-section — so stacking or averaging them produces a number with no
physical meaning whose apparent precision is entirely artificial. What is combined instead is
each sensor's *conclusion*, and every combined pixel carries a :class:`FusionEvidence` label
saying which sensors supported it.

**The arbitration principle, stated before any number was consulted: each sensor decides the
axis its physics measures directly.** Two rules follow, and nothing else in this module
resolves a disagreement.

*Rule A — radar decides built-up.* Double-bounce is a direct geometric measurement: a vertical
face beside the ground returns a wall-then-floor path straight back to the sensor, and no
natural surface produces it. Optical built-up has no geometric measurement available to it at
all; :mod:`app.services.landcover` reaches the class by combining SWIR brightness with an
*edge-density texture proxy*, because bare soil is also SWIR-bright. A proxy loses to a direct
measurement. So a double-bounce pixel is built-up whatever the optical side said, and an
optical built-up claim that radar contradicts is withdrawn.

The withdrawal half of Rule A holds only while radar actually separated a double-bounce
population. Where it separated none, the absence is a property of the histogram rather than of
the ground: :func:`analyze_sar` declined to place a bright boundary at all, so nothing was
measured about vertical structure anywhere in the frame and there is no reading to contradict
the optical one with. Vetoing on that basis would convert "radar could not tell" into "radar
disagrees". The optical claim stands instead, labelled uncorroborated, and the method is
reported as ``…-partial`` so the caller can see that half the arbitration was unavailable.

*Rule B — optical decides surface composition.* Water, vegetation and bare soil separate on
chlorophyll and liquid-water absorption in NIR/SWIR, which is again a direct measurement.
Single-polarisation VV provably cannot make that separation: :attr:`ScatteringRegime.DIFFUSE`
covers vegetation *and* bare soil and is never subdivided, and ``SMOOTH`` covers calm water
*and* dry sand. So a conflict that is not about built-up leaves the optical class standing and
is recorded, not resolved.

*Rule C — a regime names a class only where it names exactly one.* Where the optical side has
no class at all, radar supplies one only if
:attr:`ScatteringRegime.consistent_land_cover` is a single class — true only of double bounce.
Under ``SMOOTH`` or ``DIFFUSE`` the fused map stays unclassified rather than picking one of two
equally consistent surfaces.

What the rules are worth, measured on this repo's demo optical/SAR pair against its exact
ground truth (reproduced by ``tests/services/test_fusion.py``):

===================  ==============  =============
Class                Optical alone   Fused F1
===================  ==============  =============
built-up             0.7861          **0.9989**
bare soil            0.9833          0.9945
vegetation           0.9904          0.9905
water                0.9630          0.9630
overall accuracy     0.9768          **0.9923**
===================  ==============  =============

Built-up is the whole story, and the shape of the improvement is the evidence that the rules
are physics rather than a fit: precision rises from 0.6651 to 0.9997 because 3,770 SWIR-bright,
ploughing-textured bare-soil pixels stop being called buildings, while water and vegetation —
the axis Rule B forbids radar from touching — move by 0.0000 and 0.0000 respectively. A fusion
rule tuned for score would have improved everything a little; this one improves exactly the
axis it claims to and leaves the rest alone.

Both directions of the built-up conflict were resolved correctly, which matters because the
rule was fixed in advance and could have been wrong in either:

* radar asserting built-up where optical saw bare soil — 298 pixels, **298** truly built;
* radar withdrawing optical's built-up claim — 3,798 pixels, of which 3,770 are truly bare
  soil and 6 are water that the optical water test had already missed at the shoreline.

Those 6 are the rule's weakest corner and are recorded rather than smoothed over. They are
``optical=built_up`` against ``sar=smooth``: Rule A withdraws the built-up claim, and the class
falls back to bare soil because that is what the optical cascade itself reaches without its
built-up branch (built-up is carved out of the non-water, non-vegetated remainder). Radar
agrees the surface is specular but cannot say whether it is water or dry ground, so fusion
inherits the optical water miss instead of fixing it. Rule B forbids inventing a fix, and a
special case for this 6-pixel group would be fitting to one scene.

**The case fusion exists for.** Given RGB-only optical, :func:`classify_land_cover` refuses to
claim built-up at all — separating buildings from bare soil needs SWIR, so it reports 0 pixels
and says so. Overall accuracy is 0.9626 and built-up F1 is exactly 0.0000. Adding the SAR
channel recovers built-up at F1 **0.9989** and lifts overall accuracy to 0.9923. That is not a
marginal gain over a working analysis; it is the difference between declining to answer and
answering correctly, which is the argument for requiring both modalities.

**Sensor agreement is reported, not used as a gate.** 98.43% of jointly-observed pixels agree
on the demo pair, and the evidence label is *provenance* rather than a correctness prediction:
accuracy is 0.9923 on the corroborated majority against 0.9927 on the arbitrated remainder, so
the two are indistinguishable and nothing may read the label as a per-pixel reliability score.
It is also tempting to treat a rising conflict fraction as a "wrong scene"
detector, and that was tested and rejected: with the correct optical date the fraction runs
0.0157 → 0.0532 as an artificial shift grows from 0 to 20 px, while the *wrong* date perfectly
aligned reads 0.0514. A 20-pixel misregistration of the right scene therefore looks worse than
the wrong scene aligned, so the quantity cannot distinguish the two failures and must not be
thresholded as though it could. Misalignment is detected where it is actually measurable, by
:func:`measure_registration`; the conflict fraction is reported for the confidence layer to
consume continuously, and only a structural condition — more disagreement than agreement —
raises a warning.

Areas are ``None`` when the inputs are not georeferenced, as everywhere else in this package.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import ErrorCode, GeospatialError
from ..core.logging import get_logger
from ..core.types import (
    FusionEvidence,
    LandCoverClass,
    Modality,
    ScatteringRegime,
)
from ..geospatial.align import (
    REGISTRATION_WARN_PX,
    RegistrationQuality,
    measure_registration,
)
from ..geospatial.measure import pixel_area_m2
from ..geospatial.raster import RasterData
from .landcover import LandCoverResult, classify_land_cover
from .sar import SarResult, analyze_sar

logger = get_logger(__name__)

CODE_TO_EVIDENCE = {
    FusionEvidence.NEITHER: 0,
    FusionEvidence.BOTH: 1,
    FusionEvidence.CONFLICT: 2,
    FusionEvidence.OPTICAL_ONLY: 3,
    FusionEvidence.SAR_ONLY: 4,
}
EVIDENCE_FROM_CODE = {v: k for k, v in CODE_TO_EVIDENCE.items()}

# Human-readable justification attached to every conflict group, so a user reading the result
# sees *why* one sensor was preferred rather than only that it was. Keyed by the arbitrating
# rule, because the reason is a property of the rule and not of the individual pixel group.
_REASON_SAR_ASSERTS = (
    "Radar measured a double-bounce return here, which is the geometric signature of a "
    "vertical surface meeting the ground. Nothing in the optical bands measures geometry, so "
    "the radar reading decides this."
)
_REASON_SAR_WITHDRAWS = (
    "The optical bands suggested built-up land from brightness and surface texture, but radar "
    "measured no double-bounce return, so there is no vertical structure here. The built-up "
    "claim is withdrawn and the surface is reported as bare or unvegetated ground."
)
_REASON_OPTICAL_STANDS = (
    "Radar cannot separate the surfaces consistent with this return, so the optical "
    "classification stands. The disagreement is reported rather than resolved."
)
_REASON_UNRESOLVED = (
    "Radar found no separable population of strong double-bounce returns anywhere in this "
    "image, so it made no measurement of vertical structure that could confirm or contradict "
    "the optical reading. The optical classification stands, uncorroborated."
)


@dataclass
class FusedClassStats:
    """Coverage of one land-cover class in the fused map, split by evidence."""

    label: LandCoverClass
    pixel_count: int
    fraction: float
    """Share of the *classified* area, so the fractions sum to 1 whatever the nodata extent."""

    area_m2: float | None
    corroborated_fraction: float
    """Share of this class's pixels that both sensors independently supported."""

    evidence_counts: dict[FusionEvidence, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.label.value,
            "pixel_count": self.pixel_count,
            "fraction": round(self.fraction, 4),
            "percentage": round(self.fraction * 100, 2),
            "area_m2": round(self.area_m2, 2) if self.area_m2 is not None else None,
            "corroborated_fraction": round(self.corroborated_fraction, 4),
            "evidence": {k.value: v for k, v in self.evidence_counts.items() if v},
        }


@dataclass
class ConflictGroup:
    """One (optical class, SAR regime) pair the two sensors disagreed on.

    Reported in full rather than averaged away: the counts, the rule that arbitrated, and the
    surfaces radar considered consistent with its own reading.
    """

    optical_class: LandCoverClass
    sar_regime: ScatteringRegime
    pixel_count: int
    fraction: float
    area_m2: float | None
    resolved_as: LandCoverClass
    arbiter: str
    """Which modality decided: ``"sar"``, ``"optical"``, or ``"none"`` if unresolved."""

    reason: str
    candidate_surfaces: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "optical_class": self.optical_class.value,
            "sar_regime": self.sar_regime.value,
            "pixel_count": self.pixel_count,
            "fraction": round(self.fraction, 5),
            "area_m2": round(self.area_m2, 2) if self.area_m2 is not None else None,
            "resolved_as": self.resolved_as.value,
            "arbiter": self.arbiter,
            "reason": self.reason,
            "radar_candidate_surfaces": list(self.candidate_surfaces),
        }


@dataclass
class FusionResult:
    """Combined optical + SAR interpretation of one scene."""

    class_map: np.ndarray
    """``(H, W)`` uint8 of :attr:`LandCoverClass.code` values."""

    evidence_map: np.ndarray
    """``(H, W)`` uint8 of :data:`CODE_TO_EVIDENCE` values, aligned with ``class_map``."""

    stats: list[FusedClassStats]
    conflicts: list[ConflictGroup]
    registration: RegistrationQuality
    optical: LandCoverResult
    """The optical-only result, kept so a caller can show what each sensor contributed."""

    sar: SarResult
    agreement_fraction: float
    """Share of pixels *both* sensors spoke about where they were mutually consistent."""

    method: str
    warnings: list[str] = field(default_factory=list)

    def mask_for(self, label: LandCoverClass) -> np.ndarray:
        return self.class_map == label.code

    def evidence_mask(self, evidence: FusionEvidence) -> np.ndarray:
        return self.evidence_map == CODE_TO_EVIDENCE[evidence]

    def stats_for(self, label: LandCoverClass) -> FusedClassStats | None:
        return next((s for s in self.stats if s.label is label), None)

    def dominant(self) -> LandCoverClass | None:
        """Largest class by coverage, or ``None`` when nothing was classified."""
        return self.stats[0].label if self.stats else None

    @property
    def conflict_fraction(self) -> float:
        """Complement of :attr:`agreement_fraction`, for readability at call sites."""
        return 1.0 - self.agreement_fraction

    @property
    def corroborated_fraction(self) -> float:
        """Share of the *classified* map that both sensors supported.

        Differs from :attr:`agreement_fraction`, which is computed only over pixels both
        sensors spoke about: a scene where radar covers half the frame can have perfect
        agreement and still be only half corroborated.
        """
        classified = int(np.count_nonzero(self.class_map != LandCoverClass.UNCLASSIFIED.code))
        if classified == 0:
            return 0.0
        return float(np.count_nonzero(self.evidence_mask(FusionEvidence.BOTH)) / classified)

    def to_dict(self) -> dict[str, Any]:
        total = int(self.evidence_map.size)
        dominant = self.dominant()
        return {
            "method": self.method,
            "classes": [s.to_dict() for s in self.stats],
            "dominant_class": dominant.value if dominant is not None else None,
            "agreement_fraction": round(self.agreement_fraction, 4),
            "corroborated_fraction": round(self.corroborated_fraction, 4),
            "evidence_breakdown": {
                ev.value: int(np.count_nonzero(self.evidence_mask(ev)))
                for ev in FusionEvidence
                if np.any(self.evidence_mask(ev))
            },
            "evidence_fractions": {
                ev.value: round(
                    float(np.count_nonzero(self.evidence_mask(ev)) / total), 4
                )
                for ev in FusionEvidence
                if np.any(self.evidence_mask(ev))
            },
            "conflicts": [c.to_dict() for c in self.conflicts],
            "registration": self.registration.to_dict(),
            "optical_only": {
                "method": self.optical.method,
                "indices_used": list(self.optical.indices_used),
                "classes": [s.label.value for s in self.optical.stats],
            },
            "sar_only": {
                "polarization": self.sar.polarization,
                "separated_regimes": [r.value for r in self.sar.separated_regimes],
            },
            "warnings": list(self.warnings),
        }


def _modality_warnings(optical: RasterData, sar: RasterData) -> list[str]:
    """Flag inputs whose detected modality is not what the caller passed them as.

    Fusion is only meaningful between an optical and a radar acquisition, and the arbitration
    rules are justified entirely by that assumption. Handing the same optical scene in twice
    would otherwise run silently and produce a confident, meaningless result.
    """
    out: list[str] = []
    if optical.modality is Modality.SAR:
        out.append(
            "The image supplied as optical looks like radar data. Cross-modal analysis "
            "assumes one optical and one radar acquisition; check the inputs are the right "
            "way round."
        )
    if sar.modality is Modality.OPTICAL:
        out.append(
            "The image supplied as radar looks like optical data. Cross-modal analysis "
            "assumes one optical and one radar acquisition; check the inputs are the right "
            "way round."
        )
    return out


def _consistency_mask(class_map: np.ndarray, regime_map: np.ndarray) -> np.ndarray:
    """Where the optical class is one radar considers consistent with its own regime.

    Built entirely from :attr:`ScatteringRegime.consistent_land_cover`, so the compatibility
    relation lives with the radar physics that defines it and cannot drift out of step with the
    candidate surfaces reported to the user.
    """
    consistent = np.zeros(regime_map.shape, dtype=bool)
    for regime in ScatteringRegime:
        classes = regime.consistent_land_cover
        if not classes:
            continue
        in_regime = regime_map == regime.code
        if not in_regime.any():
            continue
        matches = np.zeros(class_map.shape, dtype=bool)
        for label in classes:
            matches |= class_map == label.code
        consistent |= in_regime & matches
    return consistent


def _resolve_conflicts(
    fused: np.ndarray,
    optical_map: np.ndarray,
    regime_map: np.ndarray,
    conflict: np.ndarray,
    *,
    sar_decides_built_up: bool,
    px_area: float | None,
    total: int,
) -> list[ConflictGroup]:
    """Apply Rules A and B to every disagreeing (class, regime) pair, in place on ``fused``.

    Args:
        sar_decides_built_up: Whether radar actually separated a double-bounce population. When
            it did not, Rule A has no measurement behind it and must not fire: the absence of a
            *separable* bright class is not evidence that a particular pixel lacks vertical
            structure, so an optical built-up claim is left standing and labelled
            uncorroborated rather than withdrawn on the strength of a measurement nobody made.

    Returns:
        One :class:`ConflictGroup` per pair that actually occurred, largest first.
    """
    groups: list[ConflictGroup] = []
    for regime in (
        ScatteringRegime.SMOOTH,
        ScatteringRegime.DIFFUSE,
        ScatteringRegime.DOUBLE_BOUNCE,
    ):
        in_regime = conflict & (regime_map == regime.code)
        if not in_regime.any():
            continue
        for label in LandCoverClass:
            if label is LandCoverClass.UNCLASSIFIED:
                continue
            group = in_regime & (optical_map == label.code)
            count = int(np.count_nonzero(group))
            if count == 0:
                continue

            # Rule A, both directions; then Rule B for everything else.
            if regime is ScatteringRegime.DOUBLE_BOUNCE:
                resolved, arbiter, reason = (
                    LandCoverClass.BUILT_UP, "sar", _REASON_SAR_ASSERTS,
                )
            elif label is LandCoverClass.BUILT_UP and sar_decides_built_up:
                resolved, arbiter, reason = (
                    LandCoverClass.BARE_SOIL, "sar", _REASON_SAR_WITHDRAWS,
                )
            elif label is LandCoverClass.BUILT_UP:
                resolved, arbiter, reason = label, "none", _REASON_UNRESOLVED
            else:
                resolved, arbiter, reason = label, "optical", _REASON_OPTICAL_STANDS

            fused[group] = resolved.code
            groups.append(
                ConflictGroup(
                    optical_class=label,
                    sar_regime=regime,
                    pixel_count=count,
                    fraction=count / total if total else 0.0,
                    area_m2=count * px_area if px_area is not None else None,
                    resolved_as=resolved,
                    arbiter=arbiter,
                    reason=reason,
                    candidate_surfaces=regime.candidate_surfaces,
                )
            )
    return sorted(groups, key=lambda g: g.pixel_count, reverse=True)


def _fused_stats(
    class_map: np.ndarray,
    evidence_map: np.ndarray,
    px_area: float | None,
) -> list[FusedClassStats]:
    """Per-class coverage of the fused map with its evidence split."""
    classified = int(np.count_nonzero(class_map != LandCoverClass.UNCLASSIFIED.code))
    both_code = CODE_TO_EVIDENCE[FusionEvidence.BOTH]

    out: list[FusedClassStats] = []
    for label in LandCoverClass:
        if label is LandCoverClass.UNCLASSIFIED:
            continue
        mask = class_map == label.code
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        codes = Counter(evidence_map[mask].tolist())
        out.append(
            FusedClassStats(
                label=label,
                pixel_count=count,
                fraction=count / classified if classified else 0.0,
                area_m2=count * px_area if px_area is not None else None,
                corroborated_fraction=codes.get(both_code, 0) / count,
                evidence_counts={
                    EVIDENCE_FROM_CODE[c]: n for c, n in sorted(codes.items())
                },
            )
        )
    return sorted(out, key=lambda s: s.fraction, reverse=True)


def fuse_optical_sar(
    optical: RasterData,
    sar: RasterData,
    *,
    optical_result: LandCoverResult | None = None,
    sar_result: SarResult | None = None,
) -> FusionResult:
    """Combine an optical and a radar acquisition of the same area into one interpretation.

    Args:
        optical: Multispectral or RGB imagery, already on a common grid with ``sar`` (see
            :func:`app.geospatial.align.align_to_reference`).
        sar: Backscatter imagery of the same area.
        optical_result: A land-cover classification of ``optical`` computed earlier. Supplied
            by the orchestrator so a scene the agent has already classified is not classified
            twice; validated against ``optical``'s shape rather than trusted.
        sar_result: Likewise for the SAR analysis of ``sar``.

    Returns:
        A :class:`FusionResult` whose areas are ``None`` when the inputs are not
        georeferenced.

    Raises:
        GeospatialError: if the two images are not on a common grid, if they share no valid
            pixel, or if a supplied pre-computed result belongs to a different raster. All
            three are refusals: there is no honest fused map in any of those cases.
    """
    if optical.shape != sar.shape:
        raise GeospatialError(
            "Cross-modal analysis requires the optical and radar images to cover the same "
            "area on the same pixel grid. Align them to a common grid first.",
            code=ErrorCode.SIZE_MISMATCH,
            context={"optical_shape": list(optical.shape), "sar_shape": list(sar.shape)},
        )

    overlap = optical.valid_mask() & sar.valid_mask()
    if not overlap.any():
        raise GeospatialError(
            "The optical and radar images have no valid overlapping pixels, so they cannot "
            "be analysed together.",
            code=ErrorCode.NO_SPATIAL_OVERLAP,
        )

    for name, given in (
        ("optical_result", None if optical_result is None else optical_result.class_map.shape),
        ("sar_result", None if sar_result is None else sar_result.regime_map.shape),
    ):
        if given is not None and given != optical.shape:
            raise GeospatialError(
                "A pre-computed analysis was supplied for a different image than the one "
                "given, so the two could not be combined.",
                code=ErrorCode.SIZE_MISMATCH,
                context={"argument": name, "result_shape": list(given),
                         "raster_shape": list(optical.shape)},
            )

    warnings: list[str] = _modality_warnings(optical, sar)

    registration = measure_registration(optical, sar)
    warnings.extend(registration.warnings)
    if registration.offset_magnitude > REGISTRATION_WARN_PX:
        warnings.append(
            f"The optical and radar images appear misaligned by about "
            f"{registration.offset_magnitude:.1f} pixels. Along the edges of fields, "
            f"buildings and water bodies the two sensors will then describe different "
            f"ground, so the reported disagreement between them is an over-estimate."
        )

    lc = optical_result if optical_result is not None else classify_land_cover(optical)
    sr = sar_result if sar_result is not None else analyze_sar(sar)
    warnings.extend(lc.warnings)
    warnings.extend(sr.warnings)

    optical_map = lc.class_map
    regime_map = sr.regime_map
    px_area, area_caveat = pixel_area_m2(optical.metadata)
    if area_caveat:
        warnings.append(area_caveat)

    optical_known = optical_map != LandCoverClass.UNCLASSIFIED.code
    sar_known = regime_map != ScatteringRegime.UNCLASSIFIED.code
    consistent = _consistency_mask(optical_map, regime_map)

    both = optical_known & sar_known & consistent
    conflict = optical_known & sar_known & ~consistent
    optical_only = optical_known & ~sar_known
    sar_only = ~optical_known & sar_known

    evidence_map = np.full(
        optical_map.shape, CODE_TO_EVIDENCE[FusionEvidence.NEITHER], dtype=np.uint8
    )
    evidence_map[both] = CODE_TO_EVIDENCE[FusionEvidence.BOTH]
    evidence_map[conflict] = CODE_TO_EVIDENCE[FusionEvidence.CONFLICT]
    evidence_map[optical_only] = CODE_TO_EVIDENCE[FusionEvidence.OPTICAL_ONLY]
    evidence_map[sar_only] = CODE_TO_EVIDENCE[FusionEvidence.SAR_ONLY]

    # Where the sensors agree, and where only optical spoke, its class carries through
    # unchanged; the rules below only touch pixels where that is not the case.
    fused = optical_map.copy()

    sar_decides_built_up = ScatteringRegime.DOUBLE_BOUNCE in sr.separated_regimes
    if not sar_decides_built_up:
        warnings.append(
            "The radar image showed no separable population of strong double-bounce returns, "
            "so radar could neither confirm nor rule out built-up land here. Any built-up "
            "area reported comes from the optical bands alone and is not corroborated."
        )

    conflicts = _resolve_conflicts(
        fused,
        optical_map,
        regime_map,
        conflict,
        sar_decides_built_up=sar_decides_built_up,
        px_area=px_area,
        total=optical_map.size,
    )

    # Rule C: radar alone names a class only where exactly one land cover is consistent with
    # its regime. Everywhere else the optical side was silent, the fused map stays silent too.
    refused = np.zeros(optical_map.shape, dtype=bool)
    for regime in ScatteringRegime:
        classes = regime.consistent_land_cover
        group = sar_only & (regime_map == regime.code)
        if not group.any():
            continue
        if len(classes) == 1:
            fused[group] = next(iter(classes)).code
        else:
            fused[group] = LandCoverClass.UNCLASSIFIED.code
            refused |= group
    refused_count = int(np.count_nonzero(refused))
    if refused_count:
        warnings.append(
            f"Over {refused_count} pixels the optical image carries no usable data and the "
            f"radar return is consistent with more than one surface, so no land cover is "
            f"reported there rather than guessing between them."
        )

    established = int(np.count_nonzero(optical_known & sar_known))
    agreement = (
        float(np.count_nonzero(both) / established) if established else 0.0
    )
    if established and agreement < 0.5:
        warnings.append(
            f"The two sensors disagree over {100 * (1 - agreement):.0f}% of the pixels they "
            f"both observed, which is more than they agree on. Check that both images cover "
            f"the same area and were acquired close enough in time to be comparable."
        )

    stats = _fused_stats(fused, evidence_map, px_area)
    method = "decision-level-optical-sar"
    if not sar_decides_built_up:
        method = "decision-level-optical-sar-partial"

    logger.info(
        "optical/SAR fusion complete",
        extra={
            "agreement_fraction": round(agreement, 4),
            "conflict_groups": len(conflicts),
            "registration_offset_px": round(registration.offset_magnitude, 3),
            "sar_decides_built_up": sar_decides_built_up,
            "optical_method": lc.method,
        },
    )

    return FusionResult(
        class_map=fused,
        evidence_map=evidence_map,
        stats=stats,
        conflicts=conflicts,
        registration=registration,
        optical=lc,
        sar=sr,
        agreement_fraction=agreement,
        method=method,
        warnings=list(dict.fromkeys(warnings)),
    )
