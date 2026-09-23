"""Evidence-based confidence (brief §27).

Confidence here is **computed from measurements the analysis services already produced**, never
asked of a language model and never a fixed constant. Every number that feeds it is something a
detector reported about its own output: how separable the classes were, whether two independent
detectors corroborated each other, how well the images registered, how bimodal the change
magnitude was. The job of this module is only to *aggregate* those into a single level with the
reasons attached — an accountant, not a second opinion.

Two ideas carry the design:

1. **Some factors are gates, not contributors.** A weighted average alone cannot express "this
   one measurement being near zero should sink the whole answer regardless of everything else".
   Pure sensor noise classified twice produces a perfectly healthy-looking input image — good
   dynamic range, sharp, fully georeferenced — so an average over those would report a
   comfortable score for an answer that is worthless. The change brief (§7.1 case 2) is explicit
   that this case must resolve to ``INSUFFICIENT``. That is enforced by treating evidence strength
   as a **gate**: the final score can never exceed the weakest gate, so a corroborated fraction of
   0.00 caps the result at 0.00 no matter how clean the pixels were.

2. **The reasons are the weakest factors, in order.** "Insufficient evidence" with no explanation
   is not actionable. Every :class:`ConfidenceReport` lists its factors weakest-first and names
   the one that bound the score, so a caller can tell *why* — low separability is a different
   problem from poor registration, and the fix differs.

The gate combination is ``score = min(weighted_mean(all factors), min(gate values))``. Gates also
count in the mean, so a moderately-low gate both pulls the average down and caps it; the cap is
what makes the factor *necessary* rather than merely influential.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ..core.logging import get_logger
from ..core.types import ConfidenceLevel
from . import indices

if TYPE_CHECKING:  # imported lazily elsewhere; only needed for type checking here
    from ..geospatial.align import RegistrationQuality
    from ..geospatial.validate import InputQuality
    from .change import ChangeResult
    from .fusion import FusionResult
    from .grounding import GroundingResult
    from .landcover import LandCoverResult
    from .sar import SarResult
    from .scene import SceneClassificationResult

logger = get_logger(__name__)


class FactorKind(StrEnum):
    """Whether a factor merely influences the score or can veto it."""

    CONTRIBUTOR = "contributor"
    """Averaged into the score with its weight."""

    GATE = "gate"
    """Also caps the score: the final value cannot exceed this factor's value.

    Used for measurements a defensible answer *requires* — evidence strength, corroboration —
    so that one collapsing to near zero forces ``INSUFFICIENT`` rather than being diluted by
    healthy but irrelevant factors like image sharpness.
    """


@dataclass(frozen=True)
class ConfidenceFactor:
    """One measured input to a confidence score.

    Attributes:
        name: Stable identifier (``"evidence_strength"``, ``"registration"``, …).
        value: The measurement mapped onto ``[0, 1]``; higher is more confident.
        weight: Relative influence in the weighted mean. Gates may still carry weight.
        reason: Plain-language statement of what this factor says, for the user-facing trace.
        kind: :class:`FactorKind`.
    """

    name: str
    value: float
    weight: float
    reason: str
    kind: FactorKind = FactorKind.CONTRIBUTOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "weight": round(self.weight, 4),
            "kind": self.kind.value,
            "reason": self.reason,
        }


@dataclass
class ConfidenceReport:
    """A computed confidence level with the evidence behind it.

    Attributes:
        level: The bucketed :class:`ConfidenceLevel`.
        score: The aggregate scalar in ``[0, 1]`` the level was derived from.
        factors: Every factor considered, in the order supplied.
        limiting_factor: The name of the factor that bound the score — the gate that capped it,
            or, absent an active gate, the weakest contributor. ``None`` only when there were no
            factors at all.
        reasons: Factor reasons, weakest measurement first, so the most important caveat leads.
    """

    level: ConfidenceLevel
    score: float
    factors: list[ConfidenceFactor] = field(default_factory=list)
    limiting_factor: str | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def is_sufficient(self) -> bool:
        """False when the evidence cannot support any defensible claim."""
        return self.level is not ConfidenceLevel.INSUFFICIENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "score": round(self.score, 4),
            "sufficient": self.is_sufficient,
            "limiting_factor": self.limiting_factor,
            "reasons": list(self.reasons),
            "factors": [f.to_dict() for f in self.factors],
        }


def aggregate(factors: list[ConfidenceFactor]) -> ConfidenceReport:
    """Combine factors into a :class:`ConfidenceReport`.

    ``score = min(weighted_mean(all factors), min(gate values))``. With no factors at all the
    result is ``INSUFFICIENT`` at 0.0 — declining to answer is the honest default when nothing
    was measured, not a high-confidence one.
    """
    if not factors:
        return ConfidenceReport(
            level=ConfidenceLevel.INSUFFICIENT,
            score=0.0,
            reasons=["No evidence was available to assess confidence."],
        )

    clamped = [
        ConfidenceFactor(
            name=f.name,
            value=min(max(f.value, 0.0), 1.0),
            weight=max(f.weight, 0.0),
            reason=f.reason,
            kind=f.kind,
        )
        for f in factors
    ]

    total_weight = sum(f.weight for f in clamped)
    if total_weight <= 0.0:
        base = sum(f.value for f in clamped) / len(clamped)
    else:
        base = sum(f.value * f.weight for f in clamped) / total_weight

    gates = [f for f in clamped if f.kind is FactorKind.GATE]
    gate_cap = min((f.value for f in gates), default=1.0)
    score = min(base, gate_cap)

    # The binding constraint: the gate that capped the score if a gate is active, otherwise the
    # weakest contributor. This is what the trace names as the reason confidence is not higher.
    binding_gate = min(gates, key=lambda f: f.value, default=None)
    if binding_gate is not None and binding_gate.value < base:
        limiting = binding_gate.name
    else:
        limiting = min(clamped, key=lambda f: f.value).name

    reasons = [f.reason for f in sorted(clamped, key=lambda f: f.value)]

    return ConfidenceReport(
        level=ConfidenceLevel.from_score(score),
        score=score,
        factors=clamped,
        limiting_factor=limiting,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Shared factor builders
# ---------------------------------------------------------------------------
# Default weights, kept in one place so every service balances the same way. Evidence dominates;
# input quality and alignment matter but cannot by themselves certify a finding; warnings are a
# light nudge because most warnings are informational ("not georeferenced") rather than a defect.
_W_EVIDENCE = 0.45
_W_QUALITY = 0.25
_W_ALIGNMENT = 0.20
_W_WARNINGS = 0.10

# Weight for the change-specific "can the difference be attributed to a named land-cover
# conversion?" contributor. Deliberately a contributor and not a gate: withholding the semantic
# interpretation must not drag the confidence in the *measured* pixel difference down to
# insufficient, because the measurement is still sound. It sits between alignment and warnings
# because it qualifies the answer's interpretation rather than its measurement.
_W_ATTRIBUTION = 0.15


def _bimodality_signal(bimodality: float) -> float:
    """Map Sarle's coefficient onto ``[0, 1]`` relative to the one-population boundary.

    A normal distribution scores ``1/3`` and is the "definitely one population" anchor (→ 0); the
    uniform value ``5/9`` is where "one peak" stops being a fair description and is the
    two-population threshold used everywhere else in the codebase (→ 1). Anything flatter is
    clamped to 1. This is the same boundary :data:`indices.BIMODALITY_THRESHOLD` encodes, expressed
    as a confidence rather than a yes/no.
    """
    lo = 1.0 / 3.0
    hi = indices.BIMODALITY_THRESHOLD
    return float(min(max((bimodality - lo) / (hi - lo), 0.0), 1.0))


def _input_quality_factor(quality: InputQuality | None) -> ConfidenceFactor | None:
    """Contributor from a measured :class:`InputQuality`, or ``None`` if not assessed."""
    if quality is None:
        return None
    return ConfidenceFactor(
        name="input_quality",
        value=quality.score,
        weight=_W_QUALITY,
        reason=(
            f"Input image quality scored {quality.score:.2f} "
            f"(dynamic range {quality.dynamic_range:.2f}, "
            f"saturation {quality.saturation_fraction:.2f}, "
            f"nodata {quality.nodata_fraction:.2f})."
        ),
    )


def _registration_factor(
    registration: RegistrationQuality, *, kind: FactorKind = FactorKind.CONTRIBUTOR
) -> ConfidenceFactor:
    """Factor from measured co-registration quality.

    A gate for change (a misregistered pair reports boundary pixels as change, §7.1 case 3), a
    plain contributor for fusion where a modest offset degrades but does not invalidate the
    fused map.
    """
    return ConfidenceFactor(
        name="registration",
        value=registration.score,
        weight=_W_ALIGNMENT,
        reason=(
            f"Image co-registration scored {registration.score:.2f} "
            f"(offset {registration.offset_magnitude:.2f} px, "
            f"structural NCC {registration.ncc:.2f})."
        ),
        kind=kind,
    )


def _warning_factor(warnings: list[str]) -> ConfidenceFactor:
    """Light contributor penalising the *count* of warnings, floored so it never dominates."""
    n = len(warnings)
    value = max(1.0 - 0.15 * n, 0.4)
    reason = (
        "No processing warnings were raised."
        if n == 0
        else f"{n} processing warning(s) were raised; see the result for details."
    )
    return ConfidenceFactor(
        name="warnings", value=value, weight=_W_WARNINGS, reason=reason
    )


# ---------------------------------------------------------------------------
# Per-service adapters
# ---------------------------------------------------------------------------
def _semantic_attribution_factor(result: ChangeResult) -> ConfidenceFactor | None:
    """How well the measured difference can be attributed to a named land-cover conversion.

    ``None`` when the result names no class conversion at all — a purely spectral change has no
    attribution to score, and inventing a value for it would either flatter or penalise a result
    for a claim it never made.

    A CONTRIBUTOR, not a GATE, and that is the whole point of the separation the brief asks for:
    a well-measured pixel difference whose semantic interpretation is withheld is still a good
    measurement. Gating on attribution would collapse "we measured this reliably but will not name
    the conversion" into "we measured nothing", which is the confusion this factor exists to avoid.
    """
    dominant = result.dominant_transition()
    if dominant is None:
        return None
    reportable = result.transition_is_reportable(dominant)
    return ConfidenceFactor(
        name="semantic_attribution",
        value=float(min(max(dominant.spectral_agreement, 0.0), 1.0)),
        weight=_W_ATTRIBUTION,
        reason=(
            f"Largest measured transition ({dominant.before.value} to {dominant.after.value}) is "
            f"corroborated on {dominant.spectral_agreement * 100:.0f}% of its pixels; the named "
            f"conversion is "
            + ("reported." if reportable else "withheld as a conclusion.")
        ),
        kind=FactorKind.CONTRIBUTOR,
    )


def for_change(
    result: ChangeResult, quality: InputQuality | None = None
) -> ConfidenceReport:
    """Confidence for a bi-temporal change result.

    The evidence gate is ``min(bimodality signal, corroborated fraction)``: **both** must hold
    for a change claim to be defensible — the magnitude field must actually contain two
    populations (not one unimodal spread of noise), *and* the two independent detectors must
    corroborate. On a pure-noise pair the corroborated fraction is 0.00 and the gate forces
    ``INSUFFICIENT``, which is the behaviour §7.1 case 2 requires. Registration is a second gate,
    because a misregistered pair manufactures change at every land-cover boundary.
    """
    bim = _bimodality_signal(result.bimodality)
    corr = result.corroborated_fraction
    evidence = min(bim, corr)
    factors = [
        ConfidenceFactor(
            name="evidence_strength",
            value=evidence,
            weight=_W_EVIDENCE,
            reason=(
                f"Change evidence: magnitude bimodality {result.bimodality:.2f} "
                f"(two-population threshold {indices.BIMODALITY_THRESHOLD:.2f}), "
                f"{corr * 100:.0f}% of detected change corroborated by both detectors."
            ),
            kind=FactorKind.GATE,
        ),
        _registration_factor(result.registration, kind=FactorKind.GATE),
        ConfidenceFactor(
            name="threshold_quality",
            value=result.threshold_quality,
            weight=_W_QUALITY,
            reason=(
                f"The change threshold separated the magnitude histogram with quality "
                f"{result.threshold_quality:.2f} (method: {result.threshold_method})."
            ),
        ),
        _warning_factor(result.warnings),
    ]
    attribution = _semantic_attribution_factor(result)
    if attribution is not None:
        factors.append(attribution)
    q = _input_quality_factor(quality)
    if q is not None:
        factors.append(q)
    return aggregate(factors)


def for_landcover(
    result: LandCoverResult, quality: InputQuality | None = None
) -> ConfidenceReport:
    """Confidence for a land-cover classification.

    Class separability is the gate: if the detected classes are not distinct from their
    surroundings, the map is a guess regardless of how clean the input looked.
    """
    factors = [
        ConfidenceFactor(
            name="class_separability",
            value=result.separability,
            weight=_W_EVIDENCE,
            reason=(
                f"Mean class separability {result.separability:.2f} across the detected "
                f"classes (method: {result.method})."
            ),
            kind=FactorKind.GATE,
        ),
        _warning_factor(result.warnings),
    ]
    q = _input_quality_factor(quality)
    if q is not None:
        factors.append(q)
    return aggregate(factors)


def for_sar(result: SarResult, quality: InputQuality | None = None) -> ConfidenceReport:
    """Confidence for a single-image SAR analysis.

    Separation strength is a **contributor, not a gate**: a scene that is genuinely all diffuse
    scattering reports low mode prominence, but the diffuse reading itself is not thereby
    uncertain — the low prominence only means no *additional* regime could be split out. Gating
    on it would wrongly penalise a uniform scene. The refusal to over-claim already happened
    upstream, where a mode below the prominence floor yields no threshold at all. Speckle
    handling and input quality round out the estimate.
    """
    prominences = [
        p
        for p, present in (
            (result.smooth_prominence, result.smooth_threshold_db is not None),
            (result.bright_prominence, result.bright_threshold_db is not None),
        )
        if present
    ]
    if prominences:
        # Confidence in the *extra* regimes tracks the weakest cut that was actually made,
        # scaled so the prominence floor maps to a modest 0.5 rather than to 0.
        separation = min(
            min(p / (2.0 * indices.MIN_MODE_PROMINENCE), 1.0) for p in prominences
        )
        sep_reason = (
            f"{len(result.separated_regimes)} scattering regimes separated; weakest mode "
            f"prominence {min(prominences):.2f} (floor {indices.MIN_MODE_PROMINENCE:.2f})."
        )
    else:
        separation = 0.6
        sep_reason = (
            "Only diffuse scattering was distinguishable; no additional regime was claimed."
        )
    factors = [
        ConfidenceFactor(
            name="regime_separation",
            value=separation,
            weight=_W_EVIDENCE,
            reason=sep_reason,
        ),
        ConfidenceFactor(
            name="speckle_reduction",
            value=min(result.speckle.speckle_reduction / 2.0, 1.0),
            weight=_W_ALIGNMENT,
            reason=(
                f"Speckle filter '{result.speckle.filter_name}' reduced local variability by "
                f"{result.speckle.speckle_reduction:.2f}x "
                f"(estimated ENL {result.speckle.estimated_enl:.1f})."
            ),
        ),
        _warning_factor(result.warnings),
    ]
    q = _input_quality_factor(quality)
    if q is not None:
        factors.append(q)
    return aggregate(factors)


def for_fusion(
    result: FusionResult, quality: InputQuality | None = None
) -> ConfidenceReport:
    """Confidence for a fused optical + SAR interpretation.

    The corroborated fraction — the share of the classified map both sensors supported — is the
    gate: a fused product where the sensors agreed on almost nothing is weak even if each sensor
    was internally clean. Agreement over the co-observed pixels and registration are
    contributors.
    """
    factors = [
        ConfidenceFactor(
            name="corroboration",
            value=result.corroborated_fraction,
            weight=_W_EVIDENCE,
            reason=(
                f"{result.corroborated_fraction * 100:.0f}% of the classified scene was "
                f"supported by both sensors; {result.agreement_fraction * 100:.0f}% agreement "
                f"where both observed the same pixel."
            ),
            kind=FactorKind.GATE,
        ),
        ConfidenceFactor(
            name="sensor_agreement",
            value=result.agreement_fraction,
            weight=_W_QUALITY,
            reason=(
                f"Optical and SAR agreed on {result.agreement_fraction * 100:.0f}% of the "
                f"pixels both sensors classified."
            ),
        ),
        _registration_factor(result.registration),
        _warning_factor(result.warnings),
    ]
    q = _input_quality_factor(quality)
    if q is not None:
        factors.append(q)
    return aggregate(factors)


def for_grounding(
    result: GroundingResult, quality: InputQuality | None = None
) -> ConfidenceReport:
    """Confidence for a text-grounded region query.

    Gated on the separability of the class the regions were cut from — the answer is only as
    trustworthy as the classification under it. This scores the *extent* answer; when the region
    count is not reportable (§4.8) that is surfaced as its own reason without lowering the extent
    confidence, because withholding the count is the correct handling of a real limit, not a
    defect in the regions returned.
    """
    separability = float(result.evidence.get("class_separability", 0.0))
    factors = [
        ConfidenceFactor(
            name="class_separability",
            value=separability,
            weight=_W_EVIDENCE,
            reason=(
                f"The grounded class was separable at {separability:.2f}; "
                f"{result.count} region(s) returned from {result.class_pixels} class pixels."
            ),
            kind=FactorKind.GATE,
        ),
        _warning_factor(result.warnings),
    ]
    if not result.count_reportable:
        factors.append(
            ConfidenceFactor(
                name="count_reportable",
                value=1.0,
                weight=0.0,
                reason=(
                    "Region extent is reported; the region count is withheld as unreliable "
                    "for this class and sensor set."
                ),
            )
        )
    q = _input_quality_factor(quality)
    if q is not None:
        factors.append(q)
    return aggregate(factors)


def for_adapted_scene(
    scene: SceneClassificationResult,
    landcover: LandCoverResult | None = None,
    quality: InputQuality | None = None,
) -> ConfidenceReport:
    """Confidence for an answer that leans on the adapted learned scene classifier.

    The model's own top-1 probability is the gate: a ten-class softmax that puts 0.22 on its best
    guess has effectively declined, and no amount of clean input should let that be reported as a
    confident land-use claim. The margin over the runner-up is a separate contributor because a
    0.45/0.44 split and a 0.45/0.06 split are equally "0.45 confident" by probability alone yet mean
    very different things.

    Two further contributors keep the score tied to reality rather than to the model's self-belief:
    the model's measured held-out accuracy (included only when
    ``python -m ml.adaptation.evaluate`` has actually recorded one), and the deterministic
    classifier's class separability where a land-cover result is available, which is independent
    corroboration from a different method. Neither is a gate — a specialist model can be right about
    a scene whose spectral classes overlap — but a model that measured 40% on its own test split
    cannot certify a HIGH answer.
    """
    factors = [
        ConfidenceFactor(
            name="model_probability",
            value=scene.top.probability,
            weight=_W_EVIDENCE,
            reason=(
                f"The adapted model assigned {scene.top.probability * 100:.1f}% probability to "
                f"'{scene.top.class_name}' across {len(scene.ranked)} land-use classes."
            ),
            kind=FactorKind.GATE,
        ),
        ConfidenceFactor(
            name="model_margin",
            value=scene.margin,
            weight=_W_ATTRIBUTION,
            reason=(
                f"Margin over the runner-up class was {scene.margin * 100:.1f} points"
                + (
                    f" ({scene.ranked[1].class_name} at "
                    f"{scene.ranked[1].probability * 100:.1f}%)."
                    if len(scene.ranked) > 1
                    else "."
                )
            ),
        ),
        _warning_factor(scene.warnings),
    ]
    if scene.test_accuracy is not None:
        factors.append(
            ConfidenceFactor(
                name="model_test_accuracy",
                value=scene.test_accuracy,
                weight=_W_ATTRIBUTION,
                reason=(
                    f"This component measured {scene.test_accuracy * 100:.1f}% top-1 accuracy on "
                    "its held-out remote-sensing test split."
                ),
            )
        )
    if landcover is not None:
        factors.append(
            ConfidenceFactor(
                name="class_separability",
                value=landcover.separability,
                weight=_W_ALIGNMENT,
                reason=(
                    f"The independent deterministic classifier separated its classes at "
                    f"{landcover.separability:.2f} (method: {landcover.method}), corroborating "
                    "the scene description."
                ),
            )
        )
    q = _input_quality_factor(quality)
    if q is not None:
        factors.append(q)
    return aggregate(factors)
