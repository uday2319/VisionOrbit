"""Bi-temporal change analysis (brief §3).

Change is measured by two independent detectors whose agreement is itself reported. The first
is Change Vector Analysis: build the same small stack of spectral indices for both dates,
subtract, and take the magnitude of the resulting per-pixel change vector. The second is
disagreement between the two dates' land-cover classifications. Several decisions in here are
the difference between a real result and a plausible-looking one:

* **Why two detectors rather than one.** Spectral CVA is blind to conversions between built-up
  land and bare soil: the two are near-identical in NDVI, a water index and NDBI, and separate
  only on texture. On this repo's demo pair, CVA recovered 0 of 5,800 bare-soil-to-built-up
  pixels and 0 of 864 in the opposite direction — together the *entire* recall gap. The
  classifier already uses edge density as its built-up cue, so class disagreement does see
  those conversions. Unioning the two took recall from 0.731 to 0.918 and F1 from 0.840 to
  0.868 with no new tunable quantity. The alternative considered and rejected was adding a
  scaled edge-density axis to the change vector: it reached F1 0.892, but only at a scale
  factor found by trying values against the ground truth, which is fitting to the test set
  rather than measuring on it. (The choice of vector norm was checked and is *not* such a
  factor — an L2 norm and an RMS mean give byte-identical masks, because the √n between them
  cancels against a threshold derived from the same histogram.)

* **Detector agreement is reported per transition, because it predicts correctness.** Within
  the union mask on the demo pair, pixels both detectors flagged were 98.7% correct against
  ground truth (n=18,739); pixels only the classifier flagged were 50.0% correct (n=9,456).
  That gap is measured, not assumed, so :attr:`Transition.spectral_agreement` is carried on
  every transition and the scene-level split is reported. It lets the confidence layer
  discount exactly the transitions that deserve it instead of presenting a near-coin-flip
  conversion with the same authority as a corroborated one.

* **Indices, not raw bands, form the change vector.** NDVI, a water index and NDBI are all
  natively bounded to ``[-1, 1]``, so their differences are already commensurate and can be
  combined without inventing per-axis weights. Raw reflectance bands are not — combining a
  bright SWIR band with a dark blue one silently weights the analysis towards whichever has
  the larger numeric range. Raw bands remain available as a fallback, but only with stretch
  bounds derived *jointly* from both dates: normalising each date independently would rescale
  the change away, which is the classic way to produce a confidently empty change map.

* **Radiometric normalisation is applied to the raw-band fallback only, never to the index
  path.** This is the opposite of the obvious choice and was settled by measurement. A
  normalised difference index is already a ratio, so it is invariant to the multiplicative
  gain that dominates illumination differences between dates — indices are self-normalising,
  which is why they are the standard multitemporal instrument. Layering a per-band affine
  mean/variance match on top of that does not help and actively destroys the dark end: on
  this repo's demo pair it drove stable water's SWIR1 to −59 (a physically impossible
  reflectance), flipping NDBI there from −0.30 to +0.79 and manufacturing a change of 1.17 in
  an index bounded to ``[-1, 1]``. Every pixel of the lake was then reported as changed.
  Removing the step took precision from 0.695 to 0.988 with recall unmoved. Raw-band
  differencing does still need the correction, because it compares absolute values, so it
  keeps it.

* **The per-date classifications use the original imagery.** They must, for the same reason:
  a land-cover classifier reads band ratios, and rescaling one date towards the other would
  make each date's land cover partly a function of the other date.

* **The threshold is derived from the image, and the fallback is not a literature constant.**
  There is no published universal CVA magnitude threshold — magnitude units depend entirely
  on which indices went into the stack. Otsu is used when the magnitude histogram is
  genuinely bimodal (:func:`indices.adaptive_threshold` gates on Sarle's coefficient), and
  otherwise the cut is an outlier bound on the unchanged population: ``median + k·σ`` with σ
  estimated robustly from the MAD. That is the statistically correct instrument for the shape
  this histogram usually has — a large near-zero mode with a right tail — where Otsu has no
  valley to find.

* **A class transition is reported only where both dates classified the pixel.** Pixels inside
  the change mask whose land-cover class is unchanged are reported as spectral change, not
  converted into a fabricated transition, and a pixel unclassified on either date is labelled
  ``OTHER`` rather than assigned a meaning. The class-disagreement detector likewise ignores
  pixels that either date left unclassified, so an unclassified region can never manufacture
  change on its own.

Evidence strength is deliberately *not* the separability of the magnitude field with respect
to the change mask. That number is circular — the mask is a threshold on that very field, so
the two groups differ by construction and the measure reads 1.000 on any input whatsoever,
including pure noise. What is reported instead is Sarle's bimodality coefficient of the
magnitude histogram together with Otsu's between-class variance ratio, both computed before
any mask exists, plus the measured registration quality. Those can genuinely come out low.

Areas are ``None`` when the input is not georeferenced, and the minimum change-patch size is
expressed in m² so it means the same thing at 10 m and 30 m resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import ErrorCode, GeospatialError
from ..core.logging import get_logger
from ..core.types import BandRole, ChangeType, LandCoverClass
from ..geospatial.align import (
    REGISTRATION_WARN_PX,
    RegistrationGate,
    RegistrationQuality,
    measure_registration,
    normalize_radiometry,
    registration_gate,
)
from ..geospatial.measure import pixel_area_m2, pixels_for_ground_area
from ..geospatial.raster import RasterData, band_mean
from . import features, indices, landcover

logger = get_logger(__name__)

# Indices that make up the change vector, in preference order. Each is bounded to [-1, 1] and
# each tracks a different surface property, so together they span the land-cover space the
# classifier works in without any two axes measuring the same thing.
_FEATURE_PREFERENCE: tuple[tuple[str, ...], ...] = (
    ("ndvi",),
    ("mndwi", "ndwi"),  # MNDWI first: SWIR separates water from shadow
    ("ndbi",),
)

# Multiple of the robustly-estimated standard deviation of the unchanged population above
# which a pixel's change is not explainable as noise. 3σ is the conventional scientific
# "significantly different" bound (p < 0.003 under a normal no-change model); it is a
# statistical criterion rather than a value tuned to make any particular scene look good.
CHANGE_SIGMA_K = 3.0

# 1/Φ⁻¹(0.75): scales the median absolute deviation into a standard-deviation estimate that
# agrees with σ for normal data while ignoring the changed tail.
_MAD_TO_SIGMA = 1.4826

# Smallest ground area credible as a distinct change patch. Below roughly a quarter of a
# hectare, a cluster of above-threshold pixels is far more likely to be residual
# misregistration along an edge than a real conversion of land.
MIN_CHANGE_PATCH_M2 = 2500.0

# Pixel floor for imagery with no usable pixel size, so the same denoising still applies.
_MIN_CHANGE_PATCH_PX = 25

# Share of changed pixels that must move the same way before an index is described as having
# risen or fallen; below it the honest word is "mixed".
_DIRECTION_DOMINANCE = 0.60

# Above this share of the reported change resting on land-cover disagreement alone, the result
# says so. Measured justification: class-only pixels scored 0.500 precision against ground
# truth where spectrally-corroborated pixels scored 0.987, so once a quarter of the reported
# change is uncorroborated the headline figure is materially softer than it looks.
_UNCORROBORATED_WARN_FRACTION = 0.25

# Spectral corroboration a named land-cover transition must reach before it is *asserted* as a
# semantic finding rather than merely listed as a measurement. The number is read straight off the
# precision measured on the demo pair and recorded on :attr:`Transition.spectral_agreement`:
# transitions above 0.9 agreement scored 0.91-1.00 precision against ground truth, while
# ``built_up -> bare_soil`` at 0.002 agreement scored 0.162. 0.60 sits in the empty middle of that
# distribution — comfortably above everything that measured badly and below everything that
# measured well — so it separates the two observed populations rather than being chosen to admit
# any particular result.
SEMANTIC_MIN_SPECTRAL_AGREEMENT = 0.60


@dataclass
class Transition:
    """One observed land-cover transition, measured over the change mask."""

    before: LandCoverClass
    after: LandCoverClass
    change_type: ChangeType
    pixel_count: int
    fraction_of_change: float
    area_m2: float | None
    mean_magnitude: float
    spectral_agreement: float
    """Share of this transition's pixels that the spectral detector also flagged.

    Independent corroboration, and a measured predictor of whether the transition is real: on
    the demo pair, transitions above 0.9 agreement scored 0.91–1.00 precision against ground
    truth, while ``built_up → bare_soil`` at 0.002 agreement scored 0.162. Reported so the
    confidence layer can discount an uncorroborated transition instead of ranking it alongside
    a corroborated one.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.before.value,
            "to": self.after.value,
            "change_type": self.change_type.value,
            "pixel_count": self.pixel_count,
            "fraction_of_change": round(self.fraction_of_change, 4),
            "percentage_of_change": round(self.fraction_of_change * 100, 2),
            "area_m2": round(self.area_m2, 2) if self.area_m2 is not None else None,
            "mean_magnitude": round(self.mean_magnitude, 4),
            "spectral_agreement": round(self.spectral_agreement, 4),
        }


@dataclass
class ClassDelta:
    """Net change in one class's extent between the two dates.

    Measured over the whole jointly-valid area rather than over the change mask, so it is an
    independent check on the transition table: the two should tell the same story.
    """

    label: LandCoverClass
    pixels_before: int
    pixels_after: int
    area_m2_before: float | None
    area_m2_after: float | None

    @property
    def pixel_delta(self) -> int:
        return self.pixels_after - self.pixels_before

    @property
    def relative_delta(self) -> float | None:
        """Change as a fraction of the earlier extent, or ``None`` if it was absent then."""
        if self.pixels_before == 0:
            return None
        return self.pixel_delta / self.pixels_before

    def to_dict(self) -> dict[str, Any]:
        area_delta = (
            self.area_m2_after - self.area_m2_before
            if self.area_m2_after is not None and self.area_m2_before is not None
            else None
        )
        return {
            "class": self.label.value,
            "pixels_before": self.pixels_before,
            "pixels_after": self.pixels_after,
            "pixel_delta": self.pixel_delta,
            "area_m2_before": (
                round(self.area_m2_before, 2) if self.area_m2_before is not None else None
            ),
            "area_m2_after": (
                round(self.area_m2_after, 2) if self.area_m2_after is not None else None
            ),
            "area_m2_delta": round(area_delta, 2) if area_delta is not None else None,
            "relative_delta": (
                round(self.relative_delta, 4) if self.relative_delta is not None else None
            ),
        }


@dataclass
class IndexDelta:
    """How one index moved, both overall and inside the detected change."""

    name: str
    mean_before: float
    mean_after: float
    mean_delta_in_change: float
    increase_fraction: float
    """Share of changed pixels where this index rose.

    Carried because the mean alone is misleading whenever a scene both gains and loses the
    same cover type: on this repo's demo pair, vegetation gain in the south and vegetation
    loss in the north-west cancel NDVI's mean delta down to +0.06, which would narrate as
    "vegetation barely moved" when in fact it moved a great deal in both directions.
    """

    @property
    def direction(self) -> str:
        """Direction of movement, or ``"mixed"`` when neither way clearly dominates.

        The threshold is on the *count* of pixels moving each way rather than on the mean,
        so an offsetting pair of changes is named as such instead of averaged into nothing.
        """
        if self.increase_fraction >= _DIRECTION_DOMINANCE:
            return "increase"
        if 1.0 - self.increase_fraction >= _DIRECTION_DOMINANCE:
            return "decrease"
        return "mixed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.name,
            "mean_before": round(self.mean_before, 4),
            "mean_after": round(self.mean_after, 4),
            "mean_delta_overall": round(self.mean_after - self.mean_before, 4),
            "mean_delta_in_change": round(self.mean_delta_in_change, 4),
            "increase_fraction": round(self.increase_fraction, 4),
            "direction": self.direction,
        }


@dataclass
class ChangeResult:
    """Full bi-temporal change analysis output."""

    change_mask: np.ndarray
    magnitude: np.ndarray
    threshold: float
    threshold_method: str
    changed_pixels: int
    valid_pixels: int
    changed_area_m2: float | None
    transitions: list[Transition]
    class_deltas: list[ClassDelta]
    index_deltas: list[IndexDelta]
    features_used: list[str]
    registration: RegistrationQuality
    bimodality: float
    """Sarle's coefficient of the magnitude histogram; > 5/9 indicates two populations.

    Computed before any threshold is applied, so unlike a mask-conditioned separability it can
    actually report weak evidence.
    """

    threshold_quality: float
    """Otsu's between-class variance ratio at the chosen cut, in ``[0, 1]``."""

    detectors: list[str]
    """Which detectors contributed, so a degraded run is visible rather than implied."""

    corroborated_pixels: int
    """Changed pixels both detectors flagged — the most reliable subset (0.987 precision)."""

    spectral_only_pixels: int
    """Changed pixels flagged spectrally with no land-cover class transition."""

    class_only_pixels: int
    """Changed pixels flagged by class disagreement alone (0.500 precision when measured)."""

    method: str
    before: landcover.LandCoverResult
    after: landcover.LandCoverResult
    warnings: list[str] = field(default_factory=list)

    @property
    def changed_fraction(self) -> float:
        return self.changed_pixels / self.valid_pixels if self.valid_pixels else 0.0

    @property
    def corroborated_fraction(self) -> float:
        """Share of detected change that both detectors independently support."""
        return self.corroborated_pixels / self.changed_pixels if self.changed_pixels else 0.0

    def dominant_transition(self) -> Transition | None:
        """The largest transition that is an actual class conversion.

        This is a *measurement*: the biggest class conversion the two classifications disagree on.
        It carries no claim that the conversion is real — use :meth:`semantic_transition` for that.
        """
        real = [t for t in self.transitions if t.change_type is not ChangeType.SPECTRAL_ONLY]
        return max(real, key=lambda t: t.pixel_count) if real else None

    # -- the measured / inferred boundary (§3 points 3, 5, 8, 9) -----------------------------
    @property
    def gate(self) -> RegistrationGate:
        """Whether co-registration is good enough to support per-pixel semantic claims.

        Derived rather than stored so it can never disagree with the registration it is about.
        """
        return registration_gate(self.registration)

    def transition_is_reportable(self, transition: Transition) -> bool:
        """Whether a transition may be *asserted*, as opposed to listed as a measurement.

        Two independent conditions, because they fail for different reasons and either one alone
        is enough to manufacture a transition that is not on the ground:

        * The pair must be registered well enough (:attr:`gate`). A residual offset converts one
          side of every land-cover boundary into the class on the other side, so at 82 px every
          boundary in the scene produces a confident-looking transition.
        * The spectral detector must independently corroborate this specific transition at
          :data:`SEMANTIC_MIN_SPECTRAL_AGREEMENT`. Two classifications can disagree because the
          surface changed, or because the two classes are hard to separate spectrally and the
          classifier landed differently on each date; only corroboration distinguishes them.
        """
        return (
            self.gate.passed
            and transition.spectral_agreement >= SEMANTIC_MIN_SPECTRAL_AGREEMENT
        )

    def semantic_transition(self) -> Transition | None:
        """The dominant transition, but only when it may honestly be asserted.

        ``None`` means "the measurement exists, the claim is withheld" — not "nothing changed".
        :meth:`semantic_withheld_reason` says which.
        """
        dominant = self.dominant_transition()
        if dominant is None:
            return None
        return dominant if self.transition_is_reportable(dominant) else None

    @property
    def semantic_supported(self) -> bool:
        """Whether any named land-cover transition may be asserted from this result."""
        return self.semantic_transition() is not None

    @property
    def semantic_withheld_reason(self) -> str | None:
        """Why the semantic claim is being withheld, or ``None`` when it is not withheld."""
        dominant = self.dominant_transition()
        if dominant is None:
            return None
        if self.semantic_supported:
            return None
        if not self.gate.passed:
            return (
                "The two images are not co-registered well enough to attribute a land-cover "
                "conversion to the ground rather than to the misalignment. "
                + " ".join(self.gate.reasons)
            )
        return (
            f"The largest measured transition ({dominant.before.value.replace('_', ' ')} to "
            f"{dominant.after.value.replace('_', ' ')}) is corroborated by the spectral "
            f"measurement on only {dominant.spectral_agreement * 100:.0f}% of its pixels, below "
            f"the {SEMANTIC_MIN_SPECTRAL_AGREEMENT * 100:.0f}% needed to attribute it to a real "
            f"surface conversion rather than to the two classifications landing differently on "
            f"spectrally similar classes."
        )

    def transitions_of_type(self, change_type: ChangeType) -> list[Transition]:
        return [t for t in self.transitions if t.change_type is change_type]

    def area_of_type(self, change_type: ChangeType) -> float | None:
        """Total area for a change type, or ``None`` when areas are unavailable."""
        matching = self.transitions_of_type(change_type)
        areas = [t.area_m2 for t in matching if t.area_m2 is not None]
        if not matching or len(areas) != len(matching):
            return None
        return float(sum(areas))

    def pixels_of_type(self, change_type: ChangeType) -> int:
        return sum(t.pixel_count for t in self.transitions_of_type(change_type))

    def to_dict(self) -> dict[str, Any]:
        dominant = self.dominant_transition()
        semantic = self.semantic_transition()
        gate = self.gate

        def transition_dict(t: Transition) -> dict[str, Any]:
            # Every transition carries whether it may be asserted, so the renderer never has to
            # re-derive the rule and the two views cannot drift apart.
            return {**t.to_dict(), "semantically_reportable": self.transition_is_reportable(t)}

        return {
            "method": self.method,
            "detectors": list(self.detectors),
            "features_used": list(self.features_used),
            "threshold": round(self.threshold, 5),
            "threshold_method": self.threshold_method,
            "changed_pixels": self.changed_pixels,
            "valid_pixels": self.valid_pixels,
            "changed_fraction": round(self.changed_fraction, 4),
            "changed_percentage": round(self.changed_fraction * 100, 2),
            "changed_area_m2": (
                round(self.changed_area_m2, 2) if self.changed_area_m2 is not None else None
            ),
            "bimodality": round(self.bimodality, 4),
            "threshold_quality": round(self.threshold_quality, 4),
            "evidence_split": {
                "corroborated_pixels": self.corroborated_pixels,
                "spectral_only_pixels": self.spectral_only_pixels,
                "class_only_pixels": self.class_only_pixels,
                "corroborated_fraction": round(self.corroborated_fraction, 4),
            },
            "dominant_transition": transition_dict(dominant) if dominant is not None else None,
            "transitions": [transition_dict(t) for t in self.transitions],
            "class_deltas": [d.to_dict() for d in self.class_deltas],
            "index_deltas": [d.to_dict() for d in self.index_deltas],
            "registration": self.registration.to_dict(),
            "registration_gate": gate.to_dict(),
            # The two halves of the answer, kept apart in the payload so the UI cannot present an
            # inference as a measurement by accident: `measured` is what the pixels say, `semantic`
            # is what may be concluded from it, with the reason when nothing may be.
            "semantic": {
                "supported": self.semantic_supported,
                "min_spectral_agreement": SEMANTIC_MIN_SPECTRAL_AGREEMENT,
                "transition": transition_dict(semantic) if semantic is not None else None,
                "withheld_reason": self.semantic_withheld_reason,
            },
            "warnings": list(self.warnings),
        }


def _robust_outlier_threshold(values: np.ndarray) -> float:
    """``median + k·σ`` with σ from the MAD, as a cut on the unchanged population.

    The median and MAD are both driven by the unchanged majority, so the bound they produce
    describes "larger than this scene's own no-change noise" — which is what a change
    threshold has to mean when the histogram has no valley for Otsu to find.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:  # pragma: no cover - callers pass a non-empty valid population
        return float("inf")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = mad * _MAD_TO_SIGMA
    if sigma <= 0.0:
        # A constant magnitude field means no measurable variation, so nothing is an outlier.
        # Returning a bound just above the median avoids flagging the entire scene.
        return float(np.nextafter(median, np.inf))
    return median + CHANGE_SIGMA_K * sigma


def _shared_index_stack(
    before: RasterData, after: RasterData
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Pick the index features both dates can supply, in preference order."""
    idx_b = indices.compute_all_available(before)
    idx_a = indices.compute_all_available(after)
    names: list[str] = []
    stack_b: list[np.ndarray] = []
    stack_a: list[np.ndarray] = []
    for group in _FEATURE_PREFERENCE:
        for name in group:
            if name in idx_b and name in idx_a:
                names.append(name)
                stack_b.append(idx_b[name].array)
                stack_a.append(idx_a[name].array)
                break  # one index per property; two water indices add no information
    return names, stack_b, stack_a


def _joint_band_stack(
    before: RasterData, after: RasterData
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Fallback stack of shared raw bands, scaled on *jointly* derived bounds.

    Both dates are divided by the same range, so a surface that genuinely brightened still
    reads as brighter. Stretching each date to its own percentiles would map both onto the
    same [0, 1] span and delete the very signal being measured.
    """
    shared = [
        role
        for role in before.band_roles
        if role is not BandRole.UNKNOWN and role in after.band_roles
    ]
    names: list[str] = []
    stack_b: list[np.ndarray] = []
    stack_a: list[np.ndarray] = []
    for role in shared:
        b = before.band(role)
        a = after.band(role)
        both = np.concatenate([b[np.isfinite(b)].ravel(), a[np.isfinite(a)].ravel()])
        if both.size == 0:
            continue
        lo, hi = (float(x) for x in np.percentile(both, (1.0, 99.0)))
        if hi - lo < 1e-12:
            continue
        names.append(role.value)
        stack_b.append(np.clip((b - lo) / (hi - lo), 0.0, 1.0))
        stack_a.append(np.clip((a - lo) / (hi - lo), 0.0, 1.0))
    return names, stack_b, stack_a


def _change_magnitude(
    stack_b: list[np.ndarray], stack_a: list[np.ndarray]
) -> np.ndarray:
    """Root-mean-square of the per-feature differences.

    The *mean* rather than the sum keeps the magnitude on the same scale whether two features
    or four were available, so the threshold and the reported numbers stay comparable across
    inputs with different band sets.
    """
    diffs = np.stack([a - b for b, a in zip(stack_b, stack_a, strict=True)], axis=0)
    return np.sqrt(band_mean(np.square(diffs))).astype(np.float32)


def _transitions(
    change_mask: np.ndarray,
    before: landcover.LandCoverResult,
    after: landcover.LandCoverResult,
    magnitude: np.ndarray,
    spectral_mask: np.ndarray,
    px_area: float | None,
) -> list[Transition]:
    """Cross-tabulate the two class maps over the changed pixels."""
    total = int(change_mask.sum())
    if total == 0:
        return []
    out: list[Transition] = []
    for label_b in LandCoverClass:
        mask_b = change_mask & before.mask_for(label_b)
        if not mask_b.any():
            continue
        for label_a in LandCoverClass:
            mask = mask_b & after.mask_for(label_a)
            count = int(mask.sum())
            if count == 0:
                continue
            mags = magnitude[mask]
            mags = mags[np.isfinite(mags)]
            out.append(
                Transition(
                    before=label_b,
                    after=label_a,
                    change_type=ChangeType.from_transition(label_b, label_a),
                    pixel_count=count,
                    fraction_of_change=count / total,
                    area_m2=count * px_area if px_area is not None else None,
                    mean_magnitude=float(np.mean(mags)) if mags.size else 0.0,
                    spectral_agreement=int((mask & spectral_mask).sum()) / count,
                )
            )
    return sorted(out, key=lambda t: t.pixel_count, reverse=True)


def _class_disagreement(
    before: landcover.LandCoverResult,
    after: landcover.LandCoverResult,
    valid: np.ndarray,
) -> np.ndarray:
    """Pixels whose land-cover class differs between the dates, where both are established.

    Deliberately restricted to pixels classified on *both* dates. A pixel that one date could
    not classify has an unknown class there, and calling unknown-to-vegetation a change would
    turn a gap in the analysis into a finding.
    """
    unknown = LandCoverClass.UNCLASSIFIED
    known = ~before.mask_for(unknown) & ~after.mask_for(unknown)
    return valid & known & (before.class_map != after.class_map)


def _class_deltas(
    before: landcover.LandCoverResult,
    after: landcover.LandCoverResult,
    valid: np.ndarray,
    px_area: float | None,
) -> list[ClassDelta]:
    """Net extent change per class over the jointly-valid area."""
    out: list[ClassDelta] = []
    for label in LandCoverClass:
        if label is LandCoverClass.UNCLASSIFIED:
            continue
        n_b = int((before.mask_for(label) & valid).sum())
        n_a = int((after.mask_for(label) & valid).sum())
        if n_b == 0 and n_a == 0:
            continue
        out.append(
            ClassDelta(
                label=label,
                pixels_before=n_b,
                pixels_after=n_a,
                area_m2_before=n_b * px_area if px_area is not None else None,
                area_m2_after=n_a * px_area if px_area is not None else None,
            )
        )
    return sorted(out, key=lambda d: abs(d.pixel_delta), reverse=True)


def _index_deltas(
    names: list[str],
    stack_b: list[np.ndarray],
    stack_a: list[np.ndarray],
    valid: np.ndarray,
    change_mask: np.ndarray,
) -> list[IndexDelta]:
    """Signed movement per feature, which is what gives change a *direction*.

    The magnitude says how much changed; these say whether vegetation went up or down. Both
    are measured, so narration never has to guess the sign.
    """
    out: list[IndexDelta] = []
    for name, b, a in zip(names, stack_b, stack_a, strict=True):
        vb = b[valid & np.isfinite(b)]
        va = a[valid & np.isfinite(a)]
        in_change = (a - b)[change_mask & np.isfinite(a) & np.isfinite(b)]
        out.append(
            IndexDelta(
                name=name,
                mean_before=float(np.mean(vb)) if vb.size else 0.0,
                mean_after=float(np.mean(va)) if va.size else 0.0,
                mean_delta_in_change=float(np.mean(in_change)) if in_change.size else 0.0,
                increase_fraction=(
                    float(np.count_nonzero(in_change > 0) / in_change.size)
                    if in_change.size
                    else 0.0
                ),
            )
        )
    return out


def detect_change(
    before: RasterData,
    after: RasterData,
    *,
    registration: RegistrationQuality | None = None,
) -> ChangeResult:
    """Analyse change between two co-registered dates of the same area.

    Args:
        before: The earlier acquisition.
        after: The later acquisition, already on ``before``'s grid (see
            :func:`app.geospatial.align.align_to_reference`).
        registration: An already-measured registration for this exact pair, when the caller has
            corrected the alignment itself (:func:`app.geospatial.align.coregister` measures the
            corrected pair as part of accepting the correction). Passing it in is not an
            optimisation — re-measuring here would produce the same numbers, but it would report
            them as if no correction had happened, losing the before/after record of what the
            co-registration achieved. Omit it and this measures the pair as given.

    Returns:
        A :class:`ChangeResult` whose ``changed_area_m2`` and per-transition areas are
        ``None`` when the inputs are not georeferenced.

    Raises:
        GeospatialError: if the two rasters are not on a common grid, or if they share no
            feature from which change could be measured. Both are refusals rather than
            degraded guesses, because there is no honest change map in either case.
    """
    if before.shape != after.shape:
        raise GeospatialError(
            "Change detection requires the two images to cover the same area on the same "
            "pixel grid. Align them to a common grid first.",
            code=ErrorCode.SIZE_MISMATCH,
            context={"before_shape": list(before.shape), "after_shape": list(after.shape)},
        )

    warnings: list[str] = []

    # Checked before any measurement runs. Beyond returning the refusal sooner, this keeps the
    # analysis from operating on all-nodata arrays, where every reduction is a mean of an empty
    # slice and the resulting NaNs would have to be untangled downstream.
    overlap = before.valid_mask() & after.valid_mask()
    if not overlap.any():
        raise GeospatialError(
            "The two images have no valid overlapping pixels, so change cannot be measured.",
            code=ErrorCode.NO_SPATIAL_OVERLAP,
        )

    if registration is None:
        registration = measure_registration(before, after)
    warnings.extend(registration.warnings)
    if registration.offset_magnitude > REGISTRATION_WARN_PX:
        warnings.append(
            f"The two images appear misaligned by about "
            f"{registration.offset_magnitude:.1f} pixels. Boundaries between land-cover "
            f"types will register as change even where the ground did not change, so the "
            f"reported change area is likely an over-estimate."
        )
    gate = registration_gate(registration)
    if not gate.passed:
        warnings.append(
            "Co-registration is not good enough to attribute a named land-cover conversion to "
            "the ground, so measured pixel difference is reported but no semantic transition is "
            "claimed. " + " ".join(gate.reasons)
        )

    # Indices are ratios and therefore already invariant to per-band gain, so they are
    # differenced on the *original* imagery. Only the raw-band fallback, which compares
    # absolute values, gets the radiometric correction — see the module docstring for the
    # measurement that settled this.
    names, stack_b, stack_a = _shared_index_stack(before, after)
    method = "cva-spectral-indices"
    if not names:
        normalized_after, norm_warnings = normalize_radiometry(after, before)
        warnings.extend(norm_warnings)
        names, stack_b, stack_a = _joint_band_stack(before, normalized_after)
        method = "cva-raw-bands"
        if names:
            warnings.append(
                "The two images share no spectral index that could be computed from both, so "
                "change was measured from raw band values scaled over both dates together. "
                "This detects that something changed but supports weaker conclusions about "
                "what changed than an index-based analysis."
            )
    if not names:
        raise GeospatialError(
            "The two images have no bands in common, so no change measurement is possible "
            "between them.",
            code=ErrorCode.MISSING_BAND,
            context={
                "before_bands": [r.value for r in before.band_roles],
                "after_bands": [r.value for r in after.band_roles],
            },
        )

    magnitude = _change_magnitude(stack_b, stack_a)
    valid = overlap & np.isfinite(magnitude)
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        raise GeospatialError(
            "No pixel in the overlapping area has a usable value in both images, so change "
            "cannot be measured.",
            code=ErrorCode.NO_SPATIAL_OVERLAP,
        )

    population = magnitude[valid]
    fallback = _robust_outlier_threshold(population)
    threshold, method_used, quality = indices.adaptive_threshold(population, fallback)
    # `adaptive_threshold` labels its fallback "literature", but this one is a robust outlier
    # bound computed from this very image. Renaming it keeps the reported provenance honest.
    threshold_method = "robust-outlier-mad" if method_used == "literature" else method_used

    px_area, area_caveat = pixel_area_m2(before.metadata)
    if area_caveat:
        warnings.append(area_caveat)

    lc_before = landcover.classify_land_cover(before)
    lc_after = landcover.classify_land_cover(after)
    for source, per_date in (("earlier", lc_before), ("later", lc_after)):
        for warning in per_date.warnings:
            warnings.append(f"{source.capitalize()} image: {warning}")

    # The same denoising is applied to each detector separately rather than to their union, so
    # every pixel in the final mask is attributable to at least one detector and the reported
    # evidence split is exhaustive — no pixel appears in a total without a reason behind it.
    min_patch = pixels_for_ground_area(
        before.metadata, MIN_CHANGE_PATCH_M2, default_pixels=_MIN_CHANGE_PATCH_PX
    )

    def denoise(raw: np.ndarray) -> np.ndarray:
        return features.clean_mask(raw, open_radius=1, close_radius=2, min_area=min_patch) & valid

    spectral_mask = denoise(valid & (magnitude > threshold))
    class_mask = denoise(_class_disagreement(lc_before, lc_after, valid))

    detectors = ["spectral-cva"]
    if class_mask.any():
        detectors.append("land-cover-disagreement")

    change_mask = spectral_mask | class_mask
    corroborated = int((spectral_mask & class_mask).sum())
    spectral_only = int((spectral_mask & ~class_mask).sum())
    class_only = int((class_mask & ~spectral_mask).sum())

    changed_pixels = int(change_mask.sum())
    if changed_pixels == 0:
        warnings.append(
            "No change patch large enough to be distinguished from image noise was found. "
            "This is a measurement of no detected change, not proof that nothing changed."
        )
    elif class_only / changed_pixels > _UNCORROBORATED_WARN_FRACTION:
        warnings.append(
            f"About {class_only / changed_pixels * 100:.0f}% of the detected change rests on "
            f"a difference between the two land-cover classifications that the spectral "
            f"measurement does not corroborate. Conversions between built-up land and bare "
            f"ground are the usual reason, because the two are hard to separate spectrally. "
            f"Treat the uncorroborated portion as substantially less certain."
        )

    result = ChangeResult(
        change_mask=change_mask,
        magnitude=magnitude,
        threshold=float(threshold),
        threshold_method=threshold_method,
        changed_pixels=changed_pixels,
        valid_pixels=valid_pixels,
        changed_area_m2=changed_pixels * px_area if px_area is not None else None,
        transitions=_transitions(
            change_mask, lc_before, lc_after, magnitude, spectral_mask, px_area
        ),
        class_deltas=_class_deltas(lc_before, lc_after, valid, px_area),
        index_deltas=_index_deltas(names, stack_b, stack_a, valid, change_mask),
        features_used=names,
        registration=registration,
        bimodality=float(indices.bimodality_coefficient(population)),
        threshold_quality=float(quality),
        detectors=detectors,
        corroborated_pixels=corroborated,
        spectral_only_pixels=spectral_only,
        class_only_pixels=class_only,
        method=method,
        before=lc_before,
        after=lc_after,
        warnings=warnings,
    )
    if px_area is None:
        result.warnings.append(
            "The imagery is not georeferenced, so changed extent is reported in pixels only; "
            "no ground area is available."
        )
    logger.debug(
        "change detected",
        extra={
            "features": names, "threshold": round(float(threshold), 5),
            "threshold_method": threshold_method, "otsu_quality": round(float(quality), 3),
            "changed_fraction": round(result.changed_fraction, 4),
            "corroborated_fraction": round(result.corroborated_fraction, 4),
            "registration_offset_px": round(registration.offset_magnitude, 2),
        },
    )
    return result
