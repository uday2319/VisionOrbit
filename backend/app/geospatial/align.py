"""Bringing rasters onto a common grid, measuring how well they line up, and correcting it.

Three responsibilities, deliberately separated:

* :func:`align_to_reference` puts the second raster on the first's *grid* — by coordinates
  when both sides are georeferenced, by pixel resize when they are not. Being on the same grid
  is not the same as lining up: a resize matches the array shape and says nothing about whether
  the ground under pixel (100, 100) is the same in both.

* :func:`measure_registration` measures whether they actually line up, via gradient-domain
  phase correlation plus normalised cross-correlation. The brief forbids claiming that
  co-registration is correct without validating it, so this is a measurement that flows into
  the confidence score. If the images are misaligned, the system says so.

* :func:`coregister` *corrects* a measured misalignment, and this is the part that used to be
  missing. Measuring an 82-pixel offset and then differencing the pair anyway reports the
  misalignment itself as change at every land-cover boundary. Correction is coarse-to-fine
  (a Gaussian-pyramid phase correlation, because plain full-resolution phase correlation
  cannot recover a shift of tens of pixels) with an independent ORB feature-matching estimate
  as a second candidate. Crucially the correction is **verified, never assumed**: every
  candidate shift is applied and the registration re-measured, and a candidate is kept only if
  the re-measured quality actually improved on doing nothing. A correction that does not help
  is discarded and said so.

:func:`registration_gate` then turns the final measurement into an explicit pass/fail on
whether the pair is aligned well enough to support *semantic* claims — that a particular
land-cover class became another one. A pair can be good enough to say "something differs here"
and not good enough to say "vegetation became bare soil", and those two verdicts are reported
separately rather than collapsed into one confidence number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from ..core.errors import ErrorCode, GeospatialError
from ..core.logging import get_logger
from .raster import RasterData, RasterMetadata
from .validate import AlignmentStrategy

logger = get_logger(__name__)

# Resampling choice matters: averaging is right for continuous radiometry when
# downsampling, bilinear for upsampling. Nearest is reserved for label rasters.
_CONTINUOUS_UP = Resampling.bilinear
_CONTINUOUS_DOWN = Resampling.average

# Above this measured offset, any analysis that compares two images pixel-by-pixel starts
# reporting the misalignment itself. One pixel is the natural place to draw the line: below it,
# disagreement along a feature boundary is dominated by genuine mixed pixels, and above it by
# the two grids simply not describing the same ground. Callers share the number but phrase
# their own warning, because what the misalignment corrupts differs — change detection
# over-reports changed area, optical/SAR fusion over-reports sensor conflict.
REGISTRATION_WARN_PX = 1.0

# --- thresholds for the semantic-claim gate (§3 point 5) ---------------------------------------
# A pair can be well enough registered to support "something differs in this area" and not well
# enough to support "this class became that class". The second claim compares two *per-pixel
# classifications*, so a residual offset does not merely blur it — it systematically converts one
# side of every land-cover boundary into the class on the other side, which is precisely the shape
# of a spurious transition. These three thresholds are the gate on that stronger claim.

SEMANTIC_MAX_OFFSET_PX = 2.0
"""Residual offset above which per-pixel class transitions are not attributable to the ground.

Two pixels is where a boundary displacement stops being explainable as mixed pixels: at 1 px the
disagreement along an edge is one pixel wide and dominated by genuine mixture, at 2 px it is a
two-pixel ribbon of manufactured transition following every boundary in the scene.
"""

SEMANTIC_MIN_NCC = 0.35
"""Structural correlation below which the two grids cannot be shown to describe the same ground.

Independent of offset: a pair can report a small offset simply because phase correlation found no
peak worth reporting. Requiring measured common structure is what distinguishes "aligned" from
"no evidence of misalignment".
"""

SEMANTIC_MIN_SCORE = 0.45
"""Aggregate registration score below which semantic transitions are withheld."""

# --- co-registration correction ----------------------------------------------------------------

COREGISTER_MIN_OFFSET_PX = 0.25
"""Below this measured offset there is nothing worth correcting; resampling would only blur."""

COREGISTER_MAX_LEVELS = 6
"""Gaussian-pyramid depth for the coarse-to-fine shift search.

Each level halves the resolution, so the recoverable shift roughly doubles: six levels take the
search range from the few pixels plain phase correlation manages to the order of a hundred, which
is the regime an ungeoreferenced pair of screenshots actually lands in.
"""

_PYRAMID_MIN_EDGE = 32
"""Stop coarsening below this edge length — phase correlation on a tiny array is noise."""

_ORB_FEATURES = 2000
_ORB_MIN_MATCHES = 12
_ORB_MAD_TO_SIGMA = 1.4826
_ORB_INLIER_SIGMA = 2.5


@dataclass(frozen=True)
class RegistrationGate:
    """Explicit verdict on whether a pair is aligned well enough for semantic change claims.

    Separated from :class:`RegistrationQuality` because the two answer different questions. The
    quality object says *how well* the images line up; this says *what may therefore be claimed*.
    Both the thresholds and the measurements behind the verdict are carried, so a refusal can be
    shown rather than asserted.
    """

    passed: bool
    reasons: list[str]
    measured: dict[str, float]

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "max_offset_px": SEMANTIC_MAX_OFFSET_PX,
            "min_ncc": SEMANTIC_MIN_NCC,
            "min_score": SEMANTIC_MIN_SCORE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "measured": dict(self.measured),
            "thresholds": self.thresholds,
        }


def registration_gate(registration: RegistrationQuality) -> RegistrationGate:
    """Decide whether ``registration`` supports per-pixel semantic transition claims.

    All three thresholds must hold. Failing the gate is *not* a failure of the analysis: the
    measured pixel difference is still a real observation and is still reported. What the gate
    withholds is the interpretation of that difference as a named land-cover conversion.
    """
    offset = registration.offset_magnitude
    reasons: list[str] = []
    if offset > SEMANTIC_MAX_OFFSET_PX:
        reasons.append(
            f"A residual misalignment of {offset:.1f} px remains after co-registration "
            f"(the limit for per-pixel class comparison is {SEMANTIC_MAX_OFFSET_PX:.1f} px). "
            f"At this offset a land-cover boundary in one date sits on top of the neighbouring "
            f"class in the other, so a transition would be manufactured along every boundary."
        )
    if registration.ncc < SEMANTIC_MIN_NCC:
        reasons.append(
            f"Structural agreement between the two dates is {registration.ncc:.2f}, below the "
            f"{SEMANTIC_MIN_NCC:.2f} needed to establish that the two grids describe the same "
            f"ground."
        )
    if registration.score < SEMANTIC_MIN_SCORE:
        reasons.append(
            f"Overall co-registration quality is {registration.score:.2f}, below the "
            f"{SEMANTIC_MIN_SCORE:.2f} required before a class-to-class conversion can be "
            f"attributed to the ground rather than to the alignment."
        )
    return RegistrationGate(
        passed=not reasons,
        reasons=reasons,
        measured={
            "offset_magnitude_px": round(offset, 3),
            "ncc": round(registration.ncc, 4),
            "score": round(registration.score, 4),
        },
    )


@dataclass
class RegistrationQuality:
    """Measured spatial agreement between two co-located rasters."""

    offset_x: float
    """Estimated horizontal misalignment in pixels (sub-pixel precision)."""

    offset_y: float
    """Estimated vertical misalignment in pixels."""

    phase_response: float
    """Phase-correlation peak strength in ``[0, 1]``; higher means a clearer common signal."""

    ncc: float
    """Normalised cross-correlation of gradient magnitudes in ``[-1, 1]``."""

    score: float
    """Aggregate 0..1 registration quality."""

    warnings: list[str] = field(default_factory=list)

    @property
    def offset_magnitude(self) -> float:
        return math.hypot(self.offset_x, self.offset_y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset_x_px": round(self.offset_x, 3),
            "offset_y_px": round(self.offset_y, 3),
            "offset_magnitude_px": round(self.offset_magnitude, 3),
            "phase_response": round(self.phase_response, 4),
            "ncc": round(self.ncc, 4),
            "score": round(self.score, 4),
            "warnings": list(self.warnings),
        }


@dataclass
class AlignmentReport:
    """What was done to bring a raster onto the reference grid."""

    strategy: AlignmentStrategy
    resampling: str
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    reprojected: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "resampling": self.resampling,
            "source_shape": list(self.source_shape),
            "target_shape": list(self.target_shape),
            "reprojected": self.reprojected,
            "warnings": list(self.warnings),
        }


def _representative_band(raster: RasterData) -> np.ndarray:
    """A single 2-D band summarising a raster, for correlation-based comparisons.

    Uses the mean across bands (ignoring NaN) so the estimate is not dominated by a single
    noisy channel.
    """
    band = raster.band_mean()
    return np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _normalize01(a: np.ndarray) -> np.ndarray:
    """Percentile-stretch to ``[0, 1]``, robust to outliers."""
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _gradient_magnitude(a: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude.

    Comparing gradients rather than raw intensity is what lets optical and SAR imagery be
    correlated at all: their absolute brightness is physically unrelated, but their
    *structural edges* (coastlines, field boundaries, roads) coincide.
    """
    norm = _normalize01(a)
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def measure_registration(a: RasterData, b: RasterData) -> RegistrationQuality:
    """Measure how well two same-shaped rasters line up.

    Both rasters must already be on a common grid (call :func:`align_to_reference` first).

    Raises:
        GeospatialError: if shapes differ.
    """
    if a.shape != b.shape:
        raise GeospatialError(
            "Registration quality can only be measured after the images share a grid.",
            code=ErrorCode.SIZE_MISMATCH,
            context={"a": list(a.shape), "b": list(b.shape)},
        )

    ga = _gradient_magnitude(_representative_band(a))
    gb = _gradient_magnitude(_representative_band(b))

    # Hanning window suppresses edge effects that otherwise bias phase correlation.
    warnings: list[str] = []
    try:
        window = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(ga.astype(np.float64),
                                                gb.astype(np.float64),
                                                window.astype(np.float64))
    except cv2.error as exc:  # pragma: no cover - degenerate input
        logger.warning("phase correlation failed", extra={"error": str(exc)})
        dx, dy, response = 0.0, 0.0, 0.0
        warnings.append("Sub-pixel alignment could not be estimated for these images.")

    # NCC on gradient magnitudes: structural agreement independent of brightness scale.
    fa = ga.ravel() - ga.mean()
    fb = gb.ravel() - gb.mean()
    denom = float(np.linalg.norm(fa) * np.linalg.norm(fb))
    ncc = float(np.dot(fa, fb) / denom) if denom > 1e-12 else 0.0

    offset_mag = math.hypot(dx, dy)
    if offset_mag > 5.0:
        warnings.append(
            f"The images appear misaligned by about {offset_mag:.1f} pixels. "
            f"Results near feature boundaries may be unreliable."
        )
    elif offset_mag > 2.0:
        warnings.append(
            f"A residual misalignment of about {offset_mag:.1f} pixels was measured."
        )
    if ncc < 0.15:
        warnings.append(
            "The two images share little common structure, so their co-registration "
            "could not be confirmed."
        )

    # Offsets beyond ~10 px contribute nothing; NCC is clipped at 0 (anti-correlation is
    # no better than no correlation for our purposes).
    offset_term = max(0.0, 1.0 - offset_mag / 10.0)
    score = float(np.clip(0.5 * offset_term + 0.35 * max(ncc, 0.0)
                          + 0.15 * float(np.clip(response, 0.0, 1.0)), 0.0, 1.0))

    return RegistrationQuality(
        offset_x=float(dx), offset_y=float(dy),
        phase_response=float(np.clip(response, 0.0, 1.0)),
        ncc=ncc, score=score, warnings=warnings,
    )


@dataclass
class CoregistrationResult:
    """Outcome of attempting to correct a measured misalignment.

    Attributes:
        raster: The corrected copy of the source raster, or the original when no correction was
            kept. Always safe to use downstream.
        initial: Registration measured *before* any correction.
        final: Registration measured *after* the kept correction. Equal to ``initial`` when
            nothing was applied.
        shift_x / shift_y: The translation applied, in pixels of the reference grid. Zero when
            no correction was kept.
        method: ``"none"`` (already aligned), ``"pyramid-phase-correlation"``,
            ``"orb-translation"``, or ``"rejected"`` when candidates were tried and none improved
            on doing nothing.
        candidates: Every candidate that was evaluated, with the score it achieved — the record
            that the kept shift was chosen by measurement rather than asserted.
    """

    raster: RasterData
    initial: RegistrationQuality
    final: RegistrationQuality
    shift_x: float
    shift_y: float
    method: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return self.method not in ("none", "rejected")

    @property
    def improvement(self) -> float:
        return self.final.score - self.initial.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "applied": self.applied,
            "shift_x_px": round(self.shift_x, 3),
            "shift_y_px": round(self.shift_y, 3),
            "initial": self.initial.to_dict(),
            "final": self.final.to_dict(),
            "score_improvement": round(self.improvement, 4),
            "candidates": list(self.candidates),
            "warnings": list(self.warnings),
        }


def _translate(a: np.ndarray, dx: float, dy: float, *, fill: float = 0.0) -> np.ndarray:
    """Sub-pixel translation of a 2-D array by ``(dx, dy)`` pixels (positive = right/down)."""
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        a.astype(np.float32),
        matrix,
        (a.shape[1], a.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Windowed phase correlation: ``(dx, dy, response)`` such that ``b ≈ a`` shifted by (dx, dy)."""
    try:
        window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(
            a.astype(np.float64), b.astype(np.float64), window.astype(np.float64)
        )
    except cv2.error as exc:  # pragma: no cover - degenerate input
        logger.warning("phase correlation failed", extra={"error": str(exc)})
        return 0.0, 0.0, 0.0
    return float(dx), float(dy), float(np.clip(response, 0.0, 1.0))


def _pyramid_shift(ga: np.ndarray, gb: np.ndarray) -> tuple[float, float, float]:
    """Coarse-to-fine phase-correlation shift estimate, in full-resolution pixels.

    Plain phase correlation on the full image can only recover a shift small relative to the
    dominant feature spacing — in practice a few pixels. Halving the resolution halves the shift
    in pixels, so a shift of tens of pixels becomes tractable at a coarse level, and each finer
    level then refines the residual. This is the standard construction and introduces no tunable
    quantity: the only choices are how deep to go (bounded by :data:`_PYRAMID_MIN_EDGE`) and that
    each level estimates the residual of the accumulated estimate so far.
    """
    pyr_a, pyr_b = [ga], [gb]
    for _ in range(COREGISTER_MAX_LEVELS - 1):
        if min(pyr_a[-1].shape) < _PYRAMID_MIN_EDGE * 2:
            break
        pyr_a.append(cv2.pyrDown(pyr_a[-1]))
        pyr_b.append(cv2.pyrDown(pyr_b[-1]))

    total_x = total_y = 0.0
    response = 0.0
    for level in range(len(pyr_a) - 1, -1, -1):
        scale = float(2**level)
        a_lvl, b_lvl = pyr_a[level], pyr_b[level]
        # Undo the accumulated estimate at this level's scale, then measure what is left.
        moved = _translate(b_lvl, -total_x / scale, -total_y / scale)
        dx, dy, response = _phase_shift(a_lvl, moved)
        total_x += dx * scale
        total_y += dy * scale
    return total_x, total_y, response


def _orb_shift(ga: np.ndarray, gb: np.ndarray) -> tuple[float, float, int]:
    """Translation from matched ORB keypoints, with MAD-based outlier rejection.

    An independent estimator with a completely different failure mode from phase correlation:
    it is local and sparse where phase correlation is global and dense, so it survives cases
    where a large uniform region dominates the spectrum. Restricting the model to a pure
    translation (rather than fitting a homography) is deliberate — the two dates are already on a
    common grid, so the only free parameter left is the shift, and a richer model would have more
    freedom to fit the outliers than to reject them.

    Returns:
        ``(dx, dy, inliers)`` such that ``b ≈ a`` shifted by (dx, dy). ``inliers`` is 0 when no
        usable estimate could be formed.
    """
    img_a = _to_uint8(ga)
    img_b = _to_uint8(gb)
    try:
        orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
        kp_a, des_a = orb.detectAndCompute(img_a, None)
        kp_b, des_b = orb.detectAndCompute(img_b, None)
    except cv2.error as exc:  # pragma: no cover - degenerate input
        logger.warning("ORB detection failed", extra={"error": str(exc)})
        return 0.0, 0.0, 0
    if des_a is None or des_b is None or len(kp_a) < _ORB_MIN_MATCHES or len(kp_b) < _ORB_MIN_MATCHES:
        return 0.0, 0.0, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_a, des_b)
    if len(matches) < _ORB_MIN_MATCHES:
        return 0.0, 0.0, 0

    deltas = np.array(
        [
            (kp_b[m.trainIdx].pt[0] - kp_a[m.queryIdx].pt[0],
             kp_b[m.trainIdx].pt[1] - kp_a[m.queryIdx].pt[1])
            for m in matches
        ],
        dtype=np.float64,
    )
    median = np.median(deltas, axis=0)
    residual = np.linalg.norm(deltas - median, axis=1)
    mad = float(np.median(residual))
    tolerance = max(mad * _ORB_MAD_TO_SIGMA * _ORB_INLIER_SIGMA, 1.0)
    inliers = deltas[residual <= tolerance]
    if inliers.shape[0] < _ORB_MIN_MATCHES:
        return 0.0, 0.0, 0
    dx, dy = (float(v) for v in np.mean(inliers, axis=0))
    return dx, dy, int(inliers.shape[0])


def _to_uint8(a: np.ndarray) -> np.ndarray:
    """Percentile-stretch a float array into the 8-bit range ORB requires."""
    return (_normalize01(a) * 255.0).astype(np.uint8)


def _shift_raster(src: RasterData, dx: float, dy: float) -> RasterData:
    """Translate every band of ``src`` by ``(dx, dy)`` pixels, preserving the invalid mask.

    The vacated border becomes NaN rather than zero: a shifted image genuinely has no data where
    it moved in from, and filling it with a plausible value would create pixels that appear
    measured but were invented.
    """
    out = np.empty_like(src.data, dtype=np.float32)
    for i in range(src.data.shape[0]):
        band = src.data[i]
        invalid = ~np.isfinite(band)
        filled = np.where(invalid, 0.0, band).astype(np.float32)
        moved = _translate(filled, dx, dy)
        # Translate the invalid mask the same way, and treat anything the shift brought in from
        # outside the frame as invalid too.
        mask = _translate(invalid.astype(np.float32), dx, dy, fill=1.0)
        inside = _translate(np.ones_like(filled), dx, dy, fill=0.0)
        out[i] = np.where((mask > 0.5) | (inside < 0.5), np.nan, moved)
    return _rebuild(src, out, src.metadata)


def coregister(src: RasterData, ref: RasterData) -> CoregistrationResult:
    """Correct a measured misalignment of ``src`` against ``ref``, keeping only what verifies.

    Both rasters must already be on a common grid (:func:`align_to_reference`). The measured
    offset is corrected by whichever of two independent estimators — coarse-to-fine phase
    correlation, or ORB feature matching — produces the best *re-measured* registration quality.
    Doing nothing is always one of the candidates, so a correction is applied only when it is
    demonstrably better than leaving the pair alone.

    Raises:
        GeospatialError: if the shapes differ.
    """
    initial = measure_registration(src, ref)
    if initial.offset_magnitude <= COREGISTER_MIN_OFFSET_PX:
        return CoregistrationResult(
            raster=src, initial=initial, final=initial, shift_x=0.0, shift_y=0.0, method="none",
            candidates=[{"method": "none", "score": round(initial.score, 4)}],
        )

    ga = _gradient_magnitude(_representative_band(ref))
    gb = _gradient_magnitude(_representative_band(src))

    # `_phase_shift(ga, gb)` gives the shift by which src sits *ahead* of ref, so the correction
    # that brings src back onto ref is its negation.
    proposals: list[tuple[str, float, float]] = []
    px, py, _ = _pyramid_shift(ga, gb)
    if math.hypot(px, py) > COREGISTER_MIN_OFFSET_PX:
        proposals.append(("pyramid-phase-correlation", -px, -py))
    ox, oy, inliers = _orb_shift(ga, gb)
    if inliers and math.hypot(ox, oy) > COREGISTER_MIN_OFFSET_PX:
        proposals.append(("orb-translation", -ox, -oy))

    candidates: list[dict[str, Any]] = [
        {"method": "none", "shift_x_px": 0.0, "shift_y_px": 0.0, "score": round(initial.score, 4)}
    ]
    best = ("none", 0.0, 0.0, initial, src)
    for method, dx, dy in proposals:
        shifted = _shift_raster(src, dx, dy)
        try:
            measured = measure_registration(shifted, ref)
        except GeospatialError:  # pragma: no cover - shapes are preserved by _shift_raster
            continue
        candidates.append(
            {
                "method": method,
                "shift_x_px": round(dx, 3),
                "shift_y_px": round(dy, 3),
                "score": round(measured.score, 4),
                "residual_offset_px": round(measured.offset_magnitude, 3),
            }
        )
        if measured.score > best[3].score:
            best = (method, dx, dy, measured, shifted)

    method, dx, dy, final, raster = best
    warnings: list[str] = []
    if method == "none":
        method = "rejected" if proposals else "none"
        warnings.append(
            f"An offset of about {initial.offset_magnitude:.1f} px was measured between the two "
            f"images, and automatic co-registration could not improve on it. The two images may "
            f"not cover the same ground, or may share too little structure to be aligned."
        )
    else:
        warnings.append(
            f"The two images were co-registered by {method}: a shift of "
            f"({dx:+.2f}, {dy:+.2f}) px reduced the measured offset from "
            f"{initial.offset_magnitude:.1f} px to {final.offset_magnitude:.2f} px and raised "
            f"registration quality from {initial.score:.2f} to {final.score:.2f}."
        )
    return CoregistrationResult(
        raster=raster, initial=initial, final=final, shift_x=dx, shift_y=dy,
        method=method, candidates=candidates, warnings=warnings,
    )


def align_to_reference(
    src: RasterData,
    ref: RasterData,
    strategy: AlignmentStrategy,
) -> tuple[RasterData, AlignmentReport]:
    """Warp/resample ``src`` onto ``ref``'s grid.

    Returns:
        The aligned copy of ``src`` and a report describing what was done.

    Raises:
        GeospatialError: if reprojection fails or the strategy is ``INCOMPATIBLE``.
    """
    warnings: list[str] = []
    src_shape = src.shape

    if strategy is AlignmentStrategy.INCOMPATIBLE:
        raise GeospatialError(
            "The two images cannot be placed on a common grid.",
            code=ErrorCode.NO_SPATIAL_OVERLAP,
        )

    if strategy is AlignmentStrategy.IDENTICAL_GRID and src.shape == ref.shape:
        return src, AlignmentReport(strategy, "none", src_shape, ref.shape, False)

    upsampling = (ref.height * ref.width) > (src.height * src.width)
    resampling = _CONTINUOUS_UP if upsampling else _CONTINUOUS_DOWN

    # --- true reprojection when both sides are georeferenced ---
    if strategy in (AlignmentStrategy.REPROJECT, AlignmentStrategy.RESAMPLE):
        # `is_georeferenced` already implies both transforms are present; binding them here
        # makes that invariant explicit at the point of use instead of implied by a property.
        src_tf, ref_tf = src.metadata.transform, ref.metadata.transform
        if (
            src.metadata.is_georeferenced
            and ref.metadata.is_georeferenced
            and src_tf is not None
            and ref_tf is not None
        ):
            out = np.full((src.data.shape[0], ref.height, ref.width), np.nan, dtype=np.float32)
            try:
                for i in range(src.data.shape[0]):
                    reproject(
                        source=src.data[i],
                        destination=out[i],
                        src_transform=Affine(*src_tf),
                        src_crs=CRS.from_wkt(src.metadata.crs_wkt),
                        dst_transform=Affine(*ref_tf),
                        dst_crs=CRS.from_wkt(ref.metadata.crs_wkt),
                        resampling=resampling,
                        src_nodata=np.nan,
                        dst_nodata=np.nan,
                    )
            except Exception as exc:
                raise GeospatialError(
                    "The second image could not be reprojected onto the first image's "
                    "coordinate system.",
                    code=ErrorCode.REPROJECTION_FAILED,
                    context={"error": str(exc)},
                ) from exc

            aligned = _rebuild(src, out, ref.metadata)
            return aligned, AlignmentReport(
                strategy, resampling.name, src_shape, ref.shape, True, warnings
            )

        warnings.append(
            "Georeferencing was incomplete, so the images were matched by pixel grid "
            "rather than by coordinates."
        )

    # --- pixel-space resize fallback ---
    out = np.empty((src.data.shape[0], ref.height, ref.width), dtype=np.float32)
    interp = cv2.INTER_LINEAR if upsampling else cv2.INTER_AREA
    for i in range(src.data.shape[0]):
        band = src.data[i]
        # cv2 has no NaN handling: substitute, resize, then restore the invalid region.
        nan_mask = ~np.isfinite(band)
        filled = np.where(nan_mask, 0.0, band).astype(np.float32)
        resized = cv2.resize(filled, (ref.width, ref.height), interpolation=interp)
        if nan_mask.any():
            mask_resized = cv2.resize(
                nan_mask.astype(np.float32), (ref.width, ref.height),
                interpolation=cv2.INTER_NEAREST,
            )
            resized = np.where(mask_resized > 0.5, np.nan, resized)
        out[i] = resized

    aligned = _rebuild(src, out, ref.metadata)
    return aligned, AlignmentReport(
        strategy, "bilinear" if upsampling else "area",
        src_shape, ref.shape, False, warnings,
    )


def _rebuild(src: RasterData, data: np.ndarray, ref_meta: RasterMetadata) -> RasterData:
    """Clone ``src`` with new pixel data and the reference grid's geometry.

    Provenance (path, driver, band descriptions) stays with the source; geometry comes from
    the reference, because that is now the grid the data lives on.
    """
    meta = RasterMetadata(
        path=src.metadata.path,
        driver=src.metadata.driver,
        width=int(data.shape[2]),
        height=int(data.shape[1]),
        count=int(data.shape[0]),
        dtype=str(data.dtype),
        crs_wkt=ref_meta.crs_wkt,
        crs_epsg=ref_meta.crs_epsg,
        transform=ref_meta.transform,
        bounds=ref_meta.bounds,
        nodata=src.metadata.nodata,
        band_descriptions=src.metadata.band_descriptions,
        tags=src.metadata.tags,
        decimation=src.metadata.decimation,
    )
    return RasterData(
        data=data, metadata=meta, band_roles=list(src.band_roles), modality=src.modality
    )


def normalize_radiometry(
    target: RasterData, reference: RasterData
) -> tuple[RasterData, list[str]]:
    """Relative radiometric normalisation of ``target`` towards ``reference``.

    Matches per-band mean and standard deviation over jointly-valid pixels. Without this,
    differencing two dates conflates genuine surface change with illumination, atmospheric
    and sensor-gain differences — producing "change" everywhere.

    Returns:
        The normalised copy and any warnings.
    """
    warnings: list[str] = []
    if target.shape != reference.shape:
        raise GeospatialError(
            "Radiometric normalisation requires images on a common grid.",
            code=ErrorCode.SIZE_MISMATCH,
        )
    if target.data.shape[0] != reference.data.shape[0]:
        warnings.append(
            "The two images have different band counts; radiometric normalisation was "
            "applied only to the bands they share."
        )

    n_bands = min(target.data.shape[0], reference.data.shape[0])
    out = target.data.copy()
    joint_valid = target.valid_mask() & reference.valid_mask()

    if np.count_nonzero(joint_valid) < 100:
        warnings.append(
            "Too few jointly valid pixels to normalise brightness between the two images; "
            "differences may partly reflect illumination rather than surface change."
        )
        return target, warnings

    for i in range(n_bands):
        t_vals = target.data[i][joint_valid]
        r_vals = reference.data[i][joint_valid]
        t_std = float(np.std(t_vals))
        r_std = float(np.std(r_vals))
        t_mean = float(np.mean(t_vals))
        r_mean = float(np.mean(r_vals))
        if t_std < 1e-9:
            warnings.append(f"Band {i + 1} of the second image is constant; left unscaled.")
            continue
        if r_std < 1e-9:
            # The gain factor would be r_std/t_std = 0, collapsing the target to the single
            # value `r_mean` — every difference erased, silently, producing a confidently
            # empty change map. A constant reference band carries no radiometric range to
            # match, so only the offset is applied and the target's own contrast is kept.
            warnings.append(
                f"Band {i + 1} of the first image has no variation, so brightness between "
                f"the two images could only be matched on average, not on contrast."
            )
            out[i] = target.data[i] - t_mean + r_mean
            continue
        out[i] = (target.data[i] - t_mean) * (r_std / t_std) + r_mean

    return _rebuild(target, out, target.metadata), warnings
