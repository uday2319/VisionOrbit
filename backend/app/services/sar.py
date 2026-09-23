"""SAR backscatter analysis: speckle suppression and scattering-regime segmentation.

SAR is not a grey optical image and cannot be analysed as one. Three physical facts drive
every design decision in this module, and each was verified by measurement on the repo's demo
scene before the code was written (numbers quoted below, reproduced by ``tests/services/
test_sar.py``):

**1. Speckle is multiplicative in linear power, not additive in dB.** A backscatter raster is
almost always distributed in decibels, where speckle becomes additive-ish and the Lee
estimator's variance algebra no longer holds. So filtering converts dB → linear power, filters
there, and converts back. Doing it in dB is a common shortcut and it measurably under-filters
the dark end, which is exactly where water detection happens.

**2. The number of looks must be estimated from the image.** The nominal look count in a
product header describes the processor, not the pixels: the system impulse response correlates
neighbours, so the *effective* number of looks is higher. On the demo scene, generated with 6
looks, the estimator returns 15.1 and each of the scene's four homogeneous classes
independently measures 14.5–14.7 — using the nominal 6 would over-smooth. Estimating it
*globally* is also wrong: a whole-scene coefficient of variation is dominated by the 16 dB
contrast between classes and returns ENL 0.73, below the single-look floor. The estimate is
therefore the median of *local* CV over small windows, which holds between 14.5 and 16.1 for
every window from 5 to 11 pixels.

**3. Backscatter measures a scattering mechanism, not land cover.** Segmentation therefore
yields :class:`ScatteringRegime`, and the diffuse middle is deliberately *not* split into
vegetation vs bare soil: single-polarisation VV does not separate them. That refusal is the
honest position and it is precisely what motivates optical/SAR fusion (brief §4).

The regime boundaries come from a three-class Otsu split, gated on
:func:`indices.mode_prominence` rather than on Otsu's own between-class variance ratio. That
ratio is circular — it reads *higher* on pure Gaussian noise (0.810) than on the genuine
four-class scene (0.748), because partitioning any distribution by value explains most of its
variance. Prominence asks the non-circular question instead, and separates the two cleanly:
0.720 on the raw four-class scene and 0.981 once speckle is filtered, against 0.000 on noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..core.errors import ErrorCode, ValidationError
from ..core.logging import get_logger
from ..core.types import BandRole, Modality, ScatteringRegime
from ..geospatial.measure import pixel_area_m2
from ..geospatial.raster import RasterData
from . import indices

logger = get_logger(__name__)

_REGIME_CODE = {r: r.code for r in ScatteringRegime}
CODE_TO_REGIME = {v: k for k, v in _REGIME_CODE.items()}

# Window for the Lee filter and for local-statistics ENL estimation.
#
# Task-level accuracy does *not* select this value: measured on the demo scene, smooth-regime F1
# stays within 0.9938–0.9948 and double-bounce F1 within 0.9987–0.9990 for every window from 3
# to 11. The choice is therefore made on the quality of the filtered raster itself, on two
# figures that do vary. Residual within-class scatter falls 0.602 → 0.388 → 0.283 → 0.225 dB at
# windows 3 → 5 → 7 → 9, so the gain per step collapses (0.214, then 0.105, then 0.058) before
# reversing to 0.257 dB at window 11 as the window starts spanning class boundaries. Radiometric
# bias reaches its minimum at 7 and then degrades: max deviation from nominal backscatter is
# 0.031 dB at 7, 0.041 dB at 9 and 0.103 dB at 11. Window 9 smooths marginally better and 7
# stays truer; 7 is chosen as the smallest window at which residual scatter has essentially
# converged, since going wider trades measurable bias for smoothing that improves no task metric.
SPECKLE_WINDOW = 7

# Bounds on the estimated equivalent number of looks. Below 1 is physically impossible for an
# intensity image (single-look speckle has CV = 1); above 64 the filter is a no-op anyway, and
# such a value means the local-statistics assumption has broken down on a near-flat image.
_MIN_ENL = 1.0
_MAX_ENL = 64.0

# Fallback regime boundaries in dB, from the SAR literature for C-band VV, used only when the
# histogram does not justify a data-driven cut. Deliberately conservative: -17 dB is well below
# most rough-surface returns, and -6 dB is bright enough that little but a corner reflector
# reaches it.
LITERATURE_SMOOTH_DB = -17.0
LITERATURE_BRIGHT_DB = -6.0

# A dB raster's plausible range. Backscatter above +5 dB over a whole scene, or a "dB" image
# whose values are all positive and span thousands, is not in dB — it is linear power or raw
# DN, and treating it as dB would put every threshold in the wrong place.
_DB_PLAUSIBLE_MIN = -60.0
_DB_PLAUSIBLE_MAX = 25.0


@dataclass
class SpeckleReport:
    """What the speckle filter did, and on what estimate."""

    filter_name: str
    window: int
    estimated_enl: float
    input_cv: float
    output_cv: float
    warnings: list[str] = field(default_factory=list)

    @property
    def speckle_reduction(self) -> float:
        """Factor by which local variability fell. ``1.0`` means nothing was filtered."""
        return float(self.input_cv / self.output_cv) if self.output_cv > 1e-9 else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "filter": self.filter_name,
            "window": self.window,
            "estimated_enl": round(self.estimated_enl, 2),
            "input_cv": round(self.input_cv, 4),
            "output_cv": round(self.output_cv, 4),
            "speckle_reduction_factor": round(self.speckle_reduction, 2),
            "warnings": list(self.warnings),
        }


@dataclass
class RegimeStats:
    """Coverage of one scattering regime."""

    regime: ScatteringRegime
    pixel_count: int
    fraction: float
    area_m2: float | None
    mean_db: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "candidate_surfaces": list(self.regime.candidate_surfaces),
            "pixel_count": self.pixel_count,
            "fraction": round(self.fraction, 4),
            "percentage": round(self.fraction * 100, 2),
            "area_m2": round(self.area_m2, 2) if self.area_m2 is not None else None,
            "mean_backscatter_db": round(self.mean_db, 2),
        }


@dataclass
class SarResult:
    """Full SAR analysis output."""

    regime_map: np.ndarray
    """``(H, W)`` uint8 of :data:`_REGIME_CODE` values."""

    backscatter_db: np.ndarray
    """The filtered dB raster the segmentation was computed on."""

    stats: list[RegimeStats]
    smooth_threshold_db: float | None
    bright_threshold_db: float | None
    threshold_method: str
    smooth_prominence: float
    bright_prominence: float
    speckle: SpeckleReport
    polarization: str
    warnings: list[str] = field(default_factory=list)
    histogram: dict[str, Any] = field(default_factory=dict)

    def mask_for(self, regime: ScatteringRegime) -> np.ndarray:
        return self.regime_map == _REGIME_CODE[regime]

    def stats_for(self, regime: ScatteringRegime) -> RegimeStats | None:
        return next((s for s in self.stats if s.regime is regime), None)

    @property
    def separated_regimes(self) -> list[ScatteringRegime]:
        """Regimes an actual measurement supports — the basis for any claim downstream."""
        out = [ScatteringRegime.DIFFUSE]
        if self.smooth_threshold_db is not None:
            out.insert(0, ScatteringRegime.SMOOTH)
        if self.bright_threshold_db is not None:
            out.append(ScatteringRegime.DOUBLE_BOUNCE)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "polarization": self.polarization,
            "thresholds_db": {
                "smooth": (round(self.smooth_threshold_db, 2)
                           if self.smooth_threshold_db is not None else None),
                "bright": (round(self.bright_threshold_db, 2)
                           if self.bright_threshold_db is not None else None),
                "method": self.threshold_method,
            },
            "mode_prominence": {
                "smooth": round(self.smooth_prominence, 4),
                "bright": round(self.bright_prominence, 4),
                "threshold": indices.MIN_MODE_PROMINENCE,
            },
            "separated_regimes": [r.value for r in self.separated_regimes],
            "regimes": [s.to_dict() for s in self.stats],
            "speckle": self.speckle.to_dict(),
            "histogram": self.histogram,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------------
def looks_like_db(values: np.ndarray) -> bool:
    """Whether a backscatter array is already in decibels.

    Distinguishing dB from linear power matters because the Lee filter is only valid in linear
    power and every threshold here is quoted in dB. The test is the sign structure, not the
    magnitude: calibrated backscatter in dB is overwhelmingly negative (σ⁰ below 1), whereas
    linear power is strictly positive and usually spans two or three orders of magnitude.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    if np.any(finite < 0.0):
        return bool(finite.min() >= _DB_PLAUSIBLE_MIN and np.median(finite) <= 0.0)
    # All non-negative. Linear power, DN, or an unusual scene that is genuinely bright.
    return False


def to_linear_power(db: np.ndarray) -> np.ndarray:
    """dB → linear power. Speckle is multiplicative here, which is why filtering happens here."""
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def to_db(linear: np.ndarray) -> np.ndarray:
    """Linear power → dB, floored so that exact zeros do not become ``-inf``.

    The floor is 1e-10 (−100 dB), far below any real calibrated backscatter, so it changes no
    measurement — it only prevents an invalid value from propagating into a histogram.
    """
    return 10.0 * np.log10(np.maximum(np.asarray(linear, dtype=np.float64), 1e-10))


# ---------------------------------------------------------------------------
# Speckle
# ---------------------------------------------------------------------------
def _box_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Windowed mean with edge replication, on float32 for cv2."""
    return cv2.blur(a.astype(np.float32), (window, window),
                    borderType=cv2.BORDER_REPLICATE).astype(np.float64)


def estimate_enl(linear: np.ndarray, window: int = SPECKLE_WINDOW) -> tuple[float, list[str]]:
    """Estimate the equivalent number of looks from the image itself.

    For fully-developed speckle in an intensity image the coefficient of variation within a
    homogeneous region is ``1/√ENL``, so ENL is recoverable from measured local statistics.
    The estimator is the **median of per-window CV**, not the global CV, and that distinction
    is the whole point: measured on the demo scene, the global CV of 1.170 gives ENL 0.73 —
    below the single-look floor, and therefore physically impossible — because it is dominated
    by the 16 dB contrast *between* classes rather than speckle *within* them. The per-window
    median gives 15.1, against a nominal 6 looks, the excess being real correlation introduced
    by the system impulse response; each of the scene's four homogeneous classes independently
    measures 14.5–14.7, confirming the estimate rather than the header.

    Args:
        linear: Backscatter in **linear power**. Passing dB gives a meaningless answer, since
            CV is not scale-invariant under a logarithm.

    Returns:
        The estimate clamped to ``[1, 64]``, and any warnings about the estimate's validity.
    """
    warnings: list[str] = []
    valid = np.isfinite(linear) & (linear > 0)
    if int(valid.sum()) < window * window * 4:
        warnings.append(
            "Too little valid data to estimate the speckle level from this image; a "
            "single-look assumption was used, which filters conservatively."
        )
        return _MIN_ENL, warnings

    filled = np.where(valid, linear, np.nan)
    # nan-aware local mean/variance: sum valid values and divide by the valid count, so
    # nodata borders do not drag local statistics towards zero.
    ok = valid.astype(np.float64)
    count = _box_mean(ok, window)
    mean = np.divide(_box_mean(np.nan_to_num(filled), window), count,
                     out=np.zeros_like(count), where=count > 1e-9)
    mean_sq = np.divide(_box_mean(np.nan_to_num(filled) ** 2, window), count,
                        out=np.zeros_like(count), where=count > 1e-9)
    var = np.maximum(mean_sq - mean * mean, 0.0)

    # Only windows that are fully valid and have a real mean can report a CV.
    usable = (count > 0.999) & (mean > 1e-12) & valid
    if not usable.any():  # pragma: no cover - defensive
        warnings.append("Local speckle statistics could not be measured; assuming single-look.")
        return _MIN_ENL, warnings

    cv = np.sqrt(var[usable]) / mean[usable]
    cv_median = float(np.median(cv))
    if cv_median <= 1e-6:
        warnings.append(
            "This image shows almost no local variation, which is not characteristic of SAR "
            "speckle; it may already have been filtered or resampled."
        )
        return _MAX_ENL, warnings

    enl = 1.0 / (cv_median * cv_median)
    if enl < _MIN_ENL:
        warnings.append(
            f"The measured speckle level ({enl:.1f} equivalent looks) is below the "
            f"single-look limit, which suggests texture rather than speckle; filtering was "
            f"limited to the single-look assumption."
        )
    elif enl > _MAX_ENL:
        warnings.append(
            "This image appears to have already been strongly speckle-filtered, so little "
            "further filtering was applied."
        )
    return float(np.clip(enl, _MIN_ENL, _MAX_ENL)), warnings


def lee_filter(
    linear: np.ndarray, enl: float, window: int = SPECKLE_WINDOW
) -> np.ndarray:
    """Lee (1980) minimum-mean-square-error speckle filter, in linear power.

    The estimator is ``x̂ = mean_y + b·(y − mean_y)`` with ``b = var_x/var_y``, where the
    signal variance is recovered from the observed variance by removing the speckle
    contribution: ``var_x = max(0, var_y − Cu²·mean_y²)/(1 + Cu²)`` and ``Cu² = 1/ENL``. Where
    the neighbourhood is homogeneous ``b → 0`` and the window mean is returned; at an edge
    ``b → 1`` and the pixel is left alone. That adaptivity is why it is used here instead of a
    box filter: measured on the demo scene, Lee reaches a boundary-to-interior gradient ratio
    of 9.14, while a box mean tops out at 6.57 at *any* window size from 3 to 13 — about 1.4×
    less edge contrast, and that comparison is not tilted in Lee's favour, because the best box
    mean also smooths class interiors slightly harder than Lee does (0.240 dB residual scatter
    against 0.283 dB). Lee also keeps the class means honest as it smooths, holding every class
    within 0.031 dB of its nominal backscatter, where a box mean drifts to 0.145 dB at window 11
    and 0.315 dB at window 13.

    Args:
        linear: Backscatter in linear power. NaN is preserved.
        enl: Equivalent number of looks, from :func:`estimate_enl`.
    """
    if window < 3 or window % 2 == 0:
        # A programming error, not a user input problem: the window is never user-supplied.
        raise ValueError(f"speckle window must be odd and >= 3, got {window}")
    valid = np.isfinite(linear) & (linear > 0)
    if not valid.any():
        return np.asarray(linear, dtype=np.float64).copy()

    ok = valid.astype(np.float64)
    filled = np.where(valid, linear, 0.0).astype(np.float64)
    count = _box_mean(ok, window)
    mean = np.divide(_box_mean(filled, window), count,
                     out=np.zeros_like(count), where=count > 1e-9)
    mean_sq = np.divide(_box_mean(filled ** 2, window), count,
                        out=np.zeros_like(count), where=count > 1e-9)
    var_y = np.maximum(mean_sq - mean * mean, 0.0)

    cu2 = 1.0 / max(enl, _MIN_ENL)
    var_x = np.maximum(var_y - cu2 * mean * mean, 0.0) / (1.0 + cu2)
    b = np.divide(var_x, var_y, out=np.zeros_like(var_y), where=var_y > 1e-20)

    out = mean + b * (np.asarray(linear, dtype=np.float64) - mean)
    # Negative power is not physical; it can only arise from the linear extrapolation at a
    # strong edge, and clamping to the local mean is the closest valid estimate.
    out = np.where(out > 0, out, mean)
    return np.where(valid, out, np.nan)


def despeckle(
    db: np.ndarray, window: int | None = None
) -> tuple[np.ndarray, SpeckleReport]:
    """Speckle-filter a dB backscatter raster, estimating the look count from the data.

    Args:
        window: Filter window; defaults to :data:`SPECKLE_WINDOW`. Resolved at call time
            rather than bound as a default argument so the module constant stays the single
            source of truth and remains overridable.

    Returns:
        The filtered dB raster and a report of what was done. The input/output CV figures in
        the report are measured in linear power, where they are the meaningful quantity.
    """
    win = SPECKLE_WINDOW if window is None else window
    linear = to_linear_power(db)
    enl, warnings = estimate_enl(linear, win)
    before_cv = _median_local_cv(linear, win)
    filtered = lee_filter(linear, enl, win)
    after_cv = _median_local_cv(filtered, win)
    report = SpeckleReport(
        filter_name="lee-mmse", window=win, estimated_enl=enl,
        input_cv=before_cv, output_cv=after_cv, warnings=warnings,
    )
    logger.debug("despeckle", extra=report.to_dict())
    return to_db(filtered), report


def _median_local_cv(linear: np.ndarray, window: int) -> float:
    """Median coefficient of variation over local windows, for reporting."""
    valid = np.isfinite(linear) & (linear > 0)
    if not valid.any():
        return 0.0
    ok = valid.astype(np.float64)
    filled = np.where(valid, linear, 0.0).astype(np.float64)
    count = _box_mean(ok, window)
    mean = np.divide(_box_mean(filled, window), count,
                     out=np.zeros_like(count), where=count > 1e-9)
    mean_sq = np.divide(_box_mean(filled ** 2, window), count,
                        out=np.zeros_like(count), where=count > 1e-9)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    usable = (count > 0.999) & (mean > 1e-12) & valid
    if not usable.any():  # pragma: no cover - defensive
        return 0.0
    return float(np.median(np.sqrt(var[usable]) / mean[usable]))


# ---------------------------------------------------------------------------
# Regime segmentation
# ---------------------------------------------------------------------------
def _regime_thresholds(
    values: np.ndarray,
) -> tuple[float | None, float | None, str, float, float, list[str]]:
    """Find the smooth and bright dB boundaries, refusing either when unsupported.

    Two cuts are needed, and they are found in two stages rather than by taking both from one
    three-class Otsu split. The low cut comes from that split directly. The high cut does not:
    the split's upper boundary lands where the *bulk* divides, which on the demo scene is
    -11.7 dB with a prominence of only 0.27 — it is a seed marking where the bright tail
    begins, not a decision boundary. Running Otsu again inside that tail finds the real
    corner-reflector boundary at -6.95 dB with a prominence of 0.998.

    Both cuts land essentially on the optimum without ever consulting ground truth. Sweeping the
    threshold against the demo scene's known class map, the best achievable smooth cut scores
    F1 0.9951 (at -16.55 dB) and the one found here scores 0.9938 (at -16.94 dB); the best
    achievable bright cut scores 0.9996 (at -7.80 dB) and this one scores 0.9989 (at -6.95 dB).
    Both are within 0.0013 F1 of an oracle, which is the strongest evidence available that the
    method is a property of backscatter distributions rather than a fit to this scene.

    The residual error is a mixed-pixel effect rather than a mis-set cut, which is what makes
    those scores interpretable. Every one of the 153 water disagreements lies within 3 pixels of
    the shoreline and 92% within 1 pixel, leaving the reservoir interior exact; the built-up mask
    misses 15 pixels and adds 2 against 7826, and it keeps the 50 m bare-soil road grid that cuts
    the urban blocks out of the building class entirely rather than smoothing across it.

    Each boundary is then gated independently on :func:`indices.mode_prominence`, so a scene
    with water but no buildings reports a smooth threshold and no bright one, rather than
    inventing a bright class from the top of a single population. The gate is deliberately
    conservative. On synthetic scenes with a dark patch of shrinking size it accepts from about
    5% coverage down (prominence 0.72 at 5%, F1 0.965) and refuses at 2% and below, which is
    where detection genuinely collapses: forcing a threshold anyway scores F1 0.19 at 1%
    coverage, 0.06 at 0.5% and 0.02 at 0.2%. At 2% it declines a case a forced threshold would
    have recovered at F1 0.88 — the error is in the direction of a missed regime rather than a
    phantom one, which is the trade this system is meant to make.

    Returns:
        ``(smooth_db, bright_db, method, smooth_prominence, bright_prominence, warnings)``,
        with ``None`` for a boundary that the histogram does not support.
    """
    warnings: list[str] = []
    (lo, hi), quality = indices.multilevel_otsu_threshold(values)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        warnings.append(
            "The backscatter distribution of this image is too uniform to separate scattering "
            "regimes, so reference thresholds were used instead of thresholds measured here."
        )
        return (LITERATURE_SMOOTH_DB, LITERATURE_BRIGHT_DB, "literature", 0.0, 0.0, warnings)

    smooth_prom = indices.mode_prominence(values, lo, upper=hi)
    smooth_db: float | None = float(lo)
    if smooth_prom < indices.MIN_MODE_PROMINENCE:
        smooth_db = None
        warnings.append(
            "No distinct dark-surface population was found in this image, so no smooth or "
            "specular area (such as open water) is reported. A weak or very small dark area "
            "cannot be distinguished from the low tail of the surrounding terrain."
        )

    tail = values[values >= hi]
    bright_db: float | None = None
    bright_prom = 0.0
    if tail.size >= 64:
        cut, _ = indices.otsu_threshold(tail)
        if np.isfinite(cut):
            bright_prom = indices.mode_prominence(values, float(cut), lower=lo)
            if bright_prom >= indices.MIN_MODE_PROMINENCE:
                bright_db = float(cut)
    if bright_db is None:
        warnings.append(
            "No distinct bright-scattering population was found, so no double-bounce area "
            "(such as buildings) is reported for this image."
        )

    method = "otsu-3class" if (smooth_db is not None or bright_db is not None) else "refused"
    logger.debug(
        "sar regime thresholds",
        extra={"otsu_lo": round(float(lo), 3), "otsu_hi": round(float(hi), 3),
               "otsu_quality": round(quality, 4), "smooth_db": smooth_db,
               "bright_db": bright_db, "smooth_prominence": round(smooth_prom, 4),
               "bright_prominence": round(bright_prom, 4)},
    )
    return smooth_db, bright_db, method, smooth_prom, bright_prom, warnings


def analyze_sar(
    raster: RasterData, *, filter_speckle: bool = True, window: int | None = None
) -> SarResult:
    """Segment a SAR raster into scattering regimes.

    Args:
        raster: A single-polarisation SAR raster. When several bands are present the first
            recognised polarisation band is used and the choice is reported.
        filter_speckle: Whether to run the Lee filter first. Measured effect on the demo
            scene: smooth-regime F1 rises 0.948 → 0.994, because a dark surface and the low
            tail of surrounding terrain are only a few dB apart and speckle straddles that gap.
            The double-bounce regime barely moves (0.9935 → 0.9989) since corner-reflector
            returns sit about 10 dB clear of everything else, far outside speckle's range.
            Disabling it is useful only for testing the segmentation in isolation.
        window: Speckle filter window; defaults to :data:`SPECKLE_WINDOW`.

    Raises:
        ValidationError: if the raster carries no usable backscatter values.
    """
    warnings: list[str] = []
    band, polarization, band_warning = _select_polarization(raster)
    if band_warning:
        warnings.append(band_warning)

    if raster.modality is Modality.OPTICAL:
        warnings.append(
            "This image was identified as optical, not radar. Backscatter analysis assumes "
            "radar data, so these results should be treated as unreliable."
        )

    finite = band[np.isfinite(band)]
    if finite.size < 64:
        raise ValidationError(
            "This image does not contain enough valid data to analyse radar backscatter.",
            code=ErrorCode.NO_VALID_PIXELS,
            context={"valid_pixels": int(finite.size)},
        )

    # --- units ---
    if looks_like_db(band):
        db = np.asarray(band, dtype=np.float64)
    else:
        db = to_db(band)
        warnings.append(
            "The image values were not in decibels, so they were converted before analysis. "
            "If they are uncalibrated digital numbers rather than backscatter, the decibel "
            "thresholds reported here are not physically meaningful."
        )
    if float(np.nanmax(db)) > _DB_PLAUSIBLE_MAX:
        warnings.append(
            f"Backscatter values reach {float(np.nanmax(db)):.1f} dB, which is far above the "
            f"range expected from calibrated radar data; this image may not be calibrated."
        )

    # --- speckle ---
    win = SPECKLE_WINDOW if window is None else window
    if filter_speckle:
        db, speckle = despeckle(db, win)
        warnings.extend(speckle.warnings)
    else:
        linear = to_linear_power(db)
        cv = _median_local_cv(linear, win)
        speckle = SpeckleReport("none", win, float("nan"), cv, cv)

    valid = np.isfinite(db) & raster.valid_mask()
    values = db[valid]
    if values.size < 64:  # pragma: no cover - defensive, guarded above
        raise ValidationError(
            "This image does not contain enough valid data to analyse radar backscatter.",
            code=ErrorCode.NO_VALID_PIXELS,
        )

    smooth_db, bright_db, method, smooth_prom, bright_prom, thr_warnings = _regime_thresholds(
        values
    )
    warnings.extend(thr_warnings)

    regime_map = np.zeros(db.shape, dtype=np.uint8)
    regime_map[valid] = _REGIME_CODE[ScatteringRegime.DIFFUSE]
    if smooth_db is not None:
        regime_map[valid & (db < smooth_db)] = _REGIME_CODE[ScatteringRegime.SMOOTH]
    if bright_db is not None:
        regime_map[valid & (db > bright_db)] = _REGIME_CODE[ScatteringRegime.DOUBLE_BOUNCE]

    # The diffuse middle is left whole on purpose — see the module docstring.
    warnings.append(
        "Radar backscatter measures surface roughness and geometry, not land cover. Moderate "
        "backscatter is reported as a single rough-surface class because vegetation and bare "
        "soil cannot be separated from single-polarisation data alone."
    )

    stats, area_caveat = _regime_stats(regime_map, db, valid, raster)
    if area_caveat:
        warnings.append(area_caveat)

    return SarResult(
        regime_map=regime_map, backscatter_db=db, stats=stats,
        smooth_threshold_db=smooth_db, bright_threshold_db=bright_db,
        threshold_method=method, smooth_prominence=smooth_prom, bright_prominence=bright_prom,
        speckle=speckle, polarization=polarization, warnings=warnings,
        histogram=indices.histogram_summary(values),
    )


def _select_polarization(raster: RasterData) -> tuple[np.ndarray, str, str | None]:
    """Pick the band to analyse, preferring an explicitly labelled polarisation.

    Returns:
        The 2-D band, its polarisation label, and a warning when the choice was not explicit.
    """
    roles = list(raster.band_roles)
    for role in (BandRole.VV, BandRole.VH):
        if role in roles:
            band = raster.data[roles.index(role)]
            extra = None
            if raster.data.shape[0] > 1:
                extra = (
                    f"This image has {raster.data.shape[0]} bands; the {role.value.upper()} "
                    f"polarisation was analysed."
                )
            return band, role.value.upper(), extra
    if raster.data.shape[0] == 1:
        return raster.data[0], "unknown", (
            "The polarisation of this image is not recorded, so it is reported as unknown. "
            "Backscatter thresholds differ between polarisations."
        )
    return raster.data[0], "unknown", (
        f"This image has {raster.data.shape[0]} bands but none is labelled as a radar "
        f"polarisation; the first band was analysed and its polarisation is unknown."
    )


def _regime_stats(
    regime_map: np.ndarray, db: np.ndarray, valid: np.ndarray, raster: RasterData
) -> tuple[list[RegimeStats], str | None]:
    """Per-regime coverage and mean backscatter."""
    total = int(valid.sum())
    px_area, caveat = pixel_area_m2(raster.metadata)
    out: list[RegimeStats] = []
    for regime, code in _REGIME_CODE.items():
        if regime is ScatteringRegime.UNCLASSIFIED:
            continue
        mask = regime_map == code
        count = int(mask.sum())
        if count == 0:
            continue
        out.append(RegimeStats(
            regime=regime,
            pixel_count=count,
            fraction=count / total if total else 0.0,
            area_m2=count * px_area if px_area is not None else None,
            mean_db=float(np.nanmean(db[mask])),
        ))
    return sorted(out, key=lambda s: s.fraction, reverse=True), caveat
