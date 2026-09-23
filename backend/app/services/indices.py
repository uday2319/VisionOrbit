"""Spectral indices and threshold selection.

Every index here is a published normalised-difference ratio. Two properties make them the
right backbone for this system:

* **Scale invariance.** A ratio of two bands is insensitive to a constant gain, so indices
  work on raw DN, TOA reflectance or surface reflectance alike.
* **Availability is checkable.** An index is computed only when its required bands were
  actually *resolved* (not guessed), so the system degrades explicitly instead of silently
  producing a meaningless number.

:func:`otsu_threshold` and :func:`separability` supply the evidence-strength signal the
confidence engine needs: a genuinely bimodal histogram means the threshold sits in a real
valley, and a unimodal one means the "detection" is a cut through noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.logging import get_logger
from ..core.types import BandRole
from ..geospatial.raster import RasterData, band_mean

logger = get_logger(__name__)

# Conventional decision thresholds from the remote-sensing literature. They are starting
# points: `adaptive_threshold` prefers a data-driven Otsu cut when the histogram supports
# one, and falls back to these when it does not.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "ndvi_vegetation": 0.30,
    "ndwi_water": 0.00,
    "mndwi_water": 0.00,
    "ndbi_builtup": 0.00,
    "bsi_bare": 0.00,
}


@dataclass
class IndexResult:
    """A computed index plus provenance for how it was derived."""

    name: str
    array: np.ndarray
    formula: str
    bands_used: tuple[str, ...]
    valid_fraction: float

    def to_dict(self) -> dict[str, Any]:
        finite = self.array[np.isfinite(self.array)]
        return {
            "name": self.name,
            "formula": self.formula,
            "bands_used": list(self.bands_used),
            "valid_fraction": round(self.valid_fraction, 4),
            "mean": round(float(finite.mean()), 4) if finite.size else None,
            "min": round(float(finite.min()), 4) if finite.size else None,
            "max": round(float(finite.max()), 4) if finite.size else None,
        }


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute ``(a - b) / (a + b)`` with NaN where the denominator vanishes."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / denom
    out[~np.isfinite(out)] = np.nan
    # A near-zero denominator means both bands are ~0 (deep shadow / nodata); the ratio
    # is numerically explosive and physically meaningless there.
    out[np.abs(denom) < 1e-6] = np.nan
    return np.clip(out, -1.0, 1.0)


def _valid_fraction(arr: np.ndarray) -> float:
    return float(np.count_nonzero(np.isfinite(arr)) / arr.size) if arr.size else 0.0


def available_indices(raster: RasterData) -> set[str]:
    """Names of indices computable from this raster's *resolved* bands."""
    roles = set(raster.band_roles)
    out: set[str] = set()
    if {BandRole.NIR, BandRole.RED} <= roles:
        out.add("ndvi")
    if {BandRole.GREEN, BandRole.NIR} <= roles:
        out.add("ndwi")
    if {BandRole.GREEN, BandRole.SWIR1} <= roles:
        out.add("mndwi")
    if {BandRole.SWIR1, BandRole.NIR} <= roles:
        out.add("ndbi")
    if {BandRole.SWIR1, BandRole.RED, BandRole.NIR, BandRole.BLUE} <= roles:
        out.add("bsi")
    if {BandRole.RED, BandRole.GREEN, BandRole.BLUE} <= roles:
        out.add("exg")
        out.add("brightness")
    return out


def compute_index(raster: RasterData, name: str) -> IndexResult:
    """Compute a named index.

    Raises:
        ValidationError: propagated from :meth:`RasterData.band` when a required band is
            absent — deliberately loud, because a wrong band silently poisons the result.
    """
    key = name.lower()

    if key == "ndvi":
        nir, red = raster.band(BandRole.NIR), raster.band(BandRole.RED)
        arr = normalized_difference(nir, red)
        return IndexResult("ndvi", arr, "(NIR - Red) / (NIR + Red)", ("nir", "red"),
                           _valid_fraction(arr))

    if key == "ndwi":
        green, nir = raster.band(BandRole.GREEN), raster.band(BandRole.NIR)
        arr = normalized_difference(green, nir)
        return IndexResult("ndwi", arr, "(Green - NIR) / (Green + NIR)", ("green", "nir"),
                           _valid_fraction(arr))

    if key == "mndwi":
        green, swir1 = raster.band(BandRole.GREEN), raster.band(BandRole.SWIR1)
        arr = normalized_difference(green, swir1)
        return IndexResult("mndwi", arr, "(Green - SWIR1) / (Green + SWIR1)",
                           ("green", "swir1"), _valid_fraction(arr))

    if key == "ndbi":
        swir1, nir = raster.band(BandRole.SWIR1), raster.band(BandRole.NIR)
        arr = normalized_difference(swir1, nir)
        return IndexResult("ndbi", arr, "(SWIR1 - NIR) / (SWIR1 + NIR)", ("swir1", "nir"),
                           _valid_fraction(arr))

    if key == "bsi":
        # Bare Soil Index (Rikimaru et al.).
        swir1 = raster.band(BandRole.SWIR1).astype(np.float32)
        red = raster.band(BandRole.RED).astype(np.float32)
        nir = raster.band(BandRole.NIR).astype(np.float32)
        blue = raster.band(BandRole.BLUE).astype(np.float32)
        num = (swir1 + red) - (nir + blue)
        den = (swir1 + red) + (nir + blue)
        with np.errstate(divide="ignore", invalid="ignore"):
            arr = num / den
        arr[~np.isfinite(arr)] = np.nan
        arr = np.clip(arr, -1.0, 1.0)
        return IndexResult("bsi", arr, "((SWIR1+Red)-(NIR+Blue)) / ((SWIR1+Red)+(NIR+Blue))",
                           ("swir1", "red", "nir", "blue"), _valid_fraction(arr))

    if key == "exg":
        # Excess Green — an RGB-only greenness proxy. Weaker than NDVI; used only as a
        # documented fallback when no NIR band exists.
        red = raster.band(BandRole.RED).astype(np.float32)
        green = raster.band(BandRole.GREEN).astype(np.float32)
        blue = raster.band(BandRole.BLUE).astype(np.float32)
        total = red + green + blue
        with np.errstate(divide="ignore", invalid="ignore"):
            r, g, b = red / total, green / total, blue / total
        arr = 2 * g - r - b
        arr[~np.isfinite(arr)] = np.nan
        arr = np.clip(arr, -1.0, 1.0)
        return IndexResult("exg", arr, "2g - r - b  (normalised RGB chromaticity)",
                           ("red", "green", "blue"), _valid_fraction(arr))

    if key == "brightness":
        rgb = np.stack([
            raster.band(BandRole.RED), raster.band(BandRole.GREEN), raster.band(BandRole.BLUE)
        ]).astype(np.float32)
        arr = band_mean(rgb)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            lo, hi = np.percentile(finite, [1.0, 99.0])
            if hi - lo > 1e-9:
                arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return IndexResult("brightness", arr, "mean(Red, Green, Blue), percentile-stretched",
                           ("red", "green", "blue"), _valid_fraction(arr))

    raise ValueError(f"unknown index: {name}")


def compute_all_available(raster: RasterData) -> dict[str, IndexResult]:
    """Compute every index this raster supports, skipping any that error."""
    out: dict[str, IndexResult] = {}
    for name in sorted(available_indices(raster)):
        try:
            out[name] = compute_index(raster, name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("index failed", extra={"index": name, "error": str(exc)})
    return out


# ---------------------------------------------------------------------------
# Thresholding & evidence strength
# ---------------------------------------------------------------------------
def otsu_threshold(values: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Otsu's threshold and the achieved between-class variance ratio.

    Returns:
        ``(threshold, quality)`` where ``quality`` in ``[0, 1]`` is the fraction of total
        variance explained by the split. High quality means a genuinely bimodal histogram —
        i.e. two real populations rather than one cut arbitrarily in half.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 16:
        return float("nan"), 0.0

    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        return float("nan"), 0.0

    hist, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float("nan"), 0.0
    prob = hist / total
    centers = (edges[:-1] + edges[1:]) / 2.0

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = 0.0

    idx = int(np.argmax(sigma_b))
    # When a histogram has a wide empty valley, every cut inside it yields the *same*
    # between-class variance, and `argmax` returns the leftmost — which sits at the edge of
    # the valley rather than in it, and can fall inside the lower population's tail. Taking
    # the midpoint of the tied plateau puts the threshold where a human would draw it and
    # makes the result independent of tie-breaking order. Only the contiguous run containing
    # the maximum counts, so unrelated equal-scoring bins elsewhere cannot drag it.
    tol = float(sigma_b[idx]) * 1e-9
    lo_i = hi_i = idx
    while lo_i > 0 and sigma_b[lo_i - 1] >= sigma_b[idx] - tol:
        lo_i -= 1
    while hi_i < sigma_b.size - 1 and sigma_b[hi_i + 1] >= sigma_b[idx] - tol:
        hi_i += 1
    idx = (lo_i + hi_i) // 2
    threshold = float(centers[idx])
    total_var = float(np.sum(prob * (centers - mu_t) ** 2))
    quality = float(np.clip(sigma_b[idx] / total_var, 0.0, 1.0)) if total_var > 1e-12 else 0.0
    return threshold, quality


def separability(values: np.ndarray, mask: np.ndarray) -> float:
    """Normalised separability between masked and unmasked populations.

    This is the standardised mean difference (Cohen's d) squashed into ``[0, 1]``. It answers
    "how distinct is what we detected from what we did not?" — the single most useful
    evidence-strength signal available without ground truth.
    """
    inside = values[mask & np.isfinite(values)]
    outside = values[(~mask) & np.isfinite(values)]
    if inside.size < 8 or outside.size < 8:
        return 0.0
    pooled = np.sqrt(0.5 * (inside.var() + outside.var()))
    if pooled < 1e-9:
        return 1.0 if abs(inside.mean() - outside.mean()) > 1e-9 else 0.0
    d = abs(float(inside.mean()) - float(outside.mean())) / float(pooled)
    # d = 2.0 is a large effect; treat that as full separability.
    return float(np.clip(d / 2.0, 0.0, 1.0))


def bimodality_coefficient(values: np.ndarray) -> float:
    """Sarle's bimodality coefficient ``(skew² + 1) / kurtosis``.

    Otsu's between-class variance ratio cannot answer "are there two populations here?".
    Splitting a *unimodal* Gaussian at its mean already explains ``2/π ≈ 0.637`` of the
    variance, and a uniform distribution scores about 0.75 — higher than a genuine but
    overlapping bimodal mixture. Using that ratio alone to justify a data-driven threshold
    therefore certifies bimodality that may not exist.

    This statistic is shape-based instead. The reference points are exact: a normal
    distribution gives ``1/3``, a uniform distribution gives ``5/9 ≈ 0.5556``, and a
    perfectly split two-point distribution gives ``1.0``. The conventional decision rule —
    and :data:`BIMODALITY_THRESHOLD` — is the uniform value, because a histogram flatter
    than uniform is the point at which "one peak" stops being a fair description.

    Returns:
        The coefficient in ``[0, 1]``, or ``0.0`` when there is too little data to judge.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 32:
        return 0.0
    n = float(finite.size)
    std = float(finite.std())
    if std < 1e-12:
        return 0.0
    centered = (finite - finite.mean()) / std
    skew = float(np.mean(centered**3))
    # Sarle's coefficient uses *excess* kurtosis (E[z⁴] - 3), not raw kurtosis.
    excess_kurt = float(np.mean(centered**4)) - 3.0
    # Small-sample correction (SAS formulation).
    correction = 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3)) if n > 3 else 3.0
    denom = excess_kurt + correction
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip((skew * skew + 1.0) / denom, 0.0, 1.0))


# Sarle's coefficient equals 5/9 for a uniform distribution and 1/3 for a normal one.
# Requiring at least the uniform value means a single peak, however heavy-tailed, is never
# read as two populations.
BIMODALITY_THRESHOLD = 5.0 / 9.0


def multilevel_otsu_threshold(
    values: np.ndarray, bins: int = 256
) -> tuple[tuple[float, float], float]:
    """Two thresholds splitting ``values`` into three classes, by Otsu's criterion.

    The three-class extension of :func:`otsu_threshold`, searching every ordered pair of cut
    points for the one maximising between-class variance. Needed where a distribution has
    three physically distinct regimes rather than two — SAR backscatter being the motivating
    case, with specular-dark, diffuse-moderate and double-bounce-bright populations.

    Returns:
        ``((low, high), quality)``. **The quality figure must not be used as evidence that
        three populations exist** — see :func:`mode_prominence`. Measured on this repo's demo
        SAR scene it reads 0.748 for a genuine four-class image but 0.809 for pure Gaussian
        noise, because partitioning *any* distribution by value explains most of its variance.
        It is returned only to describe the split that was made.
        ``((nan, nan), 0.0)`` when there is too little data to judge.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 32:
        return (float("nan"), float("nan")), 0.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        return (float("nan"), float("nan")), 0.0

    hist, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    total = hist.sum()
    if total <= 0:  # pragma: no cover - defensive
        return (float("nan"), float("nan")), 0.0
    prob = hist.astype(np.float64) / total
    centers = (edges[:-1] + edges[1:]) / 2.0

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = float(mu[-1])

    # All (i, j) pairs at once: class 0 is [0, i], class 1 is (i, j], class 2 is (j, end].
    w0 = omega[:, None]
    w1 = omega[None, :] - omega[:, None]
    w2 = 1.0 - omega[None, :]
    m0_num = mu[:, None]
    m1_num = mu[None, :] - mu[:, None]
    m2_num = mu_t - mu[None, :]

    i_idx = np.arange(bins)[:, None]
    j_idx = np.arange(bins)[None, :]
    valid = (j_idx > i_idx) & (w0 > 1e-12) & (w1 > 1e-12) & (w2 > 1e-12)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Between-class variance written as Σ wₖ(µₖ − µₜ)², expanded to avoid dividing
        # by the class weights twice.
        sigma_b = (
            (m0_num - w0 * mu_t) ** 2 / w0
            + (m1_num - w1 * mu_t) ** 2 / w1
            + (m2_num - w2 * mu_t) ** 2 / w2
        )
    sigma_b = np.where(valid & np.isfinite(sigma_b), sigma_b, -np.inf)
    if not np.isfinite(sigma_b).any():  # pragma: no cover - defensive
        return (float("nan"), float("nan")), 0.0

    flat = int(np.argmax(sigma_b))
    i, j = divmod(flat, bins)
    total_var = float(np.sum(prob * (centers - mu_t) ** 2))
    quality = (
        float(np.clip(sigma_b[i, j] / total_var, 0.0, 1.0)) if total_var > 1e-12 else 0.0
    )
    return (float(centers[i]), float(centers[j])), quality


def mode_prominence(
    values: np.ndarray,
    cut: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    bins: int = 128,
    smooth: int = 5,
) -> float:
    """Topographic prominence of the smaller peak flanking ``cut``, in ``[0, 1]``.

    Answers the question a between-class variance ratio cannot: *is there really a valley
    here, or is a single population being cut in half?* The histogram is smoothed, the tallest
    peak on each side of ``cut`` is located, the lowest point between those peaks is taken as
    the saddle, and the result is how far that saddle falls below the shorter peak. It
    describes the histogram's shape around the cut and never consults the partition the cut
    induces, so — unlike a variance ratio — it cannot be satisfied by construction.

    Measured on this repo's demo data, the separation is wide and unambiguous: a genuine
    four-class SAR scene scores 0.72 at its water boundary and 0.91 at its built-up boundary,
    while a single homogeneous class scores 0.00–0.06, Gaussian noise 0.00, and a uniform
    distribution 0.03–0.05. Anything in the empty band between those groups works as a
    decision threshold, which is why :data:`MIN_MODE_PROMINENCE` is not a fitted quantity.

    Args:
        lower: Restrict to values at or above this, so the measurement concerns only the two
            modes ``cut`` actually separates rather than the whole scene.
        upper: Restrict to values at or below this.

    Returns:
        ``0.0`` when there is no interior peak pair to measure — the conservative answer,
        because absence of a measurable valley is not evidence of one.
    """
    subset = values[np.isfinite(values)]
    if lower is not None:
        subset = subset[subset >= lower]
    if upper is not None:
        subset = subset[subset <= upper]
    if subset.size < 100:
        return 0.0
    span = float(subset.max()) - float(subset.min())
    if span < 1e-9 or not np.isfinite(cut):
        return 0.0

    hist, edges = np.histogram(subset, bins=bins)
    kernel = np.ones(max(1, smooth), dtype=np.float64) / max(1, smooth)
    density = np.convolve(hist.astype(np.float64), kernel, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2.0

    idx = int(np.searchsorted(centers, cut))
    if idx < 1 or idx >= bins:
        return 0.0
    left = int(np.argmax(density[:idx]))
    right = idx + int(np.argmax(density[idx:]))
    if right <= left:  # pragma: no cover - defensive
        return 0.0

    saddle = float(density[left : right + 1].min())
    peak = min(float(density[left]), float(density[right]))
    if peak <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - saddle / peak, 0.0, 1.0))


# The saddle between two modes must fall at least this far below the shorter of them before
# the cut is treated as separating real populations. Chosen from the middle of a measured
# empty band, not fitted: see :func:`mode_prominence` for the numbers on either side. On
# synthetic mixtures the refusal boundary coincides with where the method genuinely stops
# working — a dark population at 1% of the frame scores 0.00 prominence and would have been
# recovered at F1 0.11, while at 2% it scores 0.59 and is recovered at F1 0.94.
MIN_MODE_PROMINENCE = 0.25


def adaptive_threshold(
    index: np.ndarray,
    fallback: float,
    *,
    min_otsu_quality: float = 0.55,
    min_bimodality: float = BIMODALITY_THRESHOLD,
) -> tuple[float, str, float]:
    """Choose a threshold, preferring a data-driven cut when the histogram justifies it.

    Two independent conditions must hold before Otsu is trusted: the split must explain
    enough variance *and* the distribution must actually look like two populations
    (:func:`bimodality_coefficient`). Either alone is insufficient — see that function for
    why the variance ratio can be high for a single peak.

    Returns:
        ``(threshold, method, quality)`` where ``method`` is ``"otsu"`` or ``"literature"``.
        Recording which was used matters: a literature threshold applied to a unimodal
        histogram is a much weaker claim, and the confidence score reflects that.
    """
    thr, quality = otsu_threshold(index)
    # Ordered so the cheap variance check gates the moment-based shape test.
    if (
        np.isfinite(thr)
        and quality >= min_otsu_quality
        and bimodality_coefficient(index) >= min_bimodality
    ):
        return float(thr), "otsu", float(quality)
    return float(fallback), "literature", float(quality)


def histogram_summary(values: np.ndarray, bins: int = 32) -> dict[str, Any]:
    """Compact histogram payload for the UI/report."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"bins": [], "counts": [], "mean": None, "std": None}
    counts, edges = np.histogram(finite, bins=bins)
    return {
        "bins": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
        "mean": round(float(finite.mean()), 4),
        "std": round(float(finite.std()), 4),
        "p05": round(float(np.percentile(finite, 5)), 4),
        "p95": round(float(np.percentile(finite, 95)), 4),
    }
