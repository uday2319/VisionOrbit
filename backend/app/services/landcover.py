"""Per-pixel land-cover classification from spectral indices and texture.

The method is a transparent decision hierarchy, not a black box, so every class assignment is
explainable and every requirement it has on the input is checkable:

1. water     — NDWI/MNDWI positive
2. vegetation— NDVI high
3. built-up  — SWIR-bright (NDBI) *and* edge-dense; both are required because bare soil is
   also SWIR-bright, so the spectral cue alone cannot separate the two
4. bare soil — the remainder

When NIR/SWIR bands are missing, the analyser does **not** fabricate the indices it cannot
compute. It drops to an RGB colour+texture approximation and attaches a warning that lowers
downstream confidence — the honest-degradation path the brief demands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.logging import get_logger
from ..core.types import BandRole, LandCoverClass
from ..geospatial.measure import pixel_area_m2, pixels_for_ground_area
from ..geospatial.raster import RasterData
from . import features, indices

logger = get_logger(__name__)

_CLASS_CODE = {c: c.code for c in LandCoverClass}
CODE_TO_NAME = {v: k.value for k, v in _CLASS_CODE.items()}

# Edge-density reference threshold, used only when a scene's texture histogram is not
# bimodal enough for Otsu. `features.edge_density` returns a percentile-stretched [0, 1]
# fraction, so this says "more than a fifth of the local neighbourhood is edge pixels" —
# a structural-surface criterion, not a brightness one.
_EDGE_DENSITY_FALLBACK = 0.20

# Smallest ground area that is credible as a distinct built-up patch. Below roughly a
# quarter of a hectare a cluster of SWIR-bright, edge-dense pixels is far more likely to be
# threshold noise than a settlement, so it is removed rather than reported as detected area.
MIN_BUILTUP_PATCH_M2 = 2500.0

# Pixel-count floor for scenes with no usable pixel size, so the same denoising still
# applies when an image is not georeferenced.
_MIN_BUILTUP_PATCH_PX = 25


def _min_patch_pixels(raster: RasterData) -> int:
    """Convert :data:`MIN_BUILTUP_PATCH_M2` into pixels for this raster's resolution.

    Delegates the conversion so that a degree-based (geographic) CRS is handled correctly:
    reading ``pixel_size`` directly there yields a pixel area of ~8·10⁻⁹ "m²", turning this
    floor into 3·10¹¹ pixels and deleting every built-up patch without a word.
    """
    return pixels_for_ground_area(
        raster.metadata, MIN_BUILTUP_PATCH_M2, default_pixels=_MIN_BUILTUP_PATCH_PX
    )


def _region_threshold(
    index: np.ndarray, region_values: np.ndarray, fallback: float
) -> tuple[float, str, float]:
    """Pick a threshold for ``index`` using only the pixels inside ``region_values``.

    Two departures from :func:`indices.adaptive_threshold` matter here:

    * The cut is derived from the *restricted* population. Water and vegetation have already
      been removed, and leaving them in would drag Otsu's split away from the boundary that
      is actually in question.
    * When Otsu is not justified, the fallback is raised to the region median instead of
      being used as-is. Published thresholds are calibrated against whole scenes; inside an
      already water- and vegetation-excluded subset the same number is far too permissive,
      which is exactly how bare soil gets reported as built-up.
    """
    values = index[region_values]
    if values.size < 16:
        return float(fallback), "literature", 0.0
    thr, method, quality = indices.adaptive_threshold(values, fallback)
    if method == "literature":
        median = float(np.nanmedian(values))
        if np.isfinite(median):
            thr = max(thr, median)
    return float(thr), method, float(quality)


@dataclass
class ClassStats:
    """Coverage statistics for one land-cover class."""

    label: LandCoverClass
    pixel_count: int
    fraction: float
    area_m2: float | None
    mean_evidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.label.value,
            "pixel_count": self.pixel_count,
            "fraction": round(self.fraction, 4),
            "percentage": round(self.fraction * 100, 2),
            "area_m2": round(self.area_m2, 2) if self.area_m2 is not None else None,
            "mean_evidence": round(self.mean_evidence, 4),
        }


@dataclass
class LandCoverResult:
    """Full land-cover analysis output."""

    class_map: np.ndarray
    stats: list[ClassStats]
    method: str
    indices_used: list[str]
    separability: float
    warnings: list[str] = field(default_factory=list)
    index_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def dominant(self) -> ClassStats | None:
        real = [s for s in self.stats if s.label is not LandCoverClass.UNCLASSIFIED]
        return max(real, key=lambda s: s.fraction) if real else None

    def mask_for(self, label: LandCoverClass) -> np.ndarray:
        return self.class_map == _CLASS_CODE[label]

    def to_dict(self) -> dict[str, Any]:
        dominant = self.dominant()
        return {
            "method": self.method,
            "indices_used": list(self.indices_used),
            "separability": round(self.separability, 4),
            "classes": [s.to_dict() for s in self.stats],
            "dominant": dominant.to_dict() if dominant is not None else None,
            "warnings": list(self.warnings),
            "index_summaries": self.index_summaries,
        }


def classify_land_cover(raster: RasterData) -> LandCoverResult:
    """Classify a raster into land-cover classes.

    Chooses the spectral path when NIR is available, otherwise a clearly-labelled RGB
    approximation. The returned :attr:`LandCoverResult.method` records which was used.
    """
    roles = set(raster.band_roles)
    if BandRole.NIR in roles:
        return _classify_spectral(raster)
    return _classify_rgb(raster)


def _classify_spectral(raster: RasterData) -> LandCoverResult:
    """Full spectral classification (requires NIR)."""
    warnings: list[str] = []
    idx = indices.compute_all_available(raster)
    h, w = raster.shape
    valid = raster.valid_mask()
    class_map = np.zeros((h, w), dtype=np.uint8)
    used: list[str] = []
    evidence = np.zeros((h, w), dtype=np.float32)

    # --- water: prefer MNDWI (SWIR discriminates water from shadow) ---
    water_mask = np.zeros((h, w), dtype=bool)
    if "mndwi" in idx:
        arr = idx["mndwi"].array
        thr, method, _ = indices.adaptive_threshold(arr, indices.DEFAULT_THRESHOLDS["mndwi_water"])
        water_mask = np.isfinite(arr) & (arr > thr)
        used.append("mndwi")
        evidence = np.where(water_mask, np.clip(arr, 0, 1), evidence)
    elif "ndwi" in idx:
        arr = idx["ndwi"].array
        thr, method, _ = indices.adaptive_threshold(arr, indices.DEFAULT_THRESHOLDS["ndwi_water"])
        water_mask = np.isfinite(arr) & (arr > thr)
        used.append("ndwi")
        evidence = np.where(water_mask, np.clip(arr, 0, 1), evidence)

    # --- vegetation: NDVI ---
    veg_mask = np.zeros((h, w), dtype=bool)
    if "ndvi" in idx:
        arr = idx["ndvi"].array
        thr, method, _ = indices.adaptive_threshold(arr, indices.DEFAULT_THRESHOLDS["ndvi_vegetation"])
        veg_mask = np.isfinite(arr) & (arr > thr) & ~water_mask
        used.append("ndvi")
        evidence = np.where(veg_mask, np.clip(arr, 0, 1), evidence)

    # --- built-up: SWIR-bright AND edge-dense (this is what excludes bare soil) ---
    builtup_mask = np.zeros((h, w), dtype=bool)
    panchro = raster.band_mean()
    edges = features.edge_density(panchro, window=9)
    if "ndbi" in idx:
        arr = idx["ndbi"].array
        # Only the not-water, not-vegetation remainder can be built-up. Restricting first
        # matters: it is the reference population both thresholds are derived from.
        region = valid & ~water_mask & ~veg_mask & np.isfinite(arr)
        ndbi_thr, ndbi_method, ndbi_q = _region_threshold(
            arr, region, indices.DEFAULT_THRESHOLDS["ndbi_builtup"]
        )
        edge_thr, edge_method, edge_q = _region_threshold(
            edges, region, _EDGE_DENSITY_FALLBACK
        )
        # NDBI alone cannot do this: built-up and bare soil are *both* SWIR-positive, so a
        # spectral cut over-predicts built-up wherever soil dominates. Edge density is the
        # discriminating cue — roofs, roads and shadows break up the surface, ploughed
        # ground does not. Requiring both is what makes the class honest.
        builtup_mask = region & (arr > ndbi_thr) & (edges > edge_thr)
        builtup_mask = features.clean_mask(
            builtup_mask,
            open_radius=1,
            close_radius=2,
            min_area=_min_patch_pixels(raster),
        )
        used.append("ndbi")
        if ndbi_method == "literature" and edge_method == "literature":
            warnings.append(
                "Neither the built-up spectral index nor the edge-density texture showed a "
                "clearly bimodal distribution, so built-up extent was separated using "
                "reference thresholds rather than thresholds derived from this image. Treat "
                "the built-up area as indicative rather than measured."
            )
        logger.debug(
            "built-up thresholds",
            extra={"ndbi_thr": round(ndbi_thr, 4), "ndbi_method": ndbi_method,
                   "ndbi_quality": round(ndbi_q, 3), "edge_thr": round(edge_thr, 4),
                   "edge_method": edge_method, "edge_quality": round(edge_q, 3)},
        )
        ndbi01 = np.clip((arr + 1) / 2, 0, 1)
        evidence = np.where(builtup_mask, 0.5 * ndbi01 + 0.5 * edges, evidence)

    # --- bare soil: valid remainder ---
    assigned = water_mask | veg_mask | builtup_mask
    bare_mask = valid & ~assigned

    class_map[water_mask] = _CLASS_CODE[LandCoverClass.WATER]
    class_map[veg_mask] = _CLASS_CODE[LandCoverClass.VEGETATION]
    class_map[builtup_mask] = _CLASS_CODE[LandCoverClass.BUILT_UP]
    class_map[bare_mask] = _CLASS_CODE[LandCoverClass.BARE_SOIL]

    sep = _mean_class_separability(idx, water_mask, veg_mask, builtup_mask)
    stats, area_caveat = _class_stats(class_map, valid, evidence, raster)
    if area_caveat:
        warnings.append(area_caveat)
    summaries = {name: r.to_dict() for name, r in idx.items()}

    return LandCoverResult(
        class_map=class_map, stats=stats, method="spectral-indices",
        indices_used=used, separability=sep, warnings=warnings, index_summaries=summaries,
    )


def _classify_rgb(raster: RasterData) -> LandCoverResult:
    """RGB-only approximation used when no NIR band exists.

    Deliberately narrower than the spectral path. Two design points carry the honesty
    requirement, and both were chosen from measurements rather than assumed:

    * **Water is tested before vegetation.** In the visible spectrum water is
      *green-dominant*, just like vegetation — that is exactly why NDWI needs NIR. Excess
      Green therefore cannot separate them, and running vegetation first silently absorbs
      every water pixel. Ordering water first, on the ``blue > red`` signature that is unique
      to water here, is what makes the class recoverable at all.
    * **Built-up is not separated from bare soil.** Distinguishing them needs SWIR (the basis
      of NDBI); with only visible bands the two overlap so heavily that every threshold rule
      tested returned precision below 0.15 — i.e. most reported "built-up" would be soil.
      Rather than emit a plausible-looking but mostly wrong built-up map, both are reported
      as bare/unvegetated ground and a warning states the limitation.
    """
    warnings = [
        "No near-infrared band is available, so land cover is estimated from visible-colour "
        "and texture cues only. Vegetation and water separation is approximate and less "
        "reliable than a full multispectral analysis."
    ]
    roles = set(raster.band_roles)
    h, w = raster.shape
    valid = raster.valid_mask()
    class_map = np.zeros((h, w), dtype=np.uint8)
    used: list[str] = []
    evidence = np.zeros((h, w), dtype=np.float32)

    if not {BandRole.RED, BandRole.GREEN, BandRole.BLUE} <= roles:
        warnings.append(
            "The image does not have identifiable red, green and blue bands; land-cover "
            "classification could not be performed reliably."
        )
        stats, area_caveat = _class_stats(class_map, valid, evidence, raster)
        if area_caveat:
            warnings.append(area_caveat)
        return LandCoverResult(class_map, stats, "insufficient-bands", used, 0.0, warnings)

    red_raw = raster.band(BandRole.RED)
    blue_raw = raster.band(BandRole.BLUE)
    red = features.normalize01(red_raw)
    green = features.normalize01(raster.band(BandRole.GREEN))
    blue = features.normalize01(blue_raw)
    brightness = (red + green + blue) / 3.0

    # --- water first: blue exceeds red, and the surface is dark ---
    # Water absorbs strongly in red and reflects a little blue, so `blue > red` holds for
    # water and fails for soil and built-up, which are red-bright. Pairing it with a
    # darkness cut removes bright blue-grey roofs.
    bright_thr, bright_method, _ = indices.adaptive_threshold(brightness, 0.35)
    water_mask = valid & np.isfinite(red_raw) & np.isfinite(blue_raw)
    water_mask &= (blue_raw > red_raw) & (brightness < bright_thr)
    water_mask = features.clean_mask(
        water_mask, open_radius=1, close_radius=2, min_area=_min_patch_pixels(raster)
    )
    evidence = np.where(water_mask, np.clip(1.0 - brightness, 0, 1), evidence)

    # --- vegetation: Excess Green on the non-water remainder ---
    exg = indices.compute_index(raster, "exg").array
    used.append("exg")
    non_water = valid & ~water_mask
    veg_thr, veg_method, _ = _region_threshold(exg, non_water & np.isfinite(exg), 0.05)
    veg_mask = non_water & np.isfinite(exg) & (exg > veg_thr)
    evidence = np.where(veg_mask, np.clip(exg, 0, 1), evidence)

    # --- remainder: unvegetated ground, NOT split into built-up vs soil (see docstring) ---
    bare_mask = valid & ~water_mask & ~veg_mask
    warnings.append(
        "Separating built-up land from bare soil requires a short-wave infrared band, which "
        "this image does not have. Both are reported together as bare or unvegetated ground; "
        "no built-up area is claimed."
    )
    texture = features.local_variance(raster.band_mean(), window=7)
    evidence = np.where(bare_mask, texture, evidence)

    class_map[water_mask] = _CLASS_CODE[LandCoverClass.WATER]
    class_map[veg_mask] = _CLASS_CODE[LandCoverClass.VEGETATION]
    class_map[bare_mask] = _CLASS_CODE[LandCoverClass.BARE_SOIL]

    # Separability is measured, not asserted: how distinct is each detected class from the
    # rest on the cue that produced it?
    scores: list[float] = []
    if water_mask.any():
        scores.append(indices.separability(brightness, water_mask))
    if veg_mask.any():
        scores.append(indices.separability(exg, veg_mask))
    sep = float(np.mean(scores)) if scores else 0.0

    stats, area_caveat = _class_stats(class_map, valid, evidence, raster)
    if area_caveat:
        warnings.append(area_caveat)
    return LandCoverResult(
        class_map=class_map, stats=stats, method="rgb-approximation",
        indices_used=used, separability=sep, warnings=warnings,
        index_summaries={"exg": indices.compute_index(raster, "exg").to_dict()},
    )


def _class_stats(
    class_map: np.ndarray, valid: np.ndarray, evidence: np.ndarray, raster: RasterData
) -> tuple[list[ClassStats], str | None]:
    """Per-class pixel counts, coverage fractions, evidence, and (if georeferenced) area.

    Returns:
        The per-class statistics and, when the pixel area used is an approximation, the
        caveat describing why — so the caller can surface it instead of presenting an
        estimated area as a measured one.
    """
    total_valid = int(valid.sum())
    px_area, caveat = pixel_area_m2(raster.metadata)

    out: list[ClassStats] = []
    for label, code in _CLASS_CODE.items():
        if label is LandCoverClass.UNCLASSIFIED:
            continue
        mask = class_map == code
        count = int(mask.sum())
        if count == 0:
            continue
        frac = count / total_valid if total_valid else 0.0
        ev = float(np.mean(evidence[mask])) if count else 0.0
        area = count * px_area if px_area is not None else None
        out.append(ClassStats(label, count, frac, area, ev))
    return sorted(out, key=lambda s: s.fraction, reverse=True), caveat


def _mean_class_separability(idx, water_mask, veg_mask, builtup_mask) -> float:
    """Average separability across the classes that were actually detected."""
    scores: list[float] = []
    if "ndwi" in idx and water_mask.any():
        scores.append(indices.separability(idx["ndwi"].array, water_mask))
    elif "mndwi" in idx and water_mask.any():
        scores.append(indices.separability(idx["mndwi"].array, water_mask))
    if "ndvi" in idx and veg_mask.any():
        scores.append(indices.separability(idx["ndvi"].array, veg_mask))
    if "ndbi" in idx and builtup_mask.any():
        scores.append(indices.separability(idx["ndbi"].array, builtup_mask))
    return float(np.mean(scores)) if scores else 0.0
