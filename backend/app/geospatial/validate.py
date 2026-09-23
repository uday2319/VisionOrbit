"""Input validation and measured input-quality assessment.

Split into two responsibilities:

* :func:`assess_quality` — measures *how good* one raster is (dynamic range, saturation,
  nodata, sharpness). These are real measurements that feed the confidence engine, so a
  washed-out or mostly-nodata image cannot yield a HIGH-confidence answer.
* :func:`check_pair_compatibility` — decides whether two rasters may legitimately be
  compared, and if so how they must be aligned first. Comparing non-overlapping scenes is
  refused rather than silently producing a "change" map of unrelated ground.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from shapely.geometry import box

from ..core.errors import ErrorCode, GeospatialError, ValidationError
from ..core.logging import get_logger
from ..core.types import Modality
from .raster import RasterData, RasterMetadata

logger = get_logger(__name__)


class AlignmentStrategy(StrEnum):
    """How a pair must be brought onto a common grid before comparison."""

    IDENTICAL_GRID = "identical_grid"
    """Same CRS, transform and shape — directly comparable."""

    REPROJECT = "reproject"
    """Different CRS and/or transform — warp the second onto the first's grid."""

    RESAMPLE = "resample"
    """Same CRS, different resolution/shape — resample onto the first's grid."""

    PIXEL_ASSUME = "pixel_assume"
    """No georeferencing available — treat as pixel-aligned. Always warned about."""

    INCOMPATIBLE = "incompatible"
    """Cannot be compared at all."""


@dataclass
class InputQuality:
    """Measured quality of a single raster. All fields are computed, none assumed."""

    nodata_fraction: float
    dynamic_range: float
    """Mean per-band spread between the 2nd and 98th percentile, normalised 0..1."""

    saturation_fraction: float
    """Fraction of pixels at the extreme of the data range (clipped highlights/shadows)."""

    sharpness: float
    """Normalised variance-of-Laplacian; low values indicate blur or heavy resampling."""

    score: float
    """Aggregate 0..1 quality score."""

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodata_fraction": round(self.nodata_fraction, 4),
            "dynamic_range": round(self.dynamic_range, 4),
            "saturation_fraction": round(self.saturation_fraction, 4),
            "sharpness": round(self.sharpness, 4),
            "score": round(self.score, 4),
            "warnings": list(self.warnings),
        }


@dataclass
class PairCompatibility:
    """Verdict on whether two rasters may be compared, and how."""

    compatible: bool
    strategy: AlignmentStrategy
    overlap_fraction: float | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "strategy": self.strategy.value,
            "overlap_fraction": (
                round(self.overlap_fraction, 4) if self.overlap_fraction is not None else None
            ),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "details": dict(self.details),
        }


def assess_quality(raster: RasterData) -> InputQuality:
    """Measure the usability of a raster.

    Every component is an actual measurement over the pixel data; nothing here is a
    placeholder or a constant.
    """
    warnings: list[str] = []
    data = raster.data
    valid = np.isfinite(data)

    nodata_fraction = raster.nodata_fraction
    if nodata_fraction > 0.5:
        warnings.append(
            f"{nodata_fraction * 100:.0f}% of pixels are nodata or invalid; "
            f"results cover only the valid remainder."
        )
    elif nodata_fraction > 0.15:
        warnings.append(f"{nodata_fraction * 100:.0f}% of pixels are nodata or invalid.")

    # --- dynamic range, per band, on valid pixels only ---
    ranges: list[float] = []
    saturation: list[float] = []
    for b in range(data.shape[0]):
        band = data[b][valid[b]]
        if band.size == 0:
            ranges.append(0.0)
            saturation.append(1.0)
            continue
        lo, hi = np.percentile(band, [2.0, 98.0])
        bmin, bmax = float(band.min()), float(band.max())
        span = max(bmax - bmin, 1e-9)
        ranges.append(float((hi - lo) / span))
        # Pixels pinned within 0.5% of either extreme count as saturated.
        eps = span * 0.005
        sat = np.count_nonzero((band <= bmin + eps) | (band >= bmax - eps)) / band.size
        saturation.append(float(sat))

    dynamic_range = float(np.mean(ranges)) if ranges else 0.0
    saturation_fraction = float(np.mean(saturation)) if saturation else 1.0

    if dynamic_range < 0.15:
        warnings.append(
            "The image has very low contrast, which reduces the reliability of "
            "threshold-based classification."
        )
    if saturation_fraction > 0.25:
        warnings.append(
            f"{saturation_fraction * 100:.0f}% of pixels are saturated at the extremes "
            f"of the data range; spectral values there are unreliable."
        )

    # --- sharpness via variance of Laplacian on the first valid band ---
    sharpness = _sharpness(data)
    if sharpness < 0.02:
        warnings.append(
            "The image appears heavily blurred or upsampled; fine structures such as "
            "individual buildings may not be resolvable."
        )

    score = float(
        np.clip(
            0.35 * (1.0 - nodata_fraction)
            + 0.30 * min(dynamic_range / 0.5, 1.0)
            + 0.20 * (1.0 - min(saturation_fraction / 0.4, 1.0))
            + 0.15 * min(sharpness / 0.1, 1.0),
            0.0,
            1.0,
        )
    )

    return InputQuality(
        nodata_fraction=nodata_fraction,
        dynamic_range=dynamic_range,
        saturation_fraction=saturation_fraction,
        sharpness=sharpness,
        score=score,
        warnings=warnings,
    )


def _sharpness(data: np.ndarray) -> float:
    """Normalised variance-of-Laplacian sharpness estimate in ``[0, 1]``."""
    import cv2

    band = data[0]
    finite = np.isfinite(band)
    if not finite.any():
        return 0.0
    filled = np.where(finite, band, float(np.nanmedian(band)))
    lo, hi = np.percentile(filled, [1.0, 99.0])
    if hi - lo < 1e-9:
        return 0.0
    norm = np.clip((filled - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    lap = cv2.Laplacian(norm, cv2.CV_32F, ksize=3)
    return float(np.clip(lap.var() * 20.0, 0.0, 1.0))


def validate_modality(
    raster: RasterData, expected: Modality, *, strict: bool = False
) -> list[str]:
    """Check a raster's modality against what the task needs.

    Args:
        strict: if ``True`` a mismatch raises; otherwise it is returned as a warning.

    Returns:
        Warning strings (empty when the modality matches).

    Raises:
        ValidationError: on mismatch when ``strict``.
    """
    if expected is Modality.UNKNOWN or raster.modality is expected:
        return []

    if raster.modality is Modality.UNKNOWN:
        return [
            f"The modality of '{raster.metadata.path.name}' could not be determined from "
            f"its metadata; it is being processed as {expected.value}."
        ]

    message = (
        f"'{raster.metadata.path.name}' appears to be {raster.modality.value} imagery, "
        f"but this analysis requires {expected.value} imagery."
    )
    if strict:
        raise ValidationError(
            message,
            code=ErrorCode.WRONG_MODALITY,
            context={"found": raster.modality.value, "expected": expected.value},
        )
    return [message]


def _bounds_polygon(meta: RasterMetadata):
    """Shapely polygon of a raster's bounds, or ``None`` if not georeferenced."""
    if not meta.is_georeferenced or meta.bounds is None:
        return None
    minx, miny, maxx, maxy = meta.bounds
    return box(min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))


def compute_overlap_fraction(a: RasterMetadata, b: RasterMetadata) -> float | None:
    """Fraction of the *smaller* footprint covered by the intersection.

    Uses the smaller footprint as the denominator so that a small high-resolution chip fully
    contained in a large scene correctly scores ~1.0 rather than a misleadingly tiny ratio.

    Returns:
        ``0.0``–``1.0``, or ``None`` when either raster lacks georeferencing.
    """
    poly_a, poly_b = _bounds_polygon(a), _bounds_polygon(b)
    if poly_a is None or poly_b is None:
        return None
    # Implied by the polygons existing at all (both rasters are georeferenced), but stated
    # here so the reprojection below never runs on a missing CRS.
    if a.crs_wkt is None or b.crs_wkt is None:
        return None

    # Bring b into a's CRS before intersecting.
    if a.crs_wkt != b.crs_wkt:
        try:
            transformer = Transformer.from_crs(
                PyprojCRS.from_wkt(b.crs_wkt), PyprojCRS.from_wkt(a.crs_wkt), always_xy=True
            )
            minx, miny, maxx, maxy = poly_b.bounds
            xs, ys = transformer.transform(
                [minx, maxx, minx, maxx], [miny, miny, maxy, maxy]
            )
            if not all(np.isfinite(xs)) or not all(np.isfinite(ys)):
                return None
            poly_b = box(min(xs), min(ys), max(xs), max(ys))
        except Exception as exc:  # pragma: no cover - malformed CRS
            logger.warning("overlap reprojection failed", extra={"error": str(exc)})
            return None

    inter = poly_a.intersection(poly_b).area
    denom = min(poly_a.area, poly_b.area)
    if denom <= 0:
        return None
    return float(np.clip(inter / denom, 0.0, 1.0))


def check_pair_compatibility(
    a: RasterData,
    b: RasterData,
    *,
    min_overlap: float = 0.30,
    require_georeference: bool = False,
) -> PairCompatibility:
    """Decide whether ``a`` and ``b`` may be compared, and how to align them.

    Args:
        min_overlap: Minimum overlap fraction to allow comparison.
        require_georeference: Refuse ungeoreferenced input (used where real-world areas
            are essential to the answer).
    """
    warnings: list[str] = []
    errors: list[str] = []
    ma, mb = a.metadata, b.metadata
    details: dict[str, Any] = {
        "a": {"shape": [ma.height, ma.width], "crs": ma.crs_epsg or ma.crs_wkt,
              "georeferenced": ma.is_georeferenced},
        "b": {"shape": [mb.height, mb.width], "crs": mb.crs_epsg or mb.crs_wkt,
              "georeferenced": mb.is_georeferenced},
    }

    both_geo = ma.is_georeferenced and mb.is_georeferenced

    # --- neither / only one georeferenced ---
    if not both_geo:
        if require_georeference:
            errors.append(
                "Both images must carry a coordinate reference system and geotransform "
                "for this analysis. At least one does not."
            )
            return PairCompatibility(False, AlignmentStrategy.INCOMPATIBLE, None,
                                     warnings, errors, details)

        if ma.is_georeferenced != mb.is_georeferenced:
            warnings.append(
                "Only one of the two images is georeferenced. They are being compared as "
                "pixel grids; no real-world coordinates or areas can be reported."
            )
        else:
            warnings.append(
                "Neither image is georeferenced. They are assumed to cover the same area "
                "and are compared as pixel grids; results are in pixels and percentages, "
                "not real-world units."
            )

        if (ma.height, ma.width) != (mb.height, mb.width):
            warnings.append(
                f"Image dimensions differ ({ma.width}x{ma.height} vs {mb.width}x{mb.height}); "
                f"the second image is resampled to match the first. Without georeferencing "
                f"this alignment cannot be verified."
            )
        return PairCompatibility(True, AlignmentStrategy.PIXEL_ASSUME, None,
                                 warnings, errors, details)

    # --- both georeferenced: spatial overlap is decisive ---
    overlap = compute_overlap_fraction(ma, mb)
    details["overlap_fraction"] = overlap

    if overlap is None:
        warnings.append(
            "Spatial overlap between the two images could not be computed from their "
            "coordinate systems; they are compared as pixel grids."
        )
        return PairCompatibility(True, AlignmentStrategy.PIXEL_ASSUME, None,
                                 warnings, errors, details)

    if overlap <= 0.0:
        errors.append(
            "The two images do not overlap geographically, so they cannot be compared. "
            "Please supply two images covering the same area."
        )
        return PairCompatibility(False, AlignmentStrategy.INCOMPATIBLE, overlap,
                                 warnings, errors, details)

    if overlap < min_overlap:
        errors.append(
            f"The two images overlap over only {overlap * 100:.0f}% of the smaller "
            f"footprint, below the {min_overlap * 100:.0f}% minimum required for a "
            f"reliable comparison."
        )
        return PairCompatibility(False, AlignmentStrategy.INCOMPATIBLE, overlap,
                                 warnings, errors, details)

    if overlap < 0.98:
        warnings.append(
            f"The images overlap over {overlap * 100:.0f}% of the smaller footprint; "
            f"analysis is restricted to the overlapping region."
        )

    same_crs = ma.crs_wkt == mb.crs_wkt
    same_shape = (ma.height, ma.width) == (mb.height, mb.width)
    same_transform = (
        ma.transform is not None
        and mb.transform is not None
        and all(abs(x - y) < 1e-6 for x, y in zip(ma.transform, mb.transform, strict=True))
    )

    if same_crs and same_shape and same_transform:
        return PairCompatibility(True, AlignmentStrategy.IDENTICAL_GRID, overlap,
                                 warnings, errors, details)

    if not same_crs:
        warnings.append(
            f"The images use different coordinate reference systems "
            f"({ma.crs_epsg or 'custom'} and {mb.crs_epsg or 'custom'}); the second is "
            f"reprojected onto the first before comparison."
        )
        return PairCompatibility(True, AlignmentStrategy.REPROJECT, overlap,
                                 warnings, errors, details)

    pa, pb = ma.pixel_size, mb.pixel_size
    if pa and pb and (abs(pa[0] - pb[0]) > 1e-6 or abs(pa[1] - pb[1]) > 1e-6):
        warnings.append(
            f"Ground sample distance differs ({pa[0]:.2f} vs {pb[0]:.2f} CRS units); "
            f"the second image is resampled to the first image's grid. Features smaller "
            f"than the coarser pixel size cannot be resolved."
        )
    return PairCompatibility(True, AlignmentStrategy.RESAMPLE, overlap,
                             warnings, errors, details)


def require_compatible(pair: PairCompatibility) -> None:
    """Raise a user-facing error if a pair cannot be compared.

    Raises:
        GeospatialError: with the specific reason (no overlap vs insufficient overlap).
    """
    if pair.compatible:
        return
    message = pair.errors[0] if pair.errors else "The two images cannot be compared."
    code = ErrorCode.NO_SPATIAL_OVERLAP
    if pair.overlap_fraction is not None and pair.overlap_fraction > 0:
        code = ErrorCode.INSUFFICIENT_OVERLAP
    elif pair.overlap_fraction is None:
        code = ErrorCode.MISSING_GEOREFERENCE
    raise GeospatialError(message, code=code, context=pair.to_dict())
