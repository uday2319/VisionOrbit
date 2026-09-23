"""CRS-aware ground measurement.

One number — the ground area of a single pixel — underlies every reported area and every
"ignore patches smaller than X" filter in the system. Getting it wrong is silent: no
exception, just an area off by a factor of ten or a filter that deletes every detection.
Three traps are handled explicitly here, because each of them has a plausible-looking wrong
answer:

* **Geographic CRS.** A pixel measured in degrees has no fixed area. Treating ``0.0000898``
  as if it were metres makes a pixel ``8·10⁻⁹`` m², so a 2500 m² minimum patch becomes
  3·10¹¹ pixels and silently removes everything. Area is computed geodesically at the
  scene's centre latitude instead, and flagged when the scene is tall enough for that
  centre value to be a poor summary.
* **Non-metric projected CRS.** A State Plane grid in US survey feet has a "10" pixel that
  covers 9.29 m², not 100 m². The CRS's own axis unit-conversion factor is applied rather
  than assuming metres.
* **No georeferencing.** Returns ``None`` so callers report pixels, never invented metres
  (brief §8).
"""
from __future__ import annotations

import math
from typing import NamedTuple

from pyproj import CRS as PyprojCRS
from pyproj import Geod

from ..core.logging import get_logger
from .raster import RasterMetadata

logger = get_logger(__name__)

# Above this relative variation the centre-latitude pixel area stops being a fair summary of
# the whole scene, and saying so is more useful than a single number. 1% is well below any
# threshold that would change a decision, so the caveat appears before the error matters.
_LATITUDE_VARIATION_CAVEAT = 0.01


class PixelArea(NamedTuple):
    """Ground area of one pixel.

    Attributes:
        area_m2: Square metres per pixel, or ``None`` when it cannot be known.
        caveat: A user-facing sentence when the value is an approximation, else ``None``.
            Present so callers surface the limitation rather than quietly rounding it away.
    """

    area_m2: float | None
    caveat: str | None


def _linear_unit_factors(crs: PyprojCRS) -> tuple[float, float]:
    """Metres per CRS unit for the x and y axes.

    Both axes are read rather than one, because area is the product of two lengths and a CRS
    is free to declare different units per axis.
    """
    factors: list[float] = []
    for axis in list(crs.axis_info)[:2]:
        factor = getattr(axis, "unit_conversion_factor", None)
        factors.append(float(factor) if factor else 1.0)
    while len(factors) < 2:
        factors.append(1.0)
    return factors[0], factors[1]


def _geod_for(crs: PyprojCRS) -> Geod:
    """A geodesic calculator on the CRS's own ellipsoid, falling back to WGS84."""
    ellipsoid = getattr(crs, "ellipsoid", None)
    if ellipsoid is not None:
        a = getattr(ellipsoid, "semi_major_metre", None)
        rf = getattr(ellipsoid, "inverse_flattening", None)
        if a and rf:
            try:
                return Geod(a=float(a), rf=float(rf))
            except Exception:  # pragma: no cover - malformed ellipsoid definition
                pass
    return Geod(ellps="WGS84")


def _geographic_pixel_area(meta: RasterMetadata, crs: PyprojCRS) -> PixelArea:
    """Geodesic area of the centre pixel of a degree-based raster."""
    ps, bounds = meta.pixel_size, meta.bounds
    if ps is None or bounds is None:
        return PixelArea(None, None)

    lon0 = (bounds[0] + bounds[2]) / 2.0
    lat0 = (bounds[1] + bounds[3]) / 2.0
    dlon, dlat = ps
    try:
        # Corners in order: SW, SE, NE, NW. Both the third and fourth vertices must advance
        # in latitude — repeating one collapses the quad into a triangle and halves the area,
        # which looks entirely plausible in the output.
        area, _ = _geod_for(crs).polygon_area_perimeter(
            [lon0, lon0 + dlon, lon0 + dlon, lon0],
            [lat0, lat0, lat0 + dlat, lat0 + dlat],
        )
    except Exception as exc:  # pragma: no cover - degenerate coordinates
        logger.warning("geodesic pixel area failed", extra={"error": str(exc)})
        return PixelArea(None, None)

    area = abs(float(area))
    if not math.isfinite(area) or area <= 0.0:
        return PixelArea(None, None)

    # A pixel's east-west extent shrinks with cos(latitude), so its area varies down the
    # scene by roughly tan(φ)·Δφ. Deriving the caveat from that quantity rather than from a
    # fixed "is the scene big" rule means it appears exactly when it is true.
    lat_span_deg = abs(bounds[3] - bounds[1])
    variation = abs(math.tan(math.radians(min(abs(lat0), 89.0)))) * math.radians(lat_span_deg)
    caveat = None
    if variation > _LATITUDE_VARIATION_CAVEAT:
        caveat = (
            f"Areas are computed from the pixel size at the centre of the scene. Because "
            f"this image spans {lat_span_deg:.1f}° of latitude, the true ground area of a "
            f"pixel varies by about {variation * 100:.0f}% between its top and bottom edges."
        )
    return PixelArea(area, caveat)


def pixel_area_m2(meta: RasterMetadata) -> PixelArea:
    """Ground area of one pixel in square metres.

    Returns:
        A :class:`PixelArea` whose ``area_m2`` is ``None`` when the raster carries no usable
        georeferencing — the signal for callers to report pixel counts instead.
    """
    if not meta.is_georeferenced or meta.crs_wkt is None:
        return PixelArea(None, None)

    ps = meta.pixel_size
    if ps is None or ps[0] <= 0.0 or ps[1] <= 0.0:
        return PixelArea(None, None)

    try:
        crs = PyprojCRS.from_wkt(meta.crs_wkt)
    except Exception as exc:  # pragma: no cover - malformed WKT
        logger.warning("CRS could not be parsed", extra={"error": str(exc)})
        return PixelArea(None, None)

    if crs.is_geographic:
        return _geographic_pixel_area(meta, crs)

    fx, fy = _linear_unit_factors(crs)
    area = float(ps[0] * fx) * float(ps[1] * fy)
    if not math.isfinite(area) or area <= 0.0:  # pragma: no cover - defensive
        return PixelArea(None, None)
    return PixelArea(area, None)


def pixels_for_ground_area(
    meta: RasterMetadata,
    area_m2: float,
    *,
    default_pixels: int,
    max_frame_fraction: float = 0.25,
) -> int:
    """How many pixels cover ``area_m2`` of ground on this raster's grid.

    Used to express minimum-detection sizes as physical areas, which keeps them meaningful
    across resolutions: 2500 m² is 25 pixels at 10 m but a single pixel at 50 m.

    Args:
        default_pixels: Returned when pixel area is unknowable, so an ungeoreferenced image
            still gets the same denoising rather than none.
        max_frame_fraction: Ceiling as a share of the frame. A minimum-patch size larger
            than a quarter of the image cannot filter noise — it can only erase every
            detection — so an implausible pixel size is capped and logged instead of
            silently emptying the result.
    """
    area = pixel_area_m2(meta).area_m2
    if area is None or area <= 0.0:
        return max(1, default_pixels)

    wanted = max(1, int(round(area_m2 / area)))
    frame = int(meta.width) * int(meta.height)
    if frame <= 0:  # pragma: no cover - defensive
        return wanted
    cap = max(1, int(frame * max_frame_fraction))
    if wanted > cap:
        logger.warning(
            "minimum patch size exceeds a plausible share of the frame; capping",
            extra={"requested_px": wanted, "cap_px": cap, "pixel_area_m2": area},
        )
        return cap
    return wanted
