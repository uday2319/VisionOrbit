"""Turning binary masks into vector regions with honest, CRS-aware measurements.

The rule enforced throughout: **areas in m² are reported only when a real CRS and affine
transform exist.** For an ungeoreferenced image the same shapes are returned but measured in
pixels, and ``area_m2`` is ``None`` — never a fabricated real-world number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from pyproj import CRS as PyprojCRS
from pyproj import Geod
from rasterio.transform import Affine, xy
from shapely.geometry import Polygon, mapping
from shapely.geometry import shape as shapely_shape

from ..core.logging import get_logger
from .measure import pixel_area_m2
from .raster import RasterMetadata

logger = get_logger(__name__)


@dataclass
class Region:
    """One detected connected region.

    Attributes:
        pixel_area: Area in pixels — always available.
        area_m2: Area in square metres, or ``None`` when the source is not georeferenced.
        bbox_pixel: ``(col_min, row_min, col_max, row_max)`` in pixel coordinates.
        bbox_geo: World-coordinate bbox, or ``None`` when not georeferenced.
        polygon_pixel: Simplified exterior ring in ``(col, row)`` order.
        polygon_geo: Exterior ring in world coordinates, or ``None``.
        centroid_geo: World centroid ``(x, y)``, or ``None``.
        score: Optional per-region strength (e.g. mean index value) in ``[0, 1]``.
        label: This region's value in :attr:`VectorizeResult.labels`, so a caller can recover
            the region's exact pixels without labelling the mask a second time. ``0`` means the
            region did not come from a labelling pass.
    """

    region_id: int
    pixel_area: int
    area_m2: float | None
    bbox_pixel: tuple[int, int, int, int]
    bbox_geo: tuple[float, float, float, float] | None
    polygon_pixel: list[tuple[float, float]]
    polygon_geo: list[tuple[float, float]] | None
    centroid_geo: tuple[float, float] | None
    score: float | None = None
    label: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "pixel_area": self.pixel_area,
            "area_m2": round(self.area_m2, 2) if self.area_m2 is not None else None,
            "bbox_pixel": list(self.bbox_pixel),
            "bbox_geo": list(self.bbox_geo) if self.bbox_geo else None,
            "polygon_pixel": [[round(x, 1), round(y, 1)] for x, y in self.polygon_pixel],
            "polygon_geo": (
                [[round(x, 6), round(y, 6)] for x, y in self.polygon_geo]
                if self.polygon_geo else None
            ),
            "centroid_geo": (
                [round(self.centroid_geo[0], 6), round(self.centroid_geo[1], 6)]
                if self.centroid_geo else None
            ),
            "score": round(self.score, 4) if self.score is not None else None,
        }

    def to_geojson_feature(self) -> dict[str, Any] | None:
        """GeoJSON feature in lon/lat, or ``None`` if not georeferenced."""
        if self.polygon_geo is None:
            return None
        return {
            "type": "Feature",
            "geometry": mapping(Polygon(self.polygon_geo)),
            "properties": {
                "id": self.region_id,
                "area_m2": round(self.area_m2, 2) if self.area_m2 is not None else None,
                "score": round(self.score, 4) if self.score is not None else None,
            },
        }


@dataclass
class VectorizeResult:
    """Result of vectorising a mask.

    Attributes:
        labels: The connected-component label array the regions were derived from, or ``None``
            when the mask was empty. Kept so that per-region statistics against *another*
            raster — a sector mask, an index, a change magnitude — can be computed from the
            exact pixels of each region, rather than by re-labelling the mask in every caller
            or by approximating a region with its simplified polygon.
    """

    regions: list[Region]
    total_pixels: int
    total_area_m2: float | None
    coverage_fraction: float
    georeferenced: bool
    warnings: list[str] = field(default_factory=list)
    labels: np.ndarray | None = None

    @property
    def count(self) -> int:
        return len(self.regions)

    def pixels_of(self, region: Region) -> np.ndarray:
        """Boolean mask of one region's exact pixels.

        Raises:
            ValueError: If no label array is available, which would otherwise be answered with
                a silently empty mask that reads as "this region has no pixels".
        """
        if self.labels is None or region.label <= 0:
            raise ValueError(
                "region pixels are unavailable: this result carries no component labels"
            )
        return self.labels == region.label

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_pixels": self.total_pixels,
            "total_area_m2": (
                round(self.total_area_m2, 2) if self.total_area_m2 is not None else None
            ),
            "coverage_fraction": round(self.coverage_fraction, 4),
            "georeferenced": self.georeferenced,
            "regions": [r.to_dict() for r in self.regions],
            "warnings": list(self.warnings),
        }

    def to_geojson(self) -> dict[str, Any]:
        feats = [f for r in self.regions if (f := r.to_geojson_feature()) is not None]
        return {"type": "FeatureCollection", "features": feats}


def _pixel_polygon_to_geo(
    ring: list[tuple[float, float]], transform: Affine
) -> list[tuple[float, float]]:
    """Map a ``(col, row)`` ring to world coordinates via the affine transform."""
    out: list[tuple[float, float]] = []
    for col, row in ring:
        x, y = xy(transform, row, col, offset="ul")
        out.append((float(x), float(y)))
    return out


def _geodesic_area_m2(ring_geo: list[tuple[float, float]], crs_wkt: str) -> float | None:
    """Area of a geographic-CRS ring in m², via geodesic computation on the ellipsoid."""
    try:
        crs = PyprojCRS.from_wkt(crs_wkt)
        geod = Geod(ellps="WGS84") if crs.is_geographic else None
        if geod is None:
            return None
        lons = [p[0] for p in ring_geo]
        lats = [p[1] for p in ring_geo]
        area, _ = geod.polygon_area_perimeter(lons, lats)
        return abs(area)
    except Exception:  # pragma: no cover
        return None


def vectorize_mask(
    mask: np.ndarray,
    metadata: RasterMetadata,
    *,
    min_pixels: int = 25,
    max_regions: int = 200,
    simplify_tolerance: float = 1.5,
    score_map: np.ndarray | None = None,
) -> VectorizeResult:
    """Convert a boolean mask into georeferenced regions.

    Args:
        mask: 2-D boolean array; ``True`` marks detected pixels.
        metadata: Grid the mask lives on (supplies CRS/transform).
        min_pixels: Drop specks smaller than this — noise, not objects.
        max_regions: Keep only the largest N regions to bound response size.
        simplify_tolerance: Douglas–Peucker tolerance (pixels) for polygon simplification.
        score_map: Optional per-pixel strength; a region's ``score`` is its masked mean.

    Returns:
        A :class:`VectorizeResult`. When ``metadata`` is not georeferenced, geometry is
        pixel-only and every ``area_m2`` is ``None``.
    """
    warnings: list[str] = []
    mask_bool = mask.astype(bool)
    total_pixels = int(mask_bool.sum())
    frame_pixels = int(mask_bool.size)
    coverage = total_pixels / frame_pixels if frame_pixels else 0.0

    georef = metadata.is_georeferenced
    transform = Affine(*metadata.transform) if metadata.transform else None

    is_geographic = False
    px_area_m2: float | None = None
    if georef and transform is not None:
        try:
            crs = PyprojCRS.from_wkt(metadata.crs_wkt) if metadata.crs_wkt else None
            is_geographic = bool(crs and crs.is_geographic)
        except Exception:  # pragma: no cover
            is_geographic = False
        if not is_geographic:
            # Per-region geodesic area is used for geographic CRS below; for a projected one
            # a single pixel area suffices, taken CRS-unit-aware so a grid in feet is not
            # reported as if it were metres.
            px_area_m2 = pixel_area_m2(metadata).area_m2

    if total_pixels == 0:
        return VectorizeResult([], 0, 0.0 if georef else None, 0.0, georef, warnings)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), connectivity=8
    )

    candidates: list[tuple[int, int]] = [
        (lbl, int(stats[lbl, cv2.CC_STAT_AREA]))
        for lbl in range(1, num_labels)
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_pixels
    ]
    candidates.sort(key=lambda t: t[1], reverse=True)
    if len(candidates) > max_regions:
        warnings.append(
            f"{len(candidates)} regions were detected; only the {max_regions} largest are "
            f"reported individually. Totals below still reflect all detected pixels."
        )
        candidates = candidates[:max_regions]

    regions: list[Region] = []
    total_area_m2 = 0.0 if georef else None

    for out_id, (lbl, area_px) in enumerate(candidates, start=1):
        comp = (labels == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, simplify_tolerance, closed=True)
        ring = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(ring) < 3:
            x, y, w, h = cv2.boundingRect(contour)
            ring = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        x, y, w, h = cv2.boundingRect(contour)
        bbox_pixel = (int(x), int(y), int(x + w), int(y + h))

        polygon_geo: list[tuple[float, float]] | None = None
        bbox_geo: tuple[float, float, float, float] | None = None
        centroid_geo: tuple[float, float] | None = None
        area_m2: float | None = None

        if georef and transform is not None:
            polygon_geo = _pixel_polygon_to_geo(ring, transform)
            xs = [p[0] for p in polygon_geo]
            ys = [p[1] for p in polygon_geo]
            bbox_geo = (min(xs), min(ys), max(xs), max(ys))
            try:
                poly = Polygon(polygon_geo)
                c = poly.centroid
                centroid_geo = (float(c.x), float(c.y))
            except Exception:  # pragma: no cover
                centroid_geo = None

            if is_geographic and metadata.crs_wkt is not None:
                # True pixel counting scaled by geodesic frame area is more robust than
                # trusting a coarse simplified polygon for tiny regions.
                area_m2 = _geodesic_area_m2(polygon_geo, metadata.crs_wkt)
            elif px_area_m2 is not None:
                area_m2 = area_px * px_area_m2
            if area_m2 is not None and total_area_m2 is not None:
                total_area_m2 += area_m2

        score: float | None = None
        if score_map is not None:
            vals = score_map[labels == lbl]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                score = float(np.clip(np.mean(vals), 0.0, 1.0))

        regions.append(
            Region(
                region_id=out_id,
                pixel_area=area_px,
                area_m2=area_m2,
                bbox_pixel=bbox_pixel,
                bbox_geo=bbox_geo,
                polygon_pixel=ring,
                polygon_geo=polygon_geo,
                centroid_geo=centroid_geo,
                score=score,
                label=int(lbl),
            )
        )

    if not georef:
        warnings.append(
            "This image is not georeferenced, so region areas are reported in pixels only; "
            "no real-world areas or coordinates are available."
        )

    return VectorizeResult(
        regions=regions,
        total_pixels=total_pixels,
        total_area_m2=total_area_m2,
        coverage_fraction=coverage,
        georeferenced=georef,
        warnings=warnings,
        labels=labels,
    )


def pixel_bbox_to_geo(
    bbox: tuple[int, int, int, int], metadata: RasterMetadata
) -> tuple[float, float, float, float] | None:
    """Project a pixel bbox to world coordinates, or ``None`` if not georeferenced."""
    if not metadata.is_georeferenced or metadata.transform is None:
        return None
    transform = Affine(*metadata.transform)
    c0, r0, c1, r1 = bbox
    x0, y0 = xy(transform, r0, c0, offset="ul")
    x1, y1 = xy(transform, r1, c1, offset="ul")
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def to_wgs84_geojson(result: VectorizeResult, metadata: RasterMetadata) -> dict[str, Any]:
    """Reproject region polygons to WGS84 lon/lat for web mapping (Leaflet).

    Falls back to the native-CRS GeoJSON if reprojection is not possible.
    """
    if not result.georeferenced or metadata.crs_wkt is None:
        return {"type": "FeatureCollection", "features": []}
    try:
        src = PyprojCRS.from_wkt(metadata.crs_wkt)
        if src.to_epsg() == 4326:
            return result.to_geojson()
        from pyproj import Transformer

        transformer = Transformer.from_crs(src, PyprojCRS.from_epsg(4326), always_xy=True)
        feats = []
        for region in result.regions:
            if region.polygon_geo is None:
                continue
            lonlat = [transformer.transform(x, y) for x, y in region.polygon_geo]
            geom = shapely_shape(mapping(Polygon(lonlat)))
            feats.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {
                        "id": region.region_id,
                        "area_m2": (
                            round(region.area_m2, 2) if region.area_m2 is not None else None
                        ),
                        "score": (
                            round(region.score, 4) if region.score is not None else None
                        ),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": feats}
    except Exception as exc:  # pragma: no cover
        logger.warning("wgs84 reprojection failed", extra={"error": str(exc)})
        return result.to_geojson()
