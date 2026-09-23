"""Raster ingestion: metadata extraction, safe decimated reads, and band-role resolution.

Two ideas carry most of the weight here:

1. **Decimated reads.** Large scenes are downsampled *by GDAL during read* via
   ``out_shape``, so a 10000x10000 upload never materialises as a full array in RAM
   (brief §23 "huge image", §31).
2. **Honest band roles.** Spectral indices are meaningless if the wrong band is treated as
   NIR. Roles are resolved from band descriptions, then from count+driver convention, and
   otherwise left :attr:`BandRole.UNKNOWN` — never guessed.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioError, RasterioIOError

from ..core.errors import ErrorCode, ValidationError
from ..core.logging import get_logger
from ..core.types import BandRole, Modality

logger = get_logger(__name__)

# Keyword -> role, checked against lowercased band descriptions.
_DESCRIPTION_HINTS: tuple[tuple[tuple[str, ...], BandRole], ...] = (
    (("swir2", "b12", "swir 2"), BandRole.SWIR2),
    (("swir1", "b11", "swir 1", "swir"), BandRole.SWIR1),
    (("nir", "b08", "b8", "near infrared", "near-infrared"), BandRole.NIR),
    (("red", "b04", "b4"), BandRole.RED),
    (("green", "b03", "b3"), BandRole.GREEN),
    (("blue", "b02", "b2"), BandRole.BLUE),
    (("vv",), BandRole.VV),
    (("vh",), BandRole.VH),
    (("gray", "grey", "amplitude", "intensity", "backscatter", "sigma0"), BandRole.GRAY),
    (("alpha", "mask"), BandRole.ALPHA),
)

# Sentinel-2 style orderings, used when count matches and descriptions are absent.
_S2_13 = [
    BandRole.UNKNOWN, BandRole.BLUE, BandRole.GREEN, BandRole.RED,
    BandRole.UNKNOWN, BandRole.UNKNOWN, BandRole.UNKNOWN, BandRole.NIR,
    BandRole.UNKNOWN, BandRole.UNKNOWN, BandRole.UNKNOWN, BandRole.SWIR1, BandRole.SWIR2,
]
_S2_12 = [
    BandRole.UNKNOWN, BandRole.BLUE, BandRole.GREEN, BandRole.RED,
    BandRole.UNKNOWN, BandRole.UNKNOWN, BandRole.UNKNOWN, BandRole.NIR,
    BandRole.UNKNOWN, BandRole.UNKNOWN, BandRole.SWIR1, BandRole.SWIR2,
]

_NON_GEO_DRIVERS = frozenset({"PNG", "JPEG", "GIF", "BMP"})


@dataclass(frozen=True)
class RasterMetadata:
    """Everything known about a raster's geometry and provenance."""

    path: Path
    driver: str
    width: int
    height: int
    count: int
    dtype: str
    crs_wkt: str | None
    crs_epsg: int | None
    transform: tuple[float, float, float, float, float, float] | None
    bounds: tuple[float, float, float, float] | None
    nodata: float | None
    band_descriptions: tuple[str | None, ...] = ()
    tags: dict[str, str] = field(default_factory=dict)
    # Set when the read was decimated; lets callers convert analysis pixels back to
    # source pixels honestly.
    decimation: float = 1.0

    @property
    def is_georeferenced(self) -> bool:
        """True only when a real CRS *and* a non-identity transform are present.

        A plain JPEG/PNG must never be treated as carrying coordinates (brief §8).
        """
        if self.crs_wkt is None or self.transform is None:
            return False
        a, b, c, d, e, f = self.transform
        is_identity = (a, b, c, d, e, f) == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        return not is_identity

    @property
    def is_north_up(self) -> bool:
        """True when rows run north-to-south with no rotation or shear.

        This is what makes a compass reading of the pixel grid legitimate: only on a north-up
        grid is the top of the array the north of the ground. A rotated granule — a Sentinel-1
        product in its acquisition geometry, say — has the same pixels in a different
        orientation, so answering "what is in the north of this image" from row indices would
        report the wrong part of the scene. An ungeoreferenced image is never north-up, because
        without a CRS there is no north to be up.
        """
        if not self.is_georeferenced or self.transform is None:
            return False
        _, b, _, d, e, _ = self.transform
        return b == 0.0 and d == 0.0 and e < 0.0

    @property
    def pixel_size(self) -> tuple[float, float] | None:
        """(x_size, y_size) in CRS units, or ``None`` without a transform."""
        if self.transform is None:
            return None
        a, b, _, d, e, _ = self.transform
        return (math.hypot(a, d), math.hypot(b, e))

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary for API responses and reports."""
        px = self.pixel_size
        return {
            "filename": self.path.name,
            "driver": self.driver,
            "width": self.width,
            "height": self.height,
            "bands": self.count,
            "dtype": self.dtype,
            "crs": f"EPSG:{self.crs_epsg}" if self.crs_epsg else (self.crs_wkt[:80] if self.crs_wkt else None),
            "crs_epsg": self.crs_epsg,
            "transform": list(self.transform) if self.transform else None,
            "bounds": list(self.bounds) if self.bounds else None,
            "nodata": self.nodata,
            "pixel_size": list(px) if px else None,
            "georeferenced": self.is_georeferenced,
            "band_descriptions": list(self.band_descriptions),
            "decimation": self.decimation,
        }


@dataclass
class RasterData:
    """A loaded raster: float32 values plus resolved semantics.

    Attributes:
        data: ``(bands, height, width)`` float32. Nodata and masked pixels are ``NaN`` so
            index maths propagates invalidity instead of silently averaging in zeros.
        metadata: Geometry/provenance of the *read* array (``decimation`` recorded).
        band_roles: Resolved role per band, parallel to ``data``'s first axis.
        modality: Declared or inferred sensing modality.
    """

    data: np.ndarray
    metadata: RasterMetadata
    band_roles: list[BandRole]
    modality: Modality = Modality.UNKNOWN

    @property
    def height(self) -> int:
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        return int(self.data.shape[2])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def has_role(self, role: BandRole) -> bool:
        return role in self.band_roles

    def band(self, role: BandRole) -> np.ndarray:
        """Return the band with ``role``.

        Raises:
            ValidationError: if absent. Callers must check :meth:`has_role` first; failing
                loudly is deliberate so an index is never computed from a wrong band.
        """
        try:
            idx = self.band_roles.index(role)
        except ValueError as exc:
            raise ValidationError(
                f"This image does not have an identifiable {role.value} band, "
                f"which is required for the requested analysis.",
                code=ErrorCode.MISSING_BAND,
                context={"available": [r.value for r in self.band_roles], "requested": role.value},
            ) from exc
        return self.data[idx]

    def valid_mask(self) -> np.ndarray:
        """Boolean mask of pixels finite in *every* band."""
        return np.all(np.isfinite(self.data), axis=0)

    @property
    def nodata_fraction(self) -> float:
        total = self.height * self.width
        if total == 0:
            return 1.0
        return float(1.0 - (np.count_nonzero(self.valid_mask()) / total))

    def band_mean(self) -> np.ndarray:
        """Collapse every band to one 2-D array, averaging only the valid ones per pixel."""
        return band_mean(self.data)


def band_mean(data: np.ndarray) -> np.ndarray:
    """Mean across the band axis of a ``(bands, height, width)`` stack, ignoring ``NaN``.

    ``np.nanmean(data, axis=0)`` computes the same thing but emits ``RuntimeWarning: Mean of
    empty slice`` whenever some pixel has no valid band at all. That is a routine condition,
    not an anomaly: a rotated Sentinel-2 granule has blank corners, a cloud mask blanks whole
    regions, and a clipped scene has nodata margins. Every caller here already reads ``NaN``
    as "nothing measured at this pixel", so the warning reports a handled case and only
    trains readers to ignore the log.

    The mean is therefore formed from the valid count directly, and a pixel with no valid
    band comes back ``NaN`` — the same answer, without the false alarm.
    """
    if data.ndim != 3:
        raise ValueError(f"expected a (bands, height, width) stack, got shape {data.shape}")
    finite = np.isfinite(data)
    count = finite.sum(axis=0)
    total = np.where(finite, data, 0.0).sum(axis=0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return np.where(count > 0, mean, np.nan).astype(np.float32)


def _resolve_from_descriptions(descriptions: Sequence[str | None]) -> list[BandRole] | None:
    """Map band descriptions to roles. Returns ``None`` if too few are recognised."""
    roles: list[BandRole] = []
    for desc in descriptions:
        role = BandRole.UNKNOWN
        if desc:
            low = desc.strip().lower()
            for keywords, candidate in _DESCRIPTION_HINTS:
                if any(k == low for k in keywords) or any(k in low for k in keywords):
                    role = candidate
                    break
        roles.append(role)
    recognised = sum(1 for r in roles if r is not BandRole.UNKNOWN)
    # Require a majority to be recognised, otherwise the descriptions are probably
    # free-text ("Band 1") and convention is the better guide.
    return roles if recognised >= max(1, len(roles) // 2) else None


def resolve_band_roles(
    count: int,
    descriptions: Sequence[str | None] = (),
    driver: str = "GTiff",
    modality: Modality = Modality.UNKNOWN,
) -> list[BandRole]:
    """Resolve semantic roles for ``count`` bands.

    Order of preference: band descriptions -> count/driver convention -> ``UNKNOWN``.
    """
    if (
        descriptions
        and len(descriptions) == count
        and (from_desc := _resolve_from_descriptions(descriptions)) is not None
    ):
        return from_desc

    if modality is Modality.SAR:
        if count == 1:
            return [BandRole.GRAY]
        if count == 2:
            return [BandRole.VV, BandRole.VH]
        return [BandRole.UNKNOWN] * count

    if count == 1:
        return [BandRole.GRAY]
    if count == 2:
        return [BandRole.GRAY, BandRole.ALPHA]
    if count == 3:
        return [BandRole.RED, BandRole.GREEN, BandRole.BLUE]
    if count == 4:
        # RGBA for web formats; RGB+NIR is the norm for 4-band GeoTIFFs (NAIP etc.).
        if driver.upper() in _NON_GEO_DRIVERS:
            return [BandRole.RED, BandRole.GREEN, BandRole.BLUE, BandRole.ALPHA]
        return [BandRole.RED, BandRole.GREEN, BandRole.BLUE, BandRole.NIR]
    if count == 6:
        return [
            BandRole.BLUE, BandRole.GREEN, BandRole.RED,
            BandRole.NIR, BandRole.SWIR1, BandRole.SWIR2,
        ]
    if count == 12:
        return list(_S2_12)
    if count == 13:
        return list(_S2_13)
    return [BandRole.UNKNOWN] * count


def read_metadata(path: str | Path) -> RasterMetadata:
    """Read metadata without loading pixels.

    Raises:
        ValidationError: if the file cannot be opened as a raster.
    """
    p = Path(path)
    if not p.exists():
        raise ValidationError(
            "The requested image file could not be found.",
            code=ErrorCode.INVALID_FILE,
            context={"path": str(p)},
        )
    if p.stat().st_size == 0:
        raise ValidationError(
            "The uploaded file is empty.",
            code=ErrorCode.EMPTY_FILE,
            context={"path": str(p)},
        )
    try:
        with rasterio.open(p) as ds:
            crs_wkt = ds.crs.to_wkt() if ds.crs else None
            crs_epsg: int | None = None
            if ds.crs:
                try:
                    crs_epsg = ds.crs.to_epsg()
                except Exception:  # pragma: no cover - exotic CRS without EPSG
                    crs_epsg = None
            return RasterMetadata(
                path=p,
                driver=ds.driver or "unknown",
                width=int(ds.width),
                height=int(ds.height),
                count=int(ds.count),
                dtype=str(ds.dtypes[0]) if ds.dtypes else "unknown",
                crs_wkt=crs_wkt,
                crs_epsg=crs_epsg,
                transform=tuple(ds.transform)[:6] if ds.transform else None,
                bounds=tuple(ds.bounds) if ds.transform else None,
                nodata=ds.nodata,
                band_descriptions=tuple(ds.descriptions or ()),
                tags={str(k): str(v) for k, v in (ds.tags() or {}).items()},
            )
    except (RasterioIOError, RasterioError) as exc:
        raise ValidationError(
            "The uploaded file could not be interpreted as a valid raster image. "
            "Supported formats are GeoTIFF, TIFF, PNG and JPEG.",
            code=ErrorCode.INVALID_GEOTIFF,
            context={"path": str(p), "error": str(exc)},
        ) from exc


def compute_decimated_shape(
    width: int, height: int, max_edge: int
) -> tuple[int, int, float]:
    """Return ``(out_height, out_width, factor)`` fitting within ``max_edge``.

    ``factor`` is the linear downsample ratio (1.0 = no decimation).
    """
    longest = max(width, height)
    if longest <= max_edge or max_edge <= 0:
        return height, width, 1.0
    factor = longest / max_edge
    out_w = max(1, int(round(width / factor)))
    out_h = max(1, int(round(height / factor)))
    return out_h, out_w, factor


def load_raster(
    path: str | Path,
    *,
    max_edge: int | None = 1024,
    modality: Modality = Modality.UNKNOWN,
    band_roles: Sequence[BandRole] | None = None,
    max_pixels: int | None = None,
) -> RasterData:
    """Load a raster as float32 with nodata as ``NaN``.

    Args:
        path: Raster file path.
        max_edge: Decimate on read so the longest edge fits this. ``None`` reads full size.
        modality: Declared modality; informs band-role resolution.
        band_roles: Explicit override, e.g. from an operator who knows the band order.
        max_pixels: Reject sources larger than this (decompression-bomb guard).

    Raises:
        ValidationError: unreadable raster, no bands, or over the pixel budget.
    """
    meta = read_metadata(path)

    if meta.count == 0:
        raise ValidationError(
            "The raster contains no image bands.",
            code=ErrorCode.CORRUPT_RASTER,
            context={"path": str(meta.path)},
        )
    if max_pixels is not None and (meta.width * meta.height) > max_pixels:
        raise ValidationError(
            f"The image is too large to process ({meta.megapixels:.0f} megapixels). "
            f"The limit is {max_pixels / 1_000_000:.0f} megapixels.",
            code=ErrorCode.FILE_TOO_LARGE,
            context={"width": meta.width, "height": meta.height},
        )

    out_h, out_w, factor = compute_decimated_shape(
        meta.width, meta.height, max_edge if max_edge else max(meta.width, meta.height)
    )

    try:
        with rasterio.open(meta.path) as ds:
            arr = ds.read(
                out_shape=(ds.count, out_h, out_w),
                resampling=Resampling.average if factor > 1.0 else Resampling.nearest,
                masked=True,
            )
    except (RasterioIOError, RasterioError) as exc:
        raise ValidationError(
            "The raster could be opened but its pixel data could not be read; "
            "the file may be truncated or corrupt.",
            code=ErrorCode.CORRUPT_RASTER,
            context={"path": str(meta.path), "error": str(exc)},
        ) from exc

    # MaskedArray -> float32 with NaN so invalid pixels propagate through index maths.
    data = np.ma.filled(arr.astype(np.float32), np.nan)
    if meta.nodata is not None and not math.isnan(meta.nodata):
        data = np.where(data == np.float32(meta.nodata), np.nan, data)

    # Recompute the transform for the decimated grid so pixel->world stays correct.
    new_transform = meta.transform
    if meta.transform is not None and factor != 1.0:
        src = rasterio.transform.Affine(*meta.transform)
        scaled = src * rasterio.transform.Affine.scale(
            meta.width / out_w, meta.height / out_h
        )
        new_transform = tuple(scaled)[:6]

    read_meta = RasterMetadata(
        path=meta.path,
        driver=meta.driver,
        width=out_w,
        height=out_h,
        count=meta.count,
        dtype=meta.dtype,
        crs_wkt=meta.crs_wkt,
        crs_epsg=meta.crs_epsg,
        transform=new_transform,
        bounds=meta.bounds,
        nodata=meta.nodata,
        band_descriptions=meta.band_descriptions,
        tags=meta.tags,
        decimation=factor,
    )

    resolved = (
        list(band_roles)
        if band_roles is not None and len(band_roles) == meta.count
        else resolve_band_roles(meta.count, meta.band_descriptions, meta.driver, modality)
    )

    inferred = modality
    if inferred is Modality.UNKNOWN:
        inferred = infer_modality(read_meta, resolved)

    logger.debug(
        "raster loaded",
        extra={
            "file": meta.path.name, "shape": [out_h, out_w],
            "bands": meta.count, "decimation": factor,
            "roles": [r.value for r in resolved], "modality": inferred.value,
        },
    )
    return RasterData(data=data, metadata=read_meta, band_roles=resolved, modality=inferred)


def infer_modality(meta: RasterMetadata, roles: Sequence[BandRole]) -> Modality:
    """Best-effort modality inference from metadata.

    Returns :attr:`Modality.UNKNOWN` rather than guessing when signals conflict — the
    caller (or the user) should declare it instead of the system inventing sensor metadata.
    """
    haystack = " ".join(
        [meta.path.name.lower(), *(f"{k}={v}".lower() for k, v in meta.tags.items())]
    )
    sar_tokens = ("sar", "sentinel-1", "sentinel1", "s1a", "s1b", "grd", "slc",
                  "backscatter", "sigma0", "radar", "alos", "palsar", "risat", "eos-04")
    opt_tokens = ("optical", "sentinel-2", "sentinel2", "landsat", "msi", "rgb",
                  "multispectral", "resourcesat", "liss")
    if any(t in haystack for t in sar_tokens):
        return Modality.SAR
    if any(t in haystack for t in opt_tokens):
        return Modality.OPTICAL
    if BandRole.VV in roles or BandRole.VH in roles:
        return Modality.SAR
    # Colour or NIR bands are only meaningful for optical sensors.
    if any(r in roles for r in (BandRole.RED, BandRole.GREEN, BandRole.BLUE, BandRole.NIR)):
        return Modality.OPTICAL
    return Modality.UNKNOWN
