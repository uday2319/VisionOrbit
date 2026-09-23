"""API routes for file uploads and raster metadata extraction."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ...config import get_settings
from ...core.errors import ErrorCode, ValidationError
from ...core.ids import new_image_id
from ...core.logging import get_logger
from ...core.types import Modality
from ...db import get_db, save_image_record
from ...geospatial.raster import load_raster, read_metadata
from ...geospatial.viz import render_preview
from ...schemas import UploadResponse

logger = get_logger(__name__)
router = APIRouter(tags=["upload"])
settings = get_settings()


def _declared_modality(hint: str | None) -> Modality:
    """Resolve a caller-declared modality, or ``UNKNOWN`` to mean "infer it from the raster".

    The client knows something the file often does not. :func:`app.geospatial.raster.infer_modality`
    can only read filename tokens and band descriptions, so a genuine SAR scene exported as
    ``subset_1.tif`` with an unlabelled band infers as ``UNKNOWN`` — and an optical + undetermined
    pair normalises to ``bitemporal_pair``, which sends a cross-modal request to change detection and
    fails with "no bands in common". A UI that asked the user for an optical scene and a SAR scene
    already knows which is which, so it declares it here and inference is skipped.

    A hint that is not a known modality is rejected rather than ignored: the previous ``except
    ValueError: pass`` turned ``"SAR "`` or ``"radar"`` into a silent fall back to inference, so the
    upload reported a modality the caller had not asked for and the pair mode changed underneath it.

    Raises:
        ValidationError: ``hint`` is neither empty nor a value of :class:`Modality`.
    """
    if hint is None or not hint.strip():
        return Modality.UNKNOWN
    try:
        return Modality(hint.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(m.value for m in Modality)
        raise ValidationError(
            f"Unknown modality_hint {hint!r}. Expected one of: {allowed}, or omit it to infer "
            "the modality from the raster.",
            code=ErrorCode.WRONG_MODALITY,
            context={"received": hint, "allowed": [m.value for m in Modality]},
        ) from exc


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    modality_hint: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Upload a remote sensing raster image (GeoTIFF, TIFF, PNG, JPEG).

    Validates size and format, extracts CRS/metadata, generates a preview image, and returns the
    ``image_id`` an analysis request refers to.

    Args:
        modality_hint: ``"optical"`` or ``"sar"`` when the caller knows the sensor — for a client that
            asked the user for an optical scene and a SAR scene, this is the difference between the
            pair normalising to ``optical_sar_pair`` and it falling back to ``bitemporal_pair``
            because the file carried no recognisable sensor metadata. Omit it to infer from the
            raster; an unrecognised value is rejected, not ignored.
    """
    filename = file.filename or "uploaded_image.tif"
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ValidationError(
            f"Unsupported file format '{ext}'. Allowed formats: {', '.join(settings.allowed_extensions)}",
            code=ErrorCode.UNSUPPORTED_FORMAT,
        )

    image_id = new_image_id()
    upload_dir = settings.storage_root / "uploads" / image_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / filename

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        logger.error("Failed to write uploaded file", extra={"path": str(dest_path), "error": str(exc)})
        raise ValidationError("Could not save uploaded file.", code=ErrorCode.STORAGE_ERROR) from exc

    # Validate file size
    if dest_path.stat().st_size > settings.max_upload_bytes:
        dest_path.unlink(missing_ok=True)
        raise ValidationError(
            f"File size exceeds maximum allowed limit ({settings.max_upload_bytes // (1024*1024)} MB).",
            code=ErrorCode.FILE_TOO_LARGE,
        )

    try:
        meta = read_metadata(dest_path)
        raster = load_raster(
            dest_path,
            max_edge=settings.preview_edge,
            modality=_declared_modality(modality_hint),
        )
    except ValidationError:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise ValidationError(
            f"Invalid or corrupted image raster: {exc}",
            code=ErrorCode.CORRUPT_RASTER,
        ) from exc

    # Generate preview PNG
    preview_dir = settings.storage_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_filename = f"{image_id}.png"
    preview_path = preview_dir / preview_filename
    
    try:
        render_preview(raster, preview_path)
    except Exception as exc:
        logger.warning(f"Could not render preview image: {exc}")

    save_image_record(
        db=db,
        image_id=image_id,
        filename=filename,
        file_path=str(dest_path),
        width=meta.width,
        height=meta.height,
        bands=meta.count,
        dtype=meta.dtype,
        modality=raster.modality.value,
        crs_epsg=meta.crs_epsg,
        crs_wkt=meta.crs_wkt,
        georeferenced=meta.is_georeferenced,
        preview_path=str(preview_path),
    )

    px = meta.pixel_size
    bounds = list(meta.bounds) if meta.bounds else None

    return {
        "image_id": image_id,
        "filename": filename,
        "width": meta.width,
        "height": meta.height,
        "bands": meta.count,
        "dtype": meta.dtype,
        "crs": f"EPSG:{meta.crs_epsg}" if meta.crs_epsg else (meta.crs_wkt[:80] if meta.crs_wkt else None),
        "crs_epsg": meta.crs_epsg,
        "georeferenced": meta.is_georeferenced,
        "modality": raster.modality.value,
        "preview_url": f"/storage/previews/{preview_filename}",
        "decimation": raster.metadata.decimation,
        "pixel_size": list(px) if px else None,
        "bounds": bounds,
        "band_descriptions": [str(d) for d in meta.band_descriptions if d],
    }
