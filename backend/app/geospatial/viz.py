"""Rendering rasters and analysis outputs to PNG artifacts for the UI.

Everything the user sees is generated here from real arrays: RGB composites, index heatmaps,
mask overlays, change maps. No cosmetic invention — a heatmap colour always maps to a computed
value, and a legend is emitted alongside so the colour is interpretable.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.types import BandRole
from .raster import RasterData

# Overlay colours in RGB. Chosen to stay distinguishable in the default (light) UI.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "water": (33, 102, 205),
    "vegetation": (46, 154, 62),
    "built_up": (214, 96, 45),
    "bare_soil": (191, 168, 120),
    "change": (220, 40, 120),
    "increase": (214, 68, 45),
    "decrease": (58, 120, 214),
    "highlight": (240, 200, 30),
    "unclassified": (150, 150, 150),
}


def _percentile_stretch(band: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Stretch one band to ``uint8`` using robust percentiles, NaN-safe."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    stretched = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return (np.nan_to_num(stretched) * 255).astype(np.uint8)


def render_rgb(raster: RasterData) -> np.ndarray:
    """Render a display RGB image (H, W, 3) from a raster.

    Optical: true-colour from red/green/blue when present, else grayscale replication.
    SAR: dB backscatter as grayscale. The choice is driven by resolved band roles, never
    by assuming a fixed channel order.
    """
    roles = raster.band_roles
    if all(r in roles for r in (BandRole.RED, BandRole.GREEN, BandRole.BLUE)):
        r = _percentile_stretch(raster.data[roles.index(BandRole.RED)])
        g = _percentile_stretch(raster.data[roles.index(BandRole.GREEN)])
        b = _percentile_stretch(raster.data[roles.index(BandRole.BLUE)])
        return np.stack([r, g, b], axis=-1)

    # Single-band / SAR / unknown: grayscale from the first band.
    gray = _percentile_stretch(raster.data[0])
    return np.stack([gray, gray, gray], axis=-1)


def render_preview(raster: RasterData, output_path: str | Path) -> Path:
    """Render and save a display RGB preview PNG for a loaded raster."""
    rgb = render_rgb(raster)
    return save_png(output_path, rgb)



def render_index_heatmap(
    index: np.ndarray, *, vmin: float = -1.0, vmax: float = 1.0, colormap: int = cv2.COLORMAP_VIRIDIS
) -> np.ndarray:
    """Render a normalised index array as an RGB heatmap (invalid pixels -> black)."""
    valid = np.isfinite(index)
    scaled = np.clip((index - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    u8 = (np.nan_to_num(scaled) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, colormap)  # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = (0, 0, 0)
    return colored


def overlay_mask(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.45,
    outline: bool = True,
) -> np.ndarray:
    """Alpha-blend ``color`` where ``mask`` is true, optionally outlining the regions."""
    out = base_rgb.copy()
    mask_bool = mask.astype(bool)
    if mask_bool.any():
        color_arr = np.array(color, dtype=np.float32)
        blended = (1 - alpha) * out[mask_bool].astype(np.float32) + alpha * color_arr
        out[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)
        if outline:
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, color, thickness=2)
    return out


def draw_bboxes(
    base_rgb: np.ndarray,
    bboxes: Sequence[tuple[int, int, int, int]],
    color: tuple[int, int, int],
    *,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Draw labelled bounding boxes ``(x0, y0, x1, y1)`` on a copy of the image."""
    out = base_rgb.copy()
    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
        if labels and i < len(labels):
            cv2.putText(out, labels[i], (int(x0), max(0, int(y0) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def render_class_map(class_map: np.ndarray, code_to_name: dict[int, str]) -> np.ndarray:
    """Render an integer class map to an RGB image using :data:`CLASS_COLORS`."""
    h, w = class_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for code, name in code_to_name.items():
        color = CLASS_COLORS.get(name, CLASS_COLORS["unclassified"])
        out[class_map == code] = color
    return out


def save_png(path: str | Path, rgb: np.ndarray) -> Path:
    """Write an RGB (or grayscale) array to a PNG, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if rgb.ndim == 2:
        cv2.imwrite(str(p), rgb)
    else:
        cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return p


def make_legend(entries: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """Build a legend payload of ``{label, color}`` (color as ``#RRGGBB``) for the UI."""
    out: list[dict[str, str]] = []
    for label, key in entries:
        r, g, b = CLASS_COLORS.get(key, CLASS_COLORS["unclassified"])
        out.append({"label": label, "color": f"#{r:02x}{g:02x}{b:02x}", "key": key})
    return out


def thumbnail(rgb: np.ndarray, max_edge: int = 256) -> np.ndarray:
    """Downscale for a small preview while preserving aspect ratio."""
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return rgb
    scale = max_edge / longest
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def to_data_uri(rgb: np.ndarray, fmt: str = ".png") -> str:
    """Encode an image array as a base64 data URI (for inline previews/tests)."""
    import base64

    arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    ok, buf = cv2.imencode(fmt, arr)
    if not ok:  # pragma: no cover
        raise ValueError("failed to encode image")
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf).decode('ascii')}"


def summary_dict(**kwargs: Any) -> dict[str, Any]:
    """Small helper for building artifact-manifest dicts."""
    return {k: v for k, v in kwargs.items() if v is not None}
