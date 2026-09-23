"""Render the imagery a result page shows: a base composite plus the task's own overlay.

One implementation, shared by the live API (:mod:`app.api.routes.analysis`) and the recorded-fixture
generator (``scripts/generate_frontend_fixtures.py``). It previously existed *only* in that script,
which is why a recorded demo case showed a classified overlay while a live run of the very same tool
showed "No rendered imagery for this analysis." — the whole overlay toolkit in
:mod:`app.geospatial.viz` was reachable from a script and from nowhere in the running application.

Nothing here invents pixels. Every overlay is a colourised view of an array the analysis actually
produced (``class_map`` / ``regime_map`` / ``change_mask`` / ``mask``), and ``bounds_wgs84`` is
``None`` for an ungeoreferenced input so the UI falls back to a plain pixel viewer instead of
claiming world coordinates it does not have (§8).
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

from ..core.logging import get_logger
from ..core.types import LandCoverClass, ScatteringRegime
from . import viz
from .raster import RasterData

logger = get_logger(__name__)

BASE_FILENAME = "base.png"
OVERLAY_FILENAME = "overlay.png"

_LANDCOVER_NAMES: dict[int, str] = {c.code: c.value for c in LandCoverClass if c.code != 0}
_LANDCOVER_LEGEND: list[tuple[str, str]] = [
    (c.value.replace("_", " ").title(), c.value) for c in LandCoverClass if c.code != 0
]
_REGIME_COLORS: dict[int, tuple[int, int, int]] = {
    ScatteringRegime.SMOOTH.code: (33, 102, 205),
    ScatteringRegime.DIFFUSE.code: (46, 154, 62),
    ScatteringRegime.DOUBLE_BOUNCE.code: (214, 96, 45),
}
_REGIME_LABELS: dict[int, str] = {
    ScatteringRegime.SMOOTH.code: "Smooth",
    ScatteringRegime.DIFFUSE.code: "Diffuse",
    ScatteringRegime.DOUBLE_BOUNCE.code: "Double bounce",
}

# A tool may hold its map directly or carry the analysis it was phrased from: a captioning or VQA
# answer keeps the ``LandCoverResult`` behind its sentence, and the adapted scene tool wraps that
# again. Walking these named carriers is what lets those answers show the class map they rest on
# rather than a bare input image.
_CARRIER_ATTRS = ("landcover", "caption")


def wgs84_bounds(raster: RasterData) -> dict[str, float] | None:
    """Axis-aligned lat/lon box enclosing the raster footprint, for a Leaflet image overlay.

    ``None`` for an ungeoreferenced image — the UI then falls back to a plain pixel viewer rather
    than pretending the picture has coordinates (§8).
    """
    md = raster.metadata
    if not md.is_georeferenced or md.bounds is None or md.crs_epsg is None:
        return None
    minx, miny, maxx, maxy = md.bounds
    tf = Transformer.from_crs(f"EPSG:{md.crs_epsg}", "EPSG:4326", always_xy=True)
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    lonlat = [tf.transform(x, y) for x, y in corners]
    lons = [p[0] for p in lonlat]
    lats = [p[1] for p in lonlat]
    return {
        "south": min(lats),
        "west": min(lons),
        "north": max(lats),
        "east": max(lons),
    }


def input_meta(raster: RasterData) -> dict[str, Any]:
    """Per-input raster metadata for the result page's Inputs panel.

    The raster's own header plus the two things resolved at load time — the modality the analysis
    actually used and the band roles it mapped — and the WGS84 footprint when there is one.
    """
    meta = raster.metadata.to_dict()
    meta["modality"] = raster.modality.value
    meta["band_roles"] = [r.value for r in raster.band_roles]
    meta["bounds_wgs84"] = wgs84_bounds(raster)
    return meta


def _render_codes(code_map: np.ndarray, code_to_color: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Colourise an integer code map; codes with no colour stay black."""
    h, w = code_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for code, color in code_to_color.items():
        out[code_map == code] = color
    return out


def _hex(color: tuple[int, int, int]) -> str:
    r, g, b = color
    return f"#{r:02x}{g:02x}{b:02x}"


def _regime_legend() -> list[dict[str, str]]:
    """Legend for the SAR regime map, derived from the colours the map is drawn with.

    Not a parallel hardcoded list: the swatch and the pixel come from the same table, so a colour
    change cannot leave the legend describing the previous rendering.
    """
    return [
        {
            "label": _REGIME_LABELS[regime.code],
            "color": _hex(_REGIME_COLORS[regime.code]),
            "key": regime.value,
        }
        for regime in (
            ScatteringRegime.SMOOTH,
            ScatteringRegime.DIFFUSE,
            ScatteringRegime.DOUBLE_BOUNCE,
        )
    ]


def _iter_carriers(raw: Any, depth: int = 2) -> Iterator[Any]:
    """Yield ``raw`` and the nested results it carries, breadth-first, without revisiting."""
    seen: set[int] = set()
    queue: list[tuple[Any, int]] = [(raw, 0)]
    while queue:
        obj, level = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        if level >= depth:
            continue
        for name in _CARRIER_ATTRS:
            queue.append((getattr(obj, name, None), level + 1))


@dataclass(frozen=True)
class _Overlay:
    """One rendered overlay: the image, the kind/label the UI keys layers on, and its legend."""

    image: np.ndarray
    kind: str
    label: str
    legend: list[dict[str, str]]


def _mask_overlay(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Blend ``color`` into the base where ``mask`` is set, outlining the regions.

    A mask that is not on the base's grid is rendered on its own instead of blended. Alignment puts
    every pair on ``rasters[0]``'s grid, so this is a guard rather than a normal path — but a shape
    mismatch must not take the whole rendering down.
    """
    if mask.shape != base_rgb.shape[:2]:
        return _render_codes(mask.astype(np.uint8), {1: color})
    return viz.overlay_mask(base_rgb, mask, color)


def _overlay_for(raw: Any, base_rgb: np.ndarray) -> _Overlay | None:
    """The overlay for whichever map the tool's raw result holds, or ``None`` if it holds none."""
    for obj in _iter_carriers(raw):
        class_map = getattr(obj, "class_map", None)
        if class_map is not None:
            return _Overlay(
                viz.render_class_map(class_map, _LANDCOVER_NAMES),
                "class_map",
                "Land-cover classification",
                viz.make_legend(_LANDCOVER_LEGEND),
            )
        regime_map = getattr(obj, "regime_map", None)
        if regime_map is not None:
            return _Overlay(
                _render_codes(regime_map, _REGIME_COLORS),
                "regime_map",
                "SAR scattering regimes",
                _regime_legend(),
            )
        change_mask = getattr(obj, "change_mask", None)
        if change_mask is not None:
            return _Overlay(
                _mask_overlay(base_rgb, change_mask, viz.CLASS_COLORS["change"]),
                "change_mask",
                "Detected change",
                viz.make_legend([("Changed", "change")]),
            )
        mask = getattr(obj, "mask", None)
        if mask is not None:
            return _Overlay(
                _mask_overlay(base_rgb, mask, viz.CLASS_COLORS["highlight"]),
                "grounding",
                "Grounded regions",
                viz.make_legend([("Matched region", "highlight")]),
            )
    return None


def _descriptor(
    kind: str,
    label: str,
    url: str,
    bounds: dict[str, float] | None,
    legend: list[dict[str, str]],
) -> dict[str, Any]:
    return {"kind": kind, "label": label, "url": url, "bounds_wgs84": bounds, "legend": legend}


def render_artifacts(
    raw: Any,
    rasters: Sequence[RasterData],
    *,
    out_dir: Path,
    url_prefix: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Write the base and overlay PNGs for one analysis and describe them for the API.

    Args:
        raw: The tool's raw domain result — the object holding the arrays to colourise.
        rasters: The rasters the analysis read. ``rasters[0]`` is the reference grid every map is
            produced on, so it is the one the base composite is rendered from.
        out_dir: Directory the PNGs are written into (created if absent).
        url_prefix: URL prefix the written files are served under, without a trailing slash.

    Returns:
        ``(artifacts, warnings)`` — the descriptors with the base first, and any warning raised
        while rendering. A rendering failure is *reported*, not swallowed: the analysis itself is
        unaffected, so it still returns its answer, but the missing imagery is stated instead of
        looking like an input that simply had nothing to show (§27).
    """
    if not rasters:
        return [], []

    primary = rasters[0]
    bounds = wgs84_bounds(primary)
    directory = Path(out_dir)

    try:
        base = viz.render_rgb(primary)
        viz.save_png(directory / BASE_FILENAME, base)
    except Exception as exc:
        logger.exception("failed to render the base image for display", extra={"out_dir": str(out_dir)})
        return [], [
            f"The input image could not be rendered for display ({type(exc).__name__}); "
            "the analysis itself is unaffected."
        ]

    artifacts = [
        _descriptor("base", "Input image", f"{url_prefix}/{BASE_FILENAME}", bounds, [])
    ]
    warnings: list[str] = []

    try:
        overlay = _overlay_for(raw, base)
        if overlay is not None:
            viz.save_png(directory / OVERLAY_FILENAME, overlay.image)
            artifacts.append(
                _descriptor(
                    overlay.kind,
                    overlay.label,
                    f"{url_prefix}/{OVERLAY_FILENAME}",
                    bounds,
                    overlay.legend,
                )
            )
    except Exception as exc:
        logger.exception("failed to render the analysis overlay", extra={"out_dir": str(out_dir)})
        warnings.append(
            f"The analysis overlay could not be rendered for display ({type(exc).__name__}); "
            "the measurements it would have drawn are unaffected and are reported below."
        )

    return artifacts, warnings
