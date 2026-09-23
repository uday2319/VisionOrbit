"""Generate deterministic synthetic remote-sensing demo data with known ground truth.

Why synthetic data is the right call for a prototype (and not a shortcut):

* Real benchmark corpora are enormous (BigEarthNet is tens of GB) and cannot ship in a repo.
* Ground truth is **exactly known** here, so geospatial tests can assert precise coordinates
  and areas, and evaluation scripts can compute *real* IoU/precision/recall/F1 rather than
  quoting numbers from a paper.
* Spectral and backscatter signatures are taken from published class behaviour, so the
  indices and fusion logic are exercised the way real imagery would exercise them.

Everything is seeded, so regenerating produces byte-comparable rasters.

Usage::

    python scripts/generate_demo_data.py            # write data/demo
    python scripts/generate_demo_data.py --force    # overwrite existing
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO_ROOT / "data" / "demo"

SEED = 20260101
SIZE = 512               # pixels per side
PIXEL_M = 10.0           # 10 m ground sample distance (Sentinel-2 like)
UTM_EPSG = 32643         # UTM zone 43N — covers much of India
ORIGIN_X = 700000.0      # easting of upper-left corner
ORIGIN_Y = 2100000.0     # northing of upper-left corner

# Class codes used in ground-truth rasters.
CLS_UNCLASSIFIED = 0
CLS_WATER = 1
CLS_VEGETATION = 2
CLS_BUILT_UP = 3
CLS_BARE_SOIL = 4

CLASS_NAMES = {
    CLS_UNCLASSIFIED: "unclassified",
    CLS_WATER: "water",
    CLS_VEGETATION: "vegetation",
    CLS_BUILT_UP: "built_up",
    CLS_BARE_SOIL: "bare_soil",
}

# Surface reflectance per class, ordered [blue, green, red, nir, swir1, swir2].
# Chosen to reproduce the real index behaviour: water NDWI>0, vegetation NDVI high,
# built-up NDBI>0. Built-up vs bare soil stay genuinely close in reflectance — that
# confusion is real, and it is what makes SAR fusion demonstrably useful.
REFLECTANCE = {
    CLS_WATER:      (0.06, 0.08, 0.05, 0.02, 0.010, 0.010),
    CLS_VEGETATION: (0.03, 0.06, 0.04, 0.45, 0.200, 0.100),
    CLS_BUILT_UP:   (0.15, 0.16, 0.18, 0.22, 0.300, 0.280),
    CLS_BARE_SOIL:  (0.12, 0.15, 0.20, 0.28, 0.320, 0.250),
}

# Per-class texture magnitude (fraction of reflectance). Built-up is heterogeneous
# (roofs/roads/shadows); water is near-specular and very smooth.
TEXTURE = {
    CLS_WATER: 0.03,
    CLS_VEGETATION: 0.10,
    CLS_BUILT_UP: 0.28,
    CLS_BARE_SOIL: 0.12,
}

# SAR VV backscatter in dB. Water is specular (very dark); built-up double-bounces
# (very bright). This is the complementary information optical indices lack.
SAR_DB = {
    CLS_WATER: -20.0,
    CLS_VEGETATION: -10.0,
    CLS_BUILT_UP: -3.5,
    CLS_BARE_SOIL: -14.0,
}

BAND_NAMES = ("blue", "green", "red", "nir", "swir1", "swir2")
REFLECTANCE_SCALE = 10000.0  # Sentinel-2 style uint16 scaling


def base_transform(pixel_m: float = PIXEL_M) -> Affine:
    """North-up affine transform for the demo grid."""
    return Affine(pixel_m, 0.0, ORIGIN_X, 0.0, -pixel_m, ORIGIN_Y)


# ---------------------------------------------------------------------------
# Scene layout
# ---------------------------------------------------------------------------
def _draw_urban_block(labels: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                      rng: np.random.Generator, *, step: int = 26,
                      road_thickness: int = 3) -> None:
    """Fill a rectangle with a built-up block pattern separated by a road grid.

    Args:
        step: Spacing of the road grid in pixels. The default matches the original demo scene.
        road_thickness: Width of each road in pixels; ``0`` draws a solid block with no roads.
            Roads are labelled bare soil, so a dense fine grid puts a large bare-soil population
            *inside* the town — and since those pixels are surrounded by structure their texture
            reads as built-up, which blurs the very bare/built-up boundary the classifier separates
            on. A sparse arterial grid keeps the town legible without that confound.
    """
    cv2.rectangle(labels, (x0, y0), (x1, y1), CLS_BUILT_UP, thickness=-1)
    if road_thickness <= 0:
        return
    # Roads read as bare surfaces between buildings.
    for x in range(x0 + step, x1, step):
        jitter = int(rng.integers(-2, 3))
        cv2.line(labels, (x + jitter, y0), (x + jitter, y1), CLS_BARE_SOIL,
                 thickness=road_thickness)
    for y in range(y0 + step, y1, step):
        jitter = int(rng.integers(-2, 3))
        cv2.line(labels, (x0, y + jitter), (x1, y + jitter), CLS_BARE_SOIL,
                 thickness=road_thickness)


def build_label_map(rng: np.random.Generator, *, epoch: int = 1) -> np.ndarray:
    """Construct the ground-truth class map.

    Args:
        rng: Seeded generator.
        epoch: ``1`` for the earlier date, ``2`` for the later date. Epoch 2 applies known
            changes (urban expansion, reservoir shrinkage, field greening) so change
            detection can be scored against exactly-known regions.
    """
    labels = np.full((SIZE, SIZE), CLS_BARE_SOIL, dtype=np.uint8)

    # --- agricultural fields (vegetation) ---
    fields = [(40, 40, 190, 150), (210, 30, 330, 120), (30, 330, 160, 470)]
    for (x0, y0, x1, y1) in fields:
        cv2.rectangle(labels, (x0, y0), (x1, y1), CLS_VEGETATION, thickness=-1)
    cv2.ellipse(labels, (410, 400), (70, 55), 20.0, 0.0, 360.0, CLS_VEGETATION, -1)

    # --- reservoir (water) ---
    reservoir_axes = (58, 42) if epoch == 1 else (44, 30)  # shrinks by epoch 2
    cv2.ellipse(labels, (150, 240), reservoir_axes, 15.0, 0.0, 360.0, CLS_WATER, -1)

    # --- river (water) ---
    ys = np.arange(0, SIZE, 4)
    xs = (330 + 38 * np.sin(ys / 70.0)).astype(np.int32)
    river = np.stack([xs, ys.astype(np.int32)], axis=1)
    cv2.polylines(labels, [river], isClosed=False, color=CLS_WATER, thickness=9)

    # --- urban core ---
    _draw_urban_block(labels, 250, 200, 360, 300, rng)

    # --- epoch-2 known changes ---
    if epoch == 2:
        # 1. Urban expansion onto former bare soil / field edge.
        _draw_urban_block(labels, 360, 200, 440, 290, rng)
        # 2. New settlement replacing part of a vegetated field.
        _draw_urban_block(labels, 60, 60, 140, 130, rng)
        # 3. Newly irrigated land turning bare soil into cropland.
        cv2.rectangle(labels, (200, 380), (300, 470), CLS_VEGETATION, thickness=-1)

    return labels


# ---------------------------------------------------------------------------
# The reliable bi-temporal pair (brief §3 points 2 and 11)
# ---------------------------------------------------------------------------
# The default pair above is deliberately *hard*: most of its change is built-up expansion onto
# bare soil, which the two detectors cannot both see (built-up and bare soil are near-identical
# in NDVI, a water index and NDBI, so only the classifier's texture cue catches them). That is
# realistic and worth keeping — but it means the change confidence is honestly capped in the
# medium band by ``evidence_strength = min(bimodality signal, corroborated fraction)``.
#
# A demonstration of the *reliable* end of the scale needs a pair where the evidence genuinely is
# strong, not one where the thresholds were loosened until it looked strong. The change here is
# reservoir impoundment: a valley floor of cropland and bare soil goes under water. That choice is
# not arbitrary — it is the one conversion both detectors independently resolve, because water's
# signature is extreme in every index (NDVI collapses from about +0.84 to −0.43, MNDWI rises from
# about −0.54 to +0.78), so the spectral detector and the land-cover classifier flag the same
# pixels. Everything outside the reservoir keeps the same land cover and the same illumination, so
# the unchanged population stays a tight mode near zero and the magnitude histogram is genuinely
# two-population. Both dates are written as georeferenced 6-band GeoTIFFs on the identical grid,
# so registration measures at sub-pixel offset without any correction being needed.
RELIABLE_SEED = SEED + 500


def build_reliable_labels(rng: np.random.Generator, *, epoch: int = 1) -> np.ndarray:
    """Class map for the reliable pair: a valley that is flooded by epoch 2.

    Both epochs must be built from a generator in the *same* state, so the shared layout — in
    particular the jittered road grid inside the towns — is pixel-identical between the dates.
    Anything else would put a genuine structural mismatch in ground that did not change, which
    depresses the measured registration NCC and manufactures class disagreement along every road.

    Args:
        rng: Seeded generator, used for the town road jitter.
        epoch: ``1`` before impoundment, ``2`` after.
    """
    # The valley floor is irrigated cropland, and that is the *default* surface here rather than
    # bare ground. The reason is the built-up classifier: built-up and bare soil are separated on
    # NDBI plus edge density, and their distributions genuinely overlap (both surfaces are
    # SWIR-bright, and ploughed ground is not perfectly smooth). Every bare-soil pixel is therefore
    # a coin-toss near that decision boundary, and with bare ground covering a third of the frame
    # sensor noise alone flipped ~16,000 pixels between the two dates — uncorroborated class
    # disagreement that has nothing to do with the change being demonstrated. Keeping bare ground to
    # a few real features (the dry wash, the quarry, the roads) shrinks that boundary population to
    # something a denoising pass can absorb.
    labels = np.full((SIZE, SIZE), CLS_VEGETATION, dtype=np.uint8)

    # Bare ground: a dry wash draining the valley, and a quarry on the eastern ridge.
    cv2.rectangle(labels, (196, 200), (228, 500), CLS_BARE_SOIL, thickness=-1)
    cv2.rectangle(labels, (430, 40), (500, 120), CLS_BARE_SOIL, thickness=-1)

    # The natural lake on the valley floor, present on BOTH dates and deliberately large.
    #
    # Its size is not cosmetic. The built-up classifier's texture cue, `features.edge_density`, is
    # percentile-stretched over the whole frame, so the *smoothest* large surface in the image sets
    # the low end of that stretch. With only a token pond, the earlier date's smoothest population
    # was cropland, bare ground stretched up to ~0.66, and bare soil and built-up ceased to be two
    # distinguishable modes — the classifier fell back to reference thresholds on that date while
    # the later date (whose reservoir anchors the stretch) used Otsu. Two different thresholds over
    # unchanged ground is what manufactured ~25,000 bare-to-built-up flips that no spectral
    # measurement corroborated. A lake on both dates anchors the stretch identically on both.
    cv2.ellipse(labels, (150, 300), (95, 55), 8.0, 0.0, 360.0, CLS_WATER, -1)
    cv2.circle(labels, (400, 290), 22, CLS_WATER, thickness=-1)

    # Two towns, unchanged, and deliberately large. Built-up is separated by requiring both a
    # SWIR-bright index *and* high edge density, with both thresholds derived from the
    # not-water/not-vegetation remainder — so what matters is built-up's share of *that* subset,
    # not of the scene. A token settlement leaves the remainder unimodal, the classifier falls
    # back to reference thresholds, and the two dates then disagree over large tracts of bare soil:
    # uncorroborated class disagreement with nothing to do with the change being demonstrated.
    # These two districts put built-up well above bare ground in that subset. Only sparse arterial
    # roads are drawn: the built-up surface's own texture supplies the edge density, so a dense grid
    # would only add bare-soil pixels that sit on the boundary it has to separate.
    _draw_urban_block(labels, 10, 10, 250, 190, rng, step=52, road_thickness=2)
    _draw_urban_block(labels, 280, 300, 500, 480, rng, step=52, road_thickness=2)

    if epoch == 2:
        # Impoundment: the dam raises the lake, and the enlarged reservoir drowns the cropland and
        # bare terraces around its old shoreline. The change is an *extension* of the water body
        # already present on date 1, so "water increased" is a measured delta against a real
        # baseline rather than a class appearing out of nothing.
        cv2.ellipse(labels, (150, 300), (120, 72), 8.0, 0.0, 360.0, CLS_WATER, -1)

    return labels


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_optical(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Render a 6-band optical scene from a class map.

    Returns:
        ``(6, H, W)`` uint16 array, reflectance scaled by 10000.
    """
    h, w = labels.shape
    out = np.zeros((len(BAND_NAMES), h, w), dtype=np.float32)

    for cls, refl in REFLECTANCE.items():
        mask = labels == cls
        if not mask.any():
            continue
        sigma = TEXTURE[cls]
        for b, value in enumerate(refl):
            noise = rng.normal(0.0, sigma * value, size=int(mask.sum())).astype(np.float32)
            out[b][mask] = value + noise

    # Mild illumination gradient — real scenes are never perfectly flat-fielded.
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    gradient = 1.0 + 0.04 * ((gx / w) - 0.5) + 0.03 * ((gy / h) - 0.5)
    out *= gradient[None, :, :]

    # Sensor blur, so class edges are not unrealistically crisp.
    for b in range(out.shape[0]):
        out[b] = cv2.GaussianBlur(out[b], (3, 3), 0.7)

    np.clip(out, 0.0, 1.2, out=out)
    return (out * REFLECTANCE_SCALE).astype(np.uint16)


def render_optical_correlated(
    labels: np.ndarray,
    *,
    ground_seed: int,
    sensor_seed: int,
    sensor_sigma: float = 0.0015,
) -> np.ndarray:
    """Render a 6-band optical scene, separating *ground* texture from *sensor* noise.

    :func:`render_optical` draws its within-class texture from the caller's generator, so two
    epochs rendered with different generators disagree pixel-by-pixel even where the ground did
    not change. That is not how a real bi-temporal pair behaves: within-class heterogeneity —
    which roof, which furrow, which patch of gravel — is a property of the *ground*, so it repeats
    almost exactly on the second pass. Only sensor noise is independent between acquisitions.

    Modelling it that way is what lets an unchanged pair legitimately measure a high registration
    NCC and a tight near-zero change magnitude: the two dates agree because the scene is the same
    scene, not because any threshold was relaxed. The texture field is drawn once over the whole
    frame from ``ground_seed`` and modulated by the per-class amplitude, so a pixel whose class is
    unchanged receives an identical perturbation on both dates, while a pixel that converted
    (cropland to reservoir) picks up the new class's reflectance and amplitude.

    Args:
        ground_seed: Seeds the shared ground-texture field. Pass the *same* value for both epochs.
        sensor_seed: Seeds the independent per-acquisition noise. Pass a *different* value per date.
        sensor_sigma: Noise-equivalent reflectance of the independent term, in reflectance units.

    Returns:
        ``(6, H, W)`` uint16 array, reflectance scaled by 10000.
    """
    h, w = labels.shape
    bands = len(BAND_NAMES)
    ground = np.random.default_rng(ground_seed)
    # One unit-variance field per band, drawn independently of the class map so it does not shift
    # when the reservoir changes how many pixels each class owns.
    field = ground.normal(0.0, 1.0, size=(bands, h, w)).astype(np.float32)

    base = np.zeros((bands, h, w), dtype=np.float32)
    amplitude = np.zeros((h, w), dtype=np.float32)
    for cls, refl in REFLECTANCE.items():
        mask = labels == cls
        if not mask.any():
            continue
        amplitude[mask] = TEXTURE[cls]
        for b, value in enumerate(refl):
            base[b][mask] = value

    # Multiplicative in reflectance, matching render_optical's sigma * value convention.
    out = base * (1.0 + field * amplitude[None, :, :])

    # Illumination gradient: scene geometry, so shared between the two dates.
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    gradient = 1.0 + 0.04 * ((gx / w) - 0.5) + 0.03 * ((gy / h) - 0.5)
    out *= gradient[None, :, :]

    for b in range(bands):
        out[b] = cv2.GaussianBlur(out[b], (3, 3), 0.7)

    # The only term that differs between the acquisitions.
    sensor = np.random.default_rng(sensor_seed)
    out += sensor.normal(0.0, sensor_sigma, size=out.shape).astype(np.float32)

    np.clip(out, 0.0, 1.2, out=out)
    return (out * REFLECTANCE_SCALE).astype(np.uint16)


def render_sar(labels: np.ndarray, rng: np.random.Generator, looks: int = 6) -> np.ndarray:
    """Render a single-band SAR VV scene in dB, with realistic multiplicative speckle.

    Speckle for an ``looks``-look intensity image is Gamma distributed with shape
    ``looks`` and mean 1 — modelling it multiplicatively in linear power (not additively
    in dB) is what makes the Lee filter a meaningful thing to run downstream.
    """
    h, w = labels.shape
    linear = np.zeros((h, w), dtype=np.float32)
    for cls, db in SAR_DB.items():
        mask = labels == cls
        if mask.any():
            linear[mask] = 10.0 ** (db / 10.0)

    speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=(h, w)).astype(np.float32)
    linear *= speckle
    linear = cv2.GaussianBlur(linear, (3, 3), 0.5)  # system impulse response
    linear = np.maximum(linear, 1e-6)
    return (10.0 * np.log10(linear)).astype(np.float32)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_raster(
    path: Path,
    array: np.ndarray,
    *,
    transform: Affine,
    epsg: int | None,
    descriptions: tuple[str, ...] | None = None,
    tags: dict[str, str] | None = None,
    nodata: float | None = None,
) -> None:
    """Write a (bands, H, W) or (H, W) array as a GeoTIFF."""
    if array.ndim == 2:
        array = array[None, :, :]
    count, height, width = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype=array.dtype, crs=CRS.from_epsg(epsg) if epsg else None,
        transform=transform, nodata=nodata, compress="deflate",
    ) as ds:
        ds.write(array)
        if descriptions:
            for i, name in enumerate(descriptions[:count], start=1):
                ds.set_band_description(i, name)
        if tags:
            ds.update_tags(**tags)


def write_quicklook_png(path: Path, optical: np.ndarray) -> None:
    """Write a plain RGB PNG with no georeferencing.

    Used to prove the system refuses to invent coordinates for ordinary images.
    """
    rgb = optical[[2, 1, 0]].astype(np.float32)  # red, green, blue
    lo = np.percentile(rgb, 2.0)
    hi = np.percentile(rgb, 98.0)
    stretched = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = (stretched * 255).astype(np.uint8).transpose(1, 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def write_edge_cases(root: Path) -> list[str]:
    """Create the malformed and mismatched inputs the edge-case tests require (brief §23)."""
    root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    rng = np.random.default_rng(SEED + 99)
    labels = build_label_map(rng, epoch=1)
    optical = render_optical(labels, rng)

    # Zero-byte file.
    (root / "empty.tif").write_bytes(b"")
    created.append("empty.tif")

    # Valid TIFF header, truncated body.
    tmp = root / "_full.tif"
    write_raster(tmp, optical, transform=base_transform(), epsg=UTM_EPSG,
                 descriptions=BAND_NAMES)
    blob = tmp.read_bytes()
    (root / "corrupt.tif").write_bytes(blob[: len(blob) // 3])
    tmp.unlink()
    created.append("corrupt.tif")

    # Not an image at all, despite a plausible extension.
    (root / "not_an_image.pdf").write_bytes(b"%PDF-1.4\n% not really a pdf\n")
    created.append("not_an_image.pdf")

    # Same scene in a geographic CRS -> exercises reprojection.
    deg = 10.0 / 111_320.0
    write_raster(
        root / "different_crs.tif", optical,
        transform=Affine(deg, 0, 77.5, 0, -deg, 18.9), epsg=4326,
        descriptions=BAND_NAMES,
    )
    created.append("different_crs.tif")

    # Coarser resolution -> exercises safe resampling.
    coarse = optical[:, ::2, ::2]
    write_raster(root / "different_res.tif", coarse,
                 transform=base_transform(PIXEL_M * 2), epsg=UTM_EPSG,
                 descriptions=BAND_NAMES)
    created.append("different_res.tif")

    # Far-away extent -> must be rejected for pair analysis, not silently compared.
    write_raster(
        root / "disjoint_extent.tif", optical,
        transform=Affine(PIXEL_M, 0, ORIGIN_X + 500_000, 0, -PIXEL_M, ORIGIN_Y + 500_000),
        epsg=UTM_EPSG, descriptions=BAND_NAMES,
    )
    created.append("disjoint_extent.tif")

    # RGB only -> no NIR, so NDVI/NDWI are impossible and the system must degrade
    # gracefully with a warning instead of faking an index.
    write_raster(root / "rgb_only.tif", optical[[2, 1, 0]],
                 transform=base_transform(), epsg=UTM_EPSG,
                 descriptions=("red", "green", "blue"))
    created.append("rgb_only.tif")

    # Georeference absent entirely.
    write_raster(root / "no_georeference.tif", optical, transform=Affine.identity(),
                 epsg=None, descriptions=BAND_NAMES)
    created.append("no_georeference.tif")

    return created


def _rotate_and_shift(
    array: np.ndarray, *, degrees: float, dx: float, dy: float
) -> np.ndarray:
    """Apply a rotation about the centre plus a translation, band by band.

    A *rotation* is used on purpose. The co-registration stage
    (:func:`app.geospatial.align.coregister`) estimates and removes a pure translation, so a
    merely shifted pair would be corrected and would stop being an unreliable case. Uncorrected
    rotation is what actually survives translation-only co-registration — and it is a real failure
    mode, produced by an ungeoreferenced export whose orientation was never resolved.
    """
    single = array.ndim == 2
    stack = array[None, :, :] if single else array
    h, w = stack.shape[1:]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    out = np.stack(
        [
            cv2.warpAffine(
                band.astype(np.float32), matrix, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )
            for band in stack
        ]
    ).astype(array.dtype)
    return out[0] if single else out


def write_unreliable_pair(root: Path) -> list[str]:
    """Create the *unreliable* bi-temporal pair the safety tests need (brief §3 point 12).

    This reproduces, deliberately and reproducibly, the input configuration that was reported as
    a bad flagship demo: no CRS, no transform, no NIR band, and a geometric mismatch a
    translation-only co-registration cannot remove. Every one of those is a genuine defect in the
    *input*, not a tightened threshold — the pipeline must go on measuring the pixel difference
    while declining to name a land-cover conversion, and must say why.

    Returns:
        The file names created, for the manifest.
    """
    root.mkdir(parents=True, exist_ok=True)
    labels_t1 = build_reliable_labels(np.random.default_rng(RELIABLE_SEED), epoch=1)
    labels_t2 = build_reliable_labels(np.random.default_rng(RELIABLE_SEED), epoch=2)
    optical_t1 = render_optical_correlated(
        labels_t1, ground_seed=RELIABLE_SEED + 10, sensor_seed=RELIABLE_SEED + 77
    )
    optical_t2 = render_optical_correlated(
        labels_t2, ground_seed=RELIABLE_SEED + 10, sensor_seed=RELIABLE_SEED + 79
    )

    # The later date is rotated 4 degrees and shifted ~60 px: the same ground, unresolvably
    # mismatched under a translation model.
    skewed_t2 = _rotate_and_shift(optical_t2, degrees=4.0, dx=48.0, dy=-36.0)

    created: list[str] = []
    # Georeference absent entirely (identity transform, no CRS) — a plain image, not a map.
    for name, array in (("pair_t1.tif", optical_t1), ("pair_t2.tif", skewed_t2)):
        write_raster(root / name, array, transform=Affine.identity(), epsg=None,
                     descriptions=BAND_NAMES)
        created.append(name)

    # And as ordinary 8-bit RGB PNGs: no georeference, no NIR, so no vegetation or water index can
    # be computed from either date and change falls back to raw-band differencing.
    write_quicklook_png(root / "pair_t1.png", optical_t1)
    write_quicklook_png(root / "pair_t2.png", skewed_t2)
    created.extend(["pair_t1.png", "pair_t2.png"])
    return created


def class_pixel_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        CLASS_NAMES[c]: int((labels == c).sum())
        for c in sorted(CLASS_NAMES)
        if (labels == c).any()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SatQuery AI demo data")
    parser.add_argument("--out", type=Path, default=DEMO_ROOT)
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args(argv)

    out: Path = args.out
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"[skip] {out} already populated; pass --force to regenerate")
        return 0
    if out.exists() and args.force:
        shutil.rmtree(out)

    rng = np.random.default_rng(SEED)
    transform = base_transform()
    geo_tags_t1 = {"ACQUISITION_DATE": "2024-01-15", "SENSOR": "synthetic-optical-6band"}
    geo_tags_t2 = {"ACQUISITION_DATE": "2025-01-20", "SENSOR": "synthetic-optical-6band"}

    # --- epoch 1 (also the single-image demo scene) ---
    labels_t1 = build_label_map(rng, epoch=1)
    optical_t1 = render_optical(labels_t1, rng)
    sar_t1 = render_sar(labels_t1, rng)

    # --- epoch 2 ---
    rng2 = np.random.default_rng(SEED + 1)
    labels_t2 = build_label_map(rng2, epoch=2)
    optical_t2 = render_optical(labels_t2, rng2)

    opt_dir, sar_dir, tmp_dir = out / "optical", out / "sar", out / "temporal"

    write_raster(opt_dir / "scene_optical.tif", optical_t1, transform=transform,
                 epsg=UTM_EPSG, descriptions=BAND_NAMES, tags=geo_tags_t1)
    write_raster(opt_dir / "scene_landcover_gt.tif", labels_t1, transform=transform,
                 epsg=UTM_EPSG, descriptions=("landcover_class",))
    write_quicklook_png(opt_dir / "scene_quicklook.png", optical_t1)

    write_raster(sar_dir / "scene_sar_vv.tif", sar_t1, transform=transform,
                 epsg=UTM_EPSG, descriptions=("vv",),
                 tags={"SENSOR": "synthetic-SAR-VV", "UNIT": "dB",
                       "ACQUISITION_DATE": "2024-01-16"})

    write_raster(tmp_dir / "scene_t1_optical.tif", optical_t1, transform=transform,
                 epsg=UTM_EPSG, descriptions=BAND_NAMES, tags=geo_tags_t1)
    write_raster(tmp_dir / "scene_t2_optical.tif", optical_t2, transform=transform,
                 epsg=UTM_EPSG, descriptions=BAND_NAMES, tags=geo_tags_t2)
    write_raster(tmp_dir / "scene_t1_landcover_gt.tif", labels_t1, transform=transform,
                 epsg=UTM_EPSG, descriptions=("landcover_class",))
    write_raster(tmp_dir / "scene_t2_landcover_gt.tif", labels_t2, transform=transform,
                 epsg=UTM_EPSG, descriptions=("landcover_class",))

    change_gt = (labels_t1 != labels_t2).astype(np.uint8)
    write_raster(tmp_dir / "change_gt.tif", change_gt, transform=transform,
                 epsg=UTM_EPSG, descriptions=("changed",))

    # --- the reliable pair: georeferenced, aligned, one unambiguous conversion ---
    rel_dir = out / "temporal_reliable"
    # Same generator state for both label maps: the unchanged layout must be pixel-identical.
    rel_labels_t1 = build_reliable_labels(np.random.default_rng(RELIABLE_SEED), epoch=1)
    rel_labels_t2 = build_reliable_labels(np.random.default_rng(RELIABLE_SEED), epoch=2)
    # Ground texture is shared (it is the same ground); only sensor noise differs per date.
    rel_optical_t1 = render_optical_correlated(
        rel_labels_t1, ground_seed=RELIABLE_SEED + 10, sensor_seed=RELIABLE_SEED + 1
    )
    rel_optical_t2 = render_optical_correlated(
        rel_labels_t2, ground_seed=RELIABLE_SEED + 10, sensor_seed=RELIABLE_SEED + 2
    )
    rel_tags_t1 = {"ACQUISITION_DATE": "2024-03-08", "SENSOR": "synthetic-optical-6band"}
    rel_tags_t2 = {"ACQUISITION_DATE": "2025-03-11", "SENSOR": "synthetic-optical-6band"}
    write_raster(rel_dir / "reservoir_t1_optical.tif", rel_optical_t1, transform=transform,
                 epsg=UTM_EPSG, descriptions=BAND_NAMES, tags=rel_tags_t1)
    write_raster(rel_dir / "reservoir_t2_optical.tif", rel_optical_t2, transform=transform,
                 epsg=UTM_EPSG, descriptions=BAND_NAMES, tags=rel_tags_t2)
    write_raster(rel_dir / "reservoir_t1_landcover_gt.tif", rel_labels_t1, transform=transform,
                 epsg=UTM_EPSG, descriptions=("landcover_class",))
    write_raster(rel_dir / "reservoir_t2_landcover_gt.tif", rel_labels_t2, transform=transform,
                 epsg=UTM_EPSG, descriptions=("landcover_class",))
    rel_change_gt = (rel_labels_t1 != rel_labels_t2).astype(np.uint8)
    write_raster(rel_dir / "reservoir_change_gt.tif", rel_change_gt, transform=transform,
                 epsg=UTM_EPSG, descriptions=("changed",))

    unreliable = write_unreliable_pair(out / "temporal_unreliable")
    edge_cases = write_edge_cases(out / "edge_cases")

    pixel_area = PIXEL_M * PIXEL_M

    def _transition_counts(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
        """Exact pixel count per ``from -> to`` conversion, for scoring semantic claims."""
        counts: dict[str, int] = {}
        for src in sorted(CLASS_NAMES):
            for dst in sorted(CLASS_NAMES):
                if src == dst:
                    continue
                n = int(((a == src) & (b == dst)).sum())
                if n:
                    counts[f"{CLASS_NAMES[src]}->{CLASS_NAMES[dst]}"] = n
        return counts

    manifest = {
        "generator": "scripts/generate_demo_data.py",
        "seed": SEED,
        "note": (
            "Synthetic imagery with exactly-known ground truth. Spectral and backscatter "
            "signatures follow published class behaviour so indices and fusion behave as "
            "they would on real scenes. These are NOT real satellite acquisitions and the "
            "coordinates are illustrative."
        ),
        "grid": {
            "width": SIZE, "height": SIZE, "pixel_size_m": PIXEL_M,
            "crs": f"EPSG:{UTM_EPSG}", "pixel_area_m2": pixel_area,
            "transform": list(base_transform())[:6],
            "bounds": [ORIGIN_X, ORIGIN_Y - SIZE * PIXEL_M,
                       ORIGIN_X + SIZE * PIXEL_M, ORIGIN_Y],
        },
        "optical_bands": list(BAND_NAMES),
        "class_codes": {str(k): v for k, v in CLASS_NAMES.items()},
        "reflectance": {CLASS_NAMES[k]: list(v) for k, v in REFLECTANCE.items()},
        "sar_backscatter_db": {CLASS_NAMES[k]: v for k, v in SAR_DB.items()},
        "ground_truth": {
            "t1_class_pixels": class_pixel_counts(labels_t1),
            "t2_class_pixels": class_pixel_counts(labels_t2),
            "t1_class_area_m2": {k: v * pixel_area
                                 for k, v in class_pixel_counts(labels_t1).items()},
            "changed_pixels": int(change_gt.sum()),
            "changed_area_m2": float(change_gt.sum() * pixel_area),
            "changed_fraction": float(change_gt.mean()),
        },
        "known_changes": [
            "urban expansion east of the existing built-up core (bare soil -> built-up)",
            "new settlement replacing part of the north-west field (vegetation -> built-up)",
            "reservoir shrinkage (water -> bare soil)",
            "newly irrigated cropland in the south (bare soil -> vegetation)",
        ],
        # Which input a bi-temporal demonstration should use, and why. Proper GeoTIFFs on a common
        # grid are the preferred input: without a CRS and transform there is no ground area, no
        # reprojection, and no way to tell a real conversion from a misalignment.
        "preferred_bitemporal_input": {
            "path": "temporal_reliable/reservoir_t1_optical.tif"
                    " + temporal_reliable/reservoir_t2_optical.tif",
            "why": (
                "Both dates are 6-band GeoTIFFs with an explicit CRS, transform and bounds on one "
                "identical grid, so registration is sub-pixel, ground areas are real, and the one "
                "conversion present is corroborated independently by both change detectors."
            ),
            "avoid": (
                "PNG or otherwise ungeoreferenced input. It carries no CRS, transform or NIR band, "
                "so no vegetation or water index exists, change falls back to raw-band "
                "differencing, extent can only be reported in pixels, and no land-cover conversion "
                "can be attributed to the ground. See temporal_unreliable/ for that case."
            ),
        },
        "reliable_pair": {
            "note": (
                "The high-quality bi-temporal case: georeferenced, co-registered, with one "
                "spectrally unambiguous conversion (cropland and bare soil to water) that both "
                "the spectral and land-cover detectors resolve independently."
            ),
            "before": "temporal_reliable/reservoir_t1_optical.tif",
            "after": "temporal_reliable/reservoir_t2_optical.tif",
            "change_mask": "temporal_reliable/reservoir_change_gt.tif",
            "known_change": "reservoir impoundment floods the valley floor",
            "t1_class_pixels": class_pixel_counts(rel_labels_t1),
            "t2_class_pixels": class_pixel_counts(rel_labels_t2),
            "changed_pixels": int(rel_change_gt.sum()),
            "changed_area_m2": float(rel_change_gt.sum() * pixel_area),
            "changed_fraction": float(rel_change_gt.mean()),
            "transition_pixels": _transition_counts(rel_labels_t1, rel_labels_t2),
        },
        "unreliable_pair": {
            "note": (
                "The safety case: the same ground, but the later date is rotated 4 degrees and "
                "shifted about 60 px, and neither file carries a CRS or transform. The rotation is "
                "deliberate — translation-only co-registration cannot remove it — so the "
                "registration gate must fail and the pipeline must report the measured pixel "
                "difference while withholding any named land-cover conversion."
            ),
            "geotiff_pair": ["temporal_unreliable/pair_t1.tif", "temporal_unreliable/pair_t2.tif"],
            "png_pair": ["temporal_unreliable/pair_t1.png", "temporal_unreliable/pair_t2.png"],
            "defects": [
                "no CRS and identity transform (not a map)",
                "later date rotated 4 degrees about the centre and shifted (48, -36) px",
                "PNG variant is 8-bit RGB only, so NDVI/NDWI/NDBI cannot be computed",
            ],
            "files": unreliable,
        },
        "edge_cases": edge_cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[ok] demo data written to {out}")
    print(f"     changed pixels: {manifest['ground_truth']['changed_pixels']} "
          f"({manifest['ground_truth']['changed_fraction'] * 100:.2f}%)")
    for name, count in manifest["ground_truth"]["t1_class_pixels"].items():
        print(f"     t1 {name:<13} {count:>7} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
