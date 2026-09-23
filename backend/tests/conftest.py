"""Shared pytest fixtures.

Deliberately builds rasters *in code* with exactly-known geometry, so geospatial assertions
compare against arithmetic rather than against another implementation's output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Known-geometry constants used across geospatial tests.
TEST_EPSG = 32643
TEST_ORIGIN_X = 700000.0
TEST_ORIGIN_Y = 2100000.0
TEST_PIXEL_M = 10.0


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def demo_root(repo_root: Path) -> Path:
    """Demo data directory, generated on demand if absent."""
    root = repo_root / "data" / "demo"
    if not (root / "manifest.json").exists():
        import generate_demo_data

        generate_demo_data.main(["--force"])
    return root


@pytest.fixture(scope="session")
def demo_manifest(demo_root: Path) -> dict:
    import json

    return json.loads((demo_root / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def known_transform() -> Affine:
    """North-up affine with 10 m pixels at a known UTM origin."""
    return Affine(TEST_PIXEL_M, 0.0, TEST_ORIGIN_X, 0.0, -TEST_PIXEL_M, TEST_ORIGIN_Y)


def write_test_raster(
    path: Path,
    array: np.ndarray,
    *,
    transform: Affine | None = None,
    epsg: int | None = TEST_EPSG,
    descriptions: tuple[str, ...] | None = None,
    nodata: float | None = None,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write an array to a GeoTIFF for tests."""
    if array.ndim == 2:
        array = array[None, :, :]
    count, height, width = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype=array.dtype,
        crs=CRS.from_epsg(epsg) if epsg else None,
        transform=transform if transform is not None else Affine.identity(),
        nodata=nodata,
    ) as ds:
        ds.write(array)
        if descriptions:
            for i, name in enumerate(descriptions[:count], start=1):
                ds.set_band_description(i, name)
        if tags:
            ds.update_tags(**tags)
    return path


@pytest.fixture
def make_raster(tmp_path: Path):
    """Factory writing test GeoTIFFs into a temp dir."""

    def _make(name: str, array: np.ndarray, **kwargs) -> Path:
        return write_test_raster(tmp_path / name, array, **kwargs)

    return _make


@pytest.fixture
def six_band_scene() -> np.ndarray:
    """Deterministic 6-band optical scene, 64x64, with a known water square.

    Rows/cols 10:30 are water (high green, near-zero NIR); the rest is vegetation
    (high NIR). Chosen so NDWI and NDVI have unambiguous, hand-checkable signs.
    """
    rng = np.random.default_rng(7)
    h = w = 64
    # [blue, green, red, nir, swir1, swir2]
    veg = np.array([0.03, 0.06, 0.04, 0.45, 0.20, 0.10], dtype=np.float32)
    water = np.array([0.06, 0.08, 0.05, 0.02, 0.01, 0.01], dtype=np.float32)
    arr = np.repeat(veg[:, None, None], h * w, axis=1).reshape(6, h, w).copy()
    arr[:, 10:30, 10:30] = water[:, None, None]
    arr += rng.normal(0, 0.002, arr.shape).astype(np.float32)
    return (np.clip(arr, 0, 1.2) * 10000).astype(np.uint16)


@pytest.fixture
def water_square_bounds() -> tuple[int, int, int, int]:
    """``(row0, row1, col0, col1)`` of the water square in :func:`six_band_scene`."""
    return (10, 30, 10, 30)
