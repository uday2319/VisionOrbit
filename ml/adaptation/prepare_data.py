"""Acquire EuroSAT RGB and record a small, reproducible train/val/test subset.

This replaces an earlier version of this module that generated patches from a hand-written table of
"Sentinel-2 typical reflectance centroids with Gaussian noise" and wrote a manifest labelled
``BigEarthNet-Synthetic``. Nothing here is synthesised: the images are genuine Sentinel-2 RGB
patches downloaded from the pinned EuroSAT mirror, and the manifest records the real file paths that
:mod:`ml.adaptation.train` will read.

Run it as::

    python -m ml.adaptation.prepare_data

which downloads ~94 MB on first use into ``ml/datasets/eurosat_raw`` (git-ignored) and writes
``ml/datasets/processed/eurosat_subset/{manifest,train,val,test}.json``. Re-running is cheap: the
download is skipped when the folder exists, and the subset is redrawn from the same seed, so the
split is identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..datasets.eurosat_adapter import (
    DEFAULT_RAW_ROOT,
    DEFAULT_SUBSET_DIR,
    EUROSAT_CLASSES,
    build_subset,
    download_eurosat,
)


def prepare_eurosat_subset(
    *,
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    output_dir: Path | str = DEFAULT_SUBSET_DIR,
    train_size: int = 2000,
    val_size: int = 400,
    test_size: int = 400,
    seed: int = 1337,
) -> dict[str, Any]:
    """Download EuroSAT if absent, then draw and record the stratified subset.

    Returns the written manifest so a caller (or the CLI below) can report what was prepared
    without re-reading it from disk.
    """
    image_root = download_eurosat(raw_root)
    return build_subset(
        image_root=image_root,
        output_dir=output_dir,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download EuroSAT (RGB) and record a reproducible training subset."
    )
    parser.add_argument("--raw_root", type=str, default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_SUBSET_DIR))
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--val_size", type=int, default=400)
    parser.add_argument("--test_size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    print("=" * 68)
    print("  SatQuery AI - preparing EuroSAT (RGB) remote-sensing subset")
    print("=" * 68)
    print(f"  Classes ({len(EUROSAT_CLASSES)}): {', '.join(EUROSAT_CLASSES)}")
    print(f"  Raw archive root: {args.raw_root}")
    print("  Downloading if absent (~94 MB, MIT licence, pinned mirror commit)...")

    manifest = prepare_eurosat_subset(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    print("-" * 68)
    print(f"  Dataset:    {manifest['dataset']} (synthetic: {manifest['is_synthetic']})")
    print(f"  Image root: {manifest['image_root']}")
    print(f"  Seed:       {manifest['seed']}")
    for split, count in manifest["splits"].items():
        print(f"  {split:<6}     {count} images "
              f"({manifest['per_class_per_split'][split]} per class)")
    print(f"  Manifest:   {Path(args.output_dir) / 'manifest.json'}")
    print("=" * 68)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(_main())
