"""EuroSAT RGB adapter: a real, small, openly licensed remote-sensing dataset.

Why EuroSAT rather than BigEarthNet: BigEarthNet is 66 GB of Sentinel-2 patches, which cannot be
downloaded as part of a demo run, and the previous version of this package "solved" that by
generating synthetic patches from hand-written reflectance centroids and labelling the manifest
``BigEarthNet-Synthetic``. Synthetic spectra are not remote-sensing training data, so nothing
trained on them could honestly be called a remote-sensing adaptation. EuroSAT RGB is 94 MB, MIT
licensed, and made of genuine Sentinel-2 imagery — so the adaptation this package performs is real.

Source
------
Helber, Bischke, Dengel, Borth, *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use
and Land Cover Classification*, IEEE JSTARS, 2019 — https://github.com/phelber/EuroSAT (MIT).
Fetched through :class:`torchvision.datasets.EuroSAT`, which pulls the RGB archive from the
``torchgeo/eurosat`` Hugging Face mirror at a pinned commit, so the bytes are reproducible.

The dataset is 27,000 64x64 RGB patches over ten land-use classes. This adapter never uses all of
it: :func:`build_subset` draws a small class-stratified subset with a fixed seed and records the
exact file list, so a training run is reproducible and reviewable without re-deriving the split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The ten EuroSAT land-use classes, in torchvision's (alphabetical, ImageFolder) order. The index
# of a class in this tuple *is* its integer label, so it is the one thing a checkpoint and an
# inference run must agree on.
EUROSAT_CLASSES: tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

#: Where the raw archive is unpacked. Git-ignored; ~94 MB.
DEFAULT_RAW_ROOT = Path("ml/datasets/eurosat_raw")
#: Where the split manifest is written. Git-ignored.
DEFAULT_SUBSET_DIR = Path("ml/datasets/processed/eurosat_subset")

#: ImageNet statistics. The backbone is ImageNet-pretrained, so its inputs must be normalised the
#: way its weights expect; substituting EuroSAT's own statistics would fight the frozen features.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: ResNet-18 was trained at 224x224. EuroSAT patches are 64x64, so they are upsampled rather than
#: fed at native size, where the stride-32 backbone would collapse them to a 2x2 feature map.
INPUT_SIZE = 224

_DATASET_ID = "EuroSAT (RGB)"
_DATASET_SOURCE = (
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/"
    "c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip"
)
_DATASET_CITATION = (
    "Helber et al., EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land "
    "Cover Classification, IEEE JSTARS 12(7), 2019. MIT licence."
)


class DatasetNotPreparedError(RuntimeError):
    """Raised when a split is requested before :func:`build_subset` has written a manifest."""


@dataclass(frozen=True)
class SubsetSample:
    """One image in the prepared subset: a path on disk and its integer label."""

    path: str
    label: int

    @property
    def class_name(self) -> str:
        return EUROSAT_CLASSES[self.label]


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def download_eurosat(raw_root: Path | str = DEFAULT_RAW_ROOT) -> Path:
    """Ensure the real EuroSAT RGB archive is present locally, downloading it if needed.

    Returns the ``2750`` image-folder directory. Idempotent: torchvision skips the download when
    the folder already exists, so re-running costs a stat call rather than 94 MB.
    """
    from torchvision.datasets import EuroSAT  # imported lazily: torch is heavy

    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset = EuroSAT(root=str(raw_root), download=True)
    if tuple(dataset.classes) != EUROSAT_CLASSES:
        raise RuntimeError(
            "Downloaded EuroSAT class list does not match the expected ten classes: "
            f"{dataset.classes}"
        )
    return Path(raw_root) / "eurosat" / "2750"


def scan_class_folders(image_root: Path | str) -> dict[str, list[Path]]:
    """Map each class name to its image files, sorted, so a seeded shuffle is reproducible.

    ``Path.glob`` order is filesystem-dependent; sorting here is what makes the seed meaningful.
    """
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise DatasetNotPreparedError(
            f"EuroSAT image folder {image_root} not found. Run "
            "`python -m ml.adaptation.prepare_data` to download it."
        )
    found: dict[str, list[Path]] = {}
    for name in EUROSAT_CLASSES:
        files = sorted((image_root / name).glob("*.jpg"))
        if not files:
            raise DatasetNotPreparedError(
                f"No images found for EuroSAT class {name!r} under {image_root}."
            )
        found[name] = files
    return found


# ---------------------------------------------------------------------------
# Subset construction
# ---------------------------------------------------------------------------
def build_subset(
    *,
    image_root: Path | str,
    output_dir: Path | str = DEFAULT_SUBSET_DIR,
    train_size: int = 2000,
    val_size: int = 400,
    test_size: int = 400,
    seed: int = 1337,
) -> dict[str, Any]:
    """Draw a class-stratified train/val/test subset and write its manifest.

    Stratified because a 2000-image uniform draw from 27,000 would leave the ten classes unevenly
    represented by chance, and a linear probe trained on an accidentally skewed subset would be
    measuring the skew. Each class contributes ``size // 10`` images to each split, taken from
    disjoint slices of that class's seeded permutation — so no test image is ever seen in training.

    Returns the manifest dict (also written to ``output_dir/manifest.json``).
    """
    import numpy as np

    for name, size in (("train", train_size), ("val", val_size), ("test", test_size)):
        if size <= 0 or size % len(EUROSAT_CLASSES) != 0:
            raise ValueError(
                f"{name}_size must be a positive multiple of {len(EUROSAT_CLASSES)} so the subset "
                f"is class-balanced; got {size}."
            )

    per_class = {k: v // len(EUROSAT_CLASSES) for k, v in
                 (("train", train_size), ("val", val_size), ("test", test_size))}
    needed = sum(per_class.values())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_class = scan_class_folders(image_root)
    rng = np.random.default_rng(seed)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for label, name in enumerate(EUROSAT_CLASSES):
        files = by_class[name]
        if len(files) < needed:
            raise ValueError(
                f"EuroSAT class {name!r} has {len(files)} images but the requested subset needs "
                f"{needed} per class."
            )
        order = rng.permutation(len(files))
        cursor = 0
        for split in ("train", "val", "test"):
            take = per_class[split]
            for idx in order[cursor: cursor + take]:
                splits[split].append({"path": files[int(idx)].as_posix(), "label": label})
            cursor += take

    # Shuffle within each split so class order does not correlate with batch order.
    for rows in splits.values():
        rng.shuffle(rows)  # type: ignore[arg-type]

    manifest = {
        "dataset": _DATASET_ID,
        "is_synthetic": False,
        "source_url": _DATASET_SOURCE,
        "citation": _DATASET_CITATION,
        "licence": "MIT",
        "image_root": Path(image_root).as_posix(),
        "num_classes": len(EUROSAT_CLASSES),
        "class_names": list(EUROSAT_CLASSES),
        "seed": seed,
        "per_class_per_split": per_class,
        "splits": {k: len(v) for k, v in splits.items()},
        "input_size": INPUT_SIZE,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
    }

    for split, rows in splits.items():
        (output_dir / f"{split}.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(subset_dir: Path | str = DEFAULT_SUBSET_DIR) -> dict[str, Any]:
    """Read the prepared subset manifest, or explain how to create it."""
    path = Path(subset_dir) / "manifest.json"
    if not path.is_file():
        raise DatasetNotPreparedError(
            f"No subset manifest at {path}. Run `python -m ml.adaptation.prepare_data` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(
    split: str, subset_dir: Path | str = DEFAULT_SUBSET_DIR
) -> list[SubsetSample]:
    """The recorded samples for one split, in the order the manifest fixed."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}; expected train, val or test.")
    path = Path(subset_dir) / f"{split}.json"
    if not path.is_file():
        raise DatasetNotPreparedError(
            f"No {split} split at {path}. Run `python -m ml.adaptation.prepare_data` first."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [SubsetSample(path=r["path"], label=int(r["label"])) for r in rows]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def build_transform(train: bool) -> Any:
    """The preprocessing pipeline: resize to the backbone's input size, then ImageNet-normalise.

    Training adds horizontal and vertical flips only. Overhead imagery has no canonical up, so both
    flips are label-preserving; colour jitter is deliberately omitted because the frozen backbone's
    features are the only signal a linear probe has, and perturbing colour would blur the very
    spectral cues that separate Forest from Pasture.
    """
    from torchvision import transforms

    steps: list[Any] = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def load_image(path: Path | str) -> Any:
    """Open one image as RGB, raising a clear error for a missing or unreadable file."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image not found: {path}") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Not a readable image: {path} ({exc})") from exc


def build_dataset(split: str, subset_dir: Path | str = DEFAULT_SUBSET_DIR) -> Any:
    """A ``torch.utils.data.Dataset`` over one prepared split.

    Defined inside the function so importing this module does not import torch — the backend reads
    :data:`EUROSAT_CLASSES` from here and must not pay a torch import to do it.
    """
    from torch.utils.data import Dataset

    samples = load_split(split, subset_dir)
    transform = build_transform(train=(split == "train"))

    class EuroSatSubset(Dataset):  # type: ignore[misc]
        """Prepared EuroSAT subset split; returns ``(3, 224, 224)`` tensors and int labels."""

        def __init__(self) -> None:
            self.samples = samples
            self.classes = list(EUROSAT_CLASSES)
            self.split = split

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            sample = self.samples[index]
            return transform(load_image(sample.path)), sample.label

    return EuroSatSubset()


__all__ = [
    "DEFAULT_RAW_ROOT",
    "DEFAULT_SUBSET_DIR",
    "EUROSAT_CLASSES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INPUT_SIZE",
    "DatasetNotPreparedError",
    "SubsetSample",
    "build_dataset",
    "build_subset",
    "build_transform",
    "download_eurosat",
    "load_image",
    "load_manifest",
    "load_split",
    "scan_class_folders",
]
