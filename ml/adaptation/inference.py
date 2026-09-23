"""Load the trained checkpoint in a separate process and predict on unseen images.

Deliberately independent of :mod:`ml.adaptation.train`: it imports nothing from the training module
and rebuilds the architecture from the checkpoint's own recorded fields. That is what makes running
this script a real verification that the saved weights load and predict, rather than a continuation
of the training process that produced them.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..datasets.eurosat_adapter import (
    DEFAULT_SUBSET_DIR,
    build_transform,
    load_image,
    load_split,
)

DEFAULT_CHECKPOINT_DIR = Path("ml/checkpoints/satquery-rs-visual-v1")
WEIGHTS_FILENAME = "model.pt"


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the adapted checkpoint has not been trained yet."""


@dataclass(frozen=True)
class ScenePrediction:
    """One image's predicted land-use class and the full probability vector behind it."""

    path: str
    label: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
        }


class AdaptedModelInference:
    """The adapted EuroSAT scene classifier, loaded from disk and ready to predict.

    Construction loads the weights eagerly so a broken or missing checkpoint fails here rather than
    inside a request. :meth:`predict` accepts a path or an already-opened PIL image.
    """

    def __init__(self, checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR) -> None:
        import torch

        self.checkpoint_dir = Path(checkpoint_dir)
        weights_path = self.checkpoint_dir / WEIGHTS_FILENAME
        if not weights_path.is_file():
            raise CheckpointNotFoundError(
                f"No adapted checkpoint at {weights_path}. Run "
                "`python -m ml.adaptation.train` to produce one."
            )
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.checkpoint_path = weights_path
        self.model_id: str = payload.get("model_id", "unknown")
        self.class_names: list[str] = list(payload["class_names"])
        self.architecture: str = payload.get("architecture", "resnet18")
        self.pretrained_weights: str = payload.get("pretrained_weights", "unknown")
        self.adaptation_method: str = payload.get("adaptation_method", "unknown")
        self.trained_epoch: int | None = payload.get("epoch")
        self._model = self._build(len(self.class_names))
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        self._transform = build_transform(train=False)

    @staticmethod
    def _build(num_classes: int) -> Any:
        """Rebuild the architecture the checkpoint was trained with.

        ``weights=None`` here on purpose: the pretrained values are already inside the checkpoint's
        state dict, so downloading ImageNet weights just to overwrite them would be wasted I/O.
        """
        import torch.nn as nn
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def predict(self, image: Path | str | Any) -> ScenePrediction:
        """Classify one image. ``image`` may be a path or a PIL image."""
        import torch

        source = str(image) if isinstance(image, (str, Path)) else "<in-memory image>"
        pil = load_image(image) if isinstance(image, (str, Path)) else image.convert("RGB")
        tensor = self._transform(pil).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)[0]
        label = int(probs.argmax().item())
        return ScenePrediction(
            path=source,
            label=label,
            class_name=self.class_names[label],
            confidence=float(probs[label].item()),
            probabilities={
                name: float(probs[i].item()) for i, name in enumerate(self.class_names)
            },
        )

    def describe(self) -> dict[str, Any]:
        """What ran, for a trace or a report — every field read from the checkpoint itself."""
        return {
            "model_id": self.model_id,
            "checkpoint": self.checkpoint_path.as_posix(),
            "architecture": self.architecture,
            "pretrained_weights": self.pretrained_weights,
            "adaptation_method": self.adaptation_method,
            "trained_epoch": self.trained_epoch,
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names),
        }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load the adapted checkpoint and predict on held-out EuroSAT test images."
    )
    parser.add_argument("--checkpoint_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--subset_dir", type=str, default=str(DEFAULT_SUBSET_DIR))
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument(
        "--image", type=str, default=None,
        help="Classify this single image instead of sampling the held-out test split.",
    )
    args = parser.parse_args(argv)

    engine = AdaptedModelInference(args.checkpoint_dir)
    print("=" * 68)
    print("  SatQuery AI - adapted remote-sensing visual model, inference check")
    print("=" * 68)
    print(json.dumps(engine.describe(), indent=2))
    print("-" * 68)

    if args.image:
        prediction = engine.predict(args.image)
        print(f"  {prediction.class_name}  ({prediction.confidence * 100:.1f}%)  "
              f"<- {prediction.path}")
        print("=" * 68)
        return 0

    # These come from the recorded test split, so by construction none were seen during training.
    samples = load_split("test", args.subset_dir)[: max(args.num_samples, 0)]
    correct = 0
    for sample in samples:
        prediction = engine.predict(sample.path)
        hit = prediction.label == sample.label
        correct += int(hit)
        print(f"  {'OK ' if hit else 'MISS'}  truth={sample.class_name:<21}"
              f" predicted={prediction.class_name:<21} p={prediction.confidence * 100:5.1f}%")
    print("-" * 68)
    print(f"  {correct}/{len(samples)} correct on this unseen sample "
          "(not the full evaluation; run `python -m ml.adaptation.evaluate` for that)")
    print("=" * 68)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(_main())
