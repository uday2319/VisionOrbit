"""Adapt a real ImageNet-pretrained ResNet-18 to EuroSAT land use by linear probing.

What this does, precisely, so the naming stays honest:

* The backbone is ``torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`` — the
  actual published pretrained weights, downloaded by torchvision, not a look-alike CNN.
* Every backbone parameter is frozen (``requires_grad = False``) and the backbone is held in
  ``eval()`` mode throughout, so its BatchNorm running statistics stay at their ImageNet values and
  the only thing gradient descent can change is the head.
* ``model.fc`` is replaced by a fresh ``nn.Linear(512, 10)``. That single layer — 5,130 parameters —
  is what is trained. This is a **linear probe**: the standard, minimal, genuinely
  parameter-efficient adaptation. It is deliberately *not* called LoRA, because no low-rank
  decomposition of any pretrained weight matrix is involved.

Every number the run reports (losses, accuracies, parameter counts, wall-clock) is measured during
the run and written to ``metadata.json`` beside the weights. Nothing is asserted in advance.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..datasets.eurosat_adapter import (
    DEFAULT_SUBSET_DIR,
    EUROSAT_CLASSES,
    INPUT_SIZE,
    build_dataset,
    load_manifest,
)

#: Where the trained checkpoint lands. One directory per model version, as the brief asks.
DEFAULT_CHECKPOINT_DIR = Path("ml/checkpoints/satquery-rs-visual-v1")

#: The identifier the backend registry and the trace report for this component.
MODEL_ID = "satquery-rs-visual-v1"
BASE_MODEL = "torchvision.models.resnet18"
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
ADAPTATION_METHOD = "linear probe (frozen ImageNet backbone, retrained 10-class fc layer)"
WEIGHTS_FILENAME = "model.pt"
METADATA_FILENAME = "metadata.json"


def seed_everything(seed: int) -> None:
    """Fix the RNGs that affect the run, so a rerun reproduces these numbers."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int = len(EUROSAT_CLASSES)) -> Any:
    """The real pretrained ResNet-18 with a frozen backbone and a fresh classification head.

    Returns the model. The caller trains ``model.fc`` only; :func:`freeze_backbone` has already
    cleared ``requires_grad`` everywhere else.
    """
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Linear(in_features, num_classes)
    freeze_backbone(model)
    return model


def freeze_backbone(model: Any) -> None:
    """Disable gradients everywhere except the classification head."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def set_backbone_eval(model: Any) -> None:
    """Train the head but keep the frozen backbone in eval mode.

    Without this, ``model.train()`` would let the backbone's BatchNorm layers update their running
    mean/variance from EuroSAT batches. Those buffers are not parameters, so freezing
    ``requires_grad`` does not protect them — and drifting them would change the "frozen" features
    between training and evaluation.
    """
    model.train()
    for module in model.modules():
        if module.__class__.__name__.startswith("BatchNorm"):
            module.eval()


def count_parameters(model: Any) -> tuple[int, int]:
    """``(trainable, total)`` parameter counts, measured from the model itself."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _run_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any, device: Any,
               *, train: bool) -> tuple[float, float]:
    """One pass over ``loader``; returns ``(mean_loss, accuracy)``.

    Loss is averaged per sample rather than per batch so an uneven final batch cannot skew it.
    """
    import torch

    if train:
        set_backbone_eval(model)
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    seen = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)
    if seen == 0:
        raise RuntimeError("Empty data loader; the prepared subset appears to have no samples.")
    return running_loss / seen, correct / seen


def train_adapter(
    *,
    subset_dir: Path | str = DEFAULT_SUBSET_DIR,
    output_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 1337,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Run the adaptation and write ``model.pt`` + ``metadata.json`` to ``output_dir``.

    The checkpoint kept is the epoch with the lowest validation loss, not the last epoch, so a run
    that starts to overfit the 2000-image subset does not silently ship the worse weights.

    Returns the metadata dict that was written.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    subset_dir = Path(subset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(subset_dir)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = build_dataset("train", subset_dir)
    val_ds = build_dataset("val", subset_dir)
    test_ds = build_dataset("test", subset_dir)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(len(EUROSAT_CLASSES)).to(device)
    trainable, total = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )

    print("=" * 68)
    print("  SatQuery AI - remote-sensing visual adaptation (real training run)")
    print("=" * 68)
    print(f"  Dataset:      {manifest['dataset']}  (synthetic: {manifest['is_synthetic']})")
    print(f"  Splits:       {manifest['splits']}")
    print(f"  Base model:   {BASE_MODEL}  weights={PRETRAINED_WEIGHTS}")
    print(f"  Method:       {ADAPTATION_METHOD}")
    print(f"  Parameters:   {trainable:,} trainable / {total:,} total")
    print(f"  Device:       {device}   epochs={epochs} batch={batch_size} lr={lr} seed={seed}")
    print("-" * 68)

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "seconds": round(time.perf_counter() - epoch_start, 2),
        }
        history.append(entry)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            # Copy to CPU so the checkpoint loads on a machine without the training device.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"  epoch {epoch}/{epochs}  train_loss={train_loss:.4f} acc={train_acc * 100:.1f}%"
            f"  |  val_loss={val_loss:.4f} acc={val_acc * 100:.1f}%"
            f"  ({entry['seconds']}s){'  <- best' if improved else ''}"
        )

    elapsed = time.perf_counter() - started
    if best_state is None:  # pragma: no cover - epochs >= 1 always populates this
        raise RuntimeError("Training produced no checkpointable state; epochs must be >= 1.")

    weights_path = output_dir / WEIGHTS_FILENAME
    torch.save(
        {
            "model_id": MODEL_ID,
            "architecture": "resnet18",
            "pretrained_weights": PRETRAINED_WEIGHTS,
            "adaptation_method": ADAPTATION_METHOD,
            "class_names": list(EUROSAT_CLASSES),
            "input_size": INPUT_SIZE,
            "normalization": manifest["normalization"],
            "epoch": best_epoch,
            "state_dict": best_state,
        },
        weights_path,
    )

    final = history[-1]
    best = history[best_epoch - 1]
    metadata: dict[str, Any] = {
        "model_id": MODEL_ID,
        "dataset": manifest["dataset"],
        "dataset_source": manifest["source_url"],
        "dataset_citation": manifest["citation"],
        "dataset_licence": manifest["licence"],
        "dataset_is_synthetic": manifest["is_synthetic"],
        "dataset_seed": manifest["seed"],
        "num_train_images": manifest["splits"]["train"],
        "num_val_images": manifest["splits"]["val"],
        "num_test_images": manifest["splits"]["test"],
        "num_classes": len(EUROSAT_CLASSES),
        "class_names": list(EUROSAT_CLASSES),
        "base_model": BASE_MODEL,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "adaptation_method": ADAPTATION_METHOD,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "loss_function": "CrossEntropyLoss",
        "input_size": INPUT_SIZE,
        "normalization": manifest["normalization"],
        "seed": seed,
        "device": str(device),
        "train_loss": final["train_loss"],
        "val_loss": final["val_loss"],
        "best_epoch": best_epoch,
        "best_val_loss": best["val_loss"],
        "best_val_accuracy": best["val_accuracy"],
        "history": history,
        "training_seconds": round(elapsed, 2),
        "training_completed": True,
        "checkpoint_file": WEIGHTS_FILENAME,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        # Populated by ml.adaptation.evaluate, which measures it on the held-out split. It is
        # absent — not zero, not guessed — until that script has actually run.
        "test_accuracy": None,
        "test_evaluation": None,
    }
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("-" * 68)
    print(f"  trained {trainable:,} parameters in {elapsed:.1f}s")
    print(f"  best epoch {best_epoch}: val_loss={best['val_loss']:.4f} "
          f"val_accuracy={best['val_accuracy'] * 100:.2f}%")
    print(f"  weights   -> {weights_path}")
    print(f"  metadata  -> {output_dir / METADATA_FILENAME}")
    print(f"  held-out test split ({len(test_ds)} images) is untouched; run "
          "`python -m ml.adaptation.evaluate` to measure test accuracy.")
    print("=" * 68)
    return metadata


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adapt a pretrained ResNet-18 to EuroSAT by linear probing."
    )
    parser.add_argument("--subset_dir", type=str, default=str(DEFAULT_SUBSET_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args(argv)

    train_adapter(
        subset_dir=args.subset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(_main())
