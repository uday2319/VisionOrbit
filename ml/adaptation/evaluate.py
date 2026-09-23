"""Remote-sensing adaptation evaluation: measure the adapted model on the held-out test split.

This is *not* the SIH benchmark. No prescribed benchmark was run, so nothing here is labelled as
one. What it reports is exactly what it measures: top-1 accuracy, per-class accuracy and a
confusion matrix over the EuroSAT test images recorded in the subset manifest — images drawn from
disjoint slices of each class's permutation, so none of them appear in the training or validation
splits.

The confusion matrix is computed with numpy rather than scikit-learn, which is not a dependency of
this project.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..datasets.eurosat_adapter import DEFAULT_SUBSET_DIR, load_manifest, load_split
from .inference import DEFAULT_CHECKPOINT_DIR, AdaptedModelInference

EVALUATION_FILENAME = "evaluation.json"
METADATA_FILENAME = "metadata.json"

#: The label the brief requires. Kept as a constant so no caller can rename it in passing.
EVALUATION_LABEL = "Remote-sensing adaptation evaluation"


def confusion_matrix(truth: list[int], predicted: list[int], num_classes: int) -> list[list[int]]:
    """Rows are true classes, columns predicted — ``matrix[i][j]`` counts true ``i`` called ``j``."""
    import numpy as np

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(truth, predicted, strict=True):
        matrix[t, p] += 1
    return matrix.tolist()


def evaluate_adapted_model(
    *,
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    subset_dir: Path | str = DEFAULT_SUBSET_DIR,
    split: str = "test",
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the adapted model over the held-out split and write ``evaluation.json``.

    Also fills in ``test_accuracy`` / ``test_evaluation`` in the checkpoint's ``metadata.json``,
    which :mod:`ml.adaptation.train` deliberately leaves ``null`` — so that field is only ever
    populated by a measurement.

    Returns the evaluation report.
    """
    checkpoint_dir = Path(checkpoint_dir)
    engine = AdaptedModelInference(checkpoint_dir)
    samples = load_split(split, subset_dir)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError(f"The {split!r} split is empty; nothing to evaluate.")

    class_names = engine.class_names
    # Dataset provenance, read from the subset manifest rather than restated here, so the report
    # names the data that actually produced these images.
    manifest = load_manifest(subset_dir)
    truth: list[int] = []
    predicted: list[int] = []
    for sample in samples:
        prediction = engine.predict(sample.path)
        truth.append(sample.label)
        predicted.append(prediction.label)

    correct = sum(1 for t, p in zip(truth, predicted, strict=True) if t == p)
    matrix = confusion_matrix(truth, predicted, len(class_names))
    per_class = {}
    for i, name in enumerate(class_names):
        support = sum(matrix[i])
        per_class[name] = {
            "support": support,
            "correct": matrix[i][i],
            "accuracy": round(matrix[i][i] / support, 6) if support else None,
        }

    report: dict[str, Any] = {
        "evaluation": EVALUATION_LABEL,
        "is_sih_benchmark": False,
        "note": (
            "Top-1 accuracy of the adapted EuroSAT scene classifier on its own held-out test "
            "split. This is not the SIH prescribed benchmark, which was not run."
        ),
        "model_id": engine.model_id,
        "checkpoint": engine.checkpoint_path.as_posix(),
        "base_model": engine.architecture,
        "pretrained_weights": engine.pretrained_weights,
        "adaptation_method": engine.adaptation_method,
        "dataset": manifest["dataset"],
        "dataset_source": manifest["source_url"],
        "dataset_is_synthetic": manifest["is_synthetic"],
        "split": split,
        "num_test_samples": len(samples),
        "num_correct": correct,
        "test_accuracy": round(correct / len(samples), 6),
        "class_names": list(class_names),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_orientation": "rows = true class, columns = predicted class",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    (checkpoint_dir / EVALUATION_FILENAME).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    metadata_path = checkpoint_dir / METADATA_FILENAME
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["test_accuracy"] = report["test_accuracy"]
        metadata["test_evaluation"] = {
            "label": EVALUATION_LABEL,
            "num_test_samples": report["num_test_samples"],
            "num_correct": correct,
            "timestamp": report["timestamp"],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return report


def _print_report(report: dict[str, Any]) -> None:
    names = report["class_names"]
    width = max(len(n) for n in names)
    print("=" * 68)
    print(f"  {report['evaluation']}")
    print("=" * 68)
    print(f"  model:       {report['model_id']}")
    print(f"  checkpoint:  {report['checkpoint']}")
    print(f"  method:      {report['adaptation_method']}")
    print(f"  split:       {report['split']} ({report['num_test_samples']} unseen images)")
    print(f"  top-1 accuracy: {report['test_accuracy'] * 100:.2f}% "
          f"({report['num_correct']}/{report['num_test_samples']})")
    print("-" * 68)
    print("  per-class accuracy:")
    for name in names:
        stats = report["per_class"][name]
        acc = "n/a" if stats["accuracy"] is None else f"{stats['accuracy'] * 100:5.1f}%"
        print(f"    {name:<{width}}  {acc}  ({stats['correct']}/{stats['support']})")
    print("-" * 68)
    print("  confusion matrix (rows = true, columns = predicted):")
    print("    " + " ".join(f"{i:>4}" for i in range(len(names))))
    for i, row in enumerate(report["confusion_matrix"]):
        print(f"  {i:>2}" + " ".join(f"{v:>4}" for v in row) + f"   {names[i]}")
    print("=" * 68)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=EVALUATION_LABEL)
    parser.add_argument("--checkpoint_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--subset_dir", type=str, default=str(DEFAULT_SUBSET_DIR))
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    report = evaluate_adapted_model(
        checkpoint_dir=args.checkpoint_dir,
        subset_dir=args.subset_dir,
        split=args.split,
        limit=args.limit,
    )
    _print_report(report)
    print(f"  written to {Path(args.checkpoint_dir) / EVALUATION_FILENAME}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(_main())
