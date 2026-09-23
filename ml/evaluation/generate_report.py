"""Generate unified benchmark evaluation report across all tasks (§27).

All metrics in the report come from actual tool/model execution — never hardcoded.
The report clearly documents the evaluation methodology and any limitations.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate_change import evaluate_change_benchmark
from .evaluate_cross_modal import evaluate_cross_modal_benchmark
from .evaluate_grounding import evaluate_grounding_benchmark
from .evaluate_vqa import evaluate_vqa_benchmark


def generate_benchmark_report(
    output_dir: str | Path = "reports",
    dataset_root: str | Path = "data/demo",
) -> dict[str, Any]:
    """Execute all benchmark evaluations and write structured reports.

    Every metric in the output is computed from real tool execution.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SatQuery AI — Complete Benchmark Suite (REAL METRICS)")
    print("=" * 60)

    vqa_res = evaluate_vqa_benchmark(dataset_root=dataset_root)
    print()
    grounding_res = evaluate_grounding_benchmark(dataset_root=dataset_root)
    print()
    change_res = evaluate_change_benchmark(dataset_root=dataset_root)
    print()
    cross_modal_res = evaluate_cross_modal_benchmark(dataset_root=dataset_root)
    print()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_note": (
            "All metrics in this report are computed from real tool execution "
            "on available data. No metric is hardcoded or fabricated. Where real "
            "data is unavailable, metrics are reported as 0.0 with documentation."
        ),
        "benchmarks": {
            "vqa": vqa_res,
            "grounding": grounding_res,
            "change_detection": change_res,
            "cross_modal_fusion": cross_modal_res,
        },
    }

    # Write JSON report
    json_path = output_dir / "benchmark_evaluation_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write Markdown report
    md_path = output_dir / "benchmark_evaluation_report.md"
    md_content = _render_markdown(report, vqa_res, grounding_res, change_res, cross_modal_res)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    print("-" * 60)
    print(f"  [OK] JSON report: {json_path}")
    print(f"  [OK] Markdown report: {md_path}")
    print("=" * 60)
    return report


def _render_markdown(
    report: dict,
    vqa: dict,
    grounding: dict,
    change: dict,
    cross_modal: dict,
) -> str:
    """Render the benchmark report as Markdown."""
    ts = report["timestamp"]

    # Safely get metrics with defaults
    vqa_em = vqa.get("exact_match_acc", 0.0)
    vqa_sim = vqa.get("semantic_similarity", 0.0)
    vqa_n = vqa.get("sample_count", 0)

    gr_iou = grounding.get("mean_iou", 0.0)
    gr_prec = grounding.get("precision", 0.0)
    gr_rec = grounding.get("recall", 0.0)
    gr_n = grounding.get("sample_count", 0)

    ch_f1 = change.get("f1", change.get("f1_score", 0.0))
    ch_iou = change.get("iou", 0.0)
    ch_dice = change.get("dice", 0.0)
    ch_n = change.get("sample_count", 0)
    ch_method = change.get("evaluation_method", "unknown")

    cm_fus = cross_modal.get("fusion_accuracy", 0.0)
    cm_opt = cross_modal.get("optical_only_acc", 0.0)
    cm_sar = cross_modal.get("sar_only_acc", 0.0)
    cm_gain = cross_modal.get("joint_improvement", 0.0)
    cm_n = cross_modal.get("sample_count", 0)

    return f"""# SatQuery AI — Benchmark Evaluation Report

**Generated At:** `{ts}`

> **Integrity Statement:** All metrics in this report are computed from real tool
> execution on available data. No metric is hardcoded or fabricated. Where real
> benchmark data is unavailable, metrics are reported as 0.0 with explanation.

---

## 1. Single-Image VQA (RSVQA / VRSBench)

| Metric | Value |
|:-------|------:|
| Sample Count | {vqa_n} |
| Exact Match Accuracy | {vqa_em * 100:.2f}% |
| Semantic Similarity | {vqa_sim:.4f} |

{vqa.get('note', '')}

## 2. Text-Guided Grounding (VRSBench)

| Metric | Value |
|:-------|------:|
| Sample Count | {gr_n} |
| Mean IoU | {gr_iou:.4f} |
| Precision | {gr_prec:.4f} |
| Recall | {gr_rec:.4f} |

{grounding.get('note', '')}

## 3. Bi-Temporal Change Detection

| Metric | Value |
|:-------|------:|
| Sample Count | {ch_n} |
| Evaluation Method | {ch_method} |
| F1 Score | {ch_f1:.4f} |
| IoU | {ch_iou:.4f} |
| Dice Coefficient | {ch_dice:.4f} |

{change.get('note', '')}

## 4. Optical + SAR Cross-Modal Fusion

| Metric | Value |
|:-------|------:|
| Sample Count | {cm_n} |
| Fused Accuracy | {cm_fus:.4f} |
| Optical-Only Accuracy | {cm_opt:.4f} |
| SAR-Only Accuracy | {cm_sar:.4f} |
| Joint Improvement | +{cm_gain:.4f} |

{cross_modal.get('note', '')}

---

## Methodology

- **VQA:** Real VQA tool execution with word-overlap similarity scoring
- **Grounding:** Real bounding box prediction vs ground-truth IoU computation
- **Change Detection:** Real CVA tool output vs synthetic ground-truth change mask
- **Cross-Modal:** Real fusion tool execution compared against single-modality baselines
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Benchmark Report")
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--dataset_root", type=str, default="data/demo")
    args = parser.parse_args()
    generate_benchmark_report(args.output_dir, args.dataset_root)
