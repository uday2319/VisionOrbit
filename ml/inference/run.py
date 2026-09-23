"""Unified inference runner script for running SatQuery AI ML predictions offline or programmatically."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from backend.app.agents.classifier import classify
from backend.app.agents.registry import default_registry
from backend.app.agents.router import route
from backend.app.agents.tools.base import ToolContext
from backend.app.core.types import QueryTask
from backend.app.geospatial.raster import load_raster


def run_inference(
    image_path: str | Path,
    query: str,
    second_image_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Run model inference on single or paired imagery with a natural language query."""
    img1_path = Path(image_path)
    if not img1_path.exists():
        raise FileNotFoundError(f"Input image file '{img1_path}' not found.")

    rasters = [load_raster(img1_path)]
    if second_image_path:
        img2_path = Path(second_image_path)
        if not img2_path.exists():
            raise FileNotFoundError(f"Second image file '{img2_path}' not found.")
        rasters.append(load_raster(img2_path))

    # Intent classification & routing
    cls = classify(query)
    decision = route(query, rasters, classification=cls)
    registry = default_registry()
    ctx = ToolContext(query=query, rasters=rasters)

    if decision.needs_clarification or decision.task is QueryTask.UNKNOWN:
        return {
            "query": query,
            "task": decision.task.value if decision.task else "unknown",
            "answer": f"Could not determine task: {decision.reason}",
            "confidence": 0.0,
            "evidence": ["Query did not align with any supported remote-sensing workflow."],
        }

    tool_result = registry.dispatch(decision.task, ctx)

    return {
        "query": query,
        "task": decision.task.value,
        "selected_tool": tool_result.tool_name,
        "answer": tool_result.answer,
        "confidence": tool_result.confidence.score,
        "confidence_level": tool_result.confidence.level.value,
        "evidence": tool_result.evidence,
        "warnings": tool_result.warnings,
        "reasoning": decision.reason,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery AI Inference Runner")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--second_image", type=str, default=None, help="Optional second image for bi-temporal or optical-SAR pair")
    args = parser.parse_args()

    try:
        res = run_inference(args.image, args.query, args.second_image)
        print("==================================================")
        print(f"Task: {res['task'].upper()} (Tool: {res.get('selected_tool')})")
        print(f"Query: '{res['query']}'")
        print(f"Answer: {res['answer']}")
        print(f"Confidence: {res['confidence_level'].upper()} ({res['confidence']:.2f})")
        print(f"Evidence: {res['evidence']}")
        print("==================================================")
    except Exception as e:
        print(f"[Error] Inference failed: {e}", file=sys.stderr)
        sys.exit(1)

