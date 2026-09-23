"""API routes for running multimodal remote sensing analysis, traces, reports, and history.

The pipeline this module drives is, in order (§5, §6, §38):

    load rasters -> **normalise the input mode** -> classify -> route -> **check the tool contract**
    -> execute -> assess evidence -> persist -> respond

The two bolded steps are the ones that used to be missing. The mode was derived *after*
classification and then thrown away, and the only thing that checked whether a tool could accept
the inputs was the tool itself, *during* execution — so "two images sent to a single-image
analysis" came back as HTTP 200 with zero confidence, blaming the imagery for what was an
application-level contract error. Now the contract is checked before any analysis starts and the
failure surfaces as ``TOOL_INPUT_CONTRACT_ERROR`` with 422.
"""
from __future__ import annotations

import html
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ...agents.classifier import classify
from ...agents.modes import normalize_request
from ...agents.registry import default_registry
from ...agents.router import route_request
from ...agents.tools.base import ToolContext, ToolResult
from ...config import get_settings
from ...core.errors import AppError, ErrorCode, NotFoundError, UnsupportedTaskError
from ...core.logging import get_logger
from ...core.types import (
    AnalysisStatus,
    ConfidenceLevel,
    EvidenceType,
    Modality,
    QueryTask,
    StepStatus,
)
from ...db import (
    get_analysis_record,
    get_db,
    get_image_record,
    get_next_analysis_id,
    list_analysis_records,
    save_analysis_record,
)
from ...geospatial import artifacts as artifact_renderer
from ...geospatial.raster import RasterData, load_raster
from ...schemas import (
    TASK_OVERRIDE_AUTO,
    AnalyzeRequest,
    AnalyzeResponse,
    ConfidenceSchema,
    EvidenceItem,
    ExecutionTraceSchema,
    ExecutionTraceStep,
    HealthResponse,
    HistoryResponse,
    ModelInfo,
    ReportResponse,
)
from ...services.confidence import FactorKind

logger = get_logger(__name__)
router = APIRouter(tags=["analysis"])
settings = get_settings()
registry = default_registry()

# Repo-root-relative, resolved at import time. The old code used a CWD-relative path, so the same
# request found a different file depending on where uvicorn was started from.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EVALUATION_REPORT = _REPO_ROOT / "reports" / "benchmark_evaluation_report.json"

#: Where rendered analysis imagery is written, and the URL prefix the ``/storage`` mount serves it
#: at. Both are derived from one setting so the file the backend writes and the URL the frontend
#: fetches cannot drift apart.
_ARTIFACT_DIRNAME = "artifacts"


def _render_artifacts(
    analysis_id: str, result: ToolResult, rasters: list[RasterData]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Render this run's base composite and task overlay, and describe them for the response.

    The PNGs go under ``storage_root/artifacts/<analysis_id>/`` and are served by the existing
    ``/storage`` mount. Until this existed, no tool emitted any imagery: the result page's Imagery
    panel showed "No rendered imagery for this analysis." for every live run, while recorded demo
    cases rendered fully — the renderer lived in the fixture-generation script alone.
    """
    return artifact_renderer.render_artifacts(
        result.raw,
        rasters,
        out_dir=settings.storage_root / _ARTIFACT_DIRNAME / analysis_id,
        url_prefix=f"/storage/{_ARTIFACT_DIRNAME}/{analysis_id}",
    )



def _recorded_modality(raw: str | None, img_id: str) -> Modality:
    """The modality persisted for an uploaded image, or ``UNKNOWN`` to re-infer it.

    An unrecognised stored value is *not* an error worth failing the analysis over — the row may
    predate a rename — so it degrades to ``UNKNOWN``, which means "infer from the raster", the same
    behaviour as before the modality was carried across at all. It is logged, because a row that
    cannot be read back is a schema problem the operator should see.
    """
    if raw is None or not raw.strip():
        return Modality.UNKNOWN
    try:
        return Modality(raw.strip().lower())
    except ValueError:
        logger.warning(
            "uploaded image has an unrecognised stored modality; re-inferring from the raster",
            extra={"image_id": img_id, "stored_modality": raw},
        )
        return Modality.UNKNOWN


def _load_rasters_for_ids(db: Session, image_ids: list[str]) -> list[RasterData]:
    """Re-open each uploaded raster for analysis, preserving the modality settled at upload time.

    The modality is read back from the image record rather than re-inferred here. That is the
    difference between an optical+SAR pair working and not:
    :func:`app.geospatial.raster.infer_modality` can only read filename tokens and band
    descriptions, so a genuine SAR scene exported as ``subset_1.tif`` with an unlabelled band infers
    as ``UNKNOWN`` — however clearly the client declared it via ``modality_hint`` on upload. Loading
    with no ``modality=`` argument threw that declaration away, the pair then normalised to
    ``bitemporal_pair`` instead of ``optical_sar_pair``, and change detection refused it with "no
    bands in common" instead of running the fusion the user asked for.
    """
    rasters: list[RasterData] = []
    for img_id in image_ids:
        rec = get_image_record(db, img_id)
        if not rec:
            raise NotFoundError(f"Uploaded image '{img_id}' not found.", code=ErrorCode.NOT_FOUND)
        file_path = Path(rec.file_path)
        if not file_path.exists():
            raise NotFoundError(
                f"Image file for '{img_id}' is missing from storage.", code=ErrorCode.NOT_FOUND
            )
        rasters.append(
            load_raster(
                file_path,
                max_edge=settings.max_analysis_edge,
                max_pixels=settings.max_pixels,
                modality=_recorded_modality(rec.modality, img_id),
            )
        )
    return rasters


def _grid_description(rasters: list[RasterData]) -> str:
    """How to describe the grid the inputs sit on — read from metadata, never assumed.

    The former text hardcoded a fallback of "pixel" via ``getattr(raster, 'crs', 'pixel')`` on an
    attribute ``RasterData`` does not have, so *every* trace said "pixel grid" — including runs
    that reported an area in km², which is only possible with a georeference.
    """
    first = rasters[0].metadata
    if not first.is_georeferenced:
        return "pixel grid (no georeference)"
    if first.crs_epsg is not None:
        return f"EPSG:{first.crs_epsg} grid"
    return "projected grid (CRS present, EPSG code unresolved)"


def _resolve_task(routed: QueryTask, task_override: str | None) -> tuple[QueryTask, str | None]:
    """Apply an explicit task override, reporting it rather than hiding it.

    An override still goes through the contract gate downstream, so pinning a task cannot smuggle
    an unrunnable input configuration past validation — it only narrows the choice the router would
    otherwise have made. The value has already been validated by
    :class:`~app.schemas.analysis.AnalyzeRequest`, so an unknown string never reaches here.
    """
    if not task_override or task_override == TASK_OVERRIDE_AUTO:
        return routed, None
    pinned = QueryTask(task_override)
    if pinned is routed:
        return routed, None
    return pinned, (
        f"The request pinned '{pinned.value}', overriding the routed task '{routed.value}'."
    )


def _evidence_items(result: ToolResult) -> list[EvidenceItem]:
    """Turn a tool's evidence lines into structured §15 items.

    Each line keeps the confidence of the *analysis that produced it* and names the tool as its
    source. It does not receive the overall headline score, which is what made every row look
    equally strong regardless of what it said.
    """
    items = [
        EvidenceItem(
            evidence_type=EvidenceType.OBSERVATION,
            source=result.tool_name,
            confidence=None,
            description=line,
        )
        for line in result.evidence
    ]
    # The confidence report's own factors are the measured part of the evidence, so they are
    # published as measurements with their real values rather than summarised in prose. A gate
    # factor is a quality constraint, not a finding, and is labelled as such.
    for factor in result.confidence.factors:
        items.append(
            EvidenceItem(
                evidence_type=(
                    EvidenceType.QUALITY
                    if factor.kind is FactorKind.GATE
                    else EvidenceType.MEASUREMENT
                ),
                source=f"{result.tool_name}:{factor.name}",
                confidence=float(factor.value),
                description=factor.reason,
                value=float(factor.value),
                unit="fraction",
            )
        )
    return items


def _terminal_status(confidence_level: ConfidenceLevel, warnings: list[str]) -> AnalysisStatus:
    """The run's terminal state.

    ``FAILED`` is reserved for a run that did not produce an analysis at all — it is set by the
    failure paths, not derived here. A completed analysis whose evidence was too weak to support a
    claim is ``PARTIAL``: it ran correctly, and saying otherwise would blur the very distinction
    §27 exists to preserve.
    """
    if confidence_level is ConfidenceLevel.INSUFFICIENT:
        return AnalysisStatus.PARTIAL
    if warnings:
        return AnalysisStatus.PARTIAL
    return AnalysisStatus.SUCCESS


class _Timer:
    """Measured wall-clock durations for the trace (§51).

    Real elapsed milliseconds, with no floor applied. The previous code wrapped every measurement
    in ``max(15, …)`` / ``max(20, …)`` / ``max(50, …)``, which invented a minimum duration for
    steps that genuinely completed faster — and then discarded the values without using them.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._marks: dict[str, int] = {}
        self._last = self._start

    def mark(self, name: str) -> int:
        now = time.perf_counter()
        elapsed = int(round((now - self._last) * 1000))
        self._marks[name] = elapsed
        self._last = now
        return elapsed

    def get(self, name: str) -> int | None:
        return self._marks.get(name)

    @property
    def total_ms(self) -> int:
        return int(round((time.perf_counter() - self._start) * 1000))


def _run_agentic_workflow(
    db: Session,
    query: str,
    image_ids: list[str],
    task_override: str | None = None,
) -> dict[str, Any]:
    """Run one analysis end to end and return the response body.

    Raises:
        AppError: for every expected failure — a missing image, an incompatible pair, an input
            configuration the task cannot serve, an unimplemented task. These become the §29 error
            envelope with a distinct code and a 4xx/5xx status. They are deliberately *not* caught
            and rendered as a zero-confidence answer: a request that could not be run has no
            analytical result, and reporting one would misattribute an application error to the
            imagery.
    """
    timer = _Timer()
    analysis_id = get_next_analysis_id(db)

    rasters = _load_rasters_for_ids(db, image_ids)
    load_ms = timer.mark("ingest")

    # §5: the canonical mode is established from the actual rasters, before anything routes on it.
    normalized = normalize_request(query, rasters, image_ids=image_ids)

    classification = classify(query)
    decision = route_request(normalized, classification=classification)
    route_ms = timer.mark("route")

    selected_task, override_note = _resolve_task(decision.task, task_override)

    steps: list[ExecutionTraceStep] = [
        ExecutionTraceStep(
            step_index=1,
            step_name="Raster Ingestion & Geometry Validation",
            tool_name="RasterPreprocessor",
            status=StepStatus.OK,
            details=(
                f"Loaded {normalized.input_count} raster(s) "
                f"({', '.join(m.value for m in normalized.modalities)}) "
                f"on {_grid_description(rasters)}."
            ),
            duration_ms=load_ms,
            warnings=list(normalized.pair_warnings),
        ),
        ExecutionTraceStep(
            step_index=2,
            step_name="Input Mode Normalisation",
            tool_name="ModeNormalizer",
            status=StepStatus.OK,
            details=(
                f"Canonical input mode resolved to '{normalized.mode.value}' from the inputs "
                f"themselves (not from the query text)."
            ),
            duration_ms=None,
        ),
        ExecutionTraceStep(
            step_index=3,
            step_name="Query Classification & Routing",
            tool_name="AgentRouter",
            status=StepStatus.OK,
            details=decision.reason + (f" {override_note}" if override_note else ""),
            duration_ms=route_ms,
        ),
    ]
    warnings: list[str] = list(normalized.pair_warnings)

    if decision.needs_clarification or selected_task is QueryTask.UNKNOWN:
        # An unroutable request is a refusal, not an analysis. It is reported as such rather than
        # as an answer with zero confidence (§9, §28).
        raise UnsupportedTaskError(
            f"I could not determine a suitable analysis for this request. {decision.reason}",
            context={
                "analysis_id": analysis_id,
                "mode": normalized.mode.value,
                "reason": decision.reason,
            },
        )

    # §6/§38: the declared contract is checked here, before execution. A mismatch raises
    # ToolContractError (422, TOOL_INPUT_CONTRACT_ERROR) and nothing is analysed.
    accepted_tools = registry.check_contracts(selected_task, normalized)
    steps.append(
        ExecutionTraceStep(
            step_index=4,
            step_name="Tool Input Contract Validation",
            tool_name="ToolRegistry",
            status=StepStatus.OK,
            details=(
                f"Input configuration '{normalized.mode.value}' with {normalized.input_count} "
                f"image(s) satisfies the declared contract of: "
                f"{', '.join(t.name for t in accepted_tools)}."
            ),
            duration_ms=timer.mark("contract"),
        )
    )

    ctx = ToolContext(query=query, rasters=rasters)
    result = registry.dispatch(selected_task, ctx, request=normalized)
    execute_ms = timer.mark("execute")

    warnings.extend(result.warnings)
    steps.append(
        ExecutionTraceStep(
            step_index=5,
            step_name=f"Specialist Execution ({result.tool_name})",
            tool_name=result.tool_name,
            status=result.status,
            details=(
                f"Analysis executed by '{result.tool_name}' at the "
                f"'{result.tier.label}' rung of the fallback chain."
            ),
            duration_ms=execute_ms,
            warnings=list(result.warnings),
        )
    )

    # Rendered from the arrays this run produced, before the terminal status is settled: a render
    # failure adds a warning, and a warning is part of what makes a run PARTIAL.
    rendered_artifacts, render_warnings = _render_artifacts(analysis_id, result, rasters)
    warnings.extend(render_warnings)
    render_ms = timer.mark("render")

    confidence = result.confidence
    structured_evidence = _evidence_items(result)
    steps.append(
        ExecutionTraceStep(
            step_index=6,
            step_name="Evidence Grounding & Confidence Assessment",
            tool_name="ConfidenceEngine",
            status=StepStatus.OK if confidence.is_sufficient else StepStatus.WARNING,
            details=(
                f"Confidence {confidence.score:.2f} ({confidence.level.value}) from "
                f"{len(confidence.factors)} measured factor(s)"
                + (
                    f"; bounded by '{confidence.limiting_factor}'."
                    if confidence.limiting_factor
                    else "."
                )
            ),
            duration_ms=timer.mark("confidence"),
        )
    )

    status = _terminal_status(confidence.level, warnings)
    total_ms = timer.total_ms
    input_metas = [artifact_renderer.input_meta(r) for r in rasters]

    input_summary = {
        "image_ids": image_ids,
        "image_count": normalized.input_count,
        "modalities": [m.value for m in normalized.modalities],
        "dimensions": [f"{r.width}x{r.height}" for r in rasters],
        "canonical_mode": normalized.mode.value,
        "georeferenced": normalized.georeferenced,
        "normalisation": normalized.details,
        "step_durations_ms": {
            "ingest": load_ms,
            "route": route_ms,
            "execute": execute_ms,
            "render": render_ms,
        },
        "total_duration_ms": total_ms,
    }

    trace_obj = ExecutionTraceSchema(
        analysis_id=analysis_id,
        task=selected_task.value,
        source=decision.source.value,
        mode=normalized.mode,
        selected_tools=[t.name for t in accepted_tools],
        input_summary=input_summary,
        status=status,
        steps=steps,
        confidence_score=confidence.score,
        confidence_level=confidence.level,
        confidence=ConfidenceSchema.model_validate(confidence.to_dict()),
        evidence=structured_evidence,
        warnings=warnings,
        reasoning_summary=decision.reason + (f" {override_note}" if override_note else ""),
        total_duration_ms=total_ms,
        artifacts=rendered_artifacts,
        inputs=input_metas,
    )

    created_at = datetime.now(UTC).isoformat()

    save_analysis_record(
        db=db,
        analysis_id=analysis_id,
        query=query,
        mode=normalized.mode.value,
        task=selected_task.value,
        answer=result.answer,
        confidence_level=confidence.level.value,
        confidence_score=confidence.score,
        image_ids=image_ids,
        data=result.data,
        trace=trace_obj.model_dump(mode="json"),
        status=status.value,
        input_count=normalized.input_count,
        modalities=[m.value for m in normalized.modalities],
        tool=result.tool_name,
        tier=result.tier.label,
        duration_ms=total_ms,
    )

    return {
        "analysis_id": analysis_id,
        "id": analysis_id,
        "query": query,
        "mode": normalized.mode,
        "task": selected_task.value,
        "tool": result.tool_name,
        "tier": result.tier.label,
        "status": status,
        "input_count": normalized.input_count,
        "modalities": list(normalized.modalities),
        "answer": result.answer,
        "confidence_score": confidence.score,
        "confidence_level": confidence.level,
        "confidence": confidence.to_dict(),
        "evidence": list(result.evidence),
        "structured_evidence": [e.model_dump() for e in structured_evidence],
        "execution_trace": trace_obj.model_dump(mode="json"),
        "data": result.data,
        "warnings": warnings,
        "artifacts": rendered_artifacts,
        "inputs": input_metas,
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# Analysis endpoints
#
# Every handler lets :class:`AppError` propagate to the application-level handler in
# ``app.main``, which renders the §29 envelope with that error's own code and status. There is no
# blanket ``except Exception`` here: the previous version caught everything and returned HTTP 200
# with "Insufficient evidence for a reliable conclusion", which is exactly the misattribution §27
# forbids — a routing bug reported as a property of the imagery.
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Run the auto-routed analysis pipeline (§5, §6)."""
    return _run_agentic_workflow(
        db, payload.query, payload.image_ids, task_override=payload.task_override
    )


def _run_pinned(
    payload: AnalyzeRequest, db: Session, task: QueryTask
) -> dict[str, Any]:
    """Serve a task-specific endpoint by pinning ``task`` unless the caller pinned one already.

    The pinned task still passes through the contract gate, so e.g. ``POST /vqa`` with two images
    is refused with ``TOOL_INPUT_CONTRACT_ERROR`` before any analysis runs.
    """
    override = payload.task_override
    if not override or override == TASK_OVERRIDE_AUTO:
        override = task.value
    return _run_agentic_workflow(db, payload.query, payload.image_ids, task_override=override)


@router.post("/vqa", response_model=AnalyzeResponse)
def vqa(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Visual question answering on a single optical image."""
    return _run_pinned(payload, db, QueryTask.VQA)


@router.post("/ground", response_model=AnalyzeResponse)
def ground(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Region grounding on a single optical image."""
    return _run_pinned(payload, db, QueryTask.GROUNDING)


@router.post("/caption", response_model=AnalyzeResponse)
def caption(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Scene captioning for a single optical image."""
    return _run_pinned(payload, db, QueryTask.CAPTIONING)


@router.post("/change", response_model=AnalyzeResponse)
def change(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Bitemporal change detection across exactly two co-registered images."""
    return _run_pinned(payload, db, QueryTask.CHANGE_DETECTION)


@router.post("/optical-sar", response_model=AnalyzeResponse)
def optical_sar(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Cross-modal optical + SAR analysis over one optical and one SAR image."""
    return _run_pinned(payload, db, QueryTask.OPTICAL_SAR_ANALYSIS)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _require_record(db: Session, analysis_id: str) -> Any:
    rec = get_analysis_record(db, analysis_id)
    if rec is None:
        raise NotFoundError(
            f"Analysis '{analysis_id}' was not found.",
            code=ErrorCode.NOT_FOUND,
            context={"analysis_id": analysis_id},
        )
    return rec


def _stored_evidence(trace: dict[str, Any]) -> list[dict[str, Any]]:
    raw = trace.get("evidence")
    return list(raw) if isinstance(raw, list) else []


def _stored_confidence(trace: dict[str, Any], rec: Any) -> dict[str, Any]:
    """The confidence report saved with the trace, or an honest minimum reconstructed from the row.

    Rows written before the report was persisted have only the score and the level. Those two are
    reported and nothing else is claimed: ``reasons`` and ``factors`` stay empty rather than being
    back-filled with plausible text, and ``sufficient`` is derived the way the confidence engine
    derives it — from the level, not from a threshold invented here.
    """
    stored = trace.get("confidence")
    if isinstance(stored, dict) and "level" in stored:
        return stored
    level = str(rec.confidence_level)
    return {
        "level": level,
        "score": rec.confidence_score,
        "sufficient": level != ConfidenceLevel.INSUFFICIENT.value,
        "limiting_factor": None,
        "reasons": [],
        "factors": [],
    }


def _stored_list(trace: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """A list of descriptors saved with the trace, or an empty list.

    Empty means "this run recorded none" — for a row written before the field existed, or for a run
    whose imagery genuinely failed to render. Nothing is reconstructed to fill the gap: an invented
    artifact URL would 404, and an invented input metadata block would describe a file this analysis
    may never have read.
    """
    raw = trace.get(key)
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


@router.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Re-read one persisted analysis, grounding included.

    This endpoint used to raise HTTP 500: it put the stored structured evidence into the response's
    ``evidence`` field, which the schema declares as ``list[str]``, so serialisation failed for
    every analysis that had any evidence at all. The two are now kept apart — ``evidence`` carries
    the observation lines, ``structured_evidence`` the §15 items — and the record's own
    ``mode``/``tool``/``status``/``input_count``/``modalities`` columns are reported rather than
    being re-guessed from the trace blob.
    """
    rec = _require_record(db, analysis_id)
    trace = rec.trace
    items = _stored_evidence(trace)
    lines = [
        str(item.get("description", ""))
        for item in items
        if item.get("evidence_type") == EvidenceType.OBSERVATION.value
    ]
    return {
        "analysis_id": rec.id,
        "id": rec.id,
        "query": rec.query,
        "mode": rec.mode,
        "task": rec.task,
        "tool": rec.tool,
        "tier": rec.tier,
        "status": rec.status,
        "input_count": rec.input_count,
        "modalities": rec.modalities,
        "image_ids": rec.image_ids,
        "answer": rec.answer,
        "confidence_score": rec.confidence_score,
        "confidence_level": rec.confidence_level,
        "confidence": _stored_confidence(trace, rec),
        "evidence": lines,
        "structured_evidence": items,
        "execution_trace": trace,
        "data": rec.data,
        "warnings": list(trace.get("warnings", [])),
        # Read back from the trace blob the run wrote, so a reloaded analysis keeps its imagery and
        # its input metadata instead of coming back as a result page with an empty Imagery panel.
        "artifacts": _stored_list(trace, "artifacts"),
        "inputs": _stored_list(trace, "inputs"),
        "duration_ms": rec.duration_ms,
        "created_at": rec.created_at,
    }


@router.get("/analysis/{analysis_id}/trace", response_model=ExecutionTraceSchema)
def get_trace(analysis_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """The stored execution trace for one analysis (§7)."""
    rec = _require_record(db, analysis_id)
    trace = rec.trace
    if not trace:
        raise NotFoundError(
            f"Analysis '{analysis_id}' has no stored execution trace.",
            code=ErrorCode.NOT_FOUND,
            context={"analysis_id": analysis_id},
        )
    return trace


@router.get("/history", response_model=HistoryResponse)
def history(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Most recent analyses, newest first, each reporting its own stored status and mode."""
    records = list_analysis_records(db, limit=limit)
    items = [rec.to_summary() for rec in records]
    return {"total": len(items), "items": items}


# ---------------------------------------------------------------------------
# Report
#
# §1/§32: this is the *existing* report system, kept and extended — not a second one. The change
# here is that every interpolated value is HTML-escaped. The query, the answer and the evidence are
# attacker-controlled (the query arrives verbatim from the client and is stored), so the previous
# f-string concatenation let a query like ``<script>…</script>`` execute in any browser that opened
# a downloaded report: stored XSS (§33).
# ---------------------------------------------------------------------------

_REPORT_CSS = (
    "body{font-family:system-ui,Segoe UI,sans-serif;margin:0;padding:2rem;color:#0f172a;"
    "background:#f8fafc}h1{font-size:1.5rem;margin:0 0 .25rem}h2{font-size:1.05rem;"
    "margin:1.75rem 0 .5rem;border-bottom:1px solid #cbd5e1;padding-bottom:.25rem}"
    "table{border-collapse:collapse;width:100%;font-size:.875rem}"
    "th,td{border:1px solid #cbd5e1;padding:.4rem .6rem;text-align:left;vertical-align:top}"
    "th{background:#e2e8f0}code{background:#e2e8f0;padding:.1rem .3rem;border-radius:3px}"
    ".meta{color:#475569;font-size:.8rem}.answer{background:#fff;border:1px solid #cbd5e1;"
    "border-radius:6px;padding:1rem;white-space:pre-wrap}"
)


def _esc(value: Any) -> str:
    """Escape any value for HTML text content. Never interpolate a raw value into the report."""
    return html.escape("" if value is None else str(value), quote=True)


def _report_rows(pairs: list[tuple[str, Any]]) -> str:
    return "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in pairs)


def _report_html(rec: Any, trace: dict[str, Any], items: list[dict[str, Any]]) -> str:
    steps = trace.get("steps", []) if isinstance(trace, dict) else []
    step_rows = "".join(
        "<tr>"
        f"<td>{_esc(s.get('step_index'))}</td>"
        f"<td>{_esc(s.get('step_name'))}</td>"
        f"<td>{_esc(s.get('tool_name'))}</td>"
        f"<td>{_esc(s.get('status'))}</td>"
        f"<td>{_esc(s.get('duration_ms'))}</td>"
        f"<td>{_esc(s.get('details'))}</td>"
        "</tr>"
        for s in steps
        if isinstance(s, dict)
    )
    evidence_rows = "".join(
        "<tr>"
        f"<td>{_esc(i.get('evidence_type'))}</td>"
        f"<td>{_esc(i.get('source'))}</td>"
        f"<td>{_esc(i.get('description'))}</td>"
        f"<td>{_esc(i.get('value'))}</td>"
        f"<td>{_esc(i.get('unit'))}</td>"
        "</tr>"
        for i in items
        if isinstance(i, dict)
    )
    if not evidence_rows:
        evidence_rows = (
            "<tr><td colspan='5'>No structured evidence was recorded for this analysis.</td></tr>"
        )
    summary = _report_rows(
        [
            ("Analysis ID", rec.id),
            ("Created", rec.created_at),
            ("Input mode", rec.mode),
            ("Task", rec.task),
            ("Tool", rec.tool),
            ("Tier", rec.tier),
            ("Status", rec.status),
            ("Images", rec.input_count),
            ("Modalities", ", ".join(rec.modalities)),
            ("Confidence", f"{rec.confidence_score:.2f} ({rec.confidence_level})"),
            ("Duration (ms)", rec.duration_ms),
        ]
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>SatQuery AI report {_esc(rec.id)}</title><style>{_REPORT_CSS}</style></head><body>"
        f"<h1>SatQuery AI — Analysis Report</h1>"
        f"<p class='meta'>Generated from the persisted analysis record. Every value below was "
        f"measured by the run it describes; nothing is illustrative.</p>"
        f"<h2>Query</h2><div class='answer'>{_esc(rec.query)}</div>"
        f"<h2>Summary</h2><table>{summary}</table>"
        f"<h2>Answer</h2><div class='answer'>{_esc(rec.answer)}</div>"
        "<h2>Evidence</h2><table><tr><th>Type</th><th>Source</th><th>Description</th>"
        f"<th>Value</th><th>Unit</th></tr>{evidence_rows}</table>"
        "<h2>Execution trace</h2><table><tr><th>#</th><th>Step</th><th>Tool</th><th>Status</th>"
        f"<th>ms</th><th>Details</th></tr>{step_rows}</table>"
        "</body></html>"
    )


@router.get("/analysis/{analysis_id}/report", response_model=ReportResponse)
def get_report(analysis_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Render the stored analysis as a self-contained HTML report plus its JSON source."""
    rec = _require_record(db, analysis_id)
    trace = rec.trace
    items = _stored_evidence(trace)
    report_json = {
        "analysis_id": rec.id,
        "query": rec.query,
        "mode": rec.mode,
        "task": rec.task,
        "tool": rec.tool,
        "tier": rec.tier,
        "status": rec.status,
        "input_count": rec.input_count,
        "modalities": rec.modalities,
        "image_ids": rec.image_ids,
        "answer": rec.answer,
        "confidence_score": rec.confidence_score,
        "confidence_level": rec.confidence_level,
        "structured_evidence": items,
        "execution_trace": trace,
        "data": rec.data,
        "duration_ms": rec.duration_ms,
        "created_at": rec.created_at,
    }
    return {
        "analysis_id": rec.id,
        "report_html": _report_html(rec, trace, items),
        "report_json": report_json,
        "download_filename": f"satquery-report-{rec.id}.html",
    }


# ---------------------------------------------------------------------------
# Registry, health, evaluation
# ---------------------------------------------------------------------------


@router.get("/models", response_model=list[ModelInfo])
def models() -> list[dict[str, Any]]:
    """The tools this build can actually run, read from the registry (§26, §30).

    Previously a hardcoded list of invented component names with ``available: true`` written next to
    each one. Nothing here is asserted: the name, task, tier, summary and contract come from the
    tool class, and ``available`` is probed per request, so an entry reads ``ready`` only if it
    could run right now.
    """
    entries = registry.describe_all_with_contracts()
    for entry in entries:
        entry["status"] = "ready" if entry["available"] else "unavailable"
    return entries


@router.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Liveness plus the two facts a caller cannot otherwise check: storage and learned models."""
    storage_ok = False
    try:
        settings.storage_root.mkdir(parents=True, exist_ok=True)
        probe = settings.storage_root / ".health"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        storage_ok = True
    except OSError as exc:  # pragma: no cover - depends on the host filesystem
        logger.warning("storage health probe failed", extra={"error": str(exc)})
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.version,
        "learned_models_enabled": settings.enable_learned_models,
        "storage_ok": storage_ok,
    }


@router.get("/evaluation")
def evaluation() -> dict[str, Any]:
    """Serve the stored evaluation report, read-only (§21, §23, §48).

    This endpoint used to *run* the evaluation and overwrite the report on every GET, so the numbers
    a reviewer saw depended on when they loaded the page and no two loads were guaranteed to agree.
    It now only reads the artefact produced by ``make evaluate``, and says plainly when there isn't
    one instead of manufacturing a placeholder.
    """
    if not _EVALUATION_REPORT.exists():
        return {
            "status": "NOT EVALUATED",
            "report_path": str(_EVALUATION_REPORT),
            "message": (
                "No evaluation artefact is present. Run 'make evaluate' to produce "
                f"{_EVALUATION_REPORT.name}; this endpoint never generates one on request."
            ),
            "benchmarks": {},
        }
    try:
        payload = json.loads(_EVALUATION_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppError(
            "The stored evaluation report could not be read.",
            code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            context={"report_path": str(_EVALUATION_REPORT), "error": str(exc)},
        ) from exc
    payload["report_path"] = str(_EVALUATION_REPORT)
    payload["generated_by"] = "make evaluate"
    payload.setdefault("status", "EVALUATED")
    return payload
