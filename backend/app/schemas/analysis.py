"""Pydantic schemas for API requests, responses, execution trace, and evidence structures.

Two rules govern the shapes here:

1. **No field carries a plausible-looking default that the backend might not have set.** The old
   ``mode: str | None = "single_optical"`` meant a response that never computed a mode still
   claimed one, and the claim reached the UI and the report indistinguishable from a measured one.
   Provenance-bearing fields are required, or explicitly ``None``.
2. **Closed vocabularies are enums.** ``task_override`` and ``evidence_type`` were free strings, so
   an unknown value was silently ignored (the override) or silently displayed (the evidence type).
   Both now reject at the edge, with a 422 naming the accepted values.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.types import (
    AnalysisStatus,
    ConfidenceLevel,
    EvidenceType,
    InputMode,
    Modality,
    QueryTask,
    StepStatus,
)

#: The literal a client sends to mean "let the router decide" — the default, not a task.
TASK_OVERRIDE_AUTO = "auto"

#: Tasks a client may pin explicitly. Deliberately excludes UNKNOWN (not a request) and
#: REPORT_GENERATION (served by the report endpoint, not by an analysis run).
SELECTABLE_TASKS: frozenset[str] = frozenset(
    t.value
    for t in QueryTask
    if t not in (QueryTask.UNKNOWN, QueryTask.REPORT_GENERATION)
)


class UploadResponse(BaseModel):
    """Response returned after uploading a remote sensing raster image."""
    image_id: str
    filename: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: str | None = None
    crs_epsg: int | None = None
    georeferenced: bool
    modality: Modality
    preview_url: str
    decimation: float = 1.0
    pixel_size: list[float] | None = None
    bounds: list[float] | None = None
    band_descriptions: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    """Payload sent by client to analyze images with a natural language query."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    image_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="Uploaded image identifiers (1 or 2 images)",
    )
    task_override: str | None = Field(
        default=None,
        description=(
            "Optional manual task selection. 'auto' (or omitted) lets the router decide. "
            "An unrecognised value is rejected rather than ignored."
        ),
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="Execution parameters")

    @field_validator("task_override")
    @classmethod
    def _validate_task_override(cls, value: str | None) -> str | None:
        """Reject an unknown override instead of silently falling back to the routed task.

        The previous code caught the ``ValueError`` and ``pass``-ed, so a typo produced a
        confident answer to a *different* question than the one the client asked to run.
        """
        if value is None or value == TASK_OVERRIDE_AUTO:
            return value
        if value not in SELECTABLE_TASKS:
            allowed = ", ".join(sorted(SELECTABLE_TASKS | {TASK_OVERRIDE_AUTO}))
            raise ValueError(f"Unknown task_override {value!r}. Expected one of: {allowed}.")
        return value


class EvidenceItem(BaseModel):
    """One structured piece of evidence behind an answer (the single §15 evidence schema).

    Every field is populated from the analysis. ``confidence`` is this item's own confidence, not
    the overall score copied onto each row — the earlier behaviour, which made a weakly supported
    observation look exactly as strong as the headline finding.
    """

    evidence_type: EvidenceType = Field(description="What kind of evidence this is")
    source: str = Field(description="The tool or measurement that produced it, e.g. 'change-cva-cv'")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="This item's own confidence, or null when the item carries no separate score",
    )
    geometry: dict[str, Any] | None = Field(
        default=None, description="GeoJSON-style geometry when the evidence is spatial"
    )
    description: str = Field(description="Human-readable statement of the evidence")
    value: float | None = Field(default=None, description="The measured number, when there is one")
    unit: str | None = Field(default=None, description="Unit of `value`, e.g. 'km2', 'dB', 'percent'")


class ConfidenceFactorSchema(BaseModel):
    """One measured input to the confidence score, as computed by ``app.services.confidence``."""

    name: str = Field(description="Stable identifier, e.g. 'corroboration', 'registration'")
    value: float = Field(ge=0.0, le=1.0, description="The measurement mapped onto [0, 1]")
    weight: float = Field(ge=0.0, description="Relative influence in the weighted mean")
    kind: str = Field(description="'contributor' (averaged in) or 'gate' (also caps the score)")
    reason: str = Field(description="Plain-language statement of what this factor says")


class ConfidenceSchema(BaseModel):
    """The full measured confidence report behind an answer (§27).

    Only ``score`` and ``level`` used to be published, so a client had no access to *why* the score
    was what it was, nor to the backend's own sufficiency verdict. The UI filled both gaps by
    inventing them: three fixed reason strings ("Valid radiometric bounds", …) that no measurement
    produced, and its own ``score >= 0.5`` sufficiency rule, which contradicted this backend on
    every ``low`` answer — labelling a real, measured result "insufficient evidence", the one
    misattribution §0 and §27 single out. Publishing the report removes the reason to guess.
    """

    level: ConfidenceLevel
    score: float = Field(ge=0.0, le=1.0)
    sufficient: bool = Field(
        description="False only when the evidence cannot support any defensible claim"
    )
    limiting_factor: str | None = Field(
        default=None, description="Name of the factor that bound the score; null if none was measured"
    )
    reasons: list[str] = Field(
        default_factory=list, description="Factor reasons, weakest measurement first"
    )
    factors: list[ConfidenceFactorSchema] = Field(default_factory=list)


class Wgs84BoundsSchema(BaseModel):
    """Axis-aligned lat/lon box enclosing a rendered image's footprint."""
    south: float
    west: float
    north: float
    east: float


class LegendEntrySchema(BaseModel):
    """One legend swatch: the colour a class is drawn in, and the key it is drawn from."""
    label: str
    color: str = Field(description="CSS hex colour, taken from the palette the image was drawn with")
    key: str


class ArtifactSchema(BaseModel):
    """One rendered PNG the result page can display.

    ``bounds_wgs84`` is null for an ungeoreferenced input, which the UI reads as "show this in a
    plain pixel viewer" — it is never filled with a plausible box to make the map work (§8).
    """
    kind: str = Field(description="'base' for the input composite; otherwise the overlay's map")
    label: str
    url: str = Field(description="Path the file is served at, under the /storage mount")
    bounds_wgs84: Wgs84BoundsSchema | None = None
    legend: list[LegendEntrySchema] = Field(default_factory=list)


class InputMetaSchema(BaseModel):
    """Per-input raster metadata: the file's own header, plus what loading resolved.

    Extra keys are allowed through deliberately — this is
    :meth:`app.geospatial.raster.RasterMetadata.to_dict` on the wire, and a new measurement added
    there should reach the client rather than being silently dropped by this schema.
    """
    model_config = ConfigDict(extra="allow")

    filename: str
    driver: str = Field(description="The GDAL driver the file was opened with, e.g. 'GTiff'")
    width: int
    height: int
    bands: int
    dtype: str
    crs: str | None = None
    crs_epsg: int | None = None
    transform: list[float] | None = None
    bounds: list[float] | None = None
    nodata: float | None = None
    pixel_size: list[float] | None = None
    georeferenced: bool
    band_descriptions: list[str | None] = Field(
        default_factory=list,
        description=(
            "The file's own band names, positionally. An entry is null for a band the file does not "
            "name (a plain PNG names none) — the position is kept rather than compacted, so band 3's "
            "name cannot slide onto band 2"
        ),
    )
    decimation: float = 1.0
    modality: Modality
    band_roles: list[str] = Field(default_factory=list)
    bounds_wgs84: Wgs84BoundsSchema | None = None


class ExecutionTraceStep(BaseModel):
    """Individual step execution trace item."""
    step_index: int
    step_name: str
    tool_name: str | None = None
    status: StepStatus
    details: str
    duration_ms: int | None = Field(
        default=None, description="Measured wall-clock duration of this step; null if not timed"
    )
    warnings: list[str] = Field(default_factory=list)


class ExecutionTraceSchema(BaseModel):
    """Complete auditable execution trace exposing orchestration workflow without chain-of-thought."""
    analysis_id: str
    task: str
    source: str
    mode: InputMode = Field(description="The canonical input mode routing was performed against")
    selected_tools: list[str]
    input_summary: dict[str, Any]
    status: AnalysisStatus
    steps: list[ExecutionTraceStep]
    confidence_score: float
    confidence_level: ConfidenceLevel
    confidence: ConfidenceSchema | None = Field(
        default=None,
        description=(
            "The measured confidence report, stored with the trace so a re-read analysis keeps the "
            "factors behind its score. Null only for rows persisted before this field existed"
        ),
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description=(
            "The structured §15 evidence behind the answer, stored with the trace so a persisted "
            "analysis can be re-read with its grounding intact rather than only its conclusion"
        ),
    )
    warnings: list[str] = Field(default_factory=list)
    reasoning_summary: str
    total_duration_ms: int | None = Field(
        default=None, description="Measured end-to-end duration of the whole analysis (§51)"
    )
    artifacts: list[ArtifactSchema] = Field(
        default_factory=list,
        description=(
            "The imagery rendered for this analysis, stored with the trace so a re-read analysis "
            "still has its pictures. Empty for rows persisted before rendering existed, and for a "
            "run whose imagery could not be rendered — never a placeholder"
        ),
    )
    inputs: list[InputMetaSchema] = Field(
        default_factory=list,
        description=(
            "The rasters the analysis actually read, as they were read (post-decimation), stored "
            "with the trace so the Inputs panel survives a reload"
        ),
    )


class AnalyzeResponse(BaseModel):
    """Main response returned by /api/analyze, /api/vqa, /api/ground, etc."""
    analysis_id: str
    id: str | None = None
    query: str
    mode: InputMode = Field(description="Canonical input mode, derived from the actual rasters (§5)")
    task: str
    tool: str | None = Field(
        default=None, description="The tool that actually produced the answer; null if none ran"
    )
    tier: str | None = Field(default=None, description="Fallback-chain rung of that tool")
    status: AnalysisStatus = Field(description="Terminal state of the analysis")
    input_count: int = Field(description="How many rasters the analysis actually read")
    modalities: list[Modality] = Field(
        default_factory=list, description="Per-input modality, in caller order"
    )
    answer: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    confidence: ConfidenceSchema = Field(
        description="The measured confidence report: sufficiency verdict, limiting factor, factors"
    )
    evidence: list[str] = Field(default_factory=list)
    structured_evidence: list[EvidenceItem] = Field(default_factory=list)
    execution_trace: ExecutionTraceSchema
    data: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactSchema] = Field(
        default_factory=list,
        description=(
            "Imagery rendered from this analysis's own arrays — the input composite and the task's "
            "overlay. Empty means nothing was rendered, not that nothing was found"
        ),
    )
    inputs: list[InputMetaSchema] = Field(
        default_factory=list, description="Per-input raster metadata, in caller order"
    )
    created_at: str


class HistoryItem(BaseModel):
    """Summary of a past analysis for history listing."""
    analysis_id: str
    id: str | None = None
    query: str
    title: str | None = None
    mode: InputMode
    task: str
    tool: str | None = None
    confidence_level: ConfidenceLevel
    confidence: str | None = None
    confidence_score: float
    answer: str
    image_count: int
    created_at: str


class HistoryResponse(BaseModel):
    """Response returned by GET /api/history."""
    total: int
    items: list[HistoryItem]


class ReportResponse(BaseModel):
    """Response returned by GET /api/analysis/{id}/report."""
    analysis_id: str
    report_html: str
    report_json: dict[str, Any]
    download_filename: str


class ToolContractSchema(BaseModel):
    """A tool's declared input contract, as published by ``GET /api/models`` (§6)."""
    supported_modes: list[InputMode]
    min_images: int
    max_images: int
    required_modalities: list[Modality]


class ModelInfo(BaseModel):
    """One registry entry for ``GET /api/models`` (§26, §30).

    ``available`` is probed at request time, so an entry is listed as ready only if it can
    actually run. Nothing here is hardcoded: every field comes from the tool class itself.
    """

    name: str
    task: str
    tier: str
    summary: str
    available: bool
    status: str = Field(description="'ready' when available, otherwise 'unavailable'")
    contract: ToolContractSchema
    runtime: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Live provenance for a tool backed by a real artefact: mode ('LIVE' or 'UNAVAILABLE'), "
            "checkpoint path, architecture, pretrained weights, adaptation method, dataset and "
            "measured test accuracy. Absent for the deterministic analysers, which have no "
            "checkpoint. Probed per request, never cached."
        ),
    )


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    app_name: str
    version: str
    learned_models_enabled: bool
    storage_ok: bool
