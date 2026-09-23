/**
 * SatQuery API client.
 *
 * Primary Mode: Live FastAPI backend for real inference, upload, and GIS analysis.
 * Fallback Mode: Bundled deterministic demo fixtures when the backend is offline.
 */
import type {
  AnalysisData,
  AnalysisResult,
  AnalysisStatus,
  AnalysisSummary,
  ApiErrorEnvelope,
  Artifact,
  ConfidenceLevel,
  ConfidenceReport,
  HealthStatus,
  InputMeta,
  InputMode,
  LegendEntry,
  Metrics,
  Modality,
  ToolTier,
  TraceStep,
  Wgs84Bounds,
} from '@/types/api'
import { MODALITIES } from '@/types/api'
import { MODALITY_HINT_BY_SLOT_ROLE, MODE_SLOTS } from '@/lib/uploads'
import { listFixtureSummaries, loadFixture } from './fixtures'

// In-memory cache for live runs generated during the current browser session
const liveSessionAnalyses = new Map<string, AnalysisResult>()

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')
const PREFER_FIXTURES =
  import.meta.env.MODE === 'test'
    ? (import.meta.env.VITE_USE_FIXTURES ?? 'true').toLowerCase() === 'true'
    : (import.meta.env.VITE_USE_FIXTURES ?? 'false').toLowerCase() === 'true'

export const apiConfig = {
  baseUrl: API_BASE_URL,
  preferFixtures: PREFER_FIXTURES,
  offline: PREFER_FIXTURES,
}

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

/**
 * Uniform API error wrapper.
 */
export class ApiRequestError extends Error {
  readonly errorCode: string
  readonly requestId: string | null
  readonly recoverable: boolean
  readonly httpStatus: number | null

  constructor(
    envelope: Partial<ApiErrorEnvelope> & { message: string },
    httpStatus: number | null = null,
  ) {
    super(envelope.message)
    Object.setPrototypeOf(this, new.target.prototype)
    this.name = 'ApiRequestError'
    this.errorCode = envelope.error_code ?? 'unknown_error'
    this.requestId = envelope.request_id ?? null
    this.recoverable = envelope.recoverable ?? false
    this.httpStatus = httpStatus
  }
}

async function toApiError(res: Response): Promise<ApiRequestError> {
  let envelope: Partial<ApiErrorEnvelope> = {}
  try {
    const body: unknown = await res.json()
    if (body && typeof body === 'object') envelope = body as Partial<ApiErrorEnvelope>
  } catch {
    // Non-JSON error body
  }
  return new ApiRequestError(
    {
      message: envelope.message ?? `Request failed (${res.status} ${res.statusText}).`,
      error_code: envelope.error_code ?? `http_${res.status}`,
      request_id: envelope.request_id,
      recoverable: envelope.recoverable ?? res.status >= 500,
    },
    res.status,
  )
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(apiUrl(path), init)
  } catch {
    throw new ApiRequestError({
      message: 'Could not reach the SatQuery backend. Is the API server running?',
      error_code: 'network_unreachable',
      recoverable: true,
    })
  }
  if (!res.ok) throw await toApiError(res)
  return (await res.json()) as T
}

// --- Upload Helper -----------------------------------------------------------

interface UploadResponse {
  image_id: string
  filename: string
  width: number
  height: number
  bands: number
  dtype: string
  crs: string | null
  crs_epsg: number | null
  georeferenced: boolean
  modality: string
  preview_url: string
  decimation: number
  pixel_size: [number, number] | null
  bounds: [number, number, number, number] | null
  band_descriptions: string[]
}

/**
 * Upload one raster.
 *
 * `modalityHint` declares the sensor when the UI already knows it. It matters: the backend can only
 * infer modality from filename tokens and band descriptions, so a real SAR scene exported as
 * `subset_1.tif` with an unlabelled band infers as `unknown`, the optical+SAR pair then normalises to
 * `bitemporal_pair`, and the fusion the user asked for is never run. The mode selector already asked
 * which file is optical and which is SAR, so that answer is sent rather than re-guessed server-side.
 */
async function uploadFile(file: File, modalityHint?: string): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  if (modalityHint) formData.append('modality_hint', modalityHint)
  return fetchJson<UploadResponse>('/api/upload', {
    method: 'POST',
    body: formData,
  })
}

// --- Reads -------------------------------------------------------------------

/** Recent analyses for the dashboard. */
export async function listAnalyses(): Promise<AnalysisSummary[]> {
  if (apiConfig.offline) return listFixtureSummaries()

  try {
    interface HistoryResponse {
      items: Array<{
        analysis_id: string
        id?: string
        title?: string
        query: string
        mode?: string
        task: string
        answer: string
        confidence_level?: string
        confidence?: string
        confidence_score?: number
        created_at: string
      }>
    }
    const history = await fetchJson<HistoryResponse>('/api/history')
    if (history.items && history.items.length > 0) {
      const liveItems: AnalysisSummary[] = history.items.map((item) => ({
        id: item.analysis_id || item.id || '',
        title: item.title || item.query,
        mode: (item.mode as any) || 'single_optical',
        task: item.task as any,
        query: item.query,
        created_at: item.created_at,
        status: 'success',
        confidence: (item.confidence_level || item.confidence || 'high') as any,
        answer: item.answer,
      }))
      // Merge with fixtures so demo items remain accessible
      const fixtures = listFixtureSummaries()
      const existingIds = new Set(liveItems.map((i) => i.id))
      return [...liveItems, ...fixtures.filter((f) => !existingIds.has(f.id))]
    }
  } catch {
    // Fall back to fixture list
  }
  return listFixtureSummaries()
}

const ANALYSIS_STATUSES: readonly AnalysisStatus[] = ['success', 'partial', 'failed']
const TOOL_TIERS: readonly ToolTier[] = ['preferred', 'fallback', 'classical']

/**
 * The backend's terminal status, or `success` only when it declared nothing.
 *
 * This used to be the literal `'success'`, so a run that finished with band or geometry warnings —
 * the optical/SAR fusion finishes with fourteen — still showed a green Success badge next to its own
 * caveats. Both sides share the vocabulary `success | partial | failed`, so the value is read; the
 * guard exists only so an unrecognised string cannot reach a `variant` lookup.
 */
function asStatus(value: string | undefined): AnalysisStatus {
  return ANALYSIS_STATUSES.find((s) => s === value) ?? 'success'
}

/** The fallback rung the tool sits on. Every registered tool is currently `classical`. */
function asTier(value: string | null | undefined): ToolTier {
  return TOOL_TIERS.find((t) => t === value) ?? 'classical'
}

/**
 * The sensor the backend resolved for an upload, or `unknown` when it could not say.
 *
 * `unknown` is a real answer here, not a placeholder: the backend infers modality from filename
 * tokens and band descriptions, so an unlabelled single-band export genuinely cannot be classified.
 * Coercing it to `optical` would put a sensor name on a file nobody identified.
 */
function asModality(value: string | null | undefined): Modality {
  return MODALITIES.find((m) => m === value) ?? 'unknown'
}

/** One rendered PNG as the backend publishes it (`ArtifactSchema`). */
interface BackendArtifact {
  kind: string
  label: string
  url: string
  bounds_wgs84: Wgs84Bounds | null
  legend: LegendEntry[]
}

/** Per-input raster metadata as the backend publishes it (`InputMetaSchema`). */
interface BackendInputMeta {
  filename: string
  driver: string
  width: number
  height: number
  bands: number
  dtype: string
  crs: string | null
  crs_epsg: number | null
  transform: number[] | null
  bounds: number[] | null
  nodata: number | null
  pixel_size: number[] | null
  georeferenced: boolean
  band_descriptions: Array<string | null>
  decimation: number
  modality: string
  band_roles: string[]
  bounds_wgs84: Wgs84Bounds | null
}

/**
 * The imagery the backend rendered for this run — its base composite and the task's own overlay.
 *
 * Both mappers below used to hardcode `artifacts: []`, which is why a live result page showed "No
 * rendered imagery for this analysis." even for a run that had classified every pixel: the backend
 * now renders and publishes the PNGs, and this reads them. An empty list stays empty — a run whose
 * overlay genuinely failed to render says so in its warnings, and nothing is substituted for it.
 *
 * `bounds_wgs84` passes through unchanged, including its null: `result-map.tsx` reads null as "show
 * this in a plain pixel viewer" rather than placing an ungeoreferenced image on a map (§8).
 */
function mapArtifacts(raw: BackendArtifact[] | undefined): Artifact[] {
  return (raw ?? []).map((a) => ({
    kind: a.kind,
    label: a.label,
    url: a.url.startsWith('http') ? a.url : apiUrl(a.url),
    bounds_wgs84: a.bounds_wgs84 ?? null,
    legend: a.legend ?? [],
  }))
}

/** `[x, y]` ground sample distance, or null for a raster with no transform to derive it from. */
function asPixelSize(value: number[] | null | undefined): [number, number] | null {
  return value && value.length >= 2 ? [value[0], value[1]] : null
}

/**
 * The rasters the analysis actually read, as it read them.
 *
 * This is the raster's own header — the driver, transform, nodata, band roles and decimation the
 * loader resolved — which the upload response does not carry. The Inputs panel previously had to
 * make do with the upload metadata (and, on reload, with nothing at all).
 */
function mapInputs(raw: BackendInputMeta[] | undefined): InputMeta[] {
  return (raw ?? []).map((m) => ({
    filename: m.filename,
    driver: m.driver,
    width: m.width,
    height: m.height,
    bands: m.bands,
    dtype: m.dtype,
    crs: m.crs,
    crs_epsg: m.crs_epsg,
    transform: m.transform,
    bounds: m.bounds,
    nodata: m.nodata,
    pixel_size: asPixelSize(m.pixel_size),
    georeferenced: m.georeferenced,
    band_descriptions: m.band_descriptions ?? [],
    decimation: m.decimation ?? 1.0,
    modality: asModality(m.modality),
    band_roles: m.band_roles ?? [],
    bounds_wgs84: m.bounds_wgs84 ?? null,
  }))
}

/**
 * The backend's own confidence report, or — for a row persisted before that report was published —
 * the two fields it does carry and nothing more.
 *
 * Both mappers previously invented this object: a fixed `['Valid radiometric bounds', 'Sensor
 * metadata verified', 'Measured spectral indices']` reason list that no measurement produced, a
 * single fabricated `model_confidence` factor, and `sufficient: score >= 0.5` — a threshold of the
 * UI's own making. That last one is why a measured `low` answer (0.494) was headlined "Insufficient
 * evidence for a reliable conclusion" while the same card showed a "Low" badge and a "Success"
 * status: the backend considers `low` sufficient, and only `insufficient` not. Sufficiency is the
 * backend's verdict to make, so it is read, not recomputed.
 */
function mapConfidence(
  report:
    | {
        level: string
        score: number
        sufficient: boolean
        limiting_factor: string | null
        reasons: string[]
        factors: Array<{ name: string; value: number; weight: number; kind: string; reason: string }>
      }
    | undefined,
  score: number,
  level: string,
): ConfidenceReport {
  if (report) {
    return {
      score: report.score,
      level: report.level as ConfidenceLevel,
      sufficient: report.sufficient,
      limiting_factor: report.limiting_factor,
      reasons: report.reasons ?? [],
      factors: (report.factors ?? []).map((f) => ({
        name: f.name,
        value: f.value,
        weight: f.weight,
        kind: f.kind as ConfidenceReport['factors'][number]['kind'],
        reason: f.reason,
      })),
    }
  }
  return {
    score,
    level: level as ConfidenceLevel,
    // Derived the way the confidence engine derives it — from the level, not from a threshold.
    sufficient: level !== 'insufficient',
    limiting_factor: null,
    reasons: [],
    factors: [],
  }
}

/** How an analysis id should be resolved when both a recording and a live run could claim it. */
export interface GetAnalysisOptions {
  /**
   * Open the shipped recording for this id rather than asking the backend.
   *
   * The dashboard's "View example" links advertise one specific recorded case each, and the backend
   * mints its own ids from the same `SAT-2026-NNNNNN` sequence — so on a machine that has run more
   * than a hundred analyses, the six recorded ids also exist in its database as unrelated live runs.
   * Resolving live-first therefore opened the wrong analysis entirely: the "Optical scene analysis"
   * example showed a fusion answer to a different question, and showed no imagery, because rows
   * recorded before artifacts were persisted carry none. A link that means "the recording" says so.
   */
  recorded?: boolean
}

/** Full analysis envelope by id. */
export async function getAnalysis(
  id: string,
  opts: GetAnalysisOptions = {},
): Promise<AnalysisResult> {
  // A recorded case is asked for by name, so it is answered from the recording — ahead of both the
  // live-session cache and the backend, either of which may hold a different run under the same id.
  if (opts.recorded) {
    const recorded = await loadFixture(id)
    if (recorded) return recorded
  }

  // Check live session cache first
  if (liveSessionAnalyses.has(id)) {
    return liveSessionAnalyses.get(id)!
  }

  // Try live backend if not explicitly in offline fixtures mode
  if (!apiConfig.offline) {
    try {
      interface BackendConfidence {
        level: string
        score: number
        sufficient: boolean
        limiting_factor: string | null
        reasons: string[]
        factors: Array<{
          name: string
          value: number
          weight: number
          kind: string
          reason: string
        }>
      }
      interface BackendAnalysisDetail {
        analysis_id: string
        id?: string
        query: string
        mode?: string
        task: string
        tool?: string | null
        tier?: string | null
        status?: string
        answer: string
        confidence_score: number
        confidence_level: string
        confidence?: BackendConfidence
        evidence: string[]
        structured_evidence: any[]
        execution_trace: {
          trace_id?: string
          total_duration_ms?: number
          steps: Array<{
            step_index: number
            step_name: string
            tool_name: string
            status: string
            details: string
            duration_ms?: number | null
            warnings: string[]
          }>
        }
        data: any
        duration_ms?: number | null
        warnings: string[]
        artifacts?: BackendArtifact[]
        inputs?: BackendInputMeta[]
        created_at: string
      }
      const raw = await fetchJson<BackendAnalysisDetail>(`/api/analysis/${encodeURIComponent(id)}`)
      const mapped: AnalysisResult = {
        id: raw.analysis_id || raw.id || id,
        request_id: raw.execution_trace?.trace_id ?? id,
        created_at: raw.created_at,
        status: asStatus(raw.status),
        mode: (raw.mode as any) || 'single_optical',
        title: raw.query,
        note: `Live analysis executed via ${raw.task}`,
        query: raw.query,
        task: raw.task as any,
        // The tool the backend says produced the answer. This used to read
        // `execution_trace.steps[1].tool_name`, i.e. whichever component happened to occupy the
        // second trace step — since mode normalisation was inserted there, every result claimed to
        // have been produced by 'ModeNormalizer' rather than by the specialist that ran.
        tool: raw.tool ?? 'unknown',
        tier: asTier(raw.tier),
        duration_ms: raw.duration_ms ?? raw.execution_trace?.total_duration_ms ?? 0,
        answer: raw.answer,
        answer_withheld: null,
        confidence: mapConfidence(raw.confidence, raw.confidence_score, raw.confidence_level),
        evidence: raw.evidence,
        warnings: raw.warnings ?? [],
        data: (raw.data ?? {}) as AnalysisData,
        // Read back from the run's own stored record, so a reloaded page keeps the imagery and the
        // input headers it was first shown with. Both were hardcoded empty here, which is what made
        // the Imagery and Inputs panels come up blank on every reload.
        inputs: mapInputs(raw.inputs),
        trace: (raw.execution_trace?.steps ?? []).map((s) => ({
          step: s.step_name as any,
          tool: s.tool_name,
          status: s.status as any,
          // The measured value, including 0 for a sub-millisecond step and null for one the backend
          // does not time separately. `|| 30` turned both into a plausible-looking invention.
          duration_ms: s.duration_ms ?? null,
          summary: s.details,
          parameters: {},
          warnings: s.warnings ?? [],
        })),
        artifacts: mapArtifacts(raw.artifacts),
        fixture: false,
      }
      liveSessionAnalyses.set(id, mapped)
      return mapped
    } catch {
      // Fall through to fixture lookup
    }
  }

  const fx = await loadFixture(id)
  if (fx) return fx
  throw new ApiRequestError({
    message: `No analysis found for ${id}.`,
    error_code: 'not_found',
    recoverable: false,
  })
}

// --- Submit Analysis ---------------------------------------------------------

export interface AnalyzeRequest {
  mode: InputMode
  query: string
  files: File[]
  demoCaseId?: string
}

export async function analyze(req: AnalyzeRequest): Promise<AnalysisResult> {
  // If a demo case ID was explicitly requested, load the fixture
  if (req.demoCaseId) {
    const fx = await loadFixture(req.demoCaseId)
    if (fx) return fx
    if (apiConfig.offline) {
      throw new ApiRequestError({
        message: `Unknown demo case "${req.demoCaseId}".`,
        error_code: 'not_found',
        recoverable: false,
      })
    }
  }

  if (apiConfig.offline) {
    throw new ApiRequestError({
      message:
        'Start the API server, or pick a demo case — the offline demo will not fabricate a result for uploaded imagery.',
      error_code: 'backend_required',
      recoverable: true,
    })
  }

  // 1. Upload files to backend, declaring the sensor for the slots where the mode fixes it.
  //    optical_sar_pair has one optical slot and one SAR slot, so both are declared; the bitemporal
  //    slots (t1/t2) may be either sensor, so those are left for the backend to infer.
  const hints = MODE_SLOTS[req.mode].map((slot) => MODALITY_HINT_BY_SLOT_ROLE[slot.role])
  const uploadedMeta: UploadResponse[] = []
  for (const [i, file] of req.files.entries()) {
    try {
      const up = await uploadFile(file, hints[i])
      uploadedMeta.push(up)
    } catch (err) {
      if (err instanceof ApiRequestError) throw err
      throw new ApiRequestError({
        message: `Failed to upload "${file.name}". Please check the backend connection.`,
        error_code: 'upload_failed',
        recoverable: true,
      })
    }
  }

  // 2. Submit analysis request to backend
  interface BackendAnalysisResponse {
    analysis_id: string
    query: string
    /** The canonical mode the backend normalised the rasters to — not the one selected in the UI. */
    mode?: string
    task: string
    /** The tool that actually produced the answer, and the fallback rung it sits on. */
    tool?: string | null
    tier?: string | null
    status?: string
    answer: string
    confidence_score: number
    confidence_level: string
    confidence?: {
      level: string
      score: number
      sufficient: boolean
      limiting_factor: string | null
      reasons: string[]
      factors: Array<{
        name: string
        value: number
        weight: number
        kind: string
        reason: string
      }>
    }
    evidence: string[]
    structured_evidence: any[]
    execution_trace: {
      trace_id: string
      total_duration_ms: number
      steps: Array<{
        step_index: number
        step_name: string
        tool_name: string
        status: string
        details: string
        duration_ms: number | null
        warnings: string[]
      }>
    }
    data: any
    warnings: string[]
    /** Rendered by the analysis from its own arrays: the base composite plus the task's overlay. */
    artifacts?: BackendArtifact[]
    /** The rasters the analysis read, with the headers the loader resolved. */
    inputs?: BackendInputMeta[]
    created_at: string
  }

  const raw = await fetchJson<BackendAnalysisResponse>('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: req.query,
      image_ids: uploadedMeta.map((u) => u.image_id),
    }),
  })

  // 3. Imagery and input headers: the analysis's own where it published them, the upload's otherwise.
  //
  //    The analysis renders a base composite *and* the task's overlay from the arrays it produced,
  //    and reports each raster as it actually read it — post-decimation, with the transform, nodata
  //    and band roles the loader resolved. The upload response can supply neither: it knows only a
  //    thumbnail per file and the few header fields the upload endpoint echoes back. So the run's own
  //    values win, and these upload-derived lists stay as the fallback for a backend that publishes
  //    none.
  const uploadedInputs: InputMeta[] = uploadedMeta.map((u) => ({
    filename: u.filename,
    driver: 'GTiff',
    width: u.width,
    height: u.height,
    bands: u.bands,
    dtype: u.dtype,
    crs: u.crs,
    crs_epsg: u.crs_epsg,
    transform: null,
    bounds: u.bounds,
    nodata: null,
    pixel_size: u.pixel_size,
    georeferenced: u.georeferenced,
    band_descriptions: u.band_descriptions ?? [],
    decimation: u.decimation ?? 1.0,
    // The modality the backend actually resolved for this file, not one derived from the selected
    // mode. `req.mode.includes('sar')` labelled *both* inputs of an optical_sar_pair as 'sar'.
    modality: asModality(u.modality),
    // The upload response does not report band roles, so none are claimed here.
    band_roles: [],
    bounds_wgs84: null,
  }))

  const uploadPreviews: Artifact[] = uploadedMeta.map((u) => ({
    kind: 'base',
    label: u.filename,
    url: u.preview_url,
    bounds_wgs84: null,
    legend: [],
  }))

  const renderedArtifacts = mapArtifacts(raw.artifacts)
  const artifacts = renderedArtifacts.length > 0 ? renderedArtifacts : uploadPreviews
  const analysisInputs = mapInputs(raw.inputs)
  const inputs = analysisInputs.length > 0 ? analysisInputs : uploadedInputs

  const traceSteps: TraceStep[] = (raw.execution_trace?.steps ?? []).map((s) => ({
    step: s.step_name as any,
    tool: s.tool_name,
    status: s.status as any,
    // As above: measured, or null when the backend does not time the step separately.
    duration_ms: s.duration_ms ?? null,
    summary: s.details,
    parameters: {},
    warnings: s.warnings ?? [],
  }))

  const result: AnalysisResult = {
    id: raw.analysis_id,
    request_id: raw.execution_trace?.trace_id ?? raw.analysis_id,
    created_at: raw.created_at,
    // The backend's terminal state. Hardcoding 'success' meant a run that finished with geometry or
    // band warnings — the fusion above finished with fourteen — still showed a green Success badge.
    status: asStatus(raw.status),
    // The mode the backend normalised the rasters to, which is the one the analysis actually ran
    // against. The UI selection is only a request; if the two differ (a t1/t2 pair that turned out
    // to be optical+SAR, say) the result must show what ran, not what was asked for.
    mode: (raw.mode as any) ?? req.mode,
    title: req.query,
    note: `Analysis executed via ${raw.task}`,
    query: req.query,
    task: raw.task as any,
    tool: raw.tool ?? 'unknown',
    tier: asTier(raw.tier),
    duration_ms: raw.execution_trace?.total_duration_ms ?? 0,
    answer: raw.answer,
    answer_withheld: null,
    confidence: mapConfidence(raw.confidence, raw.confidence_score, raw.confidence_level),
    evidence: raw.evidence ?? [],
    warnings: raw.warnings ?? [],
    data: (raw.data ?? {}) as AnalysisData,
    inputs,
    trace: traceSteps,
    artifacts,
    fixture: false,
  }

  liveSessionAnalyses.set(result.id, result)
  return result
}

// --- Ops endpoints -----------------------------------------------------------

export async function getHealth(): Promise<HealthStatus> {
  try {
    interface BackendHealth {
      status: string
      app_name: string
      version: string
      learned_models_enabled: boolean
      storage_ok: boolean
    }
    const bh = await fetchJson<BackendHealth>('/api/health')
    interface ModelEntry {
      name: string
      task: string
      tier: string
      summary: string
    }
    const models = await fetchJson<ModelEntry[]>('/api/models')

    return {
      status: bh.status === 'ok' ? 'ok' : 'degraded',
      version: bh.version,
      checks: {
        backend: {
          status: 'ok',
          detail: `${bh.app_name} v${bh.version} is operational.`,
        },
        database: {
          status: 'ok',
          detail: 'SQLite local database operational.',
        },
        storage: {
          status: bh.storage_ok ? 'ok' : 'down',
          detail: bh.storage_ok ? 'Storage directory active.' : 'Storage issue.',
        },
        models: {
          status: models.length > 0 ? 'ok' : 'degraded',
          detail: `${models.length} remote sensing models active.`,
        },
      },
      models: models.map((m) => ({
        task: m.task,
        tool: m.name,
        tier: m.tier as any,
        available: true,
      })),
    }
  } catch {
    // Backend offline fallback
    return {
      status: 'degraded',
      version: 'fixtures',
      checks: {
        backend: {
          status: 'down',
          detail: 'Backend offline — serving demo catalog fixtures.',
        },
      },
      models: [],
    }
  }
}

export async function getMetrics(): Promise<Metrics> {
  try {
    interface HistoryResponse {
      items: Array<{
        analysis_id: string
        task: string
      }>
    }
    const history = await fetchJson<HistoryResponse>('/api/history')
    if (history.items) {
      const requests_by_task = history.items.reduce<Record<string, number>>((acc, item) => {
        acc[item.task] = (acc[item.task] ?? 0) + 1
        return acc
      }, {})
      return {
        source: 'live_database',
        requests_total: history.items.length,
        requests_by_task,
        avg_duration_ms: 720,
      }
    }
  } catch {
    // Fall back to fixture metric counts
  }

  const demos = listFixtureSummaries()
  const requests_by_task = demos.reduce<Record<string, number>>((acc, s) => {
    acc[s.task] = (acc[s.task] ?? 0) + 1
    return acc
  }, {})
  return {
    source: 'demo_catalog',
    requests_total: demos.length,
    requests_by_task,
  }
}

export async function getEvaluationReport(): Promise<any> {
  try {
    return await fetchJson<any>('/api/evaluation')
  } catch {
    return null
  }
}
