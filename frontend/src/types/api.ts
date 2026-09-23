/**
 * SatQuery API contract — the single source of truth the frontend codes against
 * and the FastAPI backend must satisfy. Every interface here mirrors a real
 * payload produced by the analysis tools (see scripts/generate_frontend_fixtures.py
 * and the recorded fixtures in src/fixtures/*.json); the enum string values match
 * backend/app/core/types.py verbatim.
 *
 * Nothing in this file is invented: the shapes were transcribed from genuine tool
 * output, so `AnalysisResult` describes exactly what /api/analyze returns.
 */

// ---------------------------------------------------------------------------
// Enumerations (string values identical to the Python enums)
// ---------------------------------------------------------------------------
export const MODALITIES = ['optical', 'sar', 'unknown'] as const
export type Modality = (typeof MODALITIES)[number]

export const QUERY_TASKS = [
  'vqa',
  'captioning',
  'grounding',
  'object_detection',
  'segmentation',
  'change_detection',
  'change_vqa',
  'optical_sar_analysis',
  'land_cover_analysis',
  'report_generation',
  'unknown',
] as const
export type QueryTask = (typeof QUERY_TASKS)[number]

export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'insufficient'
export type ToolTier = 'preferred' | 'fallback' | 'classical'
export type StepStatus = 'ok' | 'warning' | 'failed' | 'skipped'
export type AnalysisStatus = 'success' | 'partial' | 'failed'

/** The four input configurations the assistant accepts (brief §1). */
export const INPUT_MODES = [
  'single_optical',
  'single_sar',
  'optical_sar_pair',
  'bitemporal_pair',
] as const
export type InputMode = (typeof INPUT_MODES)[number]

export type LandCoverClassName = 'water' | 'vegetation' | 'built_up' | 'bare_soil' | 'unclassified'
export type ScatteringRegimeName = 'smooth' | 'diffuse' | 'double_bounce' | 'unclassified'
export type ChangeType =
  | 'water_gain'
  | 'water_loss'
  | 'vegetation_gain'
  | 'vegetation_loss'
  | 'urban_expansion'
  | 'urban_loss'
  | 'spectral_only'
  | 'other'

// ---------------------------------------------------------------------------
// Confidence (evidence-based, never model-generated — brief §27)
// ---------------------------------------------------------------------------
export type ConfidenceFactorKind = 'contributor' | 'gate'

export interface ConfidenceFactor {
  name: string
  value: number
  weight: number
  kind: ConfidenceFactorKind
  reason: string
}

export interface ConfidenceReport {
  level: ConfidenceLevel
  score: number
  sufficient: boolean
  limiting_factor: string | null
  reasons: string[]
  factors: ConfidenceFactor[]
}

// ---------------------------------------------------------------------------
// Geospatial framing + rendered artifacts
// ---------------------------------------------------------------------------
export interface Wgs84Bounds {
  south: number
  west: number
  north: number
  east: number
}

export interface LegendEntry {
  label: string
  color: string
  key: string
}

export type ArtifactKind =
  'base' | 'class_map' | 'regime_map' | 'change_mask' | 'grounding' | 'overlay'

export interface Artifact {
  /** Known kinds above; kept open for forward-compatibility. */
  kind: ArtifactKind | (string & {})
  label: string
  url: string
  /** Present only for georeferenced inputs; null => plain pixel viewer (§8). */
  bounds_wgs84: Wgs84Bounds | null
  legend: LegendEntry[]
}

// ---------------------------------------------------------------------------
// Execution trace (concise, no chain-of-thought — brief §5)
// ---------------------------------------------------------------------------
export type TraceStepName = 'classify' | 'validate' | 'preprocess' | 'execute' | 'score' | 'verify'

export interface TraceStep {
  step: TraceStepName | (string & {})
  tool: string | null
  status: StepStatus
  /**
   * Measured wall-clock milliseconds, or `null` for a step the backend does not time separately
   * (mode normalisation is counted inside routing). A step that finished in under half a millisecond
   * measures 0 — which is a result, not a missing value.
   */
  duration_ms: number | null
  summary: string
  parameters: Record<string, unknown>
  warnings: string[]
}

// ---------------------------------------------------------------------------
// Per-input raster metadata (RasterMetadata.to_dict + modality/roles/wgs84)
// ---------------------------------------------------------------------------
export interface InputMeta {
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
  pixel_size: [number, number] | null
  georeferenced: boolean
  /**
   * The file's own band names, positionally. An entry is null for a band the file does not name (a
   * plain PNG names none); the position is kept so a name cannot slide onto the wrong band.
   */
  band_descriptions: Array<string | null>
  decimation: number
  modality: Modality
  band_roles: string[]
  bounds_wgs84: Wgs84Bounds | null
}

// ---------------------------------------------------------------------------
// Per-task `data` payloads
// ---------------------------------------------------------------------------
export interface IndexSummary {
  name: string
  formula: string
  bands_used: string[]
  valid_fraction: number
  mean: number
  min: number
  max: number
}

export interface LandCoverClassStat {
  class: string
  pixel_count: number
  fraction: number
  percentage: number
  area_m2: number
  mean_evidence: number
}

export interface LandCoverData {
  method: string
  indices_used: string[]
  separability: number
  classes: LandCoverClassStat[]
  dominant: LandCoverClassStat
  warnings: string[]
  index_summaries: Record<string, IndexSummary>
}

export interface GroundingQuery {
  raw: string
  target: string | null
  target_phrase: string | null
  sector: string | null
  sector_phrase: string | null
  sector_interpretation: string | null
  min_area_m2: number | null
  max_area_m2: number | null
  min_pixels: number | null
  max_pixels: number | null
  superlative: string | null
  limit: number | null
  wants_count: boolean
  wants_area: boolean
}

export interface GroundingRegion {
  id: number
  pixel_area: number
  area_m2: number
  bbox_pixel: number[]
  bbox_geo: number[] | null
  polygon_pixel: number[][]
  polygon_geo: number[][] | null
  centroid_geo: number[] | null
  score: number
  sector_overlap: number | null
}

export interface GroundingEvidence {
  classification_method: string
  indices_used: string[]
  class_separability: number
  region_score_index: string | null
  candidate_regions: number
  min_region_pixels: number
  min_region_area_m2: number
  sector_selection: unknown
}

export interface GroundingData {
  query: GroundingQuery
  target: string
  count: number | null
  count_reportable: boolean
  count_caveat: string | null
  region_semantics: string
  source: string
  selected_pixels: number
  selected_area_m2: number
  class_pixels: number
  class_coverage_fraction: number
  georeferenced: boolean
  regions: GroundingRegion[]
  excluded: Record<string, unknown>
  evidence: GroundingEvidence
  warnings: string[]
}

export interface SarRegime {
  regime: string
  candidate_surfaces: string[]
  pixel_count: number
  fraction: number
  percentage: number
  /** Null when the scene carries no pixel size, so only the pixel count is measured. */
  area_m2: number | null
  mean_backscatter_db: number
}

export interface SpeckleStats {
  filter: string
  window: number
  estimated_enl: number
  input_cv: number
  output_cv: number
  speckle_reduction_factor: number
  warnings: string[]
}

export interface SarHistogram {
  bins: number[]
  counts: number[]
  /**
   * A raster with no finite pixels summarises as `{bins: [], counts: [], mean: null, std: null}` —
   * the percentiles are absent entirely — so every figure here is optional.
   */
  mean: number | null
  std: number | null
  p05?: number | null
  p95?: number | null
}

export interface SarData {
  polarization: string
  /**
   * The Otsu cuts, each null when that population could not be separated (`method: 'refused'` when
   * neither could). Reading them as plain numbers crashed the result route — see `formatSarThresholds`.
   */
  thresholds_db: { smooth: number | null; bright: number | null; method: string }
  mode_prominence: { smooth: number; bright: number; threshold: number }
  separated_regimes: string[]
  regimes: SarRegime[]
  speckle: SpeckleStats
  histogram: SarHistogram
  warnings: string[]
}

export interface Registration {
  offset_x_px: number
  offset_y_px: number
  offset_magnitude_px: number
  phase_response: number
  ncc: number
  score: number
  warnings: string[]
}

export interface FusionClassStat {
  class: string
  pixel_count: number
  fraction: number
  percentage: number
  area_m2: number
  corroborated_fraction: number
  /** e.g. { both: 180336, conflict: 2787 } */
  evidence: Record<string, number>
}

export interface FusionConflict {
  optical_class: string
  sar_regime: string
  pixel_count: number
  fraction: number
  area_m2: number
  resolved_as: string
  /** which sensor decided: 'sar' | 'optical' */
  arbiter: string
  reason: string
  radar_candidate_surfaces: string[]
}

export interface FusionData {
  method: string
  classes: FusionClassStat[]
  dominant_class: string
  agreement_fraction: number
  corroborated_fraction: number
  evidence_breakdown: Record<string, number>
  evidence_fractions: Record<string, number>
  conflicts: FusionConflict[]
  registration: Registration
  optical_only: { method: string; indices_used: string[]; classes: string[] }
  sar_only: { polarization: string; separated_regimes: string[] }
  warnings: string[]
}

export interface ChangeTransition {
  from: string
  to: string
  change_type: ChangeType | (string & {})
  pixel_count: number
  fraction_of_change: number
  percentage_of_change: number
  /** Null for a non-georeferenced pair: pixel counts are known, ground area is not. */
  area_m2: number | null
  mean_magnitude: number
  spectral_agreement: number
  /**
   * Whether this transition may be *named* as a land-cover conversion (brief §8/§9).
   *
   * Decided in the backend from the registration gate and the independent spectral detector's
   * agreement, and carried per row so the renderer never re-derives the rule. A transition with
   * `false` here is a measured pixel population whose interpretation is withheld — it must never
   * be presented as an observed conversion.
   */
  semantically_reportable: boolean
}

export interface ChangeClassDelta {
  class: string
  pixels_before: number
  pixels_after: number
  pixel_delta: number
  // Null whenever the pair carries no pixel size. `pixel_delta` and `relative_delta` are measured
  // either way, so the direction of a class change is always known even when its area is not.
  area_m2_before: number | null
  area_m2_after: number | null
  area_m2_delta: number | null
  relative_delta: number | null
}

export interface ChangeIndexDelta {
  index: string
  mean_before: number
  mean_after: number
  mean_delta_overall: number
  mean_delta_in_change: number
  increase_fraction: number
  direction: string
}

export interface ChangeEvidenceSplit {
  corroborated_pixels: number
  spectral_only_pixels: number
  class_only_pixels: number
  corroborated_fraction: number
}

/**
 * The explicit verdict on whether the pair is aligned well enough for semantic claims (§5).
 *
 * Separate from {@link Registration}: that says how well the dates line up, this says what may
 * therefore be claimed. Both sides of each comparison are carried so a refusal can be *shown*.
 */
export interface ChangeRegistrationGate {
  passed: boolean
  /** Empty when passed; otherwise one measured sentence per failing criterion. */
  reasons: string[]
  /** Keys: `offset_magnitude_px`, `ncc`, `score`. */
  measured: Record<string, number>
  /** Keys: `max_offset_px`, `min_ncc`, `min_score`. */
  thresholds: Record<string, number>
}

/** The model-inferred half of the result: what may be concluded, or why nothing may be (§9). */
export interface ChangeSemantic {
  /** False when the gate failed or no transition is corroborated enough to name. */
  supported: boolean
  min_spectral_agreement: number
  /** The single conversion that may be asserted, or null when none may be. */
  transition: ChangeTransition | null
  /** Present exactly when `supported` is false — the measured reason for the refusal. */
  withheld_reason: string | null
}

/** What co-registration actually did about a misalignment, and what it measurably achieved (§4). */
export interface ChangeCoregistration {
  /** `none`, `pyramid-phase-correlation`, `orb-translation`, or `rejected`. */
  method: string
  applied: boolean
  shift_x_px: number
  shift_y_px: number
  initial: Registration
  final: Registration
  score_improvement: number
  /** Every method evaluated, including the ones rejected by measurement. */
  candidates: Array<Record<string, unknown>>
  warnings: string[]
}

/**
 * How a change *question* was read, present only for the change-VQA tool (brief §7).
 *
 * Carried so a misread question is visible on the result page rather than only implied by an
 * answer that quietly addresses something else.
 */
export interface ChangeQuestion {
  /** The question as the user asked it. */
  asked: string
  /**
   * The reading that produced the answer — e.g. `net change in water`, `changed extent`,
   * `general change summary`, `no-change`.
   */
  interpreted_as: string
}

export interface ChangeData {
  method: string
  detectors: string[]
  features_used: string[]
  threshold: number
  threshold_method: string
  changed_pixels: number
  valid_pixels: number
  changed_fraction: number
  changed_percentage: number
  /** Null for a non-georeferenced pair — the changed *pixel* count is still measured. */
  changed_area_m2: number | null
  bimodality: number
  threshold_quality: number
  evidence_split: ChangeEvidenceSplit
  dominant_transition: ChangeTransition | null
  transitions: ChangeTransition[]
  class_deltas: ChangeClassDelta[]
  index_deltas: ChangeIndexDelta[]
  registration: Registration
  registration_gate: ChangeRegistrationGate
  semantic: ChangeSemantic
  /** Absent when the pair needed no correction attempt at all. */
  coregistration?: ChangeCoregistration
  /** Present only for `change-vqa-cv`: change *detection* answers no question. */
  question?: ChangeQuestion
  warnings: string[]
}

export type AnalysisData = LandCoverData | GroundingData | SarData | FusionData | ChangeData

// ---------------------------------------------------------------------------
// The envelope
// ---------------------------------------------------------------------------
export interface AnalysisResult {
  /** Report identifier, format SAT-2026-000123 (brief §13). */
  id: string
  request_id: string
  created_at: string
  status: AnalysisStatus
  mode: InputMode | (string & {})
  title: string
  note: string
  query: string
  task: QueryTask
  tool: string
  tier: ToolTier
  duration_ms: number
  /** The released answer, or the §28 decline sentence when withheld. */
  answer: string
  /** When evidence is insufficient, the phrasing that was withheld (else null). */
  answer_withheld: string | null
  confidence: ConfidenceReport
  evidence: string[]
  warnings: string[]
  data: AnalysisData
  inputs: InputMeta[]
  trace: TraceStep[]
  artifacts: Artifact[]
  /** True when served from a recorded fixture rather than a live backend run. */
  fixture?: boolean
}

/** Row shape in fixtures/index.json and GET /api/analyses. */
export interface AnalysisSummary {
  id: string
  title: string
  mode: InputMode | (string & {})
  task: QueryTask
  query: string
  created_at: string
  status: AnalysisStatus
  confidence: ConfidenceLevel
  answer: string
  /**
   * True when this row is a shipped recording rather than a live backend run.
   *
   * The listing merges both, and the two id namespaces overlap — the backend mints ids from the same
   * `SAT-2026-NNNNNN` sequence the recordings use — so a link built from the id alone is ambiguous.
   * This flag is what lets a recorded row link to its recording and a live row link to the live run.
   */
  fixture?: boolean
}

// ---------------------------------------------------------------------------
// Ops endpoints (brief §30) — fields optional so the UI degrades gracefully.
// ---------------------------------------------------------------------------
export interface HealthCheck {
  status: 'ok' | 'degraded' | 'down' | (string & {})
  detail?: string
}

export interface ModelRegistryEntry {
  task: string
  tool: string
  tier: ToolTier | (string & {})
  available: boolean
}

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'down' | (string & {})
  version?: string
  uptime_s?: number
  checks?: Record<string, HealthCheck>
  models?: ModelRegistryEntry[]
}

export interface Metrics {
  requests_total?: number
  requests_succeeded?: number
  requests_partial?: number
  requests_failed?: number
  requests_by_task?: Record<string, number>
  avg_duration_ms?: number
  p95_duration_ms?: number
  uptime_s?: number
  [key: string]: unknown
}

/** Uniform error envelope (brief §29). Never carries a stack trace. */
export interface ApiErrorEnvelope {
  success: false
  error_code: string
  message: string
  request_id: string
  recoverable: boolean
}

// ---------------------------------------------------------------------------
// Type guards — narrow AnalysisData by distinctive keys. The `task` field alone
// cannot discriminate (single-SAR and fusion both report `optical_sar_analysis`),
// so we key off structure.
// ---------------------------------------------------------------------------
export function isLandCoverData(d: AnalysisData): d is LandCoverData {
  return 'separability' in d && 'index_summaries' in d
}

export function isGroundingData(d: AnalysisData): d is GroundingData {
  return 'regions' in d && 'region_semantics' in d
}

export function isSarData(d: AnalysisData): d is SarData {
  return 'polarization' in d && 'regimes' in d && 'speckle' in d
}

export function isFusionData(d: AnalysisData): d is FusionData {
  return 'conflicts' in d && 'evidence_breakdown' in d
}

export function isChangeData(d: AnalysisData): d is ChangeData {
  return 'transitions' in d && 'detectors' in d
}
