/**
 * Report model — the single source of truth for the printable report (brief §13).
 *
 * `buildReportModel` turns a real `AnalysisResult` into a structured, display-ready
 * outline (sections → typed blocks). The same model is rendered two ways: on-screen
 * by <ReportDocument> and as a standalone downloadable file by renderReportHtml().
 * Keeping the *decisions* (what to show, the §28 count call, §27 phrasing) here — not
 * in a renderer — is what stops the two outputs from drifting, and makes those
 * decisions unit-testable.
 *
 * Every figure comes straight from the tool output. Nothing is invented: where the
 * evidence does not support a claim (e.g. an unreliable object count) the model emits
 * an honest caveat instead of a number.
 */
import type {
  AnalysisData,
  AnalysisResult,
  AnalysisStatus,
  Artifact,
  ChangeData,
  ConfidenceLevel,
  FusionData,
  GroundingData,
  LandCoverData,
  QueryTask,
  SarData,
  ToolTier,
} from '@/types/api'
import {
  isChangeData,
  isFusionData,
  isGroundingData,
  isLandCoverData,
  isSarData,
} from '@/types/api'
import {
  formatArea,
  formatExtent,
  formatFractionPct,
  formatInt,
  formatPct,
  formatSarThresholds,
  formatSignedExtent,
  formatSignedFractionPct,
  formatStepDuration,
} from '@/lib/format'
import { changeTypeLabel, prettifyKey } from '@/lib/classes'

// ---------------------------------------------------------------------------
// Model types
// ---------------------------------------------------------------------------
export type ReportTone = 'info' | 'warning' | 'success' | 'muted'

export interface ReportField {
  label: string
  value: string
  hint?: string
}

export interface ReportTableColumn {
  key: string
  label: string
  align?: 'left' | 'right'
}

export interface ReportTable {
  columns: ReportTableColumn[]
  rows: Record<string, string>[]
  caption?: string
}

export type ReportBlock =
  | { kind: 'fields'; fields: ReportField[] }
  | { kind: 'table'; table: ReportTable }
  | { kind: 'text'; text: string }
  | { kind: 'list'; items: string[]; ordered?: boolean }
  | { kind: 'callout'; tone: ReportTone; title?: string; body: string }

export interface ReportSection {
  id: string
  title: string
  blocks: ReportBlock[]
}

export interface ReportModel {
  id: string
  requestId: string
  createdAtIso: string
  generatedAtIso: string
  title: string
  note: string
  status: AnalysisStatus
  mode: string
  task: QueryTask
  tool: string
  tier: ToolTier
  durationMs: number
  fixture: boolean
  query: string
  answer: string
  /** True when §28 withheld the answer outright — `answer` is then the refusal itself. */
  withheld: boolean
  /**
   * True when the answer states real measurements that the evidence does not support concluding
   * from. Distinct from `withheld`: the figures are measured and reportable, the *conclusion* is
   * not. The two used to be one `declined` flag, which made the report print "Insufficient
   * evidence for a reliable conclusion." directly above a measured answer — declining and
   * concluding in the same breath.
   */
  unreliable: boolean
  /** The weak-evidence caveat, phrased once here so screen and file say the same thing. */
  caveat: { title: string; body: string } | null
  /** The phrasing that was withheld because evidence was insufficient (else null). */
  withheldDraft: string | null
  confidenceLevel: ConfidenceLevel
  /** Rendered natively by each presenter (images on screen / in the file). */
  artifacts: Artifact[]
  sections: ReportSection[]
  disclaimer: string
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
function yesNo(value: boolean): string {
  return value ? 'Yes' : 'No'
}

function dash(value: string | null | undefined): string {
  return value == null || value === '' ? '—' : value
}

/** Sort a copy of `rows` by a numeric key, descending — never mutates the input. */
function byDesc<T>(rows: readonly T[], key: (row: T) => number): T[] {
  return [...rows].sort((a, b) => key(b) - key(a))
}

// ---------------------------------------------------------------------------
// Per-task findings
// ---------------------------------------------------------------------------
function landCoverBlocks(d: LandCoverData): ReportBlock[] {
  const blocks: ReportBlock[] = [
    {
      kind: 'fields',
      fields: [
        { label: 'Method', value: d.method },
        {
          label: 'Class separability',
          value: formatFractionPct(d.separability),
          hint: 'higher = more distinct classes',
        },
        {
          label: 'Dominant class',
          value: prettifyKey(d.dominant.class),
          hint: formatPct(d.dominant.percentage),
        },
        { label: 'Indices used', value: d.indices_used.join(', ') || '—' },
      ],
    },
    {
      kind: 'table',
      table: {
        caption: 'Per-class area (of classified pixels)',
        columns: [
          { key: 'class', label: 'Class' },
          { key: 'share', label: 'Share', align: 'right' },
          { key: 'area', label: 'Area', align: 'right' },
          { key: 'pixels', label: 'Pixels', align: 'right' },
          { key: 'evidence', label: 'Mean evidence', align: 'right' },
        ],
        rows: byDesc(d.classes, (c) => c.fraction).map((c) => ({
          class: prettifyKey(c.class),
          share: formatPct(c.percentage),
          area: formatArea(c.area_m2),
          pixels: formatInt(c.pixel_count),
          evidence: c.mean_evidence.toFixed(2),
        })),
      },
    },
  ]

  const indexRows = Object.values(d.index_summaries)
  if (indexRows.length > 0) {
    blocks.push({
      kind: 'table',
      table: {
        caption: 'Spectral indices',
        columns: [
          { key: 'name', label: 'Index' },
          { key: 'formula', label: 'Formula' },
          { key: 'mean', label: 'Mean', align: 'right' },
          { key: 'range', label: 'Range', align: 'right' },
          { key: 'valid', label: 'Valid', align: 'right' },
        ],
        rows: indexRows.map((s) => ({
          name: d.indices_used.includes(s.name) ? `${s.name} (used)` : s.name,
          formula: s.formula,
          mean: s.mean.toFixed(3),
          range: `${s.min.toFixed(2)} … ${s.max.toFixed(2)}`,
          valid: formatFractionPct(s.valid_fraction),
        })),
      },
    })
  }
  return blocks
}

function groundingBlocks(d: GroundingData): ReportBlock[] {
  const q = d.query
  const blocks: ReportBlock[] = [
    {
      kind: 'fields',
      fields: [
        { label: 'Interpreted target', value: dash(q.target ?? d.target) },
        { label: 'Target phrase', value: dash(q.target_phrase) },
        { label: 'Sector', value: dash(q.sector_interpretation ?? q.sector) },
        { label: 'Query wants a count', value: yesNo(q.wants_count) },
        { label: 'Query wants an area', value: yesNo(q.wants_area) },
      ],
    },
    {
      kind: 'fields',
      fields: [
        { label: 'Selected area', value: formatArea(d.selected_area_m2) },
        { label: 'Selected pixels', value: formatInt(d.selected_pixels) },
        {
          label: 'Class coverage',
          value: formatFractionPct(d.class_coverage_fraction),
          hint: 'share of the scene in this class',
        },
        { label: 'Regions mapped', value: formatInt(d.regions.length) },
      ],
    },
  ]

  // §28: only state a count when it is genuinely reportable. Otherwise explain
  // why the number is not reliable — never assert the unreliable figure.
  if (d.count_reportable && d.count !== null) {
    blocks.push({
      kind: 'callout',
      tone: 'success',
      title: 'Count',
      body: `${formatInt(d.count)} discrete region(s) — this count is reportable for this input.`,
    })
  } else {
    const patches =
      d.count !== null
        ? ` (${formatInt(d.count)} connected patches were mapped, but this is not a reliable object count.)`
        : ''
    blocks.push({
      kind: 'callout',
      tone: 'warning',
      title: 'Count not reportable',
      body: `${dash(d.count_caveat) === '—' ? 'A reliable count cannot be stated from this image alone.' : d.count_caveat}${patches}`,
    })
  }

  if (d.region_semantics) {
    blocks.push({ kind: 'text', text: d.region_semantics })
  }
  return blocks
}

function sarBlocks(d: SarData): ReportBlock[] {
  return [
    {
      kind: 'fields',
      fields: [
        { label: 'Polarization', value: d.polarization },
        {
          label: 'Speckle filter',
          value: d.speckle.filter,
          hint: `${d.speckle.window}×${d.speckle.window} window`,
        },
        {
          label: 'Equivalent looks (ENL)',
          value: d.speckle.estimated_enl.toFixed(1),
        },
        {
          label: 'Speckle reduction',
          value: `${d.speckle.speckle_reduction_factor.toFixed(2)}×`,
          hint: `CV ${d.speckle.input_cv.toFixed(2)} → ${d.speckle.output_cv.toFixed(2)}`,
        },
      ],
    },
    {
      kind: 'table',
      table: {
        caption: 'Scattering regimes',
        columns: [
          { key: 'regime', label: 'Regime' },
          { key: 'share', label: 'Share', align: 'right' },
          { key: 'db', label: 'Mean backscatter', align: 'right' },
          { key: 'surfaces', label: 'Candidate surfaces' },
        ],
        rows: byDesc(d.regimes, (r) => r.fraction).map((r) => ({
          regime: prettifyKey(r.regime),
          share: formatPct(r.percentage),
          db: `${r.mean_backscatter_db.toFixed(1)} dB`,
          surfaces: r.candidate_surfaces.join(', ') || '—',
        })),
      },
    },
    {
      kind: 'text',
      text: `${formatSarThresholds(d.thresholds_db)} SAR reports scattering behaviour, not material identity — surfaces are candidates, not confirmed classes.`,
    },
  ]
}

function registrationBlock(reg: FusionData['registration']): ReportBlock {
  return {
    kind: 'fields',
    fields: [
      {
        label: 'Pixel offset',
        value: `${reg.offset_magnitude_px.toFixed(2)} px`,
        hint: `Δx ${reg.offset_x_px.toFixed(1)}, Δy ${reg.offset_y_px.toFixed(1)}`,
      },
      { label: 'NCC', value: reg.ncc.toFixed(3) },
      { label: 'Registration score', value: formatFractionPct(reg.score) },
      { label: 'Phase response', value: reg.phase_response.toFixed(3) },
    ],
  }
}

function fusionBlocks(d: FusionData): ReportBlock[] {
  const blocks: ReportBlock[] = [
    {
      kind: 'fields',
      fields: [
        { label: 'Method', value: d.method },
        { label: 'Dominant class', value: prettifyKey(d.dominant_class) },
        {
          label: 'Sensor agreement',
          value: formatFractionPct(d.agreement_fraction),
          hint: 'optical & SAR concur',
        },
        {
          label: 'Corroborated',
          value: formatFractionPct(d.corroborated_fraction),
          hint: 'backed by both sensors',
        },
      ],
    },
    {
      kind: 'table',
      table: {
        caption: 'Fused land-cover classes',
        columns: [
          { key: 'class', label: 'Class' },
          { key: 'share', label: 'Share', align: 'right' },
          { key: 'area', label: 'Area', align: 'right' },
          { key: 'corroborated', label: 'Corroborated', align: 'right' },
        ],
        rows: byDesc(d.classes, (c) => c.fraction).map((c) => ({
          class: prettifyKey(c.class),
          share: formatPct(c.percentage),
          area: formatArea(c.area_m2),
          corroborated: formatFractionPct(c.corroborated_fraction),
        })),
      },
    },
  ]

  // Cross-modal conflicts (brief §4): where optical and SAR disagree, show how it
  // was arbitrated. This is core evidence, not a footnote.
  if (d.conflicts.length > 0) {
    blocks.push({
      kind: 'table',
      table: {
        caption: 'Cross-modal conflicts (optical vs SAR)',
        columns: [
          { key: 'optical', label: 'Optical' },
          { key: 'sar', label: 'SAR regime' },
          { key: 'resolved', label: 'Resolved as' },
          { key: 'arbiter', label: 'Decided by' },
          { key: 'area', label: 'Area', align: 'right' },
        ],
        rows: byDesc(d.conflicts, (c) => c.fraction).map((c) => ({
          optical: prettifyKey(c.optical_class),
          sar: prettifyKey(c.sar_regime),
          resolved: prettifyKey(c.resolved_as),
          arbiter: c.arbiter.toUpperCase(),
          area: formatArea(c.area_m2),
        })),
      },
    })
    blocks.push({
      kind: 'list',
      items: byDesc(d.conflicts, (c) => c.fraction).map(
        (c) =>
          `${prettifyKey(c.optical_class)} vs ${prettifyKey(c.sar_regime)} → ${prettifyKey(c.resolved_as)}: ${c.reason}`,
      ),
    })
  } else {
    blocks.push({
      kind: 'text',
      text: 'No material optical/SAR conflicts: the two sensors agree across the scene.',
    })
  }

  blocks.push(registrationBlock(d.registration))
  return blocks
}

function changeBlocks(d: ChangeData): ReportBlock[] {
  const georeferenced = d.changed_area_m2 !== null
  const blocks: ReportBlock[] = []

  // Change VQA only. Printing the reading alongside the question keeps a misread question visible
  // in the downloaded report rather than only in an answer that addresses something else.
  if (d.question) {
    blocks.push({
      kind: 'fields',
      fields: [
        { label: 'Question asked', value: d.question.asked },
        { label: 'Read as', value: prettifyKey(d.question.interpreted_as) },
      ],
    })
  }

  blocks.push(
    {
      kind: 'text',
      text:
        'Measured pixel difference — counted directly from the two dates. This is an ' +
        'observation and does not depend on what the change means.',
    },
    {
      kind: 'fields',
      fields: [
        { label: 'Method', value: d.method },
        {
          // A non-georeferenced pair carries no pixel size, so the ground area is null while the
          // pixel count is measured. Headlining the area printed "—" over a measured 100,620 px.
          label: georeferenced ? 'Changed area' : 'Changed extent',
          value: formatExtent(d.changed_area_m2, d.changed_pixels),
          hint: georeferenced
            ? `${formatInt(d.changed_pixels)} px`
            : 'pixel count — no georeference, so no ground area',
        },
        { label: 'Changed', value: formatPct(d.changed_percentage) },
        {
          label: 'Threshold',
          value: d.threshold.toFixed(3),
          hint: d.threshold_method,
        },
      ],
    },
  )

  // The registration gate verdict, with both sides of every comparison, so a refusal in the
  // printed report is demonstrable rather than asserted (brief §5, §9).
  const gate = d.registration_gate
  if (gate) {
    blocks.push({
      kind: 'fields',
      fields: [
        {
          label: 'Registration gate',
          value: gate.passed ? 'Passed' : 'Failed',
          hint: gate.passed ? 'semantic claims permitted' : 'semantic claims withheld',
        },
        {
          label: 'Pixel offset',
          value: `${(gate.measured.offset_magnitude_px ?? 0).toFixed(2)} px`,
          hint: `needs ≤ ${(gate.thresholds.max_offset_px ?? 0).toFixed(2)} px`,
        },
        {
          label: 'Structural agreement',
          value: (gate.measured.ncc ?? 0).toFixed(3),
          hint: `needs ≥ ${(gate.thresholds.min_ncc ?? 0).toFixed(2)} NCC`,
        },
        {
          label: 'Registration score',
          value: (gate.measured.score ?? 0).toFixed(3),
          hint: `needs ≥ ${(gate.thresholds.min_score ?? 0).toFixed(2)}`,
        },
      ],
    })
    if (gate.reasons.length > 0) {
      blocks.push({ kind: 'list', items: gate.reasons })
    }
  }

  // The model-inferred half. The old block was an unconditional "Dominant change" callout, which
  // read as an observed conversion even when every transition had been withheld.
  const semantic = d.semantic
  const named = semantic?.supported ? semantic.transition : null
  if (named) {
    blocks.push({
      kind: 'callout',
      tone: 'info',
      title: 'Model-inferred land-cover conversion',
      body: `${prettifyKey(named.from)} → ${prettifyKey(named.to)} (${changeTypeLabel(named.change_type)}) — ${formatPct(named.percentage_of_change)} of the changed area (${formatExtent(named.area_m2, named.pixel_count)}). Named because the independent spectral detector corroborates ${formatFractionPct(named.spectral_agreement)} of its pixels, at or above the ${formatFractionPct(semantic.min_spectral_agreement)} required.`,
    })
  } else {
    const dominant = d.dominant_transition
    const reason =
      semantic?.withheld_reason ??
      'The evidence does not support attributing the measured difference to a named land-cover conversion.'
    const largest = dominant
      ? ` The largest changed pixel population is ${prettifyKey(dominant.from)} → ${prettifyKey(dominant.to)} — ${formatPct(dominant.percentage_of_change)} of the changed area (${formatExtent(dominant.area_m2, dominant.pixel_count)}), ${formatFractionPct(dominant.spectral_agreement)} spectral agreement. That is a measurement of where the two classifications differ, not an observed conversion.`
      : ''
    blocks.push({
      kind: 'callout',
      tone: 'warning',
      title: 'No land-cover conversion is claimed',
      body: `${reason}${largest}`,
    })
  }

  blocks.push({
    kind: 'table',
    table: {
      caption: 'Measured transitions (share of changed area)',
      columns: [
        { key: 'transition', label: 'Transition' },
        { key: 'type', label: 'Type' },
        { key: 'share', label: 'Share', align: 'right' },
        { key: 'area', label: georeferenced ? 'Area' : 'Extent', align: 'right' },
        { key: 'agreement', label: 'Spectral agreement', align: 'right' },
        { key: 'reportable', label: 'Reportable as conversion' },
      ],
      rows: byDesc(d.transitions, (t) => t.fraction_of_change).map((t) => ({
        transition: `${prettifyKey(t.from)} → ${prettifyKey(t.to)}`,
        type: changeTypeLabel(t.change_type),
        share: formatPct(t.percentage_of_change),
        area: formatExtent(t.area_m2, t.pixel_count),
        agreement: formatFractionPct(t.spectral_agreement),
        reportable: t.semantically_reportable ? 'Yes' : 'Withheld',
      })),
    },
  })

  if (d.class_deltas.length > 0) {
    blocks.push({
      kind: 'table',
      table: {
        caption: georeferenced
          ? 'Class area change'
          : 'Class extent change (pixel counts — the pair is not georeferenced)',
        columns: [
          { key: 'class', label: 'Class' },
          { key: 'before', label: 'Before', align: 'right' },
          { key: 'after', label: 'After', align: 'right' },
          { key: 'delta', label: georeferenced ? 'Δ area' : 'Δ extent', align: 'right' },
          { key: 'relative', label: 'Relative', align: 'right' },
        ],
        /*
         * Every cell here falls back to the measured pixel count, and the Δ takes its sign from
         * `pixel_delta`, which is measured for every pair. This row used to read
         * "Water — — +0 m² 115.3%": the delta came from `area_m2_delta >= 0 ? '+' : '−'` with
         * `formatArea(Math.abs(area_m2_delta))`, and since `null >= 0` is true and `Math.abs(null)`
         * is 0, a class that more than doubled was reported as a measured zero change.
         */
        rows: d.class_deltas.map((c) => ({
          class: prettifyKey(c.class),
          before: formatExtent(c.area_m2_before, c.pixels_before),
          after: formatExtent(c.area_m2_after, c.pixels_after),
          delta: formatSignedExtent(c.area_m2_delta, c.pixel_delta),
          relative: formatSignedFractionPct(c.relative_delta),
        })),
      },
    })
  }

  const split = d.evidence_split
  blocks.push({
    kind: 'fields',
    fields: [
      {
        label: 'Corroborated change',
        value: formatFractionPct(split.corroborated_fraction),
        hint: `${formatInt(split.corroborated_pixels)} px, both detectors`,
      },
      {
        label: 'Spectral-only',
        value: formatInt(split.spectral_only_pixels),
        hint: 'px flagged by magnitude only',
      },
      {
        label: 'Class-only',
        value: formatInt(split.class_only_pixels),
        hint: 'px flagged by label change only',
      },
    ],
  })

  blocks.push(registrationBlock(d.registration))
  const coreg = d.coregistration
  if (coreg) {
    blocks.push({
      kind: 'fields',
      fields: [
        {
          label: 'Co-registration',
          value: coreg.applied ? coreg.method : `none applied (${coreg.method})`,
          hint: `${coreg.candidates.length} method(s) evaluated, best kept`,
        },
        {
          label: 'Applied shift',
          value: `${coreg.shift_x_px.toFixed(2)}, ${coreg.shift_y_px.toFixed(2)} px`,
        },
        {
          label: 'Score improvement',
          value: `${coreg.initial.score.toFixed(3)} → ${coreg.final.score.toFixed(3)}`,
          hint: `${coreg.score_improvement >= 0 ? '+' : ''}${coreg.score_improvement.toFixed(3)}`,
        },
      ],
    })
  }
  return blocks
}

/** Dispatch to the right findings builder, with an honest fallback. */
function findingsSection(data: AnalysisData): ReportSection {
  if (isLandCoverData(data)) {
    return { id: 'findings', title: 'Land-cover findings', blocks: landCoverBlocks(data) }
  }
  if (isGroundingData(data)) {
    return { id: 'findings', title: 'Grounding & localisation', blocks: groundingBlocks(data) }
  }
  if (isSarData(data)) {
    return { id: 'findings', title: 'SAR scattering findings', blocks: sarBlocks(data) }
  }
  if (isFusionData(data)) {
    return { id: 'findings', title: 'Optical + SAR fusion', blocks: fusionBlocks(data) }
  }
  if (isChangeData(data)) {
    return { id: 'findings', title: 'Change detection', blocks: changeBlocks(data) }
  }
  return {
    id: 'findings',
    title: 'Findings',
    blocks: [{ kind: 'text', text: 'No structured detail is available for this analysis.' }],
  }
}

// ---------------------------------------------------------------------------
// Cross-cutting sections
// ---------------------------------------------------------------------------
function confidenceSection(result: AnalysisResult): ReportSection {
  const c = result.confidence
  const blocks: ReportBlock[] = [
    {
      kind: 'fields',
      fields: [
        { label: 'Level', value: prettifyKey(c.level) },
        { label: 'Evidence score', value: formatFractionPct(c.score) },
        { label: 'Sufficient to answer', value: yesNo(c.sufficient) },
        {
          label: 'Limiting factor',
          value: c.limiting_factor ? prettifyKey(c.limiting_factor) : '—',
        },
      ],
    },
  ]
  if (c.reasons.length > 0) {
    blocks.push({ kind: 'list', items: c.reasons })
  }
  if (c.factors.length > 0) {
    blocks.push({
      kind: 'table',
      table: {
        caption: 'Evidence factors',
        columns: [
          { key: 'name', label: 'Factor' },
          { key: 'kind', label: 'Type' },
          { key: 'value', label: 'Value', align: 'right' },
          { key: 'weight', label: 'Weight', align: 'right' },
          { key: 'reason', label: 'Basis' },
        ],
        rows: c.factors.map((f) => ({
          name: prettifyKey(f.name),
          kind: f.kind === 'gate' ? 'Gate' : 'Contributor',
          value: formatFractionPct(f.value),
          weight: formatFractionPct(f.weight),
          reason: f.reason,
        })),
      },
    })
  }
  blocks.push({
    kind: 'callout',
    tone: 'muted',
    body: 'Confidence is computed from measured evidence — separability, corroboration, registration and input quality — not generated by a language model.',
  })
  return { id: 'confidence', title: 'Confidence assessment', blocks }
}

function inputsSection(result: AnalysisResult): ReportSection {
  return {
    id: 'inputs',
    title: result.inputs.length > 1 ? 'Inputs' : 'Input',
    blocks: [
      {
        kind: 'table',
        table: {
          columns: [
            { key: 'file', label: 'File' },
            { key: 'modality', label: 'Modality' },
            { key: 'size', label: 'Dimensions', align: 'right' },
            { key: 'bands', label: 'Bands', align: 'right' },
            { key: 'dtype', label: 'Type' },
            { key: 'crs', label: 'CRS' },
            { key: 'pixel', label: 'Pixel size', align: 'right' },
            { key: 'geo', label: 'Georef.' },
          ],
          rows: result.inputs.map((m) => ({
            file: m.filename,
            modality: prettifyKey(m.modality),
            size: `${m.width}×${m.height}`,
            bands: String(m.bands),
            dtype: m.dtype,
            crs: m.crs_epsg ? `EPSG:${m.crs_epsg}` : dash(m.crs),
            pixel: m.pixel_size ? `${m.pixel_size[0]} m` : '—',
            geo: yesNo(m.georeferenced),
          })),
        },
      },
    ],
  }
}

function traceSection(result: AnalysisResult): ReportSection {
  return {
    id: 'trace',
    title: 'Processing trace',
    blocks: [
      {
        kind: 'table',
        table: {
          columns: [
            { key: 'step', label: 'Step' },
            { key: 'tool', label: 'Tool' },
            { key: 'status', label: 'Status' },
            { key: 'duration', label: 'Duration', align: 'right' },
            { key: 'summary', label: 'Summary' },
          ],
          rows: result.trace.map((s) => ({
            step: prettifyKey(s.step),
            tool: s.tool ?? '—',
            status: s.status.toUpperCase(),
            duration: formatStepDuration(s.duration_ms),
            summary: s.summary,
          })),
        },
      },
      {
        kind: 'callout',
        tone: 'muted',
        body: 'This is the tool pipeline the assistant ran — classification, routing, execution and evidence checks. Internal model reasoning is not shown.',
      },
    ],
  }
}

function evidenceSection(result: AnalysisResult): ReportSection | null {
  const blocks: ReportBlock[] = []
  if (result.evidence.length > 0) {
    blocks.push({ kind: 'list', items: result.evidence })
  }
  if (result.warnings.length > 0) {
    blocks.push({
      kind: 'callout',
      tone: 'warning',
      title: 'Caveats',
      body: result.warnings.join(' '),
    })
  }
  if (blocks.length === 0) return null
  return { id: 'evidence', title: 'Evidence & caveats', blocks }
}

// ---------------------------------------------------------------------------
// Public builder
// ---------------------------------------------------------------------------
export function buildReportModel(result: AnalysisResult, generatedAtIso: string): ReportModel {
  const withheld = result.answer_withheld !== null
  const unreliable = !result.confidence.sufficient
  const { limiting_factor, reasons } = result.confidence

  // Named once, beside the measurements rather than over them, and saying which measurement was
  // the limiting one instead of leaving the reader to guess (§27: weak evidence is not a failure).
  const caveat =
    unreliable && !withheld
      ? {
          title: 'These measurements do not support a reliable conclusion',
          body: `${limiting_factor ? `Limiting factor: ${prettifyKey(limiting_factor)}. ` : ''}${
            reasons[0] ?? 'The measured evidence was too weak to state a finding.'
          } Treat the figures below as measurements of these inputs, not as a finding about the ground.`,
        }
      : null

  const sections: ReportSection[] = [findingsSection(result.data), confidenceSection(result)]
  const evidence = evidenceSection(result)
  if (evidence) sections.push(evidence)
  sections.push(inputsSection(result), traceSection(result))

  const fixtureNote = result.fixture
    ? ' This report reflects a recorded analysis run bundled with the prototype.'
    : ''

  return {
    id: result.id,
    requestId: result.request_id,
    createdAtIso: result.created_at,
    generatedAtIso,
    title: result.title,
    note: result.note,
    status: result.status,
    mode: result.mode,
    task: result.task,
    tool: result.tool,
    tier: result.tier,
    durationMs: result.duration_ms,
    fixture: Boolean(result.fixture),
    query: result.query,
    answer: result.answer,
    withheld,
    unreliable,
    caveat,
    withheldDraft: result.answer_withheld,
    confidenceLevel: result.confidence.level,
    artifacts: result.artifacts,
    sections,
    disclaimer: `Every figure in this report is computed from measured image evidence — spectral separability, sensor corroboration, co-registration quality and input quality — and is not generated by a language model. Where the available evidence does not support a reliable answer, the report says so explicitly rather than guessing.${fixtureNote}`,
  }
}
