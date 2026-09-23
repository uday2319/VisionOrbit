import type { AnalysisStatus, ConfidenceLevel, InputMode } from '@/types/api'
import type { BadgeProps } from '@/components/ui/badge'

type BadgeVariant = NonNullable<BadgeProps['variant']>

export const MODE_LABELS: Record<InputMode, string> = {
  single_optical: 'Single optical',
  single_sar: 'Single SAR',
  optical_sar_pair: 'Optical + SAR',
  bitemporal_pair: 'Bi-temporal',
}

export const TASK_LABELS: Record<string, string> = {
  vqa: 'Visual Q&A',
  captioning: 'Captioning',
  grounding: 'Grounding',
  object_detection: 'Object detection',
  segmentation: 'Segmentation',
  change_detection: 'Change detection',
  change_vqa: 'Change Q&A',
  optical_sar_analysis: 'Optical / SAR analysis',
  land_cover_analysis: 'Land-cover analysis',
  report_generation: 'Report',
  unknown: 'Unknown',
}

export function modeLabel(mode: string): string {
  return MODE_LABELS[mode as InputMode] ?? mode
}

export function taskLabel(task: string): string {
  return TASK_LABELS[task] ?? task
}

export function confidenceBadgeVariant(level: ConfidenceLevel): BadgeVariant {
  switch (level) {
    case 'high':
      return 'success'
    case 'medium':
      return 'info'
    case 'low':
      return 'warning'
    case 'insufficient':
      return 'secondary'
    default:
      return 'secondary'
  }
}

export function statusBadgeVariant(status: AnalysisStatus): BadgeVariant {
  switch (status) {
    case 'success':
      return 'success'
    case 'partial':
      return 'warning'
    case 'failed':
      return 'destructive'
    default:
      return 'secondary'
  }
}

const numberFmt = new Intl.NumberFormat(undefined)

export function formatNumber(value: number): string {
  return numberFmt.format(value)
}

export function formatInt(value: number): string {
  return numberFmt.format(Math.round(value))
}

/** Format a 0..1 fraction as a percentage string, e.g. 0.691 -> "69.1%". */
export function formatFractionPct(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`
}

/** Format an already-0..100 percentage value. */
export function formatPct(percentage: number, digits = 1): string {
  return `${percentage.toFixed(digits)}%`
}

/** Human-readable area from square metres (km² above 0.1 km², else m²). */
export function formatArea(m2: number | null | undefined): string {
  if (m2 === null || m2 === undefined || !Number.isFinite(m2)) return '—'
  const km2 = m2 / 1_000_000
  if (km2 >= 0.1) return `${km2.toFixed(2)} km²`
  return `${formatInt(m2)} m²`
}

function hasArea(m2: number | null | undefined): m2 is number {
  return m2 !== null && m2 !== undefined && Number.isFinite(m2)
}

/**
 * A measured extent: ground area when the input is georeferenced, else the pixel count.
 *
 * A non-georeferenced pair carries no pixel size, so every `area_m2` in its result is null while
 * every pixel count is measured. Rendering the area alone filled whole columns with "—" and threw
 * away the measurement that does exist — the report's transition table showed nine dashes under
 * "Area" beside nine measured pixel counts it never printed.
 */
export function formatExtent(area_m2: number | null | undefined, pixels: number): string {
  return hasArea(area_m2) ? formatArea(area_m2) : `${formatInt(pixels)} px`
}

/**
 * As `formatExtent`, carrying the sign of a change.
 *
 * Never returns a bare "0" for an unmeasured area: the report used to write
 * `area_m2_delta >= 0 ? '+' : '-'` and `Math.abs(area_m2_delta)`, and because `null >= 0` is true
 * and `Math.abs(null)` is 0 in JavaScript, a class with no known ground area rendered "+0 m²" — a
 * fabricated measurement of no change, for a class that had in fact more than doubled.
 */
export function formatSignedExtent(
  area_m2_delta: number | null | undefined,
  pixel_delta: number,
): string {
  const useArea = hasArea(area_m2_delta)
  const value = useArea ? area_m2_delta : pixel_delta
  const magnitude = useArea
    ? formatArea(Math.abs(area_m2_delta))
    : `${formatInt(Math.abs(pixel_delta))} px`
  if (value > 0) return `+${magnitude}`
  if (value < 0) return `-${magnitude}`
  return magnitude
}

/**
 * A 0..1 fraction as a *signed* percentage: 1.153 -> "+115.3%", -0.313 -> "-31.3%".
 *
 * The unsigned form made a gain and a loss look identical wherever the sign was carried by a
 * neighbouring column that could itself be "—".
 */
export function formatSignedFractionPct(fraction: number | null | undefined, digits = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return '—'
  const pct = (fraction * 100).toFixed(digits)
  return fraction > 0 ? `+${pct}%` : `${pct}%`
}

/**
 * The dB thresholds a SAR run actually resolved, as one sentence.
 *
 * Otsu returns a cut only for a population it can separate. A scene with no distinct
 * bright-scattering mode therefore reports `bright: null` — the backend says as much in its own
 * warnings ("No distinct bright-scattering population was found…") — and a scene with no smooth mode
 * reports `smooth: null`; when neither separates, `method` is `refused`. Both values used to be read
 * as plain numbers, so a single-SAR run over such a scene killed the whole result route in
 * `toFixed` ("Unexpected Application Error!") even though every other measurement in it was sound.
 * A threshold that was not resolved is now reported as not resolved, next to the reason.
 */
export function formatSarThresholds(t: {
  smooth: number | null | undefined
  bright: number | null | undefined
  method: string
}): string {
  const resolved = (v: number | null | undefined): v is number =>
    typeof v === 'number' && Number.isFinite(v)
  const head = `Backscatter thresholds (${t.method}):`
  const cuts: string[] = []
  if (resolved(t.smooth)) cuts.push(`smooth ≤ ${t.smooth.toFixed(1)} dB`)
  if (resolved(t.bright)) cuts.push(`bright ≥ ${t.bright.toFixed(1)} dB`)
  if (cuts.length === 2) return `${head} ${cuts.join(', ')}.`
  if (cuts.length === 0) {
    return `${head} none — no scattering population separated from this image, so no dB threshold is reported.`
  }
  const missing = resolved(t.smooth) ? 'bright' : 'smooth'
  return `${head} ${cuts[0]}. No distinct ${missing}-scattering population was separated, so no ${missing} threshold is reported.`
}

export function formatDateTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function formatDate(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString(undefined, { dateStyle: 'medium' })
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`
}

/**
 * A trace step's duration, distinguishing "measured as zero" from "not measured".
 *
 * The API client used to substitute 25 ms (30 ms on the detail path) for any step reporting 0 or
 * null. Four of the six steps in a change run measure under half a millisecond and one — mode
 * normalisation — is not timed separately at all, so the trace showed four invented 25 ms bars and
 * summed to 479 ms against a measured total of 380 ms.
 */
export function formatStepDuration(ms: number | null): string {
  if (ms === null || ms === undefined) return 'not timed separately'
  if (ms === 0) return '<1 ms'
  return formatDuration(ms)
}
