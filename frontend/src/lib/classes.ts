/**
 * Shared display tokens for land-cover classes, SAR regimes, and change types.
 * Colours match the legend entries emitted by the real tools (see the recorded
 * fixtures), so map overlays, tables, and charts stay visually consistent.
 */

export const CLASS_COLORS: Record<string, string> = {
  water: '#2166cd',
  vegetation: '#2e9a3e',
  built_up: '#d6602d',
  bare_soil: '#bfa878',
  unclassified: '#94a3b8',
}

export const REGIME_COLORS: Record<string, string> = {
  smooth: '#2166cd',
  diffuse: '#2e9a3e',
  double_bounce: '#d6602d',
  unclassified: '#94a3b8',
}

/** Fallback colour for anything not in the maps above. */
export const NEUTRAL_COLOR = '#94a3b8'
export const CHANGE_COLOR = '#dc2878'
export const HIGHLIGHT_COLOR = '#f0c81e'

export function classColor(key: string): string {
  return CLASS_COLORS[key] ?? REGIME_COLORS[key] ?? NEUTRAL_COLOR
}

/** Turn a snake_case class/regime key into a human label ("bare_soil" -> "Bare soil"). */
export function prettifyKey(key: string): string {
  if (!key) return key
  const spaced = key.replace(/_/g, ' ')
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

/** Human labels for change transition types. */
export const CHANGE_TYPE_LABELS: Record<string, string> = {
  water_gain: 'Water gain',
  water_loss: 'Water loss',
  vegetation_gain: 'Vegetation gain',
  vegetation_loss: 'Vegetation loss',
  urban_expansion: 'Urban expansion',
  urban_loss: 'Urban loss',
  spectral_only: 'Spectral-only',
  other: 'Other',
}

export function changeTypeLabel(key: string): string {
  return CHANGE_TYPE_LABELS[key] ?? prettifyKey(key)
}

/** Colours for change transitions — gains cool/green, losses warm, other neutral. */
export const CHANGE_TYPE_COLORS: Record<string, string> = {
  water_gain: '#2166cd',
  water_loss: '#8a6d3b',
  vegetation_gain: '#2e9a3e',
  vegetation_loss: '#d68a2d',
  urban_expansion: '#d6602d',
  urban_loss: '#7c6f9c',
  spectral_only: CHANGE_COLOR,
  other: NEUTRAL_COLOR,
}

export function changeTypeColor(key: string): string {
  return CHANGE_TYPE_COLORS[key] ?? NEUTRAL_COLOR
}
