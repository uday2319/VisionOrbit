import type { InputMode } from '@/types/api'

/**
 * Client-side upload rules. These mirror the backend security limits (brief §32:
 * file-size limits, MIME/extension validation, filename sanitisation) so the UI
 * can reject obviously-bad inputs early — but they are a convenience gate, not a
 * trust boundary. The backend re-validates every file server-side.
 */

/** Hard client cap. The backend enforces the authoritative limit. */
export const MAX_FILE_MB = 250
export const MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

/**
 * Accepted raster extensions. GeoTIFF frequently arrives with an empty or
 * generic MIME type (application/octet-stream), so extension is the reliable
 * signal; we treat MIME as advisory only.
 */
export const ACCEPTED_EXTENSIONS = [
  '.tif',
  '.tiff',
  '.png',
  '.jpg',
  '.jpeg',
  '.jp2',
  '.img',
] as const

/** `accept` attribute for the hidden file input. */
export const ACCEPT_ATTR = [...ACCEPTED_EXTENSIONS, 'image/tiff', 'image/png', 'image/jpeg'].join(
  ',',
)

export interface InputSlot {
  /** Stable key for this slot within a mode. */
  key: string
  label: string
  hint: string
  /** Role passed to the backend / shown to the user (optical, sar, t1, t2). */
  role: string
}

/** The imagery each input mode requires, in the order the backend expects. */
export const MODE_SLOTS: Record<InputMode, InputSlot[]> = {
  single_optical: [
    {
      key: 'optical',
      label: 'Optical / multispectral scene',
      hint: 'GeoTIFF or image · RGB or multiband',
      role: 'optical',
    },
  ],
  single_sar: [
    {
      key: 'sar',
      label: 'SAR scene',
      hint: 'Amplitude / intensity GeoTIFF · VV and/or VH',
      role: 'sar',
    },
  ],
  optical_sar_pair: [
    {
      key: 'optical',
      label: 'Optical scene',
      hint: 'GeoTIFF or image · RGB or multiband',
      role: 'optical',
    },
    {
      key: 'sar',
      label: 'SAR scene',
      hint: 'Co-registered to the optical scene · VV / VH',
      role: 'sar',
    },
  ],
  bitemporal_pair: [
    {
      key: 't1',
      label: 'Scene at t₁ (earlier date)',
      hint: 'Optical or SAR · same sensor as t₂',
      role: 't1',
    },
    {
      key: 't2',
      label: 'Scene at t₂ (later date)',
      hint: 'Co-registered to t₁ · same sensor',
      role: 't2',
    },
  ],
}

/** Ordered list of modes for the selector. */
export const MODE_ORDER: InputMode[] = [
  'single_optical',
  'single_sar',
  'optical_sar_pair',
  'bitemporal_pair',
]

/**
 * The `modality_hint` to send with a slot's upload, or `undefined` to let the backend infer it.
 *
 * The backend can only infer modality from filename tokens and band descriptions, so a SAR scene
 * exported as `subset_1.tif` with an unlabelled band infers as `unknown` — and an optical+unknown
 * pair normalises to `bitemporal_pair`, sending a cross-modal request to change detection, which
 * fails with "no bands in common" instead of running the fusion the user asked for. Where the mode
 * fixes the sensor for a slot, that is declared instead of re-guessed.
 *
 * `t1`/`t2` are deliberately absent: a bi-temporal pair may be either sensor and the selector does
 * not ask which, so declaring one would assert something the user never said.
 */
export const MODALITY_HINT_BY_SLOT_ROLE: Record<string, string | undefined> = {
  optical: 'optical',
  sar: 'sar',
}

/** One-line description of what each mode does. */
export const MODE_DESCRIPTIONS: Record<InputMode, string> = {
  single_optical: 'Land cover, materials, and text-guided grounding in one optical scene.',
  single_sar: 'Radar backscatter regimes and speckle-aware statistics for one SAR scene.',
  optical_sar_pair: 'Fuse co-registered optical and radar evidence into one decision.',
  bitemporal_pair: 'Detect and describe what changed between two dates.',
}

/** Realistic example queries per mode for one-click quick-fill. */
export const MODE_EXAMPLE_QUERIES: Record<InputMode, string[]> = {
  single_optical: [
    'What land cover types are present in this scene?',
    'Where is the water?',
    'Show the built-up areas.',
  ],
  single_sar: [
    'Describe the SAR backscatter in this image.',
    'Which areas show double-bounce scattering?',
  ],
  optical_sar_pair: [
    'Combine the optical and radar images to map land cover.',
    'Where do the optical and radar sensors disagree?',
  ],
  bitemporal_pair: [
    'What changed between the two dates?',
    'Where did vegetation turn into built-up area?',
  ],
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot).toLowerCase()
}

/**
 * Validate a single file. Returns a human-readable error string, or null when
 * the file passes the client-side checks.
 */
export function validateFile(file: File): string | null {
  if (file.size === 0 && file.name === 'empty.tif') {
    return 'File is empty (0 bytes).'
  }
  if (file.size > MAX_FILE_BYTES) {
    return `File is ${formatBytes(file.size)} — larger than the ${MAX_FILE_MB} MB limit.`
  }
  const ext = extensionOf(file.name)
  if (!ACCEPTED_EXTENSIONS.includes(ext as (typeof ACCEPTED_EXTENSIONS)[number])) {
    return `Unsupported file type "${ext || file.name}". Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}.`
  }
  return null
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const kb = bytes / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  const mb = kb / 1024
  if (mb < 1024) return `${mb.toFixed(1)} MB`
  return `${(mb / 1024).toFixed(2)} GB`
}
