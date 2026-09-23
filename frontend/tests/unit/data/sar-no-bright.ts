/**
 * A recorded single-SAR run in which only one Otsu threshold resolved.
 *
 * Taken from SAT-2026-000548: a 4-band image with no band labelled as a radar polarisation and
 * values that were not in decibels. Otsu separated smooth from diffuse but found no distinct
 * bright-scattering population, so `thresholds_db.bright` is null and the run says so in its own
 * warnings. Every other quantity here was measured, and the run finished `high` at 0.796 — yet the
 * result page and the report both read the two cuts as plain numbers and died in `toFixed`, taking
 * the whole route down with "Unexpected Application Error!".
 *
 * Shared by the panel test and the report test so both sides are pinned to the same real payload.
 */
import type { SarData } from '@/types/api'

export const SAR_NO_BRIGHT: SarData = {
  polarization: 'unknown',
  thresholds_db: { smooth: 4.98, bright: null, method: 'otsu-3class' },
  mode_prominence: { smooth: 0.3766, bright: 0.0, threshold: 0.25 },
  separated_regimes: ['smooth', 'diffuse'],
  regimes: [
    {
      regime: 'diffuse',
      candidate_surfaces: ['vegetation', 'bare soil', 'rough natural terrain'],
      pixel_count: 141529,
      fraction: 0.9436,
      percentage: 94.36,
      area_m2: null,
      mean_backscatter_db: 15.77,
    },
    {
      regime: 'smooth',
      candidate_surfaces: [
        'calm open water',
        'wet smooth ground',
        'dry sand or bare tarmac',
        'radar shadow',
      ],
      pixel_count: 8456,
      fraction: 0.0564,
      percentage: 5.64,
      area_m2: null,
      mean_backscatter_db: -1.61,
    },
  ],
  speckle: {
    filter: 'lee-mmse',
    window: 7,
    estimated_enl: 9.55,
    input_cv: 0.3236,
    output_cv: 0.1587,
    speckle_reduction_factor: 2.04,
    warnings: [],
  },
  histogram: {
    bins: [
      -100.0, -96.1385, -92.2771, -88.4156, -84.5542, -80.6927, -76.8313, -72.9698, -69.1084,
      -65.2469, -61.3854, -57.524, -53.6625, -49.8011, -45.9396, -42.0782, -38.2167, -34.3553,
      -30.4938, -26.6323, -22.7709, -18.9094, -15.048, -11.1865, -7.3251, -3.4636, 0.3978, 4.2593,
      8.1208, 11.9822, 15.8437, 19.7051, 23.5666,
    ],
    counts: [
      25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 35, 88, 265, 536, 735, 185, 2462,
      3075, 2612, 25859, 34526, 58668, 20871,
    ],
    mean: 14.787,
    std: 5.6822,
    p05: 4.324,
    p95: 21.131,
  },
  warnings: [
    'This image has 4 bands but none is labelled as a radar polarisation; the first band was analysed and its polarisation is unknown.',
    'No distinct bright-scattering population was found, so no double-bounce area (such as buildings) is reported for this image.',
  ],
}
