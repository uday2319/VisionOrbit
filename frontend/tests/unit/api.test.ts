import { afterEach, describe, it, expect, vi } from 'vitest'
import {
  analyze,
  ApiRequestError,
  apiConfig,
  getAnalysis,
  getHealth,
  getMetrics,
  listAnalyses,
} from '@/services/api'
import {
  isChangeData,
  isFusionData,
  isGroundingData,
  isLandCoverData,
  isSarData,
} from '@/types/api'

// Tests run with VITE_USE_FIXTURES unset -> offline fixtures mode.
describe('api client (offline fixtures mode)', () => {
  it('defaults to offline in the test environment', () => {
    expect(apiConfig.offline).toBe(true)
  })

  it('lists the demo catalog newest-first', async () => {
    const rows = await listAnalyses()
    expect(rows).toHaveLength(6)
    expect(rows[0].id).toBe('SAT-2026-000106')
    expect(rows.map((r) => r.id)).toContain('SAT-2026-000101')
  })

  it('loads a full envelope and marks it as a fixture', async () => {
    const result = await getAnalysis('SAT-2026-000101')
    expect(result.id).toBe('SAT-2026-000101')
    expect(result.fixture).toBe(true)
    expect(result.trace.length).toBeGreaterThan(0)
    expect(result.confidence.level).toBe('high')
  })

  it('rejects an unknown id with a not_found ApiRequestError', async () => {
    await expect(getAnalysis('SAT-2026-999999')).rejects.toMatchObject({
      name: 'ApiRequestError',
      errorCode: 'not_found',
    })
  })

  it('replays a demo case through analyze()', async () => {
    const result = await analyze({
      mode: 'single_optical',
      query: 'What land cover types are present?',
      files: [],
      demoCaseId: 'SAT-2026-000101',
    })
    expect(result.id).toBe('SAT-2026-000101')
  })

  it('refuses to fabricate a result for uploaded imagery when offline', async () => {
    const err = await analyze({
      mode: 'single_optical',
      query: 'classify this',
      files: [new File([new Uint8Array([1, 2, 3])], 'scene.tif')],
    }).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiRequestError)
    expect((err as ApiRequestError).errorCode).toBe('backend_required')
    expect((err as ApiRequestError).recoverable).toBe(true)
  })

  it('reports honest offline health', async () => {
    const health = await getHealth()
    expect(health.status).toBe('degraded')
    expect(health.checks?.backend.status).toBe('down')
  })

  it('reports only demo-catalog metrics offline (no invented throughput)', async () => {
    const metrics = await getMetrics()
    expect(metrics.source).toBe('demo_catalog')
    expect(metrics.requests_total).toBe(6)
    expect(metrics.avg_duration_ms).toBeUndefined()
  })
})

/**
 * What the live mappers are allowed to claim.
 *
 * Every assertion here pins a field the client used to invent rather than read. The symptom the
 * user saw was one result page that simultaneously said "Tool ModeNormalizer", "Success", and
 * "Insufficient evidence for a reliable conclusion" — over a real, measured answer. None of those
 * three came from the backend: `tool` was read from whichever component happened to occupy trace
 * step 1, `status` was the literal `'success'`, and the whole confidence report was manufactured
 * (three fixed reason strings and a `score >= 0.5` sufficiency rule of the UI's own making).
 */
const BACKEND_CONFIDENCE = {
  level: 'low',
  score: 0.494,
  // The backend considers `low` sufficient to state with caveats; only `insufficient` is not. The
  // old `score >= 0.5` rule disagreed with it on exactly this value.
  sufficient: true,
  limiting_factor: 'evidence_strength',
  reasons: ['Spectral separability is weak', 'Co-registration is approximate'],
  factors: [
    {
      name: 'evidence_strength',
      value: 0.494,
      weight: 1.0,
      kind: 'gate',
      reason: 'Spectral separability is weak',
    },
    {
      name: 'registration',
      value: 0.78,
      weight: 1.0,
      kind: 'contributor',
      reason: 'Co-registration is approximate',
    },
  ],
}

/** The three strings the client used to emit no matter what the backend measured. */
const FABRICATED_REASONS = [
  'Valid radiometric bounds',
  'Sensor metadata verified',
  'Measured spectral indices',
]

function backendEnvelope(id: string, overrides: Record<string, unknown> = {}) {
  return {
    analysis_id: id,
    query: 'Combine the optical and radar images to map land cover.',
    mode: 'optical_sar_pair',
    task: 'optical_sar_analysis',
    tool: 'fusion-decision-cv',
    tier: 'classical',
    status: 'partial',
    answer: 'Built-up cover dominates the scene.',
    confidence_score: BACKEND_CONFIDENCE.score,
    confidence_level: BACKEND_CONFIDENCE.level,
    confidence: BACKEND_CONFIDENCE,
    evidence: ['Backscatter mean -13.8 dB'],
    structured_evidence: [],
    execution_trace: {
      trace_id: `trace-${id}`,
      total_duration_ms: 812,
      steps: [
        {
          step_index: 0,
          step_name: 'preprocess',
          tool_name: 'RasterPreprocessor',
          status: 'ok',
          details: 'Loaded 2 rasters',
          duration_ms: 40,
          warnings: [],
        },
        {
          // The step that gave the page its wrong headline: second in the trace, and not the analyst.
          step_index: 1,
          step_name: 'classify',
          tool_name: 'ModeNormalizer',
          status: 'ok',
          details: 'Normalised to optical_sar_pair',
          duration_ms: 5,
          warnings: [],
        },
        {
          step_index: 2,
          step_name: 'execute',
          tool_name: 'fusion-decision-cv',
          status: 'warning',
          details: 'Fused both sensors',
          duration_ms: 700,
          warnings: ['Co-registration is approximate'],
        },
      ],
    },
    data: {},
    warnings: ['Co-registration is approximate'],
    created_at: '2026-08-29T10:00:00Z',
    ...overrides,
  }
}

describe('api client (live backend mapping)', () => {
  afterEach(() => {
    apiConfig.offline = true
    vi.unstubAllGlobals()
  })

  /** Serve a fixed JSON body for every request, and record where the client went. */
  function stubBackend(bodyFor: (url: string) => unknown) {
    apiConfig.offline = false
    const calls: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        calls.push(url)
        return new Response(JSON.stringify(bodyFor(url)), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }),
    )
    return calls
  }

  it('credits the tool the backend names, not whichever component sits at trace step 1', async () => {
    stubBackend(() => backendEnvelope('SAT-LIVE-000001'))
    const result = await getAnalysis('SAT-LIVE-000001')
    // Never 'ModeNormalizer' — that step is present in the trace and must stay there...
    expect(result.tool).toBe('fusion-decision-cv')
    expect(result.trace.map((s) => s.tool)).toContain('ModeNormalizer')
    // ...but it is not the analyst, and 'remote-sensing-specialist' (the old fallback) is not a
    // registered tool at all.
    expect(result.tool).not.toBe('ModeNormalizer')
    expect(result.tool).not.toBe('remote-sensing-specialist')
  })

  it('reads tier, status and duration instead of hardcoding them', async () => {
    stubBackend(() => backendEnvelope('SAT-LIVE-000002'))
    const result = await getAnalysis('SAT-LIVE-000002')
    expect(result.tier).toBe('classical')
    // A run that finished with warnings is 'partial'. Hardcoded 'success' painted it green.
    expect(result.status).toBe('partial')
    expect(result.duration_ms).toBe(812)
  })

  it('passes the measured confidence report through untouched', async () => {
    stubBackend(() => backendEnvelope('SAT-LIVE-000003'))
    const result = await getAnalysis('SAT-LIVE-000003')
    expect(result.confidence.score).toBeCloseTo(0.494, 5)
    expect(result.confidence.level).toBe('low')
    // The verdict is the backend's: a measured `low` is sufficient, so no "Insufficient evidence"
    // banner. The old `score >= 0.5` rule inverted this exact value.
    expect(result.confidence.sufficient).toBe(true)
    expect(result.confidence.limiting_factor).toBe('evidence_strength')
    expect(result.confidence.reasons).toEqual(BACKEND_CONFIDENCE.reasons)
    expect(result.confidence.factors.map((f) => f.name)).toEqual([
      'evidence_strength',
      'registration',
    ])
    expect(result.confidence.factors.map((f) => f.kind)).toEqual(['gate', 'contributor'])
    for (const invented of FABRICATED_REASONS) {
      expect(result.confidence.reasons).not.toContain(invented)
    }
    expect(result.confidence.factors.map((f) => f.name)).not.toContain('model_confidence')
  })

  it('claims nothing beyond score and level when a row predates the report', async () => {
    stubBackend(() => {
      const { confidence: _dropped, ...withoutReport } = backendEnvelope('SAT-LIVE-000004')
      return withoutReport
    })
    const result = await getAnalysis('SAT-LIVE-000004')
    expect(result.confidence.score).toBeCloseTo(0.494, 5)
    expect(result.confidence.level).toBe('low')
    // Derived from the level the way the confidence engine derives it, not from a threshold.
    expect(result.confidence.sufficient).toBe(true)
    // Nothing is back-filled with plausible-sounding text.
    expect(result.confidence.reasons).toEqual([])
    expect(result.confidence.factors).toEqual([])
    expect(result.confidence.limiting_factor).toBeNull()
  })

  it('withholds sufficiency only when the backend measured "insufficient"', async () => {
    stubBackend(() =>
      backendEnvelope('SAT-LIVE-000005', {
        confidence_level: 'insufficient',
        confidence: { ...BACKEND_CONFIDENCE, level: 'insufficient', sufficient: false },
      }),
    )
    const result = await getAnalysis('SAT-LIVE-000005')
    expect(result.confidence.level).toBe('insufficient')
    expect(result.confidence.sufficient).toBe(false)
  })

  it('maps the same fields the same way through analyze()', async () => {
    const calls = stubBackend((url) =>
      url.includes('/api/analyze') ? backendEnvelope('SAT-LIVE-000006') : {},
    )
    const result = await analyze({
      mode: 'optical_sar_pair',
      query: 'Combine the optical and radar images to map land cover.',
      files: [],
    })
    expect(calls.some((u) => u.includes('/api/analyze'))).toBe(true)
    expect(result.tool).toBe('fusion-decision-cv')
    expect(result.tier).toBe('classical')
    expect(result.status).toBe('partial')
    // The mode the backend normalised to wins over the mode the UI selected.
    expect(result.mode).toBe('optical_sar_pair')
    expect(result.confidence.sufficient).toBe(true)
    expect(result.confidence.reasons).toEqual(BACKEND_CONFIDENCE.reasons)
    expect(result.fixture).toBe(false)
  })

  it('reports the backend-normalised mode even when it contradicts the selection', async () => {
    stubBackend((url) =>
      url.includes('/api/analyze')
        ? backendEnvelope('SAT-LIVE-000007', { mode: 'optical_sar_pair' })
        : {},
    )
    const result = await analyze({
      mode: 'bitemporal_pair',
      query: 'What changed?',
      files: [],
    })
    expect(result.mode).toBe('optical_sar_pair')
  })

  /**
   * The trace timings are measurements, so they survive the trip verbatim.
   *
   * Both mappers used to write `s.duration_ms || 25` (30 on the detail path), which caught two
   * distinct real values: a step that finished in under half a millisecond (`0`) and one the backend
   * does not time separately (`null`, mode normalisation — its cost is inside routing). In a change
   * run that is four of six steps, so the trace summed to 479 ms beside a measured 380 ms.
   */
  it('passes measured step durations through, including 0 and null', async () => {
    const trace = {
      trace_id: 'trace-durations',
      total_duration_ms: 380,
      steps: [
        {
          step_index: 0,
          step_name: 'preprocess',
          tool_name: 'RasterPreprocessor',
          status: 'ok',
          details: 'Loaded 2 rasters',
          duration_ms: 55,
          warnings: [],
        },
        {
          step_index: 1,
          step_name: 'classify',
          tool_name: 'ModeNormalizer',
          status: 'ok',
          details: 'Normalised',
          duration_ms: null,
          warnings: [],
        },
        {
          step_index: 2,
          step_name: 'classify',
          tool_name: 'AgentRouter',
          status: 'ok',
          details: 'Routed',
          duration_ms: 0,
          warnings: [],
        },
        {
          step_index: 3,
          step_name: 'execute',
          tool_name: 'change-cva-cv',
          status: 'ok',
          details: 'Executed',
          duration_ms: 324,
          warnings: [],
        },
      ],
    }

    stubBackend(() => backendEnvelope('SAT-LIVE-000008', { execution_trace: trace }))
    const result = await getAnalysis('SAT-LIVE-000008')

    expect(result.trace.map((s) => s.duration_ms)).toEqual([55, null, 0, 324])
    // The two substituted constants, neither of which was ever measured.
    expect(result.trace.map((s) => s.duration_ms)).not.toContain(25)
    expect(result.trace.map((s) => s.duration_ms)).not.toContain(30)
    // And the steps still cannot outweigh the run they belong to.
    const summed = result.trace.reduce((n, s) => n + (s.duration_ms ?? 0), 0)
    expect(summed).toBeLessThanOrEqual(result.duration_ms)
  })

  it('passes measured step durations through analyze() the same way', async () => {
    stubBackend((url) =>
      url.includes('/api/analyze')
        ? backendEnvelope('SAT-LIVE-000009', {
            execution_trace: {
              trace_id: 'trace-durations-analyze',
              total_duration_ms: 55,
              steps: [
                {
                  step_index: 0,
                  step_name: 'classify',
                  tool_name: 'ModeNormalizer',
                  status: 'ok',
                  details: 'Normalised',
                  duration_ms: null,
                  warnings: [],
                },
                {
                  step_index: 1,
                  step_name: 'validate',
                  tool_name: 'ToolRegistry',
                  status: 'ok',
                  details: 'Contract satisfied',
                  duration_ms: 0,
                  warnings: [],
                },
                {
                  step_index: 2,
                  step_name: 'execute',
                  tool_name: 'change-cva-cv',
                  status: 'ok',
                  details: 'Executed',
                  duration_ms: 55,
                  warnings: [],
                },
              ],
            },
          })
        : {},
    )
    const result = await analyze({ mode: 'bitemporal_pair', query: 'What changed?', files: [] })
    expect(result.trace.map((s) => s.duration_ms)).toEqual([null, 0, 55])
  })

  /**
   * The imagery a result page shows.
   *
   * Both mappers hardcoded `artifacts: []` and `inputs: []`, which is why a live run's Imagery panel
   * read "No rendered imagery for this analysis." over an analysis that had just classified every
   * pixel, and its Inputs panel came up empty on reload. The backend renders both from the arrays the
   * run produced and publishes them; these pin that they are read, and that nothing is substituted
   * when a run genuinely recorded none.
   */
  const BOX = { south: 18.9, west: 74.9, north: 19.0, east: 75.0 }

  const RENDERED = [
    {
      kind: 'base',
      label: 'Input image',
      url: '/storage/artifacts/SAT-LIVE-IMG/base.png',
      bounds_wgs84: BOX,
      legend: [],
    },
    {
      kind: 'class_map',
      label: 'Land-cover classification',
      url: '/storage/artifacts/SAT-LIVE-IMG/overlay.png',
      bounds_wgs84: BOX,
      legend: [{ label: 'Water', color: '#3366cd', key: 'water' }],
    },
  ]

  const BACKEND_INPUT = {
    filename: 'scene.tif',
    driver: 'GTiff',
    width: 60,
    height: 60,
    bands: 4,
    dtype: 'uint16',
    crs: 'EPSG:32643',
    crs_epsg: 32643,
    transform: [10, 0, 500000, 0, -10, 3000000],
    bounds: [500000, 2999400, 500600, 3000000],
    nodata: null,
    pixel_size: [10, 10],
    georeferenced: true,
    // The fourth band carries no name in the file; the null holds its place.
    band_descriptions: ['blue', 'green', 'red', null],
    decimation: 1,
    modality: 'optical',
    band_roles: ['blue', 'green', 'red', 'nir'],
    bounds_wgs84: BOX,
  }

  /** What the upload endpoint returns: a thumbnail and the few header fields it echoes back. */
  const UPLOADED = {
    image_id: 'img-1',
    filename: 'scene.tif',
    width: 60,
    height: 60,
    bands: 4,
    dtype: 'uint16',
    crs: 'EPSG:32643',
    crs_epsg: 32643,
    georeferenced: true,
    modality: 'optical',
    preview_url: '/storage/previews/img-1.png',
    decimation: 1,
    pixel_size: [10, 10],
    bounds: [500000, 2999400, 500600, 3000000],
    band_descriptions: ['blue', 'green', 'red'],
  }

  function withImagery(id: string) {
    return backendEnvelope(id, { artifacts: RENDERED, inputs: [BACKEND_INPUT] })
  }

  it('shows the imagery a stored analysis recorded', async () => {
    stubBackend(() => withImagery('SAT-LIVE-000010'))
    const result = await getAnalysis('SAT-LIVE-000010')
    expect(result.artifacts.map((a) => a.kind)).toEqual(['base', 'class_map'])
    expect(result.artifacts[1].url).toBe('/storage/artifacts/SAT-LIVE-IMG/overlay.png')
    expect(result.artifacts[1].legend).toEqual([
      { label: 'Water', color: '#3366cd', key: 'water' },
    ])
    // The footprint passes through as measured — the map draws the overlay on it.
    expect(result.artifacts[0].bounds_wgs84).toEqual(BOX)
  })

  it('leaves the imagery empty when the run recorded none', async () => {
    stubBackend(() => backendEnvelope('SAT-LIVE-000011'))
    const result = await getAnalysis('SAT-LIVE-000011')
    // A run whose render failed says so in its warnings; the panel's placeholder is then the truth
    // about that run, so no stand-in image is invented for it.
    expect(result.artifacts).toEqual([])
    expect(result.inputs).toEqual([])
  })

  it('keeps a null footprint null rather than making the map work', async () => {
    stubBackend(() =>
      backendEnvelope('SAT-LIVE-000012', {
        artifacts: RENDERED.map((a) => ({ ...a, bounds_wgs84: null })),
      }),
    )
    const result = await getAnalysis('SAT-LIVE-000012')
    // §8: null is what tells the result page to use its plain pixel viewer.
    expect(result.artifacts.every((a) => a.bounds_wgs84 === null)).toBe(true)
  })

  it('keeps the input headers the analysis resolved', async () => {
    stubBackend(() => withImagery('SAT-LIVE-000013'))
    const { inputs } = await getAnalysis('SAT-LIVE-000013')
    expect(inputs).toHaveLength(1)
    expect(inputs[0].band_roles).toEqual(['blue', 'green', 'red', 'nir'])
    expect(inputs[0].driver).toBe('GTiff')
    expect(inputs[0].transform).toEqual([10, 0, 500000, 0, -10, 3000000])
    expect(inputs[0].pixel_size).toEqual([10, 10])
    // An unnamed band stays a hole at its own position.
    expect(inputs[0].band_descriptions).toEqual(['blue', 'green', 'red', null])
  })

  it('will not name a sensor the backend did not resolve', async () => {
    stubBackend(() =>
      backendEnvelope('SAT-LIVE-000014', {
        inputs: [{ ...BACKEND_INPUT, modality: 'multispectral-ish' }],
      }),
    )
    const { inputs } = await getAnalysis('SAT-LIVE-000014')
    expect(inputs[0].modality).toBe('unknown')
  })

  it("prefers the run's own imagery over the upload thumbnails", async () => {
    stubBackend((url) =>
      url.includes('/api/upload') ? UPLOADED : withImagery('SAT-LIVE-000015'),
    )
    const result = await analyze({
      mode: 'single_optical',
      query: 'What land cover types are present?',
      files: [new File([new Uint8Array([1, 2, 3])], 'scene.tif')],
    })
    // The upload thumbnail is a picture of the input; the analysis also drew what it measured.
    expect(result.artifacts.map((a) => a.kind)).toEqual(['base', 'class_map'])
    expect(result.artifacts.map((a) => a.url)).not.toContain('/storage/previews/img-1.png')
    // And the raster as the analysis read it, rather than as the upload echoed it.
    expect(result.inputs[0].band_roles).toEqual(['blue', 'green', 'red', 'nir'])
  })

  it('falls back to the upload thumbnails when the backend renders none', async () => {
    stubBackend((url) =>
      url.includes('/api/upload') ? UPLOADED : backendEnvelope('SAT-LIVE-000016'),
    )
    const result = await analyze({
      mode: 'single_optical',
      query: 'What land cover types are present?',
      files: [new File([new Uint8Array([1, 2, 3])], 'scene.tif')],
    })
    expect(result.artifacts).toEqual([
      {
        kind: 'base',
        label: 'scene.tif',
        url: '/storage/previews/img-1.png',
        bounds_wgs84: null,
        legend: [],
      },
    ])
    // The upload response reports no band roles, so none are claimed on its behalf.
    expect(result.inputs[0].band_roles).toEqual([])
  })

  /**
   * A recorded case asked for by name is answered from the recording.
   *
   * The two id spaces overlap: the backend mints `SAT-2026-NNNNNN` from its own counter, so a machine
   * that has run more than a hundred analyses also holds rows at the six ids of the shipped
   * recordings. Resolving live-first made the dashboard's "View example" links open those rows instead
   * — a different question, a different tool, and an empty Imagery panel, since a row recorded before
   * artifacts were persisted carries none.
   */
  it('opens the recording for a demo case even when the backend has that id', async () => {
    const calls = stubBackend(() =>
      // What this machine's backend actually returns for SAT-2026-000101: an unrelated fusion run.
      backendEnvelope('SAT-2026-000101', { query: 'Combine the optical and radar images.' }),
    )
    const result = await getAnalysis('SAT-2026-000101', { recorded: true })
    expect(result.fixture).toBe(true)
    expect(result.query).toBe('What land cover types are present in this scene?')
    expect(result.tool).toBe('landcover-spectral-cv')
    expect(result.artifacts.map((a) => a.kind)).toEqual(['base', 'class_map'])
    // Answered entirely from the recording: the backend was never consulted for it.
    expect(calls.some((u) => u.includes('/api/analysis/'))).toBe(false)
  })

  it('still prefers the live row for an id nobody asked for as a recording', async () => {
    stubBackend(() =>
      backendEnvelope('SAT-2026-000103', { query: 'Combine the optical and radar images.' }),
    )
    // 000103 is a shipped recording too, so this is the same collision seen from the other side: an
    // ordinary link — a history row, a pasted URL — must still open the live analysis, not shadow it
    // with the recording. (The live row then stays in the session cache under that id; the recorded
    // path above is checked ahead of that cache for exactly this reason.)
    const result = await getAnalysis('SAT-2026-000103')
    expect(result.fixture).toBe(false)
    expect(result.query).toBe('Combine the optical and radar images.')
  })

  /**
   * The dashboard listing merges live history with the shipped recordings, and each row links to
   * `/analysis/<id>` — so the row itself has to say which of the two namespaces its id belongs to.
   * Without that, a merged recording's row links live-first and opens a different analysis than the
   * one it just listed, exactly as the "View example" cards did.
   */
  it('says which listed rows are recordings and which are live runs', async () => {
    stubBackend(() => ({
      items: [
        {
          analysis_id: 'SAT-2026-000544',
          query: 'Combine the optical and radar images.',
          task: 'fusion',
          mode: 'pair_optical_sar',
          answer: 'Fused land cover mapped.',
          confidence_level: 'medium',
          created_at: '2026-08-30T10:00:00Z',
        },
      ],
    }))
    const rows = await listAnalyses()
    const live = rows.find((r) => r.id === 'SAT-2026-000544')
    expect(live?.fixture).toBeFalsy()
    // The six recordings are merged in behind it, and each one is marked as such.
    const recorded = rows.filter((r) => r.id !== 'SAT-2026-000544')
    expect(recorded).toHaveLength(6)
    expect(recorded.every((r) => r.fixture === true)).toBe(true)
  })
})

describe('AnalysisData type guards against real fixtures', () => {
  it('identifies each task payload uniquely', async () => {
    const cases = [
      { id: 'SAT-2026-000101', guard: isLandCoverData },
      { id: 'SAT-2026-000102', guard: isGroundingData },
      { id: 'SAT-2026-000104', guard: isSarData },
      { id: 'SAT-2026-000105', guard: isFusionData },
      { id: 'SAT-2026-000106', guard: isChangeData },
    ] as const

    const allGuards = [isLandCoverData, isGroundingData, isSarData, isFusionData, isChangeData]

    for (const { id, guard } of cases) {
      const { data } = await getAnalysis(id)
      // The intended guard matches...
      expect(guard(data)).toBe(true)
      // ...and no other guard also matches (mutual exclusivity).
      const matches = allGuards.filter((g) => g(data))
      expect(matches).toHaveLength(1)
    }
  })
})
