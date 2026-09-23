import { useMemo, useRef, useState } from 'react'
import 'leaflet/dist/leaflet.css'
import {
  MapContainer,
  ImageOverlay,
  ZoomControl,
} from 'react-leaflet'
import type { LatLngBoundsExpression } from 'leaflet'
import type { Artifact, LegendEntry, Wgs84Bounds } from '@/types/api'
import {
  Expand,
  ImageOff,
  Layers,
  MapPin,
  Minus,
  Plus,
  RotateCcw,
} from 'lucide-react'
import { Swatch } from '@/components/result/stat'

function toLeafletBounds(b: Wgs84Bounds): LatLngBoundsExpression {
  return [
    [b.south, b.west],
    [b.north, b.east],
  ]
}

function fmtCoord(n: number): string {
  return n.toFixed(4)
}

function mergeLegends(
  overlays: Artifact[],
  visible: Record<string, boolean>,
): LegendEntry[] {
  const seen = new Map<string, LegendEntry>()

  for (const overlay of overlays) {
    if (!visible[overlay.kind]) continue

    for (const entry of overlay.legend) {
      if (!seen.has(entry.key)) {
        seen.set(entry.key, entry)
      }
    }
  }

  return [...seen.values()]
}

interface ResultMapProps {
  artifacts: Artifact[]
}

type ViewMode = 'original' | 'analysis' | 'overlay'

export function ResultMap({ artifacts }: ResultMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  const base = useMemo(
    () => artifacts.find((artifact) => artifact.kind === 'base') ?? null,
    [artifacts],
  )

  const overlays = useMemo(
    () => artifacts.filter((artifact) => artifact.kind !== 'base'),
    [artifacts],
  )

  const [viewMode, setViewMode] = useState<ViewMode>('overlay')

  const [visible, setVisible] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(
      overlays.map((overlay, index) => [
        overlay.kind,
        index === 0,
      ]),
    ),
  )

  const [opacity, setOpacity] = useState(0.7)

  const sceneBounds = useMemo(() => {
    const withBounds = artifacts.find(
      (artifact) => artifact.bounds_wgs84,
    )

    return withBounds?.bounds_wgs84 ?? null
  }, [artifacts])

  const legend = useMemo(
    () => mergeLegends(overlays, visible),
    [overlays, visible],
  )

  if (artifacts.length === 0) {
    return (
      <div className="flex h-64 flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border text-muted-foreground">
        <ImageOff className="h-6 w-6" />
        <p className="text-sm">
          No rendered imagery for this analysis.
        </p>
      </div>
    )
  }

  const georeferenced = sceneBounds !== null

  const showBase =
    viewMode === 'original' || viewMode === 'overlay'

  const showAnalysis =
    viewMode === 'analysis' || viewMode === 'overlay'

  function resetView() {
    setViewMode('overlay')
    setOpacity(0.7)

    setVisible(
      Object.fromEntries(
        overlays.map((overlay, index) => [
          overlay.kind,
          index === 0,
        ]),
      ),
    )
  }

  function toggleFullscreen() {
    if (!containerRef.current) return

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen()
    } else {
      document.exitFullscreen()
    }
  }

  return (
    <div className="space-y-3">

      {/* =========================
          IMAGE VIEWER
      ========================== */}
      <div
        ref={containerRef}
        className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-100"
      >

        {/* Top viewer toolbar */}
        <div className="absolute left-4 top-4 z-[1000] flex overflow-hidden rounded-lg border border-slate-200 bg-white shadow-md">

          <button
            type="button"
            onClick={() => setViewMode('original')}
            className={`px-3 py-2 text-xs font-medium ${
              viewMode === 'original'
                ? 'bg-blue-600 text-white'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            Original
          </button>

          <button
            type="button"
            onClick={() => setViewMode('analysis')}
            className={`px-3 py-2 text-xs font-medium ${
              viewMode === 'analysis'
                ? 'bg-blue-600 text-white'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            Analysis
          </button>

          <button
            type="button"
            onClick={() => setViewMode('overlay')}
            className={`px-3 py-2 text-xs font-medium ${
              viewMode === 'overlay'
                ? 'bg-blue-600 text-white'
                : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            Overlay
          </button>

        </div>

        {/* Right controls */}
        <div className="absolute right-4 top-4 z-[1000] flex gap-2">

          <button
            type="button"
            onClick={resetView}
            title="Reset view"
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 shadow-md hover:bg-slate-50"
          >
            <RotateCcw className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={toggleFullscreen}
            title="Fullscreen"
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 shadow-md hover:bg-slate-50"
          >
            <Expand className="h-4 w-4" />
          </button>

        </div>

        {/* =========================
            GEOREFERENCED IMAGE
        ========================== */}
        {georeferenced ? (

          <div className="h-[420px] w-full md:h-[500px]">

            <MapContainer
              bounds={toLeafletBounds(sceneBounds)}
              boundsOptions={{ padding: [10, 10] }}
              zoomControl={false}
              attributionControl={false}
              scrollWheelZoom
              className="h-full w-full"
            >

              <ZoomControl position="bottomright" />

              {/* Original image */}
              {showBase &&
                base?.bounds_wgs84 && (
                  <ImageOverlay
                    url={base.url}
                    bounds={toLeafletBounds(
                      base.bounds_wgs84,
                    )}
                  />
                )}

              {/* Analysis overlays */}
              {showAnalysis &&
                overlays.map(
                  (overlay) =>
                    overlay.bounds_wgs84 &&
                    visible[overlay.kind] && (
                      <ImageOverlay
                        key={overlay.kind}
                        url={overlay.url}
                        bounds={toLeafletBounds(
                          overlay.bounds_wgs84,
                        )}
                        opacity={opacity}
                      />
                    ),
                )}

            </MapContainer>

          </div>

        ) : (

          /* =========================
             NORMAL IMAGE
          ========================== */
          <div className="relative h-[420px] w-full bg-slate-100 md:h-[500px]">

            {showBase && base && (
              <img
                src={base.url}
                alt={base.label}
                className="absolute inset-0 h-full w-full object-contain"
              />
            )}

            {showAnalysis &&
              overlays.map(
                (overlay) =>
                  visible[overlay.kind] && (
                    <img
                      key={overlay.kind}
                      src={overlay.url}
                      alt={overlay.label}
                      style={{ opacity }}
                      className="absolute inset-0 h-full w-full object-contain"
                    />
                  ),
              )}

          </div>
        )}

        {/* Image status */}
        <div className="absolute bottom-4 left-4 z-[1000] rounded-lg border border-slate-200 bg-white/95 px-3 py-2 shadow-sm">

          <div className="flex items-center gap-2">

            <div className="h-2 w-2 animate-pulse rounded-full bg-green-500" />

            <span className="text-xs font-medium text-slate-700">
              {viewMode === 'original'
                ? 'Original imagery'
                : viewMode === 'analysis'
                  ? 'Analysis layer'
                  : 'Analysis overlay'}
            </span>

          </div>

        </div>

      </div>

      {/* =========================
          LAYERS
      ========================== */}
      <div className="rounded-xl border border-slate-200 bg-white p-4">

        <div className="flex flex-wrap items-center justify-between gap-4">

          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-blue-600" />

            <span className="text-sm font-semibold text-slate-800">
              Analysis Layers
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-4">

            {base && (
              <span className="text-xs text-slate-500">
                Base:
                <span className="ml-1 font-medium text-slate-800">
                  {base.label}
                </span>
              </span>
            )}

            {overlays.map((overlay) => (
              <label
                key={overlay.kind}
                className="flex cursor-pointer items-center gap-2 text-xs text-slate-700"
              >

                <input
                  type="checkbox"
                  checked={!!visible[overlay.kind]}
                  onChange={(event) =>
                    setVisible((previous) => ({
                      ...previous,
                      [overlay.kind]: event.target.checked,
                    }))
                  }
                  className="h-4 w-4 accent-blue-600"
                />

                {overlay.label}

              </label>
            ))}

          </div>

        </div>

        {/* Opacity */}
        {overlays.length > 0 && (
          <div className="mt-4 flex items-center gap-3 border-t border-slate-100 pt-3">

            <span className="text-xs text-slate-500">
              Overlay opacity
            </span>

            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(event) =>
                setOpacity(Number(event.target.value))
              }
              className="h-1.5 w-40 accent-blue-600"
            />

            <span className="text-xs font-medium text-slate-700">
              {Math.round(opacity * 100)}%
            </span>

          </div>
        )}

      </div>

      {/* =========================
          LEGEND
      ========================== */}
      {legend.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">

          <div className="mb-3 text-xs font-semibold text-slate-800">
            Detected Regions
          </div>

          <div className="flex flex-wrap gap-4">

            {legend.map((entry) => (
              <span
                key={entry.key}
                className="flex items-center gap-2 text-xs text-slate-600"
              >
                <Swatch color={entry.color} />
                {entry.label}
              </span>
            ))}

          </div>

        </div>
      )}

      {/* =========================
          GEO INFORMATION
      ========================== */}
      <div className="flex items-center gap-2 text-xs text-slate-500">

        <MapPin className="h-3.5 w-3.5" />

        {georeferenced && sceneBounds ? (
          <span>
            WGS84:
            S {fmtCoord(sceneBounds.south)},
            W {fmtCoord(sceneBounds.west)}
            {' → '}
            N {fmtCoord(sceneBounds.north)},
            E {fmtCoord(sceneBounds.east)}
          </span>
        ) : (
          <span>
            Non-georeferenced imagery — pixel view.
          </span>
        )}

      </div>

    </div>
  )
}