import type { SarData } from '@/types/api'
import { Stat, StatGrid, Swatch } from '@/components/result/stat'
import { Distribution } from '@/components/result/distribution'
import { classColor, prettifyKey } from '@/lib/classes'
import { formatPct, formatSarThresholds } from '@/lib/format'

/**
 * The backscatter distribution, for whatever part of it the run actually measured.
 *
 * `histogram_summary` returns `{bins: [], counts: [], mean: null, std: null}` — with no percentiles
 * at all — for a raster with no finite pixels, so neither the bars nor the summary line can be
 * assumed to exist. Reading them as plain numbers would repeat the `toFixed` crash that the
 * thresholds line used to cause.
 */
function SpeckleHistogram({
  counts,
  mean,
  p05,
  p95,
}: {
  counts: number[]
  mean: number | null | undefined
  p05: number | null | undefined
  p95: number | null | undefined
}) {
  const db = (value: number | null | undefined, label: string) =>
    typeof value === 'number' && Number.isFinite(value) ? `${label} ${value.toFixed(1)} dB` : null
  const summary = [db(p05, 'p05'), db(mean, 'mean'), db(p95, 'p95')].filter(
    (s): s is string => s !== null,
  )
  if (counts.length === 0 && summary.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No backscatter distribution was returned for this image.
      </p>
    )
  }
  const max = Math.max(...counts, 1)
  return (
    <div>
      {counts.length > 0 && (
        <div className="flex h-20 items-end gap-px" role="img" aria-label="Backscatter histogram">
          {counts.map((c, i) => (
            <div
              key={i}
              className="flex-1 rounded-sm bg-primary/70"
              style={{ height: `${Math.max(1, (c / max) * 100)}%` }}
              title={`${c}`}
            />
          ))}
        </div>
      )}
      {summary.length > 0 && (
        <div className="mt-1 flex justify-between text-xs text-muted-foreground">
          {summary.map((s) => (
            <span key={s}>{s}</span>
          ))}
        </div>
      )}
    </div>
  )
}

export function SarPanel({ data }: { data: SarData }) {
  const rows = data.regimes.map((r) => ({
    key: r.regime,
    label: prettifyKey(r.regime),
    color: classColor(r.regime),
    fraction: r.fraction,
    valueLabel: formatPct(r.percentage),
    hint: `${r.mean_backscatter_db.toFixed(1)} dB`,
  }))

  return (
    <div className="space-y-5">
      <StatGrid>
        <Stat label="Polarization" value={data.polarization} />
        <Stat
          label="Speckle filter"
          value={data.speckle.filter}
          hint={`${data.speckle.window}×${data.speckle.window} window`}
        />
        <Stat label="Equiv. looks (ENL)" value={data.speckle.estimated_enl.toFixed(1)} />
        <Stat
          label="Speckle reduction"
          value={`${data.speckle.speckle_reduction_factor.toFixed(2)}×`}
          hint={`CV ${data.speckle.input_cv.toFixed(2)} → ${data.speckle.output_cv.toFixed(2)}`}
        />
      </StatGrid>

      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">Scattering regimes</h4>
        <Distribution rows={rows} />
      </section>

      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">Physical interpretation</h4>
        <ul className="space-y-2 text-sm">
          {data.regimes.map((r) => (
            <li key={r.regime} className="flex items-start gap-2">
              <Swatch color={classColor(r.regime)} />
              <span>
                <span className="font-medium text-foreground">{prettifyKey(r.regime)}</span>
                <span className="text-muted-foreground">
                  {' '}
                  ({r.mean_backscatter_db.toFixed(1)} dB) — likely {r.candidate_surfaces.join(', ')}
                </span>
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-xs text-muted-foreground">
          {formatSarThresholds(data.thresholds_db)} SAR gives scattering behaviour, not material
          identity — surfaces are candidates, not confirmed classes.
        </p>
      </section>

      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">
          Backscatter distribution (dB)
        </h4>
        <SpeckleHistogram
          counts={data.histogram.counts}
          mean={data.histogram.mean}
          p05={data.histogram.p05}
          p95={data.histogram.p95}
        />
      </section>
    </div>
  )
}
