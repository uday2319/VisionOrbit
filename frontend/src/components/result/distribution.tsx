import { Swatch } from '@/components/result/stat'

export interface DistRow {
  key: string
  label: string
  color: string
  /** 0..1 — drives the bar width. */
  fraction: number
  /** Right-aligned primary value, e.g. "42.1%". */
  valueLabel: string
  /** Optional secondary value, e.g. an area. */
  hint?: string
}

/**
 * A labelled horizontal-bar list used by every data panel (land cover, SAR
 * regimes, fusion classes, change transitions). CSS-only — no chart library —
 * so the exact measured numbers are shown and it renders deterministically in
 * tests and print.
 */
export function Distribution({ rows }: { rows: DistRow[] }) {
  return (
    <ul className="space-y-2.5">
      {rows.map((r) => (
        <li key={r.key} className="space-y-1">
          <div className="flex items-center justify-between gap-2 text-sm">
            <span className="inline-flex items-center gap-1.5">
              <Swatch color={r.color} />
              <span className="text-foreground">{r.label}</span>
            </span>
            <span className="tabular-nums text-muted-foreground">
              {r.valueLabel}
              {r.hint && <span className="ml-1 text-xs">· {r.hint}</span>}
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.min(100, Math.max(0, r.fraction * 100)).toFixed(1)}%`,
                backgroundColor: r.color,
              }}
            />
          </div>
        </li>
      ))}
    </ul>
  )
}
