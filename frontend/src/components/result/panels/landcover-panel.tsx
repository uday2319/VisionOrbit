import type { LandCoverData } from '@/types/api'
import { Stat, StatGrid } from '@/components/result/stat'
import { Distribution } from '@/components/result/distribution'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { classColor, prettifyKey } from '@/lib/classes'
import { formatArea, formatFractionPct, formatPct } from '@/lib/format'

export function LandCoverPanel({ data }: { data: LandCoverData }) {
  const rows = data.classes.map((c) => ({
    key: c.class,
    label: prettifyKey(c.class),
    color: classColor(c.class),
    fraction: c.fraction,
    valueLabel: formatPct(c.percentage),
    hint: formatArea(c.area_m2),
  }))

  const summaries = Object.values(data.index_summaries)
  const usedSet = new Set(data.indices_used.map((i) => i.toLowerCase()))

  return (
    <div className="space-y-5">
      <StatGrid>
        <Stat label="Method" value={data.method} />
        <Stat
          label="Separability"
          value={formatFractionPct(data.separability)}
          hint="class distinctness"
        />
        <Stat
          label="Dominant"
          value={prettifyKey(data.dominant.class)}
          hint={formatPct(data.dominant.percentage)}
        />
        <Stat
          label="Indices used"
          value={data.indices_used.map((i) => i.toUpperCase()).join(', ')}
        />
      </StatGrid>

      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">Land-cover distribution</h4>
        <Distribution rows={rows} />
      </section>

      {summaries.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-semibold text-foreground">Spectral indices</h4>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Index</TableHead>
                <TableHead className="text-right">Mean</TableHead>
                <TableHead className="text-right">Range</TableHead>
                <TableHead className="text-right">Valid</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {summaries.map((s) => (
                <TableRow key={s.name}>
                  <TableCell>
                    <div className="flex items-center gap-1.5">
                      <span className="font-medium uppercase text-foreground">{s.name}</span>
                      {usedSet.has(s.name.toLowerCase()) && (
                        <Badge variant="secondary" className="text-[10px]">
                          used
                        </Badge>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground">{s.formula}</div>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{s.mean.toFixed(3)}</TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {s.min.toFixed(2)} – {s.max.toFixed(2)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {formatFractionPct(s.valid_fraction, 0)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </section>
      )}
    </div>
  )
}
