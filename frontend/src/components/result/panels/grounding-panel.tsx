import type { GroundingData } from '@/types/api'
import { Stat, StatGrid } from '@/components/result/stat'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Info, AlertTriangle } from 'lucide-react'
import { prettifyKey } from '@/lib/classes'
import { formatArea, formatFractionPct, formatInt } from '@/lib/format'

export function GroundingPanel({ data }: { data: GroundingData }) {
  const q = data.query
  return (
    <div className="space-y-5">
      {/* How the query was interpreted */}
      <section className="rounded-lg border border-border bg-muted/30 p-3 text-sm">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-muted-foreground">Interpreted as:</span>
          <span className="font-medium text-foreground">find {prettifyKey(data.target)}</span>
          {q.target_phrase && (
            <Badge variant="outline" className="text-[11px]">
              “{q.target_phrase}”
            </Badge>
          )}
          {q.wants_count && <Badge variant="secondary">wants count</Badge>}
          {q.wants_area && <Badge variant="secondary">wants area</Badge>}
          {q.sector && (
            <Badge variant="outline" className="text-[11px]">
              sector: {q.sector}
            </Badge>
          )}
        </div>
      </section>

      {/* Count — respect count_reportable (brief §28: never assert an unreliable number) */}
      {q.wants_count &&
        (data.count_reportable && data.count !== null ? (
          <Stat
            label="Count"
            value={formatInt(data.count)}
            hint={`${prettifyKey(data.target)} regions`}
            className="max-w-xs"
          />
        ) : (
          <Alert variant="warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Count not reportable</AlertTitle>
            <AlertDescription>
              {data.count_caveat ??
                'The number of separate regions cannot be stated reliably from this image alone.'}
              {data.count !== null && (
                <span className="mt-1 block text-xs">
                  ({formatInt(data.count)} connected patches were mapped, but this is not a reliable
                  count.)
                </span>
              )}
            </AlertDescription>
          </Alert>
        ))}

      <StatGrid>
        <Stat label="Selected area" value={formatArea(data.selected_area_m2)} />
        <Stat label="Selected pixels" value={formatInt(data.selected_pixels)} />
        <Stat label="Class coverage" value={formatFractionPct(data.class_coverage_fraction)} />
        <Stat label="Regions mapped" value={formatInt(data.regions.length)} />
      </StatGrid>

      <p className="flex items-start gap-1.5 text-sm text-muted-foreground">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" aria-hidden="true" />
        <span>{data.region_semantics}</span>
      </p>

      {/* Evidence backing the localisation */}
      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">Localisation evidence</h4>
        <StatGrid>
          <Stat label="Method" value={data.evidence.classification_method} />
          <Stat label="Separability" value={formatFractionPct(data.evidence.class_separability)} />
          <Stat
            label="Score index"
            value={data.evidence.region_score_index?.toUpperCase() ?? '—'}
          />
          <Stat
            label="Indices"
            value={data.evidence.indices_used.map((i) => i.toUpperCase()).join(', ')}
          />
          <Stat
            label="Min region"
            value={`${formatInt(data.evidence.min_region_pixels)} px`}
            hint={formatArea(data.evidence.min_region_area_m2)}
          />
          <Stat label="Candidates" value={formatInt(data.evidence.candidate_regions)} />
        </StatGrid>
        <p className="mt-2 text-xs text-muted-foreground">
          Region outlines are drawn on the map overlay. Coordinates are{' '}
          {data.georeferenced ? 'georeferenced' : 'in pixel space (input not georeferenced)'}.
        </p>
      </section>
    </div>
  )
}
