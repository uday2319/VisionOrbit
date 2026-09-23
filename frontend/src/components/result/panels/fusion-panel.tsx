import type { FusionData, Registration } from '@/types/api'
import { Stat, StatGrid } from '@/components/result/stat'
import { Distribution } from '@/components/result/distribution'
import { Badge } from '@/components/ui/badge'
import { classColor, prettifyKey } from '@/lib/classes'
import { formatArea, formatFractionPct, formatPct } from '@/lib/format'

export function RegistrationSummary({ reg }: { reg: Registration }) {
  return (
    <StatGrid>
      <Stat
        label="Pixel offset"
        value={`${reg.offset_magnitude_px.toFixed(2)} px`}
        hint="alignment shift"
      />
      <Stat label="NCC" value={reg.ncc.toFixed(3)} hint="cross-correlation" />
      <Stat label="Registration score" value={formatFractionPct(reg.score)} />
      <Stat label="Phase response" value={reg.phase_response.toFixed(3)} />
    </StatGrid>
  )
}

export function FusionPanel({ data }: { data: FusionData }) {
  const rows = data.classes.map((c) => ({
    key: c.class,
    label: prettifyKey(c.class),
    color: classColor(c.class),
    fraction: c.fraction,
    valueLabel: formatPct(c.percentage),
    hint: formatArea(c.area_m2),
  }))

  return (
    <div className="space-y-5">
      <StatGrid>
        <Stat label="Method" value="Decision-level fusion" />
        <Stat label="Dominant" value={prettifyKey(data.dominant_class)} />
        <Stat
          label="Agreement"
          value={formatFractionPct(data.agreement_fraction)}
          hint="optical ∩ SAR"
        />
        <Stat
          label="Corroborated"
          value={formatFractionPct(data.corroborated_fraction)}
          hint="both sensors"
        />
      </StatGrid>

      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">Fused land cover</h4>
        <Distribution rows={rows} />
      </section>

      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">Sensor agreement</h4>
        <div className="flex h-3 w-full overflow-hidden rounded-full border border-border">
          <div
            className="bg-success"
            style={{ width: `${(data.evidence_fractions.both ?? 0) * 100}%` }}
            title="Corroborated by both sensors"
          />
          <div
            className="bg-warning"
            style={{ width: `${(data.evidence_fractions.conflict ?? 0) * 100}%` }}
            title="Conflicting evidence (arbitrated)"
          />
        </div>
        <div className="mt-1.5 flex flex-wrap gap-x-4 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-success" aria-hidden="true" />
            Both agree · {formatFractionPct(data.evidence_fractions.both ?? 0)}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-warning" aria-hidden="true" />
            Conflict · {formatFractionPct(data.evidence_fractions.conflict ?? 0)}
          </span>
        </div>
      </section>

      {data.conflicts.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-semibold text-foreground">
            Cross-modal conflicts resolved ({data.conflicts.length})
          </h4>
          <ul className="space-y-3">
            {data.conflicts.map((c, i) => (
              <li key={i} className="rounded-lg border border-border p-3">
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="font-medium text-foreground">
                    {prettifyKey(c.optical_class)}
                  </span>
                  <span className="text-muted-foreground">(optical)</span>
                  <span className="text-muted-foreground">vs</span>
                  <span className="font-medium text-foreground">{prettifyKey(c.sar_regime)}</span>
                  <span className="text-muted-foreground">(SAR)</span>
                  <span aria-hidden="true" className="text-muted-foreground">
                    →
                  </span>
                  <span className="font-semibold text-foreground">
                    {prettifyKey(c.resolved_as)}
                  </span>
                  <Badge variant="secondary">{c.arbiter} decided</Badge>
                  <span className="ml-auto text-xs text-muted-foreground">
                    {formatArea(c.area_m2)} · {formatPct(c.fraction * 100)}
                  </span>
                </div>
                <p className="mt-1.5 text-sm text-muted-foreground">{c.reason}</p>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">Co-registration</h4>
        <RegistrationSummary reg={data.registration} />
        <p className="mt-2 text-xs text-muted-foreground">
          Optical: {data.optical_only.method} (
          {data.optical_only.indices_used.map((i) => i.toUpperCase()).join(', ')}). SAR:{' '}
          {data.sar_only.polarization} ({data.sar_only.separated_regimes.join(', ')}).
        </p>
      </section>
    </div>
  )
}
