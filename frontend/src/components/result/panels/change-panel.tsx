import type { ChangeData, ChangeRegistrationGate, Artifact } from '@/types/api'
import { ComparisonSlider } from '@/components/result/comparison-slider'
import { Stat, StatGrid } from '@/components/result/stat'
import { Distribution } from '@/components/result/distribution'
import { RegistrationSummary } from '@/components/result/panels/fusion-panel'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { changeTypeColor, changeTypeLabel, prettifyKey } from '@/lib/classes'
import {
  formatExtent,
  formatFractionPct,
  formatInt,
  formatPct,
  formatSignedFractionPct,
} from '@/lib/format'
import {
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Minus,
  Ruler,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react'

function signed(n: number, digits = 0): string {
  const s = n.toFixed(digits)
  return n > 0 ? `+${s}` : s
}

function DeltaIcon({ value }: { value: number }) {
  if (value > 0)
    return <TrendingUp className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
  if (value < 0)
    return <TrendingDown className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
  return <Minus className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
}

/** Small caption that names which half of the result a section belongs to (brief §9). */
function SectionKind({ kind }: { kind: 'measured' | 'inferred' }) {
  const measured = kind === 'measured'
  return (
    <span
      className={
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ' +
        (measured
          ? 'border-info/40 bg-info/10 text-info'
          : 'border-warning/40 bg-warning/10 text-warning')
      }
    >
      {measured ? <Ruler className="h-3 w-3" aria-hidden="true" /> : null}
      {measured ? 'Measured' : 'Model-inferred'}
    </span>
  )
}

/**
 * The registration gate verdict, shown with both sides of every comparison.
 *
 * A refusal has to be demonstrable rather than asserted, so the measured value and the threshold
 * it missed are both on screen — that is what separates "the evidence is weak" from "the tool
 * broke" for a reader (brief §5, §27).
 */
function GateVerdict({ gate }: { gate: ChangeRegistrationGate }) {
  const rows: Array<[string, string, string]> = [
    [
      'Pixel offset',
      `${(gate.measured.offset_magnitude_px ?? 0).toFixed(2)} px`,
      `≤ ${(gate.thresholds.max_offset_px ?? 0).toFixed(2)} px`,
    ],
    [
      'Structural agreement (NCC)',
      (gate.measured.ncc ?? 0).toFixed(3),
      `≥ ${(gate.thresholds.min_ncc ?? 0).toFixed(2)}`,
    ],
    [
      'Registration score',
      (gate.measured.score ?? 0).toFixed(3),
      `≥ ${(gate.thresholds.min_score ?? 0).toFixed(2)}`,
    ],
  ]
  return (
    <div
      className={
        'rounded-lg border p-3 ' +
        (gate.passed ? 'border-success/40 bg-success/5' : 'border-warning/50 bg-warning/5')
      }
    >
      <p className="flex items-center gap-1.5 text-sm font-semibold">
        {gate.passed ? (
          <ShieldCheck className="h-4 w-4 text-success" aria-hidden="true" />
        ) : (
          <ShieldAlert className="h-4 w-4 text-warning" aria-hidden="true" />
        )}
        <span className={gate.passed ? 'text-success' : 'text-warning'}>
          {gate.passed
            ? 'Registration gate passed — semantic claims are permitted'
            : 'Registration gate failed — semantic claims are withheld'}
        </span>
      </p>
      <dl className="mt-2 grid gap-x-4 gap-y-1 text-xs sm:grid-cols-3">
        {rows.map(([label, value, bound]) => (
          <div key={label}>
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="tabular-nums text-foreground">
              {value} <span className="text-muted-foreground">(needs {bound})</span>
            </dd>
          </div>
        ))}
      </dl>
      {gate.reasons.length > 0 && (
        <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-muted-foreground">
          {gate.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function ChangePanel({ data, artifacts }: { data: ChangeData; artifacts?: Artifact[] }) {
  const transitions = [...data.transitions].sort(
    (a, b) => b.fraction_of_change - a.fraction_of_change,
  )
  const rows = transitions.map((t, i) => ({
    key: `${t.from}-${t.to}-${i}`,
    label: `${prettifyKey(t.from)} → ${prettifyKey(t.to)}`,
    color: changeTypeColor(t.change_type),
    fraction: t.fraction_of_change,
    valueLabel: formatPct(t.percentage_of_change),
    // Each row says whether it may be read as a conversion. Without this the table looked like a
    // list of observed conversions even when the gate had withheld every one of them.
    hint:
      formatExtent(t.area_m2, t.pixel_count) +
      (t.semantically_reportable ? '' : ' · interpretation withheld'),
  }))

  const gate = data.registration_gate
  const semantic = data.semantic
  const named = semantic?.supported ? semantic.transition : null
  const coreg = data.coregistration

  const es = data.evidence_split
  const totalEv = es.corroborated_pixels + es.spectral_only_pixels + es.class_only_pixels || 1

  // Extract before/after images from artifacts for the comparison slider
  const baseArtifact = artifacts?.find((a) => a.kind === 'base')
  const overlayArtifact = artifacts?.find((a) => a.kind === 'change_mask' || a.kind === 'overlay')

  return (
    <div className="space-y-5">
      {/* Before/After comparison slider */}
      {baseArtifact && overlayArtifact && (
        <section>
          <h4 className="mb-2 text-sm font-semibold text-foreground">Before / After comparison</h4>
          <ComparisonSlider
            beforeSrc={baseArtifact.url}
            afterSrc={overlayArtifact.url}
            beforeLabel="Original"
            afterLabel="Change overlay"
          />
        </section>
      )}

      {/*
       * The two halves of the result are kept visually apart, because the reason the earlier page
       * was unsafe was not a wrong number — it was that a measured pixel population and a named
       * land-cover conversion looked identical on screen (brief §9).
       */}
      {data.question && (
        <div className="rounded-lg border border-border bg-muted/30 p-3">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Question answered
          </p>
          <p className="mt-1 text-sm text-foreground">{data.question.asked}</p>
          {/* Read as *what*, so a question the tool misread is visible here and not only implied
              by an answer that quietly addresses something else. */}
          <p className="mt-1 text-xs text-muted-foreground">
            Read as: {prettifyKey(data.question.interpreted_as)}
          </p>
        </div>
      )}

      <section>
        <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold text-foreground">
          Measured pixel difference
          <SectionKind kind="measured" />
        </h4>
        <p className="mb-2 text-xs text-muted-foreground">
          Counted directly from the two dates by {data.detectors.join(' + ')}. True regardless of
          what the change means.
        </p>
        <StatGrid>
          <Stat
            label="Changed area"
            // Without a pixel size there is no ground area, so the headline is the measurement that
            // does exist. It used to headline "—" and bury the 100,620 measured pixels in the hint.
            value={formatExtent(data.changed_area_m2, data.changed_pixels)}
            hint={
              data.changed_area_m2 === null
                ? 'pixel count — no georeference, so no ground area'
                : `${formatInt(data.changed_pixels)} px`
            }
          />
          <Stat label="Changed" value={formatPct(data.changed_percentage)} hint="of valid pixels" />
          <Stat
            label="Bimodality"
            value={formatFractionPct(data.bimodality)}
            hint="change/no-change split"
          />
          <Stat
            label="Threshold quality"
            value={formatFractionPct(data.threshold_quality)}
            hint={data.threshold_method}
          />
        </StatGrid>
      </section>

      {gate && <GateVerdict gate={gate} />}

      <section>
        <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold text-foreground">
          Model-inferred semantic change
          <SectionKind kind="inferred" />
        </h4>
        {named ? (
          <div className="rounded-lg border-l-4 border-primary bg-accent/40 p-3">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Land-cover conversion
            </p>
            <p className="mt-1 flex flex-wrap items-center gap-1.5 text-sm">
              <span className="font-semibold text-foreground">{prettifyKey(named.from)}</span>
              <ArrowRight className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              <span className="font-semibold text-foreground">{prettifyKey(named.to)}</span>
              <span
                className="ml-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium"
                style={{
                  backgroundColor: `${changeTypeColor(named.change_type)}22`,
                  color: changeTypeColor(named.change_type),
                }}
              >
                {changeTypeLabel(named.change_type)}
              </span>
              <span className="text-muted-foreground">
                — {formatPct(named.percentage_of_change)} of change ·{' '}
                {formatExtent(named.area_m2, named.pixel_count)}
              </span>
            </p>
            <p className="mt-1.5 text-xs text-muted-foreground">
              Named because the independent spectral detector corroborates{' '}
              {formatFractionPct(named.spectral_agreement)} of its pixels, at or above the{' '}
              {formatFractionPct(semantic.min_spectral_agreement)} required.
            </p>
          </div>
        ) : (
          <div className="rounded-lg border border-warning/50 bg-warning/5 p-3">
            <p className="flex items-center gap-1.5 text-sm font-semibold text-warning">
              <ShieldAlert className="h-4 w-4" aria-hidden="true" />
              No land-cover conversion is claimed
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {semantic?.withheld_reason ??
                'The evidence does not support attributing the measured difference to a named ' +
                  'land-cover conversion.'}
            </p>
            {data.dominant_transition && (
              <p className="mt-2 text-xs text-muted-foreground">
                The largest changed pixel population is{' '}
                <span className="font-medium text-foreground">
                  {prettifyKey(data.dominant_transition.from)} →{' '}
                  {prettifyKey(data.dominant_transition.to)}
                </span>{' '}
                ({formatPct(data.dominant_transition.percentage_of_change)} of change,{' '}
                {formatFractionPct(data.dominant_transition.spectral_agreement)} spectral
                agreement). That is a measurement of where the two classifications differ, not an
                observed conversion.
              </p>
            )}
          </div>
        )}
      </section>

      <section>
        <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold text-foreground">
          Transitions
          <SectionKind kind="measured" />
        </h4>
        <Distribution rows={rows} />
      </section>

      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">Change evidence</h4>
        <div className="flex h-3 w-full overflow-hidden rounded-full border border-border">
          <div
            className="bg-success"
            style={{ width: `${(es.corroborated_pixels / totalEv) * 100}%` }}
            title="Corroborated (class + spectral)"
          />
          <div
            className="bg-info"
            style={{ width: `${(es.spectral_only_pixels / totalEv) * 100}%` }}
            title="Spectral magnitude only"
          />
          <div
            className="bg-warning"
            style={{ width: `${(es.class_only_pixels / totalEv) * 100}%` }}
            title="Class change only"
          />
        </div>
        <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-success" aria-hidden="true" />
            Corroborated · {formatFractionPct(es.corroborated_fraction)}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-info" aria-hidden="true" />
            Spectral only
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-warning" aria-hidden="true" />
            Class only
          </span>
        </div>
      </section>

      {data.class_deltas.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-semibold text-foreground">Class area change</h4>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Class</TableHead>
                <TableHead className="text-right">Before</TableHead>
                <TableHead className="text-right">After</TableHead>
                <TableHead className="text-right">Δ</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.class_deltas.map((d) => (
                <TableRow key={d.class}>
                  <TableCell className="font-medium text-foreground">
                    {prettifyKey(d.class)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {formatExtent(d.area_m2_before, d.pixels_before)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {formatExtent(d.area_m2_after, d.pixels_after)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {/*
                     * The arrow reads `pixel_delta`, which is measured for every pair, and the
                     * percentage carries its sign. Driving the arrow from `area_m2_delta` meant a
                     * non-georeferenced pair (null area) fell through to the "no change" dash, so a
                     * class that grew by 115.3% rendered as "— 115.3%" — indistinguishable from a
                     * loss.
                     */}
                    <span className="inline-flex items-center justify-end gap-1">
                      <DeltaIcon value={d.pixel_delta} />
                      {formatSignedFractionPct(d.relative_delta)}
                    </span>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </section>
      )}

      {data.index_deltas.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-semibold text-foreground">Index change</h4>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Index</TableHead>
                <TableHead className="text-right">Before</TableHead>
                <TableHead className="text-right">After</TableHead>
                <TableHead className="text-right">Δ in change</TableHead>
                <TableHead>Direction</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.index_deltas.map((d) => (
                <TableRow key={d.index}>
                  <TableCell className="font-medium uppercase text-foreground">{d.index}</TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {d.mean_before.toFixed(3)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {d.mean_after.toFixed(3)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {signed(d.mean_delta_in_change, 3)}
                  </TableCell>
                  {/*
                   * Direction is a *count* of changed pixels moving each way, not the sign of the
                   * mean beside it — that is the whole point of carrying it, since offsetting gains
                   * and losses cancel a mean to nearly nothing. Shown alone it looked like a
                   * contradiction ("+0.009 … Decrease"), so the measurement behind it is stated.
                   */}
                  <TableCell className="text-muted-foreground">
                    {prettifyKey(d.direction)}
                    <span className="ml-1 text-xs">
                      · {formatFractionPct(d.increase_fraction)} of changed px rose
                    </span>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </section>
      )}

      <section>
        <h4 className="mb-2 text-sm font-semibold text-foreground">Co-registration</h4>
        <RegistrationSummary reg={data.registration} />
        {coreg && (
          <p className="mt-2 text-xs text-muted-foreground">
            {coreg.applied ? (
              <>
                Correction applied: <span className="text-foreground">{coreg.method}</span>, shift (
                {coreg.shift_x_px.toFixed(2)}, {coreg.shift_y_px.toFixed(2)}) px. Registration score{' '}
                {coreg.initial.score.toFixed(3)} → {coreg.final.score.toFixed(3)} (
                {signed(coreg.score_improvement, 3)}).
              </>
            ) : (
              <>
                No correction applied ({coreg.method}); the pair was already aligned to{' '}
                {coreg.final.offset_magnitude_px.toFixed(2)} px.
              </>
            )}
            {coreg.candidates.length > 1 && (
              <> {coreg.candidates.length} methods were evaluated and the best-scoring one kept.</>
            )}
          </p>
        )}
        <p className="mt-2 text-xs text-muted-foreground">
          Method: {data.method}. Detectors: {data.detectors.join(', ')}. Features:{' '}
          {data.features_used.map((f) => f.toUpperCase()).join(', ')}.
        </p>
      </section>
    </div>
  )
}
