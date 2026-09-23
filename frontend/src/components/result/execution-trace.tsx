import type { TraceStep, StepStatus } from '@/types/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { prettifyKey } from '@/lib/classes'
import { formatDuration, formatStepDuration } from '@/lib/format'
import { cn } from '@/lib/utils'
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  MinusCircle,
  ListChecks,
  type LucideIcon,
} from 'lucide-react'

const STEP_LABELS: Record<string, string> = {
  classify: 'Classify query',
  validate: 'Validate input',
  preprocess: 'Preprocess',
  execute: 'Execute tool',
  score: 'Score confidence',
  verify: 'Verify evidence',
}

const STATUS_META: Record<StepStatus, { icon: LucideIcon; className: string; label: string }> = {
  ok: { icon: CheckCircle2, className: 'text-success', label: 'OK' },
  warning: { icon: AlertTriangle, className: 'text-warning', label: 'Warning' },
  failed: { icon: XCircle, className: 'text-destructive', label: 'Failed' },
  skipped: { icon: MinusCircle, className: 'text-muted-foreground', label: 'Skipped' },
}

function formatParamValue(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function ExecutionTrace({ trace }: { trace: TraceStep[] }) {
  // Only measured steps are summed, and how many were not timed is stated rather than papered over.
  // Substituting a nominal 25 ms for each untimed step used to push this total past the backend's
  // own measured duration shown in the page header — two numbers for one run, neither explained.
  const measured = trace.filter((s) => typeof s.duration_ms === 'number')
  const total = measured.reduce((sum, s) => sum + (s.duration_ms ?? 0), 0)
  const untimed = trace.length - measured.length

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <ListChecks className="h-4 w-4 text-primary" aria-hidden="true" />
            Execution trace
          </CardTitle>
          <span className="text-xs text-muted-foreground">
            {trace.length} steps · {formatDuration(total)} measured
            {untimed > 0 && ` · ${untimed} not timed separately`}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <ol className="relative space-y-4 border-l border-border pl-6">
          {trace.map((step, i) => {
            const meta = STATUS_META[step.status as StepStatus] ?? STATUS_META.skipped
            const Icon = meta.icon
            const params = Object.entries(step.parameters ?? {})
            return (
              <li key={`${step.step}-${i}`} className="relative">
                <span
                  className={cn(
                    'absolute -left-[31px] flex h-5 w-5 items-center justify-center rounded-full bg-background',
                    meta.className,
                  )}
                >
                  <Icon className="h-4 w-4" aria-hidden="true" />
                </span>
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="text-sm font-semibold text-foreground">
                    {STEP_LABELS[step.step] ?? prettifyKey(String(step.step))}
                  </span>
                  {step.tool && (
                    <Badge variant="secondary" className="font-mono text-[11px]">
                      {step.tool}
                    </Badge>
                  )}
                  <span className="text-xs text-muted-foreground">
                    {meta.label} · {formatStepDuration(step.duration_ms)}
                  </span>
                </div>
                {step.summary && (
                  <p className="mt-0.5 text-sm text-muted-foreground">{step.summary}</p>
                )}
                {step.warnings.length > 0 && (
                  <ul className="mt-1 space-y-0.5">
                    {step.warnings.map((w, wi) => (
                      <li
                        key={wi}
                        // `--warning-foreground` is the colour that sits *on* a filled `bg-warning`
                        // chip; in dark mode it is near-black. Used here, over the card's own dark
                        // background, it made every step warning unreadable — the caveats a run
                        // depends on being read, silently invisible. Plain foreground text with the
                        // warning-coloured icon carries the same signal at full contrast.
                        className="flex items-start gap-1.5 text-xs text-foreground/90"
                      >
                        <AlertTriangle
                          className="mt-0.5 h-3 w-3 shrink-0 text-warning"
                          aria-hidden="true"
                        />
                        <span>{w}</span>
                      </li>
                    ))}
                  </ul>
                )}
                {params.length > 0 && (
                  <details className="mt-1 text-xs">
                    <summary className="cursor-pointer select-none text-muted-foreground hover:text-foreground">
                      Parameters
                    </summary>
                    <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 rounded-md border border-border bg-muted/30 p-2">
                      {params.map(([k, v]) => (
                        <div key={k} className="contents">
                          <dt className="font-mono text-muted-foreground">{k}</dt>
                          <dd className="break-all font-mono text-foreground">
                            {formatParamValue(v)}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  </details>
                )}
              </li>
            )
          })}
        </ol>
        <p className="mt-4 border-t border-border pt-3 text-xs text-muted-foreground">
          This is the tool pipeline the assistant ran — classification, routing, execution and
          evidence checks. Internal model reasoning is not shown.
        </p>
      </CardContent>
    </Card>
  )
}
