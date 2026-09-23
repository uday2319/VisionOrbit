import type { ConfidenceReport } from '@/types/api'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { CheckCircle2, AlertTriangle, ShieldCheck, Info } from 'lucide-react'
import { confidenceBadgeVariant, formatFractionPct } from '@/lib/format'
import { prettifyKey } from '@/lib/classes'
import { cn } from '@/lib/utils'

const INDICATOR_CLASS: Record<string, string> = {
  high: 'bg-success',
  medium: 'bg-info',
  low: 'bg-warning',
  insufficient: 'bg-muted-foreground',
}

export function ConfidenceMeter({ confidence }: { confidence: ConfidenceReport }) {
  const { level, score, sufficient, limiting_factor, reasons, factors } = confidence
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <ShieldCheck className="h-4 w-4 text-primary" aria-hidden="true" />
            Confidence
          </CardTitle>
          <Badge variant={confidenceBadgeVariant(level)}>{prettifyKey(level)}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Evidence score</span>
            <span className="font-semibold tabular-nums text-foreground">
              {formatFractionPct(score)}
            </span>
          </div>
          <Progress
            value={Math.round(score * 100)}
            indicatorClassName={INDICATOR_CLASS[level] ?? 'bg-primary'}
          />
        </div>

        <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
          <span className="inline-flex items-center gap-1.5">
            {sufficient ? (
              <CheckCircle2 className="h-4 w-4 text-success" aria-hidden="true" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-warning" aria-hidden="true" />
            )}
            {sufficient ? 'Evidence sufficient' : 'Evidence insufficient'}
          </span>
          {limiting_factor && (
            <span className="text-muted-foreground">
              Limiting factor:{' '}
              <span className="font-medium text-foreground">{prettifyKey(limiting_factor)}</span>
            </span>
          )}
        </div>

        {reasons.length > 0 && (
          <ul className="space-y-1 text-sm text-muted-foreground">
            {reasons.map((reason, i) => (
              <li key={i} className="flex gap-2">
                <span aria-hidden="true" className="text-muted-foreground/60">
                  •
                </span>
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        )}

        {factors.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Contributing factors
            </p>
            <ul className="space-y-2">
              {factors.map((factor) => (
                <li key={factor.name} className="space-y-1">
                  <div className="flex items-center justify-between gap-2 text-sm">
                    <span className="inline-flex items-center gap-1.5 font-medium text-foreground">
                      {prettifyKey(factor.name)}
                      <Badge variant={factor.kind === 'gate' ? 'warning' : 'secondary'}>
                        {factor.kind}
                      </Badge>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <button
                            type="button"
                            className="text-muted-foreground hover:text-foreground"
                            aria-label={`Why: ${factor.name}`}
                          >
                            <Info className="h-3.5 w-3.5" />
                          </button>
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">{factor.reason}</TooltipContent>
                      </Tooltip>
                    </span>
                    <span className="tabular-nums text-muted-foreground">
                      {formatFractionPct(factor.value)}
                      <span className="ml-1 text-xs">· w {factor.weight.toFixed(2)}</span>
                    </span>
                  </div>
                  <Progress
                    value={Math.round(factor.value * 100)}
                    className="h-1.5"
                    indicatorClassName={cn(factor.kind === 'gate' ? 'bg-warning' : 'bg-primary/70')}
                  />
                </li>
              ))}
            </ul>
          </div>
        )}

        <p className="border-t border-border pt-3 text-xs text-muted-foreground">
          Confidence is computed from measured evidence — separability, corroboration, registration,
          input quality — not generated by a language model.
        </p>
      </CardContent>
    </Card>
  )
}
