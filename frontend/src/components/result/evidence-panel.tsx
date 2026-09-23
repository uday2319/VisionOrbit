import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ClipboardCheck, TriangleAlert } from 'lucide-react'

interface EvidencePanelProps {
  evidence: string[]
  warnings: string[]
}

/**
 * The evidence notes and warnings that back the answer. These are surfaced
 * verbatim from the tool output — they are the audit trail behind the
 * evidence-based confidence score (brief §27), not model commentary.
 */
export function EvidencePanel({ evidence, warnings }: EvidencePanelProps) {
  const hasEvidence = evidence.length > 0
  const hasWarnings = warnings.length > 0

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <ClipboardCheck className="h-4 w-4 text-primary" aria-hidden="true" />
          Evidence
          {hasWarnings && (
            <Badge variant="warning" className="ml-auto">
              {warnings.length} warning{warnings.length === 1 ? '' : 's'}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {hasEvidence ? (
          <ul className="space-y-1.5 text-sm text-foreground">
            {evidence.map((item, i) => (
              <li key={i} className="flex gap-2">
                <span
                  aria-hidden="true"
                  className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary"
                />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground">No additional evidence notes.</p>
        )}

        {hasWarnings && (
          <div className="space-y-1.5 border-t border-border pt-3">
            {warnings.map((w, i) => (
              <p key={i} className="flex items-start gap-1.5 text-sm text-muted-foreground">
                <TriangleAlert
                  className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning"
                  aria-hidden="true"
                />
                <span>{w}</span>
              </p>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
