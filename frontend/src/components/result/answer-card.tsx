import type { AnalysisResult } from '@/types/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { MessageSquareQuote, AlertTriangle } from 'lucide-react'
import { statusBadgeVariant } from '@/lib/format'
import { prettifyKey } from '@/lib/classes'

export function AnswerCard({ result }: { result: AnalysisResult }) {
  const withheld = result.answer_withheld !== null
  const unreliable = !result.confidence.sufficient
  const { limiting_factor, reasons } = result.confidence

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <MessageSquareQuote className="h-4 w-4 text-primary" aria-hidden="true" />
            Answer
          </CardTitle>
          <Badge variant={statusBadgeVariant(result.status)}>{prettifyKey(result.status)}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <blockquote className="border-l-2 border-border pl-3 text-sm italic text-muted-foreground">
          “{result.query}”
        </blockquote>

        {withheld ? (
          <Alert variant="warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Answer withheld</AlertTitle>
            <AlertDescription>{result.answer}</AlertDescription>
          </Alert>
        ) : (
          <p className="text-base leading-relaxed text-foreground">{result.answer}</p>
        )}

        {/*
         * A weak-evidence caveat sits *beside* the measurements, not on top of them. Putting the
         * heading "Insufficient evidence for a reliable conclusion" above the answer text made one
         * card both decline to conclude and then conclude — over numbers that were genuinely
         * measured. What is insufficient is the support for treating them as a finding, and the
         * measurement that caused that is named here rather than left to be guessed at.
         */}
        {unreliable && !withheld && (
          <Alert variant="warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>These measurements do not support a reliable conclusion</AlertTitle>
            <AlertDescription>
              {limiting_factor && (
                <>
                  Limiting factor: <strong>{prettifyKey(limiting_factor)}</strong>.{' '}
                </>
              )}
              {reasons[0] ?? 'The measured evidence was too weak to state a finding.'} Treat the
              figures below as measurements of these two images, not as a finding about the ground.
            </AlertDescription>
          </Alert>
        )}

        {result.answer_withheld && (
          <div className="rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
            <p className="mb-1 font-medium text-foreground">Withheld draft</p>
            <p className="italic">“{result.answer_withheld}”</p>
            <p className="mt-1 text-xs">
              Not released — the measured evidence did not meet the threshold to state this.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
