import { Link, useParams, useSearchParams } from 'react-router-dom'
import { FileText, Plus, AlertCircle, ArrowLeft, Database } from 'lucide-react'
import { getAnalysis } from '@/services/api'
import { useAsync } from '@/lib/use-async'
import { modeLabel, taskLabel, formatDuration, formatDateTime } from '@/lib/format'
import { PageHeader } from '@/components/layout/page-header'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { AnswerCard } from '@/components/result/answer-card'
import { ResultMap } from '@/components/result/result-map'
import { DataPanel } from '@/components/result/data-panel'
import { ExecutionTrace } from '@/components/result/execution-trace'
import { ConfidenceMeter } from '@/components/result/confidence-meter'
import { EvidencePanel } from '@/components/result/evidence-panel'
import { InputsPanel } from '@/components/result/inputs-panel'

export function ResultPage() {
  const { id } = useParams<{ id: string }>()
  // `?demo=1` (the dashboard's "View example" links) names the shipped recording for this id, which
  // the backend's own run numbering can otherwise collide with.
  const [search] = useSearchParams()
  const recorded = search.get('demo') === '1'
  const { data: result, error, loading } = useAsync(
    () => getAnalysis(id ?? '', { recorded }),
    [id, recorded],
  )

  if (loading) {
    return (
      <div className="space-y-6">
        <PageHeader title="Analysis result" description={id} />
        <Skeleton className="h-28 w-full" />
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
            <Skeleton className="h-[30rem] w-full" />
            <Skeleton className="h-64 w-full" />
          </div>
          <div className="space-y-6">
            <Skeleton className="h-64 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        </div>
      </div>
    )
  }

  if (error || !result) {
    return (
      <div className="space-y-6">
        <PageHeader title="Analysis result" description={id} />
        <Alert variant="destructive">
          <AlertCircle />
          <AlertTitle>Could not load this analysis</AlertTitle>
          <AlertDescription>
            {error?.message ?? 'The requested analysis was not found.'}
          </AlertDescription>
        </Alert>
        <Button asChild variant="outline">
          <Link to="/">
            <ArrowLeft />
            Back to dashboard
          </Link>
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title={result.title}
        description={result.note}
        actions={
          <>
            <Button asChild variant="outline">
              {/* Carry the marker through, so the report is built from the same analysis as the page. */}
              <Link to={`/analysis/${result.id}/report${recorded ? '?demo=1' : ''}`}>
                <FileText />
                View report
              </Link>
            </Button>
            <Button asChild>
              <Link to="/analyze">
                <Plus />
                New analysis
              </Link>
            </Button>
          </>
        }
      />

      {/* Meta bar */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-sm">
        <span className="font-mono text-xs text-muted-foreground">{result.id}</span>
        <Badge variant="outline">{modeLabel(result.mode)}</Badge>
        <Badge variant="outline">{taskLabel(result.task)}</Badge>
        <span className="text-muted-foreground">
          Tool <span className="font-medium text-foreground">{result.tool}</span> · {result.tier}
        </span>
        <span className="text-muted-foreground">· {formatDuration(result.duration_ms)}</span>
        <span className="text-muted-foreground">· {formatDateTime(result.created_at)}</span>
        {result.fixture && (
          <Badge variant="secondary" className="gap-1">
            <Database className="h-3 w-3" aria-hidden="true" />
            recorded run
          </Badge>
        )}
      </div>

      <AnswerCard result={result} />

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Imagery</CardTitle>
            </CardHeader>
            <CardContent>
              <ResultMap artifacts={result.artifacts} />
            </CardContent>
          </Card>

          <DataPanel data={result.data} artifacts={result.artifacts} />

          <ExecutionTrace trace={result.trace} />
        </div>

        <aside className="space-y-6">
          <ConfidenceMeter confidence={result.confidence} />
          <EvidencePanel evidence={result.evidence} warnings={result.warnings} />
          <InputsPanel inputs={result.inputs} />
        </aside>
      </div>
    </div>
  )
}
