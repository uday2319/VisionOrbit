import { useCallback, useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Printer, Download, AlertCircle, Loader2 } from 'lucide-react'
import type { Artifact } from '@/types/api'
import { getAnalysis } from '@/services/api'
import { useAsync } from '@/lib/use-async'
import { buildReportModel } from '@/lib/report-model'
import { renderReportHtml } from '@/lib/report-html'
import { ReportDocument } from '@/components/report/report-document'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Skeleton } from '@/components/ui/skeleton'

/** Fetch a same-origin artifact and encode it as a base64 data URI. */
async function toDataUri(url: string): Promise<string> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`fetch ${url} → ${res.status}`)
  const blob = await res.blob()
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error ?? new Error('read failed'))
    reader.readAsDataURL(blob)
  })
}

/**
 * Resolve each artifact to an inline data URI so the downloaded HTML is truly
 * self-contained. If an image can't be inlined we fall back to its absolute URL
 * (still opens while the app is served) and warn — we never silently drop it.
 */
async function embedArtifacts(artifacts: Artifact[]): Promise<Map<string, string>> {
  const map = new Map<string, string>()
  await Promise.all(
    artifacts.map(async (a) => {
      try {
        map.set(a.url, await toDataUri(a.url))
      } catch (err) {
        console.warn('[report] could not inline artifact, using absolute URL:', a.url, err)
        map.set(a.url, new URL(a.url, window.location.origin).href)
      }
    }),
  )
  return map
}

export function ReportPage() {
  const { id } = useParams<{ id: string }>()
  // Same marker the result page uses: report the recording when the recording is what was opened.
  const [search] = useSearchParams()
  const recorded = search.get('demo') === '1'
  const { data: result, error, loading } = useAsync(
    () => getAnalysis(id ?? '', { recorded }),
    [id, recorded],
  )
  // Stamp the report once, so the on-screen document and the downloaded file agree.
  const [generatedAt] = useState(() => new Date().toISOString())
  const [downloading, setDownloading] = useState(false)

  const model = useMemo(
    () => (result ? buildReportModel(result, generatedAt) : null),
    [result, generatedAt],
  )

  const handleDownload = useCallback(async () => {
    if (!model) return
    setDownloading(true)
    try {
      const embedded = await embedArtifacts(model.artifacts)
      const html = renderReportHtml(model, (a) => embedded.get(a.url) ?? a.url)
      const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${model.id}.html`
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } finally {
      setDownloading(false)
    }
  }, [model])

  return (
    <div className="min-h-screen bg-slate-100 print:bg-white">
      {/* Action bar — omitted from print output. */}
      <div className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur print:hidden">
        <div className="mx-auto flex max-w-[820px] flex-wrap items-center justify-between gap-3 px-4 py-3">
          <Button variant="ghost" size="sm" asChild>
            <Link to={`/analysis/${id}${recorded ? '?demo=1' : ''}`}>
              <ArrowLeft />
              Back to result
            </Link>
          </Button>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
              disabled={!model || downloading}
            >
              {downloading ? <Loader2 className="animate-spin" /> : <Download />}
              Download HTML
            </Button>
            <Button size="sm" onClick={() => window.print()} disabled={!model}>
              <Printer />
              Print / Save PDF
            </Button>
          </div>
        </div>
      </div>

      {loading && (
        <div className="mx-auto my-8 max-w-[820px] space-y-4 rounded-lg bg-white p-8 shadow-xl sm:p-10">
          <Skeleton className="h-10 w-2/3" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      )}

      {!loading && (error || !model) && (
        <div className="mx-auto my-8 max-w-[820px] px-4">
          <Alert variant="destructive">
            <AlertCircle />
            <AlertTitle>Could not load this report</AlertTitle>
            <AlertDescription>
              {error?.message ?? 'The requested analysis was not found.'}
            </AlertDescription>
          </Alert>
          <Button asChild variant="outline" className="mt-4">
            <Link to="/">
              <ArrowLeft />
              Back to dashboard
            </Link>
          </Button>
        </div>
      )}

      {!loading && model && <ReportDocument model={model} />}
    </div>
  )
}
