import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Sparkles, Info, Loader2 } from 'lucide-react'
import type { InputMode } from '@/types/api'
import { analyze, apiConfig, ApiRequestError } from '@/services/api'
import { MODE_SLOTS, MODE_EXAMPLE_QUERIES, validateFile } from '@/lib/uploads'
import { PageHeader } from '@/components/layout/page-header'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { ModeSelector } from '@/components/analysis/mode-selector'
import { FileDropzone } from '@/components/analysis/file-dropzone'
import { DemoGallery } from '@/components/analysis/demo-gallery'

const INITIAL_MODE: InputMode = 'single_optical'

export function NewAnalysisPage() {
  const navigate = useNavigate()
  const [mode, setMode] = useState<InputMode>(INITIAL_MODE)
  const [files, setFiles] = useState<(File | null)[]>(() =>
    MODE_SLOTS[INITIAL_MODE].map(() => null),
  )
  const [fileErrors, setFileErrors] = useState<(string | null)[]>(() =>
    MODE_SLOTS[INITIAL_MODE].map(() => null),
  )
  const [query, setQuery] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<ApiRequestError | null>(null)

  const slots = MODE_SLOTS[mode]

  function handleModeChange(next: InputMode) {
    setMode(next)
    setFiles(MODE_SLOTS[next].map(() => null))
    setFileErrors(MODE_SLOTS[next].map(() => null))
    setSubmitError(null)
  }

  function handleFile(index: number, file: File | null) {
    setFiles((prev) => {
      const copy = [...prev]
      copy[index] = file
      return copy
    })
    setFileErrors((prev) => {
      const copy = [...prev]
      copy[index] = file ? validateFile(file) : null
      return copy
    })
    setSubmitError(null)
  }

  const allFilesProvided = slots.every((_, i) => Boolean(files[i] && typeof (files[i] as any).name === 'string'))
  const anyFileError = fileErrors.some((e) => e !== null)
  const queryProvided = query.trim().length > 0
  const canSubmit = allFilesProvided && !anyFileError && queryProvided && !submitting

  function missingHint(): string | null {
    if (!allFilesProvided) {
      const missing = slots.filter((_, i) => !Boolean(files[i] && typeof (files[i] as any).name === 'string'))
      return `Add ${missing.map((s) => s.label.toLowerCase()).join(' and ')}.`
    }
    if (anyFileError) return 'Fix the file errors above.'
    if (!queryProvided) return 'Enter a question to ask about the imagery.'
    return null
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    if (!canSubmit) return
    const selected = files.filter((f): f is File => Boolean(f && typeof (f as any).name === 'string'))
    setSubmitting(true)
    setSubmitError(null)
    try {
      const result = await analyze({ mode, query: query.trim(), files: selected })
      navigate(`/analysis/${result.id}`)
    } catch (err: any) {
      if (err instanceof ApiRequestError || (err && (err.errorCode || err.error_code))) {
        setSubmitError(err as ApiRequestError)
      } else {
        setSubmitError(
          new ApiRequestError({
            message: err?.message || 'Something went wrong submitting the analysis. Please try again.',
            error_code: err?.errorCode || err?.error_code || 'unknown_error',
            recoverable: true,
          }),
        )
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="New analysis"
        description="Pick an input mode, provide the imagery, and ask a question. Every answer is grounded in measured evidence — nothing is fabricated."
      />

      {apiConfig.offline && (
        <Alert variant="info">
          <Info className="h-4 w-4" />
          <AlertTitle>Offline demo mode</AlertTitle>
          <AlertDescription>
            No live backend is configured (VITE_USE_FIXTURES=true), so uploaded imagery cannot be
            analysed here — the demo will not invent a result. Explore a recorded analysis below to
            see genuine output, or start the API server for live runs.
          </AlertDescription>
        </Alert>
      )}

      <form onSubmit={handleSubmit} className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>1 · Choose input mode</CardTitle>
            </CardHeader>
            <CardContent>
              <ModeSelector value={mode} onChange={handleModeChange} disabled={submitting} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>
                2 · Provide imagery{' '}
                <span className="text-sm font-normal text-muted-foreground">
                  ({slots.length === 1 ? '1 file' : `${slots.length} files`})
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {slots.map((slot, i) => (
                <FileDropzone
                  key={slot.key}
                  slot={slot}
                  file={files[i] ?? null}
                  error={fileErrors[i] ?? null}
                  disabled={submitting}
                  onSelect={(file) => handleFile(i, file)}
                />
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>3 · Ask a question</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="query">Query</Label>
                <Textarea
                  id="query"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. What land cover types are present in this scene?"
                  rows={3}
                  disabled={submitting}
                />
              </div>
              <div className="flex flex-wrap gap-2">
                {MODE_EXAMPLE_QUERIES[mode].map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => setQuery(example)}
                    disabled={submitting}
                    className="rounded-full border border-border bg-muted/50 px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground disabled:opacity-50"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4 lg:sticky lg:top-24 lg:self-start">
          <Card>
            <CardHeader>
              <CardTitle>Run</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {submitError && (
                <Alert variant={submitError.recoverable ? 'warning' : 'destructive'}>
                  <AlertTitle>
                    {submitError.errorCode === 'backend_required'
                      ? 'Live backend required'
                      : 'Could not run analysis'}
                  </AlertTitle>
                  <AlertDescription>{submitError.message}</AlertDescription>
                </Alert>
              )}
              <Button type="submit" className="w-full" disabled={!canSubmit}>
                {submitting ? (
                  <>
                    <Loader2 className="animate-spin" />
                    Analysing…
                  </>
                ) : (
                  <>
                    <Sparkles />
                    Run analysis
                  </>
                )}
              </Button>
              {!submitting && missingHint() && (
                <p className="text-center text-xs text-muted-foreground">{missingHint()}</p>
              )}
              <Separator />
              <DemoGallery mode={mode} onUseExample={(q) => setQuery(q)} />
            </CardContent>
          </Card>
        </div>
      </form>
    </div>
  )
}
