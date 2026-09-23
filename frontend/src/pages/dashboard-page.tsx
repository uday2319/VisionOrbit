import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ArrowUp,
  FileImage,
  Map,
  Search,
  Send,
  Upload,
  X,
} from 'lucide-react'

import { analyze, ApiRequestError } from '@/services/api'
import { validateFile } from '@/lib/uploads'

const quickExamples = [
  { label: 'Buildings', query: 'Show buildings' },
  { label: 'Water', query: 'Detect water bodies' },
  { label: 'Vegetation', query: 'Highlight vegetation' },
  { label: 'NDVI', query: 'Calculate NDVI' },
  { label: 'Changes', query: 'Detect changes' },
]

const recentQueries = [
  'Show buildings in this area',
  'Calculate NDVI for this region',
  'What changed between these dates?',
  'Detect water bodies',
  'Is there urban expansion here?',
]

export function DashboardPage() {
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  // Create a browser preview for PNG/JPG images.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }

    const isBrowserImage =
      file.type.startsWith('image/') &&
      !file.name.toLowerCase().endsWith('.tif') &&
      !file.name.toLowerCase().endsWith('.tiff') &&
      !file.name.toLowerCase().endsWith('.jp2')

    // GeoTIFF and JP2 files are handled by the backend.
    if (!isBrowserImage) {
      setPreviewUrl(null)
      return
    }

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)

    return () => URL.revokeObjectURL(url)
  }, [file])

  // Validate and store the selected file.
  function handleFile(selected: File | null) {
    if (!selected) return

    const error = validateFile(selected)

    setFileError(error)
    setSubmitError(null)

    if (!error) {
      setFile(selected)
    } else {
      setFile(null)
    }
  }

  // Remove the selected image.
  function removeFile() {
    setFile(null)
    setFileError(null)
    setPreviewUrl(null)

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Put a predefined question into the query box.
  function useExample(exampleQuery: string) {
    setQuery(exampleQuery)
    setSubmitError(null)
  }

  // Send image + query to the existing backend.
  async function handleSubmit() {
    if (!file) {
      setSubmitError('Upload a satellite image first.')
      return
    }

    if (!query.trim()) {
      setSubmitError('Enter a question about the image.')
      return
    }

    setSubmitting(true)
    setSubmitError(null)

    try {
      const result = await analyze({
        mode: 'single_optical',
        query: query.trim(),
        files: [file],
      })

      navigate(`/analysis/${result.id}`)
    } catch (error) {
      if (error instanceof ApiRequestError) {
        setSubmitError(error.message)
      } else {
        setSubmitError('Could not run the analysis.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="h-[calc(100vh-4rem)] min-h-[650px]">

      {/* =========================================================
          PAGE HEADER
      ========================================================== */}
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-slate-900">
          Command Center
        </h1>

        <p className="mt-1 text-sm text-slate-500">
          Analyze satellite imagery using natural language
        </p>
      </div>

      {/* =========================================================
          MAIN DASHBOARD
      ========================================================== */}
      <div className="grid h-[calc(100%-4.5rem)] min-h-[570px] grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_500px]">

        {/* =======================================================
            LEFT COLUMN
        ======================================================== */}
        <section className="flex min-h-0 flex-col gap-3">

          {/* =====================================================
              UPLOADED IMAGE
          ====================================================== */}
          <div className="vision-card relative h-[52%] min-h-[330px] overflow-hidden">

            {/* Image indicator */}
            <div className="absolute left-4 top-4 z-20 flex overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
              <button
                type="button"
                className="flex items-center gap-2 bg-blue-600 px-4 py-2.5 text-sm font-medium text-white"
              >
                <Map className="h-4 w-4" />
                Image
              </button>
            </div>

            {/* Uploaded image */}
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Uploaded satellite imagery"
                className="h-full w-full bg-slate-100 object-contain"
              />
            ) : file ? (
              /* GeoTIFF / JP2 fallback */
              <div className="flex h-full items-center justify-center bg-slate-100">
                <div className="max-w-md px-6 text-center">

                  <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-xl bg-white shadow-sm">
                    <FileImage className="h-8 w-8 text-blue-600" />
                  </div>

                  <p className="truncate text-base font-semibold text-slate-800">
                    {file.name}
                  </p>

                  <p className="mt-1 text-sm text-slate-500">
                    Remote-sensing image ready for analysis
                  </p>

                </div>
              </div>
            ) : (
              /* Empty upload state */
              <div
                className="flex h-full cursor-pointer items-center justify-center bg-slate-50 transition hover:bg-slate-100"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="text-center">

                  <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-xl bg-white shadow-sm">
                    <Upload className="h-8 w-8 text-blue-600" />
                  </div>

                  <h2 className="text-lg font-semibold text-slate-800">
                    Upload satellite imagery
                  </h2>

                  <p className="mt-1 text-sm text-slate-500">
                    Click here or use Upload Image
                  </p>

                  <p className="mt-2 text-xs text-slate-400">
                    GeoTIFF, TIFF, PNG, JPG, JPEG, JP2
                  </p>

                </div>
              </div>
            )}

            {/* Image information bar */}
            {file && (
              <div className="absolute bottom-4 left-4 right-4 z-20 flex items-center justify-between rounded-xl border border-slate-200 bg-white/95 px-4 py-3 shadow-sm backdrop-blur">

                <div className="flex min-w-0 items-center gap-3">

                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-50">
                    <FileImage className="h-5 w-5 text-blue-600" />
                  </div>

                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold text-slate-800">
                      {file.name}
                    </p>

                    <p className="mt-0.5 text-xs text-slate-500">
                      {file.type || 'Satellite image'} • Ready for analysis
                    </p>
                  </div>

                </div>

                <button
                  type="button"
                  onClick={removeFile}
                  className="ml-3 flex shrink-0 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600 transition hover:border-red-200 hover:bg-red-50 hover:text-red-600"
                >
                  <X className="h-4 w-4" />
                  Remove
                </button>

              </div>
            )}

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff,.png,.jpg,.jpeg,.jp2,.img,image/tiff,image/png,image/jpeg"
              className="hidden"
              onChange={(event) => {
                handleFile(event.target.files?.[0] ?? null)
              }}
            />

          </div>

          {/* =====================================================
              QUICK ANALYSIS
          ====================================================== */}
          <div className="vision-card min-h-0 flex-1 overflow-hidden p-5">

            <div className="mb-4">
              <p className="text-sm font-semibold text-slate-800">
                Quick Analysis
              </p>

              <p className="mt-1 text-xs text-slate-500">
                Choose a common satellite analysis task
              </p>
            </div>

            {/* Quick analysis buttons */}
            <div className="flex flex-wrap gap-2.5">

              {quickExamples.map((example) => (
                <button
                  key={example.label}
                  type="button"
                  onClick={() => useExample(example.query)}
                  className="rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-600 transition hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700"
                >
                  {example.label}
                </button>
              ))}

            </div>

            {/* =================================================
                VISUAL INTELLIGENCE PANEL
            ================================================== */}
            <div className="relative mt-5 min-h-[180px] overflow-hidden rounded-xl border border-slate-200 bg-slate-50">

              {/* Subtle grid */}
              <div
                className="absolute inset-0 opacity-40"
                style={{
                  backgroundImage:
                    'linear-gradient(rgba(148,163,184,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.15) 1px, transparent 1px)',
                  backgroundSize: '28px 28px',
                }}
              />

              {/* Soft central highlight */}
              <div className="absolute left-1/2 top-1/2 h-48 w-48 -translate-x-1/2 -translate-y-1/2 rounded-full bg-blue-100/60 blur-3xl" />

              {/* Content */}
              <div className="relative flex min-h-[180px] flex-col items-center justify-center px-6 text-center">

                <div className="mb-3 flex items-center gap-2">

                  <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />

                  <span className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-600">
                    Satellite Intelligence
                  </span>

                  <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />

                </div>

                <h3 className="text-base font-semibold text-slate-800">
                  Turn imagery into meaningful insights
                </h3>

                <p className="mt-1.5 max-w-lg text-xs leading-relaxed text-slate-500">
                  Ask a natural-language question about your satellite
                  image and let VisionOrbit select the appropriate analysis.
                </p>

                {/* Capability indicators */}
                <div className="mt-4 flex flex-wrap justify-center gap-2">

                  {['Optical', 'Spectral', 'Spatial', 'Temporal'].map(
                    (item) => (
                      <span
                        key={item}
                        className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-500"
                      >
                        {item}
                      </span>
                    ),
                  )}

                </div>

              </div>
            </div>
          </div>
        </section>

        {/* =======================================================
            RIGHT COLUMN — ASK THE EARTH
        ======================================================== */}
        <aside className="vision-card flex min-h-0 flex-col overflow-hidden">

          {/* Header */}
          <div className="border-b border-slate-200 px-6 py-6">

            <div className="flex items-center justify-between">

              <div>
                <h2 className="text-2xl font-bold text-slate-900">
                  Ask the Earth
                </h2>

                <p className="mt-2 text-sm leading-relaxed text-slate-500">
                  Ask questions about your satellite imagery using
                  natural language.
                </p>
              </div>

              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-blue-50">
                <Map className="h-5 w-5 text-blue-600" />
              </div>

            </div>

          </div>

          {/* Query */}
          <div className="p-6">

            <div className="mb-3">
              <p className="text-base font-semibold text-slate-800">
                What would you like to know?
              </p>

              <p className="mt-1 text-sm text-slate-500">
                Describe what you want to detect or analyze.
              </p>
            </div>

            {/* Larger question box */}
            <div className="relative">

              <textarea
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value)
                  setSubmitError(null)
                }}
                placeholder="e.g. Show the buildings in this image"
                className="vision-input min-h-[190px] w-full resize-none p-4 pr-14 text-base leading-relaxed text-slate-700 placeholder:text-slate-400"
                disabled={submitting}
              />

              {/* Send */}
              <button
                type="button"
                onClick={handleSubmit}
                disabled={submitting}
                title="Run analysis"
                className="absolute bottom-4 right-4 flex h-11 w-11 items-center justify-center rounded-lg bg-blue-600 text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Send className="h-5 w-5" />
              </button>

            </div>

            {/* Errors */}
            {(fileError || submitError) && (
              <div className="mt-3 rounded-lg bg-red-50 px-4 py-3">
                <p className="text-sm text-red-600">
                  {fileError || submitError}
                </p>
              </div>
            )}

            {/* Upload */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={submitting}
              className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-600 transition hover:bg-slate-50 disabled:opacity-50"
            >
              <Upload className="h-4 w-4" />

              {file ? 'Change Image' : 'Upload Image'}
            </button>

            {/* Selected image */}
            {file && (
              <div className="mt-3 flex items-center gap-2 rounded-lg bg-blue-50 px-4 py-3">

                <FileImage className="h-5 w-5 shrink-0 text-blue-600" />

                <span className="truncate text-sm font-medium text-blue-800">
                  {file.name}
                </span>

                <button
                  type="button"
                  onClick={removeFile}
                  className="ml-auto shrink-0 text-blue-500 hover:text-red-600"
                  title="Remove image"
                >
                  <X className="h-4 w-4" />
                </button>

              </div>
            )}

          </div>

          {/* =====================================================
              RECENT QUERIES
          ====================================================== */}
          <div className="min-h-0 flex-1 border-t border-slate-200 p-6">

            <div className="mb-4">

              <h3 className="text-base font-semibold text-slate-800">
                Recent Queries
              </h3>

              <p className="mt-1 text-xs text-slate-500">
                Click a query to use it again
              </p>

            </div>

            <div className="space-y-2.5 overflow-y-auto">

              {recentQueries.map((recentQuery) => (
                <button
                  key={recentQuery}
                  type="button"
                  onClick={() => useExample(recentQuery)}
                  className="group flex w-full items-center gap-3 rounded-lg border border-slate-100 bg-slate-50 px-4 py-3.5 text-left text-sm text-slate-600 transition hover:border-blue-100 hover:bg-blue-50 hover:text-blue-700"
                >

                  <Search className="h-4 w-4 shrink-0 text-slate-400 group-hover:text-blue-500" />

                  <span className="truncate">
                    {recentQuery}
                  </span>

                  <ArrowUp className="ml-auto h-3.5 w-3.5 rotate-45 opacity-0 transition group-hover:opacity-100" />

                </button>
              ))}

            </div>
          </div>

          {/* Status */}
          <div className="border-t border-slate-200 bg-slate-50 px-6 py-4">

            <div className="flex items-center gap-2.5">

              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  submitting
                    ? 'animate-pulse bg-blue-500'
                    : 'bg-green-500'
                }`}
              />

              <span className="text-sm font-medium text-slate-600">
                {submitting
                  ? 'Analyzing satellite imagery...'
                  : 'Ready for analysis'}
              </span>

            </div>

          </div>

        </aside>
      </div>
    </div>
  )
}