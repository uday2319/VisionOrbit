/**
 * ReportDocument — on-screen (and printable) rendering of a ReportModel.
 *
 * Deliberately theme-independent: it uses an explicit light "paper" palette
 * (slate on white) rather than the app's theme tokens, so the document looks
 * like a document and prints legibly whatever the UI theme. Print rules use
 * Tailwind's `print:` variant; colour swatches, table headers and callouts opt
 * into `print-color-adjust: exact` so they survive "Save as PDF".
 *
 * Imagery is rendered as static figures (the artifact PNGs + legends), not the
 * interactive Leaflet map, because an interactive map does not print.
 */
import type { ReportBlock, ReportModel, ReportSection, ReportTone } from '@/lib/report-model'
import type { Artifact } from '@/types/api'
import { modeLabel, taskLabel, formatDuration, formatDateTime } from '@/lib/format'

const TONE_STYLES: Record<ReportTone, string> = {
  info: 'border-blue-300 bg-blue-50 text-blue-900',
  warning: 'border-amber-300 bg-amber-50 text-amber-900',
  success: 'border-emerald-300 bg-emerald-50 text-emerald-900',
  muted: 'border-slate-200 bg-slate-50 text-slate-600',
}

const LEVEL_STYLES: Record<string, string> = {
  high: 'border-emerald-300 bg-emerald-50 text-emerald-800',
  medium: 'border-blue-300 bg-blue-50 text-blue-800',
  low: 'border-amber-300 bg-amber-50 text-amber-800',
  insufficient: 'border-slate-300 bg-slate-100 text-slate-700',
}

function Fields({ fields }: { fields: Extract<ReportBlock, { kind: 'fields' }>['fields'] }) {
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-3 md:grid-cols-4">
      {fields.map((f) => (
        <div key={f.label}>
          <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            {f.label}
          </dt>
          <dd className="mt-0.5 text-sm font-semibold text-slate-900">{f.value}</dd>
          {f.hint && <dd className="text-xs text-slate-500">{f.hint}</dd>}
        </div>
      ))}
    </dl>
  )
}

function Table({ table }: { table: Extract<ReportBlock, { kind: 'table' }>['table'] }) {
  return (
    <figure className="break-inside-avoid">
      <div className="overflow-x-auto rounded-md border border-slate-200">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-slate-50 [print-color-adjust:exact] [-webkit-print-color-adjust:exact]">
              {table.columns.map((c) => (
                <th
                  key={c.key}
                  className={`border-b border-slate-200 px-3 py-2 font-semibold text-slate-700 ${
                    c.align === 'right' ? 'text-right' : 'text-left'
                  }`}
                >
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, i) => (
              <tr key={i} className="odd:bg-white even:bg-slate-50/60">
                {table.columns.map((c) => (
                  <td
                    key={c.key}
                    className={`border-b border-slate-100 px-3 py-1.5 text-slate-700 ${
                      c.align === 'right' ? 'text-right tabular-nums' : 'text-left'
                    }`}
                  >
                    {row[c.key] ?? '—'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.caption && (
        <figcaption className="mt-1 text-xs text-slate-500">{table.caption}</figcaption>
      )}
    </figure>
  )
}

function Callout({ block }: { block: Extract<ReportBlock, { kind: 'callout' }> }) {
  return (
    <div
      className={`break-inside-avoid rounded-md border px-4 py-3 text-sm [print-color-adjust:exact] [-webkit-print-color-adjust:exact] ${TONE_STYLES[block.tone]}`}
    >
      {block.title && <p className="font-semibold">{block.title}</p>}
      <p className={block.title ? 'mt-0.5' : ''}>{block.body}</p>
    </div>
  )
}

function Block({ block }: { block: ReportBlock }) {
  switch (block.kind) {
    case 'fields':
      return <Fields fields={block.fields} />
    case 'table':
      return <Table table={block.table} />
    case 'text':
      return <p className="text-sm leading-relaxed text-slate-700">{block.text}</p>
    case 'list':
      return block.ordered ? (
        <ol className="list-decimal space-y-1 pl-5 text-sm text-slate-700">
          {block.items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ol>
      ) : (
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
          {block.items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      )
    case 'callout':
      return <Callout block={block} />
    default:
      return null
  }
}

function Section({ section }: { section: ReportSection }) {
  return (
    <section className="space-y-3">
      <h2 className="border-b border-slate-200 pb-1 text-lg font-semibold text-slate-900">
        {section.title}
      </h2>
      {section.blocks.map((block, i) => (
        <Block key={i} block={block} />
      ))}
    </section>
  )
}

function Legend({ entries }: { entries: Artifact['legend'] }) {
  if (entries.length === 0) return null
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
      {entries.map((e) => (
        <span key={e.key} className="inline-flex items-center gap-1.5 text-xs text-slate-600">
          <span
            className="inline-block h-3 w-3 rounded-sm ring-1 ring-black/10 [print-color-adjust:exact] [-webkit-print-color-adjust:exact]"
            style={{ backgroundColor: e.color }}
            aria-hidden="true"
          />
          {e.label}
        </span>
      ))}
    </div>
  )
}

/** Static imagery: base + overlays as figures, using the artifact URLs as-is
 *  (they resolve against the running app). The standalone HTML export inlines
 *  them as data URIs separately, in renderReportHtml. */
function Imagery({ artifacts }: { artifacts: Artifact[] }) {
  if (artifacts.length === 0) return null
  return (
    <section className="space-y-3">
      <h2 className="border-b border-slate-200 pb-1 text-lg font-semibold text-slate-900">
        Imagery
      </h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {artifacts.map((a) => (
          <figure key={`${a.kind}-${a.url}`} className="break-inside-avoid">
            <div className="overflow-hidden rounded-md border border-slate-200 bg-slate-100">
              <img src={a.url} alt={a.label} className="block w-full" loading="lazy" />
            </div>
            <figcaption className="mt-1 text-xs font-medium text-slate-600">{a.label}</figcaption>
            <Legend entries={a.legend} />
          </figure>
        ))}
      </div>
    </section>
  )
}

export function ReportDocument({ model }: { model: ReportModel }) {
  const levelClass = LEVEL_STYLES[model.confidenceLevel] ?? LEVEL_STYLES.insufficient
  return (
    <article className="mx-auto my-8 max-w-[820px] space-y-6 rounded-lg bg-white p-8 text-slate-900 shadow-xl ring-1 ring-slate-200 [print-color-adjust:exact] [-webkit-print-color-adjust:exact] sm:p-10 print:my-0 print:max-w-none print:rounded-none print:p-0 print:shadow-none print:ring-0">
      {/* Masthead */}
      <header className="space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-blue-700">
              SatQuery AI
            </p>
            <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900">{model.title}</h1>
            {model.note && <p className="mt-1 text-sm text-slate-600">{model.note}</p>}
          </div>
          <div className="text-right">
            <p className="font-mono text-lg font-semibold text-slate-900">{model.id}</p>
            <p className="text-xs text-slate-500">
              Generated {formatDateTime(model.generatedAtIso)}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-y border-slate-200 py-2 text-xs text-slate-600">
          <span>
            Status <span className="font-semibold text-slate-900 capitalize">{model.status}</span>
          </span>
          <span>·</span>
          <span>{modeLabel(model.mode)}</span>
          <span>·</span>
          <span>{taskLabel(model.task)}</span>
          <span>·</span>
          <span>
            Tool <span className="font-medium text-slate-900">{model.tool}</span> ({model.tier})
          </span>
          <span>·</span>
          <span>{formatDuration(model.durationMs)}</span>
          <span>·</span>
          <span>Analysed {formatDateTime(model.createdAtIso)}</span>
          {model.fixture && (
            <span className="rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-600 [print-color-adjust:exact] [-webkit-print-color-adjust:exact]">
              recorded run
            </span>
          )}
        </div>
      </header>

      {/* Query & answer */}
      <section className="space-y-3">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">Query</p>
          <blockquote className="mt-1 border-l-2 border-slate-300 pl-3 text-sm italic text-slate-700">
            {model.query}
          </blockquote>
        </div>
        <div>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">Answer</p>
            <span
              className={`rounded-full border px-2 py-0.5 text-xs font-medium capitalize [print-color-adjust:exact] [-webkit-print-color-adjust:exact] ${levelClass}`}
            >
              {model.confidenceLevel} confidence
            </span>
          </div>
          {/*
           * The answer comes first, then the caveat about it. The old order printed
           * "Insufficient evidence for a reliable conclusion." in a banner above the measured
           * answer, so the report declined to conclude and concluded in the same section.
           */}
          {model.withheld ? (
            <div className="mt-2 break-inside-avoid rounded-md border border-amber-300 bg-amber-50 px-4 py-3 [print-color-adjust:exact] [-webkit-print-color-adjust:exact]">
              <p className="text-[11px] font-medium uppercase tracking-wide text-amber-900">
                Answer withheld
              </p>
              <p className="mt-1 text-base leading-relaxed text-slate-900">{model.answer}</p>
            </div>
          ) : (
            <p className="mt-2 text-base leading-relaxed text-slate-900">{model.answer}</p>
          )}
          {model.caveat && (
            <div className="mt-2 break-inside-avoid rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 [print-color-adjust:exact] [-webkit-print-color-adjust:exact]">
              <p className="font-semibold">{model.caveat.title}</p>
              <p className="mt-1">{model.caveat.body}</p>
            </div>
          )}
          {model.withheldDraft && (
            <div className="mt-2 rounded-md border border-dashed border-slate-300 bg-slate-50 px-4 py-3 [print-color-adjust:exact] [-webkit-print-color-adjust:exact]">
              <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                Withheld phrasing (not asserted)
              </p>
              <p className="mt-1 text-sm italic text-slate-600">{model.withheldDraft}</p>
            </div>
          )}
        </div>
      </section>

      <Imagery artifacts={model.artifacts} />

      {model.sections.map((section) => (
        <Section key={section.id} section={section} />
      ))}

      {/* Footer / disclaimer */}
      <footer className="space-y-1 border-t border-slate-200 pt-4 text-xs text-slate-500">
        <p>{model.disclaimer}</p>
        <p>
          SatQuery AI · report {model.id} · request {model.requestId} · generated{' '}
          {formatDateTime(model.generatedAtIso)}
        </p>
      </footer>
    </article>
  )
}
