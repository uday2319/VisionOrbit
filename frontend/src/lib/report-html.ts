/**
 * Standalone HTML export of a ReportModel (brief §13, the "HTML report").
 *
 * `renderReportHtml` is a pure function: model in, a complete self-contained
 * `<!doctype html>` document out, with an inline stylesheet and no external
 * dependencies — it opens and prints anywhere. It renders the *same* ReportModel
 * that <ReportDocument> renders on screen, so the two never drift.
 *
 * Images are resolved through the injected `imageSrc` callback. The default keeps
 * the artifact URLs (handy for tests and same-origin viewing); the report page
 * passes a resolver that inlines each PNG as a base64 data URI so the downloaded
 * file is truly self-contained. All dynamic text is HTML-escaped.
 */
import type { Artifact } from '@/types/api'
import type { ReportBlock, ReportModel, ReportTone } from '@/lib/report-model'
import { modeLabel, taskLabel, formatDuration, formatDateTime } from '@/lib/format'

/** Escape text for safe interpolation into HTML (also used for attributes). */
export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

const TONE_CLASS: Record<ReportTone, string> = {
  info: 'info',
  warning: 'warning',
  success: 'success',
  muted: 'muted',
}

function renderBlock(block: ReportBlock): string {
  switch (block.kind) {
    case 'fields':
      return `<dl class="fields">${block.fields
        .map(
          (f) =>
            `<div><dt>${escapeHtml(f.label)}</dt><dd class="v">${escapeHtml(f.value)}</dd>${
              f.hint ? `<dd class="h">${escapeHtml(f.hint)}</dd>` : ''
            }</div>`,
        )
        .join('')}</dl>`
    case 'table': {
      const head = block.table.columns
        .map((c) => `<th class="${c.align === 'right' ? 'r' : ''}">${escapeHtml(c.label)}</th>`)
        .join('')
      const body = block.table.rows
        .map(
          (row) =>
            `<tr>${block.table.columns
              .map(
                (c) =>
                  `<td class="${c.align === 'right' ? 'r' : ''}">${escapeHtml(row[c.key] ?? '—')}</td>`,
              )
              .join('')}</tr>`,
        )
        .join('')
      const caption = block.table.caption
        ? `<figcaption>${escapeHtml(block.table.caption)}</figcaption>`
        : ''
      return `<figure class="tbl"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>${caption}</figure>`
    }
    case 'text':
      return `<p class="body">${escapeHtml(block.text)}</p>`
    case 'list': {
      const items = block.items.map((i) => `<li>${escapeHtml(i)}</li>`).join('')
      return block.ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`
    }
    case 'callout':
      return `<div class="callout ${TONE_CLASS[block.tone]}">${
        block.title ? `<p class="t">${escapeHtml(block.title)}</p>` : ''
      }<p>${escapeHtml(block.body)}</p></div>`
    default:
      return ''
  }
}

function renderImagery(artifacts: Artifact[], imageSrc: (a: Artifact) => string): string {
  if (artifacts.length === 0) return ''
  const figures = artifacts
    .map((a) => {
      const legend =
        a.legend.length > 0
          ? `<div class="legend">${a.legend
              .map(
                (e) =>
                  `<span class="li"><span class="sw" style="background:${escapeHtml(e.color)}"></span>${escapeHtml(e.label)}</span>`,
              )
              .join('')}</div>`
          : ''
      return `<figure class="img"><img src="${escapeHtml(imageSrc(a))}" alt="${escapeHtml(a.label)}"><figcaption>${escapeHtml(a.label)}</figcaption>${legend}</figure>`
    })
    .join('')
  return `<section><h2>Imagery</h2><div class="grid">${figures}</div></section>`
}

const STYLES = `
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f1f5f9; color: #0f172a;
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.paper { max-width: 820px; margin: 32px auto; background: #fff; padding: 40px;
  border-radius: 8px; box-shadow: 0 10px 30px rgba(15,23,42,.12); }
h1 { font-size: 24px; margin: 4px 0 0; letter-spacing: -.01em; }
h2 { font-size: 18px; margin: 28px 0 10px; padding-bottom: 4px; border-bottom: 1px solid #e2e8f0; }
p { margin: 0; } p.body { color: #334155; margin: 6px 0; }
.brand { font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: #1d4ed8; }
.masthead { display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
.masthead .note { color: #475569; margin-top: 4px; }
.rid { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 18px; font-weight: 600; text-align: right; }
.gen { color: #64748b; font-size: 12px; text-align: right; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: center; color: #475569;
  font-size: 12px; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; padding: 8px 0; margin-top: 12px; }
.meta b { color: #0f172a; font-weight: 600; }
.tag { background: #f1f5f9; border-radius: 4px; padding: 1px 6px; color: #475569; font-weight: 500; }
.label { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #64748b; }
blockquote { margin: 4px 0 0; padding-left: 12px; border-left: 2px solid #cbd5e1; font-style: italic; color: #334155; }
.answer { font-size: 16px; color: #0f172a; margin-top: 8px; }
.chip { border: 1px solid; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 500; text-transform: capitalize; }
.chip.high { border-color: #6ee7b7; background: #ecfdf5; color: #065f46; }
.chip.medium { border-color: #93c5fd; background: #eff6ff; color: #1e40af; }
.chip.low { border-color: #fcd34d; background: #fffbeb; color: #92400e; }
.chip.insufficient { border-color: #cbd5e1; background: #f1f5f9; color: #334155; }
.answer-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
.fields { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px 24px; margin: 8px 0; }
.fields dt { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: #64748b; }
.fields dd { margin: 2px 0 0; } .fields dd.v { font-weight: 600; color: #0f172a; } .fields dd.h { font-size: 12px; color: #64748b; }
figure { margin: 10px 0; page-break-inside: avoid; }
figure.tbl figcaption, figure.img figcaption { font-size: 12px; color: #64748b; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden; }
th { background: #f8fafc; text-align: left; font-weight: 600; color: #334155; border-bottom: 1px solid #e2e8f0; padding: 7px 10px; }
td { border-bottom: 1px solid #f1f5f9; padding: 6px 10px; color: #334155; }
tbody tr:nth-child(even) { background: #f8fafc; }
.r { text-align: right; font-variant-numeric: tabular-nums; }
ul, ol { margin: 6px 0; padding-left: 20px; color: #334155; } li { margin: 2px 0; }
.callout { border: 1px solid; border-radius: 6px; padding: 10px 14px; margin: 10px 0; page-break-inside: avoid; }
.callout .t { font-weight: 700; margin-bottom: 2px; }
.callout.info { border-color: #93c5fd; background: #eff6ff; color: #1e3a8a; }
.callout.warning { border-color: #fcd34d; background: #fffbeb; color: #78350f; }
.callout.success { border-color: #6ee7b7; background: #ecfdf5; color: #065f46; }
.callout.muted { border-color: #e2e8f0; background: #f8fafc; color: #475569; }
.withheld { border: 1px dashed #cbd5e1; background: #f8fafc; border-radius: 6px; padding: 10px 14px; margin-top: 8px; }
.withheld p.q { font-style: italic; color: #475569; margin-top: 4px; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
figure.img img { width: 100%; display: block; border: 1px solid #e2e8f0; border-radius: 6px; background: #f1f5f9; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin-top: 6px; }
.legend .li { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: #475569; }
.sw { width: 12px; height: 12px; border-radius: 3px; box-shadow: inset 0 0 0 1px rgba(0,0,0,.1); }
footer { border-top: 1px solid #e2e8f0; margin-top: 28px; padding-top: 14px; color: #64748b; font-size: 12px; }
footer p { margin: 3px 0; }
@page { margin: 16mm; }
@media print { body { background: #fff; } .paper { margin: 0; max-width: none; box-shadow: none; border-radius: 0; padding: 0; } }
`

/**
 * Render the model as a complete standalone HTML document.
 * @param imageSrc maps each artifact to the `src` used in the file (default: its URL).
 */
export function renderReportHtml(
  model: ReportModel,
  imageSrc: (a: Artifact) => string = (a) => a.url,
): string {
  // The answer, then the caveat about it — never a banner declining to conclude above a measured
  // conclusion. `withheld` is the separate case where there is no answer to state at all.
  const answerBlock = model.withheld
    ? `<div class="callout warning"><p class="label">Answer withheld</p><p class="answer">${escapeHtml(model.answer)}</p></div>`
    : `<p class="answer">${escapeHtml(model.answer)}</p>`
  const caveat = model.caveat
    ? `<div class="callout warning"><p><strong>${escapeHtml(model.caveat.title)}</strong></p><p>${escapeHtml(model.caveat.body)}</p></div>`
    : ''
  const withheldDraft = model.withheldDraft
    ? `<div class="withheld"><p class="label">Withheld phrasing (not asserted)</p><p class="q">${escapeHtml(model.withheldDraft)}</p></div>`
    : ''
  const fixtureTag = model.fixture ? `<span class="tag">recorded run</span>` : ''

  const sections = model.sections
    .map(
      (s) =>
        `<section><h2>${escapeHtml(s.title)}</h2>${s.blocks.map(renderBlock).join('')}</section>`,
    )
    .join('')

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SatQuery report ${escapeHtml(model.id)}</title>
<style>${STYLES}</style>
</head>
<body>
<article class="paper">
<header>
  <div class="masthead">
    <div>
      <p class="brand">SatQuery AI</p>
      <h1>${escapeHtml(model.title)}</h1>
      ${model.note ? `<p class="note">${escapeHtml(model.note)}</p>` : ''}
    </div>
    <div>
      <p class="rid">${escapeHtml(model.id)}</p>
      <p class="gen">Generated ${escapeHtml(formatDateTime(model.generatedAtIso))}</p>
    </div>
  </div>
  <div class="meta">
    <span>Status <b style="text-transform:capitalize">${escapeHtml(model.status)}</b></span><span>·</span>
    <span>${escapeHtml(modeLabel(model.mode))}</span><span>·</span>
    <span>${escapeHtml(taskLabel(model.task))}</span><span>·</span>
    <span>Tool <b>${escapeHtml(model.tool)}</b> (${escapeHtml(model.tier)})</span><span>·</span>
    <span>${escapeHtml(formatDuration(model.durationMs))}</span><span>·</span>
    <span>Analysed ${escapeHtml(formatDateTime(model.createdAtIso))}</span>
    ${fixtureTag}
  </div>
</header>

<section>
  <p class="label">Query</p>
  <blockquote>${escapeHtml(model.query)}</blockquote>
  <div class="answer-head" style="margin-top:12px">
    <p class="label">Answer</p>
    <span class="chip ${escapeHtml(model.confidenceLevel)}">${escapeHtml(model.confidenceLevel)} confidence</span>
  </div>
  ${answerBlock}
  ${caveat}
  ${withheldDraft}
</section>

${renderImagery(model.artifacts, imageSrc)}
${sections}

<footer>
  <p>${escapeHtml(model.disclaimer)}</p>
  <p>SatQuery AI · report ${escapeHtml(model.id)} · request ${escapeHtml(model.requestId)} · generated ${escapeHtml(formatDateTime(model.generatedAtIso))}</p>
</footer>
</article>
</body>
</html>`
}
