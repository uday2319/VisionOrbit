import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { DashboardPage } from '@/pages/dashboard-page'

function renderDashboard() {
  return render(
    <MemoryRouter>
      <DashboardPage />
    </MemoryRouter>,
  )
}

describe('DashboardPage', () => {
  it('lists every advertised capability', async () => {
    renderDashboard()
    expect(screen.getByText('Optical scene analysis')).toBeInTheDocument()
    expect(screen.getByText('Text-guided grounding')).toBeInTheDocument()
    expect(screen.getByText('SAR backscatter analysis')).toBeInTheDocument()
    expect(screen.getByText('Optical + SAR fusion')).toBeInTheDocument()
    expect(screen.getByText('Bi-temporal change detection')).toBeInTheDocument()
    // Let the async system-status / recent-analyses loads settle before unmount.
    await screen.findByText('Demo analyses')
  })

  it('shows honest offline system status from the demo catalog', async () => {
    renderDashboard()
    expect(await screen.findByText('Demo analyses')).toBeInTheDocument()
    // Six recorded fixtures -> the "Demo analyses" tile reads 6.
    expect(screen.getByText('6')).toBeInTheDocument()
  })

  it('links recent analyses to their result pages', async () => {
    renderDashboard()
    const link = await screen.findByRole('link', { name: /Bi-temporal change analysis/i })
    // A recorded row carries the marker, so the row opens the analysis it just listed rather than a
    // live run that happens to share its id.
    expect(link).toHaveAttribute('href', '/analysis/SAT-2026-000106?demo=1')
  })

  /**
   * "View example" advertises one specific recorded case, so it must ask for the recording by name.
   *
   * The backend numbers its own runs from the same `SAT-2026-NNNNNN` sequence as the six shipped
   * recordings, so once a machine has run past a hundred analyses those ids exist in its database too
   * — and a plain `/analysis/<id>` link resolved live-first, opening an unrelated run under this
   * card's title, with no imagery because rows from before artifacts were persisted carry none.
   */
  it('asks for the recording behind each example, not whatever the backend has under that id', async () => {
    renderDashboard()
    const examples = await screen.findAllByRole('link', { name: /View example/i })
    expect(examples).toHaveLength(5)
    expect(examples.map((a) => a.getAttribute('href'))).toEqual([
      '/analysis/SAT-2026-000101?demo=1',
      '/analysis/SAT-2026-000102?demo=1',
      '/analysis/SAT-2026-000104?demo=1',
      '/analysis/SAT-2026-000105?demo=1',
      '/analysis/SAT-2026-000106?demo=1',
    ])
  })
})
