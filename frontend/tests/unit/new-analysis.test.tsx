import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { NewAnalysisPage } from '@/pages/new-analysis-page'

function renderPage() {
  return render(
    <MemoryRouter>
      <NewAnalysisPage />
    </MemoryRouter>,
  )
}

describe('NewAnalysisPage', () => {
  it('offers the four input modes and defaults to single optical', () => {
    renderPage()
    expect(screen.getByRole('radio', { name: /Single optical/i })).toBeChecked()
    expect(screen.getByRole('radio', { name: /Single SAR/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Optical \+ SAR/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Bi-temporal/i })).toBeInTheDocument()
    // Single optical => exactly one dropzone.
    expect(screen.getByLabelText('Optical / multispectral scene')).toBeInTheDocument()
  })

  it('changes the required dropzones when the mode changes', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(screen.getByRole('radio', { name: /Optical \+ SAR/i }))
    expect(screen.getByLabelText('Optical scene')).toBeInTheDocument()
    expect(screen.getByLabelText('SAR scene')).toBeInTheDocument()
  })

  it('fills the query box from an example chip', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(screen.getByRole('button', { name: 'Where is the water?' }))
    expect(screen.getByLabelText('Query')).toHaveValue('Where is the water?')
  })

  it('rejects an oversized file with a clear message', async () => {
    const user = userEvent.setup()
    renderPage()
    // A .tif passes the input `accept` filter; spoof its size to trip the limit.
    const big = new File(['x'], 'huge.tif', { type: 'image/tiff' })
    Object.defineProperty(big, 'size', { value: 300 * 1024 * 1024 })
    await user.upload(screen.getByLabelText('Optical / multispectral scene'), big)
    expect(await screen.findByText(/larger than the 250 MB limit/i)).toBeInTheDocument()
  })

  it('refuses to fabricate a result offline and asks for the backend', async () => {
    const user = userEvent.setup()
    renderPage()
    const file = new File(['fake-geotiff-bytes'], 'scene.tif', { type: 'image/tiff' })
    await user.upload(screen.getByLabelText('Optical / multispectral scene'), file)
    await user.type(screen.getByLabelText('Query'), 'What land cover is present?')
    await user.click(screen.getByRole('button', { name: /Run analysis/i }))
    // Honest offline behaviour (§42): no invented analysis, a recoverable prompt
    // to start the backend.
    expect(await screen.findByText(/Live backend required/i)).toBeInTheDocument()
  })
})
