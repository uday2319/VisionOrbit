import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { ThemeProvider } from '@/components/theme/theme-provider'
import { TooltipProvider } from '@/components/ui/tooltip'
import { AppShell } from '@/components/layout/app-shell'
import { DashboardPage } from '@/pages/dashboard-page'
import { NotFoundPage } from '@/pages/not-found-page'

function renderAt(path: string) {
  const router = createMemoryRouter(
    [
      {
        element: <AppShell />,
        children: [
          { index: true, element: <DashboardPage /> },
          { path: '*', element: <NotFoundPage /> },
        ],
      },
    ],
    { initialEntries: [path] },
  )
  return render(
    <ThemeProvider>
      <TooltipProvider>
        <RouterProvider router={router} />
      </TooltipProvider>
    </ThemeProvider>,
  )
}

describe('app shell + routing', () => {
  it('renders the dashboard at / inside the app shell', async () => {
    renderAt('/')
    expect(screen.getByRole('heading', { name: 'Dashboard', level: 1 })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /SatQuery/i })).toBeInTheDocument()
    expect(screen.getByText(/Demo · fixtures/)).toBeInTheDocument()
    // Awaiting fixture-backed content flushes the dashboard's async loads.
    expect(await screen.findByText('Optical land-cover analysis')).toBeInTheDocument()
  })

  it('renders the 404 page for unknown paths', () => {
    renderAt('/does-not-exist')
    expect(screen.getByRole('heading', { name: /not found/i })).toBeInTheDocument()
  })
})
