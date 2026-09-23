import { createBrowserRouter } from 'react-router-dom'
import { AppShell } from '@/components/layout/app-shell'
import { DashboardPage } from '@/pages/dashboard-page'
import { NewAnalysisPage } from '@/pages/new-analysis-page'
import { ResultPage } from '@/pages/result-page'
import { ReportPage } from '@/pages/report-page'
import { EvaluationPage } from '@/pages/evaluation-page'
import { NotFoundPage } from '@/pages/not-found-page'

export const router = createBrowserRouter(
  [
    // Standalone printable report — rendered outside the app shell.
    { path: '/analysis/:id/report', element: <ReportPage /> },
    {
      element: <AppShell />,
      children: [
        { index: true, element: <DashboardPage /> },
        { path: 'analyze', element: <NewAnalysisPage /> },
        { path: 'evaluation', element: <EvaluationPage /> },
        { path: 'analysis/:id', element: <ResultPage /> },
        { path: '*', element: <NotFoundPage /> },
      ],
    },
  ],
  {
    // Opt into React Router v7 relative-splat resolution early (silences the
    // upgrade warning). v7_startTransition is a RouterProvider prop, not a
    // data-router future flag, so it is not set here.
    future: { v7_relativeSplatPath: true },
  },
)
