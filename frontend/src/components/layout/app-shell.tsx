import {
  BarChart3,
  Home,
  Satellite,
} from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { apiConfig } from '@/services/api'

const navItems = [
  {
    to: '/',
    label: 'Home',
    icon: Home,
    end: true,
  },
  {
    to: '/analyze',
    label: 'Ask Satellite',
    icon: Satellite,
    end: false,
  },
  {
    to: '/evaluation',
    label: 'Analytics',
    icon: BarChart3,
    end: false,
  },
]

export function AppShell() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">

      {/* =========================
          TOP HEADER
      ========================== */}
      <header className="fixed inset-x-0 top-0 z-50 h-16 border-b border-slate-200 bg-white">
        <div className="flex h-full items-center">

          {/* VisionOrbit branding */}
          <div className="flex h-full w-64 items-center border-r border-slate-200 px-5">

            <NavLink to="/" className="flex items-center gap-3">

              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-white">
                <Satellite className="h-5 w-5" />
              </div>

              <div>
                <div className="text-sm font-bold leading-tight text-slate-900">
                  VisionOrbit
                </div>

                <div className="text-[10px] text-slate-500">
                  SatQuery AI
                </div>
              </div>

            </NavLink>

          </div>

          {/* Clean header area */}
          <div className="flex flex-1 items-center justify-end px-6">

            <div className="text-xs font-medium text-slate-400">
              Satellite Intelligence Platform
            </div>

          </div>

        </div>
      </header>

      {/* =========================
          SIDEBAR
      ========================== */}
      <aside className="fixed bottom-0 left-0 top-16 z-40 w-64 border-r border-slate-200 bg-white">

        <div className="flex h-full flex-col px-3 py-4">

          {/* Navigation */}
          <nav className="space-y-1">

            {navItems.map((item) => (
              <NavLink
                key={item.label}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900',
                  )
                }
              >
                <item.icon className="h-4 w-4" />
                <span>{item.label}</span>
              </NavLink>
            ))}

          </nav>

          {/* Bottom information card */}
          <div className="mt-auto p-1">

            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">

              <div className="mb-2 flex items-center gap-2">

                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-green-100">
                  <Satellite className="h-4 w-4 text-green-600" />
                </div>

                <span className="text-xs font-semibold text-slate-800">
                  AI for a Sustainable Earth
                </span>

              </div>

              <p className="text-[11px] leading-relaxed text-slate-500">
                From satellite data to real-world impact.
              </p>

              <div className="mt-3 flex items-center gap-1.5 text-[10px] font-medium text-green-600">

                <span
                  className={cn(
                    'h-2 w-2 rounded-full',
                    apiConfig.offline
                      ? 'bg-amber-500'
                      : 'animate-pulse bg-green-500',
                  )}
                />

                {apiConfig.offline ? 'Demo Mode' : 'System Online'}

              </div>

            </div>

          </div>

        </div>
      </aside>

      {/* =========================
          MAIN CONTENT
      ========================== */}
      <main className="ml-64 min-h-screen pt-16">

        <div className="p-5 lg:p-6">
          <Outlet />
        </div>

      </main>

    </div>
  )
}