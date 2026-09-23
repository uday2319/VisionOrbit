import type { ReactNode } from 'react'
import { AlertTriangle, Server, Cpu, Database, Activity } from 'lucide-react'
import { apiConfig, getHealth, getMetrics } from '@/services/api'
import { useAsync } from '@/lib/use-async'
import { formatDuration } from '@/lib/format'
import { Badge } from '@/components/ui/badge'
import type { BadgeProps } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'

function StatTile({
  icon,
  label,
  value,
  hint,
}: {
  icon?: ReactNode
  label: string
  value: ReactNode
  hint?: string
}) {
  return (
    <div className="rounded-lg border bg-card p-4 transition-colors hover:border-primary/40">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
        {icon && <div className="text-muted-foreground/60">{icon}</div>}
      </div>
      <div className="mt-2 text-2xl font-semibold tabular-nums">{value}</div>
      {hint ? <div className="mt-1 text-xs text-muted-foreground">{hint}</div> : null}
    </div>
  )
}

function healthBadge(status: string): {
  variant: NonNullable<BadgeProps['variant']>
  label: string
} {
  switch (status) {
    case 'ok':
      return { variant: 'success', label: 'Operational' }
    case 'degraded':
      return { variant: 'warning', label: 'Degraded' }
    case 'down':
      return { variant: 'destructive', label: 'Down' }
    default:
      return { variant: 'secondary', label: status }
  }
}

export function SystemStatus() {
  const health = useAsync(() => getHealth(), [])
  const metrics = useAsync(() => getMetrics(), [])

  const loading = health.loading || metrics.loading
  const badge = health.data ? healthBadge(health.data.status) : null
  const backendDetail = health.data?.checks?.backend?.detail
  const modelsCount = health.data?.models?.length ?? (apiConfig.offline ? 7 : 0)
  const taskCount = metrics.data?.requests_by_task
    ? Object.keys(metrics.data.requests_by_task).length
    : 0
  const avgDuration =
    typeof metrics.data?.avg_duration_ms === 'number'
      ? formatDuration(metrics.data.avg_duration_ms)
      : '—'

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <div className="space-y-1">
            <CardTitle className="text-base">System status</CardTitle>
            <CardDescription>
              {apiConfig.offline
                ? 'Offline demo — replaying recorded analysis fixtures.'
                : 'Live FastAPI backend health, remote-sensing models, and request metrics.'}
            </CardDescription>
          </div>
          {badge ? <Badge variant={badge.variant}>{badge.label}</Badge> : null}
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-[92px] w-full" />
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <StatTile
                icon={<Server className="h-4 w-4" />}
                label="Backend"
                value={badge?.label ?? 'Operational'}
                hint={apiConfig.offline ? 'Demo mode' : 'FastAPI · Uvicorn'}
              />
              <StatTile
                icon={<Cpu className="h-4 w-4" />}
                label="Models / Tools"
                value={modelsCount > 0 ? `${modelsCount} Active` : 'Ready'}
                hint="VQA, Change, Grounding, Fusion"
              />
              <StatTile
                icon={<Activity className="h-4 w-4" />}
                label={apiConfig.offline ? 'Demo analyses' : 'Total Analyses'}
                value={metrics.data?.requests_total ?? (apiConfig.offline ? 6 : '—')}
                hint={taskCount ? `${taskCount} tasks active` : undefined}
              />
              <StatTile
                icon={<Database className="h-4 w-4" />}
                label="Avg duration"
                value={avgDuration}
                hint="Inference + post-proc"
              />
            </div>
            {backendDetail ? (
              <p className="flex items-start gap-2 text-xs text-muted-foreground">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning" />
                {backendDetail}
              </p>
            ) : null}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
