import { Link } from 'react-router-dom'
import { AlertCircle } from 'lucide-react'
import type { AnalysisSummary } from '@/types/api'
import { listAnalyses } from '@/services/api'
import { useAsync } from '@/lib/use-async'
import { confidenceBadgeVariant, formatDate, modeLabel, taskLabel } from '@/lib/format'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

function ConfidenceBadge({ level }: { level: AnalysisSummary['confidence'] }) {
  return (
    <Badge variant={confidenceBadgeVariant(level)} className="capitalize">
      {level}
    </Badge>
  )
}

export function RecentAnalyses() {
  const { data, error, loading } = useAsync(() => listAnalyses(), [])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Recent analyses</CardTitle>
        <CardDescription>Completed analyses available to review.</CardDescription>
      </CardHeader>
      <CardContent>
        {error ? (
          <Alert variant="destructive">
            <AlertCircle />
            <AlertTitle>Could not load analyses</AlertTitle>
            <AlertDescription>{error.message}</AlertDescription>
          </Alert>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Analysis</TableHead>
                <TableHead className="hidden sm:table-cell">Mode</TableHead>
                <TableHead className="hidden md:table-cell">Task</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead className="hidden lg:table-cell">Date</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading
                ? Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={i}>
                      <TableCell colSpan={5}>
                        <Skeleton className="h-6 w-full" />
                      </TableCell>
                    </TableRow>
                  ))
                : (data ?? []).map((row) => (
                    <TableRow key={row.id}>
                      <TableCell>
                        <Link
                          // A recorded row links to its recording (`?demo=1`); a live row links to
                          // the live run. Both namespaces use the same id sequence, so the id alone
                          // does not say which analysis this row is.
                          to={`/analysis/${row.id}${row.fixture ? '?demo=1' : ''}`}
                          className="font-medium text-primary hover:underline"
                        >
                          {row.title}
                        </Link>
                        <div className="text-xs text-muted-foreground">{row.id}</div>
                      </TableCell>
                      <TableCell className="hidden sm:table-cell text-muted-foreground">
                        {modeLabel(row.mode)}
                      </TableCell>
                      <TableCell className="hidden md:table-cell text-muted-foreground">
                        {taskLabel(row.task)}
                      </TableCell>
                      <TableCell>
                        <ConfidenceBadge level={row.confidence} />
                      </TableCell>
                      <TableCell className="hidden lg:table-cell text-muted-foreground">
                        {formatDate(row.created_at)}
                      </TableCell>
                    </TableRow>
                  ))}
              {!loading && (data ?? []).length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="py-8 text-center text-muted-foreground">
                    No analyses yet.
                  </TableCell>
                </TableRow>
              ) : null}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
