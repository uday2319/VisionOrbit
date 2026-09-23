import { Link } from 'react-router-dom'
import { ArrowRight, PlayCircle } from 'lucide-react'
import type { InputMode } from '@/types/api'
import { CAPABILITIES } from '@/lib/capabilities'
import { Button } from '@/components/ui/button'

interface DemoGalleryProps {
  mode: InputMode
  /** Fill the query box with a capability's example query. */
  onUseExample?: (query: string) => void
}

/**
 * Recorded demo cases for the active mode. Each links to a genuine recorded
 * analysis (fixtures = real tool output, brief §42) so reviewers can see live
 * results without a running backend.
 */
export function DemoGallery({ mode, onUseExample }: DemoGalleryProps) {
  const demos = CAPABILITIES.filter((c) => c.mode === mode && c.demoCaseId)
  if (demos.length === 0) return null

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium text-foreground">
        <PlayCircle className="h-4 w-4 text-primary" aria-hidden="true" />
        Explore a recorded {demos.length === 1 ? 'analysis' : 'analysis'} for this mode
      </div>
      <ul className="space-y-2">
        {demos.map((demo) => (
          <li
            key={demo.id}
            className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-muted/40 p-3"
          >
            <div className="min-w-0">
              <p className="text-sm font-medium text-foreground">{demo.title}</p>
              <p className="truncate text-xs text-muted-foreground">“{demo.exampleQuery}”</p>
            </div>
            <div className="flex items-center gap-2">
              {onUseExample && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => onUseExample(demo.exampleQuery)}
                >
                  Use query
                </Button>
              )}
              <Button asChild variant="outline" size="sm">
                {/* `?demo=1` — this card advertises one recorded case, so it asks for the recording
                    by name rather than for whatever the live backend holds under that id. */}
                <Link to={`/analysis/${demo.demoCaseId}?demo=1`}>
                  View result
                  <ArrowRight />
                </Link>
              </Button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
