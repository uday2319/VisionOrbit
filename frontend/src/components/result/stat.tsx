import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface StatProps {
  label: string
  value: ReactNode
  hint?: ReactNode
  className?: string
}

/** A compact labelled metric used across the result data panels. */
export function Stat({ label, value, hint, className }: StatProps) {
  return (
    <div className={cn('rounded-lg border border-border bg-muted/30 p-3', className)}>
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-lg font-semibold text-foreground">{value}</p>
      {hint && <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

export function StatGrid({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4', className)}>
      {children}
    </div>
  )
}

/** A small square colour swatch for legends / class rows. */
export function Swatch({ color }: { color: string }) {
  return (
    <span
      className="inline-block h-3 w-3 shrink-0 rounded-sm border border-black/10"
      style={{ backgroundColor: color }}
      aria-hidden="true"
    />
  )
}
