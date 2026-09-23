import { Layers, Radar, Combine, GitCompareArrows, type LucideIcon } from 'lucide-react'
import type { InputMode } from '@/types/api'
import { MODE_ORDER, MODE_DESCRIPTIONS, MODE_SLOTS } from '@/lib/uploads'
import { modeLabel } from '@/lib/format'
import { cn } from '@/lib/utils'

const MODE_ICONS: Record<InputMode, LucideIcon> = {
  single_optical: Layers,
  single_sar: Radar,
  optical_sar_pair: Combine,
  bitemporal_pair: GitCompareArrows,
}

interface ModeSelectorProps {
  value: InputMode
  onChange: (mode: InputMode) => void
  disabled?: boolean
}

export function ModeSelector({ value, onChange, disabled }: ModeSelectorProps) {
  return (
    <fieldset disabled={disabled} className="space-y-3">
      <legend className="text-sm font-medium text-foreground">Input mode</legend>
      <div role="radiogroup" className="grid gap-3 sm:grid-cols-2">
        {MODE_ORDER.map((mode) => {
          const Icon = MODE_ICONS[mode]
          const selected = value === mode
          const slots = MODE_SLOTS[mode]
          return (
            <label
              key={mode}
              className={cn(
                'flex cursor-pointer gap-3 rounded-lg border p-4 transition-colors',
                'hover:border-primary/60 hover:bg-accent/50',
                selected ? 'border-primary bg-accent ring-1 ring-primary' : 'border-border',
                disabled && 'cursor-not-allowed opacity-60',
              )}
            >
              <input
                type="radio"
                name="input-mode"
                value={mode}
                checked={selected}
                onChange={() => onChange(mode)}
                disabled={disabled}
                className="sr-only"
              />
              <Icon
                className={cn(
                  'mt-0.5 h-5 w-5 shrink-0',
                  selected ? 'text-primary' : 'text-muted-foreground',
                )}
                aria-hidden="true"
              />
              <span className="space-y-1">
                <span className="block text-sm font-medium text-foreground">{modeLabel(mode)}</span>
                <span className="block text-xs text-muted-foreground">
                  {MODE_DESCRIPTIONS[mode]}
                </span>
                <span className="block text-xs font-medium text-muted-foreground">
                  {slots.length === 1 ? '1 image' : `${slots.length} images`}
                </span>
              </span>
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}
