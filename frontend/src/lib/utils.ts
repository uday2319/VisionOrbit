import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Merge conditional class names and de-conflict Tailwind utilities.
 * The standard shadcn/ui helper: `cn('px-2', cond && 'px-4')` => `px-4`.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}
