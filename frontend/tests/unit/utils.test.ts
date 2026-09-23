import { describe, it, expect } from 'vitest'
import { cn } from '@/lib/utils'

describe('cn', () => {
  it('joins truthy class names', () => {
    expect(cn('a', 'b')).toBe('a b')
  })

  it('drops falsy values', () => {
    expect(cn('a', false, undefined, null, 'b')).toBe('a b')
  })

  it('de-conflicts Tailwind utilities (last wins)', () => {
    expect(cn('px-2', 'px-4')).toBe('px-4')
  })

  it('supports conditional objects', () => {
    expect(cn('base', { active: true, hidden: false })).toBe('base active')
  })
})
