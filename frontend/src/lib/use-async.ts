import { useEffect, useState, type DependencyList } from 'react'

export interface AsyncState<T> {
  data: T | null
  error: Error | null
  loading: boolean
}

/**
 * Minimal data-fetching hook: runs `fn` on mount and whenever `deps` change,
 * tracking loading / data / error and ignoring results from stale runs. Keeps
 * the app dependency-light (no react-query) while giving pages a clean API.
 */
export function useAsync<T>(fn: () => Promise<T>, deps: DependencyList): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ data: null, error: null, loading: true })

  useEffect(() => {
    let cancelled = false
    setState({ data: null, error: null, loading: true })
    fn().then(
      (data) => {
        if (!cancelled) setState({ data, error: null, loading: false })
      },
      (error: unknown) => {
        if (!cancelled) {
          setState({
            data: null,
            error: error instanceof Error ? error : new Error(String(error)),
            loading: false,
          })
        }
      },
    )
    return () => {
      cancelled = true
    }
    // `fn` is intentionally excluded; callers control re-runs via `deps`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}
