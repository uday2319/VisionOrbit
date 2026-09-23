/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the SatQuery API. Empty => use dev proxy / same-origin. */
  readonly VITE_API_BASE_URL?: string
  /** "true" => serve recorded fixtures instead of calling the backend. */
  readonly VITE_USE_FIXTURES?: string
  /** Dev-only proxy target for /api. */
  readonly VITE_API_PROXY_TARGET?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
