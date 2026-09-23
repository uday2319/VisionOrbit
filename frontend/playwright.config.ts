import { defineConfig, devices } from '@playwright/test'

/**
 * Playwright end-to-end config (brief §21–26).
 *
 * The suite drives the real app against the bundled fixtures — the same
 * recorded tool output the UI ships with — so it needs no backend and is fully
 * deterministic. `VITE_USE_FIXTURES=true` is forced here so the run does not
 * depend on any local `.env`, and a dedicated port keeps it clear of the dev
 * server you may already have open.
 */
const PORT = 5273
const BASE_URL = `http://localhost:${PORT}`

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? [['github'], ['list']] : [['list']],
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: BASE_URL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: `npm run dev -- --port ${PORT} --strictPort`,
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: { VITE_USE_FIXTURES: 'true' },
  },
})
