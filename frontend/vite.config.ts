/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    // Proxy API calls to the FastAPI backend during development so the browser
    // talks to a same-origin URL and CORS never enters the picture locally.
    proxy: {
      // Target is 127.0.0.1, not `localhost`, on purpose. Uvicorn binds IPv4 only, while Node
      // resolves `localhost` to ::1 first on Windows — so every proxied request paid a refused
      // IPv6 connect before falling back to IPv4, and would fail outright without that fallback.
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // Upload previews are served by the backend's StaticFiles mount at /storage, and the upload
      // response returns that path verbatim. Without this entry the <img> resolved against the Vite
      // dev server instead, so every input thumbnail 404'd and rendered as a broken image.
      '/storage': {
        target: process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Split third-party code into one cacheable vendor chunk, separate from
        // app code. Kept as a *single* chunk on purpose: finer splitting (React
        // apart from its consumers) reorders chunk initialisation and breaks
        // `forwardRef` at runtime. The combined vendor stays under the 500 kB
        // warning threshold, so one chunk is both correct and sufficient.
        manualChunks(id) {
          if (id.includes('node_modules')) return 'vendor'
          return undefined
        },
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    css: true,
    include: ['src/**/*.{test,spec}.{ts,tsx}', 'tests/unit/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['tests/e2e/**', 'node_modules/**', 'dist/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/**/*.{ts,tsx}'],
      exclude: ['src/**/*.{test,spec}.{ts,tsx}', 'src/main.tsx', 'src/fixtures/**'],
    },
  },
})
