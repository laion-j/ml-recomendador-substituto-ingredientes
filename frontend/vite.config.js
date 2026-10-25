import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' }, // worker como módulo ES (permite import do tfjs)
  test: { environment: 'node', setupFiles: ['./vitest.setup.js'] },
})
