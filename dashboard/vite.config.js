import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    // Local FastAPI. Production builds use VITE_API_URL (see .env.example).
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/score': 'http://127.0.0.1:8000',
      '/lot': 'http://127.0.0.1:8000',
      '/version': 'http://127.0.0.1:8000',
    },
  },
})
