import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    // Streamlit (8501) is the QA UI — it shares ML + reports/current_results.json
    // with FastAPI. REST endpoints live on FastAPI (8000), not Streamlit.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/score': 'http://127.0.0.1:8000',
      '/lot': 'http://127.0.0.1:8000',
      '/version': 'http://127.0.0.1:8000',
    },
  },
})
