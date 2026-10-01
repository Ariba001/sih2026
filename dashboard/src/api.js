import axios from 'axios'

const api = axios.create({
  baseURL: (import.meta.env.VITE_API_URL || '').replace(/\/$/, ''),
  timeout: 120000,
})

export async function fetchHealth() {
  const { data } = await api.get('/api/health')
  return data
}

export async function fetchCurrentResults() {
  const { data } = await api.get('/api/results')
  return data
}

export async function uploadCsv(file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/api/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export default api
