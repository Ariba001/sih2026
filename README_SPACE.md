---
title: BurnTestr API
emoji: 🔥
colorFrom: slate
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# BurnTestr API (Hugging Face Space)

FastAPI backend for the BurnTestr React dashboard.

Public endpoints:
- `GET /health`
- `GET /api/results`
- `POST /api/upload`

Set Space secret `CORS_ORIGINS` to your Vercel frontend URL after deploy.
