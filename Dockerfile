# BurnTestr FastAPI — for Render / Railway / Fly / Hugging Face Spaces
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt

COPY app_api.py train.py pyproject.toml ./
COPY src ./src
COPY models ./models
COPY data ./data
RUN mkdir -p reports

EXPOSE 8000 7860

# HF Spaces uses 7860; Render/Railway set $PORT
CMD ["sh", "-c", "uvicorn app_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
