"""Deploy BurnTestr FastAPI on Modal (public ASGI URL)."""
from __future__ import annotations

import modal

app = modal.App("burntestr-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-api.txt")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("models", remote_path="/root/models")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_file("app_api.py", remote_path="/root/app_api.py")
)


@app.function(image=image, memory=4096, timeout=600, cpu=2.0)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")
    from app_api import app as fastapi

    return fastapi
