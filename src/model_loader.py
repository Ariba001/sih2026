"""Load BurnTestr joblib with optional torch-free stubs for small free hosts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class _TorchFreeVAE:
    """Stand-in when torch is unavailable or BURNTESTR_DISABLE_TORCH=1."""

    def __init__(self, *args, **kwargs):
        self.models = {}
        self.scalers = {}
        self.norm = {}

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        self.models = {}

    def fit(self, feat: pd.DataFrame):
        return self

    def score(self, feat: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(feat), dtype=float)


def _want_torch() -> bool:
    if os.getenv("BURNTESTR_DISABLE_TORCH", "").strip().lower() in {"1", "true", "yes"}:
        return False
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def load_burnin_system(model_path: Path) -> Any:
    """joblib.load with torch, or restricted unpickle that stubs VAE/torch."""
    import joblib

    if _want_torch():
        return joblib.load(model_path)

    import pickle
    import io

    class _RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "src.module_a_vae" and name in {"ModuleAVAE", "_VAE"}:
                return _TorchFreeVAE
            if module.startswith("torch"):
                return _TorchFreeVAE
            return super().find_class(module, name)

    # joblib 1.x may wrap pickle; prefer joblib.load after stubbing modules
    import sys
    import types

    stub = types.ModuleType("torch")
    stub.__dict__.update({"__version__": "0.0-stub"})
    sys.modules.setdefault("torch", stub)
    sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
    sys.modules.setdefault("torch.utils", types.ModuleType("torch.utils"))
    sys.modules.setdefault("torch.utils.data", types.ModuleType("torch.utils.data"))

    import src.module_a_vae as vae_mod

    vae_mod.ModuleAVAE = _TorchFreeVAE  # type: ignore[misc,assignment]
    vae_mod._VAE = _TorchFreeVAE  # type: ignore[misc,assignment]

    try:
        return joblib.load(model_path)
    except Exception:
        raw = Path(model_path).read_bytes()
        return _RestrictedUnpickler(io.BytesIO(raw)).load()
