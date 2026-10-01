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

    try:
        from joblib.numpy_pickle import NumpyUnpickler as _BaseUnpickler
    except Exception:  # pragma: no cover
        _BaseUnpickler = pickle.Unpickler

    class _RestrictedUnpickler(_BaseUnpickler):
        def find_class(self, module, name):
            if module == "src.module_a_vae" and name in {"ModuleAVAE", "_VAE"}:
                return _TorchFreeVAE
            if module.startswith("torch"):
                return _TorchFreeVAE
            return super().find_class(module, name)

    with open(model_path, "rb") as f:
        return _RestrictedUnpickler(f).load()
