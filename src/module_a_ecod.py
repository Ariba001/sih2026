"""ECOD (Empirical Cumulative Distribution) detector - deterministic outlier detection.

Deterministic alternative to Isolation Forest: no random_state, non-parametric,
provides exact tail probabilities for explainability. Fitted per parameter on training
data and transfers to test data.
"""
import numpy as np
import pandas as pd

try:
    from pyod.models.ecod import ECOD
    HAS_ECOD = True
except ImportError:
    HAS_ECOD = False


class ModuleAECOD:
    """Empirical Cumulative Distribution outlier detector per parameter."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ecod_models = {}
        self.ecod_norm = {}

    def fit(self, feat: pd.DataFrame):
        """Fit ECOD per parameter (deterministic, no random_state).

        Args:
            feat: DataFrame with A_FEATURES columns, grouped by Param_Name

        Returns:
            self
        """
        if not HAS_ECOD:
            return self

        from .features import A_FEATURES

        for p, g in feat.groupby("Param_Name"):
            X = g[A_FEATURES].to_numpy(float)
            if len(X) < 10:
                continue  # Skip tiny parameter groups

            model = ECOD()
            model.fit(X)
            scores = model.decision_scores_

            self.ecod_models[p] = model
            # Normalize: median -> 0, 99th percentile -> 1.0
            med = float(np.median(scores))
            q99 = float(np.quantile(scores, 0.99))
            self.ecod_norm[p] = (med, q99)

        return self

    def score(self, feat: pd.DataFrame) -> np.ndarray:
        """Return normalized ECOD scores (1.0 = at anomaly threshold).

        Args:
            feat: DataFrame with A_FEATURES columns, grouped by Param_Name

        Returns:
            1D array of normalized scores, same length as feat
        """
        if not HAS_ECOD or not self.ecod_models:
            return np.zeros(len(feat))

        from .features import A_FEATURES

        out = np.zeros(len(feat))

        for p, idx in feat.groupby("Param_Name").indices.items():
            if p not in self.ecod_models:
                continue

            X = feat.iloc[idx][A_FEATURES].to_numpy(float)
            scores = self.ecod_models[p].decision_function(X)
            med, q99 = self.ecod_norm[p]

            # Normalize: median -> 0, q99 -> 1.0
            normalized = (scores - med) / max(q99 - med, 1e-9)
            out[idx] = normalized

        return out
