"""Module A - unsupervised, lot-localized (contextual) outlier detection.

Four complementary detectors, each normalised so that 1.0 ~ "at its own outlier limit":
  * DPAT (AEC-Q001 style): value at each timepoint (and 0->168h drift) vs lot median
    +/- k robust sigma, on log scale for log-normal currents.   dpat_score = max|z| / k
  * Robust Mahalanobis (MinCovDet), fitted *per lot & parameter* on lot-normalised
    trajectory features.                                 maha_score = sqrt(D^2 / chi2_q(p))
  * Isolation Forest per parameter, trained on lot-normalised features of the training lots
    (so it transfers to unseen lots).                    if_score = (a - a_med) / (a_q99 - a_med)
  * ECOD (deterministic, distribution-free), per parameter.    ecod_score = (s - s_med) / (s_q99 - s_med)
The row anomaly score is the max of the four (a recall-first OR-combination); a component's
score is the max over its parameters.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest

from .config import Config
from .features import A_FEATURES, GROUP, dpat_measures, robust_sigma
from .module_a_ecod import ModuleAECOD
from .module_a_vae import ModuleAVAE
from .module_a_spatial import SpatialAnomalyDetector


def mahalanobis_per_lot(feat: pd.DataFrame, random_state=0):
    """Robust Mahalanobis D^2 and additive per-feature contributions, per (lot, param).

    Contribution_i = (x-mu)_i * [P (x-mu)]_i, which sums exactly to D^2.
    """
    p = len(A_FEATURES)
    d2 = np.zeros(len(feat))
    contrib = np.zeros((len(feat), p))
    for _, idx in feat.groupby(GROUP).indices.items():
        X = feat.iloc[idx][A_FEATURES].to_numpy(float)
        if len(X) >= 5 * p:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mcd = MinCovDet(random_state=random_state).fit(X)
            mu, prec = mcd.location_, mcd.get_precision()
        else:  # tiny lot: features are already robust z-scores -> diagonal metric
            mu, prec = np.zeros(p), np.eye(p)
        diff = X - mu
        c = diff * (diff @ prec)
        contrib[idx] = c
        d2[idx] = c.sum(axis=1)
    return np.maximum(d2, 0), contrib


class ModuleA:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.iforests, self.if_norm = {}, {}
        self.ecod = ModuleAECOD(cfg)
        self.vae = ModuleAVAE(
            cfg,
            latent_dim=cfg.vae_latent_dim,
            hidden_dim=cfg.vae_hidden_dim,
            epochs=cfg.vae_epochs,
            lr=cfg.vae_lr,
            beta_kl=cfg.vae_beta_kl,
        )
        self.spatial = SpatialAnomalyDetector(cfg)

    def fit(self, feat: pd.DataFrame):
        for p, g in feat.groupby("Param_Name"):
            X = g[A_FEATURES].to_numpy(float)
            m = IsolationForest(n_estimators=self.cfg.if_trees, random_state=self.cfg.random_state)
            m.fit(X)
            a = -m.score_samples(X)
            self.iforests[p] = m
            self.if_norm[p] = (float(np.median(a)), float(np.quantile(a, 0.99)))
        self.ecod.fit(feat)
        self.vae.fit(feat)
        self.spatial.fit(feat)
        return self

    def _if_scores(self, feat):
        out = np.zeros(len(feat))
        for p, idx in feat.groupby("Param_Name").indices.items():
            if p not in self.iforests:
                continue  # unseen parameter: rely on DPAT + Mahalanobis
            a = -self.iforests[p].score_samples(feat.iloc[idx][A_FEATURES].to_numpy(float))
            med, q = self.if_norm[p]
            out[idx] = (a - med) / max(q - med, 1e-9)
        return out

    def score(self, feat: pd.DataFrame) -> pd.DataFrame:
        k = self.cfg.dpat_k
        res = feat[["Lot_ID", "Component_ID", "Param_Name"]].copy()
        zcols = []
        for c, series in dpat_measures(feat).items():
            tmp = feat[GROUP].assign(_v=series.values)
            z = tmp.groupby(GROUP)["_v"].transform(lambda s: (s - s.median()) / robust_sigma(s.values))
            res[f"dpat_z_{c}"] = z.values
            zcols.append(f"dpat_z_{c}")
        res["dpat_maxz"] = res[zcols].abs().max(axis=1)
        res["dpat_score"] = res["dpat_maxz"] / k

        d2, contrib = mahalanobis_per_lot(feat, self.cfg.random_state)
        res["maha_d2"] = d2
        res["maha_score"] = np.sqrt(d2 / chi2.ppf(self.cfg.maha_quantile, len(A_FEATURES)))
        for i, f in enumerate(A_FEATURES):
            res[f"mc_{f}"] = contrib[:, i]
        res["if_score"] = self._if_scores(feat) if self.iforests else 0.0
        res["ecod_score"] = self.ecod.score(feat)
        res["vae_score"] = self.vae.score(feat)

        # PHASE 3: SPATIAL ANOMALY DETECTOR (wafer-level clustering)
        # Score components based on isolation (anomalous surrounded by good)
        # and clustering (normal surrounded by anomalous)
        # Returns spatial_score normalized to ~1.0 = 99th percentile on good parts
        spatial_scores_df = self.spatial.score(feat, rows=res)
        res["spatial_score"] = 0.0
        if len(spatial_scores_df) > 0:
            res = res.merge(spatial_scores_df[["Component_ID", "spatial_score"]],
                           on="Component_ID", how="left", suffixes=("", "_y"))
            res["spatial_score"] = res["spatial_score_y"].fillna(0.0)
            res = res.drop(columns=["spatial_score_y"])

        # PHASE 2-3 ENSEMBLE WEIGHTING (6 detectors)
        # Updated from Phase 2's 5-detector ensemble to include spatial context:
        # - DPAT (0.30): Lot-context aware baseline (was 0.35, reduced for spatial)
        # - Mahalanobis (0.22): Multivariate covariance (was 0.25)
        # - VAE (0.18): Deep non-linear normal-parts manifold (was 0.20)
        # - Spatial (0.15): NEW - Wafer-level clustering & isolation detection
        # - Isolation Forest (0.10): High-dimensional tree partitioning (was 0.12)
        # - ECOD (0.05): Deterministic empirical CDF tails (was 0.08)
        res["A_score"] = (
            0.30 * res["dpat_score"] +
            0.22 * res["maha_score"] +
            0.18 * res["vae_score"] +
            0.15 * res["spatial_score"] +
            0.10 * res["if_score"] +
            0.05 * res["ecod_score"]
        )
        return res
