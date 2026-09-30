"""VAE (Variational Autoencoder) detector for Module A ensemble.

Trained only on GOOD parts: learns the compressed manifold of "what a normal
component looks like" across the 9 lot-normalised A_FEATURES.  At inference,
reconstruction error (MSE between input and decoded output) serves as the
anomaly score — parts the VAE cannot recreate well are structurally unlike
anything it saw during training, and deserve a closer look.

Normalization follows the same convention as IsolationForest / ECOD:
    vae_score = (recon_error - median) / (q99 - median)
so 1.0 ≈ the 99th-percentile reconstruction error on training data.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .features import A_FEATURES


# --------------------------------------------------------------------------- #
# VAE architecture                                                             #
# --------------------------------------------------------------------------- #

class _VAE(nn.Module):
    """Symmetric encoder–decoder with a diagonal-Gaussian latent."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # Encoder: input → hidden → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent → hidden → input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def _vae_loss(recon_x, x, mu, logvar, beta_kl=1.0):
    """Reconstruction (MSE) + KL divergence to N(0,I)."""
    recon = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta_kl * kl, recon, kl


# --------------------------------------------------------------------------- #
# Public detector class (follows ECOD / IForest pattern)                       #
# --------------------------------------------------------------------------- #

class ModuleAVAE:
    """Variational Autoencoder anomaly detector per parameter.

    Parameters
    ----------
    cfg : Config
        Project configuration (uses random_state for reproducibility).
    latent_dim : int
        Bottleneck size.  Default 4 strikes a balance between compression
        and expressive capacity for the 9-dim A_FEATURES input.
    hidden_dim : int
        Width of each hidden layer.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    beta_kl : float
        KL-divergence weight in the ELBO.  A value < 1 prioritises
        reconstruction fidelity (β-VAE regime), which makes the
        reconstruction-error anomaly score more sensitive.
    """

    def __init__(self, cfg, *, latent_dim=4, hidden_dim=32, epochs=80,
                 lr=1e-3, batch_size=64, beta_kl=0.5):
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.beta_kl = beta_kl

        # Per-parameter state (trained models + normalisation stats)
        self.models = {}        # {param_name: _VAE}
        self.scalers = {}       # {param_name: (mean, std)}  input standardisation
        self.norm = {}          # {param_name: (median, q99)} score normalisation

    # ------------------------------------------------------------------ fit
    def fit(self, feat: pd.DataFrame):
        """Train one VAE per Param_Name on the GOOD-only rows.

        Good parts are those either unlabelled (production) or explicitly
        labelled Is_Defective == 0.  This ensures the VAE learns the
        manifold of normal behaviour only.
        """
        torch.manual_seed(self.cfg.random_state)
        np.random.seed(self.cfg.random_state)

        for p, g in feat.groupby("Param_Name"):
            # Keep only good parts for training
            if "Is_Defective" in g.columns:
                g_good = g[g["Is_Defective"] == 0]
            else:
                g_good = g  # production: all assumed good
            X = g_good[A_FEATURES].to_numpy(float)
            if len(X) < 20:
                continue  # Too few samples for meaningful VAE training

            # Standardise input (zero-mean, unit-variance per feature)
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std < 1e-9] = 1.0
            X_norm = (X - mean) / std
            self.scalers[p] = (mean, std)

            # Train
            model = _VAE(len(A_FEATURES), self.hidden_dim, self.latent_dim)
            model.train()
            optimiser = torch.optim.Adam(model.parameters(), lr=self.lr,
                                         weight_decay=1e-5)
            dataset = TensorDataset(torch.tensor(X_norm, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, drop_last=False)

            for _ in range(self.epochs):
                for (batch,) in loader:
                    recon, mu, logvar = model(batch)
                    loss, _, _ = _vae_loss(recon, batch, mu, logvar,
                                           self.beta_kl)
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

            model.eval()
            self.models[p] = model

            # Compute normalisation stats on the training good-parts
            recon_err = self._raw_recon_error(model, X_norm)
            med = float(np.median(recon_err))
            q99 = float(np.quantile(recon_err, 0.99))
            self.norm[p] = (med, q99)

        return self

    # ------------------------------------------------------------------ score
    def score(self, feat: pd.DataFrame) -> np.ndarray:
        """Return normalised VAE reconstruction-error scores.

        1.0 ≈ 99th-percentile error on the good-parts training set.
        """
        if not self.models:
            return np.zeros(len(feat))

        out = np.zeros(len(feat))

        for p, idx in feat.groupby("Param_Name").indices.items():
            if p not in self.models:
                continue
            X = feat.iloc[idx][A_FEATURES].to_numpy(float)
            mean, std = self.scalers[p]
            X_norm = (X - mean) / std

            recon_err = self._raw_recon_error(self.models[p], X_norm)
            med, q99 = self.norm[p]
            out[idx] = (recon_err - med) / max(q99 - med, 1e-9)

        return out

    # ------------------------------------------------------------------ internals
    @staticmethod
    @torch.no_grad()
    def _raw_recon_error(model: _VAE, X_norm: np.ndarray) -> np.ndarray:
        """Per-sample MSE reconstruction error (no normalisation)."""
        t = torch.tensor(X_norm, dtype=torch.float32)
        recon, _, _ = model(t)
        return ((recon - t) ** 2).mean(dim=1).numpy()
