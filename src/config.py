"""Central configuration: schema, datasheet / delta limits, physics and tunable hyper-parameters."""
from dataclasses import dataclass, field

import numpy as np

TIMEPOINTS = [0, 24, 96, 168]
VALUE_COLS = [f"Value_{t}h" for t in TIMEPOINTS]
ID_COLS = ["Lot_ID", "Component_ID", "Param_Name"]
REQUIRED_COLS = ID_COLS + VALUE_COLS
LABEL_COLS = ["Is_Defective", "Defect_Type"]
OPTIONAL_SPATIAL_COLS = ["die_x", "die_y", "wafer_id", "tool_id"]
BURNIN_HOURS = 168.0

# Datasheet limits + MIL-STD-883 / ESCC style delta limit:
#   delta_allow = max(delta_pct * |Value_0h|, delta_floor)   ("X % of initial or Y, whichever is greater")
# two_sided: drift in either direction is physically meaningful (threshold / timing parameters).
# A user CSV may override limits per row with Spec_Min / Spec_Max / Delta_Pct / Delta_Floor columns.
# log_scale: DPAT on log(value) for log-normal currents (AEC-Q001 allows transformation).
DATASHEET = {
    "Iddq": {"unit": "µA", "min": 0.0, "max": 25.0, "delta_pct": 0.50, "delta_floor": 1.0,
             "two_sided": False, "log_scale": True},
    "Leakage": {"unit": "µA", "min": 0.0, "max": 50.0, "delta_pct": 1.00, "delta_floor": 0.5,
                "two_sided": False, "log_scale": True},
    "Prop_Delay": {"unit": "ns", "min": 0.0, "max": 20.0, "delta_pct": 0.08, "delta_floor": 0.2,
                   "two_sided": True, "log_scale": False},
}
DEFAULT_SPEC = {"unit": "", "min": -np.inf, "max": np.inf, "delta_pct": np.inf, "delta_floor": np.inf,
                "two_sided": True, "log_scale": False}


def spec_for(param: str) -> dict:
    return DATASHEET.get(param, DEFAULT_SPEC)


def arrhenius_af(ea_ev: float, t_stress_c: float, t_use_c: float) -> float:
    """Acceleration factor AF = exp[(Ea/k)(1/T_use - 1/T_stress)] (T in kelvin)."""
    k = 8.617e-5
    return float(np.exp(ea_ev / k * (1 / (t_use_c + 273.15) - 1 / (t_stress_c + 273.15))))


@dataclass
class Config:
    dpat_k: float = 6.0              # DPAT: robust mean +/- k robust sigma (AEC-Q001 uses 6; small lots 3-4.5)
    maha_quantile: float = 0.999     # chi-square quantile that maps to Mahalanobis score 1.0
    if_trees: int = 300
    # Phase 2: VAE hyperparameters
    vae_latent_dim: int = 4          # Latent dimension for 9-dim input
    vae_hidden_dim: int = 32         # Encoder/Decoder hidden layer width
    vae_epochs: int = 80             # Training epochs for good-parts manifold
    vae_lr: float = 1e-3             # Adam learning rate
    vae_beta_kl: float = 0.5         # β-VAE weight (lower = prioritises reconstruction)
    # Phase 3: Spatial anomaly detection hyperparameters
    spatial_k_nn: int = 6            # k-NN neighbors for spatial graph (default 6 neighbors per die)
    spatial_distance_threshold: float = 500.0  # Max distance (μm) to consider neighbors on wafer
    slope_k: float = 6.0             # lot safety slope = median + k * robust sigma
    pi_quantile: float = 0.9         # conformalized upper (and 1-q lower) prediction bound
    # Mission-life safety slope (Arrhenius)
    ea_ev: float = 0.7
    t_burnin_c: float = 125.0
    t_field_c: float = 55.0
    mission_hours: float = 15 * 8760.0
    mission_margin: float = 0.0      # fraction of (limit - V0) kept as guard band
    # Cost-sensitive thresholding
    cost_fn: float = 50.0
    cost_fp: float = 1.0
    beta: float = 2.0
    review_frac: float = 0.85
    random_state: int = 42
    lot_split: dict = field(default_factory=lambda: {"train": 0.6, "val": 0.2, "test": 0.2})

    @property
    def af(self) -> float:
        return arrhenius_af(self.ea_ev, self.t_burnin_c, self.t_field_c)
