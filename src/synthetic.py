"""Physics-grounded synthetic burn-in dataset in the primary wide schema.

Columns: Lot_ID, Component_ID, Param_Name, Value_0h, Value_24h, Value_96h, Value_168h,
         Unit, Spec_Min, Spec_Max, Is_Defective, Defect_Type, Injected_Glitch

Model (docs/RESEARCH.md §5, §7.1; docs/DATASETS.md §5):
  * hierarchy: lot log-offset -> part baseline (log-normal; heavy-tailed for leakage) with a shared
    process corner (leaky parts are also fast) -> power-law drift  dlogP(t) = A * (t/168)^n,
    n per parameter (BTI-like 0.2-0.4), A log-normal around a lot median
  * measurement noise = relative noise + absolute noise floor, clipped at the instrument floor
  * injected latent defects (component level, 1-2 parameters) with ground-truth labels
  * tester glitches on *good* parts (single-read spikes; labelled good) to create realistic overkill
Is_Defective / Defect_Type are per row; a component is defective if any of its rows is.
"""
import numpy as np
import pandas as pd

from .config import DATASHEET, TIMEPOINTS, VALUE_COLS

T = np.array(TIMEPOINTS, dtype=float)
TN = T / 168.0

# median, lot_sd(log), part_sd(log), rel_noise, abs_noise_floor, drift_A (log units @168h), drift_n, corner_sign
PARAM_BASE = {
    "Iddq": dict(median=5.0, lot_sd=0.25, part_sd=0.25, noise=0.015, floor=0.05, A=0.04, n=0.30, corner=1.0),
    "Leakage": dict(median=3.0, lot_sd=0.35, part_sd=0.45, noise=0.02, floor=0.08, A=0.05, n=0.40, corner=1.0),
    "Prop_Delay": dict(median=12.0, lot_sd=0.04, part_sd=0.025, noise=0.003, floor=0.01, A=0.012, n=0.20,
                       corner=-1.0),
}

DEFECT_MIX = {  # relative frequency among defective components
    "gross_outlier": 0.07,
    "lot_outlier": 0.20,
    "accel_drift": 0.23,   # super-linear (n > 1) runaway, subtle at 24h
    "late_bloomer": 0.18,  # normal until 40-90h, separates only after 96h
    "step_jump": 0.12,     # sudden shift between 24h and 96h
    "erratic": 0.20,       # noisy / intermittent readings
}


def _defect_log_path(kind, rng, b, spec_max, lot_med, v0):
    """Additive log-space deviation for an injected defect (or absolute path for gross/lot outliers)."""
    scale = 0.25 if b["part_sd"] < 0.05 else 1.0  # delay moves by % where currents move by x-fold
    if kind == "accel_drift":
        m, a = rng.uniform(1.3, 2.5), rng.uniform(0.08, 0.8) * scale
        sign = -1.0 if (b["corner"] < 0 and rng.random() < 0.3) else 1.0
        return sign * a * TN ** m
    if kind == "late_bloomer":
        t_on, a = rng.uniform(40, 90), rng.uniform(0.06, 0.5) * scale
        return a * np.clip((T - t_on) / (168 - t_on), 0, None) ** 1.5
    if kind == "step_jump":
        return rng.uniform(0.08, 0.6) * (0.3 if scale < 1 else 1.0) * (T >= 96)
    if kind == "erratic":
        return rng.normal(0, rng.uniform(0.05, 0.25) * (0.3 if scale < 1 else 1.0), 4) * np.r_[0, 1, 1, 1]
    raise ValueError(kind)


def generate(n_lots=20, parts_per_lot=None, defect_rate=0.035, glitch_rate=0.003, seed=7,
             small_lot_frac=0.5, small_range=(30, 80), large_range=(250, 500), add_spatial=True) -> pd.DataFrame:
    """parts_per_lot: int (fixed), (min, max) tuple (uniform), or None (mix of small & large lots).

    add_spatial: if True, generate die_x, die_y, wafer_id, tool_id columns for spatial analysis.
    """
    rng = np.random.default_rng(seed)
    params = list(PARAM_BASE)
    kinds, probs = list(DEFECT_MIX), np.array(list(DEFECT_MIX.values()))
    probs = probs / probs.sum()
    rows = []
    for li in range(1, n_lots + 1):
        lot_id = f"L{li:02d}"
        if isinstance(parts_per_lot, int):
            n_parts = parts_per_lot
        elif parts_per_lot is not None:
            n_parts = int(rng.integers(parts_per_lot[0], parts_per_lot[1] + 1))
        else:
            lo, hi = small_range if rng.random() < small_lot_frac else large_range
            n_parts = int(rng.integers(lo, hi + 1))
        lot_shift = {p: rng.normal(0, PARAM_BASE[p]["lot_sd"]) for p in params}
        lot_A = {p: PARAM_BASE[p]["A"] * rng.uniform(0.5, 1.6) for p in params}
        lot_n = {p: PARAM_BASE[p]["n"] * np.exp(rng.normal(0, 0.1)) for p in params}

        # Generate spatial coordinates for the lot (300mm wafer simulation)
        wafer_id = rng.integers(1000, 9999)
        tool_id = rng.choice(["TOOL_A", "TOOL_B", "TOOL_C"])

        for ci in range(1, n_parts + 1):
            comp_id = f"{lot_id}-{ci:04d}"
            corner = rng.normal(0, 1)
            defective = rng.random() < defect_rate
            kind = rng.choice(kinds, p=probs) if defective else None
            n_aff = rng.choice([1, 2], p=[0.7, 0.3]) if defective else 0
            affected = set(rng.choice(params, n_aff, replace=False)) if defective else set()

            # Generate wafer coordinates (die positions on 300mm wafer, normalized to ~0-30mm die grid)
            die_x = rng.uniform(-15, 15)
            die_y = rng.uniform(-15, 15)

            for p in params:
                b, spec = PARAM_BASE[p], DATASHEET[p]
                lot_med = b["median"] * np.exp(lot_shift[p])
                v0 = lot_med * np.exp(b["part_sd"] * (0.6 * b["corner"] * corner + 0.8 * rng.normal()))
                A = lot_A[p] * np.exp(rng.normal(0, 0.3))
                n = lot_n[p] * np.exp(rng.normal(0, 0.15))
                dlog = A * TN ** n
                is_def = p in affected
                if is_def and kind == "gross_outlier":
                    base = spec["max"] * rng.uniform(1.05, 1.6)
                    true = base * np.exp(dlog)
                elif is_def and kind == "lot_outlier":
                    z = rng.uniform(4.5, 9.0)
                    target = min(lot_med * np.exp(z * b["part_sd"]), spec["max"] * 0.95)
                    true = target * np.exp(dlog)
                else:
                    extra = _defect_log_path(kind, rng, b, spec["max"], lot_med, v0) if is_def else 0.0
                    true = v0 * np.exp(dlog + extra)
                meas = true * np.exp(rng.normal(0, b["noise"], 4)) + rng.normal(0, b["floor"], 4)
                glitch = (not is_def) and rng.random() < glitch_rate
                if glitch:
                    meas[rng.choice([1, 2])] *= rng.uniform(1.2, 1.6)
                meas = np.maximum(meas, b["floor"] / 2)  # instrument floor

                row = [lot_id, comp_id, p, *np.round(meas, 4), spec["unit"], spec["min"], spec["max"],
                       int(is_def), kind if is_def else "none", int(glitch)]

                if add_spatial:
                    row.extend([round(die_x, 2), round(die_y, 2), wafer_id, tool_id])

                rows.append(row)

    cols = ["Lot_ID", "Component_ID", "Param_Name", *VALUE_COLS, "Unit", "Spec_Min", "Spec_Max",
            "Is_Defective", "Defect_Type", "Injected_Glitch"]

    if add_spatial:
        cols.extend(["die_x", "die_y", "wafer_id", "tool_id"])

    return pd.DataFrame(rows, columns=cols)
