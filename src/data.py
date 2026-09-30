"""Loading / validating the wide burn-in schema and lot-based splitting."""
import numpy as np
import pandas as pd

from .config import REQUIRED_COLS, VALUE_COLS, spec_for


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected {REQUIRED_COLS}")
    df = df.copy()
    for c in ["Lot_ID", "Component_ID", "Param_Name"]:
        df[c] = df[c].astype(str)
    for c in VALUE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[VALUE_COLS].isna().any().any():
        n = int(df[VALUE_COLS].isna().any(axis=1).sum())
        raise ValueError(f"{n} rows contain missing / non-numeric values in {VALUE_COLS}")
    if (df[VALUE_COLS] <= 0).any().any():
        # log-features need positive values; clip tiny/negative readings to a floor
        df[VALUE_COLS] = df[VALUE_COLS].clip(lower=1e-6)
    if df.duplicated(["Component_ID", "Param_Name"]).any():
        raise ValueError("Duplicate (Component_ID, Param_Name) rows found")
    # Fill datasheet / delta limits from config when not provided per row.
    for col, key in [("Spec_Min", "min"), ("Spec_Max", "max"), ("Unit", "unit"), ("Delta_Pct", "delta_pct"),
                     ("Delta_Floor", "delta_floor"), ("Two_Sided", "two_sided"), ("Log_Scale", "log_scale")]:
        default = df["Param_Name"].map(lambda p: spec_for(p)[key])
        df[col] = df[col].fillna(default) if col in df.columns else default
    for c in ["Spec_Max", "Spec_Min", "Delta_Pct", "Delta_Floor"]:
        df[c] = df[c].astype(float)
    for c in ["Two_Sided", "Log_Scale"]:
        df[c] = df[c].astype(bool)
    if "Is_Defective" in df.columns:
        df["Is_Defective"] = df["Is_Defective"].astype(int)
    if "Defect_Type" in df.columns:
        df["Defect_Type"] = df["Defect_Type"].fillna("none").astype(str)
    return df.reset_index(drop=True)


def load_csv(path_or_buffer) -> pd.DataFrame:
    return validate(pd.read_csv(path_or_buffer))


def has_labels(df: pd.DataFrame) -> bool:
    return "Is_Defective" in df.columns


def split_lots(df: pd.DataFrame, fractions: dict, seed: int = 42) -> dict:
    """Split by Lot_ID (never by row) so no lot leaks between train/val/test."""
    lots = np.array(sorted(df["Lot_ID"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(lots)
    n = len(lots)
    n_test = max(1, int(round(fractions["test"] * n)))
    n_val = max(1, int(round(fractions["val"] * n)))
    parts = {"test": lots[:n_test], "val": lots[n_test:n_test + n_val], "train": lots[n_test + n_val:]}
    return {k: df[df["Lot_ID"].isin(v)].reset_index(drop=True) for k, v in parts.items()}


def component_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Component-level ground truth: defective if any parameter row is defective."""
    g = df.groupby(["Lot_ID", "Component_ID"])
    out = g["Is_Defective"].max().rename("Is_Defective").to_frame()
    out["Defect_Type"] = g["Defect_Type"].agg(
        lambda s: ",".join(sorted(set(s) - {"none"})) or "none")
    return out.reset_index()
