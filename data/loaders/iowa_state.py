"""Real-data validation of Module B on the Meeker & Escobar accelerated-degradation datasets.

Source : Iowa State University, "Data Sets Used in Statistical Methods for Reliability Data (2nd ed.)",
         figshare DOI 10.25380/iastate.14454765 (CC BY 4.0). See docs/DATASETS.md §3.2.
Files  : DeviceB.csv (IC power drop, 34 units @150/195/237 °C), Resistor.csv (29 units @83/133/173 °C),
         LED-Afull.csv (180 units, 6 temperature x current groups).
Mapping: stress group -> Lot_ID, unit -> Component_ID; each group's test horizon is rescaled onto 168 h
         and unit paths are linearly interpolated at 0, 24, 96, 168 h-equivalent (fractions 0, 1/7, 4/7, 1).
         Values are made positive ratios where needed (DeviceB dB -> power ratio 10^(dB/10);
         Resistor % increase -> resistance ratio, with the implicit 1.0 at t=0; LED forward voltage in mV).
Eval   : leave-one-lot-out - Module B is refit without the held-out lot (physics exponent n included)
         and MAE of Value_168h is reported for physics / hybrid / XGBoost / Ridge / naive, per parameter.

Usage  : python data/loaders/iowa_state.py      (downloads ~70 KB to data/external/iowa_state/)
Manual : if the download fails, fetch https://ndownloader.figshare.com/files/<id> for DeviceB (28330509),
         Resistor (28330503), LED-Afull (28330518) into data/external/iowa_state/ and re-run.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import Config, VALUE_COLS  # noqa: E402
from src.data import validate  # noqa: E402
from src.evaluation import mae_report  # noqa: E402
from src.features import build_features  # noqa: E402
from src.module_b import MODELS, ModuleB  # noqa: E402

FILES = {"DeviceB": 28330509, "Resistor": 28330503, "LED-Afull": 28330518}
OUT_DIR = ROOT / "data" / "external" / "iowa_state"
FRACTIONS = np.array([0, 24, 96, 168]) / 168.0


def download(out_dir: Path = OUT_DIR) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, fid in FILES.items():
        path = out_dir / f"{name}.csv"
        if path.exists() and path.stat().st_size > 0:
            continue
        try:
            urllib.request.urlretrieve(f"https://ndownloader.figshare.com/files/{fid}", path)
        except Exception as e:  # network / proxy problems -> documented manual steps
            print(f"Download of {name} failed ({e}). See the module docstring for manual steps.")
            ok = False
    return ok


def _to_checkpoints(df, lot, unit, t, y, param, unit_label, spec_min=-np.inf):
    rows = []
    for L, gl in df.groupby(lot):
        horizon = gl.groupby(unit)[t].max().median()
        for U, g in gl.groupby(unit):
            g = g.sort_values(t)
            tt, yy = g[t].to_numpy(float), g[y].to_numpy(float)
            if tt.max() < horizon * 0.999 or tt.min() > 0:
                continue  # unit ended early / no initial reading
            vals = np.interp(FRACTIONS * horizon, tt, yy)
            rows.append([f"{param}_{L}", f"{param}_{L}_{U}", param, *vals, unit_label, spec_min])
    return pd.DataFrame(rows, columns=["Lot_ID", "Component_ID", "Param_Name", *VALUE_COLS, "Unit", "Spec_Min"])


def convert(src: Path = OUT_DIR) -> pd.DataFrame:
    frames = []
    db = pd.read_csv(src / "DeviceB.csv")
    db["ratio"] = 10 ** (db["Power Drop (dB)"] / 10)
    frames.append(_to_checkpoints(db, "DegreesC", "Device Number", "Hours", "ratio", "DeviceB_Power",
                                  "ratio", spec_min=10 ** (-0.05)))  # failure definition: 0.5 dB drop
    rs = pd.read_csv(src / "Resistor.csv")
    rs["ratio"] = 1 + rs["Percent Increase"] / 100
    init = rs.groupby(["Resistor ID", "DegreesC"]).size().reset_index()[["Resistor ID", "DegreesC"]]
    init = init.assign(**{"Thousands of Hours": 0.0, "ratio": 1.0})
    rs = pd.concat([rs, init], ignore_index=True)
    frames.append(_to_checkpoints(rs, "DegreesC", "Resistor ID", "Thousands of Hours", "ratio", "Resistor_R", "ratio"))
    led = pd.read_csv(src / "LED-Afull.csv")
    led["group"] = led["DegreesC"].astype(str) + "C_" + led["Current (mA)"].astype(str) + "mA"
    led["t"] = led["Hours"] - led.groupby("Unit Number")["Hours"].transform("min")
    frames.append(_to_checkpoints(led, "group", "Unit Number", "t", "Voltage (mV)", "LED_Voltage", "mV"))
    out = pd.concat(frames, ignore_index=True)
    out["Spec_Max"], out["Delta_Pct"], out["Delta_Floor"] = np.inf, np.inf, np.inf
    out["Two_Sided"], out["Log_Scale"] = True, False
    return validate(out)


def evaluate_lolo(df: pd.DataFrame, cfg: Config = Config()) -> dict:
    feat_all = build_features(df)
    preds = []
    for lot in sorted(df["Lot_ID"].unique()):
        tr, te = feat_all[feat_all["Lot_ID"] != lot], feat_all[feat_all["Lot_ID"] == lot]
        mb = ModuleB(cfg).fit(tr.reset_index(drop=True))
        p = mb.predict(te.reset_index(drop=True))
        preds.append(pd.concat([te.reset_index(drop=True)[["Param_Name", "Value_0h", "Value_168h"]],
                                p[[f"pred_168_{m}" for m in MODELS]]], axis=1))
    pred = pd.concat(preds, ignore_index=True)
    mae = mae_report(pred, {m: f"pred_168_{m}" for m in MODELS})
    n_all = ModuleB.fit_power_law(feat_all)
    return {"mae": mae, "power_law_n": n_all,
            "units_per_param": df.groupby("Param_Name")["Component_ID"].nunique().to_dict(),
            "lots": sorted(df["Lot_ID"].unique())}


def to_markdown(res: dict) -> str:
    params = list(res["units_per_param"])
    lines = ["Source: Iowa State SMRD2 degradation data (CC BY 4.0, DOI 10.25380/iastate.14454765). "
             "Stress group = lot; horizon rescaled to 168 h; leave-one-lot-out. Units: "
             + ", ".join(f"{p}: {n}" for p, n in res["units_per_param"].items()) + "; power-law n (all lots): "
             + ", ".join(f"{p}={n:.2f}" for p, n in res["power_law_n"].items()) + ".", "",
             "| Model | " + " | ".join(f"{p} MAE" for p in params) + " | " + " | ".join(f"{p} nMAE %" for p in params)
             + " |", "|---|" + "---|" * (2 * len(params))]
    for m, r in res["mae"].items():
        lines.append(f"| {m} | " + " | ".join(f"{r['per_param'][p]['mae']:.4g}" for p in params) + " | "
                     + " | ".join(f"{r['per_param'][p]['nmae_pct']:.3f}" for p in params) + " |")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not download():
        print("Skipping real-data validation (data not available).")
        return
    df = convert()
    df.to_csv(OUT_DIR / "iowa_state_schema.csv", index=False)
    print(f"Converted {len(df)} units into {df['Lot_ID'].nunique()} lots -> {OUT_DIR / 'iowa_state_schema.csv'}")
    res = evaluate_lolo(df)
    res["markdown"] = to_markdown(res)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "real_data_iowa.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(res["markdown"])


if __name__ == "__main__":
    main()
