"""Train on training lots, calibrate + tune cost-optimal thresholds on validation lots, evaluate on test lots.

Usage: python train.py [--data data/burnin_synthetic.csv] [--cost-fn 50] [--cost-fp 1] [--dpat-k 6]
Outputs: models/burnin_system.joblib, models/scored_synthetic.joblib (dashboard cache),
         reports/metrics.json, reports/metrics.md, reports/recall_overkill_curves.png, reports/test_decisions.csv
"""
import argparse
import json
import sys
from pathlib import Path

from src.config import Config
from src.data import load_csv, split_lots
from src.pipeline import BurnInSystem
from src.report import write_reports
from src.synthetic import generate

ROOT = Path(__file__).resolve().parent


def pick_examples(comp, rows):
    """A few REJECT justifications covering different reasons / binding safety-slope limits."""
    rej = comp[comp["Decision"] == "REJECT"].sort_values("C_score", ascending=False)
    out, seen = [], set()
    top = rows.sort_values("C_score", ascending=False).drop_duplicates("Component_ID").set_index("Component_ID")
    for _, c in rej.iterrows():
        r = top.loc[c["Component_ID"]]
        key = ("rule" if c["rule_fail"] else "ai", r["binding_up"] if r["B_up"] >= r["B_down"] else r["binding_lo"])
        if key not in seen:
            seen.add(key)
            out.append(c["Explanation"])
        if len(out) >= 5:
            break
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data" / "burnin_synthetic.csv"))
    ap.add_argument("--cost-fn", type=float, default=50.0)
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--dpat-k", type=float, default=6.0)
    a = ap.parse_args()

    path = Path(a.data)
    if not path.exists():
        print(f"{path} not found - generating synthetic data")
        generate().to_csv(path, index=False)
    df = load_csv(path)
    cfg = Config(cost_fn=a.cost_fn, cost_fp=a.cost_fp, dpat_k=a.dpat_k)
    splits = split_lots(df, cfg.lot_split, cfg.random_state)
    lots = {k: sorted(v["Lot_ID"].unique()) for k, v in splits.items()}
    print("Lot split:", lots)

    system = BurnInSystem(cfg).fit(splits["train"], splits["val"])
    (ROOT / "models").mkdir(exist_ok=True)
    system.save(ROOT / "models" / "burnin_system.joblib")

    metrics = system.evaluate(splits["test"])
    metrics.update(test_lots=lots["test"], val_lots=lots["val"], train_lots=lots["train"],
                   pi_quantile=cfg.pi_quantile, dpat_k=cfg.dpat_k)
    rows, comp = system.score(splits["test"])
    (ROOT / "reports").mkdir(exist_ok=True)
    comp.to_csv(ROOT / "reports" / "test_decisions.csv", index=False)
    real = ROOT / "reports" / "real_data_iowa.json"
    real_data = json.loads(real.read_text(encoding="utf-8")) if real.exists() else None
    write_reports(metrics, ROOT / "reports", pick_examples(comp, rows), real_data)

    print("Precomputing dashboard cache (scores + explanations + SHAP for all lots)...")
    system.precompute(df, ROOT / "models" / "scored_synthetic.joblib")

    for k in ["rules_only", "module_a", "combined_reject_only", "combined_reject_or_review", "module_b_early_24h"]:
        r = metrics[k]
        print(f"{k:26s} recall={r['recall']:.3f} precision={r['precision']:.3f} F2={r['f2']:.3f} "
              f"escapes={r['escapes']} (95% UB escape rate {100 * r['escape_rate_ub95']:.1f}%) "
              f"overkill={100 * r['overkill_rate']:.2f}% cm={r['confusion_matrix']}")
    print(f"Module B default: {metrics['module_b_default']}, n={metrics['power_law_n']}")
    for m, r in metrics["mae_value_168h"].items():
        pp = ", ".join(f"{p}={v['mae']:.4f}" for p, v in r["per_param"].items())
        print(f"MAE {m:9s} overall={r['overall_mae']:.4f} nMAE={r['overall_nmae_pct']:.2f}% "
              f"tail={r['tail_nmae_pct']:.2f}%  ({pp})")
    print(f"Coverage upper={metrics['pi_coverage_upper']:.3f} band={metrics['pi_coverage_band']:.3f}; "
          f"binding limits: {metrics['binding_limit_counts']}")
    t = metrics["thresholds"]
    print(f"Thresholds: Module A tau={t['module_a']['threshold']:.3f}; combined tauA={t['combined']['tau_a']:.3f}, "
          f"tauB={t['combined']['tau_b']:.3f}")
    print("Reports written to", ROOT / "reports")


if __name__ == "__main__":
    main()
