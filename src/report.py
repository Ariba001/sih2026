"""Write metrics as JSON + a concise markdown report + recall/overkill plots."""
import json
from pathlib import Path

import numpy as np


def _cls_row(name, r):
    cm = r["confusion_matrix"]
    return (f"| {name} | {r['recall']:.3f} | {r['precision']:.3f} | {r['f2']:.3f} | {r['escapes']} "
            f"| {100 * r['escape_rate_ub95']:.1f}% | {r['false_rejects']} | {100 * r['overkill_rate']:.2f}% "
            f"| {r['cost']:.0f} | TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']} |")


def save_plots(metrics: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = metrics["dpat_k_curve"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ks = [r["k"] for r in k]
    for key, lab, st in [("dpat", "Rules + DPAT only", "o-"), ("moduleA", "Rules + Module A (score ≥ 1)", "s-")]:
        ax[0].plot(ks, [r[f"{key}_recall"] for r in k], st, label=lab)
        ax[1].plot(ks, [100 * r[f"{key}_overkill"] for r in k], st, label=lab)
    ax[0].set_xlabel("DPAT k (robust σ)"); ax[0].set_ylabel("recall"); ax[0].set_title("Recall vs DPAT k (test lots)")
    ax[1].set_xlabel("DPAT k (robust σ)"); ax[1].set_ylabel("overkill %"); ax[1].set_title("Overkill vs DPAT k")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    c = metrics["cost_ratio_curve"]
    ax[2].plot([100 * r["overkill"] for r in c], [r["recall"] for r in c], "o-")
    seen = set()
    for r in c:
        pt = (round(100 * r["overkill"], 2), round(r["recall"], 4))
        if pt not in seen:
            seen.add(pt)
            ax[2].annotate(f"{r['cost_ratio']}:1", (100 * r["overkill"], r["recall"]), fontsize=7,
                           textcoords="offset points", xytext=(4, -10))
    ax[2].set_xlabel("overkill (false-reject rate, %)"); ax[2].set_ylabel("recall")
    ax[2].set_title("Combined system vs C_FN:C_FP\n(tuned on validation, scored on test)")
    fig.tight_layout()
    fig.savefig(out_dir / "recall_overkill_curves.png", dpi=120)
    plt.close(fig)


def write_reports(metrics: dict, out_dir: Path, examples: list = (), real_data: dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    save_plots(metrics, out_dir)
    t = metrics["thresholds"]
    n_def = metrics["combined_reject_only"]["n_defective"]
    lines = ["# Burn-in anomaly detection - evaluation report", "",
             f"Held-out test lots: {', '.join(metrics.get('test_lots', []))} "
             f"({metrics['combined_reject_only']['n']} components, {n_def} defective; lot sizes "
             f"{sorted(metrics['lot_sizes'].values())})  ",
             f"Cost function: cost = {t['cost_fn']:g}·FN + {t['cost_fp']:g}·FP (thresholds chosen on "
             f"validation lots {', '.join(metrics.get('val_lots', []))})  ",
             f"Module A alone: reject if A ≥ {t['module_a']['threshold']:.3f}  ",
             f"Combined: reject if rule fail (datasheet or delta limit) OR A ≥ {t['combined']['tau_a']:.3f} "
             f"OR B ≥ {t['combined']['tau_b']:.3f}  ",
             f"Module B default model: **{metrics['module_b_default']}** (lowest validation nMAE); power-law n: "
             + ", ".join(f"{p}={n:.2f}" for p, n in metrics["power_law_n"].items())
             + f"; Arrhenius AF = {metrics['arrhenius_af']:.1f}", "",
             "## Detection (component level, test lots)", "",
             "| System | Recall | Precision | F2 | Escapes | Escape-rate 95% UB (Clopper-Pearson) | "
             "False rejects | Overkill | Cost | Confusion |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, name in [("rules_only", "Rules only (datasheet + delta limits)"),
                    ("module_a", "Rules + Module A"), ("module_b_early_24h", "Module B early flag @24h"),
                    ("combined_reject_only", "Combined: REJECT"),
                    ("combined_reject_or_review", "Combined: REJECT+REVIEW")]:
        lines.append(_cls_row(name, metrics[k]))
    lines += ["", "## Recall by defect type", "",
              "| Type | n | Rules | Module A | Module B @24h | Combined REJECT |", "|---|---|---|---|---|---|"]
    for ty, r in metrics["recall_by_defect_type"].items():
        lines.append(f"| {ty} | {r['n']} | {r['rules']:.2f} | {r['module_a']:.2f} | {r['module_b_24h']:.2f} | "
                     f"{r['combined_reject']:.2f} |")
    lines += ["", "## Rule contribution (component level, individual screens at their native limits)", "",
              "| Screen | Recall | Defects caught | Good parts flagged | Overkill |", "|---|---|---|---|---|"]
    for s, r in metrics["rule_contribution"].items():
        lines.append(f"| {s} | {r['recall']:.3f} | {r['defects_caught']} | {r['good_flagged']} | "
                     f"{100 * r['overkill']:.2f}% |")
    params = list(next(iter(metrics["mae_value_168h"].values()))["per_param"])
    lines += ["", "## Module B - Value_168h MAE (test lots)", "",
              "| Model | Overall MAE | nMAE % | Tail nMAE % (top-5% drifters) | " +
              " | ".join(f"{p} MAE" for p in params) + " | " + " | ".join(f"{p} tail MAE" for p in params) + " |",
              "|---|---|---|---|" + "---|" * (2 * len(params))]
    for m, r in metrics["mae_value_168h"].items():
        star = " (default)" if m == metrics["module_b_default"] else ""
        lines.append(f"| {m}{star} | {r['overall_mae']:.4f} | {r['overall_nmae_pct']:.2f} | {r['tail_nmae_pct']:.2f} | "
                     + " | ".join(f"{r['per_param'][p]['mae']:.4f}" for p in params) + " | "
                     + " | ".join(f"{r['per_param'][p]['tail_mae']:.3f}" for p in params) + " |")
    lines += ["", f"Conformalized {int(100 * metrics.get('pi_quantile', 0.9))}% bounds: empirical upper-bound "
              f"coverage {metrics['pi_coverage_upper']:.3f}, two-sided band coverage {metrics['pi_coverage_band']:.3f}. "
              f"Binding safety-slope limit (upward) counts: {metrics['binding_limit_counts']}", "",
              "## Recall / overkill trade-offs (plot: `recall_overkill_curves.png`)", "",
              "| DPAT k | Rules+DPAT recall | overkill | escapes | Rules+Module A recall | overkill | escapes |",
              "|---|---|---|---|---|---|---|"]
    for r in metrics["dpat_k_curve"]:
        lines.append(f"| {r['k']:.1f} | {r['dpat_recall']:.3f} | {100 * r['dpat_overkill']:.2f}% | {r['dpat_escapes']} | "
                     f"{r['moduleA_recall']:.3f} | {100 * r['moduleA_overkill']:.2f}% | {r['moduleA_escapes']} |")
    lines += ["", "| C_FN:C_FP | τA | τB | Recall | Overkill | Escapes |", "|---|---|---|---|---|---|"]
    for r in metrics["cost_ratio_curve"]:
        tb = "∞" if not np.isfinite(r["tau_b"]) else f"{r['tau_b']:.2f}"
        lines.append(f"| {r['cost_ratio']}:1 | {r['tau_a']:.2f} | {tb} | {r['recall']:.3f} | "
                     f"{100 * r['overkill']:.2f}% | {r['escapes']} |")
    if real_data:
        lines += ["", "## Real-data validation (Module B, leave-one-lot-out)", "", real_data.get("markdown", "")]
    lines += ["", "![curves](recall_overkill_curves.png)", ""]
    if examples:
        lines += ["## Example QA justifications", ""] + [f"- {e}" for e in examples] + [""]
    (out_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
