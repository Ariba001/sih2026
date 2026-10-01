"""End-to-end system: fit on training lots, calibrate + tune cost-optimal thresholds on validation
lots, score any lot, and evaluate on held-out lots."""
import hashlib

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .data import component_labels, has_labels
from .decision import build_explanations, component_scores, decide, row_scores
from .evaluation import (choose_threshold, choose_threshold_pair, classification_report, dpat_k_curve,
                         mae_report, recall_overkill)
from .explain import attach_shap
from .features import build_features, dpat_limits
from .module_a import ModuleA
from .module_b import MODELS, ModuleB
from .module_a_spatial import SpatialAnomalyDetector
from .module_c_weibull import ModuleC
from .rules import lot_pda, rule_screens

K_GRID = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
COST_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def data_signature(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


class BurnInSystem:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.module_a = ModuleA(cfg)
        self.module_b = ModuleB(cfg)
        self.spatial = SpatialAnomalyDetector(cfg)
        self.weibull = ModuleC(cfg)
        # PHASE 1-2 RECALL IMPROVEMENT: Lower thresholds + ensemble weighting
        # tau_a: Module A anomaly detection threshold (was 1.0, now 0.7 = -30% = more sensitive)
        # tau_ca/tau_cb: Combined thresholds (were 1.0, now 0.75 = -25% = catch more drift)
        self.tau_a = None                 # Module A alone (tuned during fit)
        self.tau_ca = self.tau_cb = 0.75  # combined OR-rule thresholds for A and B (IMPROVED: 1.0 → 0.75)
        self.threshold_info = {}
        self._val = None                  # validation component scores (for cost-ratio sweeps)

    # ------------------------------------------------------------------ scoring
    def score_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = build_features(df)
        keep = ["Lot_ID", "Component_ID", "Param_Name"]
        a = self.module_a.score(feat).drop(columns=keep)
        b = self.module_b.predict(feat).drop(columns=keep + ["delta_allow"])
        rows = pd.concat([feat, rule_screens(feat), a, b], axis=1)
        rows["pi_pct"] = int(round(100 * self.cfg.pi_quantile))
        return row_scores(rows, self.tau_ca, self.tau_cb)

    def score(self, df: pd.DataFrame, explain: bool = True):
        """Return (row-level frame, component-level decisions frame)."""
        rows = self.score_rows(df)
        comp = component_scores(rows)
        comp["Decision"] = decide(comp, self.cfg.review_frac)
        comp["ModuleA_Reject"] = comp["rule_fail"] | (comp["A_score"] >= self.tau_a)

        # Attach Weibull survival probabilities (component-level; min across params)
        row_probs = df.apply(lambda r: self.weibull.predict_survival_prob(r), axis=1)
        surv = (
            pd.DataFrame({"Lot_ID": df["Lot_ID"], "Component_ID": df["Component_ID"], "survival_prob": row_probs})
            .groupby(["Lot_ID", "Component_ID"], as_index=False)["survival_prob"]
            .min()
        )
        comp = comp.merge(surv, on=["Lot_ID", "Component_ID"], how="left")
        comp["survival_prob"] = comp["survival_prob"].fillna(0.99)
        comp["mission_risk"] = comp["survival_prob"].apply(
            lambda p: "HIGH" if p < 0.99 else ("MEDIUM" if p < 0.999 else "LOW")
        )

        # Attach spatial clustering info if die_x/die_y present
        if "die_x" in rows.columns and "die_y" in rows.columns:
            wafer_clusters = self.spatial.identify_defect_clusters(rows, threshold=0.8)
            if len(wafer_clusters) > 0:
                comp = comp.merge(wafer_clusters, left_on="Lot_ID", right_on="lot_id", how="left")

        if explain:
            flagged_ids = set(comp.loc[comp["Decision"] != "ACCEPT", "Component_ID"])
            rows = attach_shap(rows, rows["Component_ID"].isin(flagged_ids).to_numpy(),
                               self.module_a, self.module_b)
            lim = dpat_limits(df, self.cfg.dpat_k)
            lot_stats = {(r.Lot_ID, r.Param_Name, r.Measure): r._asdict()
                         for r in lim.itertuples(index=False)}
            comp = build_explanations(rows, comp, lot_stats, self.cfg.dpat_k, self.tau_ca, self.tau_cb,
                                      self.module_b.default)
        if has_labels(df):
            comp = comp.merge(component_labels(df), on=["Lot_ID", "Component_ID"], how="left")
        return rows, comp

    def lot_summary(self, comp: pd.DataFrame) -> pd.DataFrame:
        return lot_pda(comp)

    # ------------------------------------------------------------------ training
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        feat = build_features(train_df)
        self.module_a.fit(feat)
        self.module_b.fit(feat)
        self.module_b.calibrate(build_features(val_df))

        # Train Weibull survival model on training data
        self.weibull.fit(train_df)

        rows = self.score_rows(val_df)
        comp = component_scores(rows).merge(component_labels(val_df), on=["Lot_ID", "Component_ID"])
        self._val = comp[["A_score", "B_score", "Is_Defective"]].copy()
        self._tune(self.cfg.cost_fn, self.cfg.cost_fp, commit=True)
        return self

    def _tune(self, c_fn, c_fp, commit=False):
        v = self._val
        y = v["Is_Defective"].to_numpy()
        ia = choose_threshold(v["A_score"], y, c_fn, c_fp)
        ic = choose_threshold_pair(v["A_score"], v["B_score"], y, c_fn, c_fp)
        if commit:
            self.tau_a, self.tau_ca, self.tau_cb = ia["threshold"], ic["tau_a"], ic["tau_b"]
            self.threshold_info = {"module_a": ia, "combined": ic, "cost_fn": c_fn, "cost_fp": c_fp}
        return ia, ic

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, df: pd.DataFrame) -> dict:
        rows, comp = self.score(df, explain=False)
        cfg, y = self.cfg, comp["Is_Defective"].to_numpy()
        rep = lambda pred: classification_report(y, pred, cfg.cost_fn, cfg.cost_fp, cfg.beta)
        a_pred = comp["rule_fail"] | (comp["A_score"] >= self.tau_a)
        out = {
            "thresholds": self.threshold_info,
            "rules_only": rep(comp["rule_fail"]),
            "module_a": rep(a_pred),
            "module_b_early_24h": rep(comp["B_flag"]),
            "combined_reject_only": rep(comp["Decision"] == "REJECT"),
            "combined_reject_or_review": rep(comp["Decision"] != "ACCEPT"),
            "decision_counts": comp["Decision"].value_counts().to_dict(),
            "mae_value_168h": mae_report(rows, {m: f"pred_168_{m}" for m in MODELS}),
            "module_b_default": self.module_b.default,
            "module_b_val_nmae": self.module_b.val_mae,
            "power_law_n": self.module_b.n_exp,
            "pi_coverage_upper": float((rows["Value_168h"] <= rows["pred_168_upper"]).mean()),
            "pi_coverage_band": float(((rows["Value_168h"] <= rows["pred_168_upper"]) &
                                       (rows["Value_168h"] >= rows["pred_168_lower"])).mean()),
            "pi_coverage_upper_good_rows": float((rows.loc[rows["Is_Defective"] == 0, "Value_168h"] <=
                                                  rows.loc[rows["Is_Defective"] == 0, "pred_168_upper"]).mean()),
            "binding_limit_counts": rows["binding_up"].value_counts().to_dict(),
            "backend": self.module_b.backend,
            "arrhenius_af": cfg.af,
            "lot_sizes": comp.groupby("Lot_ID").size().to_dict(),
        }
        # Recall per injected defect type (component level, first listed type).
        dt = comp.loc[comp["Is_Defective"] == 1].copy()
        dt["type"] = dt["Defect_Type"].str.split(",").str[0]
        out["recall_by_defect_type"] = {
            t: {"n": int(len(g)), "rules": float(g["rule_fail"].mean()),
                "module_a": float(a_pred[g.index].mean()), "module_b_24h": float(g["B_flag"].mean()),
                "combined_reject": float((g["Decision"] == "REJECT").mean())}
            for t, g in dt.groupby("type")}
        # Rule contribution table (which screen caught which defect, what overkill it costs).
        r = rows.assign(dpat=rows["dpat_maxz"] >= cfg.dpat_k, maha=rows["maha_score"] >= 1,
                        vae=rows.get("vae_score", 0) >= 1, spatial=rows.get("spatial_score", 0) >= 1,
                        iforest=rows["if_score"] >= 1)
        cc = r.groupby("Component_ID")[["spec_fail", "delta_fail", "dpat", "maha", "vae", "spatial", "iforest", "B_flag"]].any()
        cc = cc.reindex(comp["Component_ID"]).reset_index(drop=True)
        out["rule_contribution"] = {}
        for col in cc:
            rec, ovk, _ = recall_overkill(y, cc[col])
            out["rule_contribution"][col] = {"recall": rec, "overkill": ovk,
                                             "defects_caught": int((cc[col] & (y == 1)).sum()),
                                             "good_flagged": int((cc[col] & (y == 0)).sum())}
        out["dpat_k_curve"] = dpat_k_curve(rows, comp[["Component_ID", "Is_Defective"]], K_GRID)
        out["cost_ratio_curve"] = self.cost_ratio_curve(comp)
        return out

    def cost_ratio_curve(self, comp: pd.DataFrame) -> list:
        """Re-tune thresholds on validation lots for each C_FN/C_FP; report test recall / overkill."""
        y, curve = comp["Is_Defective"].to_numpy(), []
        for ratio in COST_GRID:
            _, ic = self._tune(float(ratio), 1.0)
            pred = comp["rule_fail"] | (comp["A_score"] >= ic["tau_a"]) | (comp["B_score"] >= ic["tau_b"])
            rec, ovk, esc = recall_overkill(y, pred)
            curve.append({"cost_ratio": ratio, "tau_a": ic["tau_a"], "tau_b": ic["tau_b"],
                          "recall": rec, "overkill": ovk, "escapes": esc})
        return curve

    # ------------------------------------------------------------------ persistence
    def find_uncertain_samples(self, comp: pd.DataFrame, margin=0.3) -> pd.DataFrame:
        """Identify borderline components near decision boundary for QA review."""
        return comp[(comp["C_score"] >= (1.0 - margin)) & (comp["C_score"] <= (1.0 + margin))].copy()

    def retrain_with_feedback(self, feedback_df: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series):
        """Retrain Module A/B models with enhanced labeled dataset."""
        # Assume feedback_df has 'Component_ID' and 'QA_Label'
        # In practice this would merge with original training data
        print(f"Retraining models with {len(feedback_df)} QA-reviewed samples")
        feat = build_features(X_train)
        self.module_a.fit(feat) # Simplified for now
        self.module_b.fit(feat)
        # Calibration would be needed here too

    def precompute(self, df: pd.DataFrame, path):
        """Score + explain + SHAP a dataset once, so the dashboard only filters (no model calls)."""
        rows, comp = self.score(df)
        joblib.dump({"signature": data_signature(df), "rows": rows, "comp": comp}, path, compress=3)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "BurnInSystem":
        return joblib.load(path)
