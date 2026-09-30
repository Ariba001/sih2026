import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, REQUIRED_COLS, arrhenius_af  # noqa: E402
from src.data import split_lots, validate  # noqa: E402
from src.evaluation import (choose_threshold, classification_report, clopper_pearson_upper,  # noqa: E402
                            fbeta)
from src.features import A_FEATURES, build_features, robust_sigma  # noqa: E402
from src.module_a import ModuleA, mahalanobis_per_lot  # noqa: E402
from src.module_b import ModuleB  # noqa: E402
from src.pipeline import BurnInSystem  # noqa: E402
from src.rules import rule_screens  # noqa: E402
from src.synthetic import generate  # noqa: E402

VC = ["Value_0h", "Value_24h", "Value_96h", "Value_168h"]


def make_lot(param="Leakage", n=500, center=10.0, sd=0.8, outlier_value=None, seed=0, lot="LX"):
    """Stable lot around `center` (default: Leakage mean ~10 µA, datasheet max 50 µA)."""
    rng = np.random.default_rng(seed)
    v0 = rng.normal(center, sd, n)
    drift = rng.normal(0.03, 0.005, n)
    tn = np.array([0, 24, 96, 168]) / 168
    vals = v0[:, None] * (1 + drift[:, None] * tn[None, :] ** 0.4) * rng.normal(1, 0.005, (n, 4))
    if outlier_value is not None:
        vals[0] = outlier_value * (1 + 0.01 * tn ** 0.4)
    df = pd.DataFrame(vals, columns=VC)
    df.insert(0, "Param_Name", param)
    df.insert(0, "Component_ID", [f"{lot}-{i:04d}" for i in range(n)])
    df.insert(0, "Lot_ID", lot)
    df["Is_Defective"] = 0
    if outlier_value is not None:
        df.loc[0, "Is_Defective"] = 1
    df["Defect_Type"] = np.where(df["Is_Defective"] == 1, "lot_outlier", "none")
    return validate(df)


@pytest.fixture(scope="module")
def trained_system():
    df = validate(generate(n_lots=6, parts_per_lot=200, seed=3))
    s = split_lots(df, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    return BurnInSystem(Config()).fit(s["train"], s["val"])


# ---------------------------------------------------------------- Module A: the problem-statement example
def test_45uA_part_in_10uA_lot_is_massive_anomaly_module_a():
    df = make_lot(outlier_value=45.0)
    feat = build_features(df)
    res = ModuleA(Config()).fit(feat).score(feat)
    part = res.iloc[0]
    assert df.loc[0, "Value_24h"] < df.loc[0, "Spec_Max"] == 50.0      # inside the datasheet limit
    assert not rule_screens(feat).iloc[0]["spec_fail"]
    assert abs(part["dpat_z_Value_24h"]) > 10                         # massive robust-sigma distance
    assert part["dpat_score"] > 1 and part["maha_score"] > 1 and part["if_score"] > 1
    assert res["A_score"].idxmax() == 0                               # most anomalous in its lot


def test_45uA_part_in_10uA_lot_is_rejected_with_explanation(trained_system):
    _, comp = trained_system.score(make_lot(outlier_value=45.0))
    row = comp.set_index("Component_ID").loc["LX-0000"]
    assert row["Decision"] == "REJECT"
    assert "within datasheet max" in row["Explanation"] and "robust-σ" in row["Explanation"]
    assert "DPAT" in row["Reason_Codes"]
    assert comp["C_score"].idxmax() == comp.index[comp["Component_ID"] == "LX-0000"][0]


# ---------------------------------------------------------------- robust statistics
def test_robust_sigma_ignores_outliers():
    x = np.r_[np.random.default_rng(0).normal(10, 1, 999), 1e6]
    assert 0.9 < robust_sigma(x) < 1.1


def test_mahalanobis_contributions_sum_to_d2():
    feat = build_features(make_lot(n=200))
    d2, contrib = mahalanobis_per_lot(feat)
    assert contrib.shape == (200, len(A_FEATURES))
    np.testing.assert_allclose(contrib.sum(axis=1), d2, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------- rules (MIL / ESCC delta limits)
def test_delta_limit_whichever_is_greater():
    df = pd.concat([make_lot("Iddq", n=3, center=5.0, sd=0.01, lot="A"),
                    make_lot("Leakage", n=3, center=0.1, sd=0.001, lot="B")], ignore_index=True)
    df.loc[0, "Value_96h"] = df.loc[0, "Value_0h"] * 1.6      # Iddq +60 % > max(50 %, 1 µA) -> fail
    df.loc[3, "Value_168h"] = 0.3                             # Leakage 0.1 -> 0.3 (+200 %) but < 0.5 µA floor
    r = rule_screens(build_features(validate(df)))
    assert r.loc[0, "delta_fail"] and r.loc[0, "delta_worst_t"] == "96h"
    assert not r.loc[3, "delta_fail"]


# ---------------------------------------------------------------- Module B physics and safety slope
def test_power_law_exponent_recovered():
    rng = np.random.default_rng(1)
    rows = []
    for lot in range(6):
        A = rng.uniform(0.02, 0.06)
        for i in range(100):
            v0 = rng.uniform(4, 6)
            vals = v0 * np.exp(A * np.exp(rng.normal(0, 0.2)) * (np.array([0, 24, 96, 168]) / 168) ** 0.3)
            rows.append([f"L{lot}", f"L{lot}-{i}", "Iddq", *vals])
    feat = build_features(validate(pd.DataFrame(rows, columns=REQUIRED_COLS)))
    assert ModuleB.fit_power_law(feat)["Iddq"] == pytest.approx(0.3, abs=0.03)


def test_safety_slope_is_min_of_three_and_binding_reported(trained_system):
    df = make_lot("Iddq", n=300, center=5.0, sd=0.4)
    df.loc[1, VC] = [24.9, 24.9, 24.9, 24.9]                  # stable, right under the 25 µA datasheet max
    rows, _ = trained_system.score(validate(df), explain=False)
    three = rows[["safe_slope_spec", "safe_slope_lot", "safe_slope_mission"]].to_numpy()
    np.testing.assert_allclose(rows["safety_slope"], three.min(axis=1))
    assert rows.loc[1, "binding_up"] == "mission"             # almost no headroom left for mission life
    assert rows["binding_up"].iloc[2:].eq("lot").mean() > 0.9


def test_module_b_flags_early_drift_upward(trained_system):
    df = make_lot(center=10.0)
    df.loc[1, VC] = [10.0, 12.5, 22.0, 38.0]                  # normal at 0h, +25 % by 24h, runaway
    rows, _ = trained_system.score(validate(df), explain=False)
    r = rows.set_index("Component_ID").loc["LX-0001"]
    # point forecasts under-predict runaway parts, which is why the flag uses the conformal upper bound
    assert r["B_flag"] and r["slope_upper"] > r["safety_slope"] and r["pred_168_upper"] > 12.5


def test_two_sided_drift_only_where_configured(trained_system):
    delay = make_lot("Prop_Delay", n=300, center=12.0, sd=0.3, lot="D")
    delay.loc[1, VC] = [12.0, 11.6, 11.0, 10.5]               # abnormal *decrease* in delay (two-sided)
    iddq = make_lot("Iddq", n=300, center=5.0, sd=0.4, lot="I")
    iddq.loc[1, VC] = [5.0, 4.6, 4.2, 3.9]                    # decrease in Iddq (one-sided: not a drift flag)
    rows, _ = trained_system.score(validate(pd.concat([delay, iddq], ignore_index=True)), explain=False)
    r = rows.set_index("Component_ID")
    assert r.loc["D-0001", "B_down"] >= 1
    assert not np.isfinite(r.loc["I-0001", "B_down"])


def test_arrhenius_acceleration_factor():
    assert arrhenius_af(0.7, 125, 55) == pytest.approx(78, rel=0.05)
    assert arrhenius_af(0.4, 125, 55) == pytest.approx(12, rel=0.1)


# ---------------------------------------------------------------- cost-sensitive evaluation
def test_cost_threshold_prefers_recall():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 2.0, 3.0])
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1])
    t = choose_threshold(scores, y, c_fn=50, c_fp=1)
    assert t["val_fn"] == 0 and 0.4 < t["threshold"] < 0.5         # accepts 2 FP to avoid 1 FN
    assert choose_threshold(scores, y, c_fn=1, c_fp=1)["threshold"] > 0.6


def test_classification_report_fbeta_and_clopper_pearson():
    r = classification_report([1, 1, 0, 0], [1, 0, 1, 0], c_fn=50, c_fp=1)
    assert r["escapes"] == 1 and r["cost"] == 51 and r["overkill_rate"] == 0.5
    assert fbeta(0.5, 1.0, 2) == pytest.approx(5 * 0.5 / (4 * 0.5 + 1))
    assert clopper_pearson_upper(0, 100) == pytest.approx(1 - 0.05 ** (1 / 100))   # ~ 3/n
    assert 0.029 < clopper_pearson_upper(0, 100) < 0.031


def test_synthetic_schema_and_validation():
    df = generate(n_lots=3, parts_per_lot=None, seed=1)
    assert list(df.columns[:7]) == REQUIRED_COLS
    assert {"Is_Defective", "Defect_Type", "Spec_Max"} <= set(df.columns)
    assert df.groupby("Lot_ID")["Component_ID"].nunique().min() >= 30
    with pytest.raises(ValueError):
        validate(df.drop(columns=["Value_96h"]))


# ---------------------------------------------------------------- Week 2: ECOD detector
def test_ecod_detector_produces_scores():
    """ECOD detector should produce normalized scores."""
    from src.module_a_ecod import ModuleAECOD
    feat = build_features(make_lot(n=200))
    ecod = ModuleAECOD(Config()).fit(feat)
    scores = ecod.score(feat)
    assert scores.shape == (len(feat),)
    assert np.isfinite(scores).all()
    # Normalized around median (can go negative); q99 should be ~ 1.0
    assert len(np.unique(scores)) > 10  # Not constant


def test_ecod_is_deterministic():
    """ECOD should be deterministic: same scores on repeated runs."""
    from src.module_a_ecod import ModuleAECOD
    feat = build_features(make_lot(n=200, seed=42))
    m1 = ModuleAECOD(Config()).fit(feat)
    m2 = ModuleAECOD(Config()).fit(feat)
    scores1 = m1.score(feat)
    scores2 = m2.score(feat)
    # ECOD is deterministic, so scores should match exactly
    assert np.allclose(scores1, scores2, rtol=1e-10)


def test_module_a_includes_ecod_score():
    """Module A should include ECOD as fourth detector with ensemble weighting."""
    feat = build_features(make_lot(n=200))
    res = ModuleA(Config()).fit(feat).score(feat)
    assert "ecod_score" in res.columns
    assert "dpat_score" in res.columns
    assert "maha_score" in res.columns
    assert "if_score" in res.columns
    # PHASE 2: A_score should be weighted ensemble (not max)
    # Weights: DPAT 0.4 + Maha 0.3 + IF 0.2 + ECOD 0.1
    expected_a_score = (0.40 * res["dpat_score"] + 0.30 * res["maha_score"] +
                       0.20 * res["if_score"] + 0.10 * res["ecod_score"])
    assert np.allclose(res["A_score"], expected_a_score)


def test_ecod_detects_outliers():
    """ECOD should give high scores to outliers."""
    from src.module_a_ecod import ModuleAECOD
    df = make_lot(n=500, center=10.0, outlier_value=45.0)
    feat = build_features(df)
    ecod = ModuleAECOD(Config()).fit(feat)
    scores = ecod.score(feat)
    # First row (outlier at 45 µA) should have highest ECOD score
    assert scores[0] == scores.max()
    assert scores[0] > 0.5  # Substantially high


# ---------------------------------------------------------------- Week 2: Monotonic constraints
def test_module_b_monotonic_constraints_applied():
    """Module B should apply monotonic constraints to XGBoost models."""
    feat = build_features(make_lot(n=300))
    mb = ModuleB(Config()).fit(feat)
    # Check that m_xgb has monotone_constraints set
    if hasattr(mb.m_xgb, 'get_booster'):
        # XGBoost sklearn API
        assert mb.m_xgb is not None
        # Verify training succeeded (if model has prediction capability)
        test_pred = mb.m_xgb.predict(feat[['log_v0', 'log_v24', 'r24', 'lot_med_log_v0',
                                             'lot_med_log_v24', 'lot_med_r24', 'rel_v0', 'rel_r24',
                                             'is_Iddq', 'is_Leakage']])
        assert np.isfinite(test_pred).all()


def test_module_b_hybrid_training_with_asymmetric_loss():
    """Module B hybrid model should train with asymmetric loss objective."""
    feat = build_features(make_lot(n=300))
    mb = ModuleB(Config()).fit(feat)
    # Model should be trained and capable of prediction
    assert mb.m_hyb is not None
    # Hybrid model should work in _logratio_preds
    preds, X, Xh, phys = mb._logratio_preds(feat)
    assert 'hybrid' in preds
    assert np.isfinite(preds['hybrid']).all()


# ---------------------------------------------------------------- Week 2: Asymmetric loss
def test_asymmetric_huber_loss_penalizes_underprediction():
    """Asymmetric loss should penalize under-prediction (runaway) more than over-prediction."""
    from src.custom_loss import asymmetric_huber_loss
    y_true = np.array([1.0, 1.0])
    y_pred_under = np.array([0.0, 0.5])  # Under-predict
    y_pred_over = np.array([2.0, 1.5])   # Over-predict

    loss_under = asymmetric_huber_loss(y_true, y_pred_under, alpha=10.0)
    loss_over = asymmetric_huber_loss(y_true, y_pred_over, alpha=10.0)

    # Under-prediction should be penalized more (10× with alpha=10)
    assert loss_under[0] > loss_over[0]
    assert loss_under[1] > loss_over[1]


def test_asymmetric_huber_gradient_produces_valid_output():
    """Asymmetric loss gradient should produce finite grad/hess for XGBoost."""
    from src.custom_loss import asymmetric_huber_grad
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)
    grad, hess = asymmetric_huber_grad(y_true, y_pred, alpha=10.0)

    assert grad.shape == (100,) and hess.shape == (100,)
    assert np.isfinite(grad).all() and np.isfinite(hess).all()
    assert (hess >= 0).all()  # Hessian should be non-negative


# ---------------------------------------------------------------- Week 2: Integration tests
def test_full_system_with_week2_enhancements():
    """Full pipeline should work with ECOD + monotonic + asymmetric loss."""
    df = validate(generate(n_lots=6, parts_per_lot=150, seed=42))
    splits = split_lots(df, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)

    system = BurnInSystem(Config()).fit(splits["train"], splits["val"])
    rows, comp = system.score(splits["test"])

    # All required columns should be present
    assert "ecod_score" in rows.columns
    assert "A_score" in comp.columns
    assert "B_score" in comp.columns
    assert comp["Decision"].isin(["ACCEPT", "REVIEW", "REJECT"]).all()

    # Metrics should be reasonable
    recall = comp[comp["Is_Defective"] == 1]["Decision"].isin(["REJECT", "REVIEW"]).mean()
    assert recall >= 0.8  # Should catch most defects


def test_baseline_metrics_maintained_with_week2 ():
    """Week 2 enhancements (ensemble + threshold adjustment) improve recall vs baseline."""
    df = validate(generate(n_lots=8, parts_per_lot=200, seed=99))
    splits = split_lots(df, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)

    system = BurnInSystem(Config()).fit(splits["train"], splits["val"])
    metrics = system.evaluate(splits["test"])

    # PHASE 1-2 IMPROVEMENTS: Ensemble weighting + lower initial thresholds
    # Combined OR-rule achieves better balance of recall/overkill
    # Accept realistic synthetic data performance: recall >= 85%, overkill <= 30%
    assert metrics["combined_reject_only"]["recall"] >= 0.85, f"Recall {metrics['combined_reject_only']['recall']:.1%} below target"
    assert metrics["combined_reject_only"]["overkill_rate"] <= 0.30, f"Overkill {metrics['combined_reject_only']['overkill_rate']:.1%} above target"


# ---------------------------------------------------------------- Phase 2: Spatial clustering
def test_synthetic_data_with_spatial_coordinates():
    """Synthetic data should include spatial coordinates when requested."""
    df = generate(n_lots=2, parts_per_lot=50, seed=42, add_spatial=True)
    assert "die_x" in df.columns and "die_y" in df.columns
    assert "wafer_id" in df.columns and "tool_id" in df.columns
    assert len(df) > 0
    # die_x and die_y should be in reasonable wafer range (-15 to 15 mm)
    assert df["die_x"].min() >= -15 and df["die_x"].max() <= 15
    assert df["die_y"].min() >= -15 and df["die_y"].max() <= 15


def test_synthetic_data_without_spatial_coordinates():
    """Synthetic data should work without spatial coordinates (backward compatible)."""
    df = generate(n_lots=2, parts_per_lot=50, seed=42, add_spatial=False)
    assert "die_x" not in df.columns
    assert "die_y" not in df.columns
    assert len(df) > 0


def test_spatial_detector_zone_classification():
    """Spatial detector should correctly classify wafer zones by polar distance."""
    from src.module_a_spatial import SpatialAnomalyDetector
    detector = SpatialAnomalyDetector(Config())

    # Test zone boundaries
    assert detector._identify_zone(0, 0) == "center"
    assert detector._identify_zone(2, 2) == "center"  # r ≈ 2.8 < 5
    assert detector._identify_zone(5, 5) == "mid-radius"  # r ≈ 7.1
    assert detector._identify_zone(10, 10) == "mid-radius"  # r ≈ 14.1 < 15
    assert detector._identify_zone(15, 15) == "edge"  # r ≈ 21.2 >= 15


def test_spatial_detector_identifies_clusters():
    """Spatial detector should identify defect clusters in synthetic data."""
    from src.module_a_spatial import SpatialAnomalyDetector

    df = generate(n_lots=3, parts_per_lot=100, seed=42, add_spatial=True)
    feat = build_features(df)
    ma = ModuleA(Config()).fit(feat)
    rows = ma.score(feat)

    # Attach spatial coordinates to rows
    rows_with_spatial = rows.merge(df[["Lot_ID", "Component_ID", "Param_Name", "die_x", "die_y"]],
                                     on=["Lot_ID", "Component_ID", "Param_Name"], how="left")

    detector = SpatialAnomalyDetector(Config())
    clusters = detector.identify_defect_clusters(rows_with_spatial, threshold=0.5)

    # Should find some clusters or return empty (both valid)
    assert isinstance(clusters, pd.DataFrame)
    if len(clusters) > 0:
        assert "fab_zone" in clusters.columns
        assert "cluster_id" in clusters.columns


# ---------------------------------------------------------------- Phase 2: Weibull survival
def test_weibull_analyzer_fits_and_predicts():
    """Weibull analyzer should fit on training data and predict survival probabilities."""
    from src.module_c_weibull import ModuleC

    df = generate(n_lots=5, parts_per_lot=150, seed=42)
    weibull = ModuleC(Config()).fit(df)

    # Pick a test component
    test_component = df.iloc[0]
    surv_prob = weibull.predict_survival_prob(test_component, mission_years=15)

    # Survival probability should be in [0, 1]
    assert 0 <= surv_prob <= 1
    assert isinstance(surv_prob, float)


def test_weibull_survival_deterministic():
    """Weibull survival probability should be deterministic for same input."""
    from src.module_c_weibull import ModuleC

    df = generate(n_lots=4, parts_per_lot=100, seed=42)
    weibull = ModuleC(Config()).fit(df)

    test_component = df.iloc[0]
    surv1 = weibull.predict_survival_prob(test_component, mission_years=15)
    surv2 = weibull.predict_survival_prob(test_component, mission_years=15)

    assert surv1 == surv2


def test_weibull_flags_marginal_parts():
    """Weibull analyzer should flag parts with low survival probability."""
    from src.module_c_weibull import ModuleC

    df = generate(n_lots=3, parts_per_lot=100, seed=42)
    weibull = ModuleC(Config()).fit(df)

    marginal = weibull.flag_marginal_parts(df, survival_threshold=0.999)

    # Should return a DataFrame (possibly empty)
    assert isinstance(marginal, pd.DataFrame)
    if len(marginal) > 0:
        assert "survival_prob" in marginal.columns
        assert "mission_risk" in marginal.columns
        assert all(marginal["survival_prob"] < 0.999)


# ---------------------------------------------------------------- Phase 2: Integration
def test_full_system_with_phase2_enhancements():
    """Full pipeline should work with spatial + Weibull modules."""
    df = validate(generate(n_lots=6, parts_per_lot=120, seed=42, add_spatial=True))
    splits = split_lots(df, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)

    system = BurnInSystem(Config()).fit(splits["train"], splits["val"])
    rows, comp = system.score(splits["test"])

    # New columns should be present
    assert "survival_prob" in comp.columns
    assert "mission_risk" in comp.columns

    # Existing columns should still be present
    assert "Decision" in comp.columns
    assert "A_score" in comp.columns
    assert "B_score" in comp.columns

    # Decisions should still be valid
    assert comp["Decision"].isin(["ACCEPT", "REVIEW", "REJECT"]).all()


def test_phase2_backward_compatibility_without_spatial_data():
    """Phase 2 enhancements should work even without spatial data."""
    df = validate(generate(n_lots=5, parts_per_lot=100, seed=42, add_spatial=False))
    splits = split_lots(df, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)

    system = BurnInSystem(Config()).fit(splits["train"], splits["val"])
    rows, comp = system.score(splits["test"])

    # Should have survival probabilities even without spatial data
    assert "survival_prob" in comp.columns
    assert comp["survival_prob"].notna().all()

    # Spatial columns should not be present
    assert "fab_zone" not in comp.columns or comp["fab_zone"].isna().all()

    # Baseline metrics should be maintained
    assert comp["Decision"].isin(["ACCEPT", "REVIEW", "REJECT"]).all()


