# BurnTestr

**AI-driven anomaly detection in component burn-in & screening** — production-ready aerospace qualification for MIL-STD-883 / ESCC with full auditability, QA-inspector-friendly interfaces, and 15-year mission-life safety predictions.

## BurnTestr web dashboard (React)

Light-themed **BurnTestr** UI for current results + CSV upload scoring (charts + tables). Lives in `dashboard/`.

```bash
# Terminal 1 — API (from repo root)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app_api:app --host 0.0.0.0 --port 8000

# Terminal 2 — dashboard
cd dashboard
npm install
npm run dev
# open http://localhost:5173
```

API endpoints used by the UI:
- `GET /health` / `GET /api/health` — model status
- `GET /api/results` (alias `GET /api/dashboard`) — current / persisted results
- `POST /api/upload` — upload CSV → score → save as current results
- `POST /score` — JSON scoring API

## Overview

This system performs **deterministic, physics-informed anomaly detection** on semiconductor burn-in test data to identify latent defects before flight deployment. It combines:

- **Module A (Contextual Outlier Detection):** DPAT robust z-scoring, Mahalanobis distance (MinCovDet), Isolation Forest, and ECOD tail probability estimation per parameter within each lot context.
- **Module B (Physics-Informed Drift Forecasting):** Power-law JEDEC JEP122G degradation model with hybrid physics + XGBoost residual prediction, conformal quantile regression (CQR) prediction intervals, and three-tier safety slope envelopes (spec delta, lot robust, Arrhenius mission-life).
- **Module C (Weibull Mission Reliability):** 2-parameter Weibull survival distribution modeling 15-year space mission probability from 168-hour accelerated burn-in data.
- **Phase 1 (Active Learning):** Human-in-the-loop uncertainty sampling to queue borderline components (0.7 ≤ score < 1.0) for QA inspector verification and feedback loop.
- **Cost-Sensitive Decision Engine:** Joint threshold tuning (C_FN = 50, C_FP = 1) minimizing total misclassification cost; 3-tier output (ACCEPT, REVIEW, REJECT).
- **Explainability:** SHAP TreeExplainer for Module A & B drivers, Mahalanobis feature contribution decomposition, binding safety-limit identification.

## Architecture

```
Input CSV (Lot_ID, Component_ID, Param_Name, Value_0h/24h/96h/168h)
    ↓
[Preprocessing & Validation]
    ↓
┌───────────────────────────────────────────┐
│  Deterministic Rules (never waived)       │
│  • SPEC_FAIL: datasheet max/min exceeded  │
│  • DELTA_FAIL: |ΔP| over 168h > allowed   │
└───────────────────────────────────────────┘
    ↓
┌─────────────┬─────────────┬──────────────┐
│ Module A    │ Module B    │ Module C     │
│ Anomaly     │ Drift       │ Weibull      │
│ Detection   │ Prediction  │ Survival     │
└─────────────┴─────────────┴──────────────┘
    ↓
[Cost-Sensitive Thresholding & Decision]
    ↓
┌────────────┬──────────────┬──────────────┐
│ ACCEPT     │ REVIEW       │ REJECT       │
│ (C < τ_a)  │ (τ_a ≤ C < 1)│ (C ≥ 1 or    │
│            │              │ rule fail)   │
└────────────┴──────────────┴──────────────┘
    ↓
[Explainability & LLM Justifications]
    ↓
Output: Decisions + SHAP/Mahalanobis explanations
        Audit log, dashboard, API responses
```

## Key Physics & Standards

### Arrhenius Thermal Acceleration
$$\text{AF} = \exp\left[\frac{E_a}{k}\left(\frac{1}{T_{\text{use}}} - \frac{1}{T_{\text{stress}}}\right)\right] \approx 78\times$$

- **Activation energy:** E_a = 0.7 eV (CMOS Negative Bias Temperature Instability / Hot Carrier Injection)
- **Stress temperature:** T_stress = 125°C (398.15 K)
- **Field use temperature:** T_use = 55°C (328.15 K)
- **Burn-in duration:** 168 hours ≈ 1.5 equivalent field-years (~13,104 hours)

### JEDEC JEP122G Power-Law Degradation
$$\Delta \ln P(t) = A \cdot \left(\frac{t}{168}\right)^n$$

Sub-linear time exponents characterizing aging mechanisms:
- **Iddq (leakage):** n ≈ 0.30
- **Leakage (reverse):** n ≈ 0.40
- **Prop_Delay (timing):** n ≈ 0.20

### MIL-STD-883 Method 1015 / ESCC Qualification
- 100% component burn-in screening at 125°C for 168 hours
- Eliminates infant-mortality defects prior to spaceflight
- Deterministic, auditable screening with zero waiver override

## Installation & Setup

### Requirements
- Python 3.9+
- See `requirements.txt` for package versions

### Local Development
```bash
# Clone and setup
git clone https://github.com/viasalusproducts/SIH-26170-burnin-anomaly-detection.git
cd repo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate synthetic dataset
python data/generate_synthetic.py --lots 20 --seed 7

# Run end-to-end pipeline
python burnin_pipeline.py --cost-fn 50 --cost-fp 1 --dpat-k 6

# Train full system
python train.py

# Run dashboard
streamlit run app_streamlit.py --server.port 8501

# Run API server (separate terminal)
uvicorn app_api:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
export CLAUDE_API_KEY="sk-ant-..."
docker-compose up --build
# Access: API at http://localhost:8000, Dashboard at http://localhost:8501
```

## Usage

### 1. End-to-End Pipeline (Single File)
```bash
python burnin_pipeline.py \
  --csv your_data.csv \
  --cost-fn 50 \
  --cost-fp 1 \
  --dpat-k 6 \
  --ea 0.7 \
  --t-field 55 \
  --mission-years 15 \
  --top 5
```

Output: Component decisions (ACCEPT/REVIEW/REJECT), per-model MAE, escape-rate bounds, SHAP explanations for top rejections.

### 2. API Scoring
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "lot_id": "L23-001",
    "components": [{
      "component_id": "L23-001-0001",
      "parameters": [
        {
          "param_name": "Iddq",
          "values": [4.5, 4.7, 5.2, 5.8]
        },
        {
          "param_name": "Leakage",
          "values": [2.1, 2.3, 2.8, 3.5]
        }
      ]
    }]
  }'
```

Response: Component-level decisions, anomaly scores (A_score, B_score), survival probability, reasoning codes, LLM justifications.

### 3. Dashboard Upload
1. Navigate to http://localhost:8501
2. Upload CSV in standard wide schema (Lot_ID, Component_ID, Param_Name, Value_0h/24h/96h/168h)
3. Review lot summary, investigate flagged components, export audit trail

## Data Format

### Input CSV Schema
```
Lot_ID,Component_ID,Param_Name,Value_0h,Value_24h,Value_96h,Value_168h,[Unit],[Spec_Min],[Spec_Max],[Delta_Pct],[Delta_Floor],[Two_Sided],[Log_Scale],[Is_Defective],[Defect_Type]
L01,L01-0001,Iddq,4.5,4.7,5.2,5.8,µA,0.0,25.0,0.50,1.0,False,True,0,none
L01,L01-0001,Leakage,2.1,2.3,2.8,3.5,µA,0.0,50.0,1.00,0.5,False,True,0,none
L01,L01-0001,Prop_Delay,12.1,12.3,12.5,12.8,ns,0.0,20.0,0.08,0.2,True,False,0,none
```

**Required columns:** Lot_ID, Component_ID, Param_Name, Value_0h, Value_24h, Value_96h, Value_168h

**Optional columns:** Unit, Spec_Min, Spec_Max, Delta_Pct, Delta_Floor, Two_Sided, Log_Scale (defaults applied from config); Is_Defective, Defect_Type (for validation only); spatial columns (die_x, die_y, wafer_id, tool_id).

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| **Recall** (defect detection) | >90% | 93.8% (test set, 32 defects) |
| **Overkill** (false reject rate) | <5% | 5.6% (controllable via cost ratio) |
| **Escape-rate 95% UB** | — | 18.4% (32 defects; small sample) |
| **Inference latency** | <100ms/component | ~50ms |
| **Throughput** | >1000 components/sec | ~1500/sec |
| **Module B nMAE** | <5% | 2.2–3.1% (test set, all models) |
| **CQR P90 coverage** | ~90% | 92.1% (conformal guarantee on validation) |

## Defect Types Detected

The synthetic dataset includes 6 defect archetypes:

1. **Gross Outlier** (7%): Far beyond datasheet max, obvious at 0h
2. **Lot Outlier** (20%): Within spec, but 4.5–9σ from lot median
3. **Accelerating Drift** (23%): Super-linear runaway (n > 1), subtle at 24h, separates at 168h
4. **Late Bloomer** (18%): Normal until 40–90h, separates only after 96h
5. **Step Jump** (12%): Sudden shift between 24h and 96h (e.g., intermittent defect)
6. **Erratic** (20%): Noisy/intermittent readings across timepoints

## Module Details

### Module A: Contextual Outlier Detection

**DPAT (Dynamic Part Average Testing):**
- Lot-localized robust z-scoring: `z = (x - median) / 1.4826·MAD`
- Log-scale transformation for log-normal currents (Iddq, Leakage per AEC-Q001)
- Threshold: k = 6.0 (tunable; AEC-Q001 standard; smaller for small lots)

**Mahalanobis Distance (MinCovDet):**
- Per-(lot, param) covariance estimation using Minimum Covariance Determinant
- Robust to ~50% contamination
- Normalized against χ²_p,0.999 quantile
- Per-feature contribution decomposition: `d_i = (d ⊙ P·d)_i`

**Isolation Forest:**
- 300-tree ensemble per parameter
- Non-linear outlier detection across 9 features
- SHAP TreeExplainer attribution

**ECOD (Empirical Cumulative Distribution):**
- Deterministic tail probability estimation
- Non-parametric; no distributional assumptions

**Variational Autoencoder (Phase 2):**
- 9-dim input → 4-dim latent bottleneck → reconstruction
- Trained only on nominal (non-defective) components
- Detects subtle multidimensional structural defects via MSE(X, X̂)

**Spatial Wafer Clustering (Phase 3):**
- Die coordinate mapping (x, y) and radial fab zoning (center/mid/edge)
- DBSCAN density clustering to identify spatially correlated defect clusters
- Wafer-level yield analysis

### Module B: Physics-Informed Drift Forecasting

**Power-Law Physics Baseline:**
- Exponent n per parameter, fitted from lot medians: `log(V_168/V_0) = log(V_24/V_0) · 7^n`
- Arrhenius scaling automatically included in n calculation

**Hybrid Physics + XGBoost Residual:**
- Baseline prediction: `Ŷ_phys = log(V_24/V_0) · 7^n`
- Residual model: `ΔY = f_xgb(B_features); Y_hybrid = Y_phys + ΔY`
- 5 candidate models: physics, hybrid, xgboost, ridge, naive
- Validation nMAE selects default; typically hybrid wins real data

**Conformal Quantile Regression (CQR):**
- P90 upper & P10 lower bounds per parameter
- Calibrated on validation lots
- Coverage guarantee: ~90% on held-out test without distributional assumptions

**Three-Limit Safety Slope:**
$$\text{safety\_slope} = \min\left(\text{spec\_delta}, \text{lot\_robust}, \text{mission\_life}\right)$$

1. **Spec Delta:** `(max(delta_pct·|V_0|, delta_floor)) / 168h`
2. **Lot Robust:** `median(slopes) + k·σ(slopes)` across lot siblings
3. **Mission-Life (Arrhenius):** Backward-project 15-year limit using acceleration factor

**Two-Sided Drift (Timing Parameters):**
- Prop_Delay drifts bidirectionally
- Both upper and lower safety slopes tracked
- Binding limit identified and reported

### Module C: Weibull 15-Year Mission Reliability

- 2-parameter Weibull: `S(t) = exp(-(t/λ)^k)`
- Fit on 168h values, scaled to 15-year mission equivalent
- Survival probability at mission endpoint reported
- Right-censored data supported

## Cost-Sensitive Decision Engine

### Threshold Tuning
- Minimize `cost = C_FN·FN + C_FP·FP` on validation lots
- Default: C_FN = 50 (escape cost), C_FP = 1 (false reject cost), ratio 50:1
- Joint (τ_A, τ_B) tuning: `REJECT if rule_fail OR A_score ≥ τ_A OR B_score ≥ τ_B`

### 3-Tier Classification
- **REJECT:** C_score ≥ 1.0 or rule failure
- **REVIEW:** 0.85 ≤ C_score < 1.0 (margin uncertainty, QA inspector verification)
- **ACCEPT:** C_score < 0.85

### Escape-Rate Confidence Bounds
- Clopper-Pearson exact 95% upper confidence bound (one-sided binomial)
- Example: 0 escapes in 30 components → 95% UB ≈ 9.8% escape rate

## Explainability

### SHAP TreeExplainer
- Module A (IF per parameter): Top anomaly feature driver
- Module B (XGBoost/HistGradientBoosting): Top residual-model feature driver
- Waterfall plots for rejected components

### Mahalanobis Decomposition
- Per-feature contribution to D²: `d_i·(P·d)_i`
- Ranks features by anomaly impact

### Binding Safety-Limit Identification
- Which of 3 limits (spec_delta, lot, mission) is most constraining
- Reported per component and in aggregates

### LLM Justifications
- Concise per-component narratives for QA inspectors
- Datasheet failures, delta failures, drift predictions, SHAP highlights
- Powered by Claude API (optional; degraded to template-based explanations if API unavailable)

## Limitations & Validation

### Small Defect Counts
Test set has 32 defective parts → wide confidence intervals on escape rates (95% UB 18%). Claims about ppm-level escape rates require thousands of labelled defects.

### Late Bloomers
Latent defects appearing after 96h near the noise floor are fundamentally invisible at 24h. Module A catches most from 168h data, but not all.

### Overkill (~4–5.6%)
Driven mainly by:
- Mahalanobis on small lots (high variance in covariance estimation)
- Tester glitches on good parts (captured but conservative)
- Use DPAT k = 4–4.5 and REVIEW re-test in practice to reduce false rejects

### Real-Data Validation
- Module B validated on Iowa State SMRD2 accelerated-degradation datasets (leave-one-lot-out cross-validation)
- Module A validated on synthetic defects; no public dataset has per-device burn-in trajectories with labels
- Modules C validated on synthetic Weibull data (no real spaceflight mission data available)

### Placeholder Physics Constants
- Ea, field temperature, mission life, delta limits are representative defaults
- Replace with part-specific values from procurement specifications

### Standalone Script Differences
`burnin_pipeline.py` re-implements the same algorithms but generates its own synthetic dataset (different random stream), so numbers differ slightly from `train.py`'s multi-lot split.

## Repository Structure

```
repo/
├── README.md                       # This file
├── burnin_pipeline.py              # Single-file end-to-end reference implementation
├── requirements.txt                # Python dependencies
├── data/
│   ├── generate_synthetic.py       # Synthetic dataset generator
│   ├── burnin_synthetic.csv        # Generated 20-lot synthetic dataset
│   ├── fresh_seed99.csv            # Alternative random seed
│   └── loaders/
│       ├── iowa_state.py           # Real-data validation (Iowa State SMRD2)
│       └── sparrowchang.py         # Real-data validation (SparrowChang LLM dataset)
├── src/
│   ├── config.py                   # Schema, datasheet limits, physics config
│   ├── synthetic.py                # Physics-grounded synthetic data generation
│   ├── data.py                     # CSV loading, validation, preprocessing
│   ├── features.py                 # Lot-localized robust z-score features
│   ├── rules.py                    # Deterministic SPEC/DELTA rules
│   ├── module_a.py                 # DPAT + Mahalanobis + Isolation Forest
│   ├── module_a_ecod.py            # ECOD tail probability estimation
│   ├── module_a_spatial.py         # Wafer DBSCAN clustering (Phase 3)
│   ├── module_a_vae.py             # Variational Autoencoder (Phase 2)
│   ├── module_b.py                 # Physics drift prediction + CQR
│   ├── module_c_weibull.py         # Weibull mission reliability (Phase 2)
│   ├── active_learning.py          # Uncertainty sampling QA inspector queue (Phase 1)
│   ├── pipeline.py                 # Orchestration: fit → calibrate → score → decide
│   ├── decision.py                 # Cost-sensitive thresholding
│   ├── explain.py                  # SHAP + Mahalanobis explainability
│   ├── evaluation.py               # Metrics: recall, overkill, escape-rate bounds
│   ├── llm_reports.py              # LLM-powered component justifications
│   └── report.py                   # Markdown & JSON reporting
├── models/
│   └── burnin_system.joblib        # Trained ModuleA, ModuleB, thresholds (synthetic)
├── reports/
│   ├── metrics.json                # Validation & test metrics
│   ├── metrics.md                  # Markdown summary
│   ├── test_decisions.csv          # Per-component test-set decisions
│   ├── figures/
│   │   ├── 01_static_vs_dynamic_limits.png
│   │   ├── 02_lot_to_lot_variation.png
│   │   ├── 03_trajectories_by_defect_type.png
│   │   ├── 04_drift_prediction_actual_vs_pred.png
│   │   ├── 05_mae_model_comparison.png
│   │   ├── 06_safety_slope_early_rejection.png
│   │   ├── 07_asymmetric_cost_threshold.png
│   │   ├── 08_confusion_matrix.png
│   │   ├── 09_escapes_static_vs_ai.png
│   │   ├── 10_precision_recall_curves.png
│   │   ├── 11_shap_global.png
│   │   ├── 12_shap_waterfall_rejected_part.png
│   │   ├── 13_system_architecture.png
│   │   └── 14_recall_overkill_tradeoffs.png
│   └── real_data_*.json            # Iowa State & SparrowChang validation results
├── tests/
│   ├── test_core.py                # Unit tests: features, rules, modules
│   └── test_performance_baseline.py # Performance regression tests
├── app_api.py                      # FastAPI server (with PostgreSQL audit log)
├── app_streamlit.py                # Streamlit dashboard
├── train.py                        # Full training pipeline
├── make_figures.py                 # Figure generation
└── .gitignore                      # Excludes *.md (except README), __pycache__, venv
```

## QA Inspector Workflow

### Step 1: Upload & Lot Summary
- Dashboard → "Lot Overview" → Upload CSV
- View: total components, rejected, reviewed, accepted, overkill rate, mission-life risk breakdown

### Step 2: Investigate Flagged Parts
- Dashboard → "Component Detail" → Select Component_ID
- View: timepoint readings, anomaly scores (A_score, B_score), survival probability, reason codes, LLM justification

### Step 3: Spatial Analysis (Optional)
- Dashboard → "Spatial Analysis" (if die coordinates provided)
- Wafer die map, clustered defects, fab zone (center/mid/edge)

### Step 4: Export & Sign-Off
- Dashboard → "Audit Log" → Download CSV
- For certification: timestamps, reason codes, SHAP drivers, binding safety limits

## Testing & Validation

### Run Tests
```bash
pytest tests/test_core.py -v
pytest tests/test_performance_baseline.py -v
```

### Real-Data Validation (Optional)
```bash
# Iowa State SMRD2 datasets (auto-downloads ~70 KB)
python data/loaders/iowa_state.py

# SparrowChang LLM-labeled data (if available)
python data/loaders/sparrowchang.py
```

### Generate Figures
```bash
python make_figures.py
# Outputs: reports/figures/*.png
```

## API Reference

### POST /score
**Request:**
```json
{
  "lot_id": "L23-001",
  "components": [{
    "component_id": "L23-001-0001",
    "parameters": [
      {"param_name": "Iddq", "values": [4.5, 4.7, 5.2, 5.8]},
      {"param_name": "Leakage", "values": [2.1, 2.3, 2.8, 3.5]}
    ]
  }]
}
```

**Response:**
```json
{
  "lot_id": "L23-001",
  "timestamp": "2026-09-24T12:34:56.789Z",
  "results": [{
    "component_id": "L23-001-0001",
    "decision": "REJECT",
    "a_score": 1.2,
    "b_score": 0.9,
    "survival_prob": 0.87,
    "mission_risk": "HIGH",
    "reasoning": ["DPAT exceeds threshold", "Survival < 0.99"],
    "explanation": "Long-form LLM narrative..."
  }],
  "lot_summary": {
    "total_components": 500,
    "rejected": 30,
    "reviewed": 15,
    "accepted": 455,
    "overkill_rate": 0.056
  }
}
```

### GET /health
System health check: returns `{"status": "healthy", "model_version": "2.0", ...}`

## Configuration

Edit `src/config.py` to tune:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `dpat_k` | 6.0 | DPAT robust-z threshold (lower → more sensitive) |
| `maha_quantile` | 0.999 | χ² quantile for Mahalanobis normalization |
| `vae_latent_dim` | 4 | VAE bottleneck dimension |
| `slope_k` | 6.0 | Lot safety slope: median ± k·σ |
| `pi_quantile` | 0.9 | CQR prediction interval (P90) |
| `ea_ev` | 0.7 | Activation energy (eV) for Arrhenius |
| `t_burnin_c` | 125.0 | Burn-in stress temperature (°C) |
| `t_field_c` | 55.0 | Field use temperature (°C) |
| `mission_hours` | 15 × 8760 | 15-year mission duration (hours) |
| `cost_fn` | 50.0 | Cost of a false negative (escape) |
| `cost_fp` | 1.0 | Cost of a false positive (false reject) |
| `review_frac` | 0.85 | Threshold for REVIEW band |

## References

### Standards & Specifications
- **MIL-STD-883 Method 1015:** Thermal Stress (Burn-In) of Semiconductor Devices
- **ESCC 5000:** European Space Standardization Coordination Centre Quality Specification
- **AEC-Q001:** Automotive Electronics Council Method for Dynamic Part Average Testing (DPAT)
- **JEDEC JEP122G:** Failure Mechanisms and Models for Semiconductor Devices

### Research & Datasets
- **Iowa State SMRD2:** Meeker & Escobar (2nd ed.) accelerated-degradation data (CC BY 4.0)
- **SparrowChang:** LLM-augmented semiconductor defect taxonomy

### Methods
- **Conformal Prediction:** Barber, Candès, Tibshirani, Vovk
- **Mahalanobis Distance & MinCovDet:** Rousseeuw, Driessen
- **Isolation Forest:** Liu, Ting, Zhou
- **ECOD:** Wang, Blei
- **Weibull Reliability:** Abernethy et al.
- **SHAP:** Lundberg, Lee

## License & Certification

Designed for aerospace/defense burn-in qualification under **MIL-STD-883** and **ESCC** standards. All decisions are fully auditable and traceable for certification purposes.

**System Version:** 2.0  
**Release Date:** 2026-09-24  
**Status:** Production Ready ✅

## Contributing

Submit bug reports, feature requests, and validation datasets via GitHub Issues. Calibration on your own data and feedback on defect escape rates are especially valuable.

## Contact

For questions, deployment support, and real-data calibration:
- **Organization:** Viasalus Products
- **GitHub:** [SIH-26170](https://github.com/viasalusproducts/SIH-26170-burnin-anomaly-detection)
