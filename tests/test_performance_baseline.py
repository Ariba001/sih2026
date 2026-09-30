import pytest
import pandas as pd
import joblib
from src.pipeline import BurnInSystem
from pathlib import Path

# Baseline metrics from reports/metrics.md
BASELINE_RECALL = 0.938
BASELINE_FP = 0.0560

def test_baseline_preservation():
    # Load model from pre-trained model file
    model_path = Path("models/burnin_system.joblib")
    assert model_path.exists(), "Model file not found"
    system = BurnInSystem.load(str(model_path))

    # Load test data
    test_data = pd.read_csv("data/burnin_synthetic.csv")

    # Evaluate
    metrics = system.evaluate(test_data)

    # Assert metrics meet baseline
    recall = metrics["module_a"]["recall"]
    # Adjusting for baseline provided in report
    assert recall >= BASELINE_RECALL, f"Recall {recall:.3f} below baseline {BASELINE_RECALL}"

    # FP check
    fp_rate = metrics["combined_reject_only"]["overkill_rate"]
    # Based on metrics.md reported overkill 5.60%
    assert fp_rate <= BASELINE_FP + 0.01, f"FP rate {fp_rate:.3%} too high"
