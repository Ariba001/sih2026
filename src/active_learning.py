"""Phase 1: Human-in-the-Loop Active Learning - Uncertainty Sampling.

Strategy: Identify "gray area" components where the model is least confident
(0.7 < C_score < 0.9, i.e., 0.7–0.9× the decision threshold). Present these
to a QA inspector for labeling. Feed the labels back into the training set.

Expected impact:
- Increases Recall: QA inspectors catch subtle defects the model misses
- Decreases False Positives: Labeling "safe-but-weird" components as Good
  teaches the model these are normal, reducing overkill
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class UncertaintySampler:
    """Extract and manage high-uncertainty samples for QA review."""

    def __init__(self, uncertainty_band: Tuple[float, float] = (0.7, 0.9)):
        """
        Args:
            uncertainty_band: (lower, upper) fraction of decision threshold.
                E.g., (0.7, 0.9) means 0.7 ≤ C_score < 0.9 (if threshold is 1.0).
        """
        self.uncertainty_band = uncertainty_band
        self.reviewed_samples: List[Dict] = []

    def find_uncertain_components(
        self, comp: pd.DataFrame, threshold: float = 1.0
    ) -> pd.DataFrame:
        """Identify borderline components in the "gray zone" near the decision boundary.

        Args:
            comp: Component-level scores (from pipeline.score()).
            threshold: Decision threshold (typically 1.0 for C_score).

        Returns:
            DataFrame of uncertain components, sorted by C_score (closest to boundary first).
        """
        lower = threshold * self.uncertainty_band[0]
        upper = threshold * self.uncertainty_band[1]

        # REVIEW-level components also included (C_score >= 0.85 but < threshold)
        uncertain = comp[
            (comp["C_score"] >= lower) & (comp["C_score"] < upper)
        ].copy()

        # Sort by distance to decision boundary (threshold), then by distance to REVIEW boundary (0.85)
        uncertain["_dist_to_threshold"] = (threshold - uncertain["C_score"]).abs()
        uncertain = uncertain.sort_values("_dist_to_threshold").drop(
            columns=["_dist_to_threshold"]
        )

        return uncertain

    def format_for_review(
        self, comp: pd.DataFrame, rows: pd.DataFrame, max_samples: int = 50
    ) -> pd.DataFrame:
        """Format uncertain components for QA inspector review.

        Includes key diagnostics: scores, primary parameter, top anomaly features.

        Args:
            comp: Component-level decisions (from pipeline.score()).
            rows: Row-level scores (from pipeline.score_rows()).
            max_samples: Max number of uncertain components to return.

        Returns:
            DataFrame ready for QA labeling interface.
        """
        uncertain = self.find_uncertain_components(comp)[:max_samples].copy()

        if len(uncertain) == 0:
            return pd.DataFrame()

        # Merge in row-level details (scores per parameter)
        row_details = rows[
            rows["Component_ID"].isin(uncertain["Component_ID"])
        ].copy()
        row_summary = (
            row_details.groupby("Component_ID")
            .agg(
                {
                    "A_score": "max",
                    "B_score": "max",
                    "dpat_maxz": "max",
                    "maha_score": "max",
                    "if_score": "max",
                    "vae_score": lambda x: x.max() if "vae_score" in row_details.columns else 0.0,
                    "spatial_score": lambda x: x.max() if "spatial_score" in row_details.columns else 0.0,
                    "Param_Name": lambda x: x.iloc[0],  # Primary param (top score)
                }
            )
            .reset_index()
            .rename(
                columns={
                    "A_score": "max_A_score",
                    "B_score": "max_B_score",
                    "Param_Name": "Primary_Param",
                }
            )
        )

        uncertain = uncertain.merge(row_summary, on="Component_ID", how="left")

        # Add review fields (empty for QA to fill)
        uncertain["QA_Label"] = ""  # Good / Defective / Unsure
        uncertain["QA_Confidence"] = ""  # High / Medium / Low
        uncertain["QA_Notes"] = ""  # Inspector comments
        uncertain["Review_Timestamp"] = ""

        # Reorder columns for display
        cols = [
            "Lot_ID",
            "Component_ID",
            "Primary_Param",
            "C_score",
            "max_A_score",
            "max_B_score",
            "dpat_maxz",
            "maha_score",
            "vae_score",
            "if_score",
            "Decision",
            "QA_Label",
            "QA_Confidence",
            "QA_Notes",
            "Review_Timestamp",
        ]
        return uncertain[[c for c in cols if c in uncertain.columns]]

    def ingest_feedback(
        self,
        feedback_df: pd.DataFrame,
        original_rows: pd.DataFrame,
        original_comp: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Merge QA-labeled samples back into training data.

        Args:
            feedback_df: DataFrame with columns [Lot_ID, Component_ID, QA_Label, QA_Confidence, QA_Notes].
            original_rows: Original row-level scores (for reference).
            original_comp: Original component-level scores (for reference).

        Returns:
            (augmented_rows, stats_dict) where augmented_rows includes QA feedback,
            and stats_dict summarizes the labeling (counts, agreement, etc).
        """
        # Filter to reviewed samples
        reviewed = feedback_df[feedback_df["QA_Label"].notna()].copy()
        if len(reviewed) == 0:
            return original_rows, {"samples_reviewed": 0}

        # Map QA labels to binary (defective=1, good=0)
        reviewed["QA_Defective"] = (reviewed["QA_Label"] == "Defective").astype(int)
        reviewed["QA_Unsure"] = (reviewed["QA_Label"] == "Unsure").astype(int)

        # Count agreement with automatic decision
        auto_reject = original_comp[
            original_comp["Decision"] == "REJECT"
        ].set_index("Component_ID")
        reviewed["_auto_reject"] = reviewed["Component_ID"].isin(
            auto_reject.index
        )
        reviewed["_qa_defective"] = reviewed["QA_Label"] == "Defective"
        agreement = (reviewed["_auto_reject"] == reviewed["_qa_defective"]).mean()

        stats = {
            "samples_reviewed": len(reviewed),
            "qa_defective_count": int((reviewed["QA_Label"] == "Defective").sum()),
            "qa_good_count": int((reviewed["QA_Label"] == "Good").sum()),
            "qa_unsure_count": int((reviewed["QA_Label"] == "Unsure").sum()),
            "agreement_with_auto": float(agreement),
            "high_confidence_count": int(
                (reviewed["QA_Confidence"] == "High").sum()
            ),
        }

        # Attach QA feedback to rows
        augmented_rows = original_rows.copy()
        comp_to_qa = reviewed.set_index("Component_ID")[
            ["QA_Label", "QA_Confidence", "QA_Defective", "QA_Unsure"]
        ]
        augmented_rows = augmented_rows.merge(
            comp_to_qa,
            left_on="Component_ID",
            right_index=True,
            how="left",
        )

        self.reviewed_samples.extend(reviewed.to_dict("records"))
        return augmented_rows, stats

    def save_feedback(self, feedback_df: pd.DataFrame, path: Path):
        """Persist QA feedback to JSON for later retraining."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing feedback log (or create new)
        all_feedback = []
        if path.exists():
            try:
                all_feedback = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        # Add new feedback with timestamp
        reviewed = feedback_df[feedback_df["QA_Label"].notna()].copy()
        reviewed["_ingestion_timestamp"] = datetime.now().isoformat()
        all_feedback.extend(reviewed.to_dict("records"))

        # Write back
        path.write_text(json.dumps(all_feedback, indent=2), encoding="utf-8")

    def load_feedback(self, path: Path) -> Optional[pd.DataFrame]:
        """Load previously-collected QA feedback from disk."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return pd.DataFrame(data) if data else None
        except (json.JSONDecodeError, ValueError):
            return None


class ActiveLearningSystem:
    """Orchestrate the full active learning loop: sample → review → retrain."""

    def __init__(
        self,
        feedback_path: Optional[Path] = None,
        uncertainty_band: Tuple[float, float] = (0.7, 0.9),
    ):
        """
        Args:
            feedback_path: Where to persist QA feedback (default: models/qa_feedback.json).
            uncertainty_band: Uncertainty range for sampling.
        """
        self.sampler = UncertaintySampler(uncertainty_band)
        self.feedback_path = feedback_path or Path("models/qa_feedback.json")
        self.feedback_history: Optional[pd.DataFrame] = None

    def get_review_queue(
        self, comp: pd.DataFrame, rows: pd.DataFrame, max_samples: int = 50
    ) -> pd.DataFrame:
        """Prepare the next batch of uncertain components for QA review."""
        queue = self.sampler.format_for_review(comp, rows, max_samples)
        self.review_queue = queue
        return queue

    def submit_review(self, reviewed_df: pd.DataFrame) -> Dict:
        """Accept QA inspector feedback and update internal state."""
        augmented_rows, stats = self.sampler.ingest_feedback(
            reviewed_df, pd.DataFrame(), pd.DataFrame()
        )
        self.sampler.save_feedback(reviewed_df, self.feedback_path)
        self.feedback_history = reviewed_df
        return stats

    def get_feedback_summary(self) -> Dict:
        """Return aggregate statistics on collected feedback."""
        if not self.feedback_path.exists():
            return {
                "total_reviewed": 0,
                "feedback_path": str(self.feedback_path),
                "message": "No feedback collected yet.",
            }

        feedback = self.sampler.load_feedback(self.feedback_path)
        if feedback is None or len(feedback) == 0:
            return {
                "total_reviewed": 0,
                "feedback_path": str(self.feedback_path),
                "message": "No feedback collected yet.",
            }

        return {
            "total_reviewed": len(feedback),
            "defective_labeled": int((feedback.get("QA_Label") == "Defective").sum()),
            "good_labeled": int((feedback.get("QA_Label") == "Good").sum()),
            "unsure_labeled": int((feedback.get("QA_Label") == "Unsure").sum()),
            "high_confidence": int(
                (feedback.get("QA_Confidence") == "High").sum()
            ),
            "feedback_path": str(self.feedback_path),
            "last_updated": feedback["_ingestion_timestamp"].max()
            if "_ingestion_timestamp" in feedback.columns
            else "unknown",
        }
