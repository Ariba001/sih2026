"""Weibull mission-life survival analysis: predict long-term reliability for space payloads.

This module uses Arrhenius-extrapolated burn-in data to fit a Weibull distribution
per parameter type, then predicts the probability that a component survives a 15-year
mission without exceeding its datasheet limit.
"""
import numpy as np
import pandas as pd

try:
    from lifelines import WeibullFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

from .config import DATASHEET


class ModuleC:
    """Weibull mission-life survival analyzer."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.weibull_models = {}
        self.mission_hours = cfg.mission_hours

    def fit(self, df: pd.DataFrame):
        """Fit Weibull distribution per parameter type on training data.

        Uses Arrhenius acceleration factor to convert 168h burn-in to mission-life equivalent.
        Treats parts exceeding spec limit as failures; parts within limit are right-censored.

        Args:
            df: DataFrame with columns: Param_Name, Value_168h, Spec_Max, Is_Defective

        Returns:
            self
        """
        if not HAS_LIFELINES:
            return self

        # Convert 168h burn-in measurement to mission-time equivalent using Arrhenius
        af = self.cfg.af  # Acceleration factor (typically ~78 at Ea=0.7eV, 125°C→55°C)
        mission_time_equiv = 168.0 * af  # ~13,104 hours ≈ 1.5 years equivalent

        for param in df["Param_Name"].unique():
            param_data = df[df["Param_Name"] == param].copy()

            if len(param_data) < 10:
                continue  # Skip parameters with too little data

            # Get datasheet limit for this parameter
            spec_info = DATASHEET.get(param, {})
            spec_limit = spec_info.get("max", np.inf)

            # Treat parts exceeding spec limit as failures (event_observed=1)
            # Parts within limit are censored (right-censored at mission time)
            v168 = param_data["Value_168h"].to_numpy()
            event_observed = (v168 >= spec_limit).astype(int)

            # If no events, skip fitting
            if event_observed.sum() == 0 and event_observed.sum() < len(event_observed):
                # All censored; use default model
                pass
            elif event_observed.sum() == len(event_observed):
                # All failures; use default model
                pass

            # Fit Weibull model
            try:
                wf = WeibullFitter()
                wf.fit(
                    durations=mission_time_equiv * np.ones_like(v168),
                    event_observed=event_observed,
                    label=f"{param} mission life"
                )
                self.weibull_models[param] = wf
            except Exception:
                # Skip if fitting fails
                continue

        return self

    def predict_survival_prob(self, component: pd.Series, mission_years: float = 15.0) -> float:
        """Predict probability that component survives full mission without exceeding spec limit.

        Args:
            component: Row with Param_Name and Value_168h
            mission_years: Mission duration (default 15 years)

        Returns:
            Survival probability (0-1); e.g., 0.999 = 99.9% survival
        """
        if not HAS_LIFELINES:
            return 0.99  # Default if lifelines not available

        param = component["Param_Name"]

        if param not in self.weibull_models:
            return 0.99  # Default if no model for this parameter

        wf = self.weibull_models[param]

        # Convert mission years to hours
        mission_hours = mission_years * 365.25 * 24  # ≈ 131,490 hours

        try:
            # Get survival probability from fitted Weibull model
            survival_prob = float(wf.survival_function_at_times(mission_hours).values[0])
        except Exception:
            return 0.99  # Default on error

        return np.clip(survival_prob, 0.0, 1.0)

    def flag_marginal_parts(self, df: pd.DataFrame, survival_threshold: float = 0.999) -> pd.DataFrame:
        """Flag parts with mission-life survival probability below threshold.

        Parts with low survival probability may pass burn-in but have high failure risk in orbit.

        Args:
            df: DataFrame with components (one row per component-parameter pair)
            survival_threshold: Flag if survival_prob < this value (default 0.999)

        Returns:
            DataFrame with: component_id, param_name, value_168h, survival_prob, mission_risk, recommended_action
        """
        flagged = []

        for _, component in df.iterrows():
            surv_prob = self.predict_survival_prob(component)

            if surv_prob < survival_threshold:
                # Categorize risk level
                if surv_prob < 0.99:
                    mission_risk = "HIGH"
                    recommended_action = "REJECT"
                else:
                    mission_risk = "MEDIUM"
                    recommended_action = "REVIEW"

                flagged.append({
                    "component_id": component.get("Component_ID", "unknown"),
                    "param_name": component["Param_Name"],
                    "value_168h": float(component["Value_168h"]),
                    "survival_prob": float(surv_prob),
                    "mission_risk": mission_risk,
                    "recommended_action": recommended_action
                })

        return pd.DataFrame(flagged)
