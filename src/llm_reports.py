"""LLM-powered report generation: Claude Opus synthesizes burn-in analysis for QA inspectors.

Converts complex multi-modal signals (anomaly scores, drift predictions, spatial clusters,
survival probabilities) into natural-language recommendations suitable for aerospace QA review.
"""
from __future__ import annotations

import os

import pandas as pd

try:
    import anthropic
except ImportError:  # optional in slim API deploys
    anthropic = None


class LLMReportGenerator:
    """Generate natural-language reports using Claude Opus."""

    def __init__(self, model: str = "claude-opus-5-5"):
        if anthropic is None:
            raise RuntimeError("anthropic package is not installed")
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic()
        self.model = model

    def generate_component_report(self, component_id: str, param_name: str,
                                  component_row: pd.Series, all_params_for_component: pd.DataFrame) -> str:
        """Generate QA-friendly report for one component-parameter pair.

        Args:
            component_id: Component identifier
            param_name: Parameter name (Iddq, Leakage, Prop_Delay)
            component_row: Row from decisions DataFrame with scores and decision
            all_params_for_component: All parameter rows for this component (for cross-param context)

        Returns:
            Markdown report with defect summary, reasoning, and recommendation
        """
        # Build structured data for Claude
        decision = component_row.get("Decision", "UNKNOWN")
        a_score = component_row.get("A_score", 0.0)
        b_score = component_row.get("B_score", 0.0)
        survival_prob = component_row.get("survival_prob", 0.99)
        mission_risk = component_row.get("mission_risk", "LOW")

        # Get parameter values
        v0 = component_row.get("Value_0h", "N/A")
        v24 = component_row.get("Value_24h", "N/A")
        v168 = component_row.get("Value_168h", "N/A")

        # Build prompt
        prompt = f"""You are a burn-in test QA analyst. Analyze this component and provide a concise,
actionable report for aerospace quality assurance.

COMPONENT DETAILS:
- Component ID: {component_id}
- Parameter: {param_name}
- Values: 0h={v0}, 24h={v24}, 168h={v168}
- Decision: {decision}

ANOMALY DETECTION (Module A):
- A_score: {a_score:.3f} (1.0 = anomaly threshold)
- Interpretation: {"ANOMALOUS" if a_score >= 1.0 else "normal"}

DRIFT PREDICTION (Module B):
- B_score: {b_score:.3f} (1.0 = drift threshold)
- Interpretation: {"ABNORMAL_DRIFT" if b_score >= 1.0 else "acceptable drift"}

MISSION-LIFE SAFETY (Module C):
- 15-year survival probability: {survival_prob:.1%}
- Mission risk: {mission_risk}
- Interpretation: {"MARGINAL" if mission_risk == "HIGH" else ("ACCEPTABLE" if mission_risk == "LOW" else "BORDERLINE")}

CROSS-PARAMETER CONTEXT:
- Other flagged parameters: {len(all_params_for_component[all_params_for_component['Decision'] != 'ACCEPT'])} of {len(all_params_for_component)}
- Pattern: {"multiple anomalies" if len(all_params_for_component[all_params_for_component['Decision'] != 'ACCEPT']) > 1 else "isolated"}

OUTPUT: Generate a concise 3-4 paragraph report with:
1. Defect Summary: What's wrong? Which modules flagged it?
2. Severity: Why should we care? (safety risk, infant mortality, latent defect?)
3. Recommended Action: REJECT / REVIEW / ACCEPT with confidence
4. Process Insight: Any fab process adjustment hint?

Keep it technical but readable for QA inspectors. Use bullet points where helpful."""

        # Call Claude Opus
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            report = message.content[0].text
        except Exception as e:
            report = f"**Error generating report:** {str(e)}\n\n**Fallback Summary:**\n- Decision: {decision}\n- Anomaly Score: {a_score:.3f}\n- Drift Score: {b_score:.3f}\n- Mission Risk: {mission_risk}"

        return report

    def generate_lot_summary(self, lot_id: str, comp_df: pd.DataFrame) -> str:
        """Generate executive summary for entire lot.

        Args:
            lot_id: Lot identifier
            comp_df: Component-level DataFrame (all components in lot, first param per component)

        Returns:
            Markdown lot summary with statistics and process insights
        """
        total = len(comp_df)
        rejected = len(comp_df[comp_df["Decision"] == "REJECT"])
        reviewed = len(comp_df[comp_df["Decision"] == "REVIEW"])
        accepted = len(comp_df[comp_df["Decision"] == "ACCEPT"])
        overkill = len(comp_df[(comp_df["Decision"] != "ACCEPT") & (comp_df["Is_Defective"] == 0)]) / total if total > 0 else 0

        high_risk = len(comp_df[comp_df["mission_risk"] == "HIGH"])
        mean_survival = comp_df["survival_prob"].mean() if "survival_prob" in comp_df.columns else 0.99

        prompt = f"""You are a burn-in test quality lead. Provide a concise executive summary for fab management.

LOT STATISTICS:
- Lot ID: {lot_id}
- Total components: {total}
- Rejected: {rejected} ({rejected/total*100:.1f}%)
- Reviewed: {reviewed} ({reviewed/total*100:.1f}%)
- Accepted: {accepted} ({accepted/total*100:.1f}%)
- Overkill rate: {overkill*100:.2f}%

MISSION-LIFE CONCERNS:
- High-risk components (survival < 0.99): {high_risk}
- Mean survival probability: {mean_survival:.2%}

OUTPUT: Generate a 2-3 paragraph executive summary with:
1. Lot Quality Verdict: PASS / CAUTION / HOLD with reasoning
2. Key Metrics: Defect rate, overkill, survival risk
3. Process Action: Any fab process adjustments recommended?

Keep it concise, actionable, and fab-friendly."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            summary = message.content[0].text
        except Exception as e:
            summary = f"**Lot Summary Error:** {str(e)}\n\n**Quick Stats:**\n- Total: {total} | Rejected: {rejected} | Overkill: {overkill*100:.1f}%\n- Mean survival: {mean_survival:.1%}"

        return summary
