"""Streamlit dashboard for burn-in anomaly detection QA review.

Interactive interface for QA inspectors to review component decisions, visualize
spatial patterns, and download audit reports for certification.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib

# Page config
st.set_page_config(
    page_title="Burn-In QA Review Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-box { background: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .reject { color: #d32f2f; font-weight: bold; }
    .review { color: #f57c00; font-weight: bold; }
    .accept { color: #388e3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_system():
    """Load trained BurnInSystem from disk."""
    model_path = Path(__file__).parent / "models" / "burnin_system.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


def score_and_sync(system, df, *, source: str = "streamlit", explain: bool = False):
    """Score with the same BurnInSystem FastAPI uses; sync results for Vite dashboard."""
    from src.results_store import persist_scored_results

    rows, comp = system.score(df, explain=explain)
    persist_scored_results(comp, df, source=source)
    return rows, comp


# Sidebar navigation
st.sidebar.title("🔥 Burn-In QA Dashboard")
page = st.sidebar.radio(
    "Select Page",
    ["Lot Overview", "Component Detail", "Spatial Analysis", "Survival Analysis", "Audit Log"]
)

system = load_system()
if not system:
    st.error("❌ Model not found. Please train the system first.")
    st.stop()

# ================================================================ Page: Lot Overview

if page == "Lot Overview":
    st.title("📊 Lot Overview")

    # File uploader
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

    if uploaded_file:
        from src.data import load_csv, validate

        try:
            df = load_csv(uploaded_file)
            df = validate(df)

            # Score (same pipeline as FastAPI; writes reports/current_results.json)
            rows, comp = score_and_sync(system, df, source="streamlit:lot_overview")

            # Extract lot_id (should be single lot for this page)
            lot_ids = df["Lot_ID"].unique()
            if len(lot_ids) > 1:
                st.warning(f"Multiple lots detected: {lot_ids}. Showing first lot only.")
            lot_id = lot_ids[0]

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Components", len(comp))
            with col2:
                reject_count = len(comp[comp["Decision"] == "REJECT"])
                st.metric("Rejected", reject_count, f"{reject_count/len(comp)*100:.1f}%")
            with col3:
                review_count = len(comp[comp["Decision"] == "REVIEW"])
                st.metric("Reviewed", review_count, f"{review_count/len(comp)*100:.1f}%")
            with col4:
                accept_count = len(comp[comp["Decision"] == "ACCEPT"])
                st.metric("Accepted", accept_count, f"{accept_count/len(comp)*100:.1f}%")

            # Decision breakdown pie chart
            decision_counts = comp["Decision"].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=decision_counts.index,
                values=decision_counts.values,
                marker=dict(colors=["#d32f2f", "#f57c00", "#388e3c"])
            )])
            fig_pie.update_layout(title="Decision Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Overkill analysis
            overkill = comp[(comp["Decision"] != "ACCEPT") & (comp["Is_Defective"] == 0)]
            st.metric("Overkill Rate", f"{len(overkill)/len(comp)*100:.2f}%")

            # Survival risk breakdown
            if "mission_risk" in comp.columns:
                risk_counts = comp["mission_risk"].value_counts()
                st.subheader("Mission-Life Risk Distribution")
                col1, col2, col3 = st.columns(3)
                with col1:
                    high_count = len(comp[comp["mission_risk"] == "HIGH"])
                    st.metric("HIGH Risk", high_count, "Recommend REJECT")
                with col2:
                    med_count = len(comp[comp["mission_risk"] == "MEDIUM"])
                    st.metric("MEDIUM Risk", med_count, "Recommend REVIEW")
                with col3:
                    low_count = len(comp[comp["mission_risk"] == "LOW"])
                    st.metric("LOW Risk", low_count, "Acceptable")

        except Exception as e:
            st.error(f"Error: {e}")

# ================================================================ Page: Component Detail

elif page == "Component Detail":
    st.title("🔍 Component Detail")

    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"], key="detail_upload")

    if uploaded_file:
        from src.data import load_csv, validate

        try:
            df = load_csv(uploaded_file)
            df = validate(df)
            rows, comp = score_and_sync(system, df, source="streamlit:component_detail")

            # Component selector
            comp_id = st.selectbox("Select Component ID", comp["Component_ID"].unique())

            if comp_id:
                comp_data = comp[comp["Component_ID"] == comp_id]
                comp_rows = rows[rows["Component_ID"] == comp_id]

                st.subheader(f"Component: {comp_id}")

                # Decision summary
                decision = comp_data.iloc[0]["Decision"]
                decision_color = {"REJECT": "red", "REVIEW": "orange", "ACCEPT": "green"}.get(decision, "gray")
                st.markdown(f"**Decision:** <span style='color:{decision_color}; font-size:20px;'>{decision}</span>",
                           unsafe_allow_html=True)

                # Scores
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("A_score (Anomaly)", f"{comp_data.iloc[0]['A_score']:.3f}")
                with col2:
                    st.metric("B_score (Drift)", f"{comp_data.iloc[0]['B_score']:.3f}")
                with col3:
                    st.metric("Survival Prob", f"{comp_data.iloc[0].get('survival_prob', 0.99):.1%}")

                # Parameter details
                st.subheader("Parameter Values")
                for param in comp_rows["Param_Name"].unique():
                    param_data = comp_rows[comp_rows["Param_Name"] == param].iloc[0]
                    with st.expander(f"{param}"):
                        cols = st.columns(4)
                        cols[0].metric("0h", f"{param_data['Value_0h']:.4f}")
                        cols[1].metric("24h", f"{param_data['Value_24h']:.4f}")
                        cols[2].metric("96h", f"{param_data['Value_96h']:.4f}")
                        cols[3].metric("168h", f"{param_data['Value_168h']:.4f}")

                # Reasoning
                if "Reason_Codes" in comp_data.columns:
                    st.subheader("Reason Codes")
                    reasons = str(comp_data.iloc[0]["Reason_Codes"]).split(",")
                    for reason in reasons:
                        st.write(f"- {reason.strip()}")

        except Exception as e:
            st.error(f"Error: {e}")

# ================================================================ Page: Spatial Analysis

elif page == "Spatial Analysis":
    st.title("🗺️ Spatial Anomaly Clustering")

    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"], key="spatial_upload")

    if uploaded_file:
        from src.data import load_csv, validate

        try:
            df = load_csv(uploaded_file)
            df = validate(df)
            rows, comp = score_and_sync(system, df, source="streamlit:spatial")

            if "die_x" in rows.columns and "die_y" in rows.columns:
                # Spatial scatter plot
                fig = px.scatter(
                    rows,
                    x="die_x",
                    y="die_y",
                    color="A_score",
                    size="survival_prob",
                    hover_data=["Component_ID", "Param_Name", "A_score"],
                    color_continuous_scale="RdYlGn_r",
                    title="Wafer Die Map (Color=Anomaly, Size=Survival)",
                    height=600
                )
                fig.update_layout(
                    xaxis_title="Die X (mm)",
                    yaxis_title="Die Y (mm)",
                    hovermode="closest"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.info("🔴 Red = High anomaly score | 🟢 Green = Normal | Larger circles = safer parts")
            else:
                st.warning("No spatial coordinates (die_x, die_y) in uploaded data.")

        except Exception as e:
            st.error(f"Error: {e}")

# ================================================================ Page: Survival Analysis

elif page == "Survival Analysis":
    st.title("⏱️ Mission-Life Survival Analysis")

    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"], key="survival_upload")

    if uploaded_file:
        from src.data import load_csv, validate

        try:
            df = load_csv(uploaded_file)
            df = validate(df)
            rows, comp = score_and_sync(system, df, source="streamlit:survival")

            if "survival_prob" in comp.columns:
                # Histogram of survival probabilities
                fig_hist = px.histogram(
                    comp,
                    x="survival_prob",
                    nbins=30,
                    title="Distribution of 15-Year Mission Survival Probability",
                    labels={"survival_prob": "Survival Probability"},
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # Mission risk breakdown
                st.subheader("Risk Categories")
                high_risk = len(comp[comp["mission_risk"] == "HIGH"])
                med_risk = len(comp[comp["mission_risk"] == "MEDIUM"])
                low_risk = len(comp[comp["mission_risk"] == "LOW"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔴 HIGH Risk (< 0.99)", high_risk, f"{high_risk/len(comp)*100:.1f}%")
                with col2:
                    st.metric("🟡 MEDIUM Risk (0.99-0.999)", med_risk, f"{med_risk/len(comp)*100:.1f}%")
                with col3:
                    st.metric("🟢 LOW Risk (> 0.999)", low_risk, f"{low_risk/len(comp)*100:.1f}%")

                # Details of high-risk parts
                if high_risk > 0:
                    st.subheader("High-Risk Components")
                    high_risk_comp = comp[comp["mission_risk"] == "HIGH"][
                        ["Component_ID", "survival_prob", "A_score", "B_score", "Decision"]
                    ].sort_values("survival_prob")
                    st.dataframe(high_risk_comp, use_container_width=True)

            else:
                st.warning("No survival probability data available.")

        except Exception as e:
            st.error(f"Error: {e}")

# ================================================================ Page: Audit Log

elif page == "Audit Log":
    st.title("📋 Audit Log (Placeholder)")

    st.info("""
    **Audit Log Features (Coming Soon):**
    - Query PostgreSQL for decision history
    - Filter by lot, date, decision type
    - Download CSV for certifications
    - Traceability for aerospace compliance
    """)

    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"], key="audit_upload")
    if uploaded_file:
        from src.data import load_csv, validate
        df = load_csv(uploaded_file)
        df = validate(df)
        rows, comp = score_and_sync(system, df, source="streamlit:audit")

        st.subheader("Component Decisions (Audit Trail)")
        audit_df = comp[[
            "Lot_ID", "Component_ID", "Param_Name", "Decision",
            "A_score", "B_score", "survival_prob", "mission_risk"
        ]].copy()
        audit_df.columns = ["Lot", "Component", "Parameter", "Decision",
                           "Anomaly_Score", "Drift_Score", "Survival_Prob", "Mission_Risk"]

        st.dataframe(audit_df, use_container_width=True)

        # Download button
        csv = audit_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Audit Trail (CSV)",
            data=csv,
            file_name="burn_in_audit_trail.csv",
            mime="text/csv"
        )

st.sidebar.markdown("---")
st.sidebar.markdown("**System Info:**")
st.sidebar.markdown(f"- Model: BurnInSystem v2.0")
st.sidebar.markdown(f"- Status: ✅ Ready")
st.sidebar.markdown("**Connected apps:**")
st.sidebar.markdown("- Vite UI: http://localhost:5173")
st.sidebar.markdown("- REST API: http://localhost:8000")
st.sidebar.caption("Uploads here sync to reports/current_results.json for the Vite dashboard.")
