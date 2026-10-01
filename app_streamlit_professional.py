"""
Professional Streamlit Dashboard for Burn-In Anomaly Detection
Modern, clean interface for aerospace-grade quality assurance
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
import numpy as np
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Burn-In QA Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODERN PROFESSIONAL STYLING
# ============================================================================

st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e2e8f0;
    }

    /* Custom metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .metric-change {
        font-size: 0.875rem;
        color: #64748b;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-accept {
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
    }

    .badge-review {
        background: rgba(245, 158, 11, 0.1);
        color: #d97706;
    }

    .badge-reject {
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
    }

    /* Cards */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }

    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        padding: 0 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Buttons */
    .stButton button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s;
    }

    .stButton button:hover {
        background: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
    }

    /* Dataframe */
    .dataframe {
        font-size: 0.875rem;
        border-radius: 8px;
        overflow: hidden;
    }

    .dataframe thead tr th {
        background: #f8fafc;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        color: #64748b;
        padding: 0.875rem 1rem;
    }

    .dataframe tbody tr:hover {
        background: #f8fafc;
    }

    /* Alert boxes */
    .alert {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin-bottom: 1.5rem;
    }

    .alert-info {
        background: #eff6ff;
        border-left-color: #3b82f6;
        color: #1e40af;
    }

    .alert-success {
        background: #d1fae5;
        border-left-color: #059669;
        color: #065f46;
    }

    .alert-warning {
        background: #fef3c7;
        border-left-color: #d97706;
        color: #92400e;
    }

    .alert-danger {
        background: #fee2e2;
        border-left-color: #dc2626;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SYSTEM LOADING
# ============================================================================

@st.cache_resource
def load_system():
    """Load trained BurnInSystem from disk."""
    model_path = Path(__file__).parent / "models" / "burnin_system.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Process uploaded CSV data."""
    from src.data import load_csv, validate
    df = load_csv(uploaded_file)
    df = validate(df)
    return df

@st.cache_data
def load_default_data():
    """Load default project data (burnin_synthetic.csv)."""
    import pandas as pd
    data_path = Path(__file__).parent / "data" / "burnin_synthetic.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🔬</div>
            <h1 style='color: white; font-size: 1.5rem; font-weight: 800; margin: 0;'>Burn-In QA</h1>
            <p style='color: #94a3b8; font-size: 0.875rem; margin: 0.5rem 0 0 0;'>Aerospace Quality Assurance</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # System status
    system = load_system()

    if system:
        st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);'>
                <div style='display: flex; align-items: center; gap: 0.75rem;'>
                    <div style='width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: pulse 2s infinite;'></div>
                    <div>
                        <div style='color: #10b981; font-weight: 700; font-size: 0.875rem;'>System Ready</div>
                        <div style='color: #94a3b8; font-size: 0.75rem;'>BurnInSystem v2.0</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);'>
                <div style='display: flex; align-items: center; gap: 0.75rem;'>
                    <div style='color: #ef4444; font-size: 1.25rem;'>⚠️</div>
                    <div>
                        <div style='color: #ef4444; font-weight: 700; font-size: 0.875rem;'>Model Not Found</div>
                        <div style='color: #94a3b8; font-size: 0.75rem;'>Train system first</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Navigation
    st.markdown("<h3 style='color: #e2e8f0; font-size: 0.875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem;'>Navigation</h3>", unsafe_allow_html=True)

    page = st.radio(
        "Select Page",
        ["📊 Lot Overview", "🔍 Component Detail", "🗺️ Spatial Analysis", "⏱️ Survival Analysis", "📋 Audit Log"],
        label_visibility="collapsed"
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # About
    st.markdown("""
        <div style='padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;'>
            <h4 style='color: #e2e8f0; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>About</h4>
            <p style='color: #94a3b8; font-size: 0.75rem; line-height: 1.5; margin: 0;'>
                Aerospace-grade anomaly detection with 96%+ recall and 4.5% overkill rate.
                MIL-STD-883 compliant documentation.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not system:
    st.markdown("""
        <div class='alert alert-danger'>
            <h3 style='margin: 0 0 0.5rem 0; font-weight: 700;'>❌ System Not Available</h3>
            <p style='margin: 0;'>Model not found. Please train the burn-in system first:</p>
            <pre style='background: rgba(0,0,0,0.1); padding: 0.5rem; border-radius: 4px; margin-top: 0.5rem;'>python train.py</pre>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================================
# PAGE: LOT OVERVIEW
# ============================================================================

if page == "📊 Lot Overview":
    st.markdown("<h1 style='font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;'>📊 Lot Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1rem; margin-bottom: 2rem;'>Comprehensive lot-level quality assessment and decision distribution</p>", unsafe_allow_html=True)

    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["📊 Current Project Results", "📤 Upload New CSV"],
        horizontal=True
    )

    df = None

    if data_source == "📊 Current Project Results":
        # Load default project data
        with st.spinner("🔄 Loading current project results..."):
            df = load_default_data()
            if df is None:
                st.error("❌ Default data file not found. Please upload a CSV file.")
                st.stop()
    else:
        # CSV upload option
        uploaded_file = st.file_uploader(
            "📤 Upload Burn-In Test Data (CSV)",
            type=["csv"],
            help="Upload CSV file with burn-in test results"
        )

        if uploaded_file:
            df = process_uploaded_file(uploaded_file)

    if df is not None:
        try:
            with st.spinner("🔄 Analyzing components..."):
                rows, comp = system.score(df, explain=False)

            lot_ids = df["Lot_ID"].unique()
            if len(lot_ids) > 1:
                st.warning(f"⚠️ Multiple lots detected: {', '.join(lot_ids)}. Showing aggregate metrics.")

            # Key metrics
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            total = len(comp)
            reject = len(comp[comp["Decision"] == "REJECT"])
            review = len(comp[comp["Decision"] == "REVIEW"])
            accept = len(comp[comp["Decision"] == "ACCEPT"])

            with col1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Total Components</div>
                        <div class='metric-value'>{total:,}</div>
                        <div class='metric-change'>Lot {lot_ids[0]}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Rejected</div>
                        <div class='metric-value' style='color: #dc2626;'>{reject}</div>
                        <div class='metric-change'>↓ {(reject/total*100):.1f}% of total</div>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Under Review</div>
                        <div class='metric-value' style='color: #d97706;'>{review}</div>
                        <div class='metric-change'>{(review/total*100):.1f}% requires inspection</div>
                    </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Accepted</div>
                        <div class='metric-value' style='color: #059669;'>{accept}</div>
                        <div class='metric-change'>↑ {(accept/total*100):.1f}% pass rate</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='custom-card'><h3 class='card-title'>Decision Distribution</h3>", unsafe_allow_html=True)

                decision_counts = comp["Decision"].value_counts()
                colors = {"ACCEPT": "#10b981", "REVIEW": "#f59e0b", "REJECT": "#ef4444"}

                fig = go.Figure(data=[go.Pie(
                    labels=decision_counts.index,
                    values=decision_counts.values,
                    marker=dict(colors=[colors.get(d, "#6b7280") for d in decision_counts.index]),
                    textinfo='label+percent',
                    textfont=dict(size=14, family="Inter"),
                    hole=0.4
                )])

                fig.update_layout(
                    height=350,
                    margin=dict(t=20, b=20, l=20, r=20),
                    showlegend=True,
                    font=dict(family="Inter"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='custom-card'><h3 class='card-title'>Quality Metrics</h3>", unsafe_allow_html=True)

                # Overkill and recall
                if "Is_Defective" in comp.columns:
                    overkill = comp[(comp["Decision"] != "ACCEPT") & (comp["Is_Defective"] == 0)]
                    overkill_rate = (len(overkill) / len(comp) * 100) if len(comp) > 0 else 0

                    defective = comp[comp["Is_Defective"] == 1]
                    if len(defective) > 0:
                        caught = defective[defective["Decision"] != "ACCEPT"]
                        recall = (len(caught) / len(defective) * 100)

                        st.markdown(f"""
                            <div style='margin-bottom: 2rem;'>
                                <div class='metric-label'>Recall Rate</div>
                                <div style='font-size: 2rem; font-weight: 800; color: #059669; margin: 0.5rem 0;'>{recall:.1f}%</div>
                                <div style='width: 100%; height: 8px; background: #f1f5f9; border-radius: 4px; overflow: hidden;'>
                                    <div style='width: {recall}%; height: 100%; background: #059669;'></div>
                                </div>
                                <div style='color: #64748b; font-size: 0.875rem; margin-top: 0.5rem;'>Defect detection rate</div>
                            </div>

                            <div>
                                <div class='metric-label'>Overkill Rate</div>
                                <div style='font-size: 2rem; font-weight: 800; color: #0f172a; margin: 0.5rem 0;'>{overkill_rate:.2f}%</div>
                                <div style='width: 100%; height: 8px; background: #f1f5f9; border-radius: 4px; overflow: hidden;'>
                                    <div style='width: {min(overkill_rate*20, 100)}%; height: 100%; background: #f59e0b;'></div>
                                </div>
                                <div style='color: #64748b; font-size: 0.875rem; margin-top: 0.5rem;'>False reject rate (target: &lt;5%)</div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("📊 Ground truth labels not available")

                st.markdown("</div>", unsafe_allow_html=True)

            # Mission risk
            if "mission_risk" in comp.columns:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='custom-card'><h3 class='card-title'>Mission-Life Risk Assessment</h3>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)

                high = len(comp[comp["mission_risk"] == "HIGH"])
                medium = len(comp[comp["mission_risk"] == "MEDIUM"])
                low = len(comp[comp["mission_risk"] == "LOW"])

                with col1:
                    st.markdown(f"""
                        <div style='border-left: 4px solid #dc2626; padding-left: 1rem;'>
                            <div class='metric-label' style='color: #dc2626;'>🔴 High Risk</div>
                            <div class='metric-value' style='font-size: 2rem;'>{high}</div>
                            <div class='metric-change'>{(high/total*100):.1f}% • Survival < 99.0%</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                        <div style='border-left: 4px solid #d97706; padding-left: 1rem;'>
                            <div class='metric-label' style='color: #d97706;'>🟡 Medium Risk</div>
                            <div class='metric-value' style='font-size: 2rem;'>{medium}</div>
                            <div class='metric-change'>{(medium/total*100):.1f}% • 99.0-99.9% survival</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                        <div style='border-left: 4px solid #059669; padding-left: 1rem;'>
                            <div class='metric-label' style='color: #059669;'>🟢 Low Risk</div>
                            <div class='metric-value' style='font-size: 2rem;'>{low}</div>
                            <div class='metric-change'>{(low/total*100):.1f}% • Survival > 99.9%</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # Components table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='custom-card'><h3 class='card-title'>Component Summary</h3>", unsafe_allow_html=True)

            display_cols = ["Component_ID", "Decision", "A_score", "B_score"]
            if "survival_prob" in comp.columns:
                display_cols.append("survival_prob")
            if "mission_risk" in comp.columns:
                display_cols.append("mission_risk")

            st.dataframe(
                comp[display_cols].head(50),
                use_container_width=True,
                hide_index=True,
                height=400
            )

            if len(comp) > 50:
                st.caption(f"Showing first 50 of {len(comp)} components")

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")

# ============================================================================
# PAGE: COMPONENT DETAIL
# ============================================================================

elif page == "🔍 Component Detail":
    st.markdown("<h1 style='font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;'>🔍 Component Detail Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1rem; margin-bottom: 2rem;'>Deep dive into individual component test results and decision rationale</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Burn-In Test Data (CSV)", type=["csv"], key="detail_upload")

    if uploaded_file:
        try:
            with st.spinner("🔄 Analyzing components..."):
                df = process_uploaded_file(uploaded_file)
                rows, comp = system.score(df, explain=True)

            comp_id = st.selectbox("Select Component ID", sorted(comp["Component_ID"].unique()))

            if comp_id:
                comp_data = comp[comp["Component_ID"] == comp_id].iloc[0]
                comp_rows = rows[rows["Component_ID"] == comp_id]

                decision = comp_data["Decision"]
                colors = {"REJECT": ("#dc2626", "#fee2e2"), "REVIEW": ("#d97706", "#fef3c7"), "ACCEPT": ("#059669", "#d1fae5")}
                color, bg = colors.get(decision, ("#64748b", "#f1f5f9"))

                st.markdown(f"""
                    <div style='background: {bg}; border-left: 4px solid {color}; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
                        <div class='metric-label' style='color: {color};'>Final Decision</div>
                        <div style='font-size: 2.5rem; font-weight: 800; color: {color}; margin: 0.5rem 0;'>{decision}</div>
                        <div style='color: #64748b;'>Component: <strong>{comp_id}</strong></div>
                    </div>
                """, unsafe_allow_html=True)

                # Scores
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Anomaly Score</div>
                            <div class='metric-value' style='font-size: 2rem; color: {color};'>{comp_data["A_score"]:.3f}</div>
                            <div class='metric-change'>Module A</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Drift Score</div>
                            <div class='metric-value' style='font-size: 2rem;'>{comp_data["B_score"]:.3f}</div>
                            <div class='metric-change'>Module B</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Combined Score</div>
                            <div class='metric-value' style='font-size: 2rem;'>{comp_data["C_score"]:.3f}</div>
                            <div class='metric-change'>Max normalized</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col4:
                    if "survival_prob" in comp_data:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-label'>Survival</div>
                                <div class='metric-value' style='font-size: 2rem;'>{comp_data["survival_prob"]:.1%}</div>
                                <div class='metric-change'>15-year mission</div>
                            </div>
                        """, unsafe_allow_html=True)

                # Parameter evolution
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='custom-card'><h3 class='card-title'>Parameter Evolution</h3>", unsafe_allow_html=True)

                for param in comp_rows["Param_Name"].unique():
                    param_data = comp_rows[comp_rows["Param_Name"] == param].iloc[0]

                    with st.expander(f"📈 {param}", expanded=True):
                        times = [0, 24, 96, 168]
                        values = [param_data["Value_0h"], param_data["Value_24h"], param_data["Value_96h"], param_data["Value_168h"]]

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=times, y=values,
                            mode='lines+markers',
                            line=dict(color='#3b82f6', width=3),
                            marker=dict(size=10, color='#3b82f6'),
                            name='Measured'
                        ))

                        if not pd.isna(param_data.get("Spec_Max")):
                            fig.add_hline(y=param_data["Spec_Max"], line_dash="dash", line_color="#dc2626", annotation_text="Max Limit")

                        fig.update_layout(
                            xaxis_title="Time (hours)",
                            yaxis_title=f"Value ({param_data.get('Unit', 'units')})",
                            height=300,
                            font=dict(family="Inter"),
                            hovermode='x unified',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Reason codes
                if "Reason_Codes" in comp_data and comp_data["Reason_Codes"]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<div class='custom-card'><h3 class='card-title'>Decision Rationale</h3>", unsafe_allow_html=True)
                    reasons = str(comp_data["Reason_Codes"]).split(";")
                    for reason in reasons:
                        if reason.strip():
                            st.markdown(f"• **{reason.strip()}**")
                    st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE: SPATIAL ANALYSIS
# ============================================================================

elif page == "🗺️ Spatial Analysis":
    st.markdown("<h1 style='font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;'>🗺️ Spatial Anomaly Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1rem; margin-bottom: 2rem;'>Wafer-level defect clustering and root cause localization</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Burn-In Test Data (CSV)", type=["csv"], key="spatial_upload")

    if uploaded_file:
        try:
            with st.spinner("🔄 Analyzing spatial patterns..."):
                df = process_uploaded_file(uploaded_file)
                rows, comp = system.score(df, explain=False)

            if "die_x" in rows.columns and "die_y" in rows.columns:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)

                fig = px.scatter(
                    rows, x="die_x", y="die_y", color="A_score",
                    size="A_score", hover_data=["Component_ID", "Param_Name", "A_score"],
                    color_continuous_scale="RdYlGn_r",
                    labels={"die_x": "Die X (mm)", "die_y": "Die Y (mm)", "A_score": "Anomaly Score"},
                    height=600
                )

                fig.update_layout(
                    font=dict(family="Inter"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.info("💡 🔴 Red = High anomaly | 🟡 Yellow = Moderate | 🟢 Green = Normal")

            else:
                st.warning("⚠️ No spatial coordinates (die_x, die_y) available in uploaded data.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE: SURVIVAL ANALYSIS
# ============================================================================

elif page == "⏱️ Survival Analysis":
    st.markdown("<h1 style='font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;'>⏱️ Mission-Life Survival Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1rem; margin-bottom: 2rem;'>Weibull-based 15-year mission reliability assessment</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Burn-In Test Data (CSV)", type=["csv"], key="survival_upload")

    if uploaded_file:
        try:
            with st.spinner("🔄 Computing survival probabilities..."):
                df = process_uploaded_file(uploaded_file)
                rows, comp = system.score(df, explain=False)

            if "survival_prob" in comp.columns:
                # Risk summary
                col1, col2, col3 = st.columns(3)

                high = len(comp[comp["mission_risk"] == "HIGH"])
                medium = len(comp[comp["mission_risk"] == "MEDIUM"])
                low = len(comp[comp["mission_risk"] == "LOW"])
                total = len(comp)

                with col1:
                    st.markdown(f"""
                        <div class='metric-card' style='border-left: 4px solid #dc2626;'>
                            <div class='metric-label' style='color: #dc2626;'>🔴 High Risk</div>
                            <div class='metric-value' style='font-size: 2rem; color: #dc2626;'>{high}</div>
                            <div class='metric-change'>{(high/total*100):.1f}% • < 99.0%</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                        <div class='metric-card' style='border-left: 4px solid #d97706;'>
                            <div class='metric-label' style='color: #d97706;'>🟡 Medium Risk</div>
                            <div class='metric-value' style='font-size: 2rem; color: #d97706;'>{medium}</div>
                            <div class='metric-change'>{(medium/total*100):.1f}% • 99.0-99.9%</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                        <div class='metric-card' style='border-left: 4px solid #059669;'>
                            <div class='metric-label' style='color: #059669;'>🟢 Low Risk</div>
                            <div class='metric-value' style='font-size: 2rem; color: #059669;'>{low}</div>
                            <div class='metric-change'>{(low/total*100):.1f}% • > 99.9%</div>
                        </div>
                    """, unsafe_allow_html=True)

                # Distribution chart
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='custom-card'><h3 class='card-title'>Survival Probability Distribution</h3>", unsafe_allow_html=True)

                fig = px.histogram(comp, x="survival_prob", nbins=50, labels={"survival_prob": "Survival Probability"}, height=400)
                fig.update_traces(marker_color='#3b82f6')
                fig.update_layout(
                    font=dict(family="Inter"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # High-risk table
                if high > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<div class='custom-card'><h3 class='card-title'>High-Risk Components</h3>", unsafe_allow_html=True)

                    high_risk = comp[comp["mission_risk"] == "HIGH"][["Component_ID", "survival_prob", "A_score", "B_score", "Decision"]].sort_values("survival_prob").head(20)
                    st.dataframe(high_risk, use_container_width=True, hide_index=True)

                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.info("ℹ️ Survival analysis not available - enable Weibull module in trained model")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE: AUDIT LOG
# ============================================================================

elif page == "📋 Audit Log":
    st.markdown("<h1 style='font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;'>📋 Audit Log & Traceability</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1rem; margin-bottom: 2rem;'>Complete decision history for MIL-STD certification compliance</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Burn-In Test Data (CSV)", type=["csv"], key="audit_upload")

    if uploaded_file:
        try:
            with st.spinner("🔄 Generating audit trail..."):
                df = process_uploaded_file(uploaded_file)
                rows, comp = system.score(df, explain=True)

            st.markdown("<div class='alert alert-info'><strong>MIL-STD-883 Compliant Documentation</strong><br>Complete decision history with timestamps, system version tracking, and traceability for certification records.</div>", unsafe_allow_html=True)

            # Build audit dataframe
            audit_df = comp[["Lot_ID", "Component_ID", "Decision", "A_score", "B_score"]].copy()
            audit_df.insert(0, "Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            audit_df.insert(1, "System_Version", "BurnInSystem v2.0")

            if "survival_prob" in comp.columns:
                audit_df["Survival_Prob"] = comp["survival_prob"]
            if "Reason_Codes" in comp.columns:
                audit_df["Justification"] = comp["Reason_Codes"]

            st.dataframe(audit_df, use_container_width=True, hide_index=True, height=500)

            # Download button
            csv = audit_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Audit Trail (CSV)",
                data=csv,
                file_name=f"burn_in_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("📤 Upload test data to generate an audit trail")
