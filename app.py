import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import plotly.express as px

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Customer Health Early Warning System",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CONSTANTS ---------------- #
REQUIRED_COLS = ['recency', 'frequency', 'monetary', 'tenure']
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.3

# ---------------- LOAD MODEL ASSETS ---------------- #
@st.cache_resource
def load_assets():
    try:
        model = XGBClassifier()
        model.load_model("churn_model.json")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        return None, None

model, scaler = load_assets()

if model is None or scaler is None:
    st.warning("⚠️ Ensure 'churn_model.json' and 'scaler.pkl' exist in the root directory.")
    st.stop()

# ---------------- HELPER FUNCTIONS ---------------- #
def validate_input(df):
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    return missing

def add_predictions(df):
    X_scaled = scaler.transform(df[REQUIRED_COLS])
    df['Churn_Prob'] = model.predict_proba(X_scaled)[:, 1]

    df['Risk_Level'] = pd.cut(
        df['Churn_Prob'],
        bins=[-1, MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 1],
        labels=['🟢 Healthy', '🟡 Medium Risk', '🔴 High Risk']
    )

    return df

def get_sample_data():
    return pd.DataFrame({
        'customer_id': [1001, 1002, 1003, 1004, 1005],
        'recency': [5, 82, 45, 98, 12],
        'frequency': [25, 2, 12, 1, 18],
        'monetary': [4500.0, 150.0, 1100.0, 45.0, 3200.0],
        'tenure': [450, 30, 200, 15, 380]
    })

# ---------------- HEADER ---------------- #
st.title("🛡️ Customer Health & Churn Early Warning System")
st.markdown("""
This tool uses an **XGBoost Machine Learning model** trained on RFM metrics  
to proactively detect customer churn risk.
""")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("🛠️ Analysis Controls")
mode = st.sidebar.radio("Choose Input Mode:", ["Bulk Analysis (CSV)", "Single Customer Simulator"])

# ==========================================================
# MODE 1: BULK ANALYSIS
# ==========================================================
if mode == "Bulk Analysis (CSV)":

    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload customer CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        st.info("Showing sample data. Upload your CSV for real analysis.")
        df = get_sample_data()

    missing_cols = validate_input(df)

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    df = add_predictions(df)

    # ---------- KPI METRICS ----------
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", len(df))
    m2.metric("High Risk Accounts", (df['Churn_Prob'] > HIGH_RISK_THRESHOLD).sum())
    m3.metric("Avg. Churn Probability", f"{df['Churn_Prob'].mean():.1%}")

    st.divider()

    # ---------- VISUALS ----------
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Customer Health Map")
        fig = px.scatter(
            df,
            x="recency",
            y="monetary",
            color="Risk_Level",
            size="frequency",
            hover_data=['customer_id'],
            labels={
                "recency": "Days Since Last Purchase",
                "monetary": "Lifetime Value ($)"
            },
            color_discrete_map={
                '🔴 High Risk': 'red',
                '🟡 Medium Risk': 'orange',
                '🟢 Healthy': 'green'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Risk Distribution")
        pie = px.pie(
            df,
            names='Risk_Level',
            color='Risk_Level',
            color_discrete_map={
                '🔴 High Risk': 'red',
                '🟡 Medium Risk': 'orange',
                '🟢 Healthy': 'green'
            }
        )
        st.plotly_chart(pie, use_container_width=True)

    # ---------- TABLE ----------
    st.subheader("Detailed Risk Table")
    st.dataframe(
        df.sort_values("Churn_Prob", ascending=False),
        use_container_width=True
    )

    # ---------- DOWNLOAD ----------
    st.download_button(
        "📩 Download Analysis Results",
        df.to_csv(index=False).encode("utf-8"),
        "churn_analysis.csv",
        "text/csv"
    )

# ==========================================================
# MODE 2: SINGLE CUSTOMER SIMULATOR
# ==========================================================
else:

    st.subheader("Single Customer 'What-If' Simulator")

    c1, c2 = st.columns(2)

    with c1:
        s_recency = st.slider("Recency (Days since last activity)", 1, 120, 30)
        s_frequency = st.slider("Frequency (Total orders)", 1, 50, 5)

    with c2:
        s_monetary = st.number_input("Monetary Value ($)", 10, 10000, 500)
        s_tenure = st.slider("Tenure (Days as customer)", 1, 1000, 100)

    single_input = pd.DataFrame(
        [[s_recency, s_frequency, s_monetary, s_tenure]],
        columns=REQUIRED_COLS
    )

    scaled_input = scaler.transform(single_input)
    prob = model.predict_proba(scaled_input)[0][1]

    st.divider()

    r1, r2 = st.columns(2)

    with r1:
        st.metric("Predicted Churn Risk", f"{prob:.1%}")

        if prob > HIGH_RISK_THRESHOLD:
            st.error("🔴 High Risk – Immediate retention action recommended")
        elif prob > MEDIUM_RISK_THRESHOLD:
            st.warning("🟡 Medium Risk – Consider targeted nurture campaign")
        else:
            st.success("🟢 Healthy – Maintain engagement strategy")

    with r2:
        st.write("### Risk Drivers (Heuristic Insight)")
        if s_recency > 60:
            st.write("• High inactivity driving churn risk")
        if s_frequency < 3:
            st.write("• Low purchase frequency")
        if s_monetary < 200:
            st.write("• Low lifetime value customer")
