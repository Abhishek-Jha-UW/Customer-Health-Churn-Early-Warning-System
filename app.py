import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Health Early Warning System",
    page_icon="🛡️",
    layout="wide"
)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    """Load the pre-trained model and scaler once and cache them."""
    try:
        model = XGBClassifier()
        model.load_model("churn_model.json")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

model, scaler = load_assets()

# --- HELPER FUNCTIONS ---
def get_sample_data():
    """Generates sample data for demonstration if no file is uploaded."""
    return pd.DataFrame({
        'customer_id': [1001, 1002, 1003, 1004, 1005],
        'recency': [5, 82, 45, 98, 12],
        'frequency': [25, 2, 12, 1, 18],
        'monetary': [4500.0, 150.0, 1100.0, 45.0, 3200.0],
        'tenure': [450, 30, 200, 15, 380]
    })

# --- UI HEADER ---
st.title("🛡️ Customer Health & Churn Early Warning System")
st.markdown("""
This tool utilizes **XGBoost Machine Learning** and **RFM Analysis** to identify at-risk customers 
within a subscription business model. 
""")

if model is None or scaler is None:
    st.warning("⚠️ Application Assets Missing: Ensure 'churn_model.json' and 'scaler.pkl' are in the root directory.")
    st.stop()

# --- SIDEBAR: NAVIGATION & UPLOAD ---
st.sidebar.header("🛠️ Analysis Controls")
mode = st.sidebar.radio("Choose Input Mode:", ["Bulk Analysis (CSV)", "Single Customer Simulator"])

# --- MODE 1: BULK ANALYSIS ---
if mode == "Bulk Analysis (CSV)":
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload customer CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        st.info("💡 Showing sample data. Upload your own CSV in the sidebar to analyze your specific metrics.")
        df = get_sample_data()

    # Feature validation
    required_cols = ['recency', 'frequency', 'monetary', 'tenure']
    if all(col in df.columns for col in required_cols):
        
        # Preprocessing & Prediction
        X_scaled = scaler.transform(df[required_cols])
        df['Churn_Prob'] = model.predict_proba(X_scaled)[:, 1]
        df['Risk_Level'] = df['Churn_Prob'].apply(
            lambda x: '🔴 High Risk' if x > 0.7 else ('🟡 Medium Risk' if x > 0.3 else '🟢 Healthy')
        )

        # Dashboard Layout
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Customers", len(df))
        m2.metric("High Risk Accounts", len(df[df['Churn_Prob'] > 0.7]))
        m3.metric("Avg. Churn Probability", f"{df['Churn_Prob'].mean():.1%}")

        st.divider()

        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("Customer Health Map")
            fig = px.scatter(
                df, x="recency", y="monetary", 
                color="Risk_Level", size="frequency",
                color_discrete_map={'🔴 High Risk': 'red', '🟡 Medium Risk': 'orange', '🟢 Healthy': 'green'},
                hover_data=['customer_id'],
                labels={"recency": "Days Since Last Purchase", "monetary": "Lifetime Value ($)"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Risk Distribution")
            dist_fig = px.pie(df, names='Risk_Level', color='Risk_Level',
                             color_discrete_map={'🔴 High Risk': 'red', '🟡 Medium Risk': 'orange', '🟢 Healthy': 'green'})
            st.plotly_chart(dist_fig, use_container_width=True)

        st.subheader("Detailed Risk Table")
        st.dataframe(df.sort_values('Churn_Prob', ascending=False), use_container_width=True)
        
        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📩 Download Analysis Results", csv, "churn_analysis.csv", "text/csv")
    else:
        st.error(f"Invalid CSV format. Please ensure columns: {required_cols}")

# --- MODE 2: SINGLE CUSTOMER SIMULATOR ---
else:
    st.subheader("Single Customer 'What-If' Simulator")
    st.write("Adjust the parameters to see how a specific customer's health score changes.")
    
    c1, c2 = st.columns(2)
    with c1:
        s_recency = st.slider("Recency (Days since last active)", 1, 120, 30)
        s_frequency = st.slider("Frequency (Total orders)", 1, 50, 5)
    with c2:
        s_monetary = st.number_input("Monetary Value ($)", 10, 10000, 500)
        s_tenure = st.slider("Tenure (Total days as customer)", 1, 1000, 100)

    # Predict single instance
    single_input = pd.DataFrame([[s_recency, s_frequency, s_monetary, s_tenure]], columns=required_cols)
    single_scaled = scaler.transform(single_input)
    prob = model.predict_proba(single_scaled)[0][1]

    st.divider()
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Individual Churn Risk", f"{prob:.1%}")
        if prob > 0.7:
            st.error("Status: High Risk - Immediate Intervention Recommended")
        elif prob > 0.3:
            st.warning("Status: Medium Risk - Nurture Campaign Suggested")
        else:
            st.success("Status: Healthy - Retain standard engagement")

    with res_col2:
        # Simple Logic explanation
        st.write("**Health Analysis:**")
        if s_recency > 60:
            st.write("❌ Risk is driven primarily by inactivity (Recency).")
        if s_frequency < 3:
            st.write("❌ Low engagement frequency increases churn probability.")
