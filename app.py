import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import plotly.express as px

# Page Config
st.set_page_config(page_title="Customer Health Dashboard", layout="wide")

st.title("🛡️ Customer Health & Churn Early Warning System")
st.markdown("Predicting churn risk using RFM metrics and XGBoost.")

# Load Model
model = XGBClassifier()
try:
    model.load_model("churn_model.json")
except:
    st.error("Please run model.py first to generate the churn_model.json file!")

# Sidebar for Input
st.sidebar.header("User Input Features")
recency = st.sidebar.slider("Recency (Days since last purchase)", 1, 100, 30)
frequency = st.sidebar.slider("Frequency (Total purchases)", 1, 50, 5)
monetary = st.sidebar.number_input("Monetary Value ($)", 10, 10000, 500)
tenure = st.sidebar.slider("Tenure (Days as customer)", 1, 1000, 100)

# Prediction Logic
features = pd.DataFrame([[recency, frequency, monetary, tenure]], 
                        columns=['recency', 'frequency', 'monetary', 'tenure'])

prediction_prob = model.predict_proba(features)[0][1]
risk_level = "High" if prediction_prob > 0.6 else "Medium" if prediction_prob > 0.3 else "Low"

# Display Results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Health Metrics")
    st.metric(label="Churn Probability", value=f"{prediction_prob:.2%}")
    st.metric(label="Risk Level", value=risk_level)

with col2:
    st.subheader("Risk Explanation")
    # Simple logic for explanation
    if recency > 50:
        st.warning("⚠️ High Recency is the primary driver for this risk score.")
    else:
        st.success("✅ Customer is still active within a healthy window.")

# Simple Chart
st.divider()
st.subheader("Segment Overview (Sample Data)")
chart_data = pd.DataFrame({
    'Segment': ['Champions', 'At Risk', 'Lost'],
    'Count': [450, 150, 400]
})
fig = px.bar(chart_data, x='Segment', y='Count', color='Segment', title="Customer Base Segmentation")
st.plotly_chart(fig)
