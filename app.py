import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import plotly.express as px

# Page Config
st.set_page_config(page_title="Customer Health Dashboard", layout="wide")

st.title("🛡️ Customer Health & Churn Early Warning System")

# --- LOAD ASSETS ---
@st.cache_resource # Use cache so it doesn't reload on every click
def load_assets():
    model = XGBClassifier()
    model.load_model("churn_model.json")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("Missing model.json or scaler.pkl. Run model.py first!")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Customer Metrics")
recency = st.sidebar.slider("Recency (Days)", 1, 100, 30)
frequency = st.sidebar.slider("Frequency (Orders)", 1, 50, 5)
monetary = st.sidebar.number_input("Monetary ($)", 10, 10000, 500)
tenure = st.sidebar.slider("Tenure (Days)", 1, 1000, 100)

# --- PREDICTION LOGIC ---
# 1. Create DataFrame
input_data = pd.DataFrame([[recency, frequency, monetary, tenure]], 
                          columns=['recency', 'frequency', 'monetary', 'tenure'])

# 2. SCALE THE DATA (Crucial step!)
input_scaled = scaler.transform(input_data)

# 3. Predict
prob = model.predict_proba(input_scaled)[0][1]
risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

# --- DISPLAY ---
col1, col2 = st.columns(2)
with col1:
    st.metric("Churn Probability", f"{prob:.1%}")
    st.progress(prob)

with col2:
    st.metric("Risk Level", risk)
    if risk == "High":
        st.error("Action Required: Immediate Outreach")
    elif risk == "Medium":
        st.warning("Action Recommended: Nurture Campaign")
    else:
        st.success("Customer Status: Healthy")

st.divider()
st.info("Note: Predictions are based on a pre-trained XGBoost model and scaled via StandardScaler.")
