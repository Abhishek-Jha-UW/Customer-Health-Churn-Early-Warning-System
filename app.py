import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
import plotly.express as px

# Page Config
st.set_page_config(page_title="Customer Health Dashboard", layout="wide")

st.title("🛡️ Customer Health & Churn Early Warning System")
st.markdown("Predicting churn risk using RFM metrics and XGBoost.")

# Load Model and Scaler
# Ensure these files (churn_model.json and scaler.pkl) are in the same directory as app.py or specify their path
try:
    model = XGBClassifier()
    model.load_model("churn_model.json")
    scaler = joblib.load("scaler.pkl")
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.info("Please ensure 'churn_model.json' and 'scaler.pkl' are in the correct directory. You might need to run model.py first.")
    st.stop() # Stop the app if model/scaler can't be loaded

# Sidebar for Input
st.sidebar.header("User Input Features")
recency = st.sidebar.slider("Recency (Days since last purchase)", 1, 100, 30)
frequency = st.sidebar.slider("Frequency (Total purchases)", 1, 50, 5)
monetary = st.sidebar.number_input("Monetary Value ($)", 10.0, 10000.0, 500.0)
tenure = st.sidebar.slider("Tenure (Days as customer)", 1, 1000, 100)

# Prepare features for prediction
# Create a DataFrame with the same column names as used during training
raw_features = pd.DataFrame([[
    recency,
    frequency,
    monetary,
    tenure
]], columns=['recency', 'frequency', 'monetary', 'tenure'])

# Scale the raw features using the loaded scaler
scaled_features = scaler.transform(raw_features)

# Prediction Logic
prediction_proba = model.predict_proba(scaled_features)[0][1]
risk_level = "High" if prediction_proba > 0.6 else "Medium" if prediction_proba > 0.3 else "Low"

# Display Results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Health Metrics")
    st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
    st.metric(label="Risk Level", value=risk_level)

with col2:
    st.subheader("Risk Explanation")
    # Example of simple logic for explanation based on features or probability
    if prediction_proba > 0.6:
        st.error("❗ This customer has a high churn probability and requires immediate attention.")
    elif prediction_proba > 0.3:
        st.warning("⚠️ This customer shows signs of potential churn. Monitor closely.")
    else:
        st.success("✅ This customer has a low churn probability and is likely healthy.")
    
    # Further explanations based on input features can be added here
    st.write(f"Recency: {recency} days, Frequency: {frequency}, Monetary: ${monetary:.2f}, Tenure: {tenure} days.")


# Simple Chart for Segment Overview
st.divider()
st.subheader("Customer Base Segmentation (Illustrative)")
chart_data = pd.DataFrame({
    'Segment': ['Champions', 'At Risk', 'Lost'],
    'Count': [450, 150, 400]
})
fig = px.bar(chart_data, x='Segment', y='Count', color='Segment', title="Distribution by Customer Segment")
st.plotly_chart(fig)
