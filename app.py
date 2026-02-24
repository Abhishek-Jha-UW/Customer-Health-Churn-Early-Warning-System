import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
import plotly.express as px

st.set_page_config(page_title="Churn Early Warning System", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = XGBClassifier()
    model.load_model("churn_model.json")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("Missing model.json or scaler.pkl. Upload them to GitHub!")
    st.stop()

st.title("🛡️ Customer Health & Churn Analysis")

# --- FILE UPLOADER SECTION ---
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV with customer metrics", type="csv")

# --- SAMPLE DATA GENERATOR ---
def get_sample_data():
    return pd.DataFrame({
        'customer_id': [101, 102, 103, 104],
        'recency': [10, 85, 45, 95],
        'frequency': [20, 2, 10, 1],
        'monetary': [5000, 50, 1200, 30],
        'tenure': [400, 20, 150, 10]
    })

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.info("Using sample data. Upload a CSV in the sidebar to analyze your own customers.")
    df = get_sample_data()

# --- PREDICTION LOGIC ---
# Ensure we only use the columns the model was trained on
feature_cols = ['recency', 'frequency', 'monetary', 'tenure']

if all(col in df.columns for col in feature_cols):
    # Scale and Predict
    scaled_data = scaler.transform(df[feature_cols])
    df['Churn_Probability'] = model.predict_proba(scaled_data)[:, 1]
    df['Risk_Level'] = df['Churn_Probability'].apply(lambda x: 'High' if x > 0.6 else ('Medium' if x > 0.3 else 'Low'))

    # --- DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("High Risk Summary")
        risk_counts = df['Risk_Level'].value_counts()
        st.write(risk_counts)
        
    with col2:
        fig = px.scatter(df, x="recency", y="monetary", color="Risk_Level",
                         size="frequency", hover_data=['customer_id'],
                         title="Customer Health Map (Recency vs Monetary)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Analysis Results")
    st.dataframe(df.style.background_gradient(subset=['Churn_Probability'], cmap='Reds'))
    
    # Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

else:
    st.error(f"CSV must contain these columns: {feature_cols}")
