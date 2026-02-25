import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Churn Early Warning System", layout="wide")

# --- CORE LOGIC: TRAINING & ASSETS ---
def generate_synthetic_data(rows=1000):
    """Generates a robust synthetic dataset for training/demo."""
    np.random.seed(42)
    data = {
        'customer_id': range(1, rows + 1),
        'recency': np.random.randint(1, 100, rows),
        'frequency': np.random.randint(1, 20, rows),
        'monetary': np.random.uniform(50, 5000, rows),
        'tenure': np.random.randint(10, 500, rows)
    }
    df = pd.DataFrame(data)
    # Target: Higher recency and lower frequency = Higher Churn probability
    df['churn'] = ((df['recency'] * 0.7) - (df['frequency'] * 2) > 20).astype(int)
    return df

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_resource
def get_trained_assets():
    """Trains the model with a 80/20 split and returns the accuracy score."""
    df = generate_synthetic_data()
    X = df[['recency', 'frequency', 'monetary', 'tenure']]
    y = df['churn']
    
    # 1. Split the data (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Scale based ONLY on the training data to prevent "data leakage"
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train the model
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train_scaled, y_train)
    
    # 4. Evaluate performance
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return model, scaler, acc

# Note: We now unpack THREE variables
model, scaler, model_accuracy = get_trained_assets()

# --- UI HEADER ---
st.title("🛡️ Customer Health & Churn Early Warning System")
st.markdown("This tool uses XGBoost to predict churn risk based on **RFM** (Recency, Frequency, Monetary) metrics.")

# --- SIDEBAR: DOWNLOAD TEMPLATE & UPLOAD ---
st.sidebar.header("1. Get the Template")
template_df = pd.DataFrame(columns=['customer_id', 'recency', 'frequency', 'monetary', 'tenure'])
template_df.loc[0] = [123, 10, 5, 500.0, 100]
st.sidebar.write(f"📈 Model Training Accuracy: {accuracy:.1%}")

# Download Template as Excel
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    template_df.to_excel(writer, index=False, sheet_name='Sheet1')
st.sidebar.download_button(
    label="📥 Download Excel Template",
    data=buffer.getvalue(),
    file_name="churn_template.xlsx",
    mime="application/vnd.ms-excel"
)

st.sidebar.divider()
st.sidebar.header("2. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# --- DATA PROCESSING ---
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)
    st.success("Custom data loaded!")
else:
    st.info("💡 Currently showing **Synthetic Demo Data**. Upload your own file in the sidebar to switch.")
    df_input = generate_synthetic_data(rows=100)

# Validate Columns
required = ['recency', 'frequency', 'monetary', 'tenure']
if all(col in df_input.columns for col in required):
    
    # Inference
    X_custom = df_input[required]
    X_custom_scaled = scaler.transform(X_custom)
    df_input['Churn_Probability'] = model.predict_proba(X_custom_scaled)[:, 1]
    df_input['Risk_Level'] = df_input['Churn_Probability'].apply(
        lambda x: '🔴 High' if x > 0.7 else ('🟡 Medium' if x > 0.3 else '🟢 Low')
    )

    # --- DASHBOARD ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Analyzed", len(df_input))
    m2.metric("At-Risk Customers", len(df_input[df_input['Churn_Probability'] > 0.5]))
    m3.metric("Avg Risk Score", f"{df_input['Churn_Probability'].mean():.1%}")

    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Customer Health Segments")
        fig = px.scatter(df_input, x="recency", y="monetary", color="Risk_Level",
                         size="frequency", hover_data=['customer_id'],
                         color_discrete_map={'🔴 High': '#ef553b', '🟡 Medium': '#fecb52', '🟢 Low': '#00cc96'})
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("Risk Distribution")
        fig_pie = px.pie(df_input, names='Risk_Level', 
                         color='Risk_Level', color_discrete_map={'🔴 High': '#ef553b', '🟡 Medium': '#fecb52', '🟢 Low': '#00cc96'})
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Detailed Prediction Results")
    st.dataframe(df_input.sort_values('Churn_Probability', ascending=False), use_container_width=True)

    # Download Results
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
        df_input.to_excel(writer, index=False)
    st.download_button("📩 Download Full Results (Excel)", output_buffer.getvalue(), "churn_predictions.xlsx")

else:
    st.error(f"Error: Your file must contain these columns: {required}")
