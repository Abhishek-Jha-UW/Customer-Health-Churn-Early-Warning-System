import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

def generate_data():
    np.random.seed(42)
    data = {
        'customer_id': range(1, 1001),
        'recency': np.random.randint(1, 100, 1000),
        'frequency': np.random.randint(1, 20, 1000),
        'monetary': np.random.uniform(50, 5000, 1000),
        'tenure': np.random.randint(10, 500, 1000)
    }
    df = pd.DataFrame(data)
    df['churn'] = (df['recency'] > 60).astype(int)
    return df

def train_model(df):
    X = df[['recency', 'frequency', 'monetary', 'tenure']]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Optional: Remove scaler if only using XGBoost
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    
    # Save artifacts
    model.save_model("churn_model.json")
    joblib.dump(scaler, "scaler.pkl")
    
    print("Model saved as churn_model.json")
    print("Scaler saved as scaler.pkl")
    
    return model, scaler

if __name__ == "__main__":
    df = generate_data()
    train_model(df)
