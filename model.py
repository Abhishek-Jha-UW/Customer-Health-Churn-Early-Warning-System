import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import datetime as dt

def generate_data():
    # Create synthetic customer data
    np.random.seed(42)
    data = {
        'customer_id': range(1, 1001),
        'recency': np.random.randint(1, 100, 1000),
        'frequency': np.random.randint(1, 20, 1000),
        'monetary': np.random.uniform(50, 5000, 1000),
        'tenure': np.random.randint(10, 500, 1000)
    }
    df = pd.DataFrame(data)
    # Target: 1 if recency > 60 days (Churned), else 0
    df['churn'] = (df['recency'] > 60).astype(int)
    return df

def train_model(df):
    X = df[['recency', 'frequency', 'monetary', 'tenure']]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # Save the model
    model.save_model("churn_model.json")
    print("Model trained and saved as churn_model.json")
    return model

if __name__ == "__main__":
    df = generate_data()
    train_model(df)
