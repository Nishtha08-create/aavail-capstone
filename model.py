import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def create_features(df, lag=7):
    df = df.copy()
    for i in range(1, lag + 1):
        df[f"lag_{i}"] = df["revenue"].shift(i)
    df = df.dropna().reset_index(drop=True)
    return df

def train_models(df, test_mode=False):
    """Trains baseline (Linear) and final (Random Forest) models, comparing metrics."""
    featured_df = create_features(df)
    X = featured_df[[c for c in featured_df.columns if c.startswith("lag_")]]
    y = featured_df["revenue"]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Baseline
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    b_pred = baseline.predict(X_test)
    b_rmse = np.sqrt(mean_squared_error(y_test, b_pred))
    
    # Final Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    prefix = "test_" if test_mode else ""
    model_path = os.path.join(MODEL_DIR, f"{prefix}rf_model.joblib")
    joblib.dump(rf, model_path)
    
    return {"baseline_rmse": b_rmse, "rf_rmse": rf_rmse, "model_path": model_path}

def predict(data_input, test_mode=False):
    prefix = "test_" if test_mode else ""
    model_path = os.path.join(MODEL_DIR, f"{prefix}rf_model.joblib")
    
    if not os.path.exists(model_path):
        from ingest import load_data, aggregate_daily
        df = aggregate_daily(load_data())
        train_models(df, test_mode=test_mode)
        
    model = joblib.load(model_path)
    features = np.array(data_input).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]