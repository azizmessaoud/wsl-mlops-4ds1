import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Generate/Load Data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Set MLflow Tracking URI to use SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Lab5_Excellence_SQLite")

print("Starting MLflow run with SQLite backend...")

# 4. Start MLflow Run
with mlflow.start_run():
    # Define Hyperparameters
    params = {
        "fit_intercept": True,
        "copy_X": True,
        "n_jobs": -1
    }
    
    # Log Parameters
    mlflow.log_params(params)
    
    # Train Model
    model = LinearRegression(**params)
    model.fit(X_train, y_train)
    
    # Make Predictions
    predictions = model.predict(X_test)
    
    # Calculate Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Log Metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    print(f"Logged metrics - MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # Log Model (and register it automatically if you want, but we'll do manual registration in UI)
    mlflow.sklearn.log_model(model, "linear_regression_model")
    print("Logged model artifact.")
    
    # Save model locally
    joblib.dump(model, "model.joblib")

print("Run complete! Check MLflow UI.")
