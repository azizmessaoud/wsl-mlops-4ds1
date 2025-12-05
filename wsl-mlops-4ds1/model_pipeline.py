import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

MODEL_PATH = 'model.joblib'

def prepare_data():
    X = np.array([[60, 3], [80, 4], [100, 3], [120, 4], [150, 5], [200, 5]])
    y = np.array([300000, 400000, 500000, 600000, 750000, 900000])
    return X, y

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_model():
    return joblib.load(MODEL_PATH)

