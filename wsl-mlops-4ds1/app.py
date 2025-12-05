from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

model = joblib.load('model.joblib')

app = FastAPI(title="House Price Predictor")

class HouseFeatures(BaseModel):
    size: float
    rooms: int

@app.get("/")
def home():
    return {"message": "MLOps API is running!"}

@app.post("/predict/")
def predict_price(features: HouseFeatures):
    X_new = np.array([[features.size, features.rooms]])
    prediction = model.predict(X_new)
    return {"predicted_price": float(prediction[0])}

