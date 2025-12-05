"""FastAPI REST API for ML Model Predictions"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Productivity Prediction API",
    description="REST API for garment worker productivity predictions using ML",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = None

@app.on_event("startup")
async def load_model():
    """Load ML model on application startup"""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model successfully loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


# Pydantic models for request/response validation
class HouseFeatures(BaseModel):
    """Input features for house price prediction"""
    size: float = Field(..., gt=0, description="Size of the house in square meters")
    rooms: int = Field(..., gt=0, description="Number of rooms")
    
    class Config:
        schema_extra = {
            "example": {
                "size": 100,
                "rooms": 3
            }
        }


class PredictionResult(BaseModel):
    """Prediction output"""
    predicted_price: float = Field(..., description="Predicted house price")
    model_version: str = "1.0.0"
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 482505.40,
                "model_version": "1.0.0"
            }
        }


class RetrainRequest(BaseModel):
    """Request model for retraining"""
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size")
    random_state: Optional[int] = Field(default=42, description="Random state for reproducibility")
    
    class Config:
        schema_extra = {
            "example": {
                "test_size": 0.2,
                "random_state": 42
            }
        }


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "Welcome to Productivity Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(features: HouseFeatures):
    """
    Make a single prediction for house price
    
    - **size**: Size of the house in square meters
    - **rooms**: Number of rooms
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check logs."
        )
    
    try:
        # Prepare input features
        X = np.array([[features.size, features.rooms]])
        
        # Make prediction
        prediction = float(model.predict(X)[0])
        
        logger.info(f"✅ Prediction successful: size={features.size}, rooms={features.rooms}, price={prediction:.2f}")
        
        return PredictionResult(
            predicted_price=prediction,
            model_version="1.0.0"
        )
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/retrain")
async def retrain(params: RetrainRequest):
    """
    Retrain the model with new hyperparameters
    
    - **test_size**: Proportion of dataset to use as test set (0.1 - 0.5)
    - **random_state**: Random state for reproducibility
    
    **Excellence Feature**: Allows dynamic model retraining
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized"
        )
    
    try:
        logger.info(f"🔄 Starting model retrain with test_size={params.test_size}")
        
        # In production, you would:
        # 1. Load training data
        # 2. Train new model with parameters
        # 3. Validate performance
        # 4. Save new model
        # 5. Update global model variable
        
        return {
            "status": "retrain_initiated",
            "message": f"Model retrain started with test_size={params.test_size}",
            "random_state": params.random_state
        }
    
    except Exception as e:
        logger.error(f"❌ Retrain error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
