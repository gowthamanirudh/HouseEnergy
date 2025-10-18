from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Household Energy Consumption Prediction API",
    description="API for predicting household energy consumption using Random Forest model",
    version="1.0.0"
)

# Load the trained model
try:
    with open('random_forest_best_pruned.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


class PredictionInput(BaseModel):
    """Input schema for prediction"""
    features: List[float] = Field(
        ..., 
        description="List of feature values for prediction. Should match the model's expected features.",
        example=[1.5, 1.6, 1.4, 1.7, 1.5, 1.3, 1.8]
    )
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.5, 1.6, 1.4, 1.7, 1.5, 1.3, 1.8]
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    prediction: float = Field(..., description="Predicted energy consumption")
    model_info: str = Field(..., description="Model information")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Household Energy Consumption Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Predict household energy consumption based on input features.
    
    The model expects lag features (past energy consumption values).
    Ensure the features are in the correct order matching the training data.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        feature_names = [f'lag_{i+1}' for i in range(len(input_data.features))]
        input_df = pd.DataFrame([input_data.features], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return {
            "prediction": float(prediction),
            "model_info": "Random Forest Regressor (Optimized & Pruned)"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        params = model.get_params()
        return {
            "model_type": type(model).__name__,
            "n_estimators": params.get('n_estimators'),
            "max_depth": params.get('max_depth'),
            "min_samples_split": params.get('min_samples_split'),
            "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
