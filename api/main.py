from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib
import os
from typing import Dict, Any
import logging

from model import DemandForecastingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Demand Forecasting API",
    description="API for predicting sales based on store, item, and date",
    version="1.0.0"
)

# Load model
model_path = os.environ.get("MODEL_PATH", "/app/models/demand_forecasting_model.joblib")
features_path = os.environ.get("FEATURES_PATH", "/app/models/features.joblib")

# Initialize model service
model_service = DemandForecastingModel(model_path, features_path)

# Define request and response models
class PredictionRequest(BaseModel):
    date: str
    store: int
    item: int

class PredictionResponse(BaseModel):
    prediction: float 
    store: int
    item: int
    date: str

class StatusResponse(BaseModel):
    status: str
    model_version: str

@app.on_event("startup")
async def startup_event():
    """Run when the API starts"""
    logger.info("Loading model and starting API")
    # Load model if not already loaded
    model_service.load_model()
    logger.info("API started successfully")

@app.get("/status", response_model=StatusResponse)
async def status():
    """Check if API is running"""
    return {
        "status": "ok",
        "model_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction based on date, store, and item
    """
    try:
        # Make prediction
        prediction = model_service.predict(
            request.date,
            request.store,
            request.item
        )
        
        return {
            "prediction": prediction,
            "store": request.store,
            "item": request.item,
            "date": request.date
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


    