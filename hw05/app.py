#!/usr/bin/env python3
"""
FastAPI application for serving the lead conversion prediction model.
"""

import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# Define the request model for input validation
class LeadData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float
    
    class Config:
        schema_extra = {
            "example": {
                "lead_source": "paid_ads",
                "number_of_courses_viewed": 2,
                "annual_income": 79276.0
            }
        }


# Define the response model
class PredictionResponse(BaseModel):
    probability: float
    probability_percentage: float
    prediction: str
    
    class Config:
        schema_extra = {
            "example": {
                "probability": 0.5336,
                "probability_percentage": 53.36,
                "prediction": "likely to convert"
            }
        }


# Initialize FastAPI app
app = FastAPI(
    title="Lead Conversion Prediction API",
    description="API for predicting lead conversion probability",
    version="1.0.0"
)

# Global variables for model and vectorizer
model = None
dv = None


def load_model(model_path: str = "/code/pipeline_v2.bin"):
    """Load the trained model and vectorizer from pickle file."""
    global model, dv
    try:
        with open(model_path, 'rb') as f:
            dv, model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Model file {model_path} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    load_model()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Lead Conversion Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None and dv is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_conversion(lead_data: LeadData):
    """
    Predict the probability of lead conversion.
    
    Args:
        lead_data: Lead information (source, courses viewed, income)
        
    Returns:
        Prediction response with probability and classification
    """
    if model is None or dv is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )
    
    try:
        # Convert input to dictionary
        lead_dict = lead_data.dict()
        
        # Transform the data using the loaded vectorizer
        X = dv.transform([lead_dict])
        
        # Get probability prediction
        probability = model.predict_proba(X)[0, 1]
        
        # Determine prediction label (you can adjust threshold as needed)
        threshold = 0.5
        if probability >= threshold:
            prediction_label = "likely to convert"
        else:
            prediction_label = "unlikely to convert"
        
        return PredictionResponse(
            probability=round(probability, 4),
            probability_percentage=round(probability * 100, 2),
            prediction=prediction_label
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        ) from e


@app.post("/predict_batch")
async def predict_conversion_batch(leads: list[LeadData]):
    """
    Predict conversion probability for multiple leads.
    
    Args:
        leads: List of lead data
        
    Returns:
        List of predictions
    """
    if model is None or dv is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )
    
    try:
        results = []
        
        for lead_data in leads:
            # Convert input to dictionary
            lead_dict = lead_data.dict()
            
            # Transform the data using the loaded vectorizer
            X = dv.transform([lead_dict])
            
            # Get probability prediction
            probability = model.predict_proba(X)[0, 1]
            
            # Determine prediction label
            threshold = 0.5
            if probability >= threshold:
                prediction_label = "likely to convert"
            else:
                prediction_label = "unlikely to convert"
            
            results.append({
                "lead_data": lead_dict,
                "probability": round(probability, 4),
                "probability_percentage": round(probability * 100, 2),
                "prediction": prediction_label
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making batch prediction: {str(e)}"
        ) from e


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
