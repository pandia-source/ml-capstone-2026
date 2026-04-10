
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
from typing import Optional, List
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('feature_names.txt', 'r') as f:
    feature_names = f.read().strip().split(',')

print(f"✓ Model loaded! Features: {len(feature_names)}")

# Create FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Professional ML API - Predict Titanic passenger survival with 82% accuracy",
    version="2.0.0"
)

# Enable CORS (for web requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Data Models
# ============================================
class PassengerData(BaseModel):
    """Passenger features for prediction"""
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1, 2, or 3)")
    Age: float = Field(..., ge=0, le=100, description="Age in years")
    Fare: float = Field(..., ge=0, description="Ticket fare in pounds")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses")
    Parch: int = Field(..., ge=0, description="Number of parents/children")
    FamilySize: int = Field(..., ge=1, description="Total family size")
    IsAlone: int = Field(..., ge=0, le=1, description="Traveling alone (0/1)")
    Sex_encoded: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    Embarked_Q: int = Field(..., ge=0, le=1, description="Embarked from Queenstown")
    Embarked_S: int = Field(..., ge=0, le=1, description="Embarked from Southampton")
    AgeGroup_Teen: int = Field(..., ge=0, le=1, description="Age 13-19")
    AgeGroup_Adult: int = Field(..., ge=0, le=1, description="Age 20-39")
    AgeGroup_Middle: int = Field(..., ge=0, le=1, description="Age 40-59")
    AgeGroup_Senior: int = Field(..., ge=0, le=1, description="Age 60+")

class PredictionResponse(BaseModel):
    """Response from prediction"""
    success: bool
    prediction: int
    prediction_text: str
    probability_did_not_survive: float
    probability_survived: float
    confidence: float
    confidence_percent: str

# ============================================
# HOME ENDPOINT
# ============================================
@app.get("/")
def home():
    """Home endpoint - returns API information"""
    return {
        "message": "Titanic Survival Prediction API v2.0",
        "model": "Logistic Regression",
        "accuracy": "82%",
        "version": "2.0.0",
        "status": "PRODUCTION",
        "endpoints": {
            "home": "GET /",
            "predict": "POST /predict (single prediction)",
            "predict-batch": "POST /predict-batch (multiple predictions)",
            "metrics": "GET /metrics (model metrics)",
            "health": "GET /health",
            "docs": "GET /docs (interactive documentation)"
        }
    }

# ============================================
# PREDICTION ENDPOINT
# ============================================
@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerData):
    """
    Predict if a passenger survived the Titanic
    
    ### Input:
    - Passenger features (14 required)
    
    ### Output:
    - Prediction (0 or 1)
    - Probability of each class
    - Confidence percentage
    """
    try:
        logger.info(f"Prediction request received for passenger class {passenger.Pclass}")
        
        # Extract features in correct order
        features = [
            passenger.Pclass,
            passenger.Age,
            passenger.Fare,
            passenger.SibSp,
            passenger.Parch,
            passenger.FamilySize,
            passenger.IsAlone,
            passenger.Sex_encoded,
            passenger.Embarked_Q,
            passenger.Embarked_S,
            passenger.AgeGroup_Teen,
            passenger.AgeGroup_Adult,
            passenger.AgeGroup_Middle,
            passenger.AgeGroup_Senior
        ]
        
        # Reshape for model
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        
        logger.info(f"Prediction: {prediction}, Confidence: {max(probabilities):.2%}")
        
        # Return results
        return {
            "success": True,
            "prediction": int(prediction),
            "prediction_text": "Survived ✓" if prediction == 1 else "Did Not Survive ✗",
            "probability_did_not_survive": round(float(probabilities[0]), 4),
            "probability_survived": round(float(probabilities[1]), 4),
            "confidence": round(float(max(probabilities)), 4),
            "confidence_percent": f"{max(probabilities)*100:.1f}%"
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# ============================================
# BATCH PREDICTION ENDPOINT
# ============================================
class BatchPassengers(BaseModel):
    """Multiple passengers for batch prediction"""
    passengers: List[PassengerData]

@app.post("/predict-batch")
def predict_batch(batch: BatchPassengers):
    """Predict for multiple passengers at once (faster than individual requests)"""
    try:
        logger.info(f"Batch prediction for {len(batch.passengers)} passengers")
        
        results = []
        
        for idx, passenger in enumerate(batch.passengers):
            features = [
                passenger.Pclass, passenger.Age, passenger.Fare,
                passenger.SibSp, passenger.Parch, passenger.FamilySize,
                passenger.IsAlone, passenger.Sex_encoded,
                passenger.Embarked_Q, passenger.Embarked_S,
                passenger.AgeGroup_Teen, passenger.AgeGroup_Adult,
                passenger.AgeGroup_Middle, passenger.AgeGroup_Senior
            ]
            
            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            probabilities = model.predict_proba(features_array)[0]
            
            results.append({
                "passenger_id": idx + 1,
                "prediction": int(prediction),
                "prediction_text": "Survived ✓" if prediction == 1 else "Did Not Survive ✗",
                "confidence": round(float(max(probabilities)), 4),
                "confidence_percent": f"{max(probabilities)*100:.1f}%"
            })
        
        logger.info(f"Batch completed: {len(results)} predictions")
        
        return {
            "success": True,
            "total_passengers": len(batch.passengers),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# ============================================
# MODEL METRICS ENDPOINT
# ============================================
@app.get("/metrics")
def get_metrics():
    """Get model performance metrics"""
    return {
        "model_name": "Logistic Regression",
        "model_version": "1.0",
        "test_accuracy": 0.82,
        "training_accuracy": 0.80,
        "precision": 0.82,
        "recall": 0.75,
        "f1_score": 0.78,
        "auc_roc": 0.84,
        "training_data_size": 712,
        "test_data_size": 179,
        "total_features": 14,
        "features": feature_names,
        "training_date": "2026-04-09",
        "deployment_date": "2026-04-09",
        "status": "PRODUCTION"
    }

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================
@app.get("/health")
def health():
    """Check if API is running and healthy"""
    return {
        "status": "healthy ✓",
        "model": "Logistic Regression",
        "accuracy": "82%",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# ROOT CAUSE ANALYSIS ENDPOINT
# ============================================
@app.post("/explain")
def explain_prediction(passenger: PassengerData):
    """Explain why model made a prediction (feature importance)"""
    try:
        features = [
            passenger.Pclass, passenger.Age, passenger.Fare,
            passenger.SibSp, passenger.Parch, passenger.FamilySize,
            passenger.IsAlone, passenger.Sex_encoded,
            passenger.Embarked_Q, passenger.Embarked_S,
            passenger.AgeGroup_Teen, passenger.AgeGroup_Adult,
            passenger.AgeGroup_Middle, passenger.AgeGroup_Senior
        ]
        
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        
        # Feature importance (coefficients)
        coefficients = model.coef_[0]
        importance = []
        
        for feature_name, coeff in zip(feature_names, coefficients):
            importance.append({
                "feature": feature_name,
                "coefficient": round(float(coeff), 4),
                "direction": "increases survival" if coeff > 0 else "decreases survival",
                "magnitude": abs(float(coeff))
            })
        
        importance.sort(key=lambda x: x["magnitude"], reverse=True)
        
        return {
            "prediction": int(prediction),
            "prediction_text": "Survived ✓" if prediction == 1 else "Did Not Survive ✗",
            "confidence": round(float(max(probabilities)), 4),
            "top_factors": importance[:5],
            "explanation": "Top factors influencing this prediction"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("FASTAPI v2.0 - PRODUCTION API")
    print("="*60)
    print("API running on: http://localhost:8080")
    print("Docs: http://localhost:8080/docs")
    print("Metrics: http://localhost:8080/metrics")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)

