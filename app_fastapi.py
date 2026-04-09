
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from typing import Optional

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
    description="Predict if a Titanic passenger survived using Logistic Regression (82% accuracy)",
    version="1.0.0"
)

# ============================================
# Define Data Model (Automatic Validation!)
# ============================================
class PassengerData(BaseModel):
    """Passenger features for prediction"""
    Pclass: int
    Age: float
    Fare: float
    SibSp: int
    Parch: int
    FamilySize: int
    IsAlone: int
    Sex_encoded: int
    Embarked_Q: int
    Embarked_S: int
    AgeGroup_Teen: int
    AgeGroup_Adult: int
    AgeGroup_Middle: int
    AgeGroup_Senior: int

# ============================================
# HOME ENDPOINT
# ============================================
@app.get("/")
def home():
    """Home endpoint - returns API information"""
    return {
        "message": "Titanic Survival Prediction API",
        "model": "Logistic Regression",
        "accuracy": "82%",
        "endpoints": {
            "home": "GET /",
            "predict": "POST /predict",
            "health": "GET /health",
            "docs": "GET /docs (interactive documentation)"
        },
        "example_passenger": {
            "Pclass": 1,
            "Age": 25,
            "Fare": 100,
            "SibSp": 0,
            "Parch": 0,
            "FamilySize": 1,
            "IsAlone": 1,
            "Sex_encoded": 0,
            "Embarked_Q": 0,
            "Embarked_S": 1,
            "AgeGroup_Teen": 0,
            "AgeGroup_Adult": 1,
            "AgeGroup_Middle": 0,
            "AgeGroup_Senior": 0
        }
    }

# ============================================
# PREDICTION ENDPOINT (With Data Validation!)
# ============================================
@app.post("/predict")
def predict(passenger: PassengerData):
    """
    Predict if a passenger survived the Titanic
    
    ### Input:
    - Passenger features (14 features required)
    
    ### Output:
    - Prediction (0 or 1)
    - Probability of each class
    - Confidence percentage
    """
    try:
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
        
        # Return results
        return {
            "success": True,
            "prediction": int(prediction),
            "prediction_text": "Survived ✓" if prediction == 1 else "Did Not Survive ✗",
            "probability_did_not_survive": round(float(probabilities[0]), 4),
            "probability_survived": round(float(probabilities[1]), 4),
            "confidence": round(float(max(probabilities)), 4),
            "confidence_percent": f"{max(probabilities)*100:.1f}%",
            "model_accuracy": "82%",
            "passenger_features": {
                "class": passenger.Pclass,
                "age": passenger.Age,
                "fare": passenger.Fare,
                "gender": "Female" if passenger.Sex_encoded == 0 else "Male"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================
@app.get("/health")
def health():
    """Check if API is running"""
    return {
        "status": "healthy ✓",
        "model": "Logistic Regression",
        "accuracy": "82%",
        "version": "1.0.0"
    }

# ============================================
# BATCH PREDICTION (NEW FEATURE!)
# ============================================
class BatchPassengers(BaseModel):
    """Multiple passengers for batch prediction"""
    passengers: list[PassengerData]

@app.post("/predict-batch")
def predict_batch(batch: BatchPassengers):
    """Predict for multiple passengers at once"""
    try:
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
                "confidence": round(float(max(probabilities)), 4)
            })
        
        return {
            "success": True,
            "total_passengers": len(batch.passengers),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("FASTAPI SERVER STARTING")
    print("="*60)
    print("API running on: http://localhost:8000")
    print("Home: http://localhost:8000/")
    print("Docs: http://localhost:8000/docs (interactive!)")
    print("ReDoc: http://localhost:8000/redoc")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
