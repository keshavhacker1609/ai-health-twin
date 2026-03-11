from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Initialize the App
app = FastAPI(title="AI Health Twin API")

# 2. Allow frontend to communicate with backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the pre-trained AI Model
model = joblib.load('diabetes_risk_model.pkl')

# 4. Define the Data Structure we expect from the user (Pydantic)
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigree: float
    Age: int

# 5. Create the Prediction Endpoint
@app.post("/api/predict/diabetes")
async def predict_diabetes_risk(data: PatientData):
    # Convert incoming data to a Pandas DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Ensure column order matches exactly how the model was trained
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age']
    input_df = input_df[columns]
    
    # Get the probability of class 1 (Diabetes)
    probability = model.predict_proba(input_df)[0][1]
    
    # Convert to percentage
    risk_percentage = round(float(probability * 100), 2)
    
    # Return the simulated risk
    return {
        "status": "success",
        "risk_score_percentage": risk_percentage,
        "message": f"Your current simulated risk for Type 2 Diabetes is {risk_percentage}%."
    }