from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np
import random

# Define request schema
class PredictionRequest(BaseModel):
    marks: float
    year: int

# Load the trained LightGBM model
model = lgb.Booster(model_file='neet_rank_model.txt')

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the NEET Rank Prediction API!"}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available."}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        marks = request.marks
        year = request.year

        # Handle cases for special conditions
        if marks > 720:
            raise HTTPException(status_code=400, detail="Marks cannot be greater than 720.")
        elif marks == 720:
            return {"predicted_rank": 1}
        elif 710 <= marks <= 719:
            # Return a random rank between 10 and 90
            random_rank = random.randint(10, 50)
            return {"predicted_rank": random_rank}
        elif 705 <= marks <= 709:
            # Return a random rank between 10 and 90
            random_rank = random.randint(50, 70)
            return {"predicted_rank": random_rank}
        
        # Prepare input for prediction
        features = np.array([[marks, year]])
        predicted_rank = model.predict(features)

        return {"predicted_rank": round(predicted_rank[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")