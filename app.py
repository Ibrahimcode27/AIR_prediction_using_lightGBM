from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np

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

        # Handle edge case for highest marks
        if marks == 720:
            return {"predicted_rank": 1}

        # Prepare input for prediction
        features = np.array([[marks, year]])
        predicted_rank = model.predict(features)

        return {"predicted_rank": round(predicted_rank[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
