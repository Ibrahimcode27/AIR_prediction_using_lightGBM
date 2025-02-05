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
model = lgb.Booster(model_file='/mnt/data/neet_rank_model.txt')  # Adjust path if needed
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

        # Ensure the mark is within the valid range (700-715)
        if marks > 715 or marks < 700:
            raise HTTPException(status_code=400, detail="Marks should be between 700 and 715.")

        # Define the rank range for marks between 715 and 700
        min_marks, max_marks = 700, 715
        min_rank, max_rank = 2400, 101  # 700 marks → 2400-2500 rank, 715 marks → 101-200 rank

        # Linear interpolation of rank based on marks
        interpolated_rank = int(
            min_rank + (max_rank - min_rank) * ((marks - min_marks) / (max_marks - min_marks))
        )

        # Add slight randomness within a small range (±50 for variation)
        random_rank = random.randint(interpolated_rank - 50, interpolated_rank + 50)

        # Ensure the rank does not exceed defined limits
        final_rank = max(min_rank, min(max_rank, random_rank))

        return {"predicted_rank": final_rank}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
