from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from trained_models.PredictionModel import PredictionModel
from PredictionService import generate_prediction

# Define the input data model using Pydantic for input validation
class PredictionInput(BaseModel):
    RowNumber: int
    CustomerId: int
    Surname: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/preprediction/")
async def predict_churn(request: PredictionInput):
    # Convert input data to a dictionary
    input_data = request.dict()
    
    try:
        # Generate prediction
        prediction = generate_prediction(input_data)
        
        # Assuming the prediction returns a float representing the churn probability
        num = np.float32(prediction)
        
        # Determine the result
        result = 'Customer will likely churn.' if float(num) > 0.5 else 'Customer is likely to unchurn11.'

        # Return structured response
        return {
            "probability": float(num),
            "prediction": result
        }
    
    except ValueError as ve:
        # Specific exception for value errors (like input validation errors)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # General exception handling for unexpected errors
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")