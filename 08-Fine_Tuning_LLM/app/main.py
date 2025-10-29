import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict as model_predict

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    API endpoint to make predictions.
    """
    prediction = model_predict(request.text)
    return PredictionResponse(prediction=prediction)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fine-Tuned Model API"}
