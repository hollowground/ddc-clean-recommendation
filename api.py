from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("cleaning_time_predictor.pkl")

app = FastAPI()


# Define the request schema
class CleaningInput(BaseModel):
    last_people_traffic: int
    last_inspection_score: int
    prev_people_traffic: int
    prev_inspection_score: int
    time_between_cleaning: float
    people_traffic_diff: int


@app.post("/predict")
def predict_cleaning_time(data: CleaningInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_time_between_cleaning": round(prediction, 0)}
