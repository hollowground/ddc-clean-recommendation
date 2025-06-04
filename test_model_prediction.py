import joblib
import pandas as pd

# Load model
model = joblib.load("cleaning_time_predictor.pkl")

# Example input (as DataFrame)
sample = pd.DataFrame([{
    "last_people_traffic": 12,
    "last_inspection_score": 95,
    "prev_people_traffic": 18,
    "prev_inspection_score": 90,
    "time_between_cleaning": 4.0,
    "people_traffic_diff": -6
}])

# Predict
predicted_time = model.predict(sample)
print(f"Predicted cleaning interval: {predicted_time[0]:.2f} hours")
