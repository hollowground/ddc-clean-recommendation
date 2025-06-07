import pickle
import pandas as pd

# Load new data and the trained pipeline
with open("cleaning_time_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

new_data = pd.read_csv("new_cleaning_data.csv")  # Must match training column names
predictions = pipeline.predict(new_data)
print(predictions)
