import pandas as pd

# Sample dataset
data = [
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 21:00:00",
        "last_people_traffic": 1,
        "last_inspection_score": 96,
        "prev_last_clean_date": "2024-04-04 17:00:00",
        "prev_people_traffic": 13,
        "prev_inspection_score": 98,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": -12
    },
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 22:00:00",
        "last_people_traffic": 15,
        "last_inspection_score": 97,
        "prev_last_clean_date": "2024-04-04 18:00:00",
        "prev_people_traffic": 23,
        "prev_inspection_score": 85,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": -8
    },
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 23:00:00",
        "last_people_traffic": 16,
        "last_inspection_score": 97,
        "prev_last_clean_date": "2024-04-04 19:00:00",
        "prev_people_traffic": 25,
        "prev_inspection_score": 93,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": -9
    },
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 13:00:00",
        "last_people_traffic": 9,
        "last_inspection_score": 99,
        "prev_last_clean_date": "2024-04-04 09:00:00",
        "prev_people_traffic": 25,
        "prev_inspection_score": 90,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": -16
    },
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 05:00:00",
        "last_people_traffic": 21,
        "last_inspection_score": 96,
        "prev_last_clean_date": "2024-04-04 01:00:00",
        "prev_people_traffic": 11,
        "prev_inspection_score": 95,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": 10
    },
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 09:00:00",
        "last_people_traffic": 16,
        "last_inspection_score": 93,
        "prev_last_clean_date": "2024-04-04 05:00:00",
        "prev_people_traffic": 1,
        "prev_inspection_score": 85,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": 15
    },
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 21:00:00",
        "last_people_traffic": 9,
        "last_inspection_score": 100,
        "prev_last_clean_date": "2024-04-04 17:00:00",
        "prev_people_traffic": 7,
        "prev_inspection_score": 85,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": 2
    },
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 03:00:00",
        "last_people_traffic": 26,
        "last_inspection_score": 99,
        "prev_last_clean_date": "2024-04-03 23:00:00",
        "prev_people_traffic": 3,
        "prev_inspection_score": 98,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": 23
    },
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 01:00:00",
        "last_people_traffic": 30,
        "last_inspection_score": 86,
        "prev_last_clean_date": "2024-04-03 21:00:00",
        "prev_people_traffic": 21,
        "prev_inspection_score": 96,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": 9
    },
    {
        "curr_date": "2024-04-05",
        "last_clean_date": "2024-04-04 07:00:00",
        "last_people_traffic": 15,
        "last_inspection_score": 98,
        "prev_last_clean_date": "2024-04-04 03:00:00",
        "prev_people_traffic": 3,
        "prev_inspection_score": 96,
        "time_between_cleaning": 4.0,
        "people_traffic_diff": 12
    }
]

# Function to compute the target (label)
def compute_target_time(row):
    score = row["last_inspection_score"]
    traffic = row["last_people_traffic"]
    base_time = row["time_between_cleaning"]

    if score >= 97 and traffic < 10:
        return round(max(base_time + 1, base_time * 1.25), 2)
    elif score >= 97 and traffic >= 10:
        return base_time
    elif 90 <= score < 97:
        if traffic < 10:
            return round(max(base_time + 1, base_time * 1.1), 2)
        else:
            return base_time
    else:
        if traffic >= 10:
            return round(max(base_time - 1, base_time * 0.75), 2)
        else:
            return base_time

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute target column
df["target_time_between_cleaning"] = df.apply(compute_target_time, axis=1)

# Optional: Drop non-feature fields like curr_date if needed
# These fields are usually not predictive and can be removed for training
features_to_drop = ["curr_date", "last_clean_date", "prev_last_clean_date"]
df_model = df.drop(columns=features_to_drop)

# Save to CSV for training
df_model.to_csv("training_dataset.csv", index=False)

# Show a preview
print("ML training dataset saved as 'training_dataset.csv'")
print(df_model.head())
