import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib  # for saving the model
import pickle

# Step 1: Load data
df = pd.read_csv("training_dataset.csv")

# Step 2: Separate features (X) and target (y)
X = df.drop(columns=["target_time_between_cleaning"])
y = df["target_time_between_cleaning"]

# Step 3: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 6: Save the trained model
#joblib.dump(model, "cleaning_time_predictor.pkl")
# Save with pickle
with open("cleaning_time_predictor.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as 'cleaning_time_predictor.pkl'")
