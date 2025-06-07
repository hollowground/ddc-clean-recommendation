import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Step 1: Load data
df = pd.read_csv("training_dataset_new.csv")

# Step 2: Separate features and target
X = df.drop(columns=["target_time_between_cleaning"])
y = df["target_time_between_cleaning"]

# Step 3: Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Step 4: Define preprocessing (transformer)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        # Numerical columns can be passed through as-is by default
    ],
    remainder='passthrough'  # Keep non-categorical columns as-is
)

# Step 5: Build pipeline with preprocessor + model
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Fit the pipeline
pipeline.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 9: Save the entire pipeline (includes encoder + model)
with open("cleaning_time_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("Pipeline saved as 'cleaning_time_pipeline.pkl'")
