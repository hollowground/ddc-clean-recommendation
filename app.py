import streamlit as st
import joblib
import pandas as pd
from io import BytesIO

# Load model
model = joblib.load("cleaning_time_predictor.pkl")

st.title("CleanSweep AI")

st.write("Upload a CSV or Excel file with the following columns:")
st.code(
    """
- last_people_traffic
- last_inspection_score
- prev_people_traffic
- prev_inspection_score
- time_between_cleaning
- people_traffic_diff
"""
)

# File uploader
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Check required columns
        required_cols = [
            "last_people_traffic",
            "last_inspection_score",
            "prev_people_traffic",
            "prev_inspection_score",
            "time_between_cleaning",
            "people_traffic_diff",
        ]

        if not all(col in df.columns for col in required_cols):
            st.error(f"File is missing required columns. Must include: {required_cols}")
        else:
            # Make predictions
            predictions = model.predict(df[required_cols])
            df["predicted_time_between_cleaning"] = predictions.round(0)

            # Display
            st.success("Recommendations Completed!")
            st.dataframe(df)

            # Download button
            def convert_df(df):
                output = BytesIO()
                df.to_excel(output, index=False, engine="xlsxwriter")
                output.seek(0)
                return output

            st.download_button(
                label="📥 Download Recommendations to Excel",
                data=convert_df(df),
                file_name="predicted_cleaning_intervals.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
