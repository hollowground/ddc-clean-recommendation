import streamlit as st
import pandas as pd
import pickle
from io import BytesIO

# Load pipeline
with open("cleaning_time_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.title("🧼 CleanSweep AI – Cleaning Intervals Recommendation")

st.write("Upload a CSV or Excel file with the following required columns:")
st.code(
    """
- campus (str)
- building (str)
- floor (str)
- location (str)
- location_number (str)
- space_type (str)
- last_people_traffic (numeric)
- last_inspection_score (numeric)
- prev_people_traffic (numeric)
- prev_inspection_score (numeric)
- time_between_cleaning (numeric)
- people_traffic_diff (numeric)
"""
)

# Required columns and their expected types
required_columns = {
    "campus": "object",
    "building": "object",
    "floor": "object",
    "location": "object",
    "location_number": "object",
    "space_type": "object",
    "last_people_traffic": "number",
    "last_inspection_score": "number",
    "prev_people_traffic": "number",
    "prev_inspection_score": "number",
    "time_between_cleaning": "number",
    "people_traffic_diff": "number",
}


# 📥 Template download button
def get_template():
    data = {
        col: ["Example" if typ == "object" else 0]
        for col, typ in required_columns.items()
    }
    df_template = pd.DataFrame(data)
    output = BytesIO()
    df_template.to_excel(output, index=False, engine="xlsxwriter")
    output.seek(0)
    return output


st.download_button(
    label="📄 Download CSV Template",
    data=get_template(),
    file_name="cleaning_prediction_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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

        # ✅ Check for required columns
        if not all(col in df.columns for col in required_columns):
            st.error(
                f"❌ File is missing required columns.\nExpected: {list(required_columns.keys())}"
            )
        else:
            # 🔍 Validate column types
            bad_types = []
            for col, expected_type in required_columns.items():
                actual_dtype = df[col].dtype
                if expected_type == "number" and not pd.api.types.is_numeric_dtype(
                    actual_dtype
                ):
                    bad_types.append((col, str(actual_dtype), "numeric"))
                elif expected_type == "object" and not pd.api.types.is_object_dtype(
                    actual_dtype
                ):
                    bad_types.append((col, str(actual_dtype), "string"))

            if bad_types:
                st.error("❌ Column type mismatches detected:")
                for col, found, expected in bad_types:
                    st.write(
                        f"- `{col}`: expected **{expected}**, but found **{found}**"
                    )
            else:
                # ✅ Make predictions
                predictions = pipeline.predict(df[list(required_columns.keys())])
                df["recommended_time_between_cleaning"] = predictions.round(0)
                df["recommended_hrs_adjustment"] = (
                    df["recommended_time_between_cleaning"]
                    - df["time_between_cleaning"]
                )

                st.success("✅ Recommendations Completed!")
                st.dataframe(df, use_container_width=True)

                # Download button
                def convert_df(df):
                    output = BytesIO()
                    df.to_excel(output, index=False, engine="xlsxwriter")
                    output.seek(0)
                    return output

                st.download_button(
                    label="📥 Download Recommendations to Excel",
                    data=convert_df(df),
                    file_name="recommended_cleaning_intervals.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
