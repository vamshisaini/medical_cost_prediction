import streamlit as st
import pandas as pd
import joblib

# Load model
medical_cost = joblib.load("medical_cost_prediction.pkl")

model = medical_cost["model"]
cat_encoder = medical_cost["cat_encoder"]
num_encoder = medical_cost["num_encoder"]

st.set_page_config(
    page_title="Medical Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Annual Medical Cost Prediction")
st.write("Enter the patient details below to predict annual medical expenses.")

# ----------------------------
# Categorical Inputs
# ----------------------------
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

physical_activity_level = st.selectbox(
    "Physical Activity Level",
    ["Low", "Medium", "High"]
)

insurance_type = st.selectbox(
    "Insurance Type",
    ["Government", "Private", "unknown"]
)

city_type = st.selectbox(
    "City Type",
    ["Rural", "Semi-Urban", "Urban"]
)

# ----------------------------
# Numerical Inputs
# ----------------------------
age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=30
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

daily_steps = st.number_input(
    "Daily Steps",
    min_value=0,
    value=5000
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

stress_level = st.slider(
    "Stress Level",
    1,
    10,
    5
)

doctor_visits_per_year = st.number_input(
    "Doctor Visits Per Year",
    min_value=0,
    value=2
)

insurance_coverage_pct = st.number_input(
    "Insurance Coverage (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

previous_year_cost = st.number_input(
    "Previous Year Medical Cost",
    min_value=0.0,
    value=10000.0
)

# ----------------------------
# Binary Inputs
# ----------------------------
diabetes = st.selectbox("Diabetes", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
asthma = st.selectbox("Asthma", [0, 1])

hospital_admissions = st.number_input(
    "Hospital Admissions",
    min_value=0,
    value=0
)

medication_count = st.number_input(
    "Medication Count",
    min_value=0,
    value=0
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Medical Cost"):

    # Categorical Data
    cat_df = pd.DataFrame({
        "gender": [gender],
        "smoker": [smoker],
        "physical_activity_level": [physical_activity_level],
        "insurance_type": [insurance_type],
        "city_type": [city_type]
    })

    cat_trans = cat_encoder.transform(cat_df)

    # Numerical Data
    num_df = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "daily_steps": [daily_steps],
        "sleep_hours": [sleep_hours],
        "stress_level": [stress_level],
        "doctor_visits_per_year": [doctor_visits_per_year],
        "insurance_coverage_pct": [insurance_coverage_pct],
        "previous_year_cost": [previous_year_cost]
    })

    num_trans = num_encoder.transform(num_df)

    # Binary Data
    binary_df = pd.DataFrame({
        "diabetes": [diabetes],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "asthma": [asthma],
        "hospital_admissions": [hospital_admissions],
        "medication_count": [medication_count]
    })

    # Final Input
    final_input = pd.concat(
        [cat_trans, num_trans, binary_df],
        axis=1
    )

    prediction = model.predict(final_input)

    prediction =float(prediction.squeeze())

    st.success(
        f"Predicted Annual Medical Cost: ₹{prediction:,.2f}"
    )