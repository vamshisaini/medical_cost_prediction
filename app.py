import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Medical Cost Prediction",
    page_icon="🏥",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD SAVED MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    medical_cost = joblib.load("medical_cost.pkl")

    model = medical_cost["model"]
    cat_encoder = medical_cost["cat_encoder"]
    num_encoder = medical_cost["num_encoder"]

    return model, cat_encoder, num_encoder


model, cat_encoder, num_encoder = load_model()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏥 Medical Cost Prediction")

st.write(
    "Predict the annual medical cost of a patient using "
    "health, lifestyle, and insurance information."
)

st.divider()


# ---------------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------------

st.header("👤 Patient Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )


with col2:

    smoker = st.selectbox(
        "Smoker",
        ["No", "Yes"]
    )

    physical_activity_level = st.selectbox(
        "Physical Activity Level",
        ["Low", "Medium", "High"]
    )

    city_type = st.selectbox(
        "City Type",
        ["Rural", "Semi-Urban", "Urban"]
    )


with col3:

    insurance_type = st.selectbox(
        "Insurance Type",
        ["Private", "Government", "unknown"]
    )

    insurance_coverage_pct = st.number_input(
        "Insurance Coverage (%)",
        min_value=0,
        max_value=100,
        value=50
    )


# ---------------------------------------------------------
# LIFESTYLE INFORMATION
# ---------------------------------------------------------

st.divider()

st.header("🏃 Lifestyle Information")

col1, col2, col3 = st.columns(3)

with col1:

    daily_steps = st.number_input(
        "Daily Steps",
        min_value=0,
        max_value=50000,
        value=5000
    )

with col2:

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.1
    )

with col3:

    stress_level = st.number_input(
        "Stress Level",
        min_value=1,
        max_value=10,
        value=5
    )

    doctor_visits_per_year = st.number_input(
        "Doctor Visits Per Year",
        min_value=0,
        max_value=20,
        value=2
    )

# ---------------------------------------------------------
# MEDICAL INFORMATION
# ---------------------------------------------------------

st.divider()

st.header("❤️ Medical Information")

col1, col2, col3 = st.columns(3)


with col1:

    diabetes = st.selectbox(
        "Diabetes",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


with col2:

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    asthma = st.selectbox(
        "Asthma",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


with col3:

    hospital_admissions = st.number_input(
        "Hospital Admissions",
        min_value=0,
        max_value=20,
        value=0
    )

    medication_count = st.number_input(
        "Medication Count",
        min_value=0,
        max_value=20,
        value=0
    )


# ---------------------------------------------------------
# PREVIOUS MEDICAL COST
# ---------------------------------------------------------

st.divider()

st.header("💰 Previous Medical Cost")

previous_year_cost = st.number_input(
    "Previous Year Medical Cost",
    min_value=0.0,
    max_value=100000.0,
    value=5000.0,
    step=100.0
)


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Annual Medical Cost",
    use_container_width=True
):

    # -----------------------------------------------------
    # CREATE CATEGORICAL DATAFRAME
    # -----------------------------------------------------

    cat_data = pd.DataFrame({

        "gender": [gender],

        "smoker": [smoker],

        "physical_activity_level": [
            physical_activity_level
        ],

        "insurance_type": [
            insurance_type
        ],

        "city_type": [
            city_type
        ]
    })


    # -----------------------------------------------------
    # CREATE NUMERICAL DATAFRAME
    # -----------------------------------------------------

    num_data = pd.DataFrame({

        "age": [age],

        "bmi": [bmi],

        "daily_steps": [daily_steps],

        "sleep_hours": [sleep_hours],

        "stress_level": [stress_level],

        "doctor_visits_per_year": [doctor_visits_per_year],

        "insurance_coverage_pct": [
            insurance_coverage_pct
        ],

        "previous_year_cost": [
            previous_year_cost
        ]
    })


    # -----------------------------------------------------
    # UNTRANSFORMED NUMERICAL/BINARY FEATURES
    # -----------------------------------------------------

    no_data = pd.DataFrame({

        "diabetes": [diabetes],

        "hypertension": [hypertension],

        "heart_disease": [heart_disease],

        "asthma": [asthma],

        "hospital_admissions": [
            hospital_admissions
        ],

        "medication_count": [
            medication_count
        ]
    })


    # -----------------------------------------------------
    # APPLY SAVED ENCODERS
    # -----------------------------------------------------

    cat_transformed = cat_encoder.transform(cat_data)

    num_transformed = num_encoder.transform(num_data)


    # -----------------------------------------------------
    # COMBINE FEATURES
    # -----------------------------------------------------

    final_input = pd.concat(
        [
            cat_transformed,
            num_transformed,
            no_data
        ],
        axis=1
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(final_input)

    predicted_cost = prediction[0]


    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------

    st.success("Prediction completed successfully!")

    st.metric(
        "Estimated Annual Medical Cost",
        f"{predicted_cost:,.2f}"
    )
