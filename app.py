import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MedCost AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

    /* ---------- MAIN PAGE ---------- */

    .stApp {
        background-color: #f4f7fb;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- HEADER ---------- */

    .app-title {
        font-size: 42px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 0;
    }

    .app-subtitle {
        font-size: 17px;
        color: #667085;
        margin-top: 5px;
        margin-bottom: 25px;
    }


    /* ---------- SECTION HEADERS ---------- */

    .section-header {
        font-size: 23px;
        font-weight: 700;
        color: #172033;
        margin-top: 25px;
        margin-bottom: 12px;
    }


    /* ---------- INPUT CONTAINERS ---------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        padding: 12px;
    }


    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 700;
        border: none;
    }


    /* ---------- METRIC ---------- */

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 16px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 34px;
        font-weight: 750;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 13px;
        margin-top: 45px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    saved_model = joblib.load(
        "medical_cost.pkl"
    )

    model = saved_model["model"]
    cat_encoder = saved_model["cat_encoder"]
    num_encoder = saved_model["num_encoder"]

    return model, cat_encoder, num_encoder


model, cat_encoder, num_encoder = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="app-title">🏥 MedCost AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'AI-powered annual medical cost prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PATIENT PROFILE
# =========================================================

st.markdown(
    '<div class="section-header">👤 Patient Profile</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col3:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )


# =========================================================
# LIFESTYLE
# =========================================================

st.markdown(
    '<div class="section-header">🏃 Lifestyle & Daily Habits</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        daily_steps = st.number_input(
            "Daily Steps",
            min_value=0,
            max_value=50000,
            value=5000,
            step=500
        )

        physical_activity_level = st.selectbox(
            "Physical Activity Level",
            ["Low", "Medium", "High"]
        )

    with col2:

        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

        stress_level = st.slider(
            "Stress Level",
            min_value=1,
            max_value=10,
            value=5
        )

    with col3:

        smoker = st.selectbox(
            "Smoking Status",
            ["No", "Yes"]
        )

        doctor_visits_per_year = st.number_input(
            "Doctor Visits / Year",
            min_value=0,
            max_value=30,
            value=2
        )


# =========================================================
# HEALTH INFORMATION
# =========================================================

st.markdown(
    '<div class="section-header">❤️ Health Information</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

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
            max_value=30,
            value=0
        )

        medication_count = st.number_input(
            "Medication Count",
            min_value=0,
            max_value=30,
            value=0
        )


# =========================================================
# INSURANCE & COST
# =========================================================

st.markdown(
    '<div class="section-header">💳 Insurance & Cost Information</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        insurance_type = st.selectbox(
            "Insurance Type",
            ["Private", "Government", "unknown"]
        )

    with col2:

        insurance_coverage_pct = st.slider(
            "Insurance Coverage (%)",
            min_value=0,
            max_value=100,
            value=50
        )

    with col3:

        previous_year_cost = st.number_input(
            "Previous Year Medical Cost",
            min_value=0.0,
            max_value=100000.0,
            value=5000.0,
            step=100.0
        )


# =========================================================
# LOCATION
# =========================================================

st.markdown(
    '<div class="section-header">🏙️ Location</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    city_type = st.selectbox(
        "City Type",
        ["Rural", "Semi-Urban", "Urban"]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

if st.button(
    "🔮  Predict Medical Cost",
    use_container_width=True
):

    # -----------------------------------------------------
    # CATEGORICAL FEATURES
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
    # NUMERICAL FEATURES
    # -----------------------------------------------------

    num_data = pd.DataFrame({

        "age": [age],

        "bmi": [bmi],

        "daily_steps": [daily_steps],

        "sleep_hours": [sleep_hours],

        "stress_level": [stress_level],

        "doctor_visits_per_year": [
            doctor_visits_per_year
        ],

        "insurance_coverage_pct": [
            insurance_coverage_pct
        ],

        "previous_year_cost": [
            previous_year_cost
        ]
    })


    # -----------------------------------------------------
    # OTHER FEATURES
    # -----------------------------------------------------

    other_data = pd.DataFrame({

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
    # APPLY SAVED PREPROCESSING
    # -----------------------------------------------------

    cat_transformed = cat_encoder.transform(
        cat_data
    )

    num_transformed = num_encoder.transform(
        num_data
    )


    # -----------------------------------------------------
    # COMBINE FEATURES
    # -----------------------------------------------------

    final_input = pd.concat(
        [
            cat_transformed,
            num_transformed,
            other_data
        ],
        axis=1
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(final_input)

    predicted_cost = float(prediction[0])


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-header">💰 Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2, result_col3 = st.columns(
        [1, 2, 1]
    )

    with result_col2:

        st.metric(
            label="Estimated Annual Medical Cost",
            value=f"{predicted_cost:,.2f}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'MedCost AI • Machine Learning Medical Cost Prediction'
    '</div>',
    unsafe_allow_html=True
)
