import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Illinois Preterm Birth Risk Index",
    page_icon="📊",
    layout="centered"
)

st.markdown(
    """
    <style>

       html, body, [class*="css"] {
        font-size: 20px;
    }

    h1 {
        font-size: 46px !important;
    }

    h2 {
        font-size: 34px !important;
    }

    h3 {
        font-size: 28px !important;
    }

       p {
        font-size: 20px !important;
    }

    label {
        font-size: 20px !important;
    }
   
    input {
        font-size: 20px !important;
    }

    div[data-baseweb="select"] {
        font-size: 20px !important;
    }

        .stButton button {
        font-size: 22px !important;
        font-weight: 600;
    }

    /* metric */
    [data-testid="stMetricLabel"] {
        font-size: 20px !important;
    }

    /* metric */
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
    }

    /* caption */
    [data-testid="stCaptionContainer"] {
        font-size: 17px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    bundle = joblib.load("ptb_rf.joblib")
    return bundle["model"], bundle["features"]


@st.cache_data
def load_reference():
    ref = pd.read_csv("ptb_reference.csv")
    return ref["PTB"].dropna().to_numpy()


model, features = load_model()
reference = load_reference()


# ============================================================
# TITLE
# ============================================================

st.title("Illinois Preterm Birth Risk Index")

st.caption(
    "Developed by Prafulla Caringula and Dr. Yu-Sheng Lee"
)

st.caption(
    "This is a county-level research tool and is not used to estimate individual pregnancy risk."
)


# ============================================================
# INPUTS
# ============================================================

st.subheader("Enter your County's Indicators")


current_ptb = st.number_input(
    "Current Preterm Birth (%); please enter 0-100",
    min_value=0.0,
    max_value=100.0,
    value=None,
    step=0.1,
    placeholder="Enter value"
)


svi = st.number_input(
    "Social Vulnerability Index (SVI); please enter 0-1",
    min_value=0.0,
    max_value=1.0,
    value=None,
    step=0.01,
    placeholder="Enter value"
)


age_lt20 = st.number_input(
    "Maternal Age <20 (%); please enter 0-100",
    min_value=0.0,
    max_value=100.0,
    value=None,
    step=0.1,
    placeholder="Enter value"
)


age_40plus = st.number_input(
    "Maternal Age 40+ (%); please enter 0-100",
    min_value=0.0,
    max_value=100.0,
    value=None,
    step=0.1,
    placeholder="Enter value"
)


black_mother = st.number_input(
    "Black Mothers (%); please enter 0-100",
    min_value=0.0,
    max_value=100.0,
    value=None,
    step=0.1,
    placeholder="Enter value"
)


multiple_gestation = st.number_input(
    "Multiple Gestation (%); please enter 0-100",
    min_value=0.0,
    max_value=100.0,
    value=None,
    step=0.1,
    placeholder="Enter value"
)


rucc = st.selectbox(
    "Rural-Urban Continuum Code (RUCC)",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    index=None,
    placeholder="Select RUCC"
)


pm25 = st.number_input(
    "Annual PM2.5 (µg/m³); please enter 0-50",
    min_value=0.0,
    max_value=50.0,
    value=None,
    step=0.1,
    placeholder="Enter value"
)


cdd = st.number_input(
    "Cooling Degree Days (CDD); please enter 0-10000",
    min_value=0,
    max_value=10000,
    value=None,
    step=1,
    placeholder="Enter value"
)


hdd = st.number_input(
    "Heating Degree Days (HDD); please enter 0-15000",
    min_value=0,
    max_value=15000,
    value=None,
    step=1,
    placeholder="Enter value"
)


caesarian = st.number_input(
    "Caesarian Delivery (%); please enter 0-100",
    min_value=0.0,
    max_value=100.0,
    value=None,
    step=0.1,
    placeholder="Enter value"
)


low_birth_weight = st.number_input(
    "Low Birth Weight (%); please enter 0-100",
    min_value=0.0,
    max_value=100.0,
    value=None,
    step=0.1,
    placeholder="Enter value"
)


unmarried = st.number_input(
    "Unmarried Mothers (%); please enter 0-100",
    min_value=0.0,
    max_value=100.0,
    value=None,
    step=0.1,
    placeholder="Enter value"
)


hpsa_label = st.selectbox(
    "HPSA primary care provider shortage",
    options=["No", "Yes"],
    index=None,
    placeholder="Select No or Yes"
)


# ============================================================
# CHECK WHETHER ALL INPUTS ARE COMPLETE
# ============================================================

required_inputs = [
    current_ptb,
    svi,
    age_lt20,
    age_40plus,
    black_mother,
    multiple_gestation,
    rucc,
    pm25,
    cdd,
    hdd,
    caesarian,
    low_birth_weight,
    unmarried,
    hpsa_label
]

all_complete = all(
    value is not None
    for value in required_inputs
)


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "Predict Next-Year Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.subheader("Prediction Results")


# Default results when inputs are incomplete
predicted_ptb = 0.0
risk_index = 0


if predict_button:

    if not all_complete:

        st.warning(
            "Please complete all required county indicators before generating a prediction."
        )

    else:

        # Convert HPSA to model value
        hpsa = 1 if hpsa_label == "Yes" else 0


        # ----------------------------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [{
                "PTB": current_ptb,
                "SVI": svi,
                "Age_lt20": age_lt20,
                "Age_40plus": age_40plus,
                "Black_Mother": black_mother,
                "Multiple_Gestation": multiple_gestation,
                "RUCC": rucc,
                "PM25": pm25,
                "CDD": cdd,
                "HDD": hdd,
                "Caesarian": caesarian,
                "Low_Birth_Weight": low_birth_weight,
                "Unmarried": unmarried,
                "HPSA_PrimaryCare": hpsa
            }]
        )


        # Ensure same predictor order used during training
        input_data = input_data[features]


        # ----------------------------------------------------
        # RANDOM FOREST PREDICTION
        # ----------------------------------------------------

        predicted_ptb = float(
            model.predict(input_data)[0]
        )


        # ----------------------------------------------------
        # CONVERT PREDICTED PTB TO 0-100 RISK INDEX
        # ----------------------------------------------------

        risk_index = int(
            round(
                100 *
                np.mean(
                    reference <= predicted_ptb
                )
            )
        )


        risk_index = max(
            0,
            min(100, risk_index)
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.metric(
        label="Predicted Next-Year PTB",
        value=f"{predicted_ptb:.2f}%"
    )


with col2:

    st.metric(
        label="Maternal Health Risk Index",
        value=f"{risk_index} / 100"
    )


# Only show interpretation after a valid prediction
if predict_button and all_complete:

    st.write(
        f"""
        A risk index of **{risk_index}** indicates that the
        predicted next-year preterm birth percentage is higher
        than approximately **{risk_index}%** of Illinois
        county-year preterm birth percentages in the historical
        reference distribution.
        """
    )

elif not all_complete:

    st.caption(
        "Complete all required fields to generate a prediction."
    )


# ============================================================
# ABOUT MODEL
# ============================================================

st.divider()

with st.expander("About the model"):

    st.write(
        """

        This prediction model is a Random Forest regression model developed using annual Illinois county-level data. Predictors from year t are used to forecast the preterm birth percentage in year t+1. The 0–100 Maternal Health Risk Index represents the percentile of the predicted preterm birth percentage relative to the historical Illinois county-year preterm birth distribution.

        This tool is intended for public health planning and research and should not be interpreted as an individual-level clinical risk assessment.

        The model and tool was developed by Prafulla Caringula of the Woodford County Health Department and Dr. Yu-Sheng Lee of the University of Illinois Springfield. If you have questions about this research, you may contact: Yu-Sheng Lee, ylee317@uis.edu, 1-217-206-7874.  
        """
    )
