import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE SETTINGS
# ============================================================

st.title("Illinois Preterm Birth Risk Index")

st.caption(
    "Developed by Prafulla Caringula and Dr. Yu-Sheng Lee"
)

st.write(
    """
    Enter your county's current maternal, social, healthcare-access,
    and environmental indicators to estimate the next-year
    preterm birth percentage.
    """
)

st.caption(
    "County-level research tool. This model does not estimate individual pregnancy risk."
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
# INPUTS
# ============================================================

st.subheader("County Indicators")


current_ptb = st.number_input(
    "Current Preterm Birth (%)",
    min_value=0.0,
    max_value=40.0,
    value=0.0,
    step=0.1
)


svi = st.number_input(
    "Social Vulnerability Index (SVI)",
    min_value=0.0,
    max_value=1.0,
    value=0.00,
    step=0.01
)


age_lt20 = st.number_input(
    "Maternal Age <20 (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)


age_40plus = st.number_input(
    "Maternal Age 40+ (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)


black_mother = st.number_input(
    "Black Mothers (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)


multiple_gestation = st.number_input(
    "Multiple Gestation (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)


rucc = st.selectbox(
    "Rural-Urban Continuum Code (RUCC)",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    index=None,
    placeholder="Select RUCC"
)


pm25 = st.number_input(
    "Annual PM2.5 (µg/m³)",
    min_value=0.0,
    max_value=50.0,
    value=0.0,
    step=0.1
)


cdd = st.number_input(
    "Cooling Degree Days (CDD); please enter 0-10000",
    min_value=0,
    max_value=10000,
    value=0,
    step=10
)


hdd = st.number_input(
    "Heating Degree Days (HDD); please enter 0-15000",
    min_value=0,
    max_value=15000,
    value=0,
    step=10
)


caesarian = st.number_input(
    "Caesarian Delivery (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)


low_birth_weight = st.number_input(
    "Low Birth Weight (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)


unmarried = st.number_input(
    "Unmarried Mothers (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)


hpsa_label = st.selectbox(
    "HPSA primary care provider shortage",
    options=[
        "No",
        "Yes"
    ]
)

hpsa = 1 if hpsa_label == "Yes" else 0


# ============================================================
# CREATE MODEL INPUT
# ============================================================

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


# Make absolutely sure the column order matches training
input_data = input_data[features]


# ============================================================
# PREDICTION
# ============================================================

st.write("")

if st.button(
    "Predict Next-Year Risk",
    type="primary",
    use_container_width=True
):

    predicted_ptb = float(
        model.predict(input_data)[0]
    )


    # --------------------------------------------------------
    # Convert predicted PTB to historical percentile
    # --------------------------------------------------------

    risk_index = int(
        round(
            100 *
            np.mean(
                reference <= predicted_ptb
            )
        )
    )


    # Keep strictly within 0-100
    risk_index = max(
        0,
        min(100, risk_index)
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.subheader("Prediction Results")


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


    st.write(
        f"""
        A risk index of **{risk_index}** indicates that the
        predicted next-year preterm birth percentage is higher
        than approximately **{risk_index}%** of Illinois
        county-year preterm birth percentages in the historical
        reference distribution.
        """
    )


# ============================================================
# METHODS NOTE
# ============================================================

st.divider()

with st.expander("About the model"):

    st.write(
        """
        The prediction model is a Random Forest regression model
        developed using annual Illinois county-level data.

        Predictors from year t are used to forecast the preterm
        birth percentage in year t+1.

        The 0–100 Maternal Health Risk Index represents the
        percentile of the predicted preterm birth percentage
        relative to the historical Illinois county-year
        preterm birth distribution.

        The tool is intended for public health planning and
        research and should not be interpreted as an
        individual-level clinical risk assessment.
        
        "This tool is developed by Dr. Yu-Sheng Lee | University of Illinois Springfield and Prafulla Caringula | Woodford County Health Department"
        """
    )
