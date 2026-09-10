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


# ============================================================
# FONT / DISPLAY SETTINGS
# ============================================================

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

    [data-testid="stMetricLabel"] {
        font-size: 20px !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 36px !important;
    }

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
# CUSTOM NUMERIC INPUT
# Allows:
# 1. Blank initial value
# 2. Direct keyboard entry
# 3. +/- buttons from blank state
# ============================================================

def adjust_numeric_value(
    text_key,
    min_value,
    max_value,
    step,
    decimals,
    direction
):
    raw = st.session_state.get(text_key, "")

    if raw is None:
        raw = ""

    raw = str(raw).strip().replace(",", "")

    try:
        current_value = float(raw) if raw != "" else None
    except ValueError:
        current_value = None

    # If currently blank:
    # + starts at min + step
    # - starts at min
    if current_value is None:

        if direction > 0:
            new_value = min_value + step
        else:
            new_value = min_value

    else:

        new_value = current_value + (direction * step)

    # Keep within allowed range
    new_value = max(
        min_value,
        min(max_value, new_value)
    )

    # Format displayed value
    if decimals == 0:

        st.session_state[text_key] = str(
            int(round(new_value))
        )

    else:

        st.session_state[text_key] = (
            f"{new_value:.{decimals}f}"
        )


def numeric_input(
    label,
    key,
    min_value,
    max_value,
    step,
    decimals=1
):
    text_key = f"{key}_text"

    # Initial blank value
    if text_key not in st.session_state:
        st.session_state[text_key] = ""

    # Input label
    st.markdown(
        f"""
        <div style="
            font-size:20px;
            margin-bottom:5px;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Minus | Input | Plus
    col_minus, col_input, col_plus = st.columns(
        [1, 8, 1],
        gap="small"
    )

    # --------------------------------------------------------
    # MINUS BUTTON
    # --------------------------------------------------------

    with col_minus:

        st.button(
            "−",
            key=f"{key}_minus",
            use_container_width=True,
            on_click=adjust_numeric_value,
            args=(
                text_key,
                min_value,
                max_value,
                step,
                decimals,
                -1
            )
        )

    # --------------------------------------------------------
    # TEXT INPUT
    # --------------------------------------------------------

    with col_input:

        raw_value = st.text_input(
            label,
            key=text_key,
            label_visibility="collapsed",
            placeholder="Enter value"
        )

    # --------------------------------------------------------
    # PLUS BUTTON
    # --------------------------------------------------------

    with col_plus:

        st.button(
            "+",
            key=f"{key}_plus",
            use_container_width=True,
            on_click=adjust_numeric_value,
            args=(
                text_key,
                min_value,
                max_value,
                step,
                decimals,
                1
            )
        )

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if raw_value is None:
        return None

    cleaned_value = (
        str(raw_value)
        .strip()
        .replace(",", "")
    )

    # Still blank
    if cleaned_value == "":
        return None

    # Must be numeric
    try:
        value = float(cleaned_value)

    except ValueError:

        st.error(
            f"{label}: please enter a numeric value."
        )

        return None

    # Must fall inside requested range
    if value < min_value or value > max_value:

        st.error(
            f"{label}: value must be between "
            f"{min_value} and {max_value}."
        )

        return None

    # Integer-only fields
    if decimals == 0:

        if not value.is_integer():

            st.error(
                f"{label}: please enter a whole number."
            )

            return None

        return int(value)

    return value


# ============================================================
# TITLE
# ============================================================

st.title("Illinois Preterm Birth Risk Index")


st.markdown(
    """
    <div style="
        font-size:15px;
        color:#8a8f98;
        margin-top:-8px;
        margin-bottom:6px;
    ">
        Developed by Prafulla Caringula and Dr. Yu-Sheng Lee
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style="
        color:#FF0000;
        font-size:18px;
        margin-top:8px;
        margin-bottom:18px;
    ">
        County-level research tool; not intended for individual pregnancy risk assessment.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUTS
# ============================================================

st.subheader(
    "Enter your county's indicators to generate a prediction"
)


# ------------------------------------------------------------
# Current PTB
# ------------------------------------------------------------

current_ptb = numeric_input(
    "Current Preterm Birth (%); please enter 0-100",
    key="current_ptb",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ------------------------------------------------------------
# Maternal age <20
# ------------------------------------------------------------

age_lt20 = numeric_input(
    "Maternal Age <20 (%); please enter 0-100",
    key="age_lt20",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ------------------------------------------------------------
# Maternal age 40+
# ------------------------------------------------------------

age_40plus = numeric_input(
    "Maternal Age 40+ (%); please enter 0-100",
    key="age_40plus",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ------------------------------------------------------------
# Black mothers
# ------------------------------------------------------------

black_mother = numeric_input(
    "Black Mothers (%); please enter 0-100",
    key="black_mother",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ------------------------------------------------------------
# Unmarried mothers
# ------------------------------------------------------------

unmarried = numeric_input(
    "Unmarried Mothers (%); please enter 0-100",
    key="unmarried",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ------------------------------------------------------------
# SVI
# ------------------------------------------------------------

svi = numeric_input(
    "Social Vulnerability Index (SVI); please enter 0-1",
    key="svi",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    decimals=2
)


# ------------------------------------------------------------
# Multiple gestation
# ------------------------------------------------------------

multiple_gestation = numeric_input(
    "Multiple Gestation (%); please enter 0-100",
    key="multiple_gestation",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ------------------------------------------------------------
# Low birth weight
# ------------------------------------------------------------

low_birth_weight = numeric_input(
    "Low Birth Weight (%); please enter 0-100",
    key="low_birth_weight",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ------------------------------------------------------------
# Caesarian delivery
# ------------------------------------------------------------

caesarian = numeric_input(
    "Caesarian Delivery (%); please enter 0-100",
    key="caesarian",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ------------------------------------------------------------
# RUCC
# ------------------------------------------------------------

rucc = st.selectbox(
    "Rural-Urban Continuum Code (RUCC)",
    options=[
        1, 2, 3, 4, 5, 6, 7, 8, 9
    ],
    index=None,
    placeholder="Select RUCC"
)


# ------------------------------------------------------------
# HPSA
# ------------------------------------------------------------

hpsa_label = st.selectbox(
    "HPSA primary care provider shortage",
    options=[
        "No",
        "Yes"
    ],
    index=None,
    placeholder="Select No or Yes"
)


# ------------------------------------------------------------
# PM2.5
# ------------------------------------------------------------

pm25 = numeric_input(
    "Annual PM2.5 (µg/m³); please enter 0-50",
    key="pm25",
    min_value=0.0,
    max_value=50.0,
    step=0.1,
    decimals=1
)


# ------------------------------------------------------------
# CDD
# ------------------------------------------------------------

cdd = numeric_input(
    "Cooling Degree Days (CDD); please enter 0-10000",
    key="cdd",
    min_value=0,
    max_value=10000,
    step=1,
    decimals=0
)


# ------------------------------------------------------------
# HDD
# ------------------------------------------------------------

hdd = numeric_input(
    "Heating Degree Days (HDD); please enter 0-15000",
    key="hdd",
    min_value=0,
    max_value=15000,
    step=1,
    decimals=0
)


# ============================================================
# CHECK WHETHER ALL INPUTS ARE COMPLETE
# ============================================================

required_inputs = [
    current_ptb,
    age_lt20,
    age_40plus,
    black_mother,
    unmarried,
    svi,
    multiple_gestation,
    low_birth_weight,
    caesarian,
    rucc,
    hpsa_label,
    pm25,
    cdd,
    hdd
]


all_complete = all(
    value is not None
    for value in required_inputs
)


# ============================================================
# DEFAULT RESULTS
# ============================================================

predicted_ptb = 0.0
risk_index = 0


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "Predict Next-Year Risk",
    type="primary",
    use_container_width=True,
    disabled=not all_complete
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button and all_complete:

    # --------------------------------------------------------
    # Convert HPSA to model value
    # --------------------------------------------------------

    hpsa = (
        1
        if hpsa_label == "Yes"
        else 0
    )


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Ensure exact predictor order used during training
    # --------------------------------------------------------

    input_data = input_data[
        features
    ]


    # --------------------------------------------------------
    # RANDOM FOREST PREDICTION
    # --------------------------------------------------------

    predicted_ptb = float(
        model.predict(
            input_data
        )[0]
    )


    # --------------------------------------------------------
    # MATERNAL HEALTH RISK INDEX
    # --------------------------------------------------------

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
        min(
            100,
            risk_index
        )
    )


# ============================================================
# INCOMPLETE INPUT MESSAGE
# ============================================================

if not all_complete:

    st.caption(
        "Please complete all required fields to generate a prediction."
    )


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.subheader(
    "Prediction Results"
)


col1, col2 = st.columns(2)

with col1:

    if predict_button and all_complete:
        ptb_display = f"{predicted_ptb:.2f}%"
    else:
        ptb_display = "—"

    st.metric(
        label="Next-Year Preterm Birth %",
        value=ptb_display
    )


with col2:

    if predict_button and all_complete:
        percentile_display = f"{risk_index} / 100"
    else:
        percentile_display = "—"

    st.metric(
        label="Next-Year Preterm Birth Percentile^",
        value=percentile_display
    )

# ============================================================
# RESULT INTERPRETATION
# ============================================================

if predict_button and all_complete:

    st.write(
        f"""
        ^A percentile of **{risk_index}** indicates that the
        predicted next-year preterm birth percentage is higher
        than approximately **{risk_index}%** of Illinois
        county-year preterm birth percentages in the historical
        reference distribution.
        """
    )


# ============================================================
# ABOUT TOOL
# ============================================================

st.divider()


with st.expander(
    "About the tool"
):

    st.write(
        """
        This prediction model is a Random Forest regression model
        developed using annual Illinois county-level data.
        Predictors from year t are used to forecast the preterm
        birth percentage in year t+1. The 0–100 Maternal Health Risk Index represents the
        percentile of the predicted preterm birth percentage
        relative to the historical Illinois county-year preterm
        birth distribution.

        This tool is intended for public health planning and
        research and should not be interpreted as an
        individual-level clinical risk assessment.

        The model and tool were developed by Prafulla Caringula
        of the Woodford County Health Department and Dr. Yu-Sheng Lee
        of the University of Illinois Springfield.

        If you have questions about this tool, please contact:
        Dr. Yu-Sheng Lee | ylee317@uis.edu | 1-217-206-7874.
        """
    )
