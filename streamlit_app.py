import streamlit as st
import pandas as pd
import numpy as np
import joblib
import html


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Illinois Preterm Birth Prediction",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# RESPONSIVE DISPLAY SETTINGS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       DESKTOP / DEFAULT
       ====================================================== */

    .block-container {
        max-width: 1000px !important;
        padding-top: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 3rem !important;
    }

    html, body, [class*="css"] {
        font-size: 20px;
    }

    h1 {
        font-size: 42px !important;
        line-height: 1.15 !important;
        white-space: nowrap !important;
        margin-bottom: 0.6rem !important;
    }

    h2 {
        font-size: 32px !important;
        line-height: 1.2 !important;
    }

    h3 {
        font-size: 27px !important;
        line-height: 1.25 !important;
    }

    p {
        font-size: 20px !important;
        line-height: 1.55 !important;
    }

    label {
        font-size: 19px !important;
        line-height: 1.35 !important;
    }

    input {
        font-size: 19px !important;
    }

    div[data-baseweb="select"] {
        font-size: 19px !important;
    }

    [data-testid="stTextInput"] input {
        min-height: 48px !important;
    }

    .stButton button {
        font-size: 20px !important;
        font-weight: 600 !important;
        min-height: 48px !important;
    }

    [data-testid="stCaptionContainer"] {
        font-size: 15px !important;
    }


    /* ------------------------------------------------------
       INPUT QUESTION LABEL
       ------------------------------------------------------ */

    .input-label {
        font-size: 19px;
        line-height: 1.35;
        font-weight: 600;

        padding: 10px 12px;

        margin-top: 22px;
        margin-bottom: 10px;

        background-color: rgba(49, 51, 63, 0.055);

        border-left: 4px solid #d16a8a;
        border-radius: 6px;
    }


    /* ------------------------------------------------------
       TITLE SUPPORT TEXT
       ------------------------------------------------------ */

    .author-line {
        font-size: 15px;
        color: #8a8f98;
        margin-top: -6px;
        margin-bottom: 8px;
    }

    .warning-line {
        color: #E20000;
        font-size: 18px;
        line-height: 1.4;
        margin-top: 8px;
        margin-bottom: 24px;
    }


    /* ------------------------------------------------------
       RESULTS
       ------------------------------------------------------ */

    .results-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 48px;

        margin-top: 10px;
        margin-bottom: 24px;
    }

    .result-item {
        min-width: 0;
    }

    .result-label {
        font-size: 20px;
        line-height: 1.35;
        font-weight: 500;

        margin-bottom: 12px;

        white-space: normal;
        overflow-wrap: normal;
        word-break: normal;
    }

    .result-value {
        font-size: 32px;
        line-height: 1.2;
        font-weight: 400;
    }


    /* ======================================================
       TABLET
       ====================================================== */

    @media (max-width: 900px) {

        .block-container {
            max-width: 100% !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }

        h1 {
            font-size: 36px !important;
            white-space: normal !important;
        }

        .results-grid {
            gap: 28px;
        }

        .result-label {
            font-size: 18px;
        }

        .result-value {
            font-size: 30px;
        }
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 700px) {

        .block-container {
            padding-top: 1.2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-bottom: 2rem !important;
        }


        /* --------------------------------------------------
           MOBILE TITLE
           -------------------------------------------------- */

        h1 {
            font-size: 30px !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            margin-bottom: 0.6rem !important;
        }

        h2 {
            font-size: 26px !important;
            line-height: 1.2 !important;
        }

        h3 {
            font-size: 22px !important;
            line-height: 1.25 !important;
        }

        p {
            font-size: 17px !important;
            line-height: 1.5 !important;
        }


        /* --------------------------------------------------
           MOBILE AUTHOR / WARNING
           -------------------------------------------------- */

        .author-line {
            font-size: 13px;
            line-height: 1.35;

            margin-top: -3px;
            margin-bottom: 8px;
        }

        .warning-line {
            font-size: 16px;
            line-height: 1.4;

            margin-top: 8px;
            margin-bottom: 22px;
        }


        /* --------------------------------------------------
           MOBILE QUESTION LABELS
           -------------------------------------------------- */

        .input-label {
            font-size: 17px;
            line-height: 1.35;
            font-weight: 650;

            padding: 10px 12px;

            margin-top: 28px;
            margin-bottom: 9px;

            background-color: rgba(49, 51, 63, 0.06);

            border-left: 4px solid #d16a8a;
            border-radius: 6px;
        }


        /* --------------------------------------------------
           KEEP - / INPUT / + IN ONE ROW ON MOBILE
           -------------------------------------------------- */

        div[data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            gap: 0.45rem !important;
            align-items: stretch !important;
        }

        div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:first-child {

            flex: 0 0 52px !important;
            width: 52px !important;
            min-width: 52px !important;
        }

        div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:nth-child(2) {

            flex: 1 1 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }

        div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:last-child {

            flex: 0 0 52px !important;
            width: 52px !important;
            min-width: 52px !important;
        }


        /* --------------------------------------------------
           MOBILE INPUT
           -------------------------------------------------- */

        input {
            font-size: 17px !important;
        }

        [data-testid="stTextInput"] input {
            min-height: 46px !important;
            padding-left: 12px !important;
            padding-right: 12px !important;
        }

        div[data-baseweb="select"] {
            font-size: 17px !important;
        }


        /* --------------------------------------------------
           MOBILE +/- BUTTON
           -------------------------------------------------- */

        .stButton button {
            font-size: 19px !important;
            font-weight: 600 !important;

            min-height: 46px !important;

            padding-left: 0 !important;
            padding-right: 0 !important;
        }


        /* --------------------------------------------------
           CAPTION
           -------------------------------------------------- */

        [data-testid="stCaptionContainer"] {
            font-size: 14px !important;
        }


        /* --------------------------------------------------
           MOBILE RESULTS
           -------------------------------------------------- */

        .results-grid {
            grid-template-columns: 1fr;

            gap: 26px;

            margin-top: 8px;
            margin-bottom: 20px;
        }

        .result-label {
            font-size: 17px;
            line-height: 1.35;
            margin-bottom: 8px;
        }

        .result-value {
            font-size: 29px;
        }

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

    bundle = joblib.load(
        "ptb_rf.joblib"
    )

    return (
        bundle["model"],
        bundle["features"]
    )


@st.cache_data
def load_reference():

    ref = pd.read_csv(
        "ptb_reference.csv"
    )

    return (
        ref["PTB"]
        .dropna()
        .to_numpy()
    )


model, features = load_model()
reference = load_reference()


# ============================================================
# VERIFY CURRENT 12-PREDICTOR MODEL
# ============================================================

expected_features = [
    "PTB",
    "SVI",
    "Age_lt20",
    "Age_40plus",
    "Black_Mother",
    "Multiple_Gestation",
    "RUCC",
    "PM25",
    "CDD",
    "Caesarian",
    "Low_Birth_Weight",
    "Unmarried"
]


if list(features) != expected_features:

    st.error(
        "The deployed model file does not match the current "
        "12-predictor version of this tool."
    )

    st.stop()


# ============================================================
# CUSTOM NUMERIC INPUT FUNCTIONS
# ============================================================

def adjust_numeric_value(
    text_key,
    min_value,
    max_value,
    step,
    decimals,
    direction
):

    raw = st.session_state.get(
        text_key,
        ""
    )

    if raw is None:

        raw = ""


    raw = (
        str(raw)
        .strip()
        .replace(",", "")
    )


    try:

        current_value = (
            float(raw)
            if raw != ""
            else None
        )

    except ValueError:

        current_value = None


    # --------------------------------------------------------
    # IF FIELD IS EMPTY
    # --------------------------------------------------------

    if current_value is None:

        if direction > 0:

            new_value = (
                min_value + step
            )

        else:

            new_value = min_value


    # --------------------------------------------------------
    # IF FIELD ALREADY HAS VALUE
    # --------------------------------------------------------

    else:

        new_value = (
            current_value +
            direction * step
        )


    # --------------------------------------------------------
    # KEEP WITHIN RANGE
    # --------------------------------------------------------

    new_value = max(
        min_value,
        min(
            max_value,
            new_value
        )
    )


    # --------------------------------------------------------
    # FORMAT
    # --------------------------------------------------------

    if decimals == 0:

        st.session_state[text_key] = str(
            int(
                round(
                    new_value
                )
            )
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


    if text_key not in st.session_state:

        st.session_state[text_key] = ""


    safe_label = html.escape(
        label
    )


    # --------------------------------------------------------
    # QUESTION LABEL
    # --------------------------------------------------------

    st.markdown(
        f'<div class="input-label">'
        f'{safe_label}'
        f'</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # - / INPUT / +
    # --------------------------------------------------------

    col_minus, col_input, col_plus = st.columns(
        [1.2, 7.6, 1.2],
        gap="small"
    )


    # --------------------------------------------------------
    # MINUS
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
    # PLUS
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
    # VALIDATION
    # --------------------------------------------------------

    if raw_value is None:

        return None


    cleaned_value = (
        str(raw_value)
        .strip()
        .replace(",", "")
    )


    if cleaned_value == "":

        return None


    try:

        value = float(
            cleaned_value
        )

    except ValueError:

        st.error(
            f"{label}: please enter a numeric value."
        )

        return None


    if (
        value < min_value
        or
        value > max_value
    ):

        st.error(
            f"{label}: value must be between "
            f"{min_value} and {max_value}."
        )

        return None


    # --------------------------------------------------------
    # INTEGER-ONLY FIELD
    # --------------------------------------------------------

    if decimals == 0:

        if not value.is_integer():

            st.error(
                f"{label}: please enter a whole number."
            )

            return None

        return int(
            value
        )


    return value


# ============================================================
# TITLE
# ============================================================

st.title(
    "Illinois Preterm Birth Prediction Tool"
)


st.markdown(
    '<div class="author-line">'
    'Developed by Prafulla Caringula and Dr. Yu-Sheng Lee'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="warning-line">'
    'County-level research tool; not intended for individual '
    'pregnancy risk assessment.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader(
    "Enter your county's indicators to generate a prediction"
)


# ============================================================
# 1. CURRENT PTB
# ============================================================

current_ptb = numeric_input(
    "Current Preterm Birth (%); please enter 0-100",
    key="current_ptb",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ============================================================
# 2. MATERNAL AGE <20
# ============================================================

age_lt20 = numeric_input(
    "Maternal Age <20 (%); please enter 0-100",
    key="age_lt20",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ============================================================
# 3. MATERNAL AGE 40+
# ============================================================

age_40plus = numeric_input(
    "Maternal Age 40+ (%); please enter 0-100",
    key="age_40plus",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ============================================================
# 4. BLACK MOTHERS
# ============================================================

black_mother = numeric_input(
    "Black Mothers (%); please enter 0-100",
    key="black_mother",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ============================================================
# 5. UNMARRIED MOTHERS
# ============================================================

unmarried = numeric_input(
    "Unmarried Mothers (%); please enter 0-100",
    key="unmarried",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ============================================================
# 6. SVI
# ============================================================

svi = numeric_input(
    "Social Vulnerability Index (SVI; 0 = lower vulnerability, 1 = higher vulnerability)",
    key="svi",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    decimals=2
)


# ============================================================
# 7. MULTIPLE GESTATION
# ============================================================

multiple_gestation = numeric_input(
    "Multiple Gestation (%); please enter 0-100",
    key="multiple_gestation",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ============================================================
# 8. LOW BIRTH WEIGHT
# ============================================================

low_birth_weight = numeric_input(
    "Low Birth Weight (%); please enter 0-100",
    key="low_birth_weight",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ============================================================
# 9. CAESARIAN DELIVERY
# ============================================================

caesarian = numeric_input(
    "Caesarian Delivery (%); please enter 0-100",
    key="caesarian",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    decimals=1
)


# ============================================================
# 10. PM2.5
# ============================================================

pm25 = numeric_input(
    "Annual PM2.5 (µg/m³); please enter 0-50",
    key="pm25",
    min_value=0.0,
    max_value=50.0,
    step=0.1,
    decimals=1
)


# ============================================================
# 11. CDD
# ============================================================

cdd = numeric_input(
    "Cooling Degree Days (CDD); please enter 0-10000",
    key="cdd",
    min_value=0,
    max_value=10000,
    step=1,
    decimals=0
)


# ============================================================
# 12. RUCC
# ============================================================

st.markdown(
    '<div class="input-label">'
    'Rural-Urban Continuum Code '
    '(RUCC; 1 = most urban, 9 = most rural)'
    '</div>',
    unsafe_allow_html=True
)


rucc = st.selectbox(
    "Rural-Urban Continuum Code",
    options=[
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
    ],
    index=None,
    placeholder="Select RUCC",
    label_visibility="collapsed"
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
    pm25,
    cdd,
    rucc
]


all_complete = all(
    value is not None
    for value in required_inputs
)


# ============================================================
# DEFAULT RESULTS
# ============================================================

predicted_ptb = None
ptb_percentile = None


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")


predict_button = st.button(
    "Predict Next-Year PTB",
    type="primary",
    use_container_width=True,
    disabled=not all_complete
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button and all_complete:

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
            "Caesarian": caesarian,
            "Low_Birth_Weight": low_birth_weight,
            "Unmarried": unmarried
        }]
    )


    # --------------------------------------------------------
    # EXACT TRAINING FEATURE ORDER
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
    # HISTORICAL PTB PERCENTILE
    # --------------------------------------------------------

    ptb_percentile = int(
        round(
            100 *
            np.mean(
                reference <= predicted_ptb
            )
        )
    )


    ptb_percentile = max(
        0,
        min(
            100,
            ptb_percentile
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


# ============================================================
# PREPARE DISPLAY VALUES
# ============================================================

if predicted_ptb is not None:

    ptb_display = (
        f"{predicted_ptb:.2f}%"
    )

else:

    ptb_display = "—"


if ptb_percentile is not None:

    percentile_display = (
        f"{ptb_percentile} / 100"
    )

else:

    percentile_display = "—"


# ============================================================
# RESPONSIVE RESULT DISPLAY
# ============================================================

results_html = (
    '<div class="results-grid">'

        '<div class="result-item">'

            '<div class="result-label">'
                'Predicted Next-Year Preterm Birth (%)'
            '</div>'

            f'<div class="result-value">'
                f'{ptb_display}'
            f'</div>'

        '</div>'


        '<div class="result-item">'

            '<div class="result-label">'
                'Predicted Next-Year Preterm Birth Percentile'
            '</div>'

            f'<div class="result-value">'
                f'{percentile_display}'
            f'</div>'

        '</div>'

    '</div>'
)


st.markdown(
    results_html,
    unsafe_allow_html=True
)


# ============================================================
# RESULT INTERPRETATION
# ============================================================

if (
    predicted_ptb is not None
    and
    ptb_percentile is not None
):

    st.write(
        f"""
A percentile of **{ptb_percentile}** indicates that the predicted
next-year preterm birth percentage is higher than approximately
**{ptb_percentile}%** of Illinois county-year preterm birth percentages
in the historical reference distribution.
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
This prediction model is a birth-weighted Random Forest regression
model developed using annual Illinois county-level data. Annual county
observations were weighted by the number of births associated with the
next-year preterm birth outcome during model fitting.

Predictors from year t are used to forecast the county-level preterm
birth percentage in year t+1.

The Predicted Next-Year Preterm Birth Percentile represents the
percentile of the predicted next-year preterm birth percentage relative
to the historical Illinois county-year preterm birth distribution.

Changes in individual input values should not be interpreted as causal
changes in preterm birth risk. Relationships learned by the Random
Forest model may be nonlinear or non-monotonic.

Prediction accuracy may be lower for counties with small annual birth
counts because annual county-level preterm birth percentages are less
stable in low-volume populations.

This tool is intended for public health planning and research and
should not be interpreted as an individual-level clinical risk
assessment.

The model and tool were developed by Prafulla Caringula of the Woodford
County Health Department and Dr. Yu-Sheng Lee of the University of
Illinois Springfield.

If you have questions about this tool, please contact:
Dr. Yu-Sheng Lee | ylee317@uis.edu | 1-217-206-7874.
        """
    )
