"""
Stroke Risk Predictor - Streamlit App
Model: Logistic Regression pipeline (StandardScaler + LogisticRegression)
Features (order matters): age, heart_disease, hypertension, avg_glucose_level
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "stroke_predictor_model.pkl"

st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS (medical theme)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef7fb 0%, #ffffff 60%);
    }
    .hero {
        background: linear-gradient(135deg, #0b7285 0%, #1098ad 100%);
        padding: 2rem 1.5rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(16, 152, 173, 0.25);
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.1rem; }
    .hero p  { margin: 0.4rem 0 0; opacity: 0.92; font-size: 1rem; }

    .card {
        background: white;
        padding: 1.4rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        border: 1px solid #e3f1f5;
        margin-bottom: 1.2rem;
    }
    .card h3 { color: #0b7285; margin-top: 0; }

    .result-box {
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-top: 0.5rem;
    }
    .low      { background: linear-gradient(135deg, #2f9e44, #51cf66); }
    .moderate { background: linear-gradient(135deg, #f08c00, #fcc419); color: #3b2a00; }
    .high     { background: linear-gradient(135deg, #c92a2a, #fa5252); }
    .result-box h2 { margin: 0; font-size: 1.8rem; }
    .result-box p  { margin: 0.3rem 0 0; font-size: 1rem; }

    .bar-bg {
        background: #e9ecef;
        border-radius: 999px;
        height: 18px;
        overflow: hidden;
        margin-top: 0.8rem;
    }
    .bar-fill { height: 100%; border-radius: 999px; }

    .disclaimer {
        background: #fff4e6;
        border-left: 5px solid #f08c00;
        padding: 0.9rem 1rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #5c3d00;
        margin-top: 1.2rem;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #0b7285, #1098ad);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #095c6b, #0c8599);
        color: white;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        f"Model file not found! Please upload '{MODEL_PATH.name}' into your "
        "project folder or update MODEL_PATH in app.py."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:  # version mismatch, corrupt file, etc.
    st.error(f"Model could not be loaded: {e}")
    st.info(
        "Tip: make sure the scikit-learn version in requirements.txt matches "
        "the version used while training the model."
    )
    st.stop()


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🩺 Stroke Risk Predictor</h1>
        <p>Machine-learning based early risk screening using key health indicators</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# INPUT FORM
# ----------------------------------------------------------------------------
st.markdown('<div class="card"><h3>📋 Patient Details</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", min_value=1, max_value=100, value=50)
    hypertension_label = st.radio(
        "Hypertension (High BP)", ["No", "Yes"], horizontal=True
    )

with col2:
    glucose = st.number_input(
        "Average Glucose Level (mg/dL)",
        min_value=40.0,
        max_value=400.0,
        value=100.0,
        step=1.0,
        help="Normal fasting range is roughly 70-100 mg/dL.",
    )
    heart_label = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)

st.markdown("</div>", unsafe_allow_html=True)

hypertension = 1 if hypertension_label == "Yes" else 0
heart_disease = 1 if heart_label == "Yes" else 0

predict_clicked = st.button("🔍 Check Stroke Risk")

# ----------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------
if predict_clicked:
    # Column names and order must match the training data exactly
    input_df = pd.DataFrame(
        {
            "age": [age],
            "heart_disease": [heart_disease],
            "hypertension": [hypertension],
            "avg_glucose_level": [glucose],
        }
    )

    stroke_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(input_df)[0][stroke_index])
    percent = round(probability * 100, 1)

    if percent < 30:
        level, css, icon = "Low Risk", "low", "✅"
        message = "Your indicators suggest a lower likelihood of stroke."
        color = "#2f9e44"
    elif percent < 60:
        level, css, icon = "Moderate Risk", "moderate", "⚠️"
        message = "Some risk factors are present. Consider a health check-up."
        color = "#f08c00"
    else:
        level, css, icon = "High Risk", "high", "🚨"
        message = "Elevated risk detected. Please consult a doctor soon."
        color = "#c92a2a"

    st.markdown(
        f"""
        <div class="result-box {css}">
            <h2>{icon} {level}</h2>
            <p>{message}</p>
            <h1 style="margin:0.6rem 0 0;">{percent}%</h1>
            <p>Estimated stroke probability</p>
        </div>
        <div class="bar-bg">
            <div class="bar-fill" style="width:{percent}%; background:{color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary of inputs
    st.markdown('<div class="card" style="margin-top:1.2rem;"><h3>🧾 Input Summary</h3>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Age", f"{age} yrs")
    m2.metric("Glucose", f"{glucose:.0f} mg/dL")
    m3.metric("Hypertension", hypertension_label)
    m4.metric("Heart Disease", heart_label)
    st.markdown("</div>", unsafe_allow_html=True)

    # Health tips
    tips = []
    if glucose > 125:
        tips.append("Your glucose level is high. Reduce sugar intake and get diabetes screening.")
    if hypertension:
        tips.append("Monitor blood pressure regularly and limit salt in your diet.")
    if heart_disease:
        tips.append("Follow your cardiologist's advice and take prescribed medicines on time.")
    if age >= 55:
        tips.append("Regular yearly health check-ups are recommended after age 55.")
    tips.append("Exercise 30 minutes daily, avoid smoking and alcohol, and sleep well.")

    st.markdown('<div class="card"><h3>💡 Health Tips</h3>', unsafe_allow_html=True)
    for t in tips:
        st.markdown(f"- {t}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.write(
        "This app uses a **Logistic Regression** model trained on the "
        "Healthcare Stroke dataset to estimate stroke risk."
    )
    st.markdown("**Features used:**")
    st.markdown("- Age\n- Hypertension\n- Heart disease\n- Average glucose level")
    st.markdown("---")
    st.markdown("**Risk levels**")
    st.markdown("🟢 Below 30% - Low\n\n🟠 30% to 60% - Moderate\n\n🔴 Above 60% - High")

# ----------------------------------------------------------------------------
# DISCLAIMER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer">
        <b>⚕️ Medical Disclaimer:</b> This tool is for educational purposes only.
        It is not a medical diagnosis and cannot replace professional medical advice.
        Always consult a qualified doctor for health concerns.
    </div>
    """,
    unsafe_allow_html=True,
)
