"""
Stroke Risk Prediction - Streamlit Frontend
--------------------------------------------------------
Run:  streamlit run app.py

Files needed in the same folder:
    stroke_predictor_model.pkl -> trained sklearn Pipeline
                                  (StandardScaler + LogisticRegression)

The scaler is already inside the pipeline, so no separate scaler file is needed.
If the model file is not found, the app runs in DEMO mode with a simple
rule-based predictor so you can still test the UI.
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Plotly is optional: used only for the gauge chart
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "stroke_predictor_model.pkl"

# Order MUST match the training order:
# X = df[["age", "heart_disease", "hypertension", "avg_glucose_level"]]
FEATURE_NAMES = ["age", "heart_disease", "hypertension", "avg_glucose_level"]

FEATURE_INFO = [
    ("Age", "Years", "1 - 100", "Patient's age in years"),
    ("Heart Disease", "0 / 1", "0 = No, 1 = Yes", "Whether the patient has any heart disease"),
    ("Hypertension", "0 / 1", "0 = No, 1 = Yes", "Whether the patient has high blood pressure"),
    ("Average Glucose Level", "mg/dL", "40 - 400", "Average blood glucose level"),
]

STROKE_LABEL = 1         # value the model returns for "Stroke"
RISK_THRESHOLD = 0.50    # probability at or above which result is "Stroke"
LOW_CUTOFF = 0.30        # below this -> Low risk
HIGH_CUTOFF = 0.60       # above this -> High risk

# ----------------------------------------------------------------------
# 2. PAGE SETUP + CUSTOM CSS
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(180deg, #eef7fb 0%, #ffffff 55%); }
        .main-title {
            font-size: 2.4rem; font-weight: 800; text-align: center;
            background: linear-gradient(90deg, #0b7285, #1098ad);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .sub-title {
            text-align: center; color: #6b7280; font-size: 1.05rem;
            margin-top: 0; margin-bottom: 1.5rem;
        }
        .result-card {
            padding: 1.6rem; border-radius: 16px; text-align: center;
            color: white; box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        }
        .low-card      { background: linear-gradient(135deg, #2f9e44, #51cf66); }
        .moderate-card { background: linear-gradient(135deg, #f08c00, #fcc419); color: #3b2a00; }
        .high-card     { background: linear-gradient(135deg, #c92a2a, #fa5252); }
        .result-card h1 { margin: 0; font-size: 2.3rem; }
        .result-card p  { margin: 0.3rem 0 0 0; font-size: 1.1rem; opacity: 0.95; }
        .tip-box {
            background: #e7f5f8; border-left: 5px solid #0b7285;
            padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;
            color: #12343b;
        }
        .disclaimer {
            background: #fff4e6; border-left: 5px solid #f08c00;
            padding: 0.9rem 1rem; border-radius: 8px;
            font-size: 0.88rem; color: #5c3d00; margin-top: 1rem;
        }
        div.stButton > button, div.stFormSubmitButton > button {
            width: 100%; border-radius: 10px; height: 3rem;
            font-weight: 700; font-size: 1.05rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 3. MODEL LOADING
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: Path):
    if not path.exists():
        return None, None
    try:
        return joblib.load(path), None
    except Exception as e:  # version mismatch, corrupt file, etc.
        return None, str(e)


model, load_error = load_model(MODEL_PATH)
DEMO_MODE = model is None


def predict(df: pd.DataFrame):
    """Return (labels, stroke_probabilities) for a dataframe of features."""
    X = df[FEATURE_NAMES].astype(float)

    if DEMO_MODE:
        # Simple weighted score, only for testing the UI
        score = (
            0.40 * (X["age"] / 100)
            + 0.20 * X["heart_disease"]
            + 0.20 * X["hypertension"]
            + 0.20 * ((X["avg_glucose_level"] - 70) / 200).clip(0, 1)
        ).values
        proba = np.clip(score, 0, 1)
    else:
        # Pass the DataFrame so column names match what the pipeline saw
        stroke_idx = list(model.classes_).index(STROKE_LABEL)
        proba = model.predict_proba(X)[:, stroke_idx]

    labels = np.where(proba >= RISK_THRESHOLD, "Stroke", "No Stroke")
    return labels, proba


def risk_level(prob: float):
    if prob < LOW_CUTOFF:
        return "Low Risk", "low-card", "✅", "Indicators suggest a lower likelihood of stroke."
    if prob < HIGH_CUTOFF:
        return "Moderate Risk", "moderate-card", "⚠️", "Some risk factors present. A health check-up is advised."
    return "High Risk", "high-card", "🚨", "Elevated risk detected. Please consult a doctor soon."


def make_gauge(prob: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Stroke Risk Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0b7285"},
                "steps": [
                    {"range": [0, LOW_CUTOFF * 100], "color": "#b2f2bb"},
                    {"range": [LOW_CUTOFF * 100, HIGH_CUTOFF * 100], "color": "#ffec99"},
                    {"range": [HIGH_CUTOFF * 100, 100], "color": "#ffc9c9"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": RISK_THRESHOLD * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def suggestions(row: dict):
    """Simple rule-based health tips based on the inputs."""
    tips = []
    if row["avg_glucose_level"] > 125:
        tips.append("🍬 Glucose level is high. Cut down sugar and get a diabetes screening.")
    if row["hypertension"] == 1:
        tips.append("🩸 Monitor blood pressure regularly and reduce salt in your diet.")
    if row["heart_disease"] == 1:
        tips.append("❤️ Follow your cardiologist's advice and take medicines on time.")
    if row["age"] >= 55:
        tips.append("📅 Yearly health check-ups are recommended after age 55.")
    tips.append("🏃 Exercise 30 minutes daily, avoid smoking and alcohol, and sleep well.")
    return tips


# ----------------------------------------------------------------------
# 4. SESSION STATE (prediction history)
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------------
# 5. SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "🔮 Single Prediction", "📂 Batch Prediction", "🕘 History", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if load_error:
        st.error("Model could not be loaded. Check the scikit-learn version.")
        st.caption(load_error)
    elif DEMO_MODE:
        st.warning(f"Demo mode: `{MODEL_PATH.name}` not found.")
    else:
        st.success("Model loaded ✔")
    st.caption(
        f"Stroke threshold: {int(RISK_THRESHOLD * 100)}%  \n"
        f"Low < {int(LOW_CUTOFF * 100)}% | High > {int(HIGH_CUTOFF * 100)}%"
    )

# ----------------------------------------------------------------------
# 6. HEADER
# ----------------------------------------------------------------------
st.markdown('<p class="main-title">🩺 Stroke Risk Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Machine Learning based early screening of stroke risk</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 7. PAGES
# ----------------------------------------------------------------------
# ---------- HOME ----------
if page == "🏠 Home":
    c1, c2, c3 = st.columns(3)
    c1.metric("Input Features", len(FEATURE_NAMES))
    c2.metric("Output Classes", "Stroke / No Stroke")
    c3.metric("Predictions Made", len(st.session_state.history))

    st.markdown("### How it works")
    st.markdown(
        """
        1. Enter the patient's details in **Single Prediction**.
        2. The model returns **Stroke** or **No Stroke**, with a risk probability.
        3. Upload a CSV in **Batch Prediction** to get results for many patients.
        4. View and download earlier predictions in **History**.
        """
    )

    st.markdown("### Features used")
    st.dataframe(
        pd.DataFrame(FEATURE_INFO, columns=["Feature", "Unit", "Range / Values", "Description"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="disclaimer">
            <b>⚕️ Medical Disclaimer:</b> This tool is for educational purposes only.
            It is not a medical diagnosis. Always consult a qualified doctor.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- SINGLE PREDICTION ----------
elif page == "🔮 Single Prediction":
    st.subheader("Enter Patient Details")

    with st.form("predict_form"):
        patient_name = st.text_input("Patient Name (optional)", placeholder="e.g. Rahul Sharma")

        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age (years)", 1, 100, 50, 1, help="Patient's age in years")
            hypertension = st.number_input(
                "Hypertension (High BP)", min_value=0, max_value=1, value=0, step=1,
                help="Enter 0 or 1  (0 = No high BP, 1 = Has high BP)",
            )
        with col2:
            glucose = st.slider(
                "Average Glucose Level (mg/dL)", 40.0, 400.0, 100.0, 1.0,
                help="Normal fasting range is roughly 70-100 mg/dL",
            )
            heart_disease = st.number_input(
                "Heart Disease", min_value=0, max_value=1, value=0, step=1,
                help="Enter 0 or 1  (0 = No heart disease, 1 = Has heart disease)",
            )

        submitted = st.form_submit_button("🔮 Predict Result")

    if submitted:
        values = {
            "age": age,
            "heart_disease": int(heart_disease),
            "hypertension": int(hypertension),
            "avg_glucose_level": glucose,
        }
        df_in = pd.DataFrame([values])
        labels, proba = predict(df_in)
        label, prob = labels[0], float(proba[0])
        level, css, emoji, message = risk_level(prob)

        st.markdown("---")
        left, right = st.columns([1, 1])

        with left:
            name_txt = patient_name if patient_name else "Patient"
            st.markdown(
                f"""
                <div class="result-card {css}">
                    <h1>{emoji} {level.upper()}</h1>
                    <p>Result for {name_txt}: <b>{label}</b></p>
                    <p>{message}</p>
                    <p><b>Stroke probability: {prob * 100:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### 💡 Health Tips")
            for t in suggestions(values):
                st.markdown(f'<div class="tip-box">{t}</div>', unsafe_allow_html=True)

        with right:
            if PLOTLY_OK:
                st.plotly_chart(make_gauge(prob), use_container_width=True)
            else:
                st.progress(min(max(prob, 0.0), 1.0), text=f"Stroke probability: {prob * 100:.1f}%")

            st.markdown("#### Input Summary")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Feature": ["Age", "Heart Disease", "Hypertension", "Avg Glucose Level"],
                        "Value": [age, int(heart_disease), int(hypertension), glucose],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            """
            <div class="disclaimer">
                <b>⚕️ Medical Disclaimer:</b> This is not a diagnosis. Please consult a doctor.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.history.append(
            {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Name": patient_name or "-",
                **values,
                "Result": label,
                "Risk Level": level,
                "Stroke Probability (%)": round(prob * 100, 2),
            }
        )

# ---------- BATCH PREDICTION ----------
elif page == "📂 Batch Prediction":
    st.subheader("Upload CSV for Bulk Prediction")
    st.caption("Your CSV must contain these columns: " + ", ".join(f"`{n}`" for n in FEATURE_NAMES))
    st.caption("`heart_disease` and `hypertension` should be 0 (No) or 1 (Yes).")

    template = pd.DataFrame([{"age": 50, "heart_disease": 0, "hypertension": 1, "avg_glucose_level": 120.0}])
    st.download_button(
        "⬇️ Download CSV Template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="patient_template.csv",
        mime="text/csv",
    )

    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file is not None:
        try:
            data = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            st.stop()

        missing = [c for c in FEATURE_NAMES if c not in data.columns]
        if missing:
            st.error(f"These columns are missing: {missing}")
        else:
            data = data.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
            labels, proba = predict(data)
            data["Predicted Result"] = labels
            data["Stroke Probability (%)"] = (proba * 100).round(2)

            total = len(data)
            stroke_n = int((data["Predicted Result"] == "Stroke").sum())
            safe_n = total - stroke_n

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Patients", total)
            m2.metric("Predicted Stroke", stroke_n)
            m3.metric("Predicted No Stroke", safe_n)
            m4.metric("Stroke Rate", f"{(stroke_n / total * 100) if total else 0:.1f}%")

            tab1, tab2 = st.tabs(["📋 Results Table", "📊 Charts"])
            with tab1:
                st.dataframe(data, use_container_width=True)
            with tab2:
                st.bar_chart(data["Predicted Result"].value_counts())
                st.caption("Stroke vs No Stroke count")

            st.download_button(
                "⬇️ Download Predictions",
                data.to_csv(index=False).encode("utf-8"),
                file_name="stroke_predictions.csv",
                mime="text/csv",
            )

# ---------- HISTORY ----------
elif page == "🕘 History":
    st.subheader("Prediction History (this session)")
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True)
        col_a, col_b = st.columns(2)
        col_a.download_button(
            "⬇️ Download History",
            hist.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
        if col_b.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions have been made yet.")

# ---------- ABOUT ----------
else:
    st.subheader("About this Project")
    st.markdown(
        """
        **Project:** Stroke Risk Prediction

        **Goal:** To estimate the likelihood of a stroke using a few key health indicators.

        **Tech Stack**
        - Python, Pandas, NumPy
        - Scikit-learn (Logistic Regression with StandardScaler pipeline)
        - Streamlit (frontend)
        - Plotly (charts)

        **Model Details**
        - Type: Binary classification (Stroke = 1, No Stroke = 0)
        - Features: age, heart_disease, hypertension, avg_glucose_level
        - Dataset: Healthcare Stroke Dataset
        - Output: Class label + probability of stroke

        > Add your accuracy, recall, and team members' names here.

        **⚕️ Disclaimer:** This project is for educational purposes only and does
        not replace professional medical advice.
        """
    )

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
