import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="NeuroCare AI | Stroke Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F4F7FB;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102A43, #173F56);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    background: linear-gradient(120deg, #102A43, #147D83);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    font-weight: 800;
}

.hero p {
    color: #D9F0F0;
    font-size: 15px;
}

.section-card {
    background: white;
    padding: 23px;
    border-radius: 15px;
    border: 1px solid #E5ECF2;
    margin-bottom: 16px;
}

.section-title {
    color: #12334B;
    font-weight: 700;
    font-size: 23px;
}

.info-card {
    background: #EAF5F5;
    border-left: 4px solid #168B8B;
    padding: 15px;
    border-radius: 8px;
    color: #173F56;
    margin: 10px 0;
}

.result-success {
    background: #E5F7EF;
    border: 1px solid #A6DFC1;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.result-warning {
    background: #FFF0F0;
    border: 1px solid #F2B6B6;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

div.stButton > button {
    background: linear-gradient(90deg, #147D83, #20A99D);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 12px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #105D68;
    color: white;
}

.footer {
    text-align: center;
    color: #7B8B99;
    font-size: 12px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

MODEL_PATH = "stroke_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 0;">
        <h1 style="font-size:42px;">🧠</h1>
        <h2 style="color:white;">NeuroCare AI</h2>
        <p style="font-size:12px;color:#B9D6E5;">
        INTELLIGENT HEALTH ANALYTICS
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "📖 About Stroke",
            "📊 Dataset Information",
            "⚙️ Methodology",
            "🔍 Stroke Prediction",
            "📈 Model Information",
            "📝 Conclusion"
        ]
    )

    st.markdown("---")

    st.caption("Machine Learning Project")
    st.caption("Binary Classification System")

# -------------------------------------------------
# COMMON HEADER
# -------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>NeuroCare AI</h1>
    <p>AI-Powered Stroke Prediction System</p>
    <p>
    An educational machine learning application
    designed to demonstrate binary classification
    using patient health parameters.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

if page == "🏠 Home":

    st.markdown(
        '<p class="section-title">Welcome to NeuroCare AI</p>',
        unsafe_allow_html=True
    )

    st.write(
        "NeuroCare AI is a machine learning based "
        "Stroke Prediction System. It uses selected "
        "health parameters to generate a binary "
        "classification output."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-card">
        <h2>🧠</h2>
        <h3>Machine Learning</h3>
        <p>Uses a trained ML model for prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
        <h2>📋</h2>
        <h3>4 Input Features</h3>
        <p>Analyzes four selected health parameters.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
        <h2>🎯</h2>
        <h3>Binary Output</h3>
        <p>Predicts Stroke or Not Stroke.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Project Objectives")

    st.markdown("""
    - Understand the application of machine learning
      in healthcare-related data analysis.
    - Accept patient health parameters as input.
    - Apply a trained classification model.
    - Display the predicted output in a simple interface.
    """)

    st.markdown("""
    <div class="info-card">
    <b>Project Goal:</b> To demonstrate how machine
    learning classification can be applied to selected
    health-related features.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Explore the Application")

    st.info(
        "Use the sidebar navigation to explore the "
        "dataset, methodology, prediction system and "
        "model information."
    )

# -------------------------------------------------
# ABOUT STROKE
# -------------------------------------------------

elif page == "📖 About Stroke":

    st.markdown(
        '<p class="section-title">Understanding Stroke</p>',
        unsafe_allow_html=True
    )

    st.write("""
    A stroke is a medical condition that occurs when
    the blood supply to a part of the brain is
    interrupted or reduced. It requires prompt
    medical attention.
    """)

    st.subheader("Health Parameters Used in This Project")

    st.markdown("""
    <div class="section-card">
    <h3>👤 Age</h3>
    Age represents the number of years a person has lived.
    It is one of the input features used by the model.
    </div>

    <div class="section-card">
    <h3>❤️ Heart Disease</h3>
    Indicates whether the patient has a history of
    heart disease. It is represented as a binary value.
    </div>

    <div class="section-card">
    <h3>🩺 Hypertension</h3>
    Represents whether the patient has hypertension
    (high blood pressure), encoded as a binary value.
    </div>

    <div class="section-card">
    <h3>💧 Average Glucose Level</h3>
    Represents the average glucose level in the
    patient's blood, measured in mg/dL.
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "This project is for educational purposes only. "
        "It cannot diagnose, prevent or rule out stroke. "
        "For medical concerns, consult a qualified doctor."
    )

# -------------------------------------------------
# DATASET INFORMATION
# -------------------------------------------------

elif page == "📊 Dataset Information":

    st.markdown(
        '<p class="section-title">Dataset Information</p>',
        unsafe_allow_html=True
    )

    st.write("""
    The model uses four selected input features
    to classify the target variable into two classes.
    """)

    feature_data = pd.DataFrame({
        "Feature": [
            "age",
            "heart_disease",
            "hypertension",
            "avg_glucose_level"
        ],
        "Description": [
            "Patient age in years",
            "Presence of heart disease",
            "Presence of hypertension",
            "Average glucose level in mg/dL"
        ],
        "Input Type": [
            "Numeric",
            "Binary (0/1)",
            "Binary (0/1)",
            "Numeric"
        ]
    })

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Target Variable")

    st.markdown("""
    | Target Value | Meaning |
    |---|---|
    | 0 | Not Stroke |
    | 1 | Stroke |
    """)

    st.info(
        "The target encoding shown above assumes that "
        "your trained model uses 0 = Not Stroke and "
        "1 = Stroke. Verify this with your dataset."
    )

# -------------------------------------------------
# METHODOLOGY
# -------------------------------------------------

elif page == "⚙️ Methodology":

    st.markdown(
        '<p class="section-title">Project Methodology</p>',
        unsafe_allow_html=True
    )

    steps = [
        ("01", "Data Collection",
         "Collect the dataset containing patient health information."),
        ("02", "Data Preprocessing",
         "Prepare the selected features in the format expected by the model."),
        ("03", "Feature Selection",
         "Use age, heart_disease, hypertension and avg_glucose_level."),
        ("04", "Model Training",
         "Train a machine learning classification model."),
        ("05", "Model Evaluation",
         "Evaluate performance using appropriate classification metrics."),
        ("06", "Prediction",
         "Pass new input values to the trained model.")
    ]

    for number, title, description in steps:
        st.markdown(f"""
        <div class="section-card">
            <h3 style="color:#168B8B;">STEP {number} — {title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Classification Type")

    st.write("""
    This is a Binary Classification problem because
    the model produces one of two possible classes:
    Stroke or Not Stroke.
    """)

# -------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------

elif page == "🔍 Stroke Prediction":

    st.markdown(
        '<p class="section-title">Patient Health Assessment</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the four health parameters below "
        "to generate a prediction from the trained model."
    )

    if model is None:
        st.error(
            "Model file not found! Please upload "
            "'stroke_model.pkl' into your project folder "
            "or update MODEL_PATH in app.py."
        )

    else:

        with st.form("stroke_prediction_form"):

            st.subheader("Patient Information")

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input(
                    "Age (Years)",
                    min_value=0.0,
                    max_value=120.0,
                    value=45.0,
                    step=1.0
                )

                heart_disease = st.selectbox(
                    "Heart Disease",
                    options=["No", "Yes"]
                )

            with col2:
                hypertension = st.selectbox(
                    "Hypertension",
                    options=["No", "Yes"]
                )

                avg_glucose_level = st.number_input(
                    "Average Glucose Level (mg/dL)",
                    min_value=0.0,
                    max_value=500.0,
                    value=100.0,
                    step=0.1
                )

            st.markdown("---")

            submitted = st.form_submit_button(
                "🔍 Predict Stroke",
                use_container_width=True
            )

        if submitted:

            input_data = pd.DataFrame(
                [[
                    age,
                    1 if heart_disease == "Yes" else 0,
                    1 if hypertension == "Yes" else 0,
                    avg_glucose_level
                ]],
                columns=[
                    "age",
                    "heart_disease",
                    "hypertension",
                    "avg_glucose_level"
                ]
            )

            try:
                prediction = model.predict(input_data)[0]

                st.markdown("---")
                st.subheader("Prediction Result")

                if prediction == 1:

                    st.markdown("""
                    <div class="result-warning">
                    <h1>⚠️</h1>
                    <h2>Stroke</h2>
                    <p>The model predicted class 1.</p>
                    </div>
                    """, unsafe_allow_html=True)

                elif prediction == 0:

                    st.markdown("""
                    <div class="result-success">
                    <h1>✓</h1>
                    <h2>Not Stroke</h2>
                    <p>The model predicted class 0.</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning(
                        f"Model returned an unexpected class: {prediction}"
                    )

                st.caption(
                    "This is a machine learning prediction, "
                    "not a medical diagnosis. Do not use it "
                    "to make healthcare decisions."
                )

            except Exception as e:
                st.error(
                    f"Prediction error: {e}"
                )

# -------------------------------------------------
# MODEL INFORMATION
# -------------------------------------------------

elif page == "📈 Model Information":

    st.markdown(
        '<p class="section-title">Model Information</p>',
        unsafe_allow_html=True
    )

    st.write("""
    This application uses a trained machine learning
    classification model to generate predictions.
    """)

    st.subheader("Model Input Features")

    st.code("""
age
heart_disease
hypertension
avg_glucose_level
""")

    st.subheader("Output Classes")

    st.success("Class 0: Not Stroke")
    st.error("Class 1: Stroke")

    st.subheader("Model Performance")

    st.info("""
    Model evaluation metrics such as Accuracy,
    Precision, Recall, F1-score and Confusion Matrix
    should be added here using the actual evaluation
    results from your training notebook.
    """)

    st.write(
        "The model algorithm and evaluation values "
        "should match the actual trained model."
    )

# -------------------------------------------------
# CONCLUSION
# -------------------------------------------------

elif page == "📝 Conclusion":

    st.markdown(
        '<p class="section-title">Project Conclusion</p>',
        unsafe_allow_html=True
    )

    st.write("""
    NeuroCare AI demonstrates the use of machine
    learning for binary classification using selected
    patient health parameters.

    The application provides an interactive interface
    for entering input values and viewing the output
    generated by the trained model.
    """)

    st.subheader("Future Scope")

    st.markdown("""
    - Add more validated health-related features.
    - Improve model evaluation and validation.
    - Compare different classification algorithms.
    - Improve the user interface and accessibility.
    - Conduct further validation before any clinical use.
    """)

    st.success(
        "Project demonstration completed!"
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("""
<div class="footer">
    NeuroCare AI | Stroke Prediction System<br>
    Built with Python, Machine Learning and Streamlit<br>
    Educational use only — Not a medical diagnostic tool
</div>
""", unsafe_allow_html=True)
