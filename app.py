import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Student Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model = joblib.load("models/student_risk_model.pkl")

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("🎓 Student Performance Risk Predictor")

st.markdown(
    "### AI-powered prediction of student academic performance risk"
)

st.write(
    "Enter the student's academic, family, and lifestyle information "
    "below to predict whether the student is **At Risk** or **Not At Risk**."
)

st.divider()
# ---------------------------------------------------
# Model Performance
# ---------------------------------------------------

st.header("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎯 Accuracy", "87.34%")

with col2:
    st.metric("🤖 Algorithm", "Random Forest")

with col3:
    st.metric("📚 Training Samples", "316")

with col4:
    st.metric("🧪 Testing Samples", "79")

st.divider()
# ---------------------------------------------------
# Academic Information
# ---------------------------------------------------

st.header("📚 Academic Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=25,
        value=17
    )

    studytime = st.number_input(
        "Study Time (1–4)",
        min_value=1,
        max_value=4,
        value=2
    )

    failures = st.number_input(
        "Past Class Failures",
        min_value=0,
        max_value=4,
        value=0
    )

with col2:
    Medu = st.number_input(
        "Mother's Education (0–4)",
        min_value=0,
        max_value=4,
        value=2
    )

    traveltime = st.number_input(
        "Travel Time (1–4)",
        min_value=1,
        max_value=4,
        value=2
    )

    G1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

with col3:
    Fedu = st.number_input(
        "Father's Education (0–4)",
        min_value=0,
        max_value=4,
        value=2
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=5
    )

    G2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10
    )

# ---------------------------------------------------
# Family & Support Information
# ---------------------------------------------------

st.divider()

st.header("🏠 Family & Support")

col1, col2, col3 = st.columns(3)

with col1:
    schoolsup = st.selectbox(
        "Extra Educational Support",
        ["no", "yes"]
    )

    famsup = st.selectbox(
        "Family Educational Support",
        ["no", "yes"]
    )

with col2:
    paid = st.selectbox(
        "Extra Paid Classes",
        ["no", "yes"]
    )

    higher = st.selectbox(
        "Wants Higher Education",
        ["no", "yes"]
    )

with col3:
    activities = st.selectbox(
        "Extra-Curricular Activities",
        ["no", "yes"]
    )

    internet = st.selectbox(
        "Internet Access at Home",
        ["no", "yes"]
    )

# ---------------------------------------------------
# Lifestyle Information
# ---------------------------------------------------

st.divider()

st.header("🌱 Lifestyle Information")

col1, col2, col3 = st.columns(3)

with col1:
    freetime = st.number_input(
        "Free Time After School (1–5)",
        min_value=1,
        max_value=5,
        value=3
    )

with col2:
    goout = st.number_input(
        "Going Out With Friends (1–5)",
        min_value=1,
        max_value=5,
        value=3
    )

with col3:
    health = st.number_input(
        "Current Health Status (1–5)",
        min_value=1,
        max_value=5,
        value=3
    )

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

st.divider()

st.subheader("🔮 Student Risk Prediction")

predict_button = st.button(
    "🔮 Predict Student Risk",
    use_container_width=True
)

if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "Medu": [Medu],
        "Fedu": [Fedu],
        "traveltime": [traveltime],
        "studytime": [studytime],
        "failures": [failures],
        "schoolsup": [schoolsup],
        "famsup": [famsup],
        "paid": [paid],
        "activities": [activities],
        "higher": [higher],
        "internet": [internet],
        "freetime": [freetime],
        "goout": [goout],
        "health": [health],
        "absences": [absences],
        "G1": [G1],
        "G2": [G2]
    })

    # Apply same encoding used during training
    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Match training columns
    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    # Get class names and probabilities
    class_names = model.classes_

    probability_dict = dict(zip(class_names, probabilities))

    at_risk_probability = probability_dict.get("At Risk", 0)
    not_at_risk_probability = probability_dict.get("Not At Risk", 0)

    # ---------------------------------------------------
    # Result
    # ---------------------------------------------------

    st.divider()

    if prediction == "At Risk":

        confidence = at_risk_probability

        st.error("## ⚠️ At Risk")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.1%}"
        )

        st.write(
             "The model predicts that this student may be at risk "
             "of poor academic performance."
        )

        st.info(
            "💡 Consider providing additional academic support, "
            "monitoring attendance, and encouraging consistent study habits."
        )

    else:

        confidence = not_at_risk_probability

        st.success("## ✅ Not At Risk")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.1%}"
        )

        st.write(
            "The model predicts that this student is not currently "
            "at risk of poor academic performance."
        )

        st.info(
            "💡 Continue maintaining consistent study habits, "
            "attendance, and academic performance."
        )

    # Probability breakdown
    st.subheader("📊 Prediction Probability")

    probability_data = pd.DataFrame({
        "Risk Category": ["At Risk", "Not At Risk"],
        "Probability": [
            at_risk_probability,
            not_at_risk_probability
        ]
    })

    st.bar_chart(
        probability_data.set_index("Risk Category")
    )