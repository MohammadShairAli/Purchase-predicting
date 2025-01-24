import streamlit as st
import joblib
import numpy as np

model = joblib.load('svm_model (2).pkl')
scaler = joblib.load('scaler (1).pkl')

def predict_purchase(gender, age, salary):
    gender_encoded = [1, 0] if gender == "Male" else [0, 1]
    input_data = gender_encoded + [age, salary]
    scaled_data = scaler.transform([input_data])  
    prediction = model.predict(scaled_data) 
    return "Purchased" if prediction[0] == 1 else "Not Purchased"


st.set_page_config(
    page_title="Predicting Purchase",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown("""
    <style>
        /* Set entire app background to white */
        body {
            background-color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }

        /* Set text color to dark blue */
        body, .stMarkdown, .stNumberInput, .stSelectbox, .stTitle, .stSubheader {
            color: #003366; /* Dark blue */
        }

        /* Style the buttons */
        .stButton button {
            background-color: #003366; /* Dark blue */
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
            margin-top: 15px;
        }

        /* Style the input fields */
        .stNumberInput, .stSelectbox {
            font-size: 16px;
        }

        /* Center the input form */
        .block-container {
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("🛒 Predicting Purchase")

# Input fields
st.subheader("Please provide the following details:")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=60, step=1, help="Enter your age (18-60).") 
salary = st.number_input("Estimated Salary", min_value=15000, step=1000, help="Enter your estimated salary (15,000+).")

# Predict button
if st.button("Predict"):
    result = predict_purchase(gender, age, salary)
    st.success(f"Prediction: {result}")
