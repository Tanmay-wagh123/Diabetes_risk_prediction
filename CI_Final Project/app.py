import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scale.pkl')

# Streamlit app UI elements
st.title("Diabetes Risk Prediction")
st.markdown("""
    <style>
        .main {
            background-color: #f0f0f0;
            color: #333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for input parameters
st.sidebar.title("Input Parameters")
age = st.sidebar.slider('Age', min_value=1, max_value=120, value=30, step=1)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
hypertension = st.sidebar.selectbox('Do you have Hypertension?', ['No', 'Yes'])
heart_disease = st.sidebar.selectbox('Do you have Heart Disease?', ['No', 'Yes'])
smoking_history = st.sidebar.selectbox('Do you have a Smoking History?', ['No', 'Yes'])
bmi = st.sidebar.slider('BMI', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
HbA1c_level = st.sidebar.slider('HbA1c Level', min_value=4.0, max_value=10.0, value=5.5, step=0.1)
blood_glucose_level = st.sidebar.slider('Blood Glucose Level', min_value=50.0, max_value=300.0, value=100.0, step=1.0)

# Display the values entered by the user
st.sidebar.markdown("### You entered:")
st.sidebar.write(f"**Age**: {age}")
st.sidebar.write(f"**Gender**: {gender}")
st.sidebar.write(f"**Hypertension**: {hypertension}")
st.sidebar.write(f"**Heart Disease**: {heart_disease}")
st.sidebar.write(f"**Smoking History**: {smoking_history}")
st.sidebar.write(f"**BMI**: {bmi}")
st.sidebar.write(f"**HbA1c Level**: {HbA1c_level}")
st.sidebar.write(f"**Blood Glucose Level**: {blood_glucose_level}")

# Map user input to numerical values
gender = 1 if gender == 'Male' else 0
hypertension = 1 if hypertension == 'Yes' else 0
heart_disease = 1 if heart_disease == 'Yes' else 0
smoking_history = 1 if smoking_history == 'Yes' else 0

# Create a DataFrame for the input values
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'smoking_history': [smoking_history],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level]
})

# Scale the input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Make prediction using the trained model
prediction = model.predict(input_data_scaled)
prediction_prob = model.predict_proba(input_data_scaled)  # Get the probability

# Display the prediction
st.subheader("Prediction Result:")
if prediction == 1:
    st.warning("⚠️ High risk of diabetes. Please consult a doctor.")
else:
    st.success("🎉 Low risk of diabetes. Keep up the good work!")

# Display the prediction probabilities
st.markdown("### Prediction Probabilities:")
st.write(f"**Probability of 'Low risk'**: {prediction_prob[0][0] * 100:.2f}%")
st.write(f"**Probability of 'High risk'**: {prediction_prob[0][1] * 100:.2f}%")

# Additional footer with credits or links
st.markdown("---")
st.markdown("Made with ❤️ by Your Name | [GitHub Repository](https://github.com/your-repo)")
