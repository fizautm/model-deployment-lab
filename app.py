import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model.pkl')

st.title('Drug Prescription Predictor')

# Collect input features from the user
age = st.number_input('Age', min_value=0, max_value=100, value=25)
sex = st.selectbox('Gender', ['Male', 'Female'])
bp = st.selectbox('Blood Pressure', ['HIGH', 'NORMAL', 'LOW'])
cholesterol = st.selectbox('Cholesterol', ['HIGH', 'NORMAL'])
na_to_k = st.number_input('Na to K Ratio', min_value=0.0, max_value=50.0, value=10.0)

# Encoding the categorical inputs
gender_encoded = 1 if sex == 'Male' else 0
bp_encoded = 1 if bp == 'HIGH' else (0 if bp == 'NORMAL' else -1)
cholesterol_encoded = 1 if cholesterol == 'HIGH' else 0

# Ensure that all 5 features are passed in the correct order (same as in training)
input_data = np.array([[age, gender_encoded, bp_encoded, cholesterol_encoded, na_to_k]])

# Make predictions when the button is clicked
if st.button('Predict Drug'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Drug: {prediction}')
