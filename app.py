import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load model
model_path = os.path.join(os.path.dirname(__file__), "churn_model.h5")
model = load_model(model_path)

st.title("Customer Churn Prediction")

# Input fields
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=60, value=12)
balance = st.number_input("Balance", min_value=0.0, value=5000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=5, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Prediction
if st.button("Predict Churn"):
    input_data = np.array([[credit_score, age, tenure, balance,
                            num_of_products, has_cr_card, is_active_member, estimated_salary]])
    prediction = model.predict(input_data)
    st.write("Prediction:", "Churn" if prediction[0][0] > 0.5 else "Not Churn")
