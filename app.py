import streamlit as st
import pickle
import numpy as np

# Load model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction")

# Input fields
credit_score = st.number_input("Credit Score")
age = st.number_input("Age")
tenure = st.number_input("Tenure")
balance = st.number_input("Balance")
num_of_products = st.number_input("Number of Products")
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary")

# Prediction button
if st.button("Predict Churn"):
    input_data = np.array([[credit_score, age, tenure, balance, num_of_products,
                            has_cr_card, is_active_member, estimated_salary]])
    prediction = model.predict(input_data)
    st.write("Prediction:", "Churn" if prediction[0]==1 else "Not Churn")
