import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction")

# Example inputs
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)")
monthly_charges = st.number_input("Monthly Charges")

if st.button("Predict"):
    x = np.array([[tenure, monthly_charges]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]

    if pred == 1:
        st.error("⚠️ This customer is likely to churn!")
    else:
        st.success("💚 This customer is safe.")
