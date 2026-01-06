import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# -----------------------------
# Load model dict
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'churn_xgb_model.joblib')
data = joblib.load(model_path)

model = data['model']
features = data['features']  # list of 39 feature names
ohe = data.get('ohe', None)
label_encoders = data.get('label_encoders', {})
threshold = data.get('threshold') 
print("threshold:", threshold)
st.title("Customer Churn Prediction")
st.write("Fill in the customer details below and hit Predict")

# -----------------------------
# USER INPUTS
# -----------------------------
st.header("Basic Info")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=130, value=10)
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=500)

st.header("Categorical Info")
gender = st.selectbox("Gender", options=["Male", "Female"])
contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
# Add other categorical features here if needed
# Example: OnlineSecurity, TechSupport, StreamingTV, etc.
multiple_lines = st.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"])
online_security = st.selectbox("Online Security", options=["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", options=["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", options=["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", options=["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", options=["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", options=["Yes", "No", "No internet service"])

# -----------------------------
# Predict Button
# -----------------------------
predict_button = st.button("Predict")

if predict_button:
    # 1. Prepare input dict
    input_dict = {
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies
        # Add any additional categorical/numerical features here
    }

    # 2. Convert to DataFrame
    df_input = pd.DataFrame([input_dict])

    # 3. Apply label encoders
    for col, le in label_encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])

    # 4. Apply OneHotEncoder if exists
    if ohe:
        try:
            df_ohe = pd.DataFrame(ohe.transform(df_input[ohe.feature_names_in_]).toarray(),
                                  columns=ohe.get_feature_names_out())
            df_input = df_ohe
        except:
            # In case some expected columns missing, fill zeros
            df_input = pd.DataFrame(0, index=np.arange(1), columns=ohe.get_feature_names_out())

    # 5. Reindex to match model features
    df_input = df_input.reindex(columns=features, fill_value=0)

    # 6. Predict probability and class
    pred_prob = model.predict_proba(df_input)[:, 1]  # probability of churn
    pred_class = (pred_prob >= threshold).astype(int)[0]
    predicted_label = "Churn" if pred_class == 1 else "Not Churn"

    st.balloons()
    st.subheader("Prediction Result")
    st.write(f"Predicted Customer Status: **{predicted_label}**")
    st.write(f"Churn Probability: {pred_prob[0]*100:.2f}%")
else:
    st.write("Click on the Predict button to get the prediction.")
