import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import re

# -----------------------------
# Load model dict
# -----------------------------

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'churn_xgb_model.joblib')
data = joblib.load(model_path)

model = data['model']
features = data['features']  # list of 39 feature names
# ohe = data.get('ohe', None)
label_encoders = data.get('label_encoders', {})
threshold = data.get('threshold') 
st.title("Customer Churn Prediction")

# -----------------------------
# USER INPUTS
# -----------------------------

st.header("Basic Information")
customer_id = st.text_input("Enter Customer ID", placeholder="WKSL-2233", help="Format: 4 capital letters+dash+4 digits, e.g., ABCD-1234")

pattern = r"^[A-Z]{4}-\d{4}$"

if customer_id:
    if re.match(pattern, customer_id):
        st.success(f"{customer_id} is a valid ID format!")
    else:
        st.error("Invalid Format! Use format like ABCD-1234")

gender = st.selectbox("Gender", options=["Male", "Female"])
internet_service = st.selectbox("Internet Service", options=["Fiber optic", "DSL", "No"])

st.header("Billing Information")
tenure = st.number_input("Tenure (months)", min_value=1, max_value=130, value=10)
total_charges = st.number_input("Total Charges", min_value=12, max_value=50000)
contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

st.header("Additional Information")
seniorCitizen = st.selectbox("SeniorCitizen", options=["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"])
partner = st.selectbox("Partner", options=["Yes", "No"])
dependents = st.selectbox("Dependents", options=["Yes", "No"])

if internet_service == "Fiber optic" or internet_service == "DSL":

    online_security = st.selectbox("Online Security", options=["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", options=["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", options=["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", options=["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", options=["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", options=["Yes", "No", "No internet service"])
else:
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies =   "No internet service"  




# Add other categorical features here if needed
# Example: OnlineSecurity, TechSupport, StreamingTV, etc.
# -----------------------------
# Predict Button
# -----------------------------
predict_button = st.button("Predict")

if predict_button:
    # 1. Prepare input dict
    input_dict = {
        "customerID": customer_id,             
        "gender": gender,
        "SeniorCitizen": 1 if seniorCitizen == "Yes" else 0,
        "tenure": tenure,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "MultipleLines": multiple_lines,
        "Partner": partner,
        "Dependents": dependents,
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
    # if ohe:
    #     try:
    #         df_ohe = pd.DataFrame(ohe.transform(df_input[ohe.feature_names_in_]).toarray(),
    #                               columns=ohe.get_feature_names_out())
    #         df_input = df_ohe
          
    #     except:
    #         # In case some expected columns missing, fill zeros
    #         df_input = pd.DataFrame(0, index=np.arange(1), columns=ohe.get_feature_names_out())
   
    # 5. Reindex to match model features

    df_input = df_input.reindex(columns=features,fill_value=0)        
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