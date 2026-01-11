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
features = data['features']
ohe = data["ohe"]
label_encoders = data["label_encoders"]
mul_cols = data["mul_category"]
bin_cols = data["bin_category"]
threshold = data.get('threshold') 
st.title("Customer Churn Prediction")


#USER INPUTS

st.header("Basic Information")
customer_id = st.text_input("Enter Customer ID", placeholder="WKSL-2233", help="Format: 4 capital letters+dash+4 digits, e.g., ABCD-1234")

pattern = r"^[A-Z]{4}-\d{4}$"

if customer_id:
    if re.match(pattern, customer_id):
        st.success(f"{customer_id} is a valid ID format!")
    else:
        st.error("Invalid Format! Use format like ABCD-1234")

gender = st.selectbox("Gender", options=["Male", "Female"])
internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
st.header("Billing Information")
tenure = st.number_input("Tenure (months)", min_value=0, max_value=130, value=10)
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

#Predict Button

predict_button = st.button("Predict")

if predict_button:
    # 1. Prepare input dict
    input_dict = {
        "CustomerID": customer_id,
        "SeniorCitizen": 1 if seniorCitizen == "Yes" else 0,    
        "Gender": gender,
        "Tenure": tenure,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Partner": partner,
        "Dependents": dependents,
        # Add any additional categorical/numerical features here
    }

    # 2. Convert to DataFrame
    df_input = pd.DataFrame([input_dict])
    
    #label encoding
    for col in bin_cols:
        if col in df_input.columns:
            df_input[col] = label_encoders[col].transform(df_input[col])


    #one hot encoding
    ohe_encoded = ohe.transform(df_input[mul_cols])
    ohe_df = pd.DataFrame(
        ohe_encoded,
        columns=ohe.get_feature_names_out(mul_cols)
    )

    #drop multi categorical raw columns
    df_input = df_input.drop(columns=mul_cols)

    #merge encoded columns
    df_input = pd.concat([df_input.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)


    df_input =df_input.reindex(columns=features, fill_value=0)

    pred_prob = model.predict_proba(df_input)[:,1]
    pred_class = (pred_prob>=threshold).astype(int)

    predict_label = "Churn" if pred_class==1 else "Not Churn"

    # st.write("Input DataFrame for prediction:")
    # st.dataframe(df_input)
    # st.write("Predicted Probability:", pred_prob)

    st.balloons()
    st.subheader("Prediction Result")
    st.write(f"Predicted Customer Status: **{predict_label}**")
    st.write(f"Churn Probability: {pred_prob[0]*100:.2f}%")