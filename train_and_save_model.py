import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("Telco_Cusomer_Churn.csv")

# Drop redundant column
df = df.drop(columns="PhoneService")

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# Encoding
cat_cols = df.select_dtypes(include="object").columns

mul_cat = [
    col for col in cat_cols
    if df[col].nunique() > 2 and col not in ["CustomerID", "TotalCharges"]
]

# One-hot encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[mul_cat])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(mul_cat))

df = df.reset_index(drop=True)
encoded_df = encoded_df.reset_index(drop=True)
df = pd.concat([df.drop(columns=mul_cat), encoded_df], axis=1)

# Binary encoding
bin_cols = [col for col in cat_cols if col in df.columns and df[col].nunique() == 2]

# Multiple binary columns need separate LabelEncoders
label_encoders = {}
for col in bin_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save each column's encoder separately

# Feature and Target
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Final Model
scale = (y == 0).sum() / (y == 1).sum()

final_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    eval_metric="auc",
    random_state=42
)

final_model.fit(X_train, y_train)

# Save Model with joblib
model_artifact = {
    "model": final_model,
    "ohe": ohe,
    "label_encoders": label_encoders,
    "threshold": 0.4,
    "features": X.columns.tolist()
}

joblib.dump(model_artifact, "churn_xgb_model.joblib")
print("Model saved successfully as churn_xgb_model.joblib")
