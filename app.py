import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("xgboost_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("💳 UPI Fraud Detection System")
st.write("Enter transaction details to predict fraud")

# User Inputs
transaction_type = st.selectbox("Transaction Type", label_encoders["transaction type"].classes_)
merchant_category = st.selectbox("Merchant Category", label_encoders["merchant_category"].classes_)
transaction_status = st.selectbox("Transaction Status", label_encoders["transaction_status"].classes_)
sender_age_group = st.selectbox("Sender Age Group", label_encoders["sender_age_group"].classes_)
receiver_age_group = st.selectbox("Receiver Age Group", label_encoders["receiver_age_group"].classes_)
sender_state = st.selectbox("Sender State", label_encoders["sender_state"].classes_)
sender_bank = st.selectbox("Sender Bank", label_encoders["sender_bank"].classes_)
receiver_bank = st.selectbox("Receiver Bank", label_encoders["receiver_bank"].classes_)
device_type = st.selectbox("Device Type", label_encoders["device_type"].classes_)
network_type = st.selectbox("Network Type", label_encoders["network_type"].classes_)

amount = st.number_input("Amount (INR)", min_value=1.0)

hour = st.slider("Hour of Transaction", 0, 23)
month = st.slider("Month", 1, 12)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6)

hour_of_day = hour
is_weekend = 1 if day_of_week >= 5 else 0

# Predict Button
if st.button("Predict Fraud"):

    input_dict = {
        "transaction type": transaction_type,
        "merchant_category": merchant_category,
        "amount (INR)": amount,
        "transaction_status": transaction_status,
        "sender_age_group": sender_age_group,
        "receiver_age_group": receiver_age_group,
        "sender_state": sender_state,
        "sender_bank": sender_bank,
        "receiver_bank": receiver_bank,
        "device_type": device_type,
        "network_type": network_type,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour": hour,
        "month": month
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical values
    for column in label_encoders:
        input_df[column] = label_encoders[column].transform(input_df[column])

    # Ensure correct feature order
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")