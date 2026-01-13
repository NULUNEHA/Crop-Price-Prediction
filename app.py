import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load model, encoder, scaler
with open("dtr.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as le_file:
    le = pickle.load(le_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load data for dropdown options
data = pd.read_csv("new_generated_food_commodities_data.csv")
data['Date'] = pd.to_datetime(data['Date'])

# App UI
st.title("🌾 Commodity Price Prediction")

with st.form("prediction_form"):
    # Inputs
    date_input = st.date_input("Select Date", value=datetime.today())
    commodity = st.selectbox("Select Commodity", sorted(data['Commodity'].unique()))
    centre = st.selectbox("Select Price Reporting Centre", sorted(data['Price Reporting Centre'].unique()))
    seasonality = st.selectbox("Seasonality Factor", sorted(data['Seasonality Factor'].unique()))
    intervention = st.selectbox("Market Intervention Decision", sorted(data['Market Intervention Decision'].unique()))
    trend = st.selectbox("Historical Trend", sorted(data['Historical Trend'].unique()))
    intelligence = st.selectbox("Market Intelligence Input", sorted(data['Market Intelligence Input'].unique()))
    buffer_stock = st.number_input("Buffer Stock (Metric Tons)", value=5000)
    crop_sowing = st.number_input("Crop Sowing Estimate (Hectares)", value=100000)
    production = st.number_input("Production Estimate (Metric Tons)", value=800000)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input data
    input_dict = {
        'Commodity': [commodity],
        'Price Reporting Centre': [centre],
        'Market Intervention Decision': [intervention],
        'Seasonality Factor': [seasonality],
        'Historical Trend': [trend],
        'Market Intelligence Input': [intelligence],
        'Buffer Stock (Metric Tons)': [buffer_stock],
        'Crop Sowing Estimate (Hectares)': [crop_sowing],
        'Production Estimate (Metric Tons)': [production]
    }

    input_df = pd.DataFrame(input_dict)

    # Encode categorical columns
    categorical_columns = [
        'Commodity', 'Price Reporting Centre', 'Market Intervention Decision',
        'Seasonality Factor', 'Historical Trend', 'Market Intelligence Input'
    ]

    for col in categorical_columns:
        if input_df[col][0] not in le.classes_:
            le.classes_ = np.append(le.classes_, input_df[col][0])
        input_df[col] = le.transform(input_df[col])

    # Ensure correct column order (from model training phase)
    correct_order = [
        'Commodity', 'Price Reporting Centre', 'Market Intervention Decision',
        'Seasonality Factor', 'Historical Trend', 'Market Intelligence Input',
        'Buffer Stock (Metric Tons)', 'Crop Sowing Estimate (Hectares)',
        'Production Estimate (Metric Tons)'
    ]

    input_df = input_df[correct_order]

    # Strip feature names (to match scaler's training input)
    input_array = input_df.values  # removes feature names

    # Scale and predict
    input_scaled = scaler.transform(input_array)
    predicted_price = model.predict(input_scaled)[0]

    # Output
    st.subheader("📈 Predicted Price:")
    st.success(f"₹ {predicted_price:.2f} per kg")