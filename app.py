import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Flight Fare Predictor")

st.title("✈️ Flight Fare Prediction")

# Inputs
date = st.date_input("Journey Date")
dep_time = st.time_input("Departure Time")
arrival_time = st.time_input("Arrival Time")

airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet", "Vistara"])
source = st.selectbox("Source", ["Delhi", "Mumbai", "Kolkata", "Chennai"])
destination = st.selectbox("Destination", ["Cochin", "Hyderabad", "Banglore", "Delhi"])

stops = st.selectbox("Stops", [0,1,2,3])
duration = st.number_input("Duration (minutes)", 30, 2000)

# Prediction
if st.button("Predict Fare"):

    data = np.zeros(len(columns))
    df_input = pd.DataFrame([data], columns=columns)

    df_input["Journey_day"] = date.day
    df_input["Journey_month"] = date.month
    df_input["Dep_hour"] = dep_time.hour
    df_input["Dep_min"] = dep_time.minute
    df_input["Arrival_hour"] = arrival_time.hour
    df_input["Arrival_min"] = arrival_time.minute
    df_input["Duration"] = duration
    df_input["Total_Stops"] = stops

    # One-hot encoding mapping
    if f"Airline_{airline}" in df_input.columns:
        df_input[f"Airline_{airline}"] = 1

    if f"Source_{source}" in df_input.columns:
        df_input[f"Source_{source}"] = 1

    if f"Destination_{destination}" in df_input.columns:
        df_input[f"Destination_{destination}"] = 1

    prediction = model.predict(df_input)[0]

    st.success(f"💰 Estimated Price: ₹{int(prediction)}")
