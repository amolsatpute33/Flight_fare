# app.py
import streamlit as st
import pandas as pd
import joblib
import gdown
import os

st.set_page_config(page_title="Flight Fare Prediction", layout="centered")
st.title("✈️ Flight Fare Prediction App")

MODEL_URL = "https://drive.google.com/uc?id=1BH0C5HxnixA4Bbt5BXmuKSLgkNiin64B"
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = joblib.load(MODEL_PATH)  # Load Pipeline directly
    return model

model = load_model()
st.success("✅ Model Loaded Successfully!")

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter Flight Details")
airline = st.selectbox("Airline", ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "IndiGo", "Air India"])
flight = st.text_input("Flight Code (e.g. SG-8709)")
source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
departure_time = st.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
arrival_time = st.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
destination_city = st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
flight_class = st.selectbox("Class", ["Economy", "Business"])
duration = st.number_input("Duration (hours)", 0.0, 20.0)
days_left = st.number_input("Days Left", 1, 50)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Price 💰"):
    try:
        # Build input DataFrame with EXACT column names from training
        input_df = pd.DataFrame([{
            "airline": airline,
            "flight": flight,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": flight_class,
            "duration": duration,
            "days_left": days_left
        }])

        # Use pipeline directly (handles encoding + prediction)
        predicted_price = model.predict(input_df)[0]
        st.success(f"✈️ Estimated Flight Price: ₹ {round(predicted_price, 2)}")

    except Exception as e:
        st.error(f"Error: {e}")
