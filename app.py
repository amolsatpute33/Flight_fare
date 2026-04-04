import streamlit as st
import joblib
import gdown
import os
import numpy as np

st.title("✈️ Flight Fare Prediction")

MODEL_URL = "https://drive.google.com/uc?id=1BH0C5HxnixA4Bbt5BXmuKSLgkNiin64B"
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return joblib.load(MODEL_PATH)

model = load_model()

st.success("Model Loaded ✅")

# -------------------------
# INPUTS (match training)
# -------------------------
airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet"])
stops = st.selectbox("Stops", ["zero", "one", "two"])
flight_class = st.selectbox("Class", ["Economy", "Business"])
duration = st.number_input("Duration", 0.0, 20.0)

# -------------------------
# MANUAL LABEL ENCODING
# -------------------------
def encode():
    airline_map = {"IndiGo": 0, "Air India": 1, "SpiceJet": 2}
    stops_map = {"zero": 0, "one": 1, "two": 2}
    class_map = {"Economy": 0, "Business": 1}

    return [
        airline_map.get(airline, 0),
        stops_map.get(stops, 0),
        class_map.get(flight_class, 0),
        duration
    ]

# -------------------------
# PREDICT
# -------------------------
if st.button("Predict"):
    try:
        input_data = np.array([encode()])

        prediction = model.predict(input_data)[0]

        st.success(f"💰 Price: ₹ {round(prediction, 2)}")

    except Exception as e:
        st.error(f"Error: {e}")
