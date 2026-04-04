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

data = load_model()

model = data["model"]
scaler = data["scaler"]
le_airline = data["le_airline"]
le_class = data["le_class"]
le_stops = data["le_stops"]

st.success("Model Loaded ✅")

airline = st.selectbox("Airline", le_airline.classes_)
stops = st.selectbox("Stops", le_stops.classes_)
flight_class = st.selectbox("Class", le_class.classes_)
duration = st.number_input("Duration", 0.0, 20.0)

if st.button("Predict"):
    a = le_airline.transform([airline])[0]
    s = le_stops.transform([stops])[0]
    c = le_class.transform([flight_class])[0]

    input_data = scaler.transform([[a, s, c, duration]])
    pred = model.predict(input_data)[0]

    st.success(f"💰 Price: ₹ {round(pred,2)}")


        
