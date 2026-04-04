import streamlit as st
import joblib
import gdown
import os
import numpy as np
import pandas as pd

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
# INPUT
# -------------------------
airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet"])
stops = st.selectbox("Stops", ["non-stop", "1 stop", "2 stops"])
flight_class = st.selectbox("Class", ["Economy", "Business"])
duration = st.number_input("Duration", 0.0, 20.0)

# -------------------------
# CREATE DATAFRAME
# -------------------------
if st.button("Predict"):
    try:
        input_dict = {
            "airline": airline,
            "stops": stops,
            "class": flight_class,
            "duration": duration
        }

        df = pd.DataFrame([input_dict])

        # 🔥 IMPORTANT: same preprocessing as training
        df = pd.get_dummies(df)

        # 🔥 MUST MATCH TRAINING COLUMNS
        train_cols = [
            'duration',
            'airline_Air India',
            'airline_IndiGo',
            'airline_SpiceJet',
            'stops_1 stop',
            'stops_2 stops',
            'stops_non-stop',
            'class_Business',
            'class_Economy'
        ]

        df = df.reindex(columns=train_cols, fill_value=0)

        pred = model.predict(df)[0]

        st.success(f"💰 Price: ₹ {round(pred,2)}")

    except Exception as e:
        st.error(f"Error: {e}")
