import streamlit as st
import joblib
import gdown
import os
import pandas as pd

st.title("✈️ Flight Fare Prediction")

MODEL_ID = "1BH0C5HxnixA4Bbt5BXmuKSLgkNiin64B"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "model.pkl"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Downloaded!")

# Safe load
@st.cache_data
def load_model():
    data = joblib.load(MODEL_PATH)
    # Check type
    if isinstance(data, dict):
        return data["model"], data["columns"]
    else:
        # If directly pipeline
        return data, None

model, model_columns = load_model()
st.success("Model Loaded ✅")

# Example input for testing
input_data = pd.DataFrame([{
    "airline": "SpiceJet",
    "flight": "SG-8709",
    "source_city": "Delhi",
    "departure_time": "Evening",
    "stops": "zero",
    "arrival_time": "Night",
    "destination_city": "Mumbai",
    "class": "Economy",
    "duration": 2.17,
    "days_left": 1
}])

# Ensure all columns exist
if model_columns is not None:
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_columns]

# Prediction
try:
    pred = model.predict(input_data)[0]
    st.write(f"Predicted Flight Price: ₹ {round(pred, 2)}")
except Exception as e:
    st.error(f"Prediction Error: {e}")
