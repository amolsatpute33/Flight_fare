# app.py
import streamlit as st
import pandas as pd
import joblib
import gdown
import os

st.title("✈️ Flight Fare Prediction")

# -----------------------------
# 1. Download model from Google Drive
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1BH0C5HxnixA4Bbt5BXmuKSLgkNiin64B"
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# -----------------------------
# 2. Load model
# -----------------------------
@st.cache_data
def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"], data["columns"]

model, model_columns = load_model()

# -----------------------------
# 3. User Input
# -----------------------------
def user_input():
    data = {
        "airline": st.selectbox("Airline", ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "IndiGo", "Air India"]),
        "flight": st.text_input("Flight Code (e.g., SG-8709)"),
        "source_city": st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]),
        "departure_time": st.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]),
        "stops": st.selectbox("Stops", ["zero", "one", "two_or_more"]),
        "arrival_time": st.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]),
        "destination_city": st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]),
        "class": st.selectbox("Class", ["Economy", "Business"]),
        "duration": st.number_input("Duration (hours)", 0.0, 20.0),
        "days_left": st.number_input("Days Left", 1, 50)
    }
    df = pd.DataFrame([data])
    
    # Ensure columns match training
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]
    return df

input_df = user_input()

# -----------------------------
# 4. Prediction
# -----------------------------
if st.button("Predict Price 💰"):
    try:
        price = model.predict(input_df)[0]
        st.success(f"✈️ Estimated Flight Price: ₹ {round(price, 2)}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
      


    
