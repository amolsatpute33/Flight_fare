import streamlit as st
import pandas as pd
import joblib
import gdown
import os

st.title("✈️ Flight Fare Prediction System")

# -------------------------------
# Download model from Google Drive
# -------------------------------
MODEL_ID = "1BH0C5HxnixA4Bbt5BXmuKSLgkNiin64B"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading trained model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Downloaded model!")

# -------------------------------
# Load model
# -------------------------------
@st.cache_data
def load_model():
    data = joblib.load(MODEL_PATH)
    if isinstance(data, dict):
        return data["model"], data["columns"]
    return data, None

model, model_columns = load_model()
st.success("Model loaded ✅")

# -------------------------------
# User Input Form
# -------------------------------
st.header("Enter Flight Details")

with st.form("flight_form"):
    airline = st.selectbox("Airline", ["Air India", "IndiGo", "SpiceJet", "Vistara", "AirAsia", "GO_FIRST"])
    flight = st.text_input("Flight Code", "SG-8709")
    source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore"])
    departure_time = st.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night"])
    stops = st.selectbox("Stops", ["zero", "one", "two", "three_or_more"])
    arrival_time = st.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night"])
    destination_city = st.selectbox("Destination City", ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore"])
    travel_class = st.selectbox("Class", ["Economy", "Business"])
    duration = st.number_input("Duration (hours)", min_value=0.5, max_value=20.0, value=2.5, step=0.1)
    days_left = st.number_input("Days left for travel", min_value=0, max_value=365, value=10, step=1)
    
    submitted = st.form_submit_button("Predict Fare")

# -------------------------------
# Make Prediction
# -------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "airline": airline,
        "flight": flight,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": travel_class,
        "duration": duration,
        "days_left": days_left
    }])
    
    # Reorder columns if model saved columns exist
    if model_columns is not None:
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]
    
    try:
        pred_price = model.predict(input_df)[0]
        st.success(f"Predicted Flight Price: ₹ {round(pred_price, 2)}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
    
