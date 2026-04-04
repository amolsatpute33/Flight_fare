# app.py
import streamlit as st
import pickle
import numpy as np
import requests
import os

st.set_page_config(page_title="Flight Fare Prediction", layout="centered")
st.title("✈️ Flight Fare Prediction System")

# --- Direct download URLs ---
SCALER_URL = "https://drive.google.com/uc?export=download&id=1XIhTSh2Lg06DqGEoXCJZAIldrCs8FoNZ"
COLUMN_URL = "https://drive.google.com/uc?export=download&id=1n5ZcUskBSywo6Lic9s5hQsyZ9NWot_-r"
MODEL_URL = "https://drive.google.com/uc?export=download&id=18mPXAbmTsLtHiTicU5zqVRnlNuh4JZtq"

# --- Local file paths ---
SCALER_PATH = "scaler.pkl"
COLUMN_PATH = "column.pkl"
MODEL_PATH = "model.pkl"

# --- Download function ---
def download_file(url, path):
    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)

# --- Download all files if missing ---
download_file(SCALER_URL, SCALER_PATH)
download_file(COLUMN_URL, COLUMN_PATH)
download_file(MODEL_URL, MODEL_PATH)

# --- Load model and preprocessing ---
model = pickle.load(open(MODEL_PATH,"rb"))
le_dict = pickle.load(open(COLUMN_PATH,"rb"))
scaler = pickle.load(open(SCALER_PATH,"rb"))

# --- Sidebar inputs ---
st.sidebar.header("Enter Flight Details")
airline = st.sidebar.selectbox("Airline", le_dict['airline'].classes_)
source = st.sidebar.selectbox("Source City", le_dict['source_city'].classes_)
destination = st.sidebar.selectbox("Destination City", le_dict['destination_city'].classes_)
flight_class = st.sidebar.selectbox("Class", le_dict['class'].classes_)
departure_time = st.sidebar.selectbox("Departure Time", le_dict['departure_time'].classes_)
arrival_time = st.sidebar.selectbox("Arrival Time", le_dict['arrival_time'].classes_)
duration = st.sidebar.number_input("Duration (hours)", 0.0, 20.0, 2.0, 0.1)
stops = st.sidebar.number_input("Number of Stops", 0, 3, 0)
days_left = st.sidebar.number_input("Days Left Before Flight", 0, 365, 1)

# --- Prepare input vector ---
input_vector = np.array([
    le_dict['airline'].transform([airline])[0],
    le_dict['source_city'].transform([source])[0],
    le_dict['destination_city'].transform([destination])[0],
    le_dict['class'].transform([flight_class])[0],
    le_dict['departure_time'].transform([departure_time])[0],
    le_dict['arrival_time'].transform([arrival_time])[0],
    duration,
    stops,
    days_left
]).reshape(1, -1)

# Scale numeric columns (duration, stops, days_left)
input_vector[:,6:] = scaler.transform(input_vector[:,6:])

# --- Prediction button ---
if st.button("Predict Fare"):
    price = model.predict(input_vector)[0]
    st.success(f"💰 Estimated Flight Fare: ₹ {round(price,2)}")

    
