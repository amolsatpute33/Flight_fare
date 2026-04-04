import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gdown
import pickle
import os

st.set_page_config(page_title="Flight Fare Prediction", layout="centered")
st.title("✈️ Flight Fare Prediction System")

# ----------------------
# 1️⃣ Google Drive file IDs
MODEL_ID = "1hbQpTK4wsh9geeRPJMxpEFlisnYLqwVv"
SCALER_ID = "1vWyKzJXs-7iJzwn_pzFRySqC4isjr6vQ"

# 2️⃣ Download model if not exists
if not os.path.exists("flight_fare_model.pkl"):
    st.info("Downloading ML model...")
    model_url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(model_url, "flight_fare_model.pkl", quiet=False)

# 3️⃣ Download scaler if not exists
if not os.path.exists("scaler.pkl"):
    st.info("Downloading scaler...")
    scaler_url = f"https://drive.google.com/uc?id={SCALER_ID}"
    gdown.download(scaler_url, "scaler.pkl", quiet=False)

# 4️⃣ Load model and scaler
with open("flight_fare_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 5️⃣ Load dataset for dropdowns
data = pd.read_csv("flight_data.csv")

# 6️⃣ Dropdown lists
airline_list = data['airline'].unique()
source_list = data['source_city'].unique()
dest_list = data['destination_city'].unique()
stops_list = data['stops'].unique()
class_list = data['class'].unique()

# 7️⃣ User inputs
airline = st.selectbox("Airline", airline_list)
source_city = st.selectbox("Source City", source_list)
destination_city = st.selectbox("Destination City", dest_list)
stops = st.selectbox("Stops", stops_list)
class_type = st.selectbox("Class", class_list)
duration = st.number_input("Duration (hours)", min_value=0.0, step=0.1)
days_left = st.number_input("Days Left for Flight", min_value=0, step=1)

# 8️⃣ Predict button
if st.button("Predict Fare"):
    # Encode categorical variables
    le_airline = LabelEncoder().fit(data['airline'])
    airline_encoded = le_airline.transform([airline])[0]

    le_source = LabelEncoder().fit(data['source_city'])
    source_encoded = le_source.transform([source_city])[0]

    le_dest = LabelEncoder().fit(data['destination_city'])
    dest_encoded = le_dest.transform([destination_city])[0]

    le_stops = LabelEncoder().fit(data['stops'])
    stops_encoded = le_stops.transform([stops])[0]

    le_class = LabelEncoder().fit(data['class'])
    class_encoded = le_class.transform([class_type])[0]

    # Prepare input
    duration_mins = duration * 60
    input_data = np.array([[airline_encoded, source_encoded, dest_encoded,
                            stops_encoded, class_encoded, duration_mins, days_left]])
    
    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Estimated Flight Fare: ₹{round(prediction, 2)}")

    
