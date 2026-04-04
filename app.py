# app.py

import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Flight Fare Prediction", layout="centered")
st.title("✈️ Flight Fare Prediction System")

# Load trained model and preprocessing objects
model = pickle.load(open("model.pkl","rb"))
le_dict = pickle.load(open("column.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ---- Sidebar / Inputs ----
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

# ---- Prepare input ----
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

# Scale numeric features (duration, stops, days_left)
input_vector[:,6:] = scaler.transform(input_vector[:,6:])

# ---- Prediction ----
if st.button("Predict Fare"):
    predicted_price = model.predict(input_vector)[0]
    st.success(f"💰 Estimated Flight Fare: ₹ {round(predicted_price, 2)}")
    
