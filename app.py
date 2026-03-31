import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open("model.pkl", "rb"))

st.title("✈️ Flight Price Prediction")

# Inputs
airline = st.number_input("Airline (encoded value)")
flight = st.number_input("Flight (encoded value)")
source_city = st.number_input("Source City")
departure_time = st.number_input("Departure Time")
stops = st.number_input("Stops")
arrival_time = st.number_input("Arrival Time")
destination_city = st.number_input("Destination City")
class_type = st.number_input("Class")
duration = st.number_input("Duration")
days_left = st.number_input("Days Left")

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[airline, flight, source_city, departure_time,
                            stops, arrival_time, destination_city,
                            class_type, duration, days_left]])

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Price: ₹ {prediction[0]:,.2f}")