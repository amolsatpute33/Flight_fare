import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("✈️ Flight Fare Prediction App")

st.write("Enter Flight Details:")

# User Inputs
airline = st.number_input("Airline", 0, 10)
flight = st.number_input("Flight Code", 0, 2000)
source_city = st.number_input("Source City", 0, 10)
departure_time = st.number_input("Departure Time", 0, 10)
stops = st.number_input("Stops", 0, 5)
arrival_time = st.number_input("Arrival Time", 0, 10)
destination_city = st.number_input("Destination City", 0, 10)
class_type = st.number_input("Class", 0, 2)
duration = st.number_input("Duration (hrs)", 0.0, 20.0)
days_left = st.number_input("Days Left", 0, 50)

# Prediction button
if st.button("Predict Price 💰"):

    input_data = pd.DataFrame([[airline, flight, source_city,
                                departure_time, stops, arrival_time,
                                destination_city, class_type,
                                duration, days_left]],
                              columns=columns)

    prediction = model.predict(input_data)

    st.success(f"Estimated Flight Price: ₹ {int(prediction[0])}")
