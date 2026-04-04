import streamlit as st
import pickle
import pandas as pd
import gdown
import os

# Google Drive file ID
file_id = "1APQFw6h_IiDvLTIe_pK13iBzR6ioU03A"
url = f"https://drive.google.com/uc?id={file_id}"

# Download model if not exists
if not os.path.exists("model.pkl"):
    gdown.download(url, "model.pkl", quiet=False)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# UI
st.title("✈️ Flight Fare Prediction")

st.write("Enter flight details:")

airline = st.number_input("Airline", 0, 10)
flight = st.number_input("Flight Code", 0, 2000)
source_city = st.number_input("Source City", 0, 10)
departure_time = st.number_input("Departure Time", 0, 10)
stops = st.number_input("Stops", 0, 5)
arrival_time = st.number_input("Arrival Time", 0, 10)
destination_city = st.number_input("Destination City", 0, 10)
class_type = st.number_input("Class", 0, 2)
duration = st.number_input("Duration", 0.0, 20.0)
days_left = st.number_input("Days Left", 0, 50)

if st.button("Predict Price 💰"):

    input_data = pd.DataFrame([[airline, flight, source_city,
                                departure_time, stops, arrival_time,
                                destination_city, class_type,
                                duration, days_left]],
                              columns=[
                                  'airline','flight','source_city','departure_time',
                                  'stops','arrival_time','destination_city',
                                  'class','duration','days_left'
                              ])

    prediction = model.predict(input_data)

    st.success(f"Estimated Price: ₹ {int(prediction[0])}")
