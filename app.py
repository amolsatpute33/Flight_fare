import streamlit as st
import numpy as np
import os
import gdown
import pickle

# ✅ Google Drive file ID
file_id = "1ocTS-tA_tRFvbWxX_iNHnFxGbsNcXXe5"

# ✅ Direct download URL
url = f"https://drive.google.com/uc?id={file_id}"

# ✅ Download model only once
if not os.path.exists("model.pkl"):
    gdown.download(url, "model.pkl", quiet=False)

# ✅ Load model
model, le = pickle.load(open("model.pkl", "rb"))

st.title("✈️ Flight Fare Prediction App")

# Inputs
airline = st.selectbox("Airline", ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST"])
flight = st.number_input("Flight Code", value=1000)
source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
departure_time = st.selectbox("Departure Time", ["Morning", "Afternoon", "Evening", "Night", "Early_Morning"])
stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
arrival_time = st.selectbox("Arrival Time", ["Morning", "Afternoon", "Evening", "Night", "Early_Morning"])
destination_city = st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
class_type = st.selectbox("Class", ["Economy", "Business"])
duration = st.number_input("Duration", value=2.5)
days_left = st.slider("Days Left", 1, 50, 5)

# Prepare input
input_data = np.array([[airline, flight, source_city, departure_time,
                        stops, arrival_time, destination_city,
                        class_type, duration, days_left]])

# Encode categorical
for i in range(input_data.shape[1]):
    try:
        input_data[:, i] = le.transform(input_data[:, i])
    except:
        pass

input_data = input_data.astype(float)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {int(prediction[0])}")