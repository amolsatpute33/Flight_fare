import streamlit as st
import numpy as np
import joblib
import requests
import os

st.set_page_config(page_title="Flight Fare Prediction", layout="centered")
st.title("✈️ Flight Fare Prediction System")

# ============================
# 🔗 YOUR GOOGLE DRIVE LINKS (FIXED)
# ============================

MODEL_URL = "https://drive.google.com/uc?export=download&id=1cEDCDXAhfdsNktxaG8wonuacICZr0Nu-"
SCALER_URL = "https://drive.google.com/uc?export=download&id=1THFl_dhkT6lVnhwH5OA7vBUziI0aD3Cx"
COLUMN_URL = "https://drive.google.com/uc?export=download&id=1csx35-zVOwVN1ghWLTrb7wS3XxBjtFOo"

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
COLUMN_PATH = "column.pkl"

# ============================
# 📥 DOWNLOAD FUNCTION
# ============================

def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            r = requests.get(url)
            with open(filename, "wb") as f:
                f.write(r.content)
        except:
            st.error(f"❌ Error downloading {filename}")
            st.stop()

download_file(MODEL_URL, MODEL_PATH)
download_file(SCALER_URL, SCALER_PATH)
download_file(COLUMN_URL, COLUMN_PATH)

# ============================
# 📦 LOAD FILES
# ============================

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le_dict = joblib.load(COLUMN_PATH)
except Exception as e:
    st.error("❌ Model Load Error")
    st.write("👉 Your model is incompatible or missing libraries")
    st.code(str(e))
    st.stop()

# ============================
# 🎛️ USER INPUT
# ============================

st.sidebar.header("Enter Flight Details")

airline = st.sidebar.selectbox("Airline", le_dict['airline'].classes_)
source = st.sidebar.selectbox("Source City", le_dict['source_city'].classes_)
destination = st.sidebar.selectbox("Destination City", le_dict['destination_city'].classes_)
flight_class = st.sidebar.selectbox("Class", le_dict['class'].classes_)
departure_time = st.sidebar.selectbox("Departure Time", le_dict['departure_time'].classes_)
arrival_time = st.sidebar.selectbox("Arrival Time", le_dict['arrival_time'].classes_)

duration = st.sidebar.number_input("Duration (hours)", 0.0, 20.0, 2.0)
stops = st.sidebar.number_input("Stops", 0, 3, 0)
days_left = st.sidebar.number_input("Days Left", 0, 365, 1)

# ============================
# 🔄 PREPROCESS INPUT
# ============================

try:
    input_data = np.array([
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

    input_data[:, 6:] = scaler.transform(input_data[:, 6:])

except Exception as e:
    st.error("❌ Input Processing Error")
    st.code(str(e))
    st.stop()

# ============================
# 🔮 PREDICT
# ============================

if st.button("Predict Fare 💰"):
    try:
        result = model.predict(input_data)[0]
        st.success(f"✈️ Estimated Fare: ₹ {round(result, 2)}")
    except Exception as e:
        st.error("❌ Prediction Error")
        st.code(str(e))



