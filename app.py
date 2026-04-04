from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
data = joblib.load("flight_fare.pkl")

model = data["model"]
scaler = data["scaler"]
le_airline = data["le_airline"]
le_class = data["le_class"]
le_stops = data["le_stops"]

@app.route('/')
def home():
    return "Flight Fare Prediction API Running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        airline = le_airline.transform([data['airline']])[0]
        stops = le_stops.transform([data['stops']])[0]
        flight_class = le_class.transform([data['class']])[0]
        duration = float(data['duration'])

        input_data = np.array([[airline, stops, flight_class, duration]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "price": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
    