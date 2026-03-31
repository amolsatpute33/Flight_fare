# ✈️ Flight Fare Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)  
[![Flask](https://img.shields.io/badge/Flask-2.3.2-orange?logo=flask)](https://flask.palletsprojects.com/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **machine learning-based web application** that predicts flight prices based on parameters such as airline, duration, stops, and class. Helps users make **data-driven decisions** when booking flights.  

---

## 📋 Table of Contents

- [Overview](#-overview)  
- [Features](#-features)  
- [Tech Stack](#-tech-stack)  
- [Dataset](#-dataset)  
- [Installation](#-installation)  
- [Project Structure](#-project-structure)  
- [Model Performance](#-model-performance)  
- [Usage](#-usage)  
- [API Endpoints](#-api-endpoints)  
- [Future Enhancements](#-future-enhancements)  
- [Authors](#-authors)  
- [Acknowledgments](#-acknowledgments)  

---

## 🎯 Overview

The **Flight Fare Prediction System** predicts ticket prices using **machine learning models** trained on historical flight data.  

**Key Components:**  
- Data preprocessing and cleaning  
- Feature engineering  
- Model training and evaluation  
- Flask web app for real-time prediction  

---

## ✨ Features

- **Real-time Price Prediction** – Instant fare estimates  
- **Multiple Airlines Support** – Air India, AirAsia, GO_FIRST, Indigo, SpiceJet, Vistara  
- **User-friendly Web Interface** – Simple input form  
- **Advanced Feature Engineering** – Normalization, encoding, outlier treatment  
- **RESTful API** – CORS-enabled for integration  

---

## 🛠️ Tech Stack

**Backend:** Python, Flask, Flask-CORS  
**Machine Learning:** scikit-learn, XGBoost, pandas, numpy, scipy  
**Visualization:** matplotlib, seaborn, sweetviz, autoviz  

---

## 📊 Dataset

- **Size:** 300,153 records  
- **Features:** 12 columns – airline, flight code, cities, departure/arrival times, stops, class, duration, days left, price  
- **Price Range:** ₹1,105 – ₹123,071  
- **Average Price:** ₹20,890  

---

## 🚀 Installation

### Prerequisites

- Python 3.7+  
- pip  

### Steps to Set Up

```bash
# Clone repository
git clone https://github.com/yourusername/flight-fare-prediction.git
cd flight-fare-prediction

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

1.Prepare Dataset: Place airlines_flights_data.csv in the project root.
2.Train Model (if flight_fare.pkl is missing):

python train_model.py

📁 Project Structure

flight-fare-prediction/
│
├── app.py                # scikit learn
├── train_model.py        # Model training script
├── flight_fare.pkl       # Trained model
├── scaler.pkl            # Scaler object
├── airlines_flights_data.csv
├── requirements.txt
└── templates/
    └── home.html         # Input form template

| Model             | Training R² | Testing R² | RMSE |
| ----------------- | ----------- | ---------- | ---- |
| Linear Regression | 0.92        | 0.92       | 0.28 |
| XGBoost Regressor | 0.94        | 0.94       | 0.24 |
| Random Forest     | 0.98        | 0.98       | 0.14 |

Selected Model: Random Forest Regressor (98% accuracy)
Top Features: Airline, Flight code, Stops, Class, Duration

Web Interface
1.Open home page
2.Enter flight details: Airline, Stops, Class, Duration
3.Click Predict to get fare estimate

Sample Input

{
  "airline": "Indigo",
  "stops": "one",
  "class": "Economy",
  "duration": 2.5
}


🔌 API Endpoints
GET / – Returns home page
POST /predict – Returns predicted fare

Request Body

{
  "airline": "string",
  "flight": "string",
  "stops": "string",
  "class": "string",
  "duration": "float"
}

Response

Your Flight price is Rs. XX,XXX

🔮 Future Enhancements
Add more airlines and international flights
Include seasonal pricing trends
Price trend visualization
Flight recommendation system
Mobile app development
Real-time integration with airline APIs
Price alerts and notifications
Historical price analysis dashboard

👥 Authors
Amol Satpute

🙏 Acknowledgments
Dataset: Kaggle – Airlines Flights Data
Inspired by flight booking platforms


