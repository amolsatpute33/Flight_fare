# ✈️ Flight_fare Prediction using Machine Learning

<p align="center">
  <a href="https://flightfare-gzkdety6mykcjzsrmf4jwj.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-blue?style=for-the-badge&logo=streamlit" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/ML-Random%20Forest-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square"/>
</p>

---

## 🌐 Live Demo

👉 **Try it here:**
🔗 https://flightfare-gzkdety6mykcjzsrmf4jwj.streamlit.app/

---

## 🎯 Project Overview

> A smart Machine Learning system to **predict flight ticket prices** based on travel details.

Flight_fare uses a **Random Forest Regressor** trained on historical airline data to estimate ticket prices with high accuracy.

✨ Designed for:

* Travelers 🧳
* Data Science learners 📊
* ML portfolio projects 🚀

---

## 💡 Problem Statement

Flight prices change dynamically due to multiple factors like:

* Airline ✈️
* Route 📍
* Duration ⏳
* Stops 🛑

👉 This project helps:

* Predict ticket prices
* Optimize booking decisions
* Analyze airline pricing patterns

---

## 📂 Dataset

📁 **File:** `Data_Train.xlsx`

### 🔑 Features:

| Feature         | Description       |
| --------------- | ----------------- |
| Airline         | Flight company    |
| Source          | Departure city    |
| Destination     | Arrival city      |
| Route           | Travel path       |
| Dep_Time        | Departure time    |
| Arrival_Time    | Arrival time      |
| Duration        | Total travel time |
| Total_Stops     | Number of stops   |
| Additional_Info | Extra details     |
| Price           | Target variable   |

---

## ⚙️ Feature Engineering

✔ Extracted:

* Journey_day, Journey_month
* Dep_hour, Dep_min
* Arrival_hour, Arrival_min

✔ Transformed:

* Duration → minutes
* Stops → numerical values

✔ Cleaned:

* Removed Route, Additional_Info

✔ Encoded:

* One-Hot Encoding for categorical data

---

## 🤖 Model Training

🔍 **Algorithm:** RandomForestRegressor

### ⚡ Parameters:

* n_estimators = 100
* max_depth = 10
* random_state = 42

---

## 📊 Model Performance

🎯 **R² Score:** `0.87`

✅ Explains **87% variance** in flight prices
✅ Strong predictive capability

---

## 💾 Project Files

| File        | Description       |
| ----------- | ----------------- |
| model.pkl   | Trained ML model  |
| columns.pkl | Feature structure |
| app.py      | Streamlit app     |

---

## 🖥️ Application UI

💡 Features of the web app:

* Clean dashboard UI
* User-friendly input system
* Real-time prediction
* Instant price output 💰

---

## 🚀 Run Locally

### 1️⃣ Clone Repo

```bash
git clone https://github.com/amolsatpute33/Flight_fare.git
cd Flight_fare
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Run App

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

🚀 Planned upgrades:

* XGBoost model (higher accuracy)
* Hyperparameter tuning
* Real-time API integration
* Advanced UI dashboard
* Price trend visualization 📊

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

---

## 👨‍💻 Author

**Amol Satpute**

---

## 🌟 Show Your Support

If you like this project:

⭐ Star the repository
🔁 Share with others
💬 Give feedback

---

<p align="center">
  🚀 Built with passion for Machine Learning
</p>
