# Demand Prediction for Micromobility in Spiders Mobility

## 📌 Project Overview
This is a Senior Graduation Project in Data Science developed in cooperation with **Spiders Mobility**. The project addresses a critical business and operational challenge: predicting daily demand for electric scooters to prevent financial losses caused by fleet oversupply or undersupply.

## 👥 Authors & Supervision
* **Team Members:** Ghadeer Abdullah Hamdi, Ruba Ahmed Alghamdi, Sadeel Mirza.
* **Supervised By:** Dr. Safa Habibullah
* **Institution:** University of Jeddah, College of Computer Sciences & Engineering.

## 🛠️ Methodology & Technical Stack
The project leverages time-series analysis and machine learning workflows to build stable forecasting models:
* **Data Preprocessing:** Outlier detection and removal using robust statistical techniques (e.g., IQR methods via `seaborn` and `matplotlib`), and data description alignments.
* **Applied Algorithms:** * Traditional Statistical Models: **ARIMA (1,0,2)**
  * Machine Learning Models: **XGBoost**, **Random Forest**
  * Deep Learning Architecture: **LSTM (Long Short-Term Memory)**

## 📊 Key Findings
* Models were rigorously evaluated using **MAE**, **MSE**, and **RMSE** metrics.
* The **ARIMA (1,0,2)** configuration demonstrated outstanding performance, delivering the lowest forecasting error rates and proving highly effective for resource allocation decisions.

## 🚀 Future Work
* Evaluate model scalability as the dataset size grows over time.
* Enhance forecasting accuracy by integrating external dynamic features such as local weather constraints, traffic congestion patterns, and major regional events (e.g., Jeddah Calendar / Riyadh Season).
