# ✈️ Flight Delay Prediction System

## 🚀 Live Demo

👉 [Open the Flight Delay Prediction App](https://flightdelayprediction-bmywou7rttg5rh35zbfzxh.streamlit.app/)
A machine learning-based web application that predicts whether a flight is likely to be delayed based on flight-related information.

The application uses a trained **Random Forest Classification model** along with a Spark ML feature preprocessing pipeline. Users can enter flight details through an interactive Streamlit interface and receive a prediction along with the estimated probability of delay.

## 📌 Project Overview

Flight delays can be caused by several factors such as airline, departure airport, destination, departure time, distance, and scheduled flight duration.

This project uses machine learning to analyze these flight characteristics and predict whether a flight is likely to experience a delay.

The trained model was developed using **Apache Spark MLlib** and deployed as an interactive web application using **Streamlit**.

## 🎯 Objectives

- Predict whether a flight will be delayed or not.
- Apply machine learning to flight-related data.
- Use Spark ML preprocessing for categorical and numerical features.
- Use a Random Forest classifier for prediction.
- Provide an easy-to-use web interface.
- Display the predicted delay probability.

## 🧠 Machine Learning Approach

The application uses the following workflow:

1. Flight data is collected.
2. Categorical features are converted into numerical representations.
3. Categorical variables are one-hot encoded.
4. Numerical and encoded features are combined using a VectorAssembler.
5. A Random Forest classification model is used for prediction.
6. The trained preprocessing pipeline and model are saved.
7. Streamlit loads the saved models and provides predictions through a web interface.

## 🔧 Input Features

The application accepts the following flight information:

- **Airline**
- **Origin Airport**
- **Destination Airport**
- **Month**
- **Day of Week**
- **Departure Hour**
- **Distance**
- **Scheduled Flight Duration**

## 🌳 Machine Learning Model

The project uses a:

**Random Forest Classifier**

Model configuration:

- Number of trees: **100**
- Number of classes: **2**
- Feature vector size: **713**

The two prediction classes are:

- `0` → Not Delayed
- `1` → Delayed

## 🖥️ Technologies Used

- Python
- Apache Spark
- PySpark
- Spark MLlib
- Random Forest
- Streamlit
- Pandas
- NumPy
- GitHub

## 📂 Project Structure

```text
Flight_Delay_Prediction/
│
├── README.md
├── app.py
├── requirements.txt
├── packages.txt
│
└── models/
    ├── feature_pipeline_model/
    │   ├── metadata/
    │   └── stages/
    │
    └── final_random_forest_model/
        └── ...
