import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import joblib
#Models evaluation metrics
MODEL_METRICS = {
    "Linear Regression": {
        "MAE": 1.16,
        "MSE": 2.40,
        "RMSE": 1.55,
        "R2": 0.885,
    },
    "Ridge Regression": {
        "MAE": 1.1603613988142674,
        "MSE": 2.39503052219139,
        "RMSE": 1.5475886152952243,
        "R2": 0.885234934477351,
    },
    "Lasso Regression": {
        "MAE": 1.5084303356570734,
        "MSE": 4.181901596762411,
        "RMSE": 2.0449698278366872,
        "R2": 0.7996116515782107,
    },
    "Random Forest Regressor": {
        "MAE": 0.64,
        "MSE": 1.02,
        "RMSE": 1.01,
        "R2": 0.95,
    },
    "Gradient Boosting Regressor": {
        "MAE": 0.61,
        "MSE": 0.93,
        "RMSE": 0.97,
        "R2": 0.96,
    },
    "XGBoost Regressor": {
        "MAE": 0.59,
        "MSE": 0.89,
        "RMSE": 0.95,
        "R2": 0.96,
    },
}

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Predict - Urban Taxi Fare",
    page_icon="🚖",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center; color: #2C3E50;'>🚖 Predict Urban Taxi Fare</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; color: #2C3E50;'>Use the form below to predict taxi fares based on trip details.</h3>",
    unsafe_allow_html=True
)

# ---------------- USER INPUTS ----------------
trip_distance_km = st.slider(
    "Select Trip Distance (KM):", 0.1, 500.0, 5.0, 0.1
)

trip_duration_min = st.slider(
    "Select Trip Duration (Minutes):", 1, 300, 15, 1
)

passenger_count = st.slider(
    "Select Passenger Count:", 1, 6, 1, 1
)

hour_of_day = st.slider(
    "Select Hour of Day:", 0, 23, 12, 1
)

day_of_week = st.slider(
    "Select Day of Week (0=Monday, 6=Sunday):", 0, 6, 2, 1
)

is_weekend = st.checkbox("Is Weekend?", value=False)

ratecode_id = st.slider(
    "Select Ratecode ID:", 1, 6, 1, 1
)

payment_type = st.slider(
    "Select Payment Type (1=Credit Card, 2=Cash, etc.):", 1, 6, 1, 1
)

model_choice = st.selectbox(
    "Select Prediction Model:",
    (
        "Gradient Boosting Regressor",
        "XGBoost Regressor",
        "Random Forest Regressor",
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
    )
)

# ---------------- MODEL SELECTION ----------------
if model_choice == "Gradient Boosting Regressor":
    model_file = "gb_model_urban_taxi_fare.pkl"
elif model_choice == "XGBoost Regressor":
    model_file = "xgb_model_urban_taxi_fare.pkl"
elif model_choice == "Random Forest Regressor":
    model_file = "rf_model_urban_taxi_fare.pkl"
elif model_choice == "Linear Regression":
    model_file = "linear_regression_model.pkl"
elif model_choice == "Ridge Regression":
    model_file = "ridge_regression_model.pkl"
elif model_choice == "Lasso Regression":
    model_file = "lasso_regression_model.pkl"

# ---------------- MANUAL PREPROCESSING ----------------
def preprocess_input(
    trip_distance,
    trip_duration,
    hour_of_day,
    day_of_week,
    is_weekend,
    passenger_count,
    RatecodeID,
    payment_type,
):
    # Log transform
    log_trip_distance = np.log1p(trip_distance)
    log_trip_duration = np.log1p(trip_duration)

    # Ratecode one-hot
    ratecode_features = {
        "RatecodeID_2": 1 if RatecodeID == 2 else 0,
        "RatecodeID_3": 1 if RatecodeID == 3 else 0,
        "RatecodeID_4": 1 if RatecodeID == 4 else 0,
        "RatecodeID_5": 1 if RatecodeID == 5 else 0,
        "RatecodeID_99": 1 if RatecodeID == 99 else 0,
    }

    # Payment type one-hot
    payment_features = {
        "payment_type_2": 1 if payment_type == 2 else 0,
        "payment_type_3": 1 if payment_type == 3 else 0,
        "payment_type_4": 1 if payment_type == 4 else 0,
    }

    features = {
        "log_trip_distance": log_trip_distance,
        "log_trip_duration": log_trip_duration,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": int(is_weekend),
        "passenger_count": passenger_count,
        **ratecode_features,
        **payment_features,
    }

    feature_order = [
        "log_trip_distance",
        "log_trip_duration",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "passenger_count",
        "RatecodeID_2",
        "RatecodeID_3",
        "RatecodeID_4",
        "RatecodeID_5",
        "RatecodeID_99",
        "payment_type_2",
        "payment_type_3",
        "payment_type_4",
    ]

    return pd.DataFrame(
        [[features[col] for col in feature_order]],
        columns=feature_order,
    )

# ---------------- PREDICTION ----------------
if st.button("Predict Fare"):
    try:
        model = joblib.load(model_file)


        input_data = preprocess_input(
            trip_distance_km,
            trip_duration_min,
            hour_of_day,
            day_of_week,
            is_weekend,
            passenger_count,
            ratecode_id,
            payment_type,
        )

        predicted_fare = model.predict(input_data)[0]

        st.success(f"💰 Predicted Taxi Fare: ${predicted_fare:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    st.markdown("---")
st.subheader("📊 Model Evaluation Metrics")

if st.button("Show Models Evaluation Metrix"):
    metrics_df = pd.DataFrame(MODEL_METRICS).T
    metrics_df = metrics_df[["MAE", "MSE", "RMSE", "R2"]].round(3)

    st.dataframe(metrics_df, use_container_width=True)

    best_model = metrics_df["RMSE"].idxmin()
    st.success(f"🏆 Best Model based on RMSE: **{best_model}**")
