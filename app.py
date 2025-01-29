import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Load the trained model
try:
    best_model = joblib.load("best_model.pkl")
    st.write("Model Loaded Successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Simulate past data for feature generation
def generate_past_data():
    dates = pd.date_range(start="2025-01-01", periods=12 * 7, freq="D")  # 12 weeks of data (84 days)
    tenderstem_demand = np.random.randint(150, 500, size=len(dates))
    babycorn_demand = np.random.randint(100, 450, size=len(dates))
    finebeans_demand = np.random.randint(200, 500, size=len(dates))

    data = pd.DataFrame({
        "Date": dates,
        "Tenderstem": tenderstem_demand,
        "babycorn": babycorn_demand,
        "finebeans": finebeans_demand
    })

    data.set_index("Date", inplace=True)
    return data

# Generate past data
data = generate_past_data()

# Function to prepare features for prediction
def prepare_features(data, product):
    data["lag_1"] = data[product].shift(1)
    data["lag_7"] = data[product].shift(7)
    data["rolling_mean_7"] = data[product].rolling(window=7).mean()

    data.dropna(inplace=True)

    # Ensure features are in the correct 2D shape
    last_data = data.iloc[-1][["lag_1", "lag_7", "rolling_mean_7"]].values.reshape(1, -1)
    return last_data

# Function to predict orders for a given date
def predict_order(date):
    forecasted_order = {}
    for product in ["Tenderstem", "babycorn", "finebeans"]:
        features = prepare_features(data, product)
        try:
            forecasted_order[product] = best_model.predict(features)[0]
        except Exception as e:
            st.error(f"Error predicting {product}: {e}")
    return forecasted_order

# Streamlit UI
st.title("Inventory Forecasting App 📈")
st.write("Enter a date to predict the order quantities for **Tenderstem, Babycorn, and Finebeans**.")

# Date input from user
input_date = st.date_input("Select a date:", min_value=datetime(2025, 1, 1))

# Predict button
if st.button("Predict Order"):
    try:
        forecasted_order = predict_order(input_date)
        st.subheader(f"Predicted Order for {input_date.strftime('%A, %d %B %Y')}:")
        for product, forecast in forecasted_order.items():
            st.write(f"**{product}:** {forecast:.2f} units")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.write("This app uses **Linear Regression** as the best model for forecasting.")
