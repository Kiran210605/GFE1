import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the saved models (Ensure they are in the same directory as this script)
xgboost_model = joblib.load('xgboost_model.pkl')  # XGBoost model
linear_model = joblib.load('linear_model.pkl')  # Linear Regression model

# Set Streamlit page configuration
st.set_page_config(
    page_title="📦 Inventory Demand Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and Description
st.title("📦 Inventory Demand Prediction App")
st.markdown("""
This app predicts inventory demand for **Tenderstem**, **Babycorn**, and **Finebeans** using:
- **XGBoost**
- **Linear Regression**

You can provide historical data and select a future date to get demand predictions.
""")

# Sidebar for Input
st.sidebar.header("🛠️ User Input")

# Date Input
selected_date = st.sidebar.date_input(
    "Select the Date for Prediction",
    min_value=datetime.today(),
    max_value=datetime.today() + timedelta(days=90),  # Limit to 3 months ahead
)

# Product Dropdown
selected_product = st.sidebar.selectbox(
    "Select Product",
    ["Tenderstem", "babycorn", "finebeans"],
)

# Lag and Rolling Features Input
st.sidebar.markdown("### Historical Data")
lag_1 = st.sidebar.number_input(f"Enter Lag-1 Demand for {selected_product}", value=300, step=10)
lag_7 = st.sidebar.number_input(f"Enter Lag-7 Demand for {selected_product}", value=280, step=10)
rolling_mean_7 = st.sidebar.number_input(f"Enter Rolling Mean for Last 7 Days for {selected_product}", value=290, step=10)

# Convert inputs into a DataFrame for predictions
input_features = pd.DataFrame(
    {
        "lag_1": [lag_1],
        "lag_7": [lag_7],
        "rolling_mean_7": [rolling_mean_7],
    }
)

# Button to Trigger Prediction
if st.sidebar.button("🔮 Predict"):
    st.header(f"📈 Predictions for {selected_product} on {selected_date.strftime('%A, %d %B %Y')}")

    # XGBoost Prediction
    xgb_forecast = xgboost_model.predict(input_features)
    st.markdown(f"**XGBoost Forecast:** {xgb_forecast[0]:.2f} units")

    # Linear Regression Prediction
    linear_forecast = linear_model.predict(input_features)
    st.markdown(f"**Linear Regression Forecast:** {linear_forecast[0]:.2f} units")

    # Average Prediction
    final_forecast = (xgb_forecast[0] + linear_forecast[0]) / 2
    st.subheader(f"📊 Final Predicted Demand: {final_forecast:.2f} units")

# Visualization Section
st.markdown("---")
st.header("📊 Historical Data and Trends")
st.markdown("Visualize past demand trends to better understand seasonality and patterns.")

# Simulate some historical data for plotting
dates = pd.date_range(start='2025-01-01', periods=90, freq='D')
historical_demand = np.random.randint(150, 500, size=len(dates))

historical_data = pd.DataFrame({
    "Date": dates,
    "Demand": historical_demand
})
historical_data.set_index("Date", inplace=True)

# Line Chart for Historical Data
st.line_chart(historical_data["Demand"])

# Footer
st.markdown("---")
st.markdown("© 2025 Demand Prediction App. Powered by Streamlit.")
