# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta

# # Load the saved models (ensure they are in the same directory as this script)
# holt_winters_model = joblib.load('holt_winters_model.pkl')  # Holt-Winters model
# xgboost_model = joblib.load('xgboost_model.pkl')  # XGBoost model
# linear_model = joblib.load('linear_model.pkl')  # Linear Regression model

# # Set Streamlit page configuration
# st.set_page_config(
#     page_title="Demand Prediction App",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Title and Description
# st.title("📦 Inventory Demand Prediction App")
# st.markdown("""
# This app predicts inventory demand for **Tenderstem**, **Babycorn**, and **Finebeans**. 
# The predictions are based on advanced forecasting models, including:
# - Holt-Winters Exponential Smoothing
# - XGBoost
# - Linear Regression

# You can provide historical data and select a future date to get demand predictions.
# """)
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# import joblib

# # Load Holt-Winters model parameters
# holt_winters_params = joblib.load('holt_winters_model.pkl')

# # Function to reinitialize the Holt-Winters model
# def initialize_holt_winters(data):
#     hw_model = ExponentialSmoothing(
#         data, trend='add', seasonal='add', seasonal_periods=7
#     )
#     hw_model_fit = hw_model.fit(smoothing_level=holt_winters_params['smoothing_level'],
#                                 smoothing_trend=holt_winters_params['smoothing_trend'],
#                                 smoothing_seasonal=holt_winters_params['smoothing_seasonal'],
#                                 initial_level=holt_winters_params['initial_level'],
#                                 initial_trend=holt_winters_params['initial_trend'],
#                                 initial_seasonal=holt_winters_params['initial_seasonal'])
#     return hw_model_fit
# # # Reinitialize Holt-Winters model with the current data
# # hw_model_fit = initialize_holt_winters(data['Tenderstem'])

# # # Forecast using the reinitialized model
# # hw_forecast = hw_model_fit.forecast(steps=1)
# # print(f"Holt-Winters Forecast: {hw_forecast[0]}")

# # Sidebar for Input
# st.sidebar.header("🛠️ User Input")

# # Date Input
# selected_date = st.sidebar.date_input(
#     "Select the Date for Prediction",
#     min_value=datetime.today(),
#     max_value=datetime.today() + timedelta(days=90),  # Limit to 3 months ahead
# )

# # Product Dropdown
# selected_product = st.sidebar.selectbox(
#     "Select Product",
#     ["Tenderstem", "babycorn", "finebeans"],
# )

# # Lag and Rolling Features Input
# st.sidebar.markdown("### Historical Data")
# lag_1 = st.sidebar.number_input(f"Enter Lag-1 Demand for {selected_product}", value=300, step=10)
# lag_7 = st.sidebar.number_input(f"Enter Lag-7 Demand for {selected_product}", value=280, step=10)
# rolling_mean_7 = st.sidebar.number_input(f"Enter Rolling Mean for Last 7 Days for {selected_product}", value=290, step=10)

# # Convert inputs into a DataFrame for predictions
# input_features = pd.DataFrame(
#     {
#         "lag_1": [lag_1],
#         "lag_7": [lag_7],
#         "rolling_mean_7": [rolling_mean_7],
#     }
# )

# # Button to Trigger Prediction
# if st.sidebar.button("🔮 Predict"):
#     # Predictions from each model
#     st.header(f"📈 Predictions for {selected_product} on {selected_date.strftime('%A, %d %B %Y')}")
    
#     # Holt-Winters Prediction
#     # hw_forecast = holt_winters_model.forecast(steps=1)
#     # st.markdown(f"**Holt-Winters Forecast:** {hw_forecast[0]:.2f} units")
#     # Reinitialize Holt-Winters model with the current data
#     hw_model_fit = initialize_holt_winters(data['Tenderstem'])

# # Forecast using the reinitialized model
#     hw_forecast = hw_model_fit.forecast(steps=1)
#     print(f"Holt-Winters Forecast: {hw_forecast[0]}")
    
#     # XGBoost Prediction
#     xgb_forecast = xgboost_model.predict(input_features)
#     st.markdown(f"**XGBoost Forecast:** {xgb_forecast[0]:.2f} units")
    
#     # Linear Regression Prediction
#     linear_forecast = linear_model.predict(input_features)
#     st.markdown(f"**Linear Regression Forecast:** {linear_forecast[0]:.2f} units")
    
#     # Average Prediction
#     final_forecast = (hw_forecast[0] + xgb_forecast[0] + linear_forecast[0]) / 3
#     st.subheader(f"📊 Final Predicted Demand: {final_forecast:.2f} units")

# # Visualization Section
# st.markdown("---")
# st.header("📊 Historical Data and Trends")
# st.markdown("Visualize past demand trends to better understand seasonality and patterns.")

# # Simulate some historical data for plotting
# dates = pd.date_range(start='2025-01-01', periods=90, freq='D')
# historical_demand = np.random.randint(150, 500, size=len(dates))

# historical_data = pd.DataFrame({
#     "Date": dates,
#     "Demand": historical_demand
# })
# historical_data.set_index("Date", inplace=True)

# # Line Chart for Historical Data
# st.line_chart(historical_data["Demand"])

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load dataset (Simulated or from a file)
dates = pd.date_range(start='2025-01-01', periods=84, freq='D')  # 12 weeks of data (84 days)
tenderstem_demand = np.random.randint(150, 500, size=len(dates))  # Random demand for Tenderstem
babycorn_demand = np.random.randint(100, 450, size=len(dates))  # Random demand for babycorn
finebeans_demand = np.random.randint(200, 500, size=len(dates))

# Create a DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Tenderstem': tenderstem_demand,
    'babycorn': babycorn_demand,
    'finebeans': finebeans_demand
})

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Load Holt-Winters model parameters
holt_winters_params = joblib.load('holt_winters_model.pkl')

# Function to reinitialize the Holt-Winters model
def initialize_holt_winters(product_data):
    hw_model = ExponentialSmoothing(
        product_data, trend='add', seasonal='add', seasonal_periods=7
    )
    hw_model_fit = hw_model.fit(
        smoothing_level=holt_winters_params['smoothing_level'],
        smoothing_trend=holt_winters_params['smoothing_trend'],
        smoothing_seasonal=holt_winters_params['smoothing_seasonal'],
        initial_level=holt_winters_params['initial_level'],
        initial_trend=holt_winters_params['initial_trend'],
        initial_seasonal=holt_winters_params['initial_seasonal']
    )
    return hw_model_fit

# Streamlit App
st.title("Inventory Demand Prediction")

# Dropdown for product selection
product = st.selectbox("Select Product", ['Tenderstem', 'babycorn', 'finebeans'])

# Trigger Holt-Winters Forecast
if st.button("Predict using Holt-Winters"):
    if product in data.columns:
        hw_model_fit = initialize_holt_winters(data[product])  # Reinitialize model with data
        hw_forecast = hw_model_fit.forecast(steps=1)  # Forecast next step
        st.write(f"Forecast for {product}: {hw_forecast[0]:.2f} units")
    else:
        st.error("Product data not found.")


# # Footer
# st.markdown("---")
# st.markdown("© 2025 Demand Prediction App. Powered by Streamlit.")
