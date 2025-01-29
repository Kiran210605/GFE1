import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Advanced Sales Forecasting", layout="wide")

# Generate synthetic dataset
dates = pd.date_range(start='2025-01-01', periods=12*7, freq='D')
tenderstem_demand = np.random.randint(150, 500, size=len(dates))
babycorn_demand = np.random.randint(100, 450, size=len(dates))
finebeans_demand = np.random.randint(200, 500, size=len(dates))

data = pd.DataFrame({
    'Date': dates,
    'Tenderstem': tenderstem_demand,
    'Babycorn': babycorn_demand,
    'Finebeans': finebeans_demand
})

data.set_index('Date', inplace=True)

# Public holidays
public_holidays = pd.to_datetime([
    '2025-01-01', '2025-02-03', '2025-03-17', '2025-04-21',
    '2025-05-05', '2025-06-02', '2025-08-04', '2025-10-27',
    '2025-12-25', '2025-12-26'
])

data['Day_of_Week'] = data.index.dayofweek
data['Weekend'] = data['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)

for col in ['Tenderstem', 'Babycorn', 'Finebeans']:
    data[col] *= np.where(data['Weekend'] == 1, 1.2, 1.0)
    data.loc[data.index.isin(public_holidays), col] *= 1.5

# Function to forecast using Holt-Winters
def forecast_holt_winters(data, product, days=7):
    model = ExponentialSmoothing(data[product], trend="add", seasonal="add", seasonal_periods=7)
    model_fit = model.fit()
    return model_fit.forecast(steps=days)

# Function to forecast using Facebook Prophet
def forecast_prophet(data, product, days=7):
    df = data.reset_index()[['Date', product]].rename(columns={'Date': 'ds', product: 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(days).set_index('ds')['yhat']

# Function to forecast using XGBoost
def forecast_xgboost(data, product):
    data['Lag_1'] = data[product].shift(1)
    data['Lag_7'] = data[product].shift(7)
    data['Rolling_Mean_7'] = data[product].rolling(window=7).mean()
    data.dropna(inplace=True)

    X = data[['Lag_1', 'Lag_7', 'Rolling_Mean_7']]
    y = data[product]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)
    return model.predict(X.tail(1))[0]

# Function to forecast using Linear Regression
def forecast_linear_regression(data, product):
    data['Lag_1'] = data[product].shift(1)
    data['Lag_7'] = data[product].shift(7)
    data['Rolling_Mean_7'] = data[product].rolling(window=7).mean()
    data.dropna(inplace=True)

    X = data[['Lag_1', 'Lag_7', 'Rolling_Mean_7']]
    y = data[product]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X.tail(1))[0]

# Function to predict orders based on selected model
def predict_order(dates, model):
    predictions = {product: [] for product in ['Tenderstem', 'Babycorn', 'Finebeans']}
    
    for date in dates:
        for product in predictions.keys():
            if model == "Holt-Winters":
                predictions[product].append(forecast_holt_winters(data, product, 7).iloc[0])
            elif model == "Prophet":
                predictions[product].append(forecast_prophet(data, product, 7).iloc[0])
            elif model == "XGBoost":
                predictions[product].append(forecast_xgboost(data, product))
            elif model == "Linear Regression":
                predictions[product].append(forecast_linear_regression(data, product))
    
    return pd.DataFrame(predictions, index=dates)

# --- Streamlit UI ---
st.title("📊 Advanced Sales Demand Forecasting")
st.markdown("Predict demand for **Tenderstem, Babycorn, and Finebeans** using different models.")

st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.radio("Select Forecasting Model", ["Holt-Winters", "Prophet", "XGBoost", "Linear Regression"])

start_date = st.sidebar.date_input("Select Start Date", min_value=pd.to_datetime('2025-01-01'), max_value=pd.to_datetime('2025-04-01'))
end_date = st.sidebar.date_input("Select End Date", min_value=start_date, max_value=pd.to_datetime('2025-04-01'))

if st.sidebar.button("Predict Demand"):
    future_dates = pd.date_range(start=start_date, end=end_date)
    forecast_df = predict_order(future_dates, selected_model)
    
    st.subheader(f"📅 Demand Forecast ({selected_model})")
    st.dataframe(forecast_df)

    fig = go.Figure()
    for product in ['Tenderstem', 'Babycorn', 'Finebeans']:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[product], mode='lines', name=product))
    fig.update_layout(title="Forecasted Demand Trends", xaxis_title="Date", yaxis_title="Demand")
    st.plotly_chart(fig)

# --- Heatmap Visualization ---
st.subheader("🔥 Demand Trends Heatmap")
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(data[['Tenderstem', 'Babycorn', 'Finebeans']].transpose(), cmap='coolwarm', annot=True, fmt=".0f", linewidths=0.5)
st.pyplot(fig)
