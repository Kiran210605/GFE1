import streamlit as st
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Simulate a dataset with data for 12 weeks (3 months)
dates = pd.date_range(start='2025-01-01', periods=12*7, freq='D')
tenderstem_demand = np.random.randint(150, 500, size=len(dates))
babycorn_demand = np.random.randint(100, 450, size=len(dates))
finebeans_demand = np.random.randint(200, 500, size=len(dates))

data = pd.DataFrame({
    'Date': dates,
    'Tenderstem': tenderstem_demand,
    'babycorn': babycorn_demand,
    'finebeans': finebeans_demand
})

data.set_index('Date', inplace=True)

public_holidays = pd.to_datetime([
    '2025-01-01', '2025-02-03', '2025-03-17', '2025-04-21',
    '2025-05-05', '2025-06-02', '2025-08-04', '2025-10-27',
    '2025-12-25', '2025-12-26'
])

data['day_of_week'] = data.index.dayofweek

data['Tenderstem'] *= np.where(data['day_of_week'] >= 5, 1.2, 1.0)
data['babycorn'] *= np.where(data['day_of_week'] >= 5, 1.2, 1.0)
data['finebeans'] *= np.where(data['day_of_week'] >= 5, 1.2, 1.0)

data.loc[data.index.isin(public_holidays), ['Tenderstem', 'babycorn', 'finebeans']] *= 1.5

def forecast_holt_winters(data, product, forecast_days=5):
    product_data = data[product]
    model = ExponentialSmoothing(product_data, trend='add', seasonal='add', seasonal_periods=7)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    return forecast

def forecast_xgboost(data, product):
    data['lag_1'] = data[product].shift(1)
    data['lag_7'] = data[product].shift(7)
    data['rolling_mean_7'] = data[product].rolling(window=7).mean()
    data.dropna(inplace=True)

    X = data[['lag_1', 'lag_7', 'rolling_mean_7']]
    y = data[product]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model.predict(X.tail(1))[0]

def forecast_linear_regression(data, product):
    data['lag_1'] = data[product].shift(1)
    data['lag_7'] = data[product].shift(7)
    data['rolling_mean_7'] = data[product].rolling(window=7).mean()
    data.dropna(inplace=True)

    X = data[['lag_1', 'lag_7', 'rolling_mean_7']]
    y = data[product]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model.predict(X.tail(1))[0]

def predict_order_on_date(input_date, data):
    input_date = pd.to_datetime(input_date)
    forecasted_order = {}

    for product in ['Tenderstem', 'babycorn', 'finebeans']:
        hw_forecast = forecast_holt_winters(data, product)
        forecasted_order[product] = hw_forecast[0]

    for product in ['Tenderstem', 'babycorn', 'finebeans']:
        xgb_forecast = forecast_xgboost(data, product)
        forecasted_order[product] = (forecasted_order[product] + xgb_forecast) / 2

    for product in ['Tenderstem', 'babycorn', 'finebeans']:
        lr_forecast = forecast_linear_regression(data, product)
        forecasted_order[product] = (forecasted_order[product] + lr_forecast) / 2

    return forecasted_order

st.title("Product Sales Prediction")
st.write("Select a date to forecast the demand for Tenderstem, Babycorn, and Finebeans.")

input_date = st.date_input("Select a date", min_value=pd.to_datetime('2025-01-01'), max_value=pd.to_datetime('2025-04-01'))

if st.button("Predict Order"):
    forecasted_order = predict_order_on_date(input_date, data)
    st.write(f"### Predicted Orders for {input_date.strftime('%A, %d %B %Y')}:")
    for product, forecast in forecasted_order.items():
        st.write(f"- **{product}:** {forecast:.2f} units")
