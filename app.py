import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---- Generate Simulated Data ----
dates = pd.date_range(start='2025-01-01', periods=12*7, freq='D')  # 12 weeks (84 days)
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
data[['Tenderstem', 'babycorn', 'finebeans']] *= np.where(data['day_of_week'] >= 5, 1.2, 1.0)
data.loc[data.index.isin(public_holidays), ['Tenderstem', 'babycorn', 'finebeans']] *= 1.5

# ---- Forecasting Models ----
def forecast_holt_winters(data, product, forecast_days=5):
    product_data = data[product]
    model = ExponentialSmoothing(product_data, trend='add', seasonal='add', seasonal_periods=7)
    model_fit = model.fit()
    return model_fit.forecast(steps=forecast_days)

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

# ---- Streamlit UI ----
st.set_page_config(page_title="Sales Demand Prediction", layout="wide")

st.title("📊 Advanced Sales Demand Prediction")
st.write("Select a date, forecasting method, and see demand predictions.")

# ---- Sidebar ----
st.sidebar.header("📅 Select Date")
input_date = st.sidebar.date_input("Choose a date", min_value=pd.to_datetime('2025-01-01'), max_value=pd.to_datetime('2025-04-01'))

st.sidebar.header("🔍 Choose Forecasting Method")
forecasting_method = st.sidebar.selectbox(
    "Select a model",
    ["Holt-Winters", "XGBoost", "Linear Regression"]
)

# ---- Predict Demand ----
def predict_order_on_date(input_date, data, method):
    input_date = pd.to_datetime(input_date)
    forecasted_order = {}

    for product in ['Tenderstem', 'babycorn', 'finebeans']:
        if method == "Holt-Winters":
            forecasted_order[product] = forecast_holt_winters(data, product)[0]
        elif method == "XGBoost":
            forecasted_order[product] = forecast_xgboost(data, product)
        elif method == "Linear Regression":
            forecasted_order[product] = forecast_linear_regression(data, product)

    return forecasted_order

if st.sidebar.button("🔮 Predict Demand"):
    forecasted_order = predict_order_on_date(input_date, data, forecasting_method)
    
    st.subheader(f"📅 Predicted Orders for {input_date.strftime('%A, %d %B %Y')}")
    for product, forecast in forecasted_order.items():
        st.write(f"- **{product}:** {forecast:.2f} units")
    
    # Convert predictions to DataFrame
    forecast_df = pd.DataFrame.from_dict(forecasted_order, orient='index', columns=['Predicted Demand'])
    forecast_df.index.name = "Product"
    
    # Show data as table
    st.dataframe(forecast_df)

    # Download CSV button
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=forecast_df.to_csv().encode('utf-8'),
        file_name=f"predicted_demand_{input_date}.csv",
        mime="text/csv"
    )

# ---- Visualization ----
st.subheader("📈 Historical Demand Trends")
fig = px.line(data, x=data.index, y=['Tenderstem', 'babycorn', 'finebeans'], title="Demand Trends Over Time")
st.plotly_chart(fig, use_container_width=True)

