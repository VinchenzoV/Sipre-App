import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.set_page_config(page_title="Sipre - Trading Signal Dashboard", layout="wide")

st.title("📈 Sipre - Live Trading Signal Dashboard")

# 🔍 Symbol input with autocomplete
symbol = st.text_input("Enter a stock/crypto/forex symbol (e.g. AAPL, BTC-USD, EURUSD=X)", "AAPL").upper()

# ⏱️ Date range
period = st.selectbox("Select time period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# 📥 Fetch data
@st.cache_data(ttl=3600)
def load_data(symbol, period):
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_data(symbol, period)

if df.empty:
    st.error("No data found. Please check the symbol.")
    st.stop()

# 📈 Indicators
df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

latest = df.iloc[-1]
ema9_latest = latest["EMA9"]
ema21_latest = latest["EMA21"]
rsi_latest = latest["RSI"]

signal = "HOLD"
if ema9_latest > ema21_latest and rsi_latest > 50:
    signal = "BUY"
elif ema9_latest < ema21_latest and rsi_latest < 50:
    signal = "SELL"

# 📊 Signal Summary Table
st.subheader("📊 Signal Summary")
signal_table = pd.DataFrame({
    "Date": [latest.name.strftime('%Y-%m-%d')],
    "Close": [latest["Close"]],
    "EMA9": [ema9_latest],
    "EMA21": [ema21_latest],
    "RSI": [round(rsi_latest, 2)],
    "Signal": [signal]
})
st.table(signal_table)

# 📈 Price chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA9"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA21"))
fig.update_layout(title=f"{symbol} Price & EMAs", xaxis_title="Date", yaxis_title="Price", height=500)
st.plotly_chart(fig, use_container_width=True)

# 📅 Prophet Forecasting
st.subheader("📅 Prophet Forecast (7 Days)")
prophet_df = df.reset_index()[["Date", "Close"]]
prophet_df.columns = ["ds", "y"]
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors='coerce')

m = Prophet(daily_seasonality=True)
m.fit(prophet_df)

future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)

fig1 = m.plot(forecast)
st.plotly_chart(fig1, use_container_width=True)

# 📄 Prophet Forecast Table
st.subheader("📄 Prophet Forecast Data")
prophet_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
prophet_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
prophet_display['Date'] = prophet_display['Date'].dt.strftime('%Y-%m-%d')
st.table(prophet_display)

# 🤖 LSTM Forecasting
st.subheader("🤖 LSTM Future Price Prediction")

if len(df) <= 60:
    st.warning("Not enough data to run LSTM prediction (need > 60 data points). Skipping LSTM.")
else:
    data = df[["Close"]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Predict next 7 days
    last_60 = scaled_data[-60:]
    lstm_input = np.reshape(last_60, (1, 60, 1))
    predictions = []
    for _ in range(7):
        pred = model.predict(lstm_input, verbose=0)
        predictions.append(pred[0][0])
        lstm_input = np.append(lstm_input[:, 1:, :], [[[pred[0][0]]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)
    df_future = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions.flatten()})

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Historical"))
    fig2.add_trace(go.Scatter(x=df_future["Date"], y=df_future["Predicted Price"], name="LSTM Forecast"))
    fig2.update_layout(title=f"{symbol} LSTM 7-Day Forecast", xaxis_title="Date", yaxis_title="Price", height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # 📄 LSTM Forecast Table
    st.subheader("📄 LSTM Forecast Data")
    st.table(df_future.set_index("Date").round(2))
