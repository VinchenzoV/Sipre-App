import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime
import traceback

# App Config
st.set_page_config(page_title="Sipre Pro", layout="wide")
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Enter Symbol (e.g., AAPL or BTC-USD):", "AAPL")
timeframe = st.sidebar.selectbox("Select Timeframe:", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
predict_days_prophet = st.sidebar.number_input("Days to Predict (Prophet):", min_value=7, max_value=90, value=30)
predict_days_lstm = st.sidebar.number_input("Days to Predict (LSTM):", min_value=7, max_value=90, value=30)
email = st.sidebar.text_input("Email for Alerts (optional):")

# --- Load Data ---
@st.cache_data(show_spinner=False)
def load_data(symbol, timeframe):
    end = datetime.datetime.today()
    delta_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    start = end - datetime.timedelta(days=delta_map[timeframe])
    df = yf.download(symbol, start=start, end=end)
    df = df.dropna()
    return df

try:
    df = load_data(symbol, timeframe)
    st.subheader(f"Using timeframe: {timeframe}")
except Exception as e:
    st.error("Error loading data. Please check the symbol or try again later.")
    st.stop()

# --- Signal Calculation ---
df["EMA9"] = df["Close"].ewm(span=9).mean()
df["EMA21"] = df["Close"].ewm(span=21).mean()
delta = df["Close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# --- Signal Display ---
latest = df.iloc[-1]
prev = df.iloc[-2]
ema9_latest, ema21_latest = latest["EMA9"], latest["EMA21"]
ema9_prev, ema21_prev = prev["EMA9"], prev["EMA21"]
rsi_latest = latest["RSI"]

signal = "Neutral"
if (ema9_prev < ema21_prev) and (ema9_latest > ema21_latest) and (rsi_latest > 30):
    signal = "Buy"
elif (ema9_prev > ema21_prev) and (ema9_latest < ema21_latest) and (rsi_latest < 70):
    signal = "Sell"

st.markdown(f"### 📌 Signal: {signal}")
st.markdown(f"RSI: {rsi_latest:.2f}")

# --- Prophet Forecast ---
st.markdown(f"### 🗕️ Prophet Forecast (Next {predict_days_prophet} Days)")
df_prophet = df[["Close"].copy()]
df_prophet.reset_index(inplace=True)
df_prophet.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

try:
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=predict_days_prophet)
    forecast = model.predict(future)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], name="Historical"))
    fig1.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    st.plotly_chart(fig1, use_container_width=True)
except Exception as e:
    st.error("Prophet model error: " + str(e))

# --- LSTM Forecast ---
st.markdown(f"### 🤖 LSTM Future Price Prediction (Next {predict_days_lstm} Days)")

def lstm_forecast(data, n_steps=60, n_days=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_sequence = scaled_data[-n_steps:]
    forecasted = []
    for _ in range(n_days):
        last_input = last_sequence.reshape((1, n_steps, 1))
        next_price = model.predict(last_input, verbose=0)[0][0]
        forecasted.append(next_price)
        last_sequence = np.append(last_sequence[1:], [[next_price]], axis=0)

    forecasted = scaler.inverse_transform(np.array(forecasted).reshape(-1, 1)).flatten()
    return forecasted

try:
    lstm_preds = lstm_forecast(df, n_days=predict_days_lstm)
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=predict_days_lstm)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Historical"))
    fig2.add_trace(go.Scatter(x=future_dates, y=lstm_preds, name="LSTM Prediction"))
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error("LSTM model error: " + str(e))

# --- Footer ---
st.markdown("---")
st.caption("Built with Streamlit | Predictive models by Prophet and LSTM | © 2025 Sipre Pro")
