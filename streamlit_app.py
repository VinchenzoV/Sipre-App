import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go
import datetime
import traceback

# --- Streamlit Config ---
st.set_page_config(page_title="📈 Sipre Pro", layout="wide")
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")
st.markdown("Professional-grade predictive tool for stock and crypto markets using Prophet and LSTM models.")

# --- User Inputs ---
symbol = st.text_input("Enter symbol (e.g. AAPL, LNR.TO, BTC-USD):", "AAPL")
timeframe = st.selectbox("Select timeframe:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
predict_days_prophet = st.number_input("Number of days to predict (Prophet):", min_value=7, max_value=90, value=30)
predict_days_lstm = st.number_input("Number of days to predict (LSTM):", min_value=7, max_value=60, value=30)
email = st.text_input("Enter your email for alerts (optional):")

st.write(f"Using timeframe: {timeframe}")

# --- Download Data ---
data = yf.download(symbol, period=timeframe, interval="1d")
if data.empty:
    st.error("No data found for the selected symbol.")
    st.stop()

data = data.dropna()
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])

# --- Feature Engineering ---
data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# --- Signal ---
try:
    ema9_prev, ema21_prev = data['EMA9'].iloc[-2], data['EMA21'].iloc[-2]
    ema9_latest, ema21_latest = data['EMA9'].iloc[-1], data['EMA21'].iloc[-1]
    rsi_latest = data['RSI'].iloc[-1]
    signal = "Neutral"
    if (ema9_prev < ema21_prev) and (ema9_latest > ema21_latest) and (rsi_latest > 30):
        signal = "Buy"
    elif (ema9_prev > ema21_prev) and (ema9_latest < ema21_latest) and (rsi_latest < 70):
        signal = "Sell"
    st.subheader(f"📌 Signal: {signal}")
    st.write(f"RSI: {rsi_latest:.2f}")
except Exception:
    st.write("Unable to compute signal.")

# --- Prophet Forecast ---
st.subheader(f"📅 Prophet Forecast (Next {predict_days_prophet} Days)")
prophet_df = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
prophet_df = prophet_df[prophet_df['y'].notnull()]

try:
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=predict_days_prophet)
    forecast = model.predict(future)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig1.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig1, use_container_width=True)
except Exception as e:
    st.error(f"❌ Error in Prophet prediction: {e}")
    st.text(traceback.format_exc())

# --- LSTM Forecast ---
st.subheader(f"🤖 LSTM Future Price Prediction (Next {predict_days_lstm} Days)")
try:
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(data[['Close']])

    window_size = 60
    X, y = [], []
    for i in range(window_size, len(close_scaled)):
        X.append(close_scaled[i - window_size:i, 0])
        y.append(close_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_60 = close_scaled[-60:]
    predictions = []
    current_input = last_60.reshape(1, 60, 1)
    for _ in range(predict_days_lstm):
        pred = model.predict(current_input, verbose=0)
        predictions.append(pred[0, 0])
        current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

    forecast_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(data['Date'].iloc[-1] + datetime.timedelta(days=1), periods=predict_days_lstm)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': forecast_prices.flatten()})

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical'))
    fig2.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Price'], name='LSTM Prediction'))
    fig2.update_layout(title="LSTM Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"❌ LSTM Error: {e}")
    st.text(traceback.format_exc())
