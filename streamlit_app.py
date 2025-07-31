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

st.set_page_config(page_title="Sipre Pro", layout="wide")
st.title("📈 Sipre Pro: Smart Indicator & Forecast Dashboard")

# --- Input Section ---
ticker = st.text_input("Enter Ticker Symbol (e.g. TSLA, AAPL, BTC-USD)", value="TSLA")
period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# --- Data Download ---
st.subheader("🔍 Price Chart & Indicators")
data_load_state = st.text("Loading data...")
df = yf.download(ticker, period=period)
data_load_state.text("Data loaded!")

if df.empty:
    st.warning("No data found for this ticker.")
    st.stop()

# --- Indicators ---
df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# --- Chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA 9'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA 21'))
fig.update_layout(title=f"{ticker} Price Chart with EMA Indicators", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# --- Prophet Forecast ---
st.subheader("📅 Prophet Forecast (Next 15 Days)")
df_reset = df.reset_index()
df_prophet = df_reset[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

if len(df_prophet) >= 30:
    try:
        m = Prophet()
        m.fit(df_prophet)

        future = m.make_future_dataframe(periods=15)
        forecast = m.predict(future)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Actual'))
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
        st.plotly_chart(fig1, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Prophet Error: {e}")
else:
    st.warning("Not enough data for Prophet forecasting (need at least 30 data points).")

# --- LSTM Forecast ---
st.subheader("🤖 LSTM Future Price Prediction")
if len(df) < 70:
    st.warning("LSTM Skipped: Not enough data points for LSTM prediction (need at least ~70).")
else:
    try:
        look_back = 60
        df_lstm = df[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_lstm)

        x_train, y_train = [], []
        for i in range(look_back, len(scaled_data)):
            x_train.append(scaled_data[i - look_back:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

        last_60_days = scaled_data[-look_back:]
        input_seq = last_60_days.reshape(1, look_back, 1)
        future_prices = []
        for _ in range(15):
            pred = model.predict(input_seq, verbose=0)
            future_prices.append(pred[0, 0])
            input_seq = np.append(input_seq[:, 1:, :], [[pred[0]]], axis=1)

        future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1)).flatten()

        # Bound the forecast
        current_price = df['Close'].iloc[-1]
        price_floor = current_price * 0.8
        price_ceiling = current_price * 1.2
        clipped_prices = np.clip(future_prices, price_floor, price_ceiling)

        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=15)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical"))
        fig2.add_trace(go.Scatter(x=future_dates, y=clipped_prices, name="LSTM Forecast"))
        fig2.update_layout(title=f"{ticker} LSTM Forecast (15 Days)", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"❌ LSTM Error: {e}")
