import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go
import datetime
import traceback

st.set_page_config(page_title="Sipre Pro", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>📈 Sipre Pro — Predictive Trading Signal Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Inputs
symbol = st.text_input("Enter symbol (e.g. LNR.TO or AAPL):", value="AAPL")
timeframe = st.selectbox("Select timeframe:", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
email_alert = st.text_input("Enter your email for alerts (optional):")
n_prophet_days = st.number_input("Number of days to predict (Prophet):", min_value=1, max_value=365, value=15)
n_lstm_days = st.number_input("Number of days to predict (LSTM):", min_value=1, max_value=60, value=30)

st.markdown(f"**Using timeframe:** {timeframe}")

# Load data
try:
    df = yf.download(symbol, period=timeframe)
    if df.empty:
        st.error("No data found. Please check the symbol.")
        st.stop()

    df['RSI'] = df['Close'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x.diff().dropna().apply(lambda y: max(y, 0)).sum() / max(-x.diff().dropna().apply(lambda y: min(y, 0)).sum(), 1)))))
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    ema9_prev = prev['EMA9']
    ema21_prev = prev['EMA21']
    ema9_latest = latest['EMA9']
    ema21_latest = latest['EMA21']
    rsi_latest = latest['RSI']

    signal = "Neutral"
    if (ema9_prev < ema21_prev) and (ema9_latest > ema21_latest) and (rsi_latest > 30):
        signal = "Buy 🔼"
    elif (ema9_prev > ema21_prev) and (ema9_latest < ema21_latest) and (rsi_latest < 70):
        signal = "Sell 🔽"

    st.markdown(f"### 📌 Signal: {signal}")
    st.markdown(f"**RSI:** {rsi_latest:.2f}")

    # Prophet Forecast
    st.subheader(f"📅 Prophet Forecast (Next {n_prophet_days} Days)")
    df_reset = df.reset_index()
    df_reset = df_reset[['Date', 'Close']]
    df_reset.columns = ['ds', 'y']
    df_reset['y'] = np.log(df_reset['y'].clip(lower=1.0))

    prophet_model = Prophet()
    prophet_model.fit(df_reset)

    future = prophet_model.make_future_dataframe(periods=n_prophet_days)
    forecast = prophet_model.predict(future)

    fig_prophet = go.Figure()
    fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=np.exp(forecast['yhat']), name='Forecast'))
    fig_prophet.add_trace(go.Scatter(x=df_reset['ds'], y=np.exp(df_reset['y']), name='Actual'))
    fig_prophet.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_prophet, use_container_width=True)

    st.write("Sample of data used for Prophet:")
    st.dataframe(df_reset.tail())

    # LSTM Forecast
    st.subheader(f"🤖 LSTM Future Price Prediction (Next {n_lstm_days} Days)")

    prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    sequence_length = 60
    X = []
    y = []

    for i in range(sequence_length, len(scaled_prices)):
        X.append(scaled_prices[i-sequence_length:i])
        y.append(scaled_prices[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_sequence = scaled_prices[-sequence_length:]
    future_preds_scaled = []
    current_seq = last_sequence.copy()

    for _ in range(n_lstm_days):
        prediction = model.predict(current_seq.reshape(1, sequence_length, 1), verbose=0)
        future_preds_scaled.append(prediction[0][0])
        current_seq = np.append(current_seq[1:], prediction, axis=0)

    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_lstm_days)
    lstm_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_preds})
    st.line_chart(lstm_df.set_index('Date'))

except Exception as e:
    st.error("Unexpected error: " + str(e))
    st.text(traceback.format_exc())
