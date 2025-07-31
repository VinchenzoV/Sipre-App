import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
import plotly.graph_objs as go
import traceback

st.set_page_config(page_title="Sipre Pro", layout="wide")
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

@st.cache_data
def load_symbols():
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().str.upper().tolist()
    except Exception:
        return ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "SPY", "BTC-USD", "ETH-USD"]

symbols_list = load_symbols()

user_input = st.text_input("Enter symbol (e.g. LNR.TO or AAPL):").upper().strip()
filtered_symbols = [s for s in symbols_list if user_input in s] if user_input else symbols_list
selected_symbol = st.selectbox("Or select from suggestions:", filtered_symbols) if filtered_symbols else None
symbol = user_input if user_input else selected_symbol

if not symbol:
    st.warning("Please enter or select a valid symbol.")
    st.stop()

timeframe = st.selectbox("Select timeframe:", ["1mo", "3mo", "6mo", "1y"])
alert_email = st.text_input("Enter your email for alerts (optional):")

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_lstm_data(df, sequence_length=60):
    if df.shape[0] <= sequence_length:
        sequence_length = max(10, df.shape[0] - 1)
        if sequence_length < 10:
            raise ValueError("Still too little data for LSTM.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    
    X = np.array(X)
    y = np.array(y)
    if X.ndim != 3:
        raise ValueError(f"Unexpected LSTM input shape: {X.shape}")
    return X, y, scaler

def fetch_news_sentiment(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        response = requests.get(url)
        return "Sentiment: [Mock sentiment placeholder]"
    except:
        return "Sentiment unavailable"

def send_email_alert(recipient, signal, symbol):
    try:
        st.success(f"Alert email would be sent to {recipient} (demo)")
    except:
        st.error("Failed to send email alert.")

if st.button("Get Prediction & Signal"):
    try:
        timeframes_to_try = [timeframe, "3mo", "6mo", "1y"]
        for tf in timeframes_to_try:
            df = yf.download(symbol, period=tf, interval="1d")
            if len(df) >= 30:
                st.info(f"Using timeframe: {tf}")
                break
        else:
            st.warning("⚠️ Not enough data available for this symbol.")
            st.stop()

        df.dropna(inplace=True)
        df['EMA9'] = calculate_ema(df['Close'], 9)
        df['EMA21'] = calculate_ema(df['Close'], 21)
        df['RSI'] = calculate_rsi(df['Close'])
        df.dropna(inplace=True)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        ema9_latest = float(latest["EMA9"])
        ema21_latest = float(latest["EMA21"])
        ema9_prev = float(prev["EMA9"])
        ema21_prev = float(prev["EMA21"])
        rsi_latest = float(latest["RSI"])

        signal = "Neutral"
        if (ema9_prev < ema21_prev) and (ema9_latest > ema21_latest) and (rsi_latest > 30):
            signal = "Buy ✅"
        elif (ema9_prev > ema21_prev) and (ema9_latest < ema21_latest) and (rsi_latest < 70):
            signal = "Sell ❌"

        st.subheader(f"📌 Signal: {signal}")
        st.markdown(f"**RSI:** {round(rsi_latest, 2)}")

        if alert_email and signal != "Neutral":
            send_email_alert(alert_email, signal, symbol)

        st.subheader("📰 News Sentiment (Mocked)")
        st.markdown(fetch_news_sentiment(symbol))

        st.subheader("📅 Prophet Forecast (Next 30 Days)")
        prophet_df = df.reset_index()
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(prophet_df[prophet_df.columns[0]]),
            'y': pd.to_numeric(prophet_df['Close'], errors='coerce')
        }).dropna()

        if prophet_df.shape[0] < 30:
            st.warning("Not enough data for Prophet forecasting.")
        else:
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            fig1 = m.plot(forecast)
            st.pyplot(fig1.figure)

            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10), use_container_width=True)
            st.download_button("📥 Download Prophet Forecast", forecast.to_csv(index=False), file_name=f"{symbol}_prophet_forecast.csv")

        st.subheader("🤖 LSTM Future Price Prediction")
        try:
            X, y, scaler = prepare_lstm_data(df)
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)

            future_input = X[-1].reshape(1, X.shape[1], 1)
            future_preds = []
            for _ in range(10):
                pred = model.predict(future_input, verbose=0)[0][0]
                future_preds.append(pred)
                pred_array = np.array([[[pred]]], dtype=np.float32)
                future_input = np.concatenate((future_input[:, 1:, :], pred_array), axis=1)

            future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')

            df_future = pd.DataFrame({
                'Date': pd.to_datetime(future_dates),
                'Predicted Close': future_prices
            })

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index.to_list(), y=df['Close'].to_list(), name="Historical"))
            fig2.add_trace(go.Scatter(x=df_future['Date'].to_list(), y=df_future['Predicted Close'].to_list(),
                                      name="LSTM Forecast", line=dict(dash='dot')))
            fig2.update_layout(title=f"{symbol} — Combined Forecast View")
            st.plotly_chart(fig2)

            st.dataframe(df_future, use_container_width=True)
            st.download_button("📥 Download LSTM Forecast", df_future.to_csv(index=False), file_name=f"{symbol}_lstm_forecast.csv")

        except ValueError as ve:
            st.warning(f"LSTM Skipped: {ve}")
        except Exception as e:
            st.error(f"❌ LSTM Error: {e}")
            st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.text(traceback.format_exc())
