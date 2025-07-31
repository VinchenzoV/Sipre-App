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

# Load symbols with fallback (using a reliable source)
@st.cache_data
def load_symbols():
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().str.upper().tolist()
    except Exception:
        # fallback hardcoded list
        return ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "SPY", "BTC-USD", "ETH-USD"]

symbols_list = load_symbols()

# 1. User types partial symbol here:
user_input = st.text_input("Enter symbol (type to filter):").upper().strip()

# 2. Dynamically filter the list based on input and show dropdown:
if user_input:
    filtered_symbols = [s for s in symbols_list if user_input in s]
else:
    filtered_symbols = symbols_list

if filtered_symbols:
    symbol = st.selectbox("Select symbol:", filtered_symbols, index=0)
else:
    st.warning("No matching symbols found.")
    symbol = None

if symbol:
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
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']])
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])
        return np.array(X), np.array(y), scaler

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
                if len(df) >= 60:
                    st.info(f"Using timeframe: {tf}")
                    break
            else:
                st.warning("⚠️ Not enough data available for this symbol across all timeframes.")
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
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

            # Clean data to fix Prophet error:
            prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
            prophet_df = prophet_df.dropna(subset=['y'])

            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            fig1 = m.plot(forecast)
            st.pyplot(fig1)

            st.subheader("🤖 LSTM Future Price Prediction")
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
                pred = model.predict(future_input)[0][0]
                future_preds.append(pred)
                pred_array = np.array([[[pred]]], dtype=np.float32)
                future_input = np.concatenate((future_input[:, 1:, :], pred_array), axis=1)

            future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
            future_dates = future_dates.to_pydatetime().tolist()

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

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.text(traceback.format_exc())
