# Sipre Pro: Full-featured Predictive Trading Dashboard
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
import smtplib
from email.mime.text import MIMEText
import plotly.graph_objs as go

st.set_page_config(page_title="Sipre Pro", layout="wide")
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

popular_symbols = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BTC-USD", "ETH-USD", "SPY"]
symbol_choice = st.selectbox("Choose a popular symbol:", popular_symbols)
custom_symbol = st.text_input("Or enter a custom symbol:", value=symbol_choice)
timeframe = st.selectbox("Select timeframe:", ["1mo", "3mo", "6mo", "1y"])
alert_email = st.text_input("Enter your email for alerts (optional):")

# --- Helper Functions ---
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
        msg = MIMEText(f"Signal alert for {symbol}: {signal}")
        msg['Subject'] = f"Sipre Signal Alert — {symbol}"
        msg['From'] = "sipre.alerts@example.com"
        msg['To'] = recipient
        # Placeholder for real SMTP server setup
        # s = smtplib.SMTP('smtp.example.com', 587)
        # s.starttls()
        # s.login('username', 'password')
        # s.sendmail(msg['From'], [msg['To']], msg.as_string())
        # s.quit()
        st.success(f"Alert email would be sent to {recipient} (demo)")
    except:
        st.error("Failed to send email alert.")

# --- Main Signal Logic ---
if st.button("Get Prediction & Signal"):
    try:
        df = yf.download(custom_symbol, period=timeframe, interval="1d")
        if df.empty or len(df) < 100:
            st.warning("⚠️ Not enough data for analysis.")
        else:
            df.dropna(inplace=True)
            df['EMA9'] = calculate_ema(df['Close'], 9)
            df['EMA21'] = calculate_ema(df['Close'], 21)
            df['RSI'] = calculate_rsi(df['Close'])
            df.dropna(inplace=True)

            latest = df.iloc[-1]
            prev = df.iloc[-2]
            signal = "Neutral"
            if prev['EMA9'] < prev['EMA21'] and latest['EMA9'] > latest['EMA21'] and latest['RSI'] > 30:
                signal = "Buy ✅"
            elif prev['EMA9'] > prev['EMA21'] and latest['EMA9'] < latest['EMA21'] and latest['RSI'] < 70:
                signal = "Sell ❌"
            st.subheader(f"📌 Signal: {signal}")
            st.markdown(f"**RSI:** {round(latest['RSI'], 2)}")

            if alert_email and signal != "Neutral":
                send_email_alert(alert_email, signal, custom_symbol)

            st.subheader("📰 News Sentiment (Mocked)")
            st.markdown(fetch_news_sentiment(custom_symbol))

            st.subheader("📅 Prophet Forecast (Next 30 Days)")
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
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
                future_input = np.append(future_input[:, 1:, :], [[[pred]]], axis=1)

            future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
            future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=10)
            df_future = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_prices.flatten()})

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical"))
            fig2.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Predicted Close'], name="LSTM Forecast", line=dict(dash='dot')))
            fig2.update_layout(title=f"{custom_symbol} — Combined Forecast View")
            st.plotly_chart(fig2)

    except Exception as e:
        st.error(f"❌ Error: {e}")
