import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Sipre", layout="wide")
st.title("📈 Sipre — Stock Signal App (Free & Live with Yahoo Finance)")

symbol = st.text_input("Enter a stock symbol (e.g. AAPL, TSLA, MSFT)", value="AAPL")

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if st.button("Get Signal"):
    try:
        df = yf.download(symbol, period="7d", interval="1h")
        df.dropna(inplace=True)
        df["EMA9"] = calculate_ema(df["Close"], 9)
        df["EMA21"] = calculate_ema(df["Close"], 21)
        df["RSI"] = calculate_rsi(df["Close"])

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        signal = "Neutral"
        if prev["EMA9"] < prev["EMA21"] and latest["EMA9"] > latest["EMA21"] and latest["RSI"] > 30:
            signal = "Buy ✅"
        elif prev["EMA9"] > prev["EMA21"] and latest["EMA9"] < latest["EMA21"] and latest["RSI"] < 70:
            signal = "Sell ❌"

        st.subheader(f"Signal: {signal}")
        st.text(f"RSI: {round(latest['RSI'], 2)}")

        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Close"], label="Price", color="blue")
        ax.plot(df.index, df["EMA9"], label="EMA9", color="orange")
        ax.plot(df.index, df["EMA21"], label="EMA21", color="red")
        ax.set_title(f"{symbol.upper()} Price Chart with EMA9, EMA21")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading data: {e}")
