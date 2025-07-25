import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sipre", layout="wide")
st.title("📊 Sipre — Live Trading Dashboard (Single Symbol)")

# ========== SYMBOL + TIMEFRAME ==========
popular_symbols = ["AAPL", "TSLA", "MSFT", "SPY", "BTC-USD"]
symbol = st.selectbox("Select a symbol:", popular_symbols)
timeframe = st.selectbox("Timeframe:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
interval = st.selectbox("Interval:", ["15m", "30m", "1h", "1d"])

# ========== FUNCTIONS ==========
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(price):
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def get_news_headlines(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}"
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        headlines = soup.find_all("h3", limit=3)
        return [h.text.strip() for h in headlines if h.text.strip()]
    except:
        return ["No news found"]

# ========== PROCESS & DISPLAY ==========
if st.button("Analyze"):
    df = yf.download(symbol, period=timeframe, interval=interval)

    if df.empty or len(df) < 2:
        st.warning("⚠️ No data available. Try a different symbol, timeframe, or interval.")
    else:
        df["EMA9"] = calculate_ema(df["Close"], 9)
        df["EMA21"] = calculate_ema(df["Close"], 21)
        df["RSI"] = calculate_rsi(df["Close"])
        df["MACD"], df["MACD_Signal"] = calculate_macd(df["Close"])
        df.dropna(inplace=True)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        signal = "Neutral"
        if prev["EMA9"] < prev["EMA21"] and latest["EMA9"] > latest["EMA21"] and latest["RSI"] > 30:
            signal = "Buy ✅"
        elif prev["EMA9"] > prev["EMA21"] and latest["EMA9"] < latest["EMA21"] and latest["RSI"] < 70:
            signal = "Sell ❌"

        st.subheader(f"Signal: {signal}")
        st.write(f"**Price:** {round(latest['Close'], 2)} | **RSI:** {round(latest['RSI'], 2)} | **MACD:** {round(latest['MACD'], 2)}")

        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Close"], label="Price", color="blue")
        ax.plot(df.index, df["EMA9"], label="EMA9", color="orange")
        ax.plot(df.index, df["EMA21"], label="EMA21", color="red")
        ax.set_title(f"{symbol} — Price + EMA")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # News
        st.markdown("---")
        st.subheader("📰 Recent News")
        headlines = get_news_headlines(symbol)
        for h in headlines:
            st.markdown(f"- {h}")
