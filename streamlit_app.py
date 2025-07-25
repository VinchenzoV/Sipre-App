import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import time
import os

# ========== CONFIG ==========
st.set_page_config(page_title="Sipre", layout="wide")
st.title("📊 Sipre — Live Trading Dashboard (Single Symbol)")
REFRESH_INTERVAL = 60  # seconds

# ========== SYMBOL INPUT ==========
popular_symbols = ["AAPL", "TSLA", "MSFT", "SPY", "BTC-USD"]
default_symbol = popular_symbols[0]
symbol_input = st.text_input("Enter a symbol (e.g. AAPL, BTC-USD, ETH-USD, EURUSD=X):", value=default_symbol)
symbol = symbol_input.strip().upper()
timeframe = st.selectbox("Timeframe:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
interval = st.selectbox("Interval:", ["15m", "30m", "1h", "1d"])
auto_refresh = st.checkbox("🔄 Auto-refresh every 60s", value=True)

# ========== CALCULATIONS ==========
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

def send_telegram_alert(message):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")  # Set as environment variable
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"⚠️ Telegram alert failed: {e}")

# ========== MAIN LOGIC ==========
placeholder = st.empty()
run = st.button("Analyze")

if run or auto_refresh:
    if auto_refresh:
        time.sleep(1)  # Prevent infinite rerun loop

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
        suggestion = "Hold / Wait 📉"

        if float(prev["EMA9"]) < float(prev["EMA21"]) and float(latest["EMA9"]) > float(latest["EMA21"]) and float(latest["RSI"]) > 30:
            signal = "Buy ✅"
            suggestion = "📈 Suggestion: Buy — Uptrend and momentum building."
        elif float(prev["EMA9"]) > float(prev["EMA21"]) and float(latest["EMA9"]) < float(latest["EMA21"]) and float(latest["RSI"]) < 70:
            signal = "Sell ❌"
            suggestion = "📉 Suggestion: Sell — Weakening price action and crossover down."

        st.subheader(f"Signal: {signal}")
        st.write(f"**Price:** {round(latest['Close'], 2)} | **RSI:** {round(latest['RSI'], 2)} | **MACD:** {round(latest['MACD'], 2)}")
        st.markdown(f"<div style='color:yellow; font-weight:bold'>{suggestion}</div>", unsafe_allow_html=True)

        # ========== CHART ==========
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA9", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA21", line=dict(color="red")))
        fig.update_layout(title=f"{symbol} — Price with EMAs", xaxis_title="Time", yaxis_title="Price", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # ========== NEWS ==========
        st.markdown("---")
        st.subheader("📰 Recent News")
        headlines = get_news_headlines(symbol)
        for h in headlines:
            st.markdown(f"- {h}")

        # ========== TELEGRAM ==========
        send_telegram_alert(f"{symbol} Signal: {signal} | Price: {round(latest['Close'], 2)}")

    if auto_refresh:
        st.experimental_rerun()
