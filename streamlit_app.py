import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import requests
import time
import datetime
from bs4 import BeautifulSoup
import openai

# ----- Streamlit Config -----
st.set_page_config(page_title="Sipre", layout="wide")
st.title("📊 Sipre — AI-Enhanced Live Trading Dashboard")

openai.api_key = "your-openai-api-key"  # Optional: replace with real key if using OpenAI sentiment

# ----- Global Symbol List -----
symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BTC-USD", "ETH-USD", "SPY", "QQQ"]

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
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
        return [h.text for h in headlines if h.text.strip()]
    except:
        return ["No news available."]

def get_sentiment(text):
    try:
        prompt = f"Summarize the sentiment of this news headline: '{text}'"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except:
        return "Sentiment unavailable"

def analyze_symbol(symbol, timeframe="5d", interval="1h"):
    try:
        df = yf.download(symbol, period=timeframe, interval=interval)
        if df.empty or len(df) < 26:
            return None

        df["EMA9"] = calculate_ema(df["Close"], 9)
        df["EMA21"] = calculate_ema(df["Close"], 21)
        df["RSI"] = calculate_rsi(df["Close"])
        df["MACD"], df["MACD_Signal"] = calculate_macd(df["Close"])
        df.dropna(inplace=True)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        ema_cross_up = prev["EMA9"] < prev["EMA21"] and latest["EMA9"] > latest["EMA21"]
        ema_cross_down = prev["EMA9"] > prev["EMA21"] and latest["EMA9"] < latest["EMA21"]
        rsi = float(latest["RSI"])

        if ema_cross_up and rsi > 30:
            signal = "Buy ✅"
        elif ema_cross_down and rsi < 70:
            signal = "Sell ❌"
        else:
            signal = "Neutral"

        return {
            "Symbol": symbol,
            "Price": round(latest["Close"], 2),
            "RSI": round(rsi, 2),
            "Signal": signal,
            "MACD": round(latest["MACD"], 2),
            "Volume": int(latest["Volume"]),
            "News": get_news_headlines(symbol)
        }

    except Exception as e:
        return None

# ----- Sidebar Controls -----
st.sidebar.header("⚙️ Dashboard Controls")

selected_symbols = st.sidebar.multiselect("Select tickers to monitor:", symbols, default=symbols[:5])
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "5d", "1mo", "3mo", "6mo"])
interval = st.sidebar.selectbox("Interval", ["15m", "30m", "1h", "1d"])
refresh_interval = st.sidebar.number_input("Auto-refresh (minutes)", min_value=0, max_value=60, value=0)
filter_signal = st.sidebar.selectbox("Filter signals", ["All", "Buy ✅", "Sell ❌", "Neutral"])

st.sidebar.markdown("---")
export_btn = st.sidebar.button("📤 Export Data to CSV")

st.subheader("📡 Live Signal Scanner")

signal_log = []

placeholder = st.empty()
last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

def run_scan():
    results = []
    for symbol in selected_symbols:
        result = analyze_symbol(symbol, timeframe=timeframe, interval=interval)
        if result:
            if filter_signal == "All" or result["Signal"] == filter_signal:
                results.append(result)
                signal_log.append({
                    "Time": last_updated,
                    **result
                })
    return pd.DataFrame(results)

data = run_scan()

if not data.empty:
    st.dataframe(data[["Symbol", "Price", "RSI", "MACD", "Volume", "Signal"]].set_index("Symbol"), use_container_width=True)
else:
    st.warning("No data to display.")

# ----- Export -----
if export_btn and not data.empty:
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "sipre_signals.csv", "text/csv", key='download-csv')

# ----- Auto-refresh -----
if refresh_interval > 0:
    st.info(f"⏳ Auto-refresh every {refresh_interval} minutes is enabled.")
    time.sleep(refresh_interval * 60)
    st.rerun()

st.markdown("---")
st.subheader("🗞️ News & AI Sentiment")

for symbol in selected_symbols:
    news = get_news_headlines(symbol)
    st.markdown(f"**{symbol} Headlines:**")
    for headline in news:
        st.markdown(f"- {headline}")
        sentiment = get_sentiment(headline)
        st.caption(f"> 💬 *Sentiment:* {sentiment}")
    st.markdown("---")
