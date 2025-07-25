import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import openai

# ========== SETTINGS ==========
st.set_page_config(page_title="Sipre", layout="wide")
st.title("📊 Sipre — Free Live Trading Signal Dashboard")

openai.api_key = "your-openai-api-key"  # Optional: replace if using sentiment

default_symbols = ["AAPL", "TSLA", "MSFT", "SPY", "BTC-USD"]

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

def get_sentiment(text):
    try:
        prompt = f"Summarize the sentiment of this headline: '{text}'"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except:
        return "Unavailable"

def safe_yf_download(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty or len(df) < 30:
            return None
        return df
    except:
        return None

def analyze_symbol(symbol, period, interval):
    df = safe_yf_download(symbol, period, interval)
    if df is None or len(df) < 30:
        return None

    df["EMA9"] = calculate_ema(df["Close"], 9)
    df["EMA21"] = calculate_ema(df["Close"], 21)
    df["RSI"] = calculate_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"] = calculate_macd(df["Close"])
    df.dropna(inplace=True)

    if len(df) < 2:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        ema_cross_up = prev["EMA9"] < prev["EMA21"] and latest["EMA9"] > latest["EMA21"]
        ema_cross_down = prev["EMA9"] > prev["EMA21"] and latest["EMA9"] < latest["EMA21"]
    except:
        return None

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
        "MACD": round(latest["MACD"], 2),
        "Volume": int(latest["Volume"]),
        "Signal": signal,
        "News": get_news_headlines(symbol)
    }

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Settings")

symbols = st.sidebar.multiselect("Symbols", default_symbols, default=default_symbols[:3])

timeframes = {
    "1d": ["15m", "30m"],
    "5d": ["30m", "1h"],
    "1mo": ["1h", "1d"],
    "3mo": ["1d"],
    "6mo": ["1d"]
}

timeframe = st.sidebar.selectbox("Timeframe", list(timeframes.keys()), index=1)
interval = st.sidebar.selectbox("Interval", timeframes[timeframe])
filter_signal = st.sidebar.selectbox("Filter", ["All", "Buy ✅", "Sell ❌", "Neutral"])
auto_refresh = st.sidebar.number_input("Auto-refresh (min)", 0, 60, 0)
export_btn = st.sidebar.button("Export CSV")

# ========== MAIN SCAN ==========
st.subheader("📡 Live Trading Signals")
signal_log = []
errors = []
now = time.strftime("%Y-%m-%d %H:%M:%S")

results = []
for symbol in symbols:
    res = analyze_symbol(symbol, period=timeframe, interval=interval)
    if res:
        if filter_signal == "All" or res["Signal"] == filter_signal:
            results.append(res)
            signal_log.append({ "Time": now, **res })
    else:
        errors.append(symbol)

if errors:
    st.warning(f"⚠️ No data for: {', '.join(errors)}")

df = pd.DataFrame(results)
if not df.empty:
    st.dataframe(df[["Symbol", "Price", "RSI", "MACD", "Volume", "Signal"]].set_index("Symbol"), use_container_width=True)
else:
    st.error("No signals found. Try different symbols or interval.")

# ========== CSV EXPORT ==========
if export_btn and not df.empty:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "sipre_signals.csv", "text/csv")

# ========== AUTO REFRESH ==========
if auto_refresh > 0:
    st.info(f"🔄 Refreshing every {auto_refresh} min...")
    time.sleep(auto_refresh * 60)
    st.rerun()

# ========== AI SENTIMENT ==========
st.markdown("---")
st.subheader("🗞️ News & AI Sentiment")

for symbol in symbols:
    headlines = get_news_headlines(symbol)
    st.markdown(f"**{symbol}**")
    for h in headlines:
        st.markdown(f"- {h}")
        sentiment = get_sentiment(h)
        st.caption(f"> 🧠 *Sentiment:* {sentiment}")
    st.markdown("---")
