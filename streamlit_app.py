import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Sipre - Trading Dashboard", layout="wide")

st.title("📈 Sipre - Trading Signal Dashboard")
st.markdown("Enter a **symbol** to view EMA & RSI-based signals, chart, and trading suggestion.")

# Input controls
symbol = st.text_input("Enter Symbol", value="AAPL")
interval = st.selectbox("Interval", ["1d", "1h", "15m"])
period = st.selectbox("Period", ["7d", "30d", "90d", "180d", "1y"])

@st.cache_data(show_spinner=False)
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        return None
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df.dropna()

df = load_data(symbol, period, interval)

if df is None or df.empty:
    st.error(f"No data found for symbol '{symbol}'.")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2]

# Signal logic
def get_signal(prev, latest):
    try:
        if prev["EMA9"] < prev["EMA21"] and latest["EMA9"] > latest["EMA21"] and latest["RSI"] > 30:
            return "Buy"
        elif prev["EMA9"] > prev["EMA21"] and latest["EMA9"] < latest["EMA21"] and latest["RSI"] < 70:
            return "Sell"
        else:
            return "Hold"
    except:
        return "Hold"

signal = get_signal(prev, latest)
color = {"Buy": "green", "Sell": "red", "Hold": "gray"}

# Chart
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode='lines', name="EMA9"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], mode='lines', name="EMA21"))
fig.update_layout(title=f"{symbol.upper()} Price Chart", xaxis_rangeslider_visible=False)

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Close", f"${latest['Close']:.2f}")
    st.metric("EMA9", f"{latest['EMA9']:.2f}")
    st.metric("EMA21", f"{latest['EMA21']:.2f}")
    st.metric("RSI", f"{latest['RSI']:.2f}")
    st.markdown(f"### 📌 Recommendation: **:{color[signal]}[{signal}]**")

st.caption("Built with Streamlit · Powered by Yahoo Finance · EMA + RSI Strategy")
