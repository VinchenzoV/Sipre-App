import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Sipre - Trading Signal App", layout="wide")

st.title("📈 Sipre - Live Trading Signal Dashboard")
st.markdown("Enter a stock, crypto, or forex symbol below (e.g., AAPL, BTC-USD, EURUSD=X):")

symbol = st.text_input("Symbol", value="AAPL").upper()
interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=4)
period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=3)

@st.cache_data(show_spinner=False)
def fetch_data(symbol, interval, period):
    try:
        df = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
        if df.empty:
            return None
        df.dropna(inplace=True)
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df
    except Exception as e:
        return None

if symbol:
    df = fetch_data(symbol, interval, period)

    if df is None or df.empty:
        st.warning(f"⚠️ No data for: {symbol}")
    else:
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Safely extract float values
        ema9_prev = float(prev["EMA9"])
        ema21_prev = float(prev["EMA21"])
        ema9_latest = float(latest["EMA9"])
        ema21_latest = float(latest["EMA21"])
        rsi_latest = float(latest["RSI"])

        signal = "HOLD"
        suggestion = "No clear signal."

        if ema9_prev < ema21_prev and ema9_latest > ema21_latest and rsi_latest > 30:
            signal = "BUY"
            suggestion = "🔼 EMA crossover + RSI improving → Consider Buying"
        elif ema9_prev > ema21_prev and ema9_latest < ema21_latest and rsi_latest < 70:
            signal = "SELL"
            suggestion = "🔽 EMA crossover down + RSI weakening → Consider Selling"

        col1, col2 = st.columns(2)
        col1.metric("Latest Close", f"${latest['Close']:.2f}")
        col2.metric("Signal", signal)

        st.markdown(f"### 💡 Suggestion: {suggestion}")

        st.markdown("### 📊 Price Chart with EMA9 & EMA21")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode='lines', name='EMA9'))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], mode='lines', name='EMA21'))
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📉 RSI Indicator")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)

        with st.expander("📄 Show Raw Data"):
            st.dataframe(df.tail(50), use_container_width=True)
