import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Sipre", layout="wide")
st.title("📈 Sipre — Trading Signal App (Live Yahoo Finance)")

# --- Popular symbols + custom input ---
popular_symbols = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BTC-USD", "ETH-USD", "SPY"]
symbol_choice = st.selectbox("Choose a popular symbol:", popular_symbols)
custom_symbol = st.text_input("Or enter a custom symbol:", value=symbol_choice)

# --- Timeframe selector ---
timeframe = st.selectbox("Select timeframe:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

# --- Functions ---
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Get Signal ---
if st.button("Get Signal"):
    try:
        interval = "1h" if timeframe in ["1d", "5d", "1mo"] else "1d"
        df = yf.download(custom_symbol, period=timeframe, interval=interval)

        if df.empty or len(df) < 2:
            st.warning("⚠️ No data found for this symbol/timeframe. Try a different one.")
        else:
            df.dropna(inplace=True)
            df["EMA9"] = calculate_ema(df["Close"], 9)
            df["EMA21"] = calculate_ema(df["Close"], 21)
            df["RSI"] = calculate_rsi(df["Close"])
            df.dropna(inplace=True)

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            ema9_latest = float(latest["EMA9"])
            ema21_latest = float(latest["EMA21"])
            ema9_prev = float(prev["EMA9"])
            ema21_prev = float(prev["EMA21"])
            rsi_latest = float(latest["RSI"])

            signal = "Neutral"
            if ema9_prev < ema21_prev and ema9_latest > ema21_latest and rsi_latest > 30:
                signal = "Buy ✅"
            elif ema9_prev > ema21_prev and ema9_latest < ema21_latest and rsi_latest < 70:
                signal = "Sell ❌"

            st.subheader(f"Signal: {signal}")
            rsi_color = "green" if rsi_latest < 30 else "red" if rsi_latest > 70 else "white"
            st.markdown(f"**RSI:** <span style='color:{rsi_color}'>{round(rsi_latest, 2)}</span>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, df["Close"], label="Price", color="blue")
            ax.plot(df.index, df["EMA9"], label="EMA9", color="orange")
            ax.plot(df.index, df["EMA21"], label="EMA21", color="red")
            ax.set_title(f"{custom_symbol.upper()} Price Chart ({timeframe})")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading data: {e}")
