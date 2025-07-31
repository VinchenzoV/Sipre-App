import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sipre", layout="wide")
st.title("📈 Sipre — Trading Signal App (Live Yahoo Finance)")

# --- Popular symbols + custom input ---
popular_symbols = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BTC-USD", "ETH-USD", "SPY"]
symbol_choice = st.selectbox("Choose a popular symbol:", popular_symbols)
custom_symbol = st.text_input("Or enter a custom symbol:", value=symbol_choice)

# --- Timeframe selector ---
timeframe = st.selectbox("Select timeframe:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

# --- EMA ---
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# --- RSI ---
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- ADX ---
def calculate_adx(df, period=14):
    df = df.copy()
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ])

    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    tr_smooth = df['TR'].rolling(window=period).mean()
    plus_dm_smooth = df['+DM'].rolling(window=period).mean()
    minus_dm_smooth = df['-DM'].rolling(window=period).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()

    return adx

# --- Main logic ---
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
            df["ADX"] = calculate_adx(df)

            df.dropna(inplace=True)

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            ema9_latest = float(latest["EMA9"])
            ema21_latest = float(latest["EMA21"])
            ema9_prev = float(prev["EMA9"])
            ema21_prev = float(prev["EMA21"])
            rsi_latest = float(latest["RSI"])
            adx_latest = float(latest["ADX"])

            signal = "Neutral"
            if ema9_prev < ema21_prev and ema9_latest > ema21_latest and rsi_latest > 30 and adx_latest > 20:
                signal = "Buy ✅"
            elif ema9_prev > ema21_prev and ema9_latest < ema21_latest and rsi_latest < 70 and adx_latest > 20:
                signal = "Sell ❌"

            st.subheader(f"Signal: {signal}")

            rsi_color = "green" if rsi_latest < 30 else "red" if rsi_latest > 70 else "white"
            st.markdown(f"**RSI:** <span style='color:{rsi_color}'>{round(rsi_latest, 2)}</span>", unsafe_allow_html=True)
            st.markdown(f"**ADX:** {round(adx_latest, 2)}")

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
