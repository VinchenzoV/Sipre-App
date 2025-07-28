import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import requests
from datetime import datetime

st.set_page_config(page_title="Sipre", layout="wide")
st.title("📈 Sipre — Trading Signal App (Live Yahoo Finance)")

# --- Popular symbols + custom input ---
popular_symbols = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BTC-USD", "ETH-USD", "SPY"]
symbol_choice = st.selectbox("Choose a popular symbol:", popular_symbols)
custom_symbol = st.text_input("Or enter a custom symbol:", value=symbol_choice)

# --- Timeframe selector ---
timeframe = st.selectbox("Select timeframe:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

# --- Auto-refresh toggle ---
auto_refresh = st.checkbox("🔄 Auto-refresh every 1 minute")

# --- Indicator calculation functions ---
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df):
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

# --- Alerts (Discord) ---
def send_discord_alert(message):
    webhook_url = "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
    try:
        requests.post(webhook_url, json={"content": message})
    except:
        st.warning("Failed to send Discord alert.")

# --- Cached data fetch ---
@st.cache_data(ttl=300)
def get_data(symbol, timeframe, interval):
    return yf.download(symbol, period=timeframe, interval=interval)

# --- Signal generation and plotting ---
if st.button("Get Signal") or auto_refresh:
    try:
        interval = "1h" if timeframe in ["1d", "5d", "1mo"] else "1d"
        df = get_data(custom_symbol, timeframe, interval)

        if df.empty or len(df) < 2:
            st.warning("⚠️ No data found for this symbol/timeframe. Try a different one.")
        else:
            df.dropna(inplace=True)
            df["EMA9"] = calculate_ema(df["Close"], 9)
            df["EMA21"] = calculate_ema(df["Close"], 21)
            df["RSI"] = calculate_rsi(df["Close"])
            df["MACD"], df["MACD_Signal"] = calculate_macd(df)
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

            # --- Log Signal ---
            signal_log = {
                "Symbol": custom_symbol.upper(),
                "Timeframe": timeframe,
                "Signal": signal,
                "RSI": round(rsi_latest, 2),
                "DateTime": latest.name.strftime("%Y-%m-%d %H:%M")
            }
            df_log = pd.DataFrame([signal_log])
            history_file = "signal_history.csv"
            df_log.to_csv(history_file, mode="a", header=not os.path.exists(history_file), index=False)

            # --- Send Alert ---
            if signal != "Neutral":
                send_discord_alert(f"{custom_symbol.upper()} Signal: {signal} | RSI: {rsi_latest:.2f} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

            # --- Display results ---
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

            # --- MACD Plot ---
            fig_macd, ax_macd = plt.subplots(figsize=(10, 3))
            ax_macd.plot(df.index, df["MACD"], label="MACD", color="purple")
            ax_macd.plot(df.index, df["MACD_Signal"], label="Signal", color="pink")
            ax_macd.axhline(0, linestyle='--', color='gray')
            ax_macd.set_title("MACD")
            ax_macd.legend()
            ax_macd.grid(True)
            st.pyplot(fig_macd)

            # --- Export history ---
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    st.download_button("📥 Download Signal History", f.read(), file_name="signals.csv")

    except Exception as e:
        st.error(f"Error loading data: {e}")

# --- Auto-refresh rerun logic ---
if auto_refresh:
    time.sleep(60)
    st.experimental_rerun()
