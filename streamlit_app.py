import streamlit as st
import requests
from datetime import datetime
import matplotlib.pyplot as plt

API_KEY = "d21r7ihr01qquiqnqu8gd21r7ihr01qquiqnqu90"
BASE_URL = "https://finnhub.io/api/v1"

def get_data(symbol):
    to_time = int(datetime.now().timestamp())
    from_time = to_time - 7 * 86400
    params = {
        "symbol": symbol,
        "resolution": "60",
        "from": from_time,
        "to": to_time,
        "token": API_KEY
    }
    res = requests.get(f"{BASE_URL}/stock/candle", params=params)
    res.raise_for_status()
    return res.json()

def calculate_ema(data, length):
    k = 2 / (length + 1)
    ema_array = []
    ema = data[0]
    for price in data:
        ema = price * k + ema * (1 - k)
        ema_array.append(round(ema, 2))
    return ema_array

def calculate_rsi(prices, period=14):
    gains = losses = 0
    for i in range(1, period + 1):
        delta = prices[i] - prices[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return round(100 - (100 / (1 + rs)), 2)

# === Streamlit App ===
st.set_page_config(page_title="Sipre", layout="wide")
st.title("📈 Sipre: Live Trading Signal App")

symbol = st.text_input("Enter stock/crypto/forex symbol (e.g., AAPL, TSLA, BTC/USD)", "AAPL")

if st.button("Get Signal"):
    try:
        data = get_data(symbol)
        prices = data["c"]
        timestamps = [datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") for t in data["t"]]
        ema9 = calculate_ema(prices, 9)
        ema21 = calculate_ema(prices, 21)
        rsi = calculate_rsi(prices)

        # Signal Logic
        last, prev = len(prices) - 1, len(prices) - 2
        signal = "Neutral"
        if ema9[prev] < ema21[prev] and ema9[last] > ema21[last] and rsi > 30:
            signal = "Buy ✅"
        elif ema9[prev] > ema21[prev] and ema9[last] < ema21[last] and rsi < 70:
            signal = "Sell ❌"

        st.subheader(f"Signal: {signal}")
        st.text(f"RSI: {rsi} {'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else ''}")

        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(timestamps, prices, label="Price", color="blue")
        ax.plot(timestamps, ema9, label="EMA9", color="orange")
        ax.plot(timestamps, ema21, label="EMA21", color="red")
        ax.set_title(f"{symbol} Price Chart with EMA & RSI")
        ax.set_xticks(timestamps[::len(timestamps)//6])
        ax.set_xticklabels(timestamps[::len(timestamps)//6], rotation=45)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to load data: {e}")
