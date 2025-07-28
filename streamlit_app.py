# sipre_v2_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 SIPRE v2 - Trading Signal Dashboard")

# --- Sidebar ---
st.sidebar.header("Configuration")
tickers_input = st.sidebar.text_input("Enter symbols (comma-separated)", "AAPL, MSFT, NVDA")
show_macd = st.sidebar.checkbox("Include MACD", True)
show_volume = st.sidebar.checkbox("Show Volume", True)
show_forecast = st.sidebar.checkbox("Show Forecast", True)
forecast_days = st.sidebar.slider("Forecast days ahead", 1, 30, 7)

@st.cache_data(show_spinner=False)
def load_data(symbol, period="6mo"):
    try:
        df = yf.download(symbol, period=period)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading {symbol}: {e}")
        return None

def calculate_indicators(df):
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    if show_macd:
        exp12 = df["Close"].ewm(span=12, adjust=False).mean()
        exp26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def generate_signal(df):
    if df.empty or len(df) < 21:
        return "Hold"
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    if latest["EMA9"] > latest["EMA21"] and previous["EMA9"] <= previous["EMA21"] and latest["RSI"] < 70:
        return "Buy"
    elif latest["EMA9"] < latest["EMA21"] and previous["EMA9"] >= previous["EMA21"] and latest["RSI"] > 30:
        return "Sell"
    else:
        return "Hold"

def forecast_price(df, days):
    df = df.copy()
    df = df.dropna()
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days
    model = LinearRegression()
    model.fit(df[["Days"]], df["Close"])
    last_day = df["Days"].iloc[-1]
    future_days = np.array(range(last_day + 1, last_day + days + 1)).reshape(-1, 1)
    forecasted_prices = model.predict(future_days)
    future_dates = [df["Date"].max() + timedelta(days=i) for i in range(1, days + 1)]
    return pd.DataFrame({"Date": future_dates, "Forecast": forecasted_prices})

# --- Main View ---
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

for ticker in tickers:
    st.subheader(f"📈 {ticker} Analysis")
    df = load_data(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        st.warning(f"No data for {ticker}")
        continue

    df = calculate_indicators(df)
    signal = generate_signal(df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
        st.metric("Signal", signal)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Close"], label="Close", color="black")
    ax.plot(df["Date"], df["EMA9"], label="EMA9", color="blue", linestyle="--")
    ax.plot(df["Date"], df["EMA21"], label="EMA21", color="red", linestyle="--")

    if show_forecast:
        forecast_df = forecast_price(df, forecast_days)
        ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="green", linestyle=":")

    ax.set_title(f"{ticker} Price Chart")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    if show_macd:
        fig_macd, ax_macd = plt.subplots(figsize=(10, 2))
        ax_macd.plot(df["Date"], df["MACD"], label="MACD", color="purple")
        ax_macd.plot(df["Date"], df["Signal"], label="Signal", color="orange")
        ax_macd.set_title("MACD Indicator")
        ax_macd.legend()
        st.pyplot(fig_macd)

    if show_volume:
        fig_vol, ax_vol = plt.subplots(figsize=(10, 2))
        ax_vol.bar(df["Date"], df["Volume"], color="gray")
        ax_vol.set_title("Volume")
        st.pyplot(fig_vol)

    st.markdown("---")

st.info("SIPRE v2: Forecasting + Multi-Symbol RSI/EMA/MACD-based signal system.")
