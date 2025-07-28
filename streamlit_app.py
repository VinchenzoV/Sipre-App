import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sipre Pro — Advanced Trading Signals", layout="wide")

st.title("📊 Sipre Pro — Advanced Trading Signal Dashboard")

# --- Sidebar controls ---
st.sidebar.header("Settings")

# Popular symbols + custom input
popular_symbols = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BTC-USD", "ETH-USD", "SPY"]
symbol_choice = st.sidebar.selectbox("Choose a symbol", popular_symbols)
custom_symbol = st.sidebar.text_input("Or enter a custom symbol", value=symbol_choice)

# Timeframe and interval
timeframe = st.sidebar.selectbox("Select timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
interval_options = {"1mo": "1d", "3mo": "1d", "6mo": "1d", "1y": "1d", "2y": "1d", "5y": "1d"}
interval = interval_options[timeframe]

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh data every 5 minutes", value=False)

# --- Technical Indicator Functions ---

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = calculate_sma(prices, window)
    std = prices.rolling(window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return sma, upper_band, lower_band

def calculate_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    adx = dx.rolling(window=period, min_periods=period).mean()

    adx = adx.reindex(df.index).astype(float)
    adx.name = "ADX"
    return adx

def calculate_obv(df):
    obv = np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
                   np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    obv = pd.Series(obv, index=df.index).cumsum()
    obv.name = "OBV"
    return obv

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    atr.name = "ATR"
    return atr

# --- Composite Signal Logic ---

def generate_signal(df):
    signal = "Neutral"
    if len(df) < 2:
        return signal

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    ema_fast = latest['EMA12']
    ema_slow = latest['EMA26']
    prev_ema_fast = prev['EMA12']
    prev_ema_slow = prev['EMA26']

    rsi = latest['RSI']
    macd_hist = latest['MACD_hist']
    adx = latest['ADX']
    close = latest['Close']
    upper_band = latest['BB_upper']
    lower_band = latest['BB_lower']
    obv = latest['OBV']
    prev_obv = prev['OBV']

    if (prev_ema_fast < prev_ema_slow) and (ema_fast > ema_slow) and (rsi < 70) and (macd_hist > 0) and (adx > 25) and (close > upper_band) and (obv > prev_obv):
        signal = "Strong Buy 🔥"
    elif (ema_fast > ema_slow) and (rsi < 70) and (macd_hist > 0) and (adx > 20):
        signal = "Buy ✅"
    elif (adx < 20) or (40 < rsi < 60):
        signal = "Hold 🤝"
    elif (ema_fast < ema_slow) and (rsi > 70) and (macd_hist < 0) and (adx > 20):
        signal = "Sell ❌"
    elif (prev_ema_fast > prev_ema_slow) and (ema_fast < ema_slow) and (rsi > 30) and (macd_hist < 0) and (adx > 25) and (close < lower_band) and (obv < prev_obv):
        signal = "Strong Sell 🚨"
    else:
        signal = "Neutral"

    return signal

@st.cache_data(ttl=300)
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

def main():
    st.markdown("### Market Data & Analytics")

    symbol = custom_symbol.strip().upper()
    if not symbol:
        st.warning("Please enter a valid symbol.")
        return

    with st.spinner(f"Loading data for {symbol} ..."):
        df = load_data(symbol, timeframe, interval)
    if df.empty:
        st.error(f"No data found for {symbol}. Try another symbol or timeframe.")
        return

    df['EMA12'] = calculate_ema(df['Close'], 12)
    df['EMA26'] = calculate_ema(df['Close'], 26)
    df['RSI'] = calculate_rsi(df['Close'])
    macd_line, macd_signal, macd_hist = calculate_macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_mid'] = bb_mid
    df['BB_upper'] = bb_upper
    df['BB_lower'] = bb_lower
    df['ADX'] = calculate_adx(df)
    df['OBV'] = calculate_obv(df)
    df['ATR'] = calculate_atr(df)

    df.dropna(inplace=True)

    signal = generate_signal(df)

    st.subheader(f"Symbol: {symbol}")
    st.markdown(f"**Latest Signal:** {signal}")

    latest = df.iloc[-1]
    st.markdown(f"""
        - Close Price: ${latest['Close']:.2f}  
        - RSI: {latest['RSI']:.2f}  
        - MACD Histogram: {latest['MACD_hist']:.4f}  
        - ADX: {latest['ADX']:.2f}  
        - ATR (Volatility): {latest['ATR']:.2f}  
        - OBV (Volume): {int(latest['OBV'])}
    """)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax.plot(df.index, df['EMA12'], label='EMA 12', color='orange')
    ax.plot(df.index, df['EMA26'], label='EMA 26', color='red')
    ax.plot(df.index, df['BB_upper'], label='Bollinger Upper', linestyle='--', color='grey')
    ax.plot(df.index, df['BB_lower'], label='Bollinger Lower', linestyle='--', color='grey')
    ax.fill_between(df.index, df['BB_lower'], df['BB_upper'], color='grey', alpha=0.1)
    ax.set_title(f"{symbol} Price & Indicators")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(df.index, df['MACD'], label='MACD Line', color='purple')
    ax2.plot(df.index, df['MACD_signal'], label='Signal Line', color='green')
    ax2.bar(df.index, df['MACD_hist'], label='Histogram', color='grey')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 2))
    ax3.plot(df.index, df['RSI'], label='RSI', color='magenta')
    ax3.axhline(70, color='red', linestyle='--')
    ax3.axhline(30, color='green', linestyle='--')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    st.subheader("📜 Signal History")

    signals = []
    for i in range(len(df)):
        if i == 0:
            signals.append("Neutral")
        else:
            signals.append(generate_signal(df.iloc[:i+1]))

    signal_history = pd.DataFrame({
        'Date': df.index,
        'Close': df['Close'],
        'Signal': signals
    })

    st.dataframe(signal_history.tail(20))

    csv = signal_history.to_csv(index=False)
    st.download_button(label="Export Signal History CSV", data=csv, file_name=f"{symbol}_signal_history.csv", mime="text/csv")

    if auto_refresh:
        st.experimental_rerun()

main()
