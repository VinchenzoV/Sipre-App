import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import plotly.graph_objs as go
import traceback
import datetime

st.set_page_config(page_title="📈 Sipre Pro — Predictive Trading Signal Dashboard", layout="wide")
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

@st.cache_data
def load_symbols():
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().str.upper().tolist()
    except Exception:
        return ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "SPY", "BTC-USD", "ETH-USD"]

symbols_list = load_symbols()

with st.sidebar:
    st.header("Settings")
    user_input = st.text_input("Enter symbol (e.g. LNR.TO or AAPL):").upper().strip()
    filtered_symbols = [s for s in symbols_list if user_input in s] if user_input else symbols_list
    selected_symbol = st.selectbox("Or select from suggestions:", filtered_symbols) if filtered_symbols else None
    symbol = user_input if user_input else selected_symbol

    timeframe = st.selectbox("Select timeframe for historical data:", ["1mo", "3mo", "6mo", "1y"], index=2)
    alert_email = st.text_input("Enter your email for alerts (optional):")
    prophet_period = st.number_input("Number of days to predict (Prophet):", min_value=5, max_value=90, value=30, step=5)
    lstm_period = st.number_input("Number of days to predict (LSTM):", min_value=5, max_value=90, value=30, step=5)
    run_button = st.button("Run Prediction")

if not symbol:
    st.warning("Please enter or select a valid symbol on the sidebar.")
    st.stop()

# === Helper functions ===

def calculate_volatility(df, window=14):
    return df['Close'].rolling(window=window).std()

def detect_trend(df):
    if len(df) < 10:
        return "Unknown"
    slope = np.polyfit(range(len(df[-10:])), df['Close'][-10:], 1)[0]
    return "Uptrend" if slope > 0 else "Downtrend" if slope < 0 else "Sideways"

def detect_gaps(df):
    gaps = df['Open'][1:].values - df['Close'][:-1].values
    return np.count_nonzero(np.abs(gaps) > df['Close'].mean() * 0.02)

def annotate_signals(df, signal_log):
    annotations = []
    for log in signal_log:
        annotations.append(dict(
            x=log['date'],
            y=log.get('price', df['Close'].iloc[-1]),
            xref='x',
            yref='y',
            text=f"{log['signal']} ({log['confidence']*100:.0f}%)",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        ))
    return annotations

# (rest of script remains unchanged)
