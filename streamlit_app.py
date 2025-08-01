import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go
import traceback
import requests
import datetime

st.set_page_config(page_title="📈 Sipre Pro — Predictive Trading Signal Dashboard", layout="wide")
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("🔎 Select Stock or Crypto")
all_symbols = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN', 'BTC-USD', 'ETH-USD', 'NVDA', 'META']
symbol = st.sidebar.text_input("Enter symbol (e.g., AAPL or BTC-USD):", value="AAPL")
symbol = symbol.upper().strip()
timeframe = st.sidebar.selectbox("Select timeframe", ['1y', '6mo', '3mo', '1mo'])

st.sidebar.markdown("---")
st.sidebar.header("📬 Alerts")
st.sidebar.text_input("Email (for future alerts)", "")

st.sidebar.markdown("---")
st.sidebar.header("📊 Forecast Settings")
predict_days = st.sidebar.slider("Days to Predict (Prophet & LSTM)", 5, 60, 20)

run_button = st.sidebar.button("Run Analysis")

# ------------------ LOAD DATA ------------------
@st.cache_data

def load_data(symbol, period):
    interval = '1d'
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

# ------------------ TECHNICAL INDICATORS ------------------
def add_indicators(df):
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    df['Upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['Lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# ------------------ SIGNAL GENERATION ------------------
def generate_signals(df):
    df['Signal'] = 0
    df.loc[(df['EMA9'] > df['EMA21']) & (df['RSI'] > 30), 'Signal'] = 1
    df.loc[(df['EMA9'] < df['EMA21']) & (df['RSI'] < 70), 'Signal'] = -1
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
    return df

# ------------------ BACKTESTING ------------------
def backtest_signals(df, initial_cash=1000):
    cash = initial_cash
    shares = 0
    trade_log = []

    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1 and cash > 0:
            shares = cash // df['Close'].iloc[i]
            cash -= shares * df['Close'].iloc[i]
            trade_log.append((df['Date'].iloc[i], 'BUY', df['Close'].iloc[i]))
        elif df['Signal'].iloc[i] == -1 and shares > 0:
            cash += shares * df['Close'].iloc[i]
            trade_log.append((df['Date'].iloc[i], 'SELL', df['Close'].iloc[i]))
            shares = 0

    portfolio_value = cash + shares * df['Close'].iloc[-1]
    return_pct = (portfolio_value - initial_cash) / initial_cash * 100

    return portfolio_value, return_pct, trade_log

# ------------------ FORECASTING WITH PROPHET ------------------
def run_prophet(df, predict_days):
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet['y'] = np.log1p(df_prophet['y'])
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=predict_days)
    forecast = model.predict(future)
    forecast['yhat'] = np.expm1(forecast['yhat'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("📥 Download Prophet Forecast", forecast.to_csv(index=False), file_name="prophet_forecast.csv")

# ------------------ FORECASTING WITH LSTM ------------------
def run_lstm(df, predict_days):
    try:
        df_model = df[['Close']].copy()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_model)

        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i, 0])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        last_sequence = scaled[-60:]
        predictions = []
        input_seq = last_sequence.reshape(1, 60, 1)

        for _ in range(predict_days):
            next_pred = model.predict(input_seq, verbose=0)
            predictions.append(next_pred[0, 0])
            input_seq = np.append(input_seq[:, 1:, :], [[next_pred]], axis=1)

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        last_date = df['Date'].iloc[-1]
        future_dates = pd.bdate_range(last_date, periods=predict_days + 1, freq='B')[1:]

        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': predicted_prices
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Actual'))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name='LSTM Forecast', line=dict(dash='dot', color='orange')))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("📥 Download LSTM Forecast", forecast_df.to_csv(index=False), file_name="lstm_forecast.csv")

    except Exception as e:
        st.error("❌ LSTM forecasting failed")
        st.text(traceback.format_exc())

# ------------------ RUN MAIN ------------------
if run_button:
    try:
        df = load_data(symbol, timeframe)
        required_cols = ['Open', 'High', 'Low', 'Close']
        df = df.dropna(subset=required_cols)
        df = add_indicators(df)
        df = generate_signals(df)

        st.subheader(f"📊 {symbol} Price Chart with Signals")
        fig = go.Figure(data=[
            go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'),
            go.Scatter(x=df['Date'], y=df['EMA9'], line=dict(color='blue'), name='EMA9'),
            go.Scatter(x=df['Date'], y=df['EMA21'], line=dict(color='red'), name='EMA21'),
            go.Scatter(x=df['Date'], y=df['Upper'], line=dict(color='green', dash='dot'), name='Upper Band'),
            go.Scatter(x=df['Date'], y=df['Lower'], line=dict(color='green', dash='dot'), name='Lower Band')
        ])
        st.plotly_chart(fig, use_container_width=True)

        portfolio_value, return_pct, trade_log = backtest_signals(df)
        st.markdown(f"💰 Final Portfolio Value: **${portfolio_value:.2f}**")
        st.markdown(f"📈 Return: **{return_pct:.2f}%**")

        st.subheader("📬 Trade Log")
        for trade in trade_log:
            st.write(trade)

        st.subheader("🔮 Prophet Forecast")
        run_prophet(df, predict_days)

        st.subheader("🤖 LSTM Forecast")
        run_lstm(df, predict_days)

    except Exception as e:
        st.error("❌ An error occurred while processing.")
        st.text(traceback.format_exc())
