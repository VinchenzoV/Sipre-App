import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
import plotly.graph_objs as go
import datetime
import traceback

st.set_page_config(page_title="Sipre Pro", layout="wide")
st.title("📈 Sipre Pro - AI Stock Signal Dashboard")

# Sidebar
with st.sidebar:
    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL")
    start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())
    show_news = st.checkbox("📰 Show News Feed")
    show_backtest = st.checkbox("📊 Show Backtest")

# Load data
@st.cache_data

def load_data(ticker):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        return df
    except:
        st.error("Failed to load data.")
        return pd.DataFrame()

# Technical indicators
def add_indicators(df):
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Upper'] = df['EMA20'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower'] = df['EMA20'] - 2 * df['Close'].rolling(window=20).std()

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    return df

# Signal generation logic
def generate_signals(df):
    df['Signal'] = 0
    df['Signal'] = np.where((df['Close'] > df['EMA20']) & (df['MACD'] > df['Signal']), 1, df['Signal'])
    df['Signal'] = np.where((df['Close'] < df['EMA20']) & (df['MACD'] < df['Signal']), -1, df['Signal'])
    df['Position'] = df['Signal'].replace(0, method='ffill')
    return df

# Backtesting
def backtest_signals(df):
    trades = []
    position = 0
    buy_price = 0

    for i, row in df.iterrows():
        if position == 0 and row['Position'] == 1:
            position = 1
            buy_price = row['Close']
        elif position == 1 and row['Position'] == -1:
            position = 0
            sell_price = row['Close']
            trades.append((buy_price, sell_price))

    returns = [(sell - buy) / buy * 100 for buy, sell in trades]
    trades_df = pd.DataFrame(trades, columns=['Buy', 'Sell'])
    trades_df['Return %'] = returns

    total_return = sum(returns)
    win_rate = (trades_df['Return %'] > 0).mean() * 100 if not trades_df.empty else 0
    return trades_df, total_return, win_rate, len(trades)

# Prophet Forecast
def prophet_forecast(df):
    prophet_df = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    prophet_df['y'] = np.log1p(prophet_df['y'])
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast['yhat'] = np.expm1(forecast['yhat'])
    return forecast

# LSTM Forecast
def lstm_forecast(df):
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i - 60:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=3, batch_size=16, verbose=0)

    last_60_days = scaled[-60:]
    predictions = []
    input_seq = last_60_days
    for _ in range(30):
        pred = model.predict(input_seq.reshape(1, 60, 1), verbose=0)
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq[1:], pred, axis=0)

    future_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
    return pd.Series(future_prices, index=future_dates)

# Display logic
def plot_chart(df, forecast=None, lstm=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], line=dict(color='blue'), name='EMA20'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], line=dict(color='gray', dash='dot'), name='Upper BB'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], line=dict(color='gray', dash='dot'), name='Lower BB'))

    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', marker=dict(color='green', size=10), name='Buy'))
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', marker=dict(color='red', size=10), name='Sell'))

    if forecast is not None:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Forecast'))

    if lstm is not None:
        fig.add_trace(go.Scatter(x=lstm.index, y=lstm.values, mode='lines', name='LSTM Forecast', line=dict(dash='dot', color='orange')))

    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

# Main logic
df = load_data(ticker)
if not df.empty:
    df = add_indicators(df)
    df = generate_signals(df)
    st.subheader("📊 Stock Price & Signals")
    forecast = prophet_forecast(df)
    lstm_series = lstm_forecast(df)
    plot_chart(df, forecast, lstm_series)

    if show_backtest:
        st.subheader("📊 Backtesting Performance")
        trades_df, total_return, win_rate, num_trades = backtest_signals(df)
        st.metric("Total Return (%)", f"{total_return:.2f}%")
        st.metric("Win Rate", f"{win_rate:.2f}%")
        st.metric("Number of Trades", num_trades)
        st.dataframe(trades_df)

    if show_news:
        st.subheader("📰 Latest News")
        news_url = f"https://finance.yahoo.com/quote/{ticker}"
        st.markdown(f"[View {ticker} News on Yahoo Finance]({news_url})")
else:
    st.warning("No data loaded. Check the ticker symbol.")
