import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go
import traceback

st.set_page_config(page_title="📈 Sipre Pro — Predictive Trading Signal Dashboard", layout="wide")

st.title("📈 Sipre Pro")
st.subheader("Predictive Trading Signal Dashboard using Prophet and LSTM")

@st.cache_data

def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        df.reset_index(inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def compute_indicators(df):
    df = df.copy()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()

    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def simulate_trades(df):
    capital = 1000
    position = 0
    buy_signals = []
    sell_signals = []
    capital_over_time = []

    for i in range(1, len(df)):
        if df['RSI'][i-1] < 30 and df['MACD'][i-1] > df['Signal'][i-1]:
            if capital > 0:
                position = capital / df['Close'][i]
                capital = 0
                buy_signals.append((df['Date'][i], df['Close'][i]))
        elif df['RSI'][i-1] > 70 and df['MACD'][i-1] < df['Signal'][i-1]:
            if position > 0:
                capital = position * df['Close'][i]
                position = 0
                sell_signals.append((df['Date'][i], df['Close'][i]))
        total_value = capital + (position * df['Close'][i])
        capital_over_time.append(total_value)

    return buy_signals, sell_signals, capital_over_time

def plot_backtest(df, buy_signals, sell_signals, capital_over_time):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Candlesticks'))

    for date, price in buy_signals:
        fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers', name='Buy', marker=dict(color='green', size=10)))

    for date, price in sell_signals:
        fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers', name='Sell', marker=dict(color='red', size=10)))

    st.plotly_chart(fig, use_container_width=True)
    st.line_chart(capital_over_time)

def forecast_with_lstm(df):
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    sequence_length = 60
    x_train = []
    y_train = []

    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i])
        y_train.append(scaled_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    test_input = scaled_data[-sequence_length:]
    test_input = np.reshape(test_input, (1, test_input.shape[0], 1))
    predicted = model.predict(test_input)
    predicted_price = scaler.inverse_transform(predicted)[0][0]

    forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=2, freq='D')[1:]
    df_future = pd.DataFrame({'Date': forecast_dates, 'LSTM Forecast': [predicted_price]})

    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Close', line=dict(color='blue', dash='dot')))
    fig_lstm.add_trace(go.Scatter(x=df_future['Date'], y=df_future['LSTM Forecast'], mode='lines+markers', name='LSTM Forecast', line=dict(color='orange')))
    fig_lstm.add_trace(go.Scatter(x=[df['Date'].iloc[-1], df_future['Date'].iloc[0]], y=[df['Close'].iloc[-1], df_future['LSTM Forecast'].iloc[0]], mode='lines', name='Forecast Bridge', line=dict(color='orange', dash='dash')))
    st.plotly_chart(fig_lstm, use_container_width=True)

def main():
    try:
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("today"))

        if symbol:
            df = load_data(symbol, start_date, end_date)
            if not df.empty:
                df = compute_indicators(df)
                buy_signals, sell_signals, capital_over_time = simulate_trades(df)
                st.subheader("Backtesting Results")
                plot_backtest(df, buy_signals, sell_signals, capital_over_time)
                st.subheader("LSTM Forecast")
                forecast_with_lstm(df)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
