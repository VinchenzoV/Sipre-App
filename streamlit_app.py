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
import datetime

st.set_page_config(page_title="📈 Sipre Pro — Predictive Trading Signal Dashboard", layout="wide")
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

st.sidebar.header("Symbol & Settings")
symbol = st.sidebar.text_input("Enter stock/crypto symbol:", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
predict_days = st.sidebar.slider("Forecast Days (LSTM & Prophet)", 7, 60, 30)

@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.dropna(inplace=True)
    return data

def compute_indicators(df):
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def generate_signals(df):
    df['Position'] = 0
    df.loc[(df['EMA9'] > df['EMA21']) & (df['RSI'] > 30), 'Position'] = 1
    df.loc[(df['EMA9'] < df['EMA21']) & (df['RSI'] < 70), 'Position'] = -1
    return df

def simulate_trades(df):
    capital = 1000.0
    cash = capital
    shares = 0
    position = 0
    entry_price = 0
    trades = []

    for idx, row in df.iterrows():
        if position == 0 and row['Position'] == 1:
            entry_price = row['Close']
            shares = cash / entry_price
            cash = 0
            position = 1
            trades.append({'Entry Date': idx, 'Entry Price': entry_price})
        elif position == 1 and row['Position'] == -1:
            exit_price = row['Close']
            cash = shares * exit_price
            return_pct = (exit_price - entry_price) / entry_price * 100
            trades[-1].update({
                'Exit Date': idx,
                'Exit Price': exit_price,
                'Return %': return_pct,
                'Final Capital': cash
            })
            shares = 0
            position = 0

    if position == 1:
        exit_price = df['Close'].iloc[-1]
        cash = shares * exit_price
        return_pct = (exit_price - entry_price) / entry_price * 100
        trades[-1].update({
            'Exit Date': df.index[-1],
            'Exit Price': exit_price,
            'Return %': return_pct,
            'Final Capital': cash
        })

    return pd.DataFrame(trades), cash

def plot_backtest(df, trades):
    fig = go.Figure(data=[
        go.Candlestick(x=df.index,
                       open=df['Open'], high=df['High'],
                       low=df['Low'], close=df['Close'],
                       name='Candlesticks')
    ])

    for _, trade in trades.iterrows():
        fig.add_shape(type="line", x0=trade['Entry Date'], y0=trade['Entry Price'],
                      x1=trade['Exit Date'], y1=trade['Exit Price'],
                      line=dict(color="green" if trade['Return %'] > 0 else "red", width=2))

    fig.update_layout(title="Backtest Trades", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def lstm_forecast(df, predict_days):
    df_lstm = df[['Close']].copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_lstm)

    sequence_length = 60
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=10, batch_size=32, verbose=0)

    test_input = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    predictions = []
    for _ in range(predict_days):
        next_pred = model.predict(test_input)[0][0]
        predictions.append(next_pred)
        test_input = np.append(test_input[:, 1:, :], [[[next_pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=predict_days)
    df_future = pd.DataFrame({'Date': future_dates, 'LSTM Forecast': predictions.flatten()})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:],
                             mode='lines', name='Historical Close', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=df_future['Date'], y=df_future['LSTM Forecast'],
                             mode='lines+markers', name='LSTM Forecast', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=[df.index[-1], df_future['Date'].iloc[0]],
                             y=[df['Close'].iloc[-1], df_future['LSTM Forecast'].iloc[0]],
                             mode='lines', name='Forecast Bridge', line=dict(color='orange', dash='dash')))
    fig.update_layout(title="LSTM Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def prophet_forecast(df, predict_days):
    df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=predict_days)
    forecast = model.predict(future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

try:
    df = load_data(symbol, start_date, end_date)
    df = compute_indicators(df)
    df = generate_signals(df)
    trades_df, final_cash = simulate_trades(df)

    st.subheader("💸 Backtest Results")
    st.metric("Final Capital", f"${final_cash:,.2f}")
    st.dataframe(trades_df)
    plot_backtest(df, trades_df)

    st.subheader("🤖 LSTM Forecast")
    lstm_forecast(df, predict_days)

    st.subheader("📅 Prophet Forecast")
    prophet_forecast(df.reset_index(), predict_days)

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.text(traceback.format_exc())
