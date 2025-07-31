import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go
import traceback

st.set_page_config(page_title="Sipre Pro", layout="wide")
st.title("📈 Sipre Pro - Advanced Stock/Crypto Signal Dashboard")

@st.cache_data
def load_data(ticker):
    try:
        df = yf.download(ticker, period="2y")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error("Error loading data")
        st.exception(e)
        return None

def generate_signals(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Signal'] = 0
    df['Position'] = 0
    df['Confidence'] = 0

    for i in range(1, len(df)):
        if df['MA20'].iloc[i] > df['MA50'].iloc[i] and df['RSI'].iloc[i] < 70:
            df.at[i, 'Signal'] = 1
            df.at[i, 'Confidence'] = round(min(100, 100 - df['RSI'].iloc[i]), 2)
        elif df['MA20'].iloc[i] < df['MA50'].iloc[i] and df['RSI'].iloc[i] > 30:
            df.at[i, 'Signal'] = -1
            df.at[i, 'Confidence'] = round(min(100, df['RSI'].iloc[i]), 2)
        df.at[i, 'Position'] = df.at[i-1, 'Position'] if df.at[i, 'Signal'] == 0 else df.at[i, 'Signal']
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def explain_signal(signal):
    return "Buy Signal" if signal == 1 else "Sell Signal" if signal == -1 else "Hold"

def forecast_with_prophet(df):
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat']].tail(30)

def forecast_with_lstm(df):
    try:
        data = df[['Close']].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        x, y = [], []
        for i in range(60, len(scaled_data)):
            x.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        x = np.array(x)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x, np.array(y), epochs=5, batch_size=1, verbose=0)

        future_input = scaled_data[-60:].reshape(1, 60, 1)
        predictions = []
        for _ in range(30):
            pred = model.predict(future_input)[0, 0]
            predictions.append(pred)
            future_input = np.append(future_input[:, 1:, :], [[[pred]]], axis=1)

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)
        return pd.DataFrame({"Date": future_dates, "Predicted": predicted_prices.flatten()})
    except Exception as e:
        st.error("LSTM Error")
        st.exception(e)
        return pd.DataFrame()

def backtest_signals(df):
    trades = []
    position = 0
    entry_price = 0
    for i in range(1, len(df)):
        row = df.iloc[i]
        if position == 0 and row['Position'] == 1:
            position = 1
            entry_price = row['Close']
            entry_date = row['Date']
        elif position == 1 and row['Position'] == -1:
            exit_price = row['Close']
            exit_date = row['Date']
            return_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({'Entry': entry_date, 'Exit': exit_date, 'Return %': return_pct})
            position = 0

    trades_df = pd.DataFrame(trades)
    total_return = trades_df['Return %'].sum() if not trades_df.empty else 0
    win_rate = (trades_df['Return %'] > 0).mean() * 100 if not trades_df.empty else 0
    return trades_df, total_return, win_rate, len(trades_df)

# Main app
symbol = st.text_input("Enter Stock/Crypto Symbol (e.g. AAPL, BTC-USD)", "AAPL")
if symbol:
    df = load_data(symbol)
    if df is not None and not df.empty:
        df = generate_signals(df)

        st.subheader("📉 Price Chart with Signals")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='MA20'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], mode='lines', name='MA50'))
        buys = df[df['Signal'] == 1]
        sells = df[df['Signal'] == -1]
        fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Close'], mode='markers', name='Buy', marker=dict(color='green', size=8)))
        fig.add_trace(go.Scatter(x=sells['Date'], y=sells['Close'], mode='markers', name='Sell', marker=dict(color='red', size=8)))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Signal History")
        st.dataframe(df[['Date', 'Close', 'MA20', 'MA50', 'RSI', 'Signal', 'Confidence']].dropna().tail(10))

        st.subheader("🤖 Prophet Forecast (Next 30 Days)")
        prophet_forecast = forecast_with_prophet(df)
        st.line_chart(prophet_forecast.set_index('ds'))

        st.subheader("🔮 LSTM Forecast (Next 30 Days)")
        lstm_forecast = forecast_with_lstm(df)
        if not lstm_forecast.empty:
            st.line_chart(lstm_forecast.set_index('Date'))

        st.subheader("📊 Backtesting Performance")
        trades_df, total_return, win_rate, num_trades = backtest_signals(df)
        st.metric("Total Return (%)", f"{total_return:.2f}%")
        st.metric("Win Rate", f"{win_rate:.2f}%")
        st.metric("Number of Trades", num_trades)
        if not trades_df.empty:
            st.dataframe(trades_df)
    else:
        st.warning("No data loaded.")
