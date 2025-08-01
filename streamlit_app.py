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

st.set_page_config(page_title="📈 AI Trading Dashboard", layout="wide")
st.title("📈 AI Trading Dashboard with Forecasting and Signals")

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Settings")
    symbol_input = st.text_input("Enter symbol (e.g. AAPL, MSFT, TSLA):", value="AAPL").upper().strip()
    timeframe = st.selectbox("Select data period:", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    prophet_days = st.slider("Prophet forecast days", 5, 90, 30)
    lstm_days = st.slider("LSTM forecast days", 5, 90, 30)
    alert_email = st.text_input("Email for alerts (demo, optional):")
    run_button = st.button("Run Analysis")

# --- Helper functions ---
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(df['Close'], fast)
    ema_slow = calculate_ema(df['Close'], slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_bollinger_bands(df, period=20, std_dev=2):
    sma = df['Close'].rolling(window=period).mean()
    rstd = df['Close'].rolling(window=period).std()
    upper = sma + std_dev * rstd
    lower = sma - std_dev * rstd
    return upper, lower

def generate_signals(df):
    df = df.copy()
    df['Signal'] = 0
    # Buy signal: EMA9 crosses above EMA21 and RSI > 30
    cross_up = (df['EMA9'].shift(1) < df['EMA21'].shift(1)) & (df['EMA9'] > df['EMA21']) & (df['RSI'] > 30)
    df.loc[cross_up, 'Signal'] = 1
    # Sell signal: EMA9 crosses below EMA21 and RSI < 70
    cross_down = (df['EMA9'].shift(1) > df['EMA21'].shift(1)) & (df['EMA9'] < df['EMA21']) & (df['RSI'] < 70)
    df.loc[cross_down, 'Signal'] = -1
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0).astype(int)
    return df

def backtest(df, initial_cash=1000):
    position = 0
    cash = initial_cash
    shares = 0
    portfolio_vals = []
    trades = []
    for i, row in df.iterrows():
        price = row['Close']
        signal = row['Signal']
        date = i
        if position == 0 and signal == 1:
            shares = cash // price
            if shares > 0:
                cash -= shares * price
                position = 1
                trades.append({'Entry Date': date, 'Entry Price': price, 'Exit Date': None, 'Exit Price': None, 'Return %': None})
        elif position == 1 and signal == -1 and shares > 0:
            cash += shares * price
            position = 0
            trades[-1]['Exit Date'] = date
            trades[-1]['Exit Price'] = price
            trades[-1]['Return %'] = (price - trades[-1]['Entry Price']) / trades[-1]['Entry Price'] * 100
            shares = 0
        portfolio_val = cash + shares * price
        portfolio_vals.append({'Date': date, 'Portfolio Value': portfolio_val})

    # Close open position at last date
    if position == 1 and shares > 0:
        price = df['Close'][-1]
        cash += shares * price
        trades[-1]['Exit Date'] = df.index[-1]
        trades[-1]['Exit Price'] = price
        trades[-1]['Return %'] = (price - trades[-1]['Entry Price']) / trades[-1]['Entry Price'] * 100
        shares = 0
        portfolio_vals.append({'Date': df.index[-1], 'Portfolio Value': cash})

    trades_df = pd.DataFrame(trades)
    portfolio_df = pd.DataFrame(portfolio_vals).set_index('Date')
    total_return = (portfolio_df['Portfolio Value'][-1] - initial_cash) / initial_cash * 100 if not portfolio_df.empty else 0
    win_rate = trades_df['Return %'].gt(0).mean() * 100 if not trades_df.empty else 0
    return trades_df, portfolio_df, total_return, win_rate

def prepare_lstm_data(df, seq_len=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

def ai_commentary(signal, confidence):
    if signal == 1:
        return f"Buy signal detected with confidence {confidence:.1%}. Market conditions look favorable."
    elif signal == -1:
        return f"Sell signal detected with confidence {confidence:.1%}. Caution advised."
    return "No clear buy or sell signal detected."

# --- Main ---
if run_button:
    try:
        # Download data
        df = yf.download(symbol_input, period=timeframe, interval="1d", progress=False)
        if df.empty:
            st.error("No data found for this symbol.")
            st.stop()

        df.dropna(inplace=True)

        # Indicators
        df['EMA9'] = calculate_ema(df['Close'], 9)
        df['EMA21'] = calculate_ema(df['Close'], 21)
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df)
        df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df)

        df.dropna(inplace=True)

        # Signals
        df = generate_signals(df)

        # Latest signal and confidence (simple abs diff * RSI influence)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        signal = latest['Signal']
        ema_diff = abs(latest['EMA9'] - latest['EMA21'])
        rsi_score = (latest['RSI'] - 30) / 40 if signal == 1 else (70 - latest['RSI']) / 40 if signal == -1 else 0
        confidence = min(max(ema_diff * rsi_score, 0), 1)

        st.subheader(f"Latest Signal: {'Buy ✅' if signal==1 else 'Sell ❌' if signal==-1 else 'Hold ⚪️'} (Confidence: {confidence:.2%})")
        st.markdown(ai_commentary(signal, confidence))

        # Backtest
        st.subheader("Backtest Results")
        trades_df, portfolio_df, total_return, win_rate = backtest(df)
        st.markdown(f"Total Return: **{total_return:.2f}%**")
        st.markdown(f"Win Rate: **{win_rate:.2f}%**")
        if not trades_df.empty:
            st.dataframe(trades_df)

        # Backtest chart
        fig_bt = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Price"
        )])
        if not trades_df.empty:
            fig_bt.add_trace(go.Scatter(
                x=trades_df['Entry Date'], y=trades_df['Entry Price'],
                mode='markers', marker=dict(symbol='triangle-up', color='green', size=10),
                name='Buy'
            ))
            fig_bt.add_trace(go.Scatter(
                x=trades_df['Exit Date'], y=trades_df['Exit Price'],
                mode='markers', marker=dict(symbol='triangle-down', color='red', size=10),
                name='Sell'
            ))
        fig_bt.update_layout(title=f"{symbol_input} Backtest", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_bt, use_container_width=True)

        # Prophet forecast
        st.subheader(f"Prophet Forecast (Next {prophet_days} Days)")
        df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet = df_prophet.dropna(subset=['y'])
        if len(df_prophet) < 30:
            st.warning("Not enough data for Prophet forecast.")
        else:
            model = Prophet()
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=prophet_days)
            forecast = model.predict(future)

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
            fig_p.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself', fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True, name='Confidence Interval'))
            fig_p.update_layout(title=f"{symbol_input} Prophet Forecast", yaxis_title="Price", xaxis_title="Date")
            st.plotly_chart(fig_p, use_container_width=True)
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prophet_days))

        # LSTM forecast
        st.subheader(f"LSTM Forecast (Next {lstm_days} Days)")
        try:
            seq_len = min(60, len(df) - 1)
            X, y, scaler = prepare_lstm_data(df, seq_len)
            model_lstm = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            model_lstm.fit(X, y, epochs=15, batch_size=32, verbose=0)

            input_seq = X[-1].reshape(1, seq_len, 1)
            preds_scaled = []
            for _ in range(lstm_days):
                pred = model_lstm.predict(input_seq)[0][0]
                preds_scaled.append(pred)
                input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
            future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=lstm_days)
            df_lstm_forecast = pd.DataFrame({'Date': future_dates, 'LSTM Forecast': preds})

            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Close'))
            fig_lstm.add_trace(go.Scatter(x=df_lstm_forecast['Date'], y=df_lstm_forecast['LSTM Forecast'], mode='lines', name='LSTM Forecast'))
            fig_lstm.update_layout(title=f"{symbol_input} LSTM Forecast", yaxis_title='Price', xaxis_title='Date')
            st.plotly_chart(fig_lstm, use_container_width=True)
            st.dataframe(df_lstm_forecast)
        except Exception as e:
            st.error(f"LSTM forecasting failed: {e}")
            st.text(traceback.format_exc())

        # Technical indicators chart
        st.subheader("Technical Indicators")
        fig_ind = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        )])
        fig_ind.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA9'))
        fig_ind.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21'))
        fig_ind.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', line=dict(dash='dot'), name='BB Upper'))
        fig_ind.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', line=dict(dash='dot'), name='BB Lower'))
        st.plotly_chart(fig_ind, use_container_width=True)

        st.subheader("RSI and MACD")
        fig_rsi_macd = go.Figure()
        fig_rsi_macd.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig_rsi_macd.add_hline(y=30, line_dash="dot", line_color='red')
        fig_rsi_macd.add_hline(y=70, line_dash="dot", line_color='green')
        fig_rsi_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
        fig_rsi_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='MACD Signal'))
        st.plotly_chart(fig_rsi_macd, use_container_width=True)

        # Mock alert (demo only)
        if alert_email and signal != 0:
            st.info(f"Demo alert sent to {alert_email} for signal: {'Buy' if signal==1 else 'Sell'}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.text(traceback.format_exc())
else:
    st.info("Enter symbol and press Run Analysis to start.")

