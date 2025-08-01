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

# --- Page Config ---
st.set_page_config(page_title="📈 Sipre Pro — Predictive Trading Signal Dashboard", layout="wide")
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Enter ticker symbol (e.g. AAPL, TSLA, BTC-USD):", value="AAPL").upper().strip()
    timeframe = st.selectbox("Select historical data period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    prophet_days = st.number_input("Prophet forecast days:", min_value=5, max_value=90, value=30, step=5)
    lstm_days = st.number_input("LSTM forecast days:", min_value=5, max_value=90, value=30, step=5)
    alert_email = st.text_input("Email for alerts (optional):")
    run_btn = st.button("Run Analysis")

# --- Helper Functions ---
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta).clip(lower=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(df['Close'], fast)
    ema_slow = calculate_ema(df['Close'], slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_bollinger(df, period=20):
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def prepare_lstm_data(df, seq_len=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def generate_signal(df):
    df['Signal'] = 0
    df.loc[(df['EMA9'] > df['EMA21']) & (df['RSI'] > 30), 'Signal'] = 1
    df.loc[(df['EMA9'] < df['EMA21']) & (df['RSI'] < 70), 'Signal'] = -1
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0).astype(int)
    return df

def explain_signal(latest, prev):
    ema9_curr, ema21_curr = latest['EMA9'], latest['EMA21']
    ema9_prev, ema21_prev = prev['EMA9'], prev['EMA21']
    rsi_curr = latest['RSI']
    signal = "Neutral"
    confidence = 0.0
    explanation = ""

    cross_up = (ema9_prev < ema21_prev) and (ema9_curr > ema21_curr) and (rsi_curr > 30)
    cross_down = (ema9_prev > ema21_prev) and (ema9_curr < ema21_curr) and (rsi_curr < 70)

    if cross_up:
        signal = "Buy ✅"
        explanation = "EMA9 crossed above EMA21 and RSI > 30 indicates bullish momentum."
        confidence = min(1.0, abs(ema9_curr - ema21_curr)) * ((rsi_curr - 30) / 40)
    elif cross_down:
        signal = "Sell ❌"
        explanation = "EMA9 crossed below EMA21 and RSI < 70 indicates bearish momentum."
        confidence = min(1.0, abs(ema9_curr - ema21_curr)) * ((70 - rsi_curr) / 40)
    else:
        explanation = "No clear crossover or RSI in neutral zone."

    confidence = round(confidence, 2)
    return signal, explanation, confidence

def send_email_alert(email, signal, symbol):
    # Placeholder - implement your email logic here
    st.info(f"Alert would be sent to {email} with signal: {signal} for {symbol}")

def ai_commentary(signal, explanation, confidence):
    comments = {
        "Buy ✅": [
            "The indicators suggest a strong buying opportunity. Consider entering a position soon.",
            "Bullish crossover with RSI confirmation; positive momentum expected.",
        ],
        "Sell ❌": [
            "Signals show potential for downward movement; consider protecting your position.",
            "Bearish crossover suggests caution and potential exit.",
        ],
        "Neutral": [
            "Market is indecisive; best to wait for clearer signals.",
            "No strong trend detected; stay alert for upcoming changes.",
        ],
    }
    import random
    comment = random.choice(comments.get(signal, ["No additional commentary available."]))
    return f"AI Commentary: {comment} (Confidence: {confidence * 100:.0f}%)"

# --- Main Logic ---
if run_btn:
    try:
        st.info(f"Downloading data for {symbol} ...")
        data = yf.download(symbol, period=timeframe, interval="1d", progress=False)
        if data.empty:
            st.error(f"No data found for symbol {symbol}.")
            st.stop()
        data.dropna(inplace=True)

        # Indicators
        data['EMA9'] = calculate_ema(data['Close'], 9)
        data['EMA21'] = calculate_ema(data['Close'], 21)
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = calculate_macd(data)
        data['BB_upper'], data['BB_lower'] = calculate_bollinger(data)
        data.dropna(inplace=True)

        # Signals
        data = generate_signal(data)
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        signal, explanation, confidence = explain_signal(latest, prev)

        # Display signal & explanation
        st.subheader(f"Trading Signal for {symbol}: {signal}")
        st.write(f"Explanation: {explanation}")
        st.write(f"Confidence: {confidence * 100:.0f}%")
        st.write(ai_commentary(signal, explanation, confidence))

        # Send alert email if requested
        if alert_email and signal != "Neutral":
            send_email_alert(alert_email, signal, symbol)

        # Price Chart with indicators
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'))

        fig.add_trace(go.Scatter(x=data.index, y=data['EMA9'], mode='lines', name='EMA9'))
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA21'], mode='lines', name='EMA21'))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], mode='lines', line=dict(dash='dot'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], mode='lines', line=dict(dash='dot'), name='BB Lower'))
        st.plotly_chart(fig, use_container_width=True)

        # RSI & MACD Chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
        fig2.add_hline(y=30, line_dash="dot", line_color='red')
        fig2.add_hline(y=70, line_dash="dot", line_color='green')
        fig2.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
        fig2.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], mode='lines', name='MACD Signal'))
        st.plotly_chart(fig2, use_container_width=True)

        # --- Prophet Forecast ---
        st.subheader(f"Prophet Forecast for {prophet_days} Days")
        df_prophet = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(subset=['y'], inplace=True)

        if len(df_prophet) < 30:
            st.warning("Not enough data for Prophet forecasting (need at least 30 days).")
        else:
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=prophet_days)
            forecast = model.predict(future)

            fig_prophet = go.Figure()
            fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
            fig_prophet.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 100, 80, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                hoverinfo="skip"
            ))
            fig_prophet.update_layout(title=f"Prophet Forecast for {symbol}", yaxis_title='Price', xaxis_title='Date')
            st.plotly_chart(fig_prophet, use_container_width=True)

            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

            csv = forecast.to_csv(index=False)
            st.download_button("Download Prophet Forecast CSV", csv, file_name=f"{symbol}_prophet_forecast.csv")

        # --- LSTM Forecast ---
        st.subheader(f"LSTM Forecast for {lstm_days} Days")
        try:
            seq_len = min(60, len(data) - 1)
            X, y, scaler = prepare_lstm_data(data, seq_len)
            model_lstm = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            model_lstm.fit(X, y, epochs=15, batch_size=32, verbose=0)

            # Predict future
            last_seq = X[-1]
            preds_scaled = []
            curr_seq = last_seq.reshape(1, seq_len, 1)
            for _ in range(lstm_days):
                pred = model_lstm.predict(curr_seq)[0][0]
                preds_scaled.append(pred)
                curr_seq = np.append(curr_seq[:,1:,:], [[[pred]]], axis=1)

            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
            future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=lstm_days)
            df_forecast = pd.DataFrame({"Date": future_dates, "LSTM Forecast": preds})

            # Plot
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close'))
            fig_lstm.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['LSTM Forecast'], mode='lines', name='LSTM Forecast'))
            fig_lstm.update_layout(title=f"LSTM Forecast for {symbol}", yaxis_title='Price', xaxis_title='Date')
            st.plotly_chart(fig_lstm, use_container_width=True)

            st.dataframe(df_forecast)
            csv2 = df_forecast.to_csv(index=False)
            st.download_button("Download LSTM Forecast CSV", csv2, file_name=f"{symbol}_lstm_forecast.csv")

        except Exception as e:
            st.error(f"LSTM forecasting error: {e}")
            st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.text(traceback.format_exc())
