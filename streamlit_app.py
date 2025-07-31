import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import plotly.graph_objs as go
import traceback

st.set_page_config(page_title="📈 Sipre Pro — Smart Predictive Trading Dashboard", layout="wide")
st.title("📈 Sipre Pro — Smart Predictive Trading Signal Dashboard")

# --- Utils ---
@st.cache_data(show_spinner=False)
def load_symbols():
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().str.upper().tolist()
    except Exception:
        # fallback list
        return ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "SPY", "BTC-USD", "ETH-USD"]

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def prepare_lstm_data(df, sequence_length=60):
    # Use multiple features for richer input: Close, Volume, EMA9, EMA21, RSI, MACD Histogram
    features = ['Close', 'Volume', 'EMA9', 'EMA21', 'RSI', 'MACD_Hist']
    data = df[features].copy()
    data.fillna(method='ffill', inplace=True)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i, 0])  # Predict scaled 'Close'
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def fetch_news_sentiment(symbol):
    # For demo, keep placeholder - ideally integrate real sentiment API
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        requests.get(url, timeout=2)
        return "Sentiment: [Mock sentiment placeholder]"
    except:
        return "Sentiment unavailable"

def send_email_alert(recipient, signal, symbol):
    # Placeholder for real email alert integration
    try:
        st.success(f"Alert email would be sent to {recipient} for {signal} on {symbol} (demo mode).")
    except:
        st.error("Failed to send email alert.")

def backtest_signal(df):
    # Basic backtesting on EMA9/EMA21 crossover + RSI filter signals
    df = df.copy()
    df['Signal'] = 0
    df.loc[(df['EMA9'] > df['EMA21']) & (df['RSI'] > 30), 'Signal'] = 1
    df.loc[(df['EMA9'] < df['EMA21']) & (df['RSI'] < 70), 'Signal'] = -1
    
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift(1)
    
    cumulative_returns = (1 + df[['Returns', 'Strategy_Returns']].fillna(0)).cumprod() - 1
    return cumulative_returns

# --- Sidebar ---
with st.sidebar:
    st.header("Configure Prediction")
    symbols_list = load_symbols()
    user_input = st.text_input("Enter symbol (e.g. AAPL, TSLA, LNR.TO):", "AAPL").upper().strip()
    filtered_symbols = [s for s in symbols_list if user_input in s] if user_input else symbols_list
    selected_symbol = st.selectbox("Or select from suggestions:", filtered_symbols) if filtered_symbols else None
    symbol = user_input if user_input else selected_symbol

    timeframe = st.selectbox("Historical data timeframe:", ["1mo", "3mo", "6mo", "1y"], index=2)

    alert_email = st.text_input("Email for alerts (optional):")

    pred_horizon_num = st.number_input("Prediction horizon:", min_value=1, max_value=90, value=15, step=1)
    pred_horizon_unit = st.selectbox("Horizon unit:", ["Days", "Months"], index=0)
    pred_horizon_days = pred_horizon_num if pred_horizon_unit == "Days" else pred_horizon_num * 30

    st.markdown("---")
    st.markdown("**Note:** Predictions use LSTM with multiple indicators & Prophet with advanced settings.")

if not symbol:
    st.warning("Please enter or select a valid symbol in the sidebar.")
    st.stop()

if st.button("Run Predictions"):

    # --- Data Download ---
    try:
        df = yf.download(symbol, period=timeframe, interval="1d", progress=False)
        if df.empty or len(df) < 30:
            st.error("Not enough data to perform prediction.")
            st.stop()

        df.dropna(inplace=True)
        # Indicators
        df['EMA9'] = calculate_ema(df['Close'], 9)
        df['EMA21'] = calculate_ema(df['Close'], 21)
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD_Line'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])

        df.dropna(inplace=True)
        
        # Signal
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        signal = "Neutral"
        if (prev['EMA9'] < prev['EMA21']) and (latest['EMA9'] > latest['EMA21']) and (latest['RSI'] > 30):
            signal = "Buy ✅"
        elif (prev['EMA9'] > prev['EMA21']) and (latest['EMA9'] < latest['EMA21']) and (latest['RSI'] < 70):
            signal = "Sell ❌"

        # Show Signal & Basic Info
        st.markdown(f"### Trading Signal for {symbol}: {signal}")
        st.markdown(f"- **RSI:** {latest['RSI']:.2f}")
        st.markdown(f"- **EMA9:** {latest['EMA9']:.2f}")
        st.markdown(f"- **EMA21:** {latest['EMA21']:.2f}")
        st.markdown(f"- **MACD Histogram:** {latest['MACD_Hist']:.4f}")

        if alert_email and signal != "Neutral":
            send_email_alert(alert_email, signal, symbol)

        st.markdown("---")

        # --- Backtesting ---
        st.subheader("Backtesting Strategy Performance")
        cumulative_returns = backtest_signal(df)
        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['Returns'], mode='lines', name='Buy & Hold'))
        fig_backtest.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['Strategy_Returns'], mode='lines', name='EMA+RSI Strategy'))
        fig_backtest.update_layout(
            yaxis_title="Cumulative Returns",
            xaxis_title="Date",
            height=400,
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig_backtest, use_container_width=True)

        # --- Tabs for Prophet & LSTM ---
        tabs = st.tabs(["Prophet Forecast", "LSTM Forecast"])

        # --- Prophet Forecast ---
        with tabs[0]:
            st.subheader(f"Prophet Forecast (Next {pred_horizon_days} Days)")
            df_reset = df.reset_index()
            prices = df_reset['Close'].values.flatten()
            dates = pd.to_datetime(df_reset['Date'])

            # Prepare prophet dataframe with log transform
            prophet_df = pd.DataFrame({
                'ds': dates,
                'y': np.log(prices.clip(min=1))
            }).dropna()

            if len(prophet_df) < 30:
                st.warning("Not enough data for Prophet forecasting.")
            else:
                # Use holidays & uncertainty settings
                m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, interval_width=0.95)
                m.fit(prophet_df)

                future = m.make_future_dataframe(periods=pred_horizon_days)
                forecast = m.predict(future)

                forecast['yhat_exp'] = np.exp(forecast['yhat'])
                forecast['yhat_lower_exp'] = np.exp(forecast['yhat_lower'].clip(lower=np.log(1e-3)))
                forecast['yhat_upper_exp'] = np.exp(forecast['yhat_upper'])

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_exp'], mode='lines', name='Forecast'))
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                    y=pd.concat([forecast['yhat_upper_exp'], forecast['yhat_lower_exp'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='Confidence Interval'
                ))
                fig.update_layout(
                    title=f"{symbol} Prophet Forecast",
                    yaxis_title="Price (USD)",
                    xaxis_title="Date"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.download_button("📥 Download Prophet Forecast CSV", forecast.to_csv(index=False), file_name=f"{symbol}_prophet_forecast.csv")

        # --- LSTM Forecast ---
        with tabs[1]:
            st.subheader(f"LSTM Future Price Prediction (Next {pred_horizon_days} Days)")
            try:
                seq_len = min(60, len(df) - 1)
                X, y, scaler = prepare_lstm_data(df, sequence_length=seq_len)

                # Build LSTM Model with Dropout
                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                    Dropout(0.2),
                    LSTM(32),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X, y, epochs=15, batch_size=32, verbose=0)

                future_input = X[-1].reshape(1, X.shape[1], X.shape[2])
                future_preds = []
                for _ in range(pred_horizon_days):
                    pred = model.predict(future_input, verbose=0)[0][0]
                    future_preds.append(pred)
                    pred_array = np.array([[[pred]]], dtype=np.float32)
                    future_input = np.concatenate((future_input[:, 1:, :], pred_array), axis=1)

                future_prices = scaler.inverse_transform(np.hstack([np.array(future_preds).reshape(-1,1),
                    np.zeros((pred_horizon_days, X.shape[2]-1))]))[:, 0]

                last_close = df['Close'].iloc[-1]
                clipped_prices = np.clip(future_prices, last_close * 0.9, None)

                future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=pred_horizon_days, freq='D')
                df_future = pd.DataFrame({'Date': future_dates, 'Predicted Close': clipped_prices})

                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical"))
                fig_lstm.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Predicted Close'], mode='lines+markers', name="LSTM Forecast"))
                fig_lstm.update_layout(title=f"{symbol} LSTM Price Prediction", yaxis_title="Price (USD)", xaxis_title="Date")
                st.plotly_chart(fig_lstm, use_container_width=True)

                st.dataframe(df_future, use_container_width=True)
                st.download_button("📥 Download LSTM Forecast CSV", df_future.to_csv(index=False), file_name=f"{symbol}_lstm_forecast.csv")

            except Exception as e:
                st.error(f"LSTM Forecast Error: {e}")
                st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.text(traceback.format_exc())
