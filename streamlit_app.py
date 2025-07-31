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

# --- 1. Setup and session state ---
st.set_page_config(page_title="📈 Sipre Pro — Predictive Trading Signal Dashboard", layout="wide")

# Initialize session state for signal history
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

# --- 1. Expanded symbol universe & multi-interval support ---
@st.cache_data(show_spinner=False)
def load_symbols(market):
    if market == "S&P 500":
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().str.upper().tolist()
    elif market == "TSX":
        # If no reliable online list, fallback symbols provided
        fallback = ["RY.TO", "TD.TO", "BNS.TO", "ENB.TO"]
        try:
            url = "https://raw.githubusercontent.com/datasets/tsx/master/data/tsx.csv"
            df = pd.read_csv(url)
            return df['Symbol'].dropna().str.upper().tolist()
        except:
            return fallback
    elif market == "Crypto":
        return ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOGE-USD"]
    else:
        return ["AAPL", "MSFT", "TSLA"]

# Sidebar inputs
with st.sidebar:
    st.header("Settings")

    market = st.selectbox("Select Market:", ["S&P 500", "TSX", "Crypto"])
    symbols_list = load_symbols(market)

    user_input = st.text_input("Enter symbol (e.g. AAPL, TSLA, BTC-USD):").upper().strip()
    filtered_symbols = [s for s in symbols_list if user_input in s] if user_input else symbols_list
    selected_symbol = st.selectbox("Or select from suggestions:", filtered_symbols) if filtered_symbols else None
    symbol = user_input if user_input else selected_symbol

    interval = st.selectbox("Select data interval:", ["1d", "1h", "15m"], index=0)

    timeframe = st.selectbox("Select timeframe:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

    alert_email = st.text_input("Enter your email for alerts (optional):")

    prophet_period = st.number_input("Days to predict (Prophet):", min_value=5, max_value=90, value=30, step=5)
    lstm_period = st.number_input("Days to predict (LSTM):", min_value=5, max_value=90, value=30, step=5)

    # Indicator toggles
    show_macd = st.checkbox("Show MACD Indicator", value=True)
    show_bbands = st.checkbox("Show Bollinger Bands", value=True)
    show_rsi = st.checkbox("Show RSI Indicator", value=True)
    show_ema = st.checkbox("Show EMA Indicators (9 & 21)", value=True)

    run_button = st.button("Run Prediction")

if not symbol:
    st.warning("Please enter or select a valid symbol.")
    st.stop()

# --- 2. Helper functions ---
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema(macd_line, 9)
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def prepare_lstm_data(df, sequence_length=60):
    n = df.shape[0]
    if n <= sequence_length:
        sequence_length = max(10, n - 1)
        if sequence_length < 10:
            raise ValueError("Still too little data for LSTM.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    X = np.array(X)
    y = np.array(y)
    if X.ndim != 3:
        raise ValueError(f"Unexpected LSTM input shape: {X.shape}")
    return X, y, scaler

def build_lstm_model(input_shape, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=50))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def monte_carlo_dropout_predict(model, input_data, n_sim=50):
    # Enable dropout at inference by using Keras backend function (MC Dropout)
    f = tf.function(lambda x: model(x, training=True))
    preds = [f(input_data).numpy() for _ in range(n_sim)]
    preds = np.array(preds).squeeze()
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred, std_pred

def backtest_signals(df):
    trades = []
    position = 0
    entry_price = 0
    returns = []

    for i, row in df.iterrows():
        if position == 0 and row['Position'] == 1:
            position = 1
            entry_price = row['Close']
            trades.append({'Entry Date': i, 'Entry Price': entry_price})
        elif position == 1 and row['Position'] == -1:
            exit_price = row['Close']
            ret = (exit_price - entry_price) / entry_price * 100
            trades[-1].update({'Exit Date': i, 'Exit Price': exit_price, 'Return %': ret})
            returns.append(ret)
            position = 0

    trades_df = pd.DataFrame(trades)
    total_return = np.sum(returns) if returns else 0
    win_rate = (np.array(returns) > 0).mean() * 100 if returns else 0
    num_trades = len(trades)
    return trades_df, total_return, win_rate, num_trades

def explain_signal(ema9_prev, ema21_prev, ema9_curr, ema21_curr, rsi_curr):
    # Confidence calculation example (simple)
    conf = 0
    if ema9_prev < ema21_prev and ema9_curr > ema21_curr:
        conf += 0.6
    if 30 < rsi_curr < 70:
        conf += 0.4
    conf_pct = int(conf * 100)
    if conf_pct > 100:
        conf_pct = 100
    if conf_pct < 0:
        conf_pct = 0
    return conf_pct

def fetch_news_sentiment(symbol):
    # Placeholder for sentiment integration
    return "Sentiment data currently unavailable."

def send_email_alert(recipient, signal, symbol):
    # Placeholder demo
    st.success(f"Demo: Would send alert '{signal}' for {symbol} to {recipient}")

# --- 3. Main app logic ---
if run_button:
    try:
        # Fetch data with retries and validation
        df = yf.download(symbol, period=timeframe, interval=interval, progress=False)
        if df.empty:
            st.error(f"No data found for {symbol} at interval {interval}.")
            st.stop()

        # Data validation
        if df['Close'].isnull().sum() / len(df) > 0.1:
            st.warning("Warning: More than 10% missing closing price data.")

        # Calculate indicators
        if show_ema:
            df['EMA9'] = calculate_ema(df['Close'], 9)
            df['EMA21'] = calculate_ema(df['Close'], 21)
        if show_rsi:
            df['RSI'] = calculate_rsi(df['Close'])
        if show_macd:
            df['MACD'], df['SignalLine'], df['MACDHist'] = calculate_macd(df['Close'])
        if show_bbands:
            df['BBUpper'], df['BBLower'] = calculate_bollinger_bands(df['Close'])

        df.dropna(inplace=True)
        if len(df) < 30:
            st.warning("Not enough data after indicator calculation.")

        # Signal generation: EMA crossover with RSI filter
        df['Position'] = 0
        if show_ema and show_rsi:
            df.loc[(df['EMA9'] > df['EMA21']) & (df['RSI'] > 30), 'Position'] = 1
            df.loc[(df['EMA9'] < df['EMA21']) & (df['RSI'] < 70), 'Position'] = -1

        # Backtesting
        trades_df, total_return, win_rate, num_trades = backtest_signals(df)

        st.subheader(f"📊 Backtesting Performance for {symbol}")
        st.write(f"Total Return: {total_return:.2f}%")
        st.write(f"Win Rate: {win_rate:.2f}%")
        st.write(f"Number of Trades: {num_trades}")
        if not trades_df.empty:
            st.dataframe(trades_df)

        # Latest signal & confidence
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        conf_pct = explain_signal(prev['EMA9'], prev['EMA21'], latest['EMA9'], latest['EMA21'], latest['RSI'])
        signal_str = "Neutral"
        if latest['Position'] == 1:
            signal_str = f"Buy ✅ (Confidence: {conf_pct}%)"
        elif latest['Position'] == -1:
            signal_str = f"Sell ❌ (Confidence: {conf_pct}%)"

        st.subheader(f"📌 Latest Signal: {signal_str}")

        # Add signal to history
        st.session_state.signal_history.append({
            "Date": df.index[-1],
            "Symbol": symbol,
            "Signal": signal_str,
            "Close": latest['Close']
        })
        st.markdown("### Signal History")
        history_df = pd.DataFrame(st.session_state.signal_history)
        st.dataframe(history_df)

        # Send alert if email provided and signal active
        if alert_email and latest['Position'] != 0:
            send_email_alert(alert_email, signal_str, symbol)

        # --- Prophet Forecast ---
        st.subheader(f"📅 Prophet Forecast (Next {prophet_period} Days)")

        df_reset = df.reset_index()
        prophet_df = pd.DataFrame({
            'ds': df_reset['Date'],
            'y': np.log1p(df_reset['Close'])
        }).dropna()

        if len(prophet_df) < 30:
            st.warning("Not enough data for Prophet forecasting.")
        else:
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=prophet_period)
            forecast = m.predict(future)

            forecast['yhat_exp'] = np.expm1(forecast['yhat'])
            forecast['yhat_lower_exp'] = np.expm1(forecast['yhat_lower'])
            forecast['yhat_upper_exp'] = np.expm1(forecast['yhat_upper'])

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_exp'], mode='lines', name='Forecast'))
            fig1.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper_exp'], forecast['yhat_lower_exp'][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Confidence Interval'
            ))
            fig1.update_layout(title=f"{symbol} Prophet Forecast (Next {prophet_period} Days)",
                               yaxis_title='Price (USD)', xaxis_title='Date')
            st.plotly_chart(fig1)
            st.download_button("📥 Download Prophet Forecast CSV", forecast.to_csv(index=False), file_name=f"{symbol}_prophet_forecast.csv")

        # --- LSTM Forecast with MC Dropout Uncertainty ---
        st.subheader(f"🤖 LSTM Forecast (Next {lstm_period} Days)")

        try:
            if len(df) < 50:
                raise ValueError("Not enough data for LSTM (need 50+ points).")

            seq_len = min(60, len(df)-1)
            X, y, scaler = prepare_lstm_data(df, sequence_length=seq_len)
            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=15, batch_size=32, verbose=0)

            future_input = X[-1].reshape(1, X.shape[1], X.shape[2])
            preds_mean = []
            preds_std = []

            for _ in range(lstm_period):
                mean_pred, std_pred = monte_carlo_dropout_predict(model, future_input, n_sim=30)
                preds_mean.append(mean_pred[0])
                preds_std.append(std_pred[0])

                next_pred = np.array([[[mean_pred[0]]]])
                future_input = np.concatenate((future_input[:, 1:, :], next_pred), axis=1)

            preds_mean = np.array(preds_mean).reshape(-1,1)
            preds_std = np.array(preds_std).reshape(-1,1)

            # Inverse transform predictions
            future_prices_mean = scaler.inverse_transform(preds_mean).flatten()
            future_prices_upper = scaler.inverse_transform(preds_mean + preds_std).flatten()
            future_prices_lower = scaler.inverse_transform(np.clip(preds_mean - preds_std, 0, None)).flatten()

            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=lstm_period, freq='D')

            df_future = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close (Mean)': future_prices_mean,
                'Predicted Close (Upper)': future_prices_upper,
                'Predicted Close (Lower)': future_prices_lower,
            })

            # Plot combined chart with indicators and forecasts
            fig2 = go.Figure()

            # Candlestick
            fig2.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Historical"
            ))

            # EMA lines
            if show_ema:
                fig2.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA9', line=dict(color='blue')))
                fig2.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21', line=dict(color='red')))

            # Bollinger Bands
            if show_bbands:
                fig2.add_trace(go.Scatter(x=df.index, y=df['BBUpper'], fill=None, mode='lines', line=dict(color='rgba(173,216,230,0.4)'), name='BB Upper'))
                fig2.add_trace(go.Scatter(x=df.index, y=df['BBLower'], fill='tonexty', mode='lines', line=dict(color='rgba(173,216,230,0.4)'), name='BB Lower'))

            # Buy/Sell markers
            buy_signals = df[(df['Position'] == 1)]
            sell_signals = df[(df['Position'] == -1)]
            fig2.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'))
            fig2.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'))

            # LSTM forecast with uncertainty
            fig2.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Predicted Close (Mean)'], mode='lines', name='LSTM Forecast', line=dict(color='orange', dash='dot')))
            fig2.add_trace(go.Scatter(x=pd.concat([df_future['Date'], df_future['Date'][::-1]]),
                                      y=pd.concat([df_future['Predicted Close (Upper)'], df_future['Predicted Close (Lower)'][::-1]]),
                                      fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,165,0,0)'),
                                      hoverinfo='skip', showlegend=True, name='LSTM Uncertainty'))

            fig2.update_layout(
                title=f"{symbol} Historical + Forecast",
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(df_future[['Date', 'Predicted Close (Mean)', 'Predicted Close (Upper)', 'Predicted Close (Lower)']], use_container_width=True)
            st.download_button("📥 Download LSTM Forecast CSV", df_future.to_csv(index=False), file_name=f"{symbol}_lstm_forecast.csv")

        except Exception as e:
            st.error(f"LSTM Prediction Error: {e}")
            st.text(traceback.format_exc())

        # --- News & Sentiment ---
        st.subheader("📰 News & Sentiment")
        sentiment = fetch_news_sentiment(symbol)
        st.markdown(sentiment)

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.text(traceback.format_exc())

# --- 4. Export signal history to CSV ---
st.sidebar.markdown("---")
if st.sidebar.button("Export Signal History CSV"):
    if st.session_state.signal_history:
        df_hist = pd.DataFrame(st.session_state.signal_history)
        csv = df_hist.to_csv(index=False)
        st.sidebar.download_button("Download Signal History CSV", csv, file_name="signal_history.csv")
    else:
        st.sidebar.warning("No signal history to export.")

# --- 5. Mobile Responsive tweaks ---
st.markdown(
    """
    <style>
    @media (max-width: 600px) {
        .css-1d391kg {padding-left: 1rem; padding-right: 1rem;}
    }
    </style>
    """,
    unsafe_allow_html=True
)
