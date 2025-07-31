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
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(df, period=20, std_dev=2):
    sma = df['Close'].rolling(window=period).mean()
    rstd = df['Close'].rolling(window=period).std()
    upper_band = sma + std_dev * rstd
    lower_band = sma - std_dev * rstd
    return upper_band, lower_band

def prepare_lstm_data(df, sequence_length=60):
    n = df.shape[0]
    if n < 20:
        raise ValueError("Too little data for LSTM.")
    sequence_length = min(sequence_length, n - 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def fetch_news_sentiment(symbol):
    try:
        return "Sentiment: [Mock sentiment placeholder]"
    except:
        return "Sentiment unavailable"

def send_email_alert(recipient, signal, symbol):
    try:
        st.success(f"Alert email would be sent to {recipient} (demo only)")
    except:
        st.error("Failed to send email alert.")

def backtest_signals(df):
    """
    Backtest Buy/Sell signals based on EMA9/EMA21 crossovers and RSI.
    Returns DataFrame with trades, and performance stats.
    """
    df = df.copy()
    df['Position'] = 0
    # Buy signal
    buy_cond = (df['EMA9'].shift(1) < df['EMA21'].shift(1)) & (df['EMA9'] > df['EMA21']) & (df['RSI'] > 30)
    # Sell signal
    sell_cond = (df['EMA9'].shift(1) > df['EMA21'].shift(1)) & (df['EMA9'] < df['EMA21']) & (df['RSI'] < 70)

    df.loc[buy_cond, 'Position'] = 1
    df.loc[sell_cond, 'Position'] = -1

    trades = []
    position = 0
    entry_price = 0.0

    for idx, row in df.iterrows():
        if position == 0 and row['Position'] == 1:
            # Enter long
            position = 1
            entry_price = row['Close']
            trades.append({'Entry Date': idx, 'Entry Price': entry_price, 'Exit Date': None, 'Exit Price': None, 'Return %': None})
        elif position == 1 and row['Position'] == -1:
            # Exit long
            exit_price = row['Close']
            position = 0
            trades[-1]['Exit Date'] = idx
            trades[-1]['Exit Price'] = exit_price
            trades[-1]['Return %'] = (exit_price - entry_price) / entry_price * 100

    # Close open position on last day
    if position == 1:
        exit_price = df['Close'].iloc[-1]
        trades[-1]['Exit Date'] = df.index[-1]
        trades[-1]['Exit Price'] = exit_price
        trades[-1]['Return %'] = (exit_price - entry_price) / entry_price * 100

    trades_df = pd.DataFrame(trades)

    total_return = trades_df['Return %'].sum() if not trades_df.empty else 0
    win_rate = (trades_df['Return %'] > 0).mean() * 100 if not trades_df.empty else 0
    num_trades = len(trades_df)

    return trades_df, total_return, win_rate, num_trades

def explain_signal(latest, prev):
    """Explain the signal in simple terms with a confidence score."""
    ema9_latest = float(latest["EMA9"])
    ema21_latest = float(latest["EMA21"])
    ema9_prev = float(prev["EMA9"])
    ema21_prev = float(prev["EMA21"])
    rsi_latest = float(latest["RSI"])

    explanation = []
    confidence = 0

    if (ema9_prev < ema21_prev) and (ema9_latest > ema21_latest) and (rsi_latest > 30):
        explanation.append("EMA9 crossed above EMA21 and RSI > 30")
        confidence += (rsi_latest - 30) / 70  # scaled confidence based on RSI
        signal = "Buy ✅"
    elif (ema9_prev > ema21_prev) and (ema9_latest < ema21_latest) and (rsi_latest < 70):
        explanation.append("EMA9 crossed below EMA21 and RSI < 70")
        confidence += (70 - rsi_latest) / 70
        signal = "Sell ❌"
    else:
        signal = "Neutral"
        explanation.append("No clear crossover or RSI in neutral zone")

    confidence = round(min(max(confidence, 0), 1), 2)  # clamp 0-1
    return signal, "; ".join(explanation), confidence


if "signal_log" not in st.session_state:
    st.session_state.signal_log = []

if run_button:
    with st.spinner("Running predictions and analysis..."):
        try:
            # Download data
            for tf in [timeframe, "3mo", "6mo", "1y"]:
                df = yf.download(symbol, period=tf, interval="1d", progress=False)
                if df.shape[0] >= 50:  # we need at least 50 for LSTM
                    st.info(f"Using timeframe: {tf}")
                    break
            else:
                st.error("Not enough data for this symbol.")
                st.stop()

            df.dropna(inplace=True)
            if df.empty:
                st.error("No data found.")
                st.stop()

            # Indicators
            df['EMA9'] = calculate_ema(df['Close'], 9)
            df['EMA21'] = calculate_ema(df['Close'], 21)
            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df)
            df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df)

            df.dropna(inplace=True)
            latest, prev = df.iloc[-1], df.iloc[-2]

            # Signal & explainability
            signal, explanation, confidence = explain_signal(latest, prev)
            st.subheader(f"Signal: {signal} (Confidence: {confidence*100}%)")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown(f"**RSI:** {round(latest['RSI'], 2)}")

            # Save to signal log
            if signal != "Neutral":
                st.session_state.signal_log.append({
                    "symbol": symbol,
                    "signal": signal,
                    "rsi": round(latest['RSI'], 2),
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "confidence": confidence
                })

            if alert_email and signal != "Neutral":
                send_email_alert(alert_email, signal, symbol)

            if st.checkbox("Show Signal History"):
                st.dataframe(pd.DataFrame(st.session_state.signal_log))

            st.subheader("News Sentiment (Mocked)")
            st.markdown(fetch_news_sentiment(symbol))

            # Backtesting
            st.subheader("📊 Backtesting Performance")
            trades_df, total_return, win_rate, num_trades = backtest_signals(df)
            st.markdown(f"**Number of trades:** {num_trades}")
            st.markdown(f"**Total return:** {total_return:.2f}%")
            st.markdown(f"**Win rate:** {win_rate:.2f}%")
            if not trades_df.empty:
                st.dataframe(trades_df)

            # Prophet forecast
            st.subheader(f"Prophet Forecast (Next {int(prophet_period)} Days)")
            df_reset = df.reset_index()
            prices = df_reset['Close'].clip(lower=1.0).values.flatten()
            dates = pd.to_datetime(df_reset[df_reset.columns[0]]).values.flatten()

            prophet_df = pd.DataFrame({'ds': dates, 'y': np.log1p(prices)}).dropna()
            if len(prophet_df) < 30:
                st.warning("Not enough data for Prophet.")
            else:
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=int(prophet_period))
                forecast = m.predict(future)

                forecast['yhat_exp'] = np.expm1(forecast['yhat'])
                forecast['yhat_lower_exp'] = np.expm1(forecast['yhat_lower'].clip(lower=np.log1p(1e-3)))
                forecast['yhat_upper_exp'] = np.expm1(forecast['yhat_upper'])

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_exp'], mode='lines', name='Forecast'))
                fig1.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                    y=pd.concat([forecast['yhat_upper_exp'], forecast['yhat_lower_exp'][::-1]]),
                    fill='toself', fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", showlegend=True, name='Confidence Interval'))
                fig1.update_layout(title=f"{symbol} Prophet Forecast", yaxis_title='Price (USD)', xaxis_title='Date')
                st.plotly_chart(fig1)
                st.dataframe(forecast[['ds', 'yhat_exp', 'yhat_lower_exp', 'yhat_upper_exp']].tail(10), use_container_width=True)
                st.download_button("Download Prophet Forecast", forecast.to_csv(index=False), file_name=f"{symbol}_prophet.csv")

            # LSTM Forecast with Dropout & Candlestick chart + signal markers
            st.subheader(f"LSTM Forecast (Next {int(lstm_period)} Days)")
            try:
                seq_len = min(60, df.shape[0]-1)
                X, y, scaler = prepare_lstm_data(df, sequence_length=seq_len)

                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                    Dropout(0.2),
                    LSTM(50),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X, y, epochs=15, batch_size=32, verbose=0)

                future_input = X[-1].reshape(1, X.shape[1], X.shape[2])
                future_preds = []
                for _ in range(int(lstm_period)):
                    pred_scaled = model.predict(future_input, verbose=0)[0][0]
                    future_preds.append(pred_scaled)
                    pred_array = np.array([[[pred_scaled]]])
                    future_input = np.concatenate((future_input[:, 1:, :], pred_array), axis=1)

                future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
                last_close = float(df['Close'].iloc[-1])
                clipped_prices = np.clip(future_prices, last_close * 0.9, None)

                future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=int(lstm_period), freq='D')
                df_future = pd.DataFrame({'Date': future_dates, 'Predicted Close': clipped_prices})

                # Candlestick chart with indicators and signals
                fig2 = go.Figure()

                fig2.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                    name='Historical'
                ))

                # Bollinger bands
                fig2.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='rgba(255,0,0,0.3)'), name='BB Upper'))
                fig2.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='rgba(0,0,255,0.3)'), name='BB Lower'))

                # MACD histogram as bar chart (secondary y-axis)
                fig2.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram', marker_color='grey', yaxis='y2'))

                # Buy/Sell signal markers on price chart
                buy_signals = (df['EMA9'].shift(1) < df['EMA21'].shift(1)) & (df['EMA9'] > df['EMA21']) & (df['RSI'] > 30)
                sell_signals = (df['EMA9'].shift(1) > df['EMA21'].shift(1)) & (df['EMA9'] < df['EMA21']) & (df['RSI'] < 70)

                fig2.add_trace(go.Scatter(
                    x=df.index[buy_signals],
                    y=df['Close'][buy_signals],
                    mode='markers', marker=dict(symbol='triangle-up', color='green', size=12),
                    name='Buy Signal'
                ))

                fig2.add_trace(go.Scatter(
                    x=df.index[sell_signals],
                    y=df['Close'][sell_signals],
                    mode='markers', marker=dict(symbol='triangle-down', color='red', size=12),
                    name='Sell Signal'
                ))

                # LSTM forecast line
                fig2.add_trace(go.Scatter(
                    x=df_future['Date'],
                    y=df_future['Predicted Close'],
                    mode='lines',
                    name='LSTM Forecast',
                    line=dict(color='orange', dash='dot')
                ))

                # Layout with secondary y-axis for MACD histogram
                fig2.update_layout(
                    title=f"{symbol} Price Chart with Indicators and LSTM Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    yaxis2=dict(
                        title='MACD Histogram',
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='grey'
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig2)

                st.dataframe(df_future, use_container_width=True)
                st.download_button("Download LSTM Forecast", df_future.to_csv(index=False), file_name=f"{symbol}_lstm.csv")

            except Exception as e:
                st.error(f"LSTM Error: {e}")
                st.text(traceback.format_exc())

        except Exception as e:
            st.error(f"Unexpected Error: {e}")
            st.text(traceback.format_exc())
