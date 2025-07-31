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
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

@st.cache_data
def load_symbols():
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().str.upper().tolist()
    except Exception:
        # Fallback symbols list
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
    # Placeholder for real sentiment analysis integration
    return "Sentiment: [Mock sentiment placeholder]"

def send_email_alert(recipient, signal, symbol):
    # Placeholder for email sending implementation
    try:
        st.success(f"Alert email would be sent to {recipient} (demo only)")
    except:
        st.error("Failed to send email alert.")

def generate_signals(df):
    df = df.copy()
    df['Signal'] = 0

    # Buy signal: EMA9 crosses above EMA21 and RSI > 30
    df.loc[(df['EMA9'] > df['EMA21']) & (df['RSI'] > 30), 'Signal'] = 1
    # Sell signal: EMA9 crosses below EMA21 and RSI < 70
    df.loc[(df['EMA9'] < df['EMA21']) & (df['RSI'] < 70), 'Signal'] = -1

    # Forward fill Position based on last signal (to hold position)
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0).astype(int)

    return df

def backtest_signals(df):
    df = df.copy()
    df['Trade_Signal'] = 0

    # Buy when EMA9 crosses above EMA21 and RSI > 30
    buy_cond = (df['EMA9'].shift(1) < df['EMA21'].shift(1)) & (df['EMA9'] > df['EMA21']) & (df['RSI'] > 30)
    # Sell when EMA9 crosses below EMA21 and RSI < 70
    sell_cond = (df['EMA9'].shift(1) > df['EMA21'].shift(1)) & (df['EMA9'] < df['EMA21']) & (df['RSI'] < 70)

    df.loc[buy_cond, 'Trade_Signal'] = 1
    df.loc[sell_cond, 'Trade_Signal'] = -1

    trades = []
    position = 0
    entry_price = 0.0
    entry_date = None

    for idx, row in df.iterrows():
        signal = row['Trade_Signal']
        price = row['Close']
        date = idx

        if position == 0 and signal == 1:
            # Enter long position
            position = 1
            entry_price = price
            entry_date = date
        elif position == 1 and signal == -1:
            # Exit position
            exit_price = price
            exit_date = date
            ret = (exit_price - entry_price) / entry_price * 100
            trades.append({
                'Entry Date': entry_date,
                'Entry Price': entry_price,
                'Exit Date': exit_date,
                'Exit Price': exit_price,
                'Return %': ret
            })
            position = 0
            entry_price = 0.0
            entry_date = None

    # Close position at end of data if still open
    if position == 1:
        exit_price = df['Close'].iloc[-1]
        exit_date = df.index[-1]
        ret = (exit_price - entry_price) / entry_price * 100
        trades.append({
            'Entry Date': entry_date,
            'Entry Price': entry_price,
            'Exit Date': exit_date,
            'Exit Price': exit_price,
            'Return %': ret
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        total_return = 0.0
        win_rate = 0.0
        num_trades = 0
    else:
        total_return = trades_df['Return %'].sum()
        win_rate = (trades_df['Return %'] > 0).mean() * 100
        num_trades = len(trades_df)

    return trades_df, total_return, win_rate, num_trades

def explain_signal(latest, prev):
    ema9_latest = float(latest["EMA9"])
    ema21_latest = float(latest["EMA21"])
    ema9_prev = float(prev["EMA9"])
    ema21_prev = float(prev["EMA21"])
    rsi_latest = float(latest["RSI"])

    explanation = []
    confidence = 0
    signal = "Neutral"

    ema_diff_prev = ema9_prev - ema21_prev
    ema_diff_latest = ema9_latest - ema21_latest

    ema_strength = abs(ema_diff_latest)

    rsi_buy_conf = max(0, min(1, (rsi_latest - 30) / 40))   # RSI 30-70 scaled 0-1
    rsi_sell_conf = max(0, min(1, (70 - rsi_latest) / 40))  # RSI 70-30 scaled 0-1

    if (ema_diff_prev < 0) and (ema_diff_latest > 0) and (rsi_latest > 30):
        signal = "Buy ✅"
        explanation.append("EMA9 crossed above EMA21 and RSI > 30")
        confidence = round(min(1, ema_strength * 10) * rsi_buy_conf, 2)
    elif (ema_diff_prev > 0) and (ema_diff_latest < 0) and (rsi_latest < 70):
        signal = "Sell ❌"
        explanation.append("EMA9 crossed below EMA21 and RSI < 70")
        confidence = round(min(1, abs(ema_diff_latest) * 10) * rsi_sell_conf, 2)
    else:
        explanation.append("No clear crossover or RSI in neutral zone")
        confidence = 0

    return signal, "; ".join(explanation), confidence

if "signal_log" not in st.session_state:
    st.session_state.signal_log = []

if run_button:
    with st.spinner("Running predictions and analysis..."):
        try:
            # Download data with validation
            df = yf.download(symbol, period=timeframe, interval="1d", progress=False)

            required_cols = ['Open', 'High', 'Low', 'Close']

            if df.empty:
                st.error(f"No data found for symbol '{symbol}'. Please check the ticker and try again.")
                st.stop()

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Downloaded data missing required columns: {missing_cols}. Symbol: {symbol}")
                st.stop()

            df = df.dropna(subset=required_cols)
            if df.empty:
                st.error("Data after dropping rows with missing OHLC values is empty.")
                st.stop()

            # Calculate indicators
            df['EMA9'] = calculate_ema(df['Close'], 9)
            df['EMA21'] = calculate_ema(df['Close'], 21)
            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df)
            df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df)
            df.dropna(inplace=True)

            # Generate trading signals & positions
            df = generate_signals(df)

            # Latest and previous for explanation
            latest, prev = df.iloc[-1], df.iloc[-2]

            # Show signal & explanation
            signal, explanation, confidence = explain_signal(latest, prev)
            st.subheader(f"Signal: {signal} (Confidence: {confidence * 100:.0f}%)")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown(f"**RSI:** {round(latest['RSI'], 2)}")

            # Log signals in session state
            if signal != "Neutral":
                st.session_state.signal_log.append({
                    "symbol": symbol,
                    "signal": signal,
                    "rsi": round(latest['RSI'], 2),
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "confidence": confidence
                })

            # Alert email placeholder
            if alert_email and signal != "Neutral":
                send_email_alert(alert_email, signal, symbol)

            if st.checkbox("Show Signal History"):
                st.dataframe(pd.DataFrame(st.session_state.signal_log))

            # News sentiment (mocked)
            st.subheader("News Sentiment (Mocked)")
            st.markdown(fetch_news_sentiment(symbol))

            # Backtesting performance and trades
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

            # LSTM forecast
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

                st.dataframe(df_future, use_container_width=True)
                st.download_button("Download LSTM Forecast", df_future.to_csv(index=False), file_name=f"{symbol}_lstm.csv")

            except Exception as e:
                st.error(f"LSTM Error: {e}")
                st.text(traceback.format_exc())

            # Price chart with buy/sell signals
            st.subheader("Price Chart with Buy/Sell Signals")
            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price"
            ))

            buy_signals = df[df['Trade_Signal'] == 1]
            sell_signals = df[df['Trade_Signal'] == -1]

            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Buy Signal'
            ))

            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Sell Signal'
            ))

            fig.update_layout(
                title=f"{symbol} Price with Trading Signals",
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("An unexpected error occurred:")
            st.error(traceback.format_exc())
