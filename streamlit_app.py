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

# --- Indicator calculation functions ---

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
    y = np.array(y).reshape(-1)
    return X, y, scaler

def generate_signals(df):
    df = df.copy()
    df['Signal'] = 0
    df.loc[(df['EMA9'] > df['EMA21']) & (df['RSI'] > 30), 'Signal'] = 1
    df.loc[(df['EMA9'] < df['EMA21']) & (df['RSI'] < 70), 'Signal'] = -1
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0).astype(int)
    return df

def backtest_signals(df, initial_cash=1000):
    df = df.copy()
    df['Position'] = 0

    buy_cond = (df['EMA9'].shift(1) < df['EMA21'].shift(1)) & (df['EMA9'] > df['EMA21']) & (df['RSI'] > 30)
    sell_cond = (df['EMA9'].shift(1) > df['EMA21'].shift(1)) & (df['EMA9'] < df['EMA21']) & (df['RSI'] < 70)

    df.loc[buy_cond, 'Position'] = 1
    df.loc[sell_cond, 'Position'] = -1
    df['Position'] = df['Position'].astype(int)

    position = 0
    shares = 0
    cash = initial_cash
    portfolio_values = []
    trades = []

    for idx, pos in zip(df.index, df['Position']):
        close_price = float(df.loc[idx, 'Close'])

        if position == 0 and pos == 1:
            shares = cash // close_price
            if shares > 0:
                cash -= shares * close_price
                position = 1
                trades.append({'Entry Date': idx, 'Entry Price': close_price, 'Exit Date': None, 'Exit Price': None, 'Return %': None})

        elif position == 1 and pos == -1 and shares > 0:
            cash += shares * close_price
            position = 0
            trades[-1]['Exit Date'] = idx
            trades[-1]['Exit Price'] = close_price
            trades[-1]['Return %'] = (close_price - trades[-1]['Entry Price']) / trades[-1]['Entry Price'] * 100
            shares = 0

        portfolio_value = cash + shares * close_price
        portfolio_values.append({'Date': idx, 'Portfolio Value': portfolio_value})

    if position == 1 and shares > 0:
        close_price = float(df['Close'].iloc[-1])
        cash += shares * close_price
        trades[-1]['Exit Date'] = df.index[-1]
        trades[-1]['Exit Price'] = close_price
        trades[-1]['Return %'] = (close_price - trades[-1]['Entry Price']) / trades[-1]['Entry Price'] * 100
        shares = 0
        portfolio_values.append({'Date': df.index[-1], 'Portfolio Value': cash})

    trades_df = pd.DataFrame(trades)
    portfolio_df = pd.DataFrame(portfolio_values).set_index('Date')

    if not trades_df.empty:
        trades_df['Return %'] = pd.to_numeric(trades_df['Return %'], errors='coerce')
        trades_df_clean = trades_df.dropna(subset=['Return %'])
        total_return = (portfolio_df['Portfolio Value'][-1] - initial_cash) / initial_cash * 100
        win_rate = (trades_df_clean['Return %'] > 0).mean() * 100 if not trades_df_clean.empty else 0
        num_trades = len(trades_df_clean)
    else:
        total_return = 0
        win_rate = 0
        num_trades = 0

    return trades_df, portfolio_df, total_return, win_rate, num_trades

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
    rsi_buy_conf = max(0, min(1, (rsi_latest - 30) / 40))
    rsi_sell_conf = max(0, min(1, (70 - rsi_latest) / 40))

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
            for tf in [timeframe, "3mo", "6mo", "1y"]:
                price_df = yf.download(symbol, period=tf, interval="1d", progress=False)
                if price_df.shape[0] >= 50:
                    st.info(f"Using timeframe: {tf}")
                    break
            else:
                st.error("Not enough data for this symbol.")
                st.stop()

            price_df.dropna(inplace=True)
            if price_df.empty:
                st.error("No data found.")
                st.stop()

            price_df.index = pd.to_datetime(price_df.index)
            df = price_df.copy()

            df['EMA9'] = calculate_ema(df['Close'], 9)
            df['EMA21'] = calculate_ema(df['Close'], 21)
            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df)
            df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df)

            df.dropna(inplace=True)

            df = generate_signals(df)

            latest, prev = df.iloc[-1], df.iloc[-2]

            signal, explanation, confidence = explain_signal(latest, prev)
            st.subheader(f"Signal: {signal} (Confidence: {confidence * 100:.0f}%)")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown(f"**RSI:** {round(latest['RSI'], 2)}")

            if signal != "Neutral":
                st.session_state.signal_log.append({
                    "symbol": symbol,
                    "signal": signal,
                    "rsi": round(latest['RSI'], 2),
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "confidence": confidence
                })

            if alert_email and signal != "Neutral":
                # Here you can implement real email alert sending
                st.success(f"Alert email would be sent to {alert_email} (demo only)")

            if st.checkbox("Show Signal History"):
                st.dataframe(pd.DataFrame(st.session_state.signal_log))

            st.subheader("News Sentiment (Mocked)")
            st.markdown("Sentiment: [Mock sentiment placeholder]")

            st.subheader("📊 Backtesting Performance")
            trades_df, portfolio_df, total_return, win_rate, num_trades = backtest_signals(df)
            st.markdown(f"**Number of trades:** {num_trades}")
            st.markdown(f"**Total return:** {total_return:.2f}%")
            st.markdown(f"**Win rate:** {win_rate:.2f}%")
            if not trades_df.empty:
                st.dataframe(trades_df)

            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Candlestick(
                x=price_df.index,
                open=price_df['Open'], high=price_df['High'], low=price_df['Low'], close=price_df['Close'],
                name='Historical'
            ))

            if not trades_df.empty:
                fig_backtest.add_trace(go.Scatter(
                    x=trades_df['Entry Date'],
                    y=trades_df['Entry Price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=12),
                    name='Buy'
                ))
                fig_backtest.add_trace(go.Scatter(
                    x=trades_df['Exit Date'],
                    y=trades_df['Exit Price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=12),
                    name='Sell'
                ))

                annotations = []
                for _, row in trades_df.dropna(subset=['Exit Date']).iterrows():
                    ret = row['Return %']
                    exit_date = row['Exit Date']
                    exit_price = row['Exit Price']
                    color = 'green' if ret > 0 else 'red'
                    annotations.append(dict(
                        x=exit_date,
                        y=exit_price,
                        xref='x',
                        yref='y',
                        text=f"{ret:.2f}%",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-20,
                        font=dict(color=color, size=12),
                        arrowcolor=color
                    ))

                fig_backtest.update_layout(annotations=annotations)

            fig_backtest.update_layout(
                title=f"{symbol} Backtest Trades",
                yaxis_title='Price (USD)',
                xaxis_title='Date',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_backtest, use_container_width=True)

            st.subheader(f"Prophet Forecast (Next {int(prophet_period)} Days)")

            df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            # Convert 'y' to 1D Series of numeric values
            df_prophet['y'] = pd.to_numeric(df_prophet['y'].values.flatten(), errors='coerce')
            df_prophet = df_prophet.dropna(subset=['y'])

            if len(df_prophet) < 30:
                st.warning("Not enough data for Prophet forecasting.")
            else:
                model_prophet = Prophet()
                model_prophet.fit(df_prophet)
                future = model_prophet.make_future_dataframe(periods=int(prophet_period))
                forecast = model_prophet.predict(future)

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                fig1.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", showlegend=True, name='Confidence Interval'))
                fig1.update_layout(title=f"{symbol} Prophet Forecast", yaxis_title='Price (USD)', xaxis_title='Date')
                st.plotly_chart(fig1, use_container_width=True)
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10), use_container_width=True)
                st.download_button("Download Prophet Forecast CSV", forecast.to_csv(index=False), file_name=f"{symbol}_prophet_forecast.csv")

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
                lstm_forecast_scaled = []
                for _ in range(int(lstm_period)):
                    pred = model.predict(future_input)[0][0]
                    lstm_forecast_scaled.append(pred)
                    new_seq = np.append(future_input[:,1:,:], [[[pred]]], axis=1)
                    future_input = new_seq

                lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1,1)).flatten()

                future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=int(lstm_period))
                df_lstm_forecast = pd.DataFrame({'Date': future_dates, 'LSTM Forecast': lstm_forecast})

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Close'))
                fig2.add_trace(go.Scatter(x=df_lstm_forecast['Date'], y=df_lstm_forecast['LSTM Forecast'], mode='lines', name='LSTM Forecast'))
                fig2.update_layout(title=f"{symbol} LSTM Forecast", yaxis_title='Price (USD)', xaxis_title='Date')
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(df_lstm_forecast, use_container_width=True)
                st.download_button("Download LSTM Forecast CSV", df_lstm_forecast.to_csv(index=False), file_name=f"{symbol}_lstm_forecast.csv")

            except Exception as e:
                st.error(f"LSTM forecasting failed: {e}")
                st.text(traceback.format_exc())

            st.subheader("Technical Indicators and Candlestick Chart")
            fig_indicators = go.Figure()
            fig_indicators.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='Candlestick'))
            fig_indicators.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA9'))
            fig_indicators.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21'))
            fig_indicators.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='BB Upper', line=dict(dash='dot')))
            fig_indicators.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='BB Lower', line=dict(dash='dot')))
            fig_indicators.update_layout(title=f"{symbol} Price with Indicators", yaxis_title='Price (USD)', xaxis_title='Date')
            st.plotly_chart(fig_indicators, use_container_width=True)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.text(traceback.format_exc())
