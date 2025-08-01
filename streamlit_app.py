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

def generate_signals(df):
    df = df.copy()
    df['Signal'] = 0

    # Signal = 1 when EMA9 > EMA21 and RSI > 30
    df.loc[(df['EMA9'] > df['EMA21']) & (df['RSI'] > 30), 'Signal'] = 1
    # Signal = -1 when EMA9 < EMA21 and RSI < 70
    df.loc[(df['EMA9'] < df['EMA21']) & (df['RSI'] < 70), 'Signal'] = -1

    # Forward fill Position based on Signal changes
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

    cash = initial_cash
    shares = 0
    portfolio_values = []
    trades = []
    position = 0
    entry_price = 0.0
    entry_date = None

    for idx, pos in zip(df.index, df['Position']):
        price = df.loc[idx, 'Close']
        if position == 0 and pos == 1:
            # Buy as many shares as possible
            shares = cash // price
            if shares > 0:
                cash -= shares * price
                position = 1
                entry_price = price
                entry_date = pd.Timestamp(idx)
                trades.append({'Entry Date': entry_date, 'Entry Price': entry_price, 'Exit Date': None, 'Exit Price': None, 'Return %': None, 'Shares': shares})
        elif position == 1 and pos == -1:
            # Sell all shares
            cash += shares * price
            exit_price = price
            exit_date = pd.Timestamp(idx)
            position = 0
            trades[-1]['Exit Date'] = exit_date
            trades[-1]['Exit Price'] = exit_price
            trades[-1]['Return %'] = (exit_price - trades[-1]['Entry Price']) / trades[-1]['Entry Price'] * 100
            trades[-1]['Shares'] = shares
            shares = 0
        # Update portfolio value daily
        current_portfolio_value = cash + shares * price
        portfolio_values.append({'Date': idx, 'Portfolio Value': current_portfolio_value})

    # Close open position at last date if any
    if position == 1:
        price = df['Close'].iloc[-1]
        cash += shares * price
        exit_date = df.index[-1]
        trades[-1]['Exit Date'] = exit_date
        trades[-1]['Exit Price'] = price
        trades[-1]['Return %'] = (price - trades[-1]['Entry Price']) / trades[-1]['Entry Price'] * 100
        trades[-1]['Shares'] = shares
        shares = 0
        current_portfolio_value = cash
        portfolio_values[-1]['Portfolio Value'] = current_portfolio_value

    trades_df = pd.DataFrame(trades)
    portfolio_df = pd.DataFrame(portfolio_values).set_index('Date')

    total_return = (portfolio_df['Portfolio Value'].iloc[-1] - initial_cash) / initial_cash * 100 if not portfolio_df.empty else 0
    win_rate = (trades_df['Return %'] > 0).mean() * 100 if not trades_df.empty else 0
    num_trades = len(trades_df)

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

    # Strength of EMA difference
    ema_strength = abs(ema_diff_latest)

    # Normalize RSI distances for confidence scaling
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
            # Download data
            for tf in [timeframe, "3mo", "6mo", "1y"]:
                df = yf.download(symbol, period=tf, interval="1d", progress=False)
                if df.shape[0] >= 50:
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

            # Generate trading signals
            df = generate_signals(df)

            latest, prev = df.iloc[-1], df.iloc[-2]

            # Signal & explainability
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
                send_email_alert(alert_email, signal, symbol)

            if st.checkbox("Show Signal History"):
                st.dataframe(pd.DataFrame(st.session_state.signal_log))

            st.subheader("News Sentiment (Mocked)")
            st.markdown(fetch_news_sentiment(symbol))

            # Backtesting
            st.subheader("📊 Backtesting Performance")
            trades_df, portfolio_df, total_return, win_rate, num_trades = backtest_signals(df, initial_cash=1000)
            st.markdown(f"**Initial cash:** $1000")
            st.markdown(f"**Number of trades:** {num_trades}")
            st.markdown(f"**Total return:** {total_return:.2f}%")
            st.markdown(f"**Win rate:** {win_rate:.2f}%")
            if not trades_df.empty:
                st.dataframe(trades_df)

            # Plot backtest trades on candlestick chart with annotations and portfolio value line
            fig_backtest = go.Figure()

            fig_backtest.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='Historical'
            ))

            # Plot buy/sell markers
            buy_signals = trades_df.dropna(subset=['Exit Date']).copy()
            buy_signals = buy_signals[['Entry Date', 'Entry Price', 'Return %', 'Shares']]

            sell_signals = trades_df.dropna(subset=['Exit Date']).copy()
            sell_signals = sell_signals[['Exit Date', 'Exit Price', 'Return %', 'Shares']]

            fig_backtest.add_trace(go.Scatter(
                x=buy_signals['Entry Date'],
                y=buy_signals['Entry Price'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=12),
                name='Buy'
            ))

            fig_backtest.add_trace(go.Scatter(
                x=sell_signals['Exit Date'],
                y=sell_signals['Exit Price'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=12),
                name='Sell'
            ))

            # Add return % annotations on sell signals
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

            # Add portfolio value line
            fig_backtest.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['Portfolio Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))

            fig_backtest.update_layout(
                title=f"{symbol} Backtest Trades & Portfolio Value",
                annotations=annotations,
                yaxis_title='Price (USD)',
                yaxis2=dict(
                    title='Portfolio Value (USD)',
                    overlaying='y',
                    side='right'
                ),
                xaxis_title='Date',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_backtest)

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
                future_preds_scaled = []
                for _ in range(int(lstm_period)):
                    pred_scaled = model.predict(future_input, verbose=0)[0][0]
                    future_preds_scaled.append(pred_scaled)
                    pred_array = np.array([[[pred_scaled]]])
                    future_input = np.concatenate((future_input[:, 1:, :], pred_array), axis=1)

                future_prices = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
                last_close = float(df['Close'].iloc[-1])
                clipped_prices = np.clip(future_prices, last_close * 0.9, None)

                future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=int(lstm_period), freq='D')
                df_future = pd.DataFrame({'Date': future_dates, 'LSTM Forecast': clipped_prices})

                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Historical'
                ))
                fig_lstm.add_trace(go.Scatter(
                    x=df_future['Date'], y=df_future['LSTM Forecast'],
                    mode='lines+markers', name='LSTM Forecast', line=dict(color='orange')
                ))
                fig_lstm.update_layout(title=f"{symbol} LSTM Forecast", yaxis_title='Price (USD)', xaxis_title='Date')
                st.plotly_chart(fig_lstm)
                st.dataframe(df_future, use_container_width=True)
                st.download_button("Download LSTM Forecast", df_future.to_csv(index=False), file_name=f"{symbol}_lstm.csv")

            except Exception as e:
                st.error("Error in LSTM prediction: " + str(e))
                st.text(traceback.format_exc())

        except Exception as e:
            st.error(f"Error fetching or processing data: {e}")
            st.text(traceback.format_exc())
