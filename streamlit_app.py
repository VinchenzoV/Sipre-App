import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objs as go
import datetime

# --- Page Setup ---
st.set_page_config(
    page_title="📈 Sipre Pro — Predictive Trading Signal Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark Mode Toggle ---
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #0E1117;
            color: #D7D9DB;
        }
        .stButton>button {
            background-color: #1F2937;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# --- Indicator functions ---
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_sma(series, period):
    return series.rolling(window=period).mean()

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

# --- Signal generation ---
def generate_signals(df):
    df['EMA9'] = calculate_ema(df['Close'], 9)
    df['EMA21'] = calculate_ema(df['Close'], 21)
    df['RSI'] = calculate_rsi(df['Close'])

    df['Signal'] = 0
    buy_cond = (df['EMA9'].shift(1) < df['EMA21'].shift(1)) & (df['EMA9'] > df['EMA21']) & (df['RSI'] > 30)
    sell_cond = (df['EMA9'].shift(1) > df['EMA21'].shift(1)) & (df['EMA9'] < df['EMA21']) & (df['RSI'] < 70)

    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0).astype(int)
    return df

# --- Backtest ---
def backtest_signals(df):
    trades = []
    position = 0
    entry_price = 0

    for idx, row in df.iterrows():
        signal = row['Signal']
        price = row['Close']

        if position == 0 and signal == 1:
            position = 1
            entry_price = price
            trades.append({'Entry Date': idx, 'Entry Price': entry_price, 'Exit Date': None, 'Exit Price': None, 'Return %': None})
        elif position == 1 and signal == -1:
            position = 0
            exit_price = price
            trades[-1]['Exit Date'] = idx
            trades[-1]['Exit Price'] = exit_price
            trades[-1]['Return %'] = (exit_price - entry_price) / entry_price * 100

    if position == 1:
        exit_price = df['Close'].iloc[-1]
        trades[-1]['Exit Date'] = df.index[-1]
        trades[-1]['Exit Price'] = exit_price
        trades[-1]['Return %'] = (exit_price - entry_price) / entry_price * 100

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, 0, 0, 0

    trades_df['Return %'] = pd.to_numeric(trades_df['Return %'], errors='coerce')
    total_return = trades_df['Return %'].sum()
    win_rate = (trades_df['Return %'] > 0).mean() * 100
    num_trades = len(trades_df)
    return trades_df, total_return, win_rate, num_trades

# --- Explain signal ---
def explain_signal(df):
    if len(df) < 2:
        return "Neutral", "Insufficient data to explain signals.", 0

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    ema_diff_prev = prev['EMA9'] - prev['EMA21']
    ema_diff_latest = latest['EMA9'] - latest['EMA21']
    rsi_latest = latest['RSI']

    signal = "Neutral"
    explanation = []
    confidence = 0

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

# --- Fetch stock data with caching ---
@st.cache_data(show_spinner=False)
def fetch_data(symbol, period):
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    df.dropna(inplace=True)
    return df

# --- Prophet forecasting ---
def prophet_forecast(df, days):
    df_reset = df.reset_index()
    prophet_df = pd.DataFrame({
        'ds': df_reset['Date'] if 'Date' in df_reset.columns else df_reset[df_reset.columns[0]],
        'y': np.log1p(df_reset['Close'])
    })
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    forecast['yhat_exp'] = np.expm1(forecast['yhat'])
    return forecast

# --- Prepare data for LSTM ---
def prepare_lstm_data(df, seq_len=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

# --- Build and train LSTM model ---
def build_train_lstm(X_train, y_train, epochs=20, batch_size=32):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])
    return model

# --- Plotly charts ---
def plot_price_chart(df, signals, show_ema=True, show_sma=True, show_bbands=True, show_macd=True, show_rsi=True, show_volume=True):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
    ))

    # EMA
    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA 9', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA 21', line=dict(color='blue')))

    # SMA
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA 200', line=dict(color='green')))

    # Bollinger Bands
    if show_bbands:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='BB Upper', line=dict(color='rgba(255,0,0,0.3)')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='BB Lower', line=dict(color='rgba(0,0,255,0.3)')))

    # Buy/Sell markers
    buy_signals = signals['Signal'] == 1
    sell_signals = signals['Signal'] == -1
    fig.add_trace(go.Scatter(
        x=signals.index[buy_signals],
        y=df['Close'][buy_signals],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=12),
        name='Buy Signal',
        hovertemplate='Buy<br>Date: %{x}<br>Price: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=signals.index[sell_signals],
        y=df['Close'][sell_signals],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell Signal',
        hovertemplate='Sell<br>Date: %{x}<br>Price: %{y}<extra></extra>'
    ))

    # Volume
    if show_volume:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue', yaxis='y2', opacity=0.3))

    # Layout with 2 y axes
    fig.update_layout(
        height=600,
        xaxis=dict(title='Date', rangeslider=dict(visible=False)),
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False, position=1.0),
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='x unified'
    )
    return fig

# --- Main App ---

st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

with st.sidebar:
    st.header("Settings")

    symbols_list = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "SPY", "BTC-USD", "ETH-USD"]
    multi_select = st.multiselect("Select one or more symbols:", options=symbols_list, default=["AAPL", "MSFT"])

    timeframe = st.selectbox("Select timeframe for historical data:", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    prophet_days = st.slider("Prophet forecast days:", 5, 90, 30, 5)
    lstm_days = st.slider("LSTM forecast days:", 5, 90, 30, 5)

    st.subheader("Indicators to display:")
    show_ema = st.checkbox("EMA (9 & 21)", value=True)
    show_sma = st.checkbox("SMA (50 & 200)", value=False)
    show_bbands = st.checkbox("Bollinger Bands", value=True)
    show_macd = st.checkbox("MACD", value=False)  # not plotted here yet (can add later)
    show_rsi = st.checkbox("RSI", value=True)
    show_volume = st.checkbox("Volume", value=True)

    alert_email = st.text_input("Enter your email for alerts (optional):")
    alert_threshold = st.slider("RSI Alert Threshold:", 10, 90, 30)
    enable_alerts = st.checkbox("Enable Alerts", value=False)

    run_button = st.button("Run Prediction & Analysis")

if run_button:
    for symbol in multi_select:
        st.header(f"### {symbol}")

        df = fetch_data(symbol, timeframe)
        if df.empty or len(df) < 60:
            st.error(f"Not enough data for {symbol} to proceed.")
            continue

        # Calculate indicators
        df['EMA9'] = calculate_ema(df['Close'], 9)
        df['EMA21'] = calculate_ema(df['Close'], 21)
        df['SMA50'] = calculate_sma(df['Close'], 50)
        df['SMA200'] = calculate_sma(df['Close'], 200)
        df['RSI'] = calculate_rsi(df['Close'])
        df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df)

        # Signals and backtest
        df_signals = generate_signals(df)
        trades_df, total_return, win_rate, num_trades = backtest_signals(df_signals)
        signal, explanation, confidence = explain_signal(df_signals)

        # Show main signal and explanation
        st.markdown(f"**Current Signal:** {signal} (Confidence: {confidence*100:.0f}%)")
        st.markdown(f"**Explanation:** {explanation}")

        # Alert placeholder
        if enable_alerts and alert_email and signal != "Neutral":
            if (signal == "Buy ✅" and df_signals['RSI'].iloc[-1] >= alert_threshold) or \
               (signal == "Sell ❌" and df_signals['RSI'].iloc[-1] <= alert_threshold):
                st.success(f"Alert: Signal '{signal}' for {symbol} would be sent to {alert_email}")

        # Price + indicators chart
        fig = plot_price_chart(df, df_signals, show_ema, show_sma, show_bbands, show_macd, show_rsi, show_volume)
        st.plotly_chart(fig, use_container_width=True)

        # Backtest trades summary
        st.subheader("Backtest Trade Log")
        if trades_df.empty:
            st.info("No trades generated by strategy in backtest period.")
        else:
            st.dataframe(trades_df)
            st.markdown(f"**Total Return:** {total_return:.2f}%")
            st.markdown(f"**Win Rate:** {win_rate:.2f}%")
            st.markdown(f"**Number of Trades:** {num_trades}")

        # Prophet forecast
        st.subheader("Prophet Forecast (Log-scale)")
        prophet_forecast_df = prophet_forecast(df, prophet_days)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=prophet_forecast_df['ds'],
            y=prophet_forecast_df['yhat_exp'],
            mode='lines',
            name='Prophet Forecast'
        ))
        fig2.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Historical Close'
        ))
        fig2.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=30), xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig2, use_container_width=True)

        # LSTM forecast
        st.subheader("LSTM Forecast")
        try:
            seq_len = 60
            X, y, scaler = prepare_lstm_data(df, seq_len)
            model = build_train_lstm(X, y, epochs=25)

            # Predict future lstm_days
            last_seq = X[-1]
            preds = []
            cur_seq = last_seq
            for _ in range(lstm_days):
                pred = model.predict(cur_seq[np.newaxis, :, :])[0]
                preds.append(pred)
                cur_seq = np.vstack([cur_seq[1:], pred])

            preds = scaler.inverse_transform(preds).flatten()

            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=lstm_days, freq='B')
            lstm_forecast_df = pd.DataFrame({'Date': future_dates, 'LSTM Forecast': preds})

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Close'))
            fig3.add_trace(go.Scatter(x=lstm_forecast_df['Date'], y=lstm_forecast_df['LSTM Forecast'], mode='lines', name='LSTM Forecast'))
            fig3.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=30), xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"LSTM forecast failed: {e}")

# --- End of script ---
