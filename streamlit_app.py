import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
import plotly.graph_objs as go
import traceback

# --- Page config ---
st.set_page_config(page_title="📈 Sipre Pro — Predictive Trading Signal Dashboard", layout="wide")

# --- Title ---
st.title("📈 Sipre Pro — Predictive Trading Signal Dashboard")

# --- Load S&P 500 symbols with caching ---
@st.cache_data
def load_symbols():
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].dropna().str.upper().tolist()
    except Exception:
        # Fallback list if loading fails
        return ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "SPY", "BTC-USD", "ETH-USD"]

symbols_list = load_symbols()

# --- Sidebar inputs ---
with st.sidebar:
    st.header("🔧 Settings")

    user_input = st.text_input("Enter symbol (e.g. LNR.TO or AAPL):", value="AAPL").upper().strip()
    filtered_symbols = [s for s in symbols_list if user_input in s] if user_input else symbols_list
    selected_symbol = st.selectbox("Or select from suggestions:", filtered_symbols) if filtered_symbols else None
    symbol = user_input if user_input else selected_symbol

    timeframe = st.selectbox("Select timeframe for historical data:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    days_to_predict = st.number_input(
        "Days to predict into the future:", min_value=5, max_value=60, value=15, step=5
    )

    alert_email = st.text_input("Email for alerts (optional):")

    st.markdown("---")
    st.markdown("Press the button below to get predictions & signals.")

# --- Validate symbol ---
if not symbol:
    st.warning("Please enter or select a valid symbol from the sidebar.")
    st.stop()

# --- Utility functions ---
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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

def fetch_news_sentiment(symbol):
    # Placeholder for real news sentiment
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        response = requests.get(url)
        # Real sentiment extraction logic would go here
        return "Sentiment: [Mock sentiment placeholder]"
    except:
        return "Sentiment unavailable"

def send_email_alert(recipient, signal, symbol):
    # Placeholder for sending email alerts
    try:
        st.success(f"Alert email would be sent to {recipient} (demo mode).")
    except:
        st.error("Failed to send email alert.")

# --- Main action button ---
if st.sidebar.button("Get Prediction & Signal"):
    try:
        # Fetch historical data with fallback timeframes
        timeframes_to_try = [timeframe, "3mo", "6mo", "1y", "2y", "5y"]
        for tf in timeframes_to_try:
            df = yf.download(symbol, period=tf, interval="1d")
            if len(df) >= 30:
                st.info(f"Using timeframe: {tf}")
                break
        else:
            st.warning("⚠️ Not enough data available for this symbol.")
            st.stop()

        df.dropna(inplace=True)
        if df.empty:
            st.error("No data retrieved for the symbol.")
            st.stop()

        # Calculate indicators
        df['EMA9'] = calculate_ema(df['Close'], 9)
        df['EMA21'] = calculate_ema(df['Close'], 21)
        df['RSI'] = calculate_rsi(df['Close'])
        df.dropna(inplace=True)

        # Trading signal logic
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        ema9_latest = float(latest["EMA9"])
        ema21_latest = float(latest["EMA21"])
        ema9_prev = float(prev["EMA9"])
        ema21_prev = float(prev["EMA21"])
        rsi_latest = float(latest["RSI"])

        signal = "Neutral"
        if (ema9_prev < ema21_prev) and (ema9_latest > ema21_latest) and (rsi_latest > 30):
            signal = "Buy ✅"
        elif (ema9_prev > ema21_prev) and (ema9_latest < ema21_latest) and (rsi_latest < 70):
            signal = "Sell ❌"

        # --- Layout ---
        st.subheader(f"📌 Trading Signal for {symbol}: {signal}")
        st.markdown(f"**RSI:** {round(rsi_latest, 2)}")

        if alert_email and signal != "Neutral":
            send_email_alert(alert_email, signal, symbol)

        st.subheader("📰 News Sentiment (Mocked)")
        st.markdown(fetch_news_sentiment(symbol))

        # --- Prophet Forecast ---
        st.subheader(f"📅 Prophet Forecast (Next {days_to_predict} Days)")

        df_reset = df.reset_index()

        close_col = 'Close'
        if close_col not in df_reset.columns:
            st.error("No 'Close' column found.")
            st.stop()

        # Ensure min close price is a scalar number
        min_close_val = df_reset[close_col].min()
        if isinstance(min_close_val, (pd.Series, np.ndarray)):
            min_close_val = min_close_val.min()
        min_close_val = float(min_close_val)

        min_price_clip = max(1.0, min_close_val)
        prices_clipped = df_reset[close_col].clip(lower=min_price_clip)

        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df_reset[df_reset.columns[0]]),
            'y': np.log(prices_clipped)  # log-transform for Prophet stability
        }).dropna()

        if prophet_df.shape[0] < 30:
            st.warning("Not enough data for Prophet forecasting.")
        else:
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=days_to_predict)
            forecast = m.predict(future)

            # Exponentiate carefully (clip lower bound to avoid negatives)
            min_positive = 1e-3
            forecast['yhat_exp'] = np.exp(forecast['yhat'])
            forecast['yhat_lower_exp'] = np.exp(forecast['yhat_lower'].clip(lower=np.log(min_positive)))
            forecast['yhat_upper_exp'] = np.exp(forecast['yhat_upper'])

            # Plot forecast with confidence intervals
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
            fig1.update_layout(title=f"{symbol} Prophet Forecast (Next {days_to_predict} Days)",
                               yaxis_title='Price (USD)',
                               xaxis_title='Date')
            st.plotly_chart(fig1, use_container_width=True)

            st.dataframe(forecast[['ds', 'yhat_exp', 'yhat_lower_exp', 'yhat_upper_exp']].tail(days_to_predict), use_container_width=True)
            st.download_button("📥 Download Prophet Forecast CSV", forecast.to_csv(index=False), file_name=f"{symbol}_prophet_forecast.csv")

        # --- LSTM Forecast ---
        st.subheader(f"🤖 LSTM Future Price Prediction (Next {days_to_predict} Days)")

        try:
            if df.shape[0] < 50:
                raise ValueError("Not enough data points for LSTM prediction (need at least 50).")

            seq_len = min(60, df.shape[0] - 1)
            X, y, scaler = prepare_lstm_data(df, sequence_length=seq_len)

            # Build LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
            model.add(LSTM(units=50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            # Generate predictions iteratively
            future_input = X[-1].reshape(1, X.shape[1], X.shape[2])
            future_preds = []
            for _ in range(days_to_predict):
                pred = model.predict(future_input, verbose=0)[0][0]
                future_preds.append(pred)
                pred_array = np.array([[[pred]]], dtype=np.float32)
                future_input = np.concatenate((future_input[:, 1:, :], pred_array), axis=1)

            future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
            last_close = df['Close'].iloc[-1]

            # Avoid unrealistic price drops below 90% of last close
            clipped_prices = np.clip(future_prices, last_close * 0.9, None)
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days_to_predict, freq='D')

            df_future = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close': clipped_prices
            })

            min_price = min(df['Close'].min(), df_future['Predicted Close'].min())
            yaxis_min = max(min_price * 0.95, 0)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical"))
            fig2.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Predicted Close'],
                                      name="LSTM Forecast", line=dict(dash='dot')))
            fig2.update_layout(
                title=f"{symbol} — Combined Price & LSTM Forecast",
                yaxis=dict(range=[yaxis_min, None]),
                xaxis_title="Date",
                yaxis_title="Price (USD)"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(df_future, use_container_width=True)
            st.download_button("📥 Download LSTM Forecast CSV", df_future.to_csv(index=False), file_name=f"{symbol}_lstm_forecast.csv")

        except ValueError as ve:
            st.warning(f"LSTM Skipped: {ve}")
        except Exception as e:
            st.error(f"❌ LSTM Error: {e}")
            st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.text(traceback.format_exc())
