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

st.set_page_config(page_title="Sipre Pro", layout="wide")
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

user_input = st.text_input("Enter symbol (e.g. LNR.TO or AAPL):").upper().strip()
filtered_symbols = [s for s in symbols_list if user_input in s] if user_input else symbols_list
selected_symbol = st.selectbox("Or select from suggestions:", filtered_symbols) if filtered_symbols else None
symbol = user_input if user_input else selected_symbol

if not symbol:
    st.warning("Please enter or select a valid symbol.")
    st.stop()

timeframe = st.selectbox("Select timeframe:", ["1mo", "3mo", "6mo", "1y"])
alert_email = st.text_input("Enter your email for alerts (optional):")

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_news_sentiment(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        response = requests.get(url)
        return "Sentiment: [Mock sentiment placeholder]"
    except:
        return "Sentiment unavailable"

def send_email_alert(recipient, signal, symbol):
    try:
        st.success(f"Alert email would be sent to {recipient} (demo)")
    except:
        st.error("Failed to send email alert.")

if st.button("Get Prediction & Signal"):
    try:
        timeframes_to_try = [timeframe, "3mo", "6mo", "1y"]
        for tf in timeframes_to_try:
            df = yf.download(symbol, period=tf, interval="1d")
            if len(df) >= 30:
                st.info(f"Using timeframe: {tf}")
                break
        else:
            st.warning("⚠️ Not enough data available for this symbol.")
            st.stop()

        df.dropna(inplace=True)
        df['EMA9'] = calculate_ema(df['Close'], 9)
        df['EMA21'] = calculate_ema(df['Close'], 21)
        df['RSI'] = calculate_rsi(df['Close'])
        df.dropna(inplace=True)

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

        st.subheader(f"📌 Signal: {signal}")
        st.markdown(f"**RSI:** {round(rsi_latest, 2)}")

        if alert_email and signal != "Neutral":
            send_email_alert(alert_email, signal, symbol)

        st.subheader("📰 News Sentiment (Mocked)")
        st.markdown(fetch_news_sentiment(symbol))

        # --- Prophet Forecast ---
        st.subheader("📅 Prophet Forecast (Next 15 Days)")

        df_reset = df.reset_index()

        close_col = 'Close'
        if close_col not in df_reset.columns:
            st.error("No 'Close' column found.")
            st.stop()

        min_price_clip = max(1.0, df_reset[close_col].min())
        prices_clipped = df_reset[close_col].clip(lower=min_price_clip)

        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df_reset[df_reset.columns[0]]),
            'y': np.log(prices_clipped)
        }).dropna()

        st.write("Sample of data used for Prophet:")
        st.dataframe(prophet_df.head())

        if prophet_df.shape[0] < 30:
            st.warning("Not enough data for Prophet forecasting.")
        else:
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=15)
            forecast = m.predict(future)

            min_positive = 1e-3
            forecast['yhat_exp'] = np.exp(forecast['yhat'])
            forecast['yhat_lower_exp'] = np.exp(forecast['yhat_lower'].clip(lower=np.log(min_positive)))
            forecast['yhat_upper_exp'] = np.exp(forecast['yhat_upper'])

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
            fig1.update_layout(title=f"{symbol} Prophet Forecast (Log-Transformed, Next 15 Days)",
                               yaxis_title='Price (USD)',
                               xaxis_title='Date')
            st.plotly_chart(fig1)

            st.dataframe(forecast[['ds', 'yhat_exp', 'yhat_lower_exp', 'yhat_upper_exp']].tail(10), use_container_width=True)
            st.download_button("📥 Download Prophet Forecast", forecast.to_csv(index=False), file_name=f"{symbol}_prophet_forecast.csv")

        # --- LSTM Forecast ---
        st.subheader("🤖 LSTM Future Price Prediction")

        try:
            if df.shape[0] < 50:
                raise ValueError("Not enough data points for LSTM prediction (need at least 50).")

            seq_len = min(60, df.shape[0] - 1)

            # Compute log returns
            df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
            df.dropna(inplace=True)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_returns = scaler.fit_transform(df[['LogReturn']])

            X, y = [], []
            for i in range(seq_len, len(scaled_returns)):
                X.append(scaled_returns[i - seq_len:i])
                y.append(scaled_returns[i])

            X = np.array(X)
            y = np.array(y)

            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

            future_input = X[-1].reshape(1, X.shape[1], 1)
            future_preds_scaled = []

            for _ in range(10):
                pred_scaled = model.predict(future_input, verbose=0)[0][0]
                pred_scaled = np.clip(pred_scaled, 0, 1)  # Strict clip to [0,1]
                future_preds_scaled.append(pred_scaled)
                pred_array = np.array([[[pred_scaled]]], dtype=np.float32)
                future_input = np.concatenate((future_input[:, 1:, :], pred_array), axis=1)

            future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

            last_close = df['Close'].iloc[-1]
            future_prices = []
            current_price = last_close

            for r in future_preds:
                next_price = current_price * np.exp(r)
                if next_price < last_close * 0.9:
                    next_price = last_close * 0.9
                future_prices.append(next_price)
                current_price = next_price

            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')

            df_future = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close': future_prices
            })

            st.write("Debug: Predicted log returns (unscaled):", future_preds)
            st.write("Debug: Predicted future prices:", future_prices)

            min_price = min(df['Close'].min(), df_future['Predicted Close'].min())
            yaxis_min = max(min_price * 0.95, 0)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index.to_list(), y=df['Close'].values.flatten().tolist(), name="Historical"))
            fig2.add_trace(go.Scatter(x=df_future['Date'].to_list(), y=df_future['Predicted Close'].to_list(),
                                      name="LSTM Forecast", line=dict(dash='dot')))
            fig2.update_layout(
                title=f"{symbol} — Combined Forecast View",
                yaxis=dict(range=[yaxis_min, None]),
                xaxis_title="Date",
                yaxis_title="Price (USD)"
            )
            st.plotly_chart(fig2)

            st.dataframe(df_future, use_container_width=True)
            st.download_button("📥 Download LSTM Forecast", df_future.to_csv(index=False), file_name=f"{symbol}_lstm_forecast.csv")

        except Exception as e:
            st.error(f"❌ LSTM Error: {e}")
            st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.text(traceback.format_exc())
