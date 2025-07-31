import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go
import traceback

st.set_page_config(page_title="Sipre Pro", layout="wide")
st.title("📈 Sipre: Smart Investment Predictor")

# --- Sidebar ---
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., TSLA)", value="TSLA")
date_range = st.sidebar.selectbox("Date Range", ["1y", "2y", "5y", "10y"], index=0)
predict_days = st.sidebar.slider("LSTM: Days to Predict", min_value=7, max_value=90, value=15)

try:
    df = yf.download(symbol, period=date_range)
    if df.empty:
        st.error("No data found. Please check the symbol and try again.")
        st.stop()
except Exception as e:
    st.error(f"Data retrieval error: {e}")
    st.stop()

st.subheader(f"📊 Historical Data: {symbol}")
st.line_chart(df['Close'])

# --- Prophet Forecast ---
st.subheader("📅 Prophet Forecast (Next 15 Days)")

try:
    df_reset = df.reset_index()
    safe_min_price = max(1.0, df_reset['Close'].min())
    prices_clipped = df_reset['Close'].clip(lower=safe_min_price)

    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df_reset[df_reset.columns[0]]),
        'y': np.log(prices_clipped)
    }).dropna()

    if prophet_df.shape[0] < 30:
        st.warning("Not enough data for Prophet forecasting.")
    else:
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=15)
        forecast = m.predict(future)

        min_log_price = np.log(safe_min_price)
        forecast['yhat'] = forecast['yhat'].clip(lower=min_log_price)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=min_log_price)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=min_log_price)

        forecast['yhat_exp'] = np.exp(forecast['yhat'])
        forecast['yhat_lower_exp'] = np.exp(forecast['yhat_lower'])
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
        fig1.update_layout(title=f"{symbol} Prophet Forecast (Next 15 Days)",
                           yaxis_title='Price (USD)', xaxis_title='Date')
        st.plotly_chart(fig1)

        st.dataframe(forecast[['ds', 'yhat_exp', 'yhat_lower_exp', 'yhat_upper_exp']].tail(10), use_container_width=True)

except Exception as e:
    st.error(f"❌ Prophet Error: {e}")
    st.text(traceback.format_exc())

# --- LSTM Forecast ---
st.subheader("🤖 LSTM Future Price Prediction")

try:
    close_data = df[['Close']].values

    if len(close_data) < 70:
        st.warning("LSTM Skipped: Not enough data points for LSTM prediction (need at least ~70).")
    else:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        X = []
        y = []
        sequence_length = 60

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        test_input = scaled_data[-sequence_length:]
        future_prices = []

        for _ in range(predict_days):
            test_seq = np.reshape(test_input, (1, sequence_length, 1))
            pred = model.predict(test_seq, verbose=0)[0][0]
            future_prices.append(pred)
            test_input = np.append(test_input[1:], [[pred]], axis=0)

        future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1)).flatten()
        last_close = df['Close'].iloc[-1]
        price_floor = last_close * 0.9
        price_ceiling = last_close * 1.1
        clipped_prices = np.clip(future_prices, price_floor, price_ceiling)

        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=predict_days)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical"))
        fig2.add_trace(go.Scatter(x=future_dates, y=clipped_prices, name="LSTM Forecast"))
        fig2.update_layout(title=f"{symbol} LSTM Forecast (Next {predict_days} Days)",
                           yaxis_title='Price (USD)', xaxis_title='Date')
        st.plotly_chart(fig2)

        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": clipped_prices})
        st.dataframe(forecast_df, use_container_width=True)

except Exception as e:
    st.error(f"❌ LSTM Error: {e}")
    st.text(traceback.format_exc())
