import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from prophet import Prophet
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

st.title("📈 NIFTY50 Stock Forecasting App")

# Upload CSV
uploaded_file = st.file_uploader("Upload NIFTY50 CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    data = df[['Close']].dropna()
    data = data.tail(1000)

    st.subheader("Raw Data Preview")
    st.write(data.tail())

    # Plot raw data
    st.line_chart(data['Close'])

    # Model selection
    model_choice = st.selectbox("Select Model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])
    horizon = st.slider("Forecast Horizon (days)", 30, 365, 90)

    if st.button("Run Forecast"):
        if model_choice == "ARIMA":
            model = joblib.load("arima_model.pkl")
            forecast = model.forecast(steps=horizon)

            st.subheader("ARIMA Forecast")
            future_index = pd.date_range(data.index[-1], periods=horizon+1, freq="D")[1:]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index, data['Close'], label="Actual")
            ax.plot(future_index, forecast, label="Forecast", color="red")
            ax.legend()
            st.pyplot(fig)

        elif model_choice == "SARIMA":
            model = joblib.load("sarima_model.pkl")
            forecast = model.forecast(steps=horizon)

            st.subheader("SARIMA Forecast")
            future_index = pd.date_range(data.index[-1], periods=horizon+1, freq="D")[1:]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index, data['Close'], label="Actual")
            ax.plot(future_index, forecast, label="Forecast", color="red")
            ax.legend()
            st.pyplot(fig)

        elif model_choice == "Prophet":
            model = joblib.load("prophet_model.pkl")
            prophet_df = data.reset_index()[['Date', 'Close']]
            prophet_df.columns = ['ds', 'y']
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            st.subheader("Prophet Forecast")
            fig = model.plot(forecast)
            st.pyplot(fig)

        elif model_choice == "LSTM":
            model = keras.models.load_model("lstm_model.keras")
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data[['Close']])

            time_step = 60
            input_seq = scaled[-time_step:].reshape(1, time_step, 1)

            forecast = []
            for _ in range(horizon):
                    pred = model.predict(input_seq, verbose=0)        # shape (1,1)
                    pred = pred.reshape(1, 1, 1)                      # reshape to (1,1,1)
                    input_seq = np.append(input_seq[:, 1:, :], pred, axis=1)
                    forecast.append(pred[0, 0, 0])                    # save scalar
      # save scalar


            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

            st.subheader("LSTM Forecast")
            future_index = pd.date_range(data.index[-1], periods=horizon+1, freq="D")[1:]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index, data['Close'], label="Actual")
            ax.plot(future_index, forecast, label="Forecast", color="red")
            ax.legend()
            st.pyplot(fig)
