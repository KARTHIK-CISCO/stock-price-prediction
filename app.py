
import streamlit as st
import pandas as pd
import numpy as np

import os
import joblib  # For model loading

# --- Configuration ---
PROCESSED_DATA_PATH = "Processed_AAPL.csv"
MODEL_PATH = "models/best_xgb.pkl"   # Use your saved tuned model

# --- Streamlit UI Setup ---
st.title("📈 AAPL Stock Price Forecasting App")
st.write("Forecast future stock prices using a trained ML model.")

# --- Data Loading ---
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Processed data file '{path}' not found.")
        st.stop()
    df_loaded = pd.read_csv(path)
    df_loaded['Date'] = pd.to_datetime(df_loaded['Date'])
    df_loaded = df_loaded.sort_values('Date').reset_index(drop=True)
    return df_loaded

df_loaded = load_data(PROCESSED_DATA_PATH)

# Create next-day target
df_loaded['target_next_close'] = df_loaded['Close'].shift(-1)
df_loaded.dropna(inplace=True)

# Feature selection
exclude_cols = ['Date', 'target_next_close']
feature_cols = [col for col in df_loaded.columns if col not in exclude_cols and np.issubdtype(df_loaded[col].dtype, np.number)]

X_app = df_loaded[feature_cols]
y_app = df_loaded['target_next_close']


# --- Load Tuned Model ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file '{path}' not found. Please upload the trained model.")
        st.stop()
    return joblib.load(path)

model = load_model(MODEL_PATH)


# --- User Input ---
forecast_days = st.slider("Select number of days to forecast:", 1, 30, 7)

# --- Forecasting ---
last_row = X_app.iloc[-1:].copy()
future_predictions = []

for _ in range(forecast_days):
    pred = model.predict(last_row)[0]
    future_predictions.append(pred)

    # Update "Close" feature for recursive forecast
    last_row.loc[last_row.index[0], 'Close'] = pred

# --- Display Results ---
st.subheader("📊 Forecasted Prices (Normalized)")

if len(future_predictions) > 0:
    last_date = df_loaded['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=forecast_days, freq='D')
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_predictions})
    st.dataframe(forecast_df.set_index('Date'))
else:
    st.write("No predictions generated.")

# --- Plot Forecast ---
fig, ax = plt.subplots(figsize=(10,5))

# Plot historical close
ax.plot(df_loaded['Date'].tail(60), df_loaded['Close'].tail(60),
        label="Historical Close", linestyle='--', color='gray')

# Plot forecast
ax.plot(future_dates, future_predictions, marker='o',
        label="Forecasted Close", color='blue')

ax.set_title("AAPL Stock Price Forecast (Normalized)")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Normalized Close Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)
