import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor # Use XGBoost as it was tuned
import matplotlib.pyplot as plt
import os

# --- Configuration ---
PROCESSED_DATA_PATH = "Processed_AAPL.csv"

# --- Streamlit UI Setup ---
st.title("📈 AAPL Stock Price Forecasting App")
st.write("Forecast future stock prices using a trained ML model.")

# --- Data Loading and Preparation ---
@st.cache_data # Cache data loading for performance
def load_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Processed data file '{path}' not found. Please ensure it is generated.")
        st.stop()
    df_loaded = pd.read_csv(path)
    df_loaded['Date'] = pd.to_datetime(df_loaded['Date'])
    df_loaded = df_loaded.sort_values('Date').reset_index(drop=True)
    return df_loaded

df_loaded = load_data(PROCESSED_DATA_PATH)

# Create the target variable: next day's closing price
df_loaded['target_next_close'] = df_loaded['Close'].shift(-1)
df_loaded.dropna(inplace=True)

# Define features and target
exclude_cols_for_features = ['Date', 'target_next_close']
feature_cols_for_app = [col for col in df_loaded.columns if col not in exclude_cols_for_features and np.issubdtype(df_loaded[col].dtype, np.number)]

X_app = df_loaded[feature_cols_for_app]
y_app = df_loaded['target_next_close']

# --- Model Training ---
@st.cache_resource # Cache the model to avoid re-training on every rerun
def train_model(X, y):
    # Using parameters close to a common tuned result, or what was intended for best_xgb
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X, y)
    return model

model = train_model(X_app, y_app)

# --- User Input ---
forecast_days = st.slider("Select number of days to forecast:", 1, 30, 7)

# --- Forecasting Logic ---
last_features_row_df = X_app.iloc[-1:].copy() # Get the last row of features as a DataFrame
future_predictions = []

for _ in range(forecast_days):
    # Predict the next day's close price (which is scaled)
    next_day_close_prediction = model.predict(last_features_row_df)[0]
    future_predictions.append(next_day_close_prediction)

    # Prepare features for the next prediction step (recursive forecasting)
    # Update 'Close' feature with the scaled prediction.
    # For simplicity in recursive forecasting, other features will carry over their last known values.
    # This is a strong simplification for features like MAs/Volatility/Open/High/Low/Adj Close
    # and could limit long-term forecast accuracy, but ensures the app runs without complex re-calculations.
    last_features_row_df.loc[last_features_row_df.index[0], 'Close'] = next_day_close_prediction

# --- Display Results ---
st.subheader("📊 Forecasted Prices (Normalized)")
# Create a DataFrame for display with proper date indexing if desired, or just list
if len(future_predictions) > 0:
    # Generate future dates starting from the day after the last date in df_loaded
    last_date = df_loaded['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast_df_display = pd.DataFrame({"Date": future_dates, "Predicted Close": future_predictions})
    st.dataframe(forecast_df_display.set_index('Date'))
else:
    st.write("No predictions to display.")

# --- Plot Forecast ---
fig, ax = plt.subplots(figsize=(10,5))
# Plot historical 'Close' data if desired for context
ax.plot(df_loaded['Date'].tail(60), df_loaded['Close'].tail(60), label='Historical Close', color='gray', linestyle='--')

# Plot forecasted prices
ax.plot(future_dates, future_predictions, marker='o', linestyle='-', color='blue', label='Forecasted Close')

ax.set_title("AAPL Stock Price Forecast (Normalized Close)")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Normalized Close Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)
