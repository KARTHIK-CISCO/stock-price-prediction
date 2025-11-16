import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Config ---
PROCESSED_DATA_PATH = "Processed_AAPL.csv"

# --- UI ---
st.title("📈 AAPL Stock Price Forecasting App")
st.write("Forecast future stock prices using Pure NumPy Linear Regression (Cloud-Safe).")

# --- Load Data ---
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        st.error(f"❌ '{path}' missing. Upload Processed_AAPL.csv")
        st.stop()
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data(PROCESSED_DATA_PATH)

# --- Feature Engineering ---
df["target_next_close"] = df["Close"].shift(-1)
df.dropna(inplace=True)

exclude_cols = ["Date", "target_next_close"]
feature_cols = [
    col for col in df.columns
    if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)
]

X = df[feature_cols].values
y = df["target_next_close"].values

# Add intercept term
X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

# --- Train Pure NumPy Linear Regression ---
@st.cache_resource
def train_numpy_model(Xb, y):
    # Compute weights using least squares
    weights, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return weights

weights = train_numpy_model(X_bias, y)

# --- Forecast ---
forecast_days = st.slider("Forecast Next Days:", 1, 30, 7)

last_row = X[-1:].copy()
future_predictions = []

for _ in range(forecast_days):
    last_row_bias = np.hstack([1, last_row.flatten()])
    pred = last_row_bias @ weights
    future_predictions.append(pred)

    # Update "Close" recursively
    last_row[0][0] = pred

# --- Display ---
st.subheader("📊 Forecasted Prices")

future_dates = pd.date_range(
    df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close": future_predictions
})

st.dataframe(forecast_df.set_index("Date"))

# --- Plot using Streamlit ---
st.subheader("📈 Forecast Chart")

combined_df = pd.DataFrame({
    "Date": pd.concat([df["Date"].tail(60), forecast_df["Date"]]),
    "Close": pd.concat([df["Close"].tail(60), pd.Series([None] * forecast_days)]),
    "Predicted Close": pd.concat([pd.Series([None] * 60), forecast_df["Predicted Close"]])
}).set_index("Date")

st.line_chart(combined_df)
