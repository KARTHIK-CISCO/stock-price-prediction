import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Config ---
PROCESSED_DATA_PATH = "Processed_AAPL.csv"
RAW_DATA_PATH = "AAPL.csv"    # raw data with real prices

# --- UI ---
st.title("📈 AAPL Stock Price Forecasting App")
st.write("Forecast future REAL stock prices using Pure NumPy Regression.")

# --- Load Data ---
@st.cache_data
def load_processed(path):
    if not os.path.exists(path):
        st.error(f" '{path}' missing. Upload Processed_AAPL.csv")
        st.stop()
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_raw(path):
    if not os.path.exists(path):
        st.error(f" Raw file '{path}' missing. Upload AAPL.csv")
        st.stop()
    df = pd.read_csv(path)
    return df

df_norm = load_processed(PROCESSED_DATA_PATH)
df_raw = load_raw(RAW_DATA_PATH)

# --- Compute real price min/max for inverse scaling ---
real_min = df_raw["Close"].min()
real_max = df_raw["Close"].max()

# --- Prepare features ---
df_norm = df_norm.sort_values("Date").reset_index(drop=True)
df_norm["target_next_close"] = df_norm["Close"].shift(-1)
df_norm.dropna(inplace=True)

exclude_cols = ["Date", "target_next_close"]
feature_cols = [
    col for col in df_norm.columns
    if col not in exclude_cols and np.issubdtype(df_norm[col].dtype, np.number)
]

X = df_norm[feature_cols].values
y = df_norm["target_next_close"].values

# Add bias
Xb = np.hstack([np.ones((X.shape[0], 1)), X])

# --- Train numpy regression model ---
@st.cache_resource
def train_numpy(Xb, y):
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return w

weights = train_numpy(Xb, y)

# --- Forecast ---
forecast_days = st.slider("Forecast Next Days:", 1, 30, 7)

last_row = X[-1:].copy()
future_norm = []
future_real = []

for _ in range(forecast_days):
    xb = np.hstack([1, last_row.flatten()])
    pred_norm = xb @ weights
    future_norm.append(pred_norm)

    # Convert normalized --> real price
    pred_real = pred_norm * (real_max - real_min) + real_min
    future_real.append(pred_real)

    last_row[0][0] = pred_norm

# --- Show REAL prices ---
st.subheader(" Forecasted REAL Prices (USD)")

future_dates = pd.date_range(
    df_norm["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price (USD)": future_real
})

st.dataframe(forecast_df.set_index("Date"))

# --- Plot REAL prices ---
st.subheader(" Forecast Chart (Real Price)")

hist_dates = df_norm["Date"].tail(60).reset_index(drop=True)
hist_real = (df_norm["Close"].tail(60) * (real_max - real_min) + real_min).reset_index(drop=True)

combined_df = pd.DataFrame({
    "Date": pd.concat([hist_dates, forecast_df["Date"]], ignore_index=True),
    "Actual Price": pd.concat([hist_real, pd.Series([None] * forecast_days)], ignore_index=True),
    "Predicted Price": pd.concat([pd.Series([None] * 60), pd.Series(future_real)], ignore_index=True)
}).set_index("Date")

st.line_chart(combined_df)
