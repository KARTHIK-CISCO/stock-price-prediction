import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import os

# --- Configuration ---
PROCESSED_DATA_PATH = "Processed_AAPL.csv"

# --- Streamlit UI ---
st.title("📈 AAPL Stock Price Forecasting App")
st.write("Forecast future stock prices using an automatically trained ML model.")

# --- Load Data ---
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Processed data file '{path}' not found.")
        st.stop()
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data(PROCESSED_DATA_PATH)

# --- Prepare Features ---
df["target_next_close"] = df["Close"].shift(-1)
df.dropna(inplace=True)

exclude_cols = ["Date", "target_next_close"]
feature_cols = [
    col for col in df.columns
    if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)
]

X = df[feature_cols]
y = df["target_next_close"]

# --- Train Model Inside App ---
@st.cache_resource
def train_model(X, y):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X, y)
    return model

model = train_model(X, y)

# --- User Input ---
forecast_days = st.slider("Select number of days to forecast:", 1, 30, 7)

# --- Forecasting ---
last_row = X.iloc[-1:].copy()
future_predictions = []

for _ in range(forecast_days):
    pred = model.predict(last_row)[0]
    future_predictions.append(pred)
    last_row.loc[last_row.index[0], "Close"] = pred  # Recursive update

# --- Display Results ---
st.subheader("📊 Forecasted Prices (Normalized)")

last_date = df["Date"].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                              periods=forecast_days)

forecast_df = pd.DataFrame(
    {"Date": future_dates, "Predicted Close": future_predictions}
)

st.dataframe(forecast_df.set_index("Date"))

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df["Date"].tail(60), df["Close"].tail(60),
        label="Historical Close", linestyle="--", color="gray")

ax.plot(future_dates, future_predictions, marker="o",
        label="Forecasted Close", color="blue")

ax.set_title("AAPL Stock Price Forecast (Normalized)")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Normalized Close Price")
ax.legend()
ax.grid(True)

st.pyplot(fig)
