import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

# --- Config ---
PROCESSED_DATA_PATH = "Processed_AAPL.csv"

# --- UI ---
st.title("📈 AAPL Stock Price Forecasting App")
st.write("Forecast future stock prices using Random Forest (Streamlit Cloud Safe Version).")

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

X = df[feature_cols]
y = df["target_next_close"]

# --- Train Model ---
@st.cache_resource
def train_rf(X, y):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

model = train_rf(X, y)

# --- Forecast ---
forecast_days = st.slider("Forecast Next Days:", 1, 30, 7)

last_row = X.iloc[-1:].copy()
future_predictions = []

for _ in range(forecast_days):
    pred = model.predict(last_row)[0]
    future_predictions.append(pred)
    last_row.loc[last_row.index[0], "Close"] = pred  # Recursive forecasting

# --- Display Results ---
st.subheader("📊 Forecasted Normalized Prices")

future_dates = pd.date_range(
    df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close": future_predictions
})

st.dataframe(forecast_df.set_index("Date"))

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df["Date"].tail(60), df["Close"].tail(60),
        label="Historical Close", linestyle="--", color="gray")

ax.plot(future_dates, future_predictions, marker="o",
        label="Forecasted Close", color="blue")

ax.set_title("AAPL Stock Price Forecast (Normalized Close Price)")
ax.set_xlabel("Date")
ax.set_ylabel("Normalized Price")
ax.legend()
ax.grid(True)

st.pyplot(fig)
