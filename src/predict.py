"""Generate a future demand forecast using the best model from Day 10 (LSTM).

LSTM only predicts one day at a time from a 14-day window, so forecasting
further ahead means feeding its own predictions back in as input — a
"recursive" forecast. Errors can compound the further out we go, which is
exactly why we cap this at a couple of weeks rather than months.
"""

import logging

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src import config
from src.lstm_model import WINDOW_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FORECAST_DAYS = 14


def load_model_and_scaler():
    model = tf.keras.models.load_model(config.MODELS_DIR / "lstm_model.keras")
    scaler = joblib.load(config.MODELS_DIR / "lstm_scaler.pkl")
    return model, scaler


def recursive_forecast(model, scaler, last_known_values: np.ndarray, n_days: int) -> np.ndarray:
    """Predict n_days ahead, one day at a time, feeding each prediction
    back in as if it were an observed value for the next step."""
    window = scaler.transform(last_known_values.reshape(-1, 1)).flatten().tolist()
    predictions_scaled = []

    for _ in range(n_days):
        x = np.array(window[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
        next_scaled = model.predict(x, verbose=0)[0, 0]
        predictions_scaled.append(next_scaled)
        window.append(next_scaled)

    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    return np.clip(predictions, a_min=0, a_max=None)  # demand can't be negative


def build_forecast() -> pd.DataFrame:
    daily = pd.read_csv(config.PROCESSED_DATA_DIR / "daily_demand.csv", parse_dates=[config.COL_RIDE_START_DATE])
    daily = daily.sort_values(config.COL_RIDE_START_DATE)

    model, scaler = load_model_and_scaler()
    last_known = daily["Ride_Count"].values[-WINDOW_SIZE:]
    forecast_values = recursive_forecast(model, scaler, last_known, FORECAST_DAYS)

    last_date = daily[config.COL_RIDE_START_DATE].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")

    forecast_df = pd.DataFrame({
        config.COL_RIDE_START_DATE: future_dates,
        "Predicted_Ride_Count": forecast_values.round().astype(int),
    })
    logger.info("Forecast for %d days ahead:\n%s", FORECAST_DAYS, forecast_df.to_string(index=False))
    return forecast_df


def run() -> pd.DataFrame:
    forecast_df = build_forecast()
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PREDICTIONS_DIR / "future_predictions.csv"
    forecast_df.to_csv(out_path, index=False)
    logger.info("Saved forecast to %s", out_path.name)
    return forecast_df


if __name__ == "__main__":
    run()
