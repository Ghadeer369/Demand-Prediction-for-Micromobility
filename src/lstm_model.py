"""Train an LSTM to predict daily Ride_Count from the previous 14 days.

LSTM (Long Short-Term Memory) is a recurrent neural network: it reads a
*sequence* of past values one step at a time and keeps an internal memory,
so — unlike XGBoost — it can in principle learn how much recent history
matters on its own. It needs scaled inputs (neural nets train badly on
raw values like "0 to 1271") and a 3D input shape: (samples, timesteps,
features).
"""

import logging

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET = "Ride_Count"
WINDOW_SIZE = 14  # predict day t from the previous 14 days


def load_daily_demand() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DATA_DIR / "daily_demand.csv", parse_dates=[config.COL_RIDE_START_DATE])
    return df.sort_values(config.COL_RIDE_START_DATE).reset_index(drop=True)


def make_sequences(scaled_values: np.ndarray, window_size: int):
    """Turn a 1D array into (X, y) sliding-window pairs:
    X[i] = values[i : i+window_size], y[i] = values[i+window_size]."""
    X, y = [], []
    for i in range(len(scaled_values) - window_size):
        X.append(scaled_values[i:i + window_size])
        y.append(scaled_values[i + window_size])
    return np.array(X), np.array(y)


def prepare_data(df: pd.DataFrame):
    """Fit the scaler on the TRAINING portion only (never let the test
    period influence the scaling), then build sequences from the full
    scaled series so test sequences can still see recent history."""
    n_test = config.TEST_DAYS
    train_values = df[TARGET].values[:-n_test].reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.fit(train_values)
    scaled_all = scaler.transform(df[TARGET].values.reshape(-1, 1)).flatten()

    X, y = make_sequences(scaled_all, WINDOW_SIZE)
    # y[i] corresponds to day (WINDOW_SIZE + i) in the original series
    n_test_sequences = n_test
    X_train, X_test = X[:-n_test_sequences], X[-n_test_sequences:]
    y_train, y_test = y[:-n_test_sequences], y[-n_test_sequences:]

    # LSTM expects 3D input: (samples, timesteps, features)
    X_train = X_train.reshape(-1, WINDOW_SIZE, 1)
    X_test = X_test.reshape(-1, WINDOW_SIZE, 1)

    logger.info("Train sequences: %d | Test sequences: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, scaler


def build_lstm() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def quick_evaluate(model, X_test, y_test, scaler) -> dict:
    scaled_preds = model.predict(X_test, verbose=0).flatten()
    # inverse_transform to get back to real ride counts, not the 0-1 scaled values
    preds = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = {
        "MAE": mean_absolute_error(actuals, preds),
        "RMSE": mean_squared_error(actuals, preds) ** 0.5,
        "R2": r2_score(actuals, preds),
    }
    logger.info("LSTM test metrics: MAE=%.1f  RMSE=%.1f  R2=%.3f", metrics["MAE"], metrics["RMSE"], metrics["R2"])
    return metrics


def run() -> dict:
    tf.random.set_seed(config.RANDOM_SEED)
    df = load_daily_demand()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    model = build_lstm()
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0,
              validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])

    metrics = quick_evaluate(model, X_test, y_test, scaler)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODELS_DIR / "lstm_model.keras")
    joblib.dump(scaler, config.MODELS_DIR / "lstm_scaler.pkl")
    logger.info("Saved model and scaler to %s", config.MODELS_DIR)
    return metrics


if __name__ == "__main__":
    run()
