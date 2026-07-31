"""Train a SARIMA model on daily Ride_Count.

Unlike XGBoost, SARIMA understands time directly: it models a value as a
function of its own past values (AR), past forecast errors (MA), and — via
the seasonal terms — a repeating weekly pattern. We confirmed with an
Augmented Dickey-Fuller test that the raw series is non-stationary
(p=0.46) but becomes stationary after one differencing (p<0.0001), which
is why d=1 below.
"""

import logging

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config, utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET = "Ride_Count"
ORDER = (1, 1, 1)            # (p, d, q): AR order, differencing, MA order
SEASONAL_ORDER = (1, 1, 1, 7)  # same, plus a 7-day seasonal cycle (the weekly pattern from Day 5/6)


def load_daily_demand() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DATA_DIR / "daily_demand.csv", parse_dates=[config.COL_RIDE_START_DATE])
    return df.sort_values(config.COL_RIDE_START_DATE).reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    train, test = utils.chronological_split_by_days(df, config.COL_RIDE_START_DATE, config.TEST_DAYS)
    logger.info("Train: %d days | Test: %d days", len(train), len(test))
    return train, test


def train_sarima(train_series: pd.Series) -> SARIMAX:
    model = SARIMAX(train_series, order=ORDER, seasonal_order=SEASONAL_ORDER,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    return fitted


def quick_evaluate(fitted, test: pd.DataFrame) -> dict:
    preds = fitted.forecast(steps=len(test))
    metrics = {
        "MAE": mean_absolute_error(test[TARGET], preds),
        "RMSE": mean_squared_error(test[TARGET], preds) ** 0.5,
        "R2": r2_score(test[TARGET], preds),
    }
    logger.info("SARIMA test metrics: MAE=%.1f  RMSE=%.1f  R2=%.3f", metrics["MAE"], metrics["RMSE"], metrics["R2"])
    return metrics


def run() -> dict:
    df = load_daily_demand()
    train, test = chronological_split(df)

    fitted = train_sarima(train[TARGET])
    metrics = quick_evaluate(fitted, test)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "arima_model.pkl"
    joblib.dump(fitted, model_path)
    logger.info("Saved model to %s", model_path.name)
    return metrics


if __name__ == "__main__":
    run()
