"""Train an XGBoost regressor to predict daily Ride_Count.

XGBoost can't see "time" on its own — it just sees rows of numbers — so the
model's whole job is done by the features we hand it: lag values, a rolling
average, and calendar signals (weekday, month, a day-index for trend).
"""

import logging

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config, utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET = "Ride_Count"


def load_daily_demand() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DATA_DIR / "daily_demand.csv", parse_dates=[config.COL_RIDE_START_DATE])
    return df.sort_values(config.COL_RIDE_START_DATE).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features. XGBoost has no concept of
    'yesterday' or 'this month' unless we compute it and hand it a column."""
    df = df.copy()

    df["Ride_Count_Lag1"] = df[TARGET].shift(1)                       # yesterday
    df["Ride_Count_Lag7"] = df[TARGET].shift(7)                       # same weekday last week
    df["Rolling_Mean_7"] = df[TARGET].shift(1).rolling(window=7).mean()  # past 7 days, excluding today

    df["Month"] = df[config.COL_RIDE_START_DATE].dt.month
    df = pd.get_dummies(df, columns=[config.COL_DAY_OF_WEEK], prefix="Day")

    before = len(df)
    df = df.dropna().reset_index(drop=True)  # first 7 rows have no lag/rolling history yet
    logger.info("Dropped %d row(s) with incomplete lag history", before - len(df))
    return df


def chronological_split(df: pd.DataFrame):
    """Split by time order, NOT randomly — we're simulating forecasting the
    future from the past, so the test set must be the most recent days.
    Uses config.TEST_DAYS so every model shares the exact same test window."""
    train, test = utils.chronological_split_by_days(df, config.COL_RIDE_START_DATE, config.TEST_DAYS)
    logger.info("Train: %d days (%s -> %s) | Test: %d days (%s -> %s)",
                len(train), train[config.COL_RIDE_START_DATE].min().date(), train[config.COL_RIDE_START_DATE].max().date(),
                len(test), test[config.COL_RIDE_START_DATE].min().date(), test[config.COL_RIDE_START_DATE].max().date())
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {TARGET, config.COL_RIDE_START_DATE, "Total_Time_Sum", "Is_Weekend"}
    return [col for col in df.columns if col not in exclude]


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=config.RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def quick_evaluate(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Sanity-check metrics for this model alone. The full cross-model
    comparison table is built later in evaluation.py."""
    preds = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        "R2": r2_score(y_test, preds),
    }
    logger.info("XGBoost test metrics: MAE=%.1f  RMSE=%.1f  R2=%.3f", metrics["MAE"], metrics["RMSE"], metrics["R2"])
    return metrics


def run() -> dict:
    df = build_features(load_daily_demand())
    train, test = chronological_split(df)
    feature_cols = get_feature_columns(df)

    model = train_xgboost(train[feature_cols], train[TARGET])
    metrics = quick_evaluate(model, test[feature_cols], test[TARGET])

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "xgboost_model.pkl"
    joblib.dump({"model": model, "feature_columns": feature_cols}, model_path)
    logger.info("Saved model to %s", model_path.name)
    return metrics


if __name__ == "__main__":
    run()
