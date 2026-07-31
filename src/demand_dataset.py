"""Aggregate ride-level data into a daily demand time series — the table
every forecasting model (XGBoost, ARIMA, LSTM) will be trained on.

We keep two candidate demand targets:
  - Ride_Count: number of rides that day (the standard "demand" metric)
  - Total_Time_Sum: total ride minutes that day (matches the original report)
Both are cheap to compute now; we decide which one to model later.
"""

import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_daily_demand(df: pd.DataFrame = None) -> pd.DataFrame:
    """Group rides by day and compute daily demand metrics."""
    if df is None:
        df = pd.read_csv(config.PROCESSED_RIDES_FILE)

    df = df.copy()
    df[config.COL_RIDE_START_DATE] = pd.to_datetime(df[config.COL_RIDE_START_DATE])

    daily = (
        df.groupby(config.COL_RIDE_START_DATE)
        .agg(
            Ride_Count=(config.COL_RIDE_ID, "count"),
            Total_Time_Sum=(config.COL_TOTAL_TIME, "sum"),
        )
        .reset_index()
        .sort_values(config.COL_RIDE_START_DATE)
    )

    daily[config.COL_DAY_OF_WEEK] = daily[config.COL_RIDE_START_DATE].dt.day_name()
    daily["Is_Weekend"] = daily[config.COL_DAY_OF_WEEK].isin({"Friday", "Saturday"})

    _check_no_gaps(daily)
    return daily


def _check_no_gaps(daily: pd.DataFrame) -> None:
    """Forecasting models need a continuous daily series — fail loudly if a day is missing."""
    full_range = pd.date_range(daily[config.COL_RIDE_START_DATE].min(),
                                daily[config.COL_RIDE_START_DATE].max(), freq="D")
    missing = set(full_range) - set(daily[config.COL_RIDE_START_DATE])
    if missing:
        raise ValueError(f"{len(missing)} day(s) missing from the daily series: {sorted(missing)[:5]}...")
    logger.info("Daily series is continuous: %d days, no gaps", len(daily))


def build() -> pd.DataFrame:
    daily = build_daily_demand()
    daily_path = config.PROCESSED_DATA_DIR / "daily_demand.csv"
    daily.to_csv(daily_path, index=False)
    logger.info("Saved daily demand dataset to %s (%d rows)", daily_path.name, len(daily))
    return daily


if __name__ == "__main__":
    build()
