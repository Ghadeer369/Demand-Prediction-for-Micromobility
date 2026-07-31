"""Clean the raw RidesDetails export into an analysis-ready CSV.

Pipeline: read raw file -> drop Customer -> split date/time -> fix the one
missing Total_Time -> remove Total_Time outliers (IQR) -> save to data/processed/.
"""

import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(path=config.RAW_RIDES_FILE) -> pd.DataFrame:
    """Read the raw CSV with the encoding/separator RidesDetails.csv actually uses."""
    df = pd.read_csv(path, sep=config.RAW_CSV_SEPARATOR, encoding=config.RAW_CSV_ENCODING)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # drop the trailing empty column
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path.name)
    return df


def drop_customer_column(df: pd.DataFrame) -> pd.DataFrame:
    """Customer names are inconsistent (mixed language, first-name-only) and
    are not needed to predict aggregate demand, so we drop the column."""
    return df.drop(columns=[config.COL_CUSTOMER])


def split_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ride_Start_Time / Ride_End_Time hold date+time together (e.g. "03/11/21 21:22").
    Parse them once, then derive separate date, hour, weekday, and time-of-day columns."""
    df = df.copy()
    start = pd.to_datetime(df[config.COL_RIDE_START_TIME], format="%d/%m/%y %H:%M")

    df[config.COL_RIDE_START_DATE] = start.dt.date
    df["Ride_Start_Hour"] = start.dt.hour
    df[config.COL_DAY_OF_WEEK] = start.dt.day_name()
    df[config.COL_TIMEZONE] = start.dt.hour.apply(_hour_to_timezone)
    return df


def _hour_to_timezone(hour: int) -> str:
    """Bucket an hour (0-23) into the same four periods used in the original report."""
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 20:
        return "Evening"
    return "Night"


def fill_missing_total_time(df: pd.DataFrame) -> pd.DataFrame:
    """The one row missing Total_Time gets it recomputed from start/end timestamps."""
    df = df.copy()
    missing_mask = df[config.COL_TOTAL_TIME].isna()
    if missing_mask.any():
        start = pd.to_datetime(df.loc[missing_mask, config.COL_RIDE_START_TIME], format="%d/%m/%y %H:%M")
        end = pd.to_datetime(df.loc[missing_mask, config.COL_RIDE_END_TIME], format="%d/%m/%y %H:%M")
        df.loc[missing_mask, config.COL_TOTAL_TIME] = (end - start).dt.total_seconds() / 60
        logger.info("Filled %d missing Total_Time value(s) from start/end timestamps", missing_mask.sum())
    return df


def remove_total_time_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where Total_Time falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1 = df[config.COL_TOTAL_TIME].quantile(0.25)
    q3 = df[config.COL_TOTAL_TIME].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    before = len(df)
    df = df[(df[config.COL_TOTAL_TIME] >= lower_bound) & (df[config.COL_TOTAL_TIME] <= upper_bound)]
    logger.info("Removed %d outlier rows (kept range %.2f–%.2f minutes)", before - len(df), lower_bound, upper_bound)
    return df


def preprocess() -> pd.DataFrame:
    """Run the full cleaning pipeline and save the result to data/processed/."""
    df = load_raw_data()
    df = drop_customer_column(df)
    df = split_datetime_columns(df)
    df = fill_missing_total_time(df)
    df = remove_total_time_outliers(df)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_RIDES_FILE, index=False)
    logger.info("Saved cleaned dataset to %s (%d rows)", config.PROCESSED_RIDES_FILE.name, len(df))
    return df


if __name__ == "__main__":
    preprocess()
