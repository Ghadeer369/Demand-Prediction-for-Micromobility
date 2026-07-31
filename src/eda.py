"""Exploratory Data Analysis on the daily demand series: summary stats,
the weekend effect, and a first-look trend chart. Polished, presentation-
ready charts come later in visualization.py — this file is for building
our own understanding of the data before choosing a model.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_daily_demand() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DATA_DIR / "daily_demand.csv", parse_dates=[config.COL_RIDE_START_DATE])
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print summary statistics and flag skew (mean vs. median gap)."""
    stats = df["Ride_Count"].describe()
    logger.info("Ride_Count summary:\n%s", stats)

    mean, median = df["Ride_Count"].mean(), df["Ride_Count"].median()
    skew_pct = (mean - median) / median * 100
    logger.info("Mean (%.1f) vs median (%.1f): %.0f%% higher — a few very high-demand days pull the mean up",
                mean, median, skew_pct)


def weekend_effect(df: pd.DataFrame) -> pd.Series:
    """Average Ride_Count on weekend days vs weekdays."""
    result = df.groupby("Is_Weekend")["Ride_Count"].mean()
    logger.info("Average daily rides — weekday: %.0f, weekend: %.0f", result[False], result[True])
    return result


def plot_daily_trend(df: pd.DataFrame) -> None:
    """Save a first-look line chart of daily ride count over the full period."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[config.COL_RIDE_START_DATE], df["Ride_Count"], linewidth=0.8, color=config.BRAND_PRIMARY)
    ax.set_title("Daily Ride Count Over Time", color=config.BRAND_TEXT)
    ax.set_xlabel("Date")
    ax.set_ylabel("Ride Count")
    fig.tight_layout()

    out_path = config.FIGURES_DIR / "eda_daily_trend.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved trend chart to %s", out_path.name)


def run_eda() -> None:
    df = load_daily_demand()
    summarize(df)
    weekend_effect(df)
    plot_daily_trend(df)


if __name__ == "__main__":
    run_eda()
