"""Presentation-ready charts for the README and dashboard: weekday demand,
monthly seasonality, and a smoothed trend line. Each function saves one
figure to outputs/figures/ and returns nothing — these are for humans,
not for feeding back into the pipeline.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEEKDAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def load_daily_demand() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DATA_DIR / "daily_demand.csv", parse_dates=[config.COL_RIDE_START_DATE])


def _save(fig, filename: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", filename)


def plot_weekday_comparison(df: pd.DataFrame) -> None:
    """Average ride count per weekday, ordered Sunday->Saturday (the Saudi week)."""
    avg_by_day = df.groupby(config.COL_DAY_OF_WEEK)["Ride_Count"].mean().reindex(WEEKDAY_ORDER)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [config.BRAND_PRIMARY if day not in {"Friday", "Saturday"} else config.BRAND_ACCENT for day in WEEKDAY_ORDER]
    ax.bar(avg_by_day.index, avg_by_day.values, color=colors)
    ax.set_title("Average Daily Rides by Weekday (lavender = weekend)", color=config.BRAND_TEXT)
    ax.set_ylabel("Average Ride Count")
    ax.tick_params(axis="x", rotation=30)

    _save(fig, "weekday_avg_demand.png")


def plot_monthly_seasonality(df: pd.DataFrame) -> None:
    """Average ride count per calendar month, pooled across all years."""
    month_names = df[config.COL_RIDE_START_DATE].dt.month_name()
    avg_by_month = df.groupby(month_names)["Ride_Count"].mean().reindex(MONTH_ORDER)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(avg_by_month.index, avg_by_month.values, color=config.BRAND_PRIMARY)
    ax.set_title("Average Daily Rides by Month (pooled across 2021-2023)", color=config.BRAND_TEXT)
    ax.set_ylabel("Average Ride Count")
    ax.tick_params(axis="x", rotation=45)

    _save(fig, "monthly_avg_demand.png")


def plot_rolling_average(df: pd.DataFrame, window: int = 7) -> None:
    """Raw daily series vs. a rolling mean — the smoothed line makes the
    underlying trend visible through the day-to-day noise."""
    rolling = df["Ride_Count"].rolling(window=window, center=True).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[config.COL_RIDE_START_DATE], df["Ride_Count"], linewidth=0.5, alpha=0.4,
            color=config.BRAND_TEXT, label="Daily")
    ax.plot(df[config.COL_RIDE_START_DATE], rolling, linewidth=1.8, color=config.BRAND_PRIMARY,
            label=f"{window}-day rolling average")
    ax.set_title("Daily Ride Count with Rolling Average", color=config.BRAND_TEXT)
    ax.set_xlabel("Date")
    ax.set_ylabel("Ride Count")
    ax.legend()

    _save(fig, "rolling_avg_trend.png")


def build_all_visuals() -> None:
    df = load_daily_demand()
    plot_weekday_comparison(df)
    plot_monthly_seasonality(df)
    plot_rolling_average(df)


if __name__ == "__main__":
    build_all_visuals()
