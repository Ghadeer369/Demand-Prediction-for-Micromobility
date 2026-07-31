"""Run all three models on the identical held-out window and build one
comparison table — plus a naive baseline, because a model's MAE only
means something in contrast to "what if we barely tried."
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import arima_model, config, lstm_model, xgboost_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def naive_baseline_metrics() -> dict:
    """Predict 'same value as the same weekday last week' — the simplest
    forecast that still respects weekly seasonality. Any real model should
    beat this by a healthy margin, or it isn't earning its complexity."""
    df = xgboost_model.build_features(xgboost_model.load_daily_demand())
    _, test = xgboost_model.chronological_split(df)
    preds = test["Ride_Count_Lag7"]
    actual = test["Ride_Count"]
    return {
        "MAE": mean_absolute_error(actual, preds),
        "RMSE": mean_squared_error(actual, preds) ** 0.5,
        "R2": r2_score(actual, preds),
    }


def build_comparison_table() -> pd.DataFrame:
    logger.info("Training XGBoost...")
    xgb_metrics = xgboost_model.run()

    logger.info("Training SARIMA...")
    arima_metrics = arima_model.run()

    logger.info("Training LSTM...")
    lstm_metrics = lstm_model.run()

    logger.info("Computing naive baseline...")
    naive_metrics = naive_baseline_metrics()

    table = pd.DataFrame({
        "XGBoost": xgb_metrics,
        "SARIMA": arima_metrics,
        "LSTM": lstm_metrics,
        "Naive (last week)": naive_metrics,
    }).T
    table = table.sort_values("MAE")
    return table


def plot_comparison(table: pd.DataFrame) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.index, table["MAE"], color=config.BRAND_PRIMARY)
    ax.set_title(f"Model Comparison — MAE on the last {config.TEST_DAYS} days", color=config.BRAND_TEXT)
    ax.set_ylabel("MAE (rides/day)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison_mae.png", dpi=150)
    plt.close(fig)


def run() -> pd.DataFrame:
    table = build_comparison_table()
    logger.info("\n%s", table.to_string())

    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.PREDICTIONS_DIR / "model_comparison.csv")
    plot_comparison(table)

    best_model = table["MAE"].idxmin()
    logger.info("Best model by MAE: %s", best_model)
    return table


if __name__ == "__main__":
    run()
