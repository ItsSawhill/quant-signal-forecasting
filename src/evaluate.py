from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score

from models import compute_spearman


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"


def compute_ic_by_date(predictions: pd.DataFrame) -> pd.DataFrame:
    ic_series = predictions.groupby("date").apply(
        lambda df: df["prediction"].corr(df["realized_return"], method="spearman")
    )
    return ic_series.rename("information_coefficient").reset_index()


def compute_regression_metrics(predictions: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    y_true = predictions["target"]
    y_pred = predictions["prediction"]
    ic_by_date = compute_ic_by_date(predictions)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "correlation": float(y_true.corr(y_pred)),
        "spearman_rank_correlation": compute_spearman(y_true, y_pred),
        "daily_information_coefficient_mean": float(ic_by_date["information_coefficient"].mean()),
        "daily_information_coefficient_std": float(ic_by_date["information_coefficient"].std(ddof=0)),
    }
    return metrics, ic_by_date


def compute_classification_metrics(predictions: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    y_true = predictions["target"].astype(int)
    y_score = predictions["prediction"]
    y_pred = (y_score >= 0.5).astype(int)
    ic_by_date = compute_ic_by_date(predictions)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "daily_information_coefficient_mean": float(ic_by_date["information_coefficient"].mean()),
        "daily_information_coefficient_std": float(ic_by_date["information_coefficient"].std(ddof=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics, ic_by_date


def compute_bucket_returns(predictions: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """Compute average realized return by daily prediction bucket."""
    frame = predictions.copy()
    bucket_frames = []
    for date, group in frame.groupby("date"):
        if group["prediction"].nunique() < n_buckets:
            continue
        group = group.copy()
        group["bucket"] = pd.qcut(group["prediction"], q=n_buckets, labels=False, duplicates="drop")
        bucket_frames.append(group)

    if not bucket_frames:
        return pd.DataFrame(columns=["bucket", "mean_realized_return"])

    bucket_data = pd.concat(bucket_frames, ignore_index=True)
    return (
        bucket_data.groupby("bucket", as_index=False)["realized_return"]
        .mean()
        .rename(columns={"realized_return": "mean_realized_return"})
        .sort_values("bucket")
    )


def compute_quantile_analysis(predictions: pd.DataFrame, n_quantiles: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute quantile average returns and daily top-minus-bottom spread."""
    quantile_frames = []
    spread_rows = []

    for date, group in predictions.groupby("date"):
        if group["prediction"].nunique() < n_quantiles:
            continue
        group = group.copy()
        group["quantile"] = pd.qcut(group["prediction"], q=n_quantiles, labels=False, duplicates="drop")
        quantile_frames.append(group)
        quantile_means = group.groupby("quantile")["realized_return"].mean()
        if not quantile_means.empty:
            spread_rows.append(
                {
                    "date": date,
                    "top_quantile_return": float(quantile_means.iloc[-1]),
                    "bottom_quantile_return": float(quantile_means.iloc[0]),
                    "spread_return": float(quantile_means.iloc[-1] - quantile_means.iloc[0]),
                }
            )

    if not quantile_frames:
        empty_summary = pd.DataFrame(columns=["quantile", "mean_realized_return"])
        empty_spread = pd.DataFrame(columns=["date", "top_quantile_return", "bottom_quantile_return", "spread_return"])
        return empty_summary, empty_spread

    quantile_data = pd.concat(quantile_frames, ignore_index=True)
    summary = (
        quantile_data.groupby("quantile", as_index=False)["realized_return"]
        .mean()
        .rename(columns={"realized_return": "mean_realized_return"})
        .sort_values("quantile")
    )
    spread = pd.DataFrame(spread_rows).sort_values("date").reset_index(drop=True)
    return summary, spread


def save_metrics(metrics: dict[str, float], file_name: str) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / file_name
    pd.DataFrame([metrics]).to_csv(path, index=False)
    return path


def plot_predicted_vs_realized(predictions: pd.DataFrame, file_name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / file_name
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predictions["prediction"], predictions["target"], alpha=0.35)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Training Target")
    ax.set_title("Predicted vs Target")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(importance_frame: pd.DataFrame, file_name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / file_name
    top = importance_frame.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Tree Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
