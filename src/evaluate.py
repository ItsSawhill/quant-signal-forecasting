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


def compute_regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = predictions["target"]
    y_pred = predictions["prediction"]
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "correlation": float(y_true.corr(y_pred)),
        "spearman_rank_correlation": compute_spearman(y_true, y_pred),
    }

    ic_by_day = predictions.groupby("date").apply(
        lambda df: df["prediction"].corr(df["realized_return"], method="spearman")
    )
    metrics["daily_information_coefficient_mean"] = float(ic_by_day.mean())
    metrics["daily_information_coefficient_std"] = float(ic_by_day.std(ddof=0))
    return metrics


def compute_classification_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = predictions["target"].astype(int)
    y_score = predictions["prediction"]
    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    ic_by_day = predictions.groupby("date").apply(
        lambda df: df["prediction"].corr(df["realized_return"], method="spearman")
    )
    metrics["daily_information_coefficient_mean"] = float(ic_by_day.mean())
    metrics["daily_information_coefficient_std"] = float(ic_by_day.std(ddof=0))
    return metrics


def compute_bucket_returns(predictions: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """Compute average target return by daily prediction bucket."""
    frame = predictions.copy()
    bucket_frames = []
    for date, group in frame.groupby("date"):
        if group["prediction"].nunique() < n_buckets:
            continue
        group = group.copy()
        group["bucket"] = pd.qcut(group["prediction"], q=n_buckets, labels=False, duplicates="drop")
        group["date"] = date
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


def save_metrics(metrics: dict[str, float], file_name: str) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / file_name
    pd.DataFrame([metrics]).to_csv(path, index=False)
    return path


def plot_predicted_vs_realized(predictions: pd.DataFrame, file_name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / file_name
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predictions["prediction"], predictions["target"], alpha=0.5)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Realized Target")
    ax.set_title("Predicted vs Realized")
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
