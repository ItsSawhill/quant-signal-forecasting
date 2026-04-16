from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest import run_backtest, save_backtest_outputs
from data_loader import DEFAULT_TICKERS, download_market_data
from evaluate import (
    compute_bucket_returns,
    compute_classification_metrics,
    compute_regression_metrics,
    plot_feature_importance,
    plot_predicted_vs_realized,
    save_metrics,
)
from features import FEATURE_COLUMNS, create_features
from labels import create_labels
from models import extract_feature_importance, fit_and_predict
from portfolio import attach_positions, build_positions


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
METRICS_DIR = ROOT / "outputs" / "metrics"


def build_modeling_dataset(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
    horizon: int,
) -> pd.DataFrame:
    market_data = download_market_data(tickers=tickers, start_date=start_date, end_date=end_date)
    featured = create_features(market_data)
    labeled = create_labels(featured, horizon=horizon)
    dataset = labeled.dropna(subset=[col for col in FEATURE_COLUMNS if col in labeled.columns] + ["forward_return"]).copy()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATA_DIR / "modeling_dataset.csv", index=False)
    try:
        dataset.to_parquet(DATA_DIR / "modeling_dataset.parquet", index=False)
    except Exception:
        pass
    return dataset


def walk_forward_predictions(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    task: str,
    target_column: str,
    test_size: int,
    min_train_size: int,
    run_prefix: str,
) -> tuple[pd.DataFrame, object | None]:
    unique_dates = sorted(dataset["date"].unique())
    predictions = []
    last_model = None

    for train_end_idx in range(min_train_size, len(unique_dates), test_size):
        train_dates = unique_dates[:train_end_idx]
        test_dates = unique_dates[train_end_idx : train_end_idx + test_size]
        if not test_dates:
            break

        train_frame = dataset[dataset["date"].isin(train_dates)].copy()
        test_frame = dataset[dataset["date"].isin(test_dates)].copy()
        if train_frame.empty or test_frame.empty:
            continue

        val_cutoff = max(1, int(train_frame["date"].nunique() * 0.8))
        split_dates = sorted(train_frame["date"].unique())
        fit_dates = split_dates[:val_cutoff]
        val_dates = split_dates[val_cutoff:]
        if not val_dates:
            val_dates = fit_dates[-max(1, min(10, len(fit_dates))):]
            fit_dates = fit_dates[: -len(val_dates)]

        fit_frame = train_frame[train_frame["date"].isin(fit_dates)].copy()
        val_frame = train_frame[train_frame["date"].isin(val_dates)].copy()
        if fit_frame.empty or val_frame.empty:
            continue

        checkpoint_path = METRICS_DIR / f"{run_prefix}_mlp_checkpoint.pt"
        model, preds = fit_and_predict(
            model_name=model_name,
            task=task,
            x_train=fit_frame[feature_columns],
            y_train=fit_frame[target_column],
            x_val=val_frame[feature_columns],
            y_val=val_frame[target_column],
            x_test=test_frame[feature_columns],
            checkpoint_path=checkpoint_path,
        )
        last_model = model
        fold_predictions = test_frame[["date", "asset", "forward_return"]].copy()
        fold_predictions = fold_predictions.rename(columns={"forward_return": "realized_return"})
        fold_predictions["target"] = test_frame[target_column].values
        fold_predictions["prediction"] = preds
        predictions.append(fold_predictions)

    if not predictions:
        raise ValueError("No out-of-sample predictions were generated. Try reducing min_train_size or test_size.")

    result = pd.concat(predictions, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    return result, last_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train forecasting models and run a portfolio backtest.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--model", choices=["ridge", "tree", "mlp"], default="ridge")
    parser.add_argument("--test-size", type=int, default=63, help="Number of daily observations per out-of-sample block.")
    parser.add_argument("--min-train-size", type=int, default=252, help="Minimum number of daily observations before walk-forward testing.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--holding-horizon", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_prefix = f"{args.model}_{args.task}"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_modeling_dataset(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
    )
    feature_columns = [col for col in FEATURE_COLUMNS if col in dataset.columns]
    target_column = "forward_return" if args.task == "regression" else "forward_return_binary"

    predictions, last_model = walk_forward_predictions(
        dataset=dataset,
        feature_columns=feature_columns,
        model_name=args.model,
        task=args.task,
        target_column=target_column,
        test_size=args.test_size,
        min_train_size=args.min_train_size,
        run_prefix=run_prefix,
    )

    prediction_path_csv = PREDICTIONS_DIR / f"{run_prefix}_predictions.csv"
    predictions.to_csv(prediction_path_csv, index=False)
    try:
        predictions.to_parquet(PREDICTIONS_DIR / f"{run_prefix}_predictions.parquet", index=False)
    except Exception:
        pass

    if args.task == "regression":
        metrics = compute_regression_metrics(predictions)
    else:
        metrics = compute_classification_metrics(predictions)
    save_metrics(metrics, f"{run_prefix}_metrics_summary.csv")

    bucket_returns = compute_bucket_returns(predictions)
    bucket_returns.to_csv(METRICS_DIR / f"{run_prefix}_bucket_returns.csv", index=False)
    plot_predicted_vs_realized(predictions, f"{run_prefix}_predicted_vs_realized.png")

    if args.model == "tree" and last_model is not None:
        importance_frame = extract_feature_importance(last_model, feature_columns)
        if importance_frame is not None:
            importance_frame.to_csv(METRICS_DIR / f"{run_prefix}_feature_importance.csv", index=False)
            plot_feature_importance(importance_frame, f"{run_prefix}_feature_importance.png")

    positions = build_positions(predictions, top_k=args.top_k, holding_horizon=args.holding_horizon)
    positions.to_csv(METRICS_DIR / f"{run_prefix}_positions.csv", index=False)

    portfolio_frame = attach_positions(predictions, positions)
    daily_returns, backtest_metrics = run_backtest(portfolio_frame, transaction_cost_bps=args.transaction_cost_bps)
    save_backtest_outputs(daily_returns, backtest_metrics, prefix=run_prefix)

    print(f"Saved predictions to {prediction_path_csv}")
    print(f"Saved metrics to {METRICS_DIR}")
    print(f"Saved figures to {ROOT / 'outputs' / 'figures'}")


if __name__ == "__main__":
    main()
