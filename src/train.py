from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest import run_backtest, save_backtest_outputs
from data_loader import BENCHMARK_TICKER, DEFAULT_START_DATE, DEFAULT_TICKERS, download_market_data
from evaluate import (
    compute_bucket_returns,
    compute_classification_metrics,
    compute_quantile_analysis,
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
    benchmark_ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_data = download_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        benchmark_ticker=benchmark_ticker,
    )
    featured = create_features(market_data, benchmark_ticker=benchmark_ticker)
    labeled = create_labels(featured, horizon=horizon, ranking_universe=tickers)
    dataset = labeled[labeled["asset"].isin(tickers)].copy()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATA_DIR / "modeling_dataset.csv", index=False)
    try:
        dataset.to_parquet(DATA_DIR / "modeling_dataset.parquet", index=False)
    except Exception:
        pass
    return dataset, market_data


def build_benchmark_returns(market_data: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    benchmark = market_data.loc[market_data["asset"] == benchmark_ticker, ["date", "close"]].copy().sort_values("date")
    benchmark["benchmark_next_day_return"] = benchmark["close"].shift(-1) / benchmark["close"] - 1.0
    return benchmark[["date", "benchmark_next_day_return"]].dropna().reset_index(drop=True)


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
        fold_predictions = test_frame[["date", "asset", "forward_return", "next_day_return"]].copy()
        fold_predictions = fold_predictions.rename(columns={"forward_return": "realized_return"})
        fold_predictions["target"] = test_frame[target_column].values
        fold_predictions["prediction"] = preds
        predictions.append(fold_predictions)

    if not predictions:
        raise ValueError("No out-of-sample predictions were generated. Try reducing min_train_size or test_size.")

    result = pd.concat(predictions, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    return result, last_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cross-sectional signal models and run a portfolio backtest.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--target-mode", choices=["forward_return", "cross_sectional_rank"], default="cross_sectional_rank")
    parser.add_argument("--model", choices=["ridge", "tree", "mlp"], default="ridge")
    parser.add_argument("--test-size", type=int, default=63, help="Number of daily observations per out-of-sample block.")
    parser.add_argument("--min-train-size", type=int, default=252, help="Minimum number of daily observations before walk-forward testing.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--portfolio-mode", choices=["topk", "quantile"], default="quantile")
    parser.add_argument("--weight-scheme", choices=["equal", "signal"], default="signal")
    parser.add_argument("--quantile", type=float, default=0.1)
    parser.add_argument("--gross-exposure", type=float, default=1.0)
    parser.add_argument("--max-turnover", type=float, default=0.5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--holding-horizon", type=int, default=5)
    parser.add_argument("--rebalance-frequency", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_label = args.target_mode
    run_prefix = f"{args.model}_{args.task}_{target_label}"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    dataset, market_data = build_modeling_dataset(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        benchmark_ticker=args.benchmark_ticker,
    )
    feature_columns = [col for col in FEATURE_COLUMNS if col in dataset.columns]

    if args.task == "classification":
        target_column = "forward_return_binary"
    elif args.target_mode == "cross_sectional_rank":
        target_column = "forward_return_rank"
    else:
        target_column = "forward_return"

    required_columns = feature_columns + [target_column, "forward_return", "next_day_return"]
    dataset = dataset.dropna(subset=required_columns).copy()

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
        metrics, ic_by_date = compute_regression_metrics(predictions)
    else:
        metrics, ic_by_date = compute_classification_metrics(predictions)
    save_metrics(metrics, f"{run_prefix}_metrics_summary.csv")
    ic_by_date.to_csv(METRICS_DIR / f"{run_prefix}_ic_by_date.csv", index=False)

    bucket_returns = compute_bucket_returns(predictions, n_buckets=5)
    bucket_returns.to_csv(METRICS_DIR / f"{run_prefix}_bucket_returns.csv", index=False)
    quantile_summary, quantile_spread = compute_quantile_analysis(predictions, n_quantiles=10)
    quantile_summary.to_csv(METRICS_DIR / f"{run_prefix}_quantile_returns.csv", index=False)
    quantile_spread.to_csv(METRICS_DIR / f"{run_prefix}_quantile_spread_returns.csv", index=False)
    plot_predicted_vs_realized(predictions, f"{run_prefix}_predicted_vs_realized.png")

    if args.model == "tree" and last_model is not None:
        importance_frame = extract_feature_importance(last_model, feature_columns)
        if importance_frame is not None:
            importance_frame.to_csv(METRICS_DIR / f"{run_prefix}_feature_importance.csv", index=False)
            plot_feature_importance(importance_frame, f"{run_prefix}_feature_importance.png")

    positions = build_positions(
        predictions,
        top_k=args.top_k,
        holding_horizon=args.holding_horizon,
        portfolio_mode=args.portfolio_mode,
        weight_scheme=args.weight_scheme,
        quantile=args.quantile,
        gross_exposure=args.gross_exposure,
        max_turnover=args.max_turnover,
        rebalance_frequency=args.rebalance_frequency,
    )
    positions.to_csv(METRICS_DIR / f"{run_prefix}_positions.csv", index=False)

    benchmark_returns = build_benchmark_returns(market_data, benchmark_ticker=args.benchmark_ticker)
    portfolio_frame = attach_positions(predictions, positions)
    daily_returns, backtest_metrics = run_backtest(
        portfolio_frame,
        transaction_cost_bps=args.transaction_cost_bps,
        benchmark_returns=benchmark_returns,
    )
    save_backtest_outputs(daily_returns, backtest_metrics, prefix=run_prefix)

    print(f"Saved predictions to {prediction_path_csv}")
    print(f"Saved metrics to {METRICS_DIR}")
    print(f"Saved figures to {ROOT / 'outputs' / 'figures'}")


if __name__ == "__main__":
    main()
