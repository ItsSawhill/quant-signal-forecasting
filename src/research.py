from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import run_backtest, save_backtest_outputs
from data_loader import BENCHMARK_TICKER, DEFAULT_START_DATE, DEFAULT_TICKERS, get_market_data
from evaluate import compute_quantile_analysis, compute_regression_metrics, save_metrics
from features import EXTENDED_FEATURE_COLUMNS, FEATURE_COLUMNS, create_features
from labels import create_labels
from portfolio import attach_positions, build_positions
from train import build_benchmark_returns, get_realized_return_column, get_target_column, walk_forward_predictions


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
FIGURES_DIR = ROOT / "outputs" / "figures"

BOOTSTRAP_SEED = 42
BOOTSTRAP_REPS = 300
SMALL_DENOMINATOR = 1e-8


def prepare_dataset(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
    benchmark_ticker: str,
    feature_set: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_data = get_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        benchmark_ticker=benchmark_ticker,
    )
    featured = create_features(
        market_data,
        benchmark_ticker=benchmark_ticker,
        include_phase4_features=feature_set == "extended",
    )
    labeled = create_labels(featured, horizon=5, ranking_universe=tickers)
    dataset = labeled[labeled["asset"].isin(tickers)].copy()
    return dataset, market_data


def run_single_experiment(
    dataset: pd.DataFrame,
    market_data: pd.DataFrame,
    model: str,
    horizon: int,
    target_mode: str,
    benchmark_ticker: str,
    test_size: int,
    min_train_size: int,
    portfolio_mode: str = "quantile",
    weight_scheme: str = "signal",
    top_k: int = 5,
    quantile: float = 0.1,
    gross_exposure: float = 1.0,
    max_turnover: float | None = 0.5,
    transaction_cost_bps: float = 5.0,
    rebalance_frequency: int | None = None,
) -> dict[str, float]:
    feature_columns = [col for col in FEATURE_COLUMNS if col in dataset.columns]
    target_column = get_target_column(task="regression", target_mode=target_mode, horizon=horizon)
    realized_return_column = get_realized_return_column(horizon)

    working = dataset.dropna(subset=feature_columns + [target_column, realized_return_column, "next_day_return"]).copy()
    working["forward_return"] = working[realized_return_column]

    run_prefix = f"{model}_regression_{target_mode}_h{horizon}"
    predictions, _ = walk_forward_predictions(
        dataset=working,
        feature_columns=feature_columns,
        model_name=model,
        task="regression",
        target_column=target_column,
        test_size=test_size,
        min_train_size=min_train_size,
        run_prefix=run_prefix,
    )

    predictions.to_csv(PREDICTIONS_DIR / f"{run_prefix}_predictions.csv", index=False)
    regression_metrics, ic_by_date = compute_regression_metrics(predictions)
    quantile_summary, quantile_spread = compute_quantile_analysis(predictions, n_quantiles=10)
    positions = build_positions(
        predictions,
        top_k=top_k,
        portfolio_mode=portfolio_mode,
        weight_scheme=weight_scheme,
        quantile=quantile,
        gross_exposure=gross_exposure,
        max_turnover=max_turnover,
        rebalance_frequency=rebalance_frequency,
    )
    portfolio_frame = attach_positions(predictions, positions)
    benchmark_returns = build_benchmark_returns(market_data, benchmark_ticker=benchmark_ticker)
    daily_returns, backtest_metrics = run_backtest(
        portfolio_frame,
        transaction_cost_bps=transaction_cost_bps,
        benchmark_returns=benchmark_returns,
    )

    row = {
        "model": model,
        "target_mode": target_mode,
        "horizon": horizon,
        "mean_ic": regression_metrics["daily_information_coefficient_mean"],
        "ic_volatility": regression_metrics["daily_information_coefficient_std"],
        "rmse": regression_metrics["rmse"],
        "correlation": regression_metrics["correlation"],
        "spearman_rank_correlation": regression_metrics["spearman_rank_correlation"],
        "top_quantile_mean_return": float(quantile_summary["mean_realized_return"].iloc[-1]) if not quantile_summary.empty else float("nan"),
        "bottom_quantile_mean_return": float(quantile_summary["mean_realized_return"].iloc[0]) if not quantile_summary.empty else float("nan"),
        "mean_quantile_spread": float(quantile_spread["spread_return"].mean()) if not quantile_spread.empty else float("nan"),
        "annualized_return": backtest_metrics["annualized_return"],
        "annualized_gross_return": backtest_metrics["annualized_gross_return"],
        "annualized_volatility": backtest_metrics["annualized_volatility"],
        "sharpe_ratio": backtest_metrics["sharpe_ratio"],
        "max_drawdown": backtest_metrics["max_drawdown"],
        "average_turnover": backtest_metrics["average_turnover"],
        "return_per_unit_turnover": backtest_metrics["return_per_unit_turnover"],
        "benchmark_annualized_return": backtest_metrics.get("benchmark_annualized_return", float("nan")),
        "excess_annualized_return_vs_benchmark": backtest_metrics.get("excess_annualized_return_vs_benchmark", float("nan")),
    }
    save_metrics(regression_metrics, f"{run_prefix}_metrics_summary.csv")
    ic_by_date.to_csv(METRICS_DIR / f"{run_prefix}_ic_by_date.csv", index=False)
    quantile_summary.to_csv(METRICS_DIR / f"{run_prefix}_quantile_returns.csv", index=False)
    quantile_spread.to_csv(METRICS_DIR / f"{run_prefix}_quantile_spread_returns.csv", index=False)
    positions.to_csv(METRICS_DIR / f"{run_prefix}_positions.csv", index=False)
    save_backtest_outputs(daily_returns, backtest_metrics, prefix=run_prefix)
    pd.DataFrame([regression_metrics | backtest_metrics]).to_csv(METRICS_DIR / f"{run_prefix}_summary.csv", index=False)
    return row


def run_horizon_study(args: argparse.Namespace) -> Path:
    dataset, market_data = prepare_dataset(args.tickers, args.start_date, args.end_date, args.benchmark_ticker)
    rows = []
    for model in args.models:
        for horizon in args.horizons:
            rows.append(
                run_single_experiment(
                    dataset=dataset,
                    market_data=market_data,
                    model=model,
                    horizon=horizon,
                    target_mode=args.target_mode,
                    benchmark_ticker=args.benchmark_ticker,
                    test_size=args.test_size,
                    min_train_size=args.min_train_size,
                    transaction_cost_bps=args.transaction_cost_bps,
                )
            )
    comparison = pd.DataFrame(rows).sort_values(["target_mode", "horizon", "sharpe_ratio"], ascending=[True, True, False]).reset_index(drop=True)
    output_path = METRICS_DIR / f"horizon_study_{args.target_mode}.csv"
    comparison.to_csv(output_path, index=False)
    return output_path


def run_ranking_study(args: argparse.Namespace) -> Path:
    dataset, market_data = prepare_dataset(args.tickers, args.start_date, args.end_date, args.benchmark_ticker)
    rows = []
    for model in args.models:
        for horizon in args.horizons:
            rows.append(
                run_single_experiment(
                    dataset=dataset,
                    market_data=market_data,
                    model=model,
                    horizon=horizon,
                    target_mode="cross_sectional_rank",
                    benchmark_ticker=args.benchmark_ticker,
                    test_size=args.test_size,
                    min_train_size=args.min_train_size,
                    transaction_cost_bps=args.transaction_cost_bps,
                )
            )

    comparison = (
        pd.DataFrame(rows)[
            [
                "model",
                "horizon",
                "mean_ic",
                "ic_volatility",
                "mean_quantile_spread",
                "annualized_return",
                "sharpe_ratio",
                "average_turnover",
                "max_drawdown",
            ]
        ]
        .rename(
            columns={
                "mean_ic": "ic",
                "mean_quantile_spread": "quantile_spread",
                "annualized_return": "return",
                "average_turnover": "turnover",
            }
        )
        .sort_values(["horizon", "model"])
        .reset_index(drop=True)
    )
    output_path = METRICS_DIR / "ranking_vs_regression_comparison.csv"
    comparison.to_csv(output_path, index=False)
    return output_path


def run_portfolio_study(args: argparse.Namespace) -> Path:
    prediction_path = PREDICTIONS_DIR / args.prediction_file
    predictions = pd.read_csv(prediction_path, parse_dates=["date"])

    specs = [
        {"portfolio_mode": "quantile", "weight_scheme": "equal", "quantile": 0.1, "top_k": 5, "rebalance_frequency": 1, "max_turnover": None},
        {"portfolio_mode": "quantile", "weight_scheme": "signal", "quantile": 0.1, "top_k": 5, "rebalance_frequency": 1, "max_turnover": 0.5},
        {"portfolio_mode": "quantile", "weight_scheme": "equal", "quantile": 0.2, "top_k": 5, "rebalance_frequency": 5, "max_turnover": None},
        {"portfolio_mode": "quantile", "weight_scheme": "signal", "quantile": 0.2, "top_k": 5, "rebalance_frequency": 5, "max_turnover": 0.5},
        {"portfolio_mode": "topk", "weight_scheme": "equal", "quantile": 0.1, "top_k": 10, "rebalance_frequency": 1, "max_turnover": None},
        {"portfolio_mode": "topk", "weight_scheme": "signal", "quantile": 0.1, "top_k": 10, "rebalance_frequency": 10, "max_turnover": 0.5},
    ]

    benchmark_returns = pd.read_csv(args.benchmark_file, parse_dates=["date"])
    rows = []
    for spec in specs:
        positions = build_positions(
            predictions,
            top_k=spec["top_k"],
            portfolio_mode=spec["portfolio_mode"],
            weight_scheme=spec["weight_scheme"],
            quantile=spec["quantile"],
            gross_exposure=1.0,
            max_turnover=spec["max_turnover"],
            rebalance_frequency=spec["rebalance_frequency"],
        )
        portfolio_frame = attach_positions(predictions, positions)
        _, metrics = run_backtest(
            portfolio_frame,
            transaction_cost_bps=args.transaction_cost_bps,
            benchmark_returns=benchmark_returns,
        )
        rows.append(
            {
                **spec,
                "annualized_gross_return": metrics["annualized_gross_return"],
                "annualized_return": metrics["annualized_return"],
                "average_turnover": metrics["average_turnover"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "return_per_unit_turnover": metrics["return_per_unit_turnover"],
            }
        )

    output = pd.DataFrame(rows)
    output_path = METRICS_DIR / "portfolio_construction_study.csv"
    output.to_csv(output_path, index=False)
    return output_path


def _ensure_candidate_predictions(
    dataset: pd.DataFrame,
    market_data: pd.DataFrame,
    benchmark_ticker: str,
    test_size: int,
    min_train_size: int,
    transaction_cost_bps: float,
) -> list[tuple[str, int, pd.DataFrame]]:
    candidates = [("ranker", 10), ("ranker", 20), ("tree", 20)]
    loaded = []
    for model, horizon in candidates:
        prediction_path = PREDICTIONS_DIR / f"{model}_regression_cross_sectional_rank_h{horizon}_predictions.csv"
        if not prediction_path.exists():
            run_single_experiment(
                dataset=dataset,
                market_data=market_data,
                model=model,
                horizon=horizon,
                target_mode="cross_sectional_rank",
                benchmark_ticker=benchmark_ticker,
                test_size=test_size,
                min_train_size=min_train_size,
                transaction_cost_bps=transaction_cost_bps,
            )
        predictions = pd.read_csv(prediction_path, parse_dates=["date"])
        loaded.append((model, horizon, predictions))
    return loaded


def _phase3_specs() -> list[dict[str, object]]:
    specs = []
    selection_specs = [
        {"selection_method": "decile", "portfolio_mode": "quantile", "quantile": 0.1, "top_k": 5},
        {"selection_method": "quintile", "portfolio_mode": "quantile", "quantile": 0.2, "top_k": 5},
        {"selection_method": "topk", "portfolio_mode": "topk", "quantile": 0.1, "top_k": 10},
    ]
    for selection in selection_specs:
        for weighting_scheme in ["equal", "signal"]:
            for rebalance_frequency in [1, 5, 10]:
                for turnover_cap in [False, True]:
                    specs.append(
                        {
                            **selection,
                            "weighting_scheme": weighting_scheme,
                            "rebalance_frequency": rebalance_frequency,
                            "turnover_cap": turnover_cap,
                            "max_turnover": 0.5 if turnover_cap else None,
                        }
                    )
    return specs


def _phase3_config_label(row: pd.Series) -> str:
    turnover_label = "cap_on" if bool(row["turnover_cap"]) else "cap_off"
    return (
        f"{row['model']}_h{int(row['horizon'])}_"
        f"{row['selection_method']}_{row['weighting_scheme']}_"
        f"reb{int(row['rebalance_frequency'])}_{turnover_label}"
    )


def _write_phase3_markdown(results: pd.DataFrame, output_path: Path) -> None:
    best_by_candidate = (
        results.sort_values("sharpe", ascending=False)
        .groupby(["model", "horizon"], as_index=False)
        .first()[
            [
                "model",
                "horizon",
                "selection_method",
                "weighting_scheme",
                "rebalance_frequency",
                "turnover_cap",
                "gross_return",
                "net_return",
                "sharpe",
                "max_drawdown",
                "turnover",
                "annualized_volatility",
                "mean_IC",
                "quantile_spread",
            ]
        ]
    )
    top_configs = results.sort_values("sharpe", ascending=False).head(15).copy()

    def to_markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for col in display.columns:
            if pd.api.types.is_bool_dtype(display[col]):
                display[col] = display[col].map({True: "on", False: "off"})
            elif pd.api.types.is_numeric_dtype(display[col]):
                if col in {"horizon", "rebalance_frequency"}:
                    display[col] = display[col].map(lambda x: f"{int(x)}")
                else:
                    display[col] = display[col].map(lambda x: f"{x:.4f}")
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    content = "\n".join(
        [
            "# Phase 3 Portfolio Sensitivity",
            "",
            "## Best Configuration By Candidate",
            "",
            to_markdown_table(best_by_candidate),
            "",
            "## Top 15 Configurations By Sharpe",
            "",
            to_markdown_table(
                top_configs[
                    [
                        "model",
                        "horizon",
                        "selection_method",
                        "weighting_scheme",
                        "rebalance_frequency",
                        "turnover_cap",
                        "gross_return",
                        "net_return",
                        "sharpe",
                        "turnover",
                        "mean_IC",
                        "quantile_spread",
                    ]
                ]
            ),
            "",
        ]
    )
    output_path.write_text(content)


def _save_phase3_figure(curves: dict[str, pd.DataFrame], results: pd.DataFrame) -> Path | None:
    if not curves:
        return None
    top_labels = [
        _phase3_config_label(row)
        for _, row in results.sort_values("sharpe", ascending=False).head(3).iterrows()
    ]
    selected = [(label, curves[label]) for label in top_labels if label in curves]
    if not selected:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "phase3_portfolio_sensitivity_best_configs.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, frame in selected:
        ax.plot(frame["date"], frame["equity_curve"], linewidth=2, label=label)
    ax.set_title("Phase 3 Best Portfolio Configurations")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_phase3_portfolio_study(args: argparse.Namespace) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    dataset, market_data = prepare_dataset(args.tickers, args.start_date, args.end_date, args.benchmark_ticker)
    benchmark_returns = build_benchmark_returns(market_data, benchmark_ticker=args.benchmark_ticker)
    candidates = _ensure_candidate_predictions(
        dataset=dataset,
        market_data=market_data,
        benchmark_ticker=args.benchmark_ticker,
        test_size=args.test_size,
        min_train_size=args.min_train_size,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    specs = _phase3_specs()
    rows = []
    curves: dict[str, pd.DataFrame] = {}

    for model, horizon, predictions in candidates:
        regression_metrics, _ = compute_regression_metrics(predictions)
        _, quantile_spread = compute_quantile_analysis(predictions, n_quantiles=10)
        mean_ic = regression_metrics["daily_information_coefficient_mean"]
        mean_quantile_spread = float(quantile_spread["spread_return"].mean()) if not quantile_spread.empty else float("nan")

        for spec in specs:
            positions = build_positions(
                predictions,
                top_k=int(spec["top_k"]),
                holding_horizon=5,
                portfolio_mode=str(spec["portfolio_mode"]),
                weight_scheme=str(spec["weighting_scheme"]),
                quantile=float(spec["quantile"]),
                gross_exposure=1.0,
                max_turnover=spec["max_turnover"],
                rebalance_frequency=int(spec["rebalance_frequency"]),
            )
            portfolio_frame = attach_positions(predictions, positions)
            daily_returns, metrics = run_backtest(
                portfolio_frame,
                transaction_cost_bps=args.transaction_cost_bps,
                benchmark_returns=benchmark_returns,
            )

            row = {
                "model": model,
                "horizon": horizon,
                "selection_method": spec["selection_method"],
                "weighting_scheme": spec["weighting_scheme"],
                "rebalance_frequency": int(spec["rebalance_frequency"]),
                "turnover_cap": bool(spec["turnover_cap"]),
                "gross_return": metrics["annualized_gross_return"],
                "net_return": metrics["annualized_return"],
                "sharpe": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["average_turnover"],
                "annualized_volatility": metrics["annualized_volatility"],
                "return_per_unit_turnover": metrics["return_per_unit_turnover"],
                "mean_IC": mean_ic,
                "quantile_spread": mean_quantile_spread,
            }
            rows.append(row)
            curves[_phase3_config_label(pd.Series(row))] = daily_returns.copy()

    results = pd.DataFrame(rows).sort_values(["sharpe", "net_return"], ascending=[False, False]).reset_index(drop=True)
    csv_path = METRICS_DIR / "phase3_portfolio_sensitivity.csv"
    md_path = METRICS_DIR / "phase3_portfolio_sensitivity.md"
    results.to_csv(csv_path, index=False)
    _write_phase3_markdown(results, md_path)
    _save_phase3_figure(curves, results)
    return csv_path


def _write_phase4_markdown(results: pd.DataFrame, output_path: Path) -> None:
    display = results.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            if col == "horizon":
                display[col] = display[col].map(lambda x: f"{int(x)}")
            else:
                display[col] = display[col].map(lambda x: f"{x:.4f}")

    headers = list(display.columns)
    lines = [
        "# Phase 4 Feature Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    output_path.write_text("\n".join(lines) + "\n")


def run_phase4_feature_study(args: argparse.Namespace) -> Path:
    candidates = [("ranker", 10), ("ranker", 20), ("tree", 20)]
    rows = []

    for feature_set in ["baseline", "extended"]:
        dataset, market_data = prepare_dataset(
            args.tickers,
            args.start_date,
            args.end_date,
            args.benchmark_ticker,
            feature_set=feature_set,
        )
        feature_template = EXTENDED_FEATURE_COLUMNS if feature_set == "extended" else FEATURE_COLUMNS
        benchmark_returns = build_benchmark_returns(market_data, benchmark_ticker=args.benchmark_ticker)

        for model, horizon in candidates:
            target_column = get_target_column(task="regression", target_mode="cross_sectional_rank", horizon=horizon)
            realized_return_column = get_realized_return_column(horizon)
            feature_columns = [col for col in feature_template if col in dataset.columns]
            working = dataset.dropna(subset=feature_columns + [target_column, realized_return_column, "next_day_return"]).copy()
            working["forward_return"] = working[realized_return_column]

            run_prefix = f"{model}_regression_cross_sectional_rank_h{horizon}_{feature_set}"
            predictions, _ = walk_forward_predictions(
                dataset=working,
                feature_columns=feature_columns,
                model_name=model,
                task="regression",
                target_column=target_column,
                test_size=args.test_size,
                min_train_size=args.min_train_size,
                run_prefix=run_prefix,
            )

            predictions.to_csv(PREDICTIONS_DIR / f"{run_prefix}_predictions.csv", index=False)
            regression_metrics, ic_by_date = compute_regression_metrics(predictions)
            quantile_summary, quantile_spread = compute_quantile_analysis(predictions, n_quantiles=10)
            positions = build_positions(
                predictions,
                top_k=5,
                holding_horizon=5,
                portfolio_mode="quantile",
                weight_scheme="equal",
                quantile=0.1,
                gross_exposure=1.0,
                max_turnover=0.5,
                rebalance_frequency=5,
            )
            positions.to_csv(METRICS_DIR / f"{run_prefix}_positions.csv", index=False)
            portfolio_frame = attach_positions(predictions, positions)
            daily_returns, backtest_metrics = run_backtest(
                portfolio_frame,
                transaction_cost_bps=args.transaction_cost_bps,
                benchmark_returns=benchmark_returns,
            )

            save_metrics(regression_metrics, f"{run_prefix}_metrics_summary.csv")
            ic_by_date.to_csv(METRICS_DIR / f"{run_prefix}_ic_by_date.csv", index=False)
            quantile_summary.to_csv(METRICS_DIR / f"{run_prefix}_quantile_returns.csv", index=False)
            quantile_spread.to_csv(METRICS_DIR / f"{run_prefix}_quantile_spread_returns.csv", index=False)
            save_backtest_outputs(daily_returns, backtest_metrics, prefix=run_prefix)
            pd.DataFrame([regression_metrics | backtest_metrics]).to_csv(METRICS_DIR / f"{run_prefix}_summary.csv", index=False)

            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "feature_set": feature_set,
                    "IC": regression_metrics["daily_information_coefficient_mean"],
                    "IC_volatility": regression_metrics["daily_information_coefficient_std"],
                    "quantile_spread": float(quantile_spread["spread_return"].mean()) if not quantile_spread.empty else float("nan"),
                    "net_return": backtest_metrics["annualized_return"],
                    "Sharpe": backtest_metrics["sharpe_ratio"],
                    "turnover": backtest_metrics["average_turnover"],
                    "max_drawdown": backtest_metrics["max_drawdown"],
                }
            )

    results = pd.DataFrame(rows).sort_values(["model", "horizon", "feature_set"]).reset_index(drop=True)
    csv_path = METRICS_DIR / "phase4_feature_comparison.csv"
    md_path = METRICS_DIR / "phase4_feature_comparison.md"
    results.to_csv(csv_path, index=False)
    _write_phase4_markdown(results, md_path)
    return csv_path


def _phase5_reference_candidates() -> list[dict[str, object]]:
    return [
        {"model": "ranker", "horizon": 10, "feature_set": "extended"},
        {"model": "ranker", "horizon": 20, "feature_set": "extended"},
        {"model": "tree", "horizon": 20, "feature_set": "extended"},
    ]


def _phase5_reference_portfolio() -> dict[str, object]:
    return {
        "selection": "decile",
        "portfolio_mode": "quantile",
        "quantile": 0.1,
        "top_k": 5,
        "weighting": "equal",
        "rebalance": 5,
        "max_turnover": 0.5,
    }


def _phase5_periods(max_date: pd.Timestamp) -> list[dict[str, object]]:
    return [
        {"period": "2018-2020", "start": pd.Timestamp("2018-01-01"), "end": pd.Timestamp("2020-01-01")},
        {"period": "2020-2022", "start": pd.Timestamp("2020-01-01"), "end": pd.Timestamp("2022-01-01")},
        {"period": "2022-2026", "start": pd.Timestamp("2022-01-01"), "end": max_date + pd.Timedelta(days=1)},
    ]


def _load_phase5_predictions(candidate: dict[str, object]) -> pd.DataFrame:
    prediction_path = (
        PREDICTIONS_DIR
        / f"{candidate['model']}_regression_cross_sectional_rank_h{candidate['horizon']}_{candidate['feature_set']}_predictions.csv"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing Phase 5 reference predictions: {prediction_path}")
    return pd.read_csv(prediction_path, parse_dates=["date"])


def _run_phase5_backtest(
    predictions: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    selection: str,
    weighting: str,
    rebalance: int,
    transaction_cost_bps: float,
    max_turnover: float | None,
) -> dict[str, float]:
    portfolio_mode = "quantile" if selection in {"decile", "quintile"} else "topk"
    quantile = 0.1 if selection == "decile" else 0.2
    top_k = 10 if selection == "topk" else 5

    positions = build_positions(
        predictions,
        top_k=top_k,
        holding_horizon=5,
        portfolio_mode=portfolio_mode,
        weight_scheme=weighting,
        quantile=quantile,
        gross_exposure=1.0,
        max_turnover=max_turnover,
        rebalance_frequency=rebalance,
    )
    portfolio_frame = attach_positions(predictions, positions)
    _, backtest_metrics = run_backtest(
        portfolio_frame,
        transaction_cost_bps=transaction_cost_bps,
        benchmark_returns=benchmark_returns,
    )
    regression_metrics, _ = compute_regression_metrics(predictions)
    return {
        "net_return": backtest_metrics["annualized_return"],
        "Sharpe": backtest_metrics["sharpe_ratio"],
        "turnover": backtest_metrics["average_turnover"],
        "max_drawdown": backtest_metrics["max_drawdown"],
        "IC": regression_metrics["daily_information_coefficient_mean"],
    }


def _write_phase5_markdown(results: pd.DataFrame, output_path: Path) -> None:
    subperiod = results[results["period"] != "full"].copy()
    cost = results[
        (results["period"] == "full")
        & (results["selection"] == "decile")
        & (results["weighting"] == "equal")
        & (results["rebalance"] == 5)
    ].copy()
    portfolio = results[(results["period"] == "full") & (results["cost_level"] == 5)].copy()

    def to_markdown(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for col in display.columns:
            if pd.api.types.is_numeric_dtype(display[col]):
                if col in {"horizon", "cost_level", "rebalance"}:
                    display[col] = display[col].map(lambda x: f"{int(x)}")
                else:
                    display[col] = display[col].map(lambda x: f"{x:.4f}")
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    sections = [
        "# Phase 5 Robustness Results",
        "",
        "## Subperiod Analysis",
        "",
        to_markdown(subperiod[["model", "horizon", "period", "net_return", "Sharpe", "turnover", "max_drawdown", "IC"]]),
        "",
        "## Cost Sensitivity",
        "",
        to_markdown(cost[["model", "horizon", "cost_level", "net_return", "Sharpe", "turnover", "IC"]]),
        "",
        "## Portfolio Variations",
        "",
        to_markdown(portfolio[["model", "horizon", "selection", "weighting", "rebalance", "net_return", "Sharpe", "turnover", "max_drawdown", "IC"]]),
        "",
    ]
    output_path.write_text("\n".join(sections))


def _save_phase5_subperiod_figure(results: pd.DataFrame) -> Path | None:
    subperiod = results[results["period"] != "full"].copy()
    if subperiod.empty:
        return None
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "phase5_subperiod_sharpe.png"
    pivot = subperiod.pivot(index="period", columns=["model", "horizon"], values="Sharpe")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Phase 5 Subperiod Sharpe")
    ax.set_xlabel("Period")
    ax.set_ylabel("Sharpe")
    ax.legend(title="Model / Horizon", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def run_phase5_robustness_study(args: argparse.Namespace) -> Path:
    market_data = get_market_data(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_ticker=args.benchmark_ticker,
    )
    benchmark_returns = build_benchmark_returns(market_data, benchmark_ticker=args.benchmark_ticker)
    max_date = benchmark_returns["date"].max()

    reference_portfolio = _phase5_reference_portfolio()
    periods = _phase5_periods(max_date)
    rows = []

    for candidate in _phase5_reference_candidates():
        predictions = _load_phase5_predictions(candidate)

        for period_spec in periods:
            subset = predictions[(predictions["date"] >= period_spec["start"]) & (predictions["date"] < period_spec["end"])].copy()
            benchmark_subset = benchmark_returns[
                (benchmark_returns["date"] >= period_spec["start"]) & (benchmark_returns["date"] < period_spec["end"])
            ].copy()
            if subset.empty or benchmark_subset.empty:
                continue
            metrics = _run_phase5_backtest(
                predictions=subset,
                benchmark_returns=benchmark_subset,
                selection=str(reference_portfolio["selection"]),
                weighting=str(reference_portfolio["weighting"]),
                rebalance=int(reference_portfolio["rebalance"]),
                transaction_cost_bps=5.0,
                max_turnover=reference_portfolio["max_turnover"],
            )
            rows.append(
                {
                    "model": candidate["model"],
                    "horizon": candidate["horizon"],
                    "period": period_spec["period"],
                    "cost_level": 5,
                    "selection": reference_portfolio["selection"],
                    "weighting": reference_portfolio["weighting"],
                    "rebalance": reference_portfolio["rebalance"],
                    **metrics,
                }
            )

        for cost_level in [2, 5, 10]:
            metrics = _run_phase5_backtest(
                predictions=predictions,
                benchmark_returns=benchmark_returns,
                selection=str(reference_portfolio["selection"]),
                weighting=str(reference_portfolio["weighting"]),
                rebalance=int(reference_portfolio["rebalance"]),
                transaction_cost_bps=float(cost_level),
                max_turnover=reference_portfolio["max_turnover"],
            )
            rows.append(
                {
                    "model": candidate["model"],
                    "horizon": candidate["horizon"],
                    "period": "full",
                    "cost_level": cost_level,
                    "selection": reference_portfolio["selection"],
                    "weighting": reference_portfolio["weighting"],
                    "rebalance": reference_portfolio["rebalance"],
                    **metrics,
                }
            )

        for selection, weighting, rebalance in [
            ("decile", "equal", 5),
            ("decile", "equal", 10),
            ("decile", "signal", 5),
            ("decile", "signal", 10),
            ("quintile", "equal", 5),
            ("quintile", "equal", 10),
            ("quintile", "signal", 5),
            ("quintile", "signal", 10),
        ]:
            metrics = _run_phase5_backtest(
                predictions=predictions,
                benchmark_returns=benchmark_returns,
                selection=selection,
                weighting=weighting,
                rebalance=rebalance,
                transaction_cost_bps=5.0,
                max_turnover=reference_portfolio["max_turnover"],
            )
            rows.append(
                {
                    "model": candidate["model"],
                    "horizon": candidate["horizon"],
                    "period": "full",
                    "cost_level": 5,
                    "selection": selection,
                    "weighting": weighting,
                    "rebalance": rebalance,
                    **metrics,
                }
            )

    results = pd.DataFrame(rows).drop_duplicates().sort_values(
        ["model", "horizon", "period", "cost_level", "selection", "weighting", "rebalance"]
    ).reset_index(drop=True)
    csv_path = METRICS_DIR / "phase5_robustness_results.csv"
    md_path = METRICS_DIR / "phase5_robustness_results.md"
    results.to_csv(csv_path, index=False)
    _write_phase5_markdown(results, md_path)
    _save_phase5_subperiod_figure(results)
    return csv_path


def _build_reference_positions(predictions: pd.DataFrame) -> pd.DataFrame:
    reference_portfolio = _phase5_reference_portfolio()
    return build_positions(
        predictions,
        top_k=int(reference_portfolio["top_k"]),
        holding_horizon=5,
        portfolio_mode=str(reference_portfolio["portfolio_mode"]),
        weight_scheme=str(reference_portfolio["weighting"]),
        quantile=float(reference_portfolio["quantile"]),
        gross_exposure=1.0,
        max_turnover=reference_portfolio["max_turnover"],
        rebalance_frequency=int(reference_portfolio["rebalance"]),
    )


def _run_reference_backtest(
    predictions: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    positions = _build_reference_positions(predictions)
    portfolio_frame = attach_positions(predictions, positions)
    return run_backtest(
        portfolio_frame,
        transaction_cost_bps=transaction_cost_bps,
        benchmark_returns=benchmark_returns,
    )


def _annualized_return_from_series(net_returns: pd.Series) -> float:
    if net_returns.empty:
        return float("nan")
    equity = float((1.0 + net_returns).prod())
    return float(equity ** (252 / len(net_returns)) - 1.0)


def _annualized_volatility_from_series(net_returns: pd.Series) -> float:
    if net_returns.empty:
        return float("nan")
    return float(net_returns.std(ddof=0) * np.sqrt(252))


def _annualized_sharpe_from_series(net_returns: pd.Series) -> float:
    annualized_return = _annualized_return_from_series(net_returns)
    annualized_volatility = _annualized_volatility_from_series(net_returns)
    if not np.isfinite(annualized_return) or not np.isfinite(annualized_volatility) or annualized_volatility <= 0:
        return float("nan")
    return float(annualized_return / annualized_volatility)


def _bootstrap_portfolio_statistics(
    daily_returns: pd.DataFrame,
    quantile_spread: pd.DataFrame,
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = BOOTSTRAP_REPS,
) -> dict[str, float]:
    if daily_returns.empty:
        return {}

    rng = np.random.default_rng(seed)
    n_obs = len(daily_returns)
    net_returns = daily_returns["net_return"].to_numpy()
    spread_values = quantile_spread["spread_return"].to_numpy() if not quantile_spread.empty else np.array([])

    annualized_returns = []
    sharpes = []
    spread_means = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_obs, size=n_obs)
        sampled_returns = pd.Series(net_returns[indices])
        annualized_returns.append(_annualized_return_from_series(sampled_returns))
        sharpes.append(_annualized_sharpe_from_series(sampled_returns))
        if spread_values.size:
            spread_indices = rng.integers(0, spread_values.size, size=spread_values.size)
            spread_means.append(float(np.mean(spread_values[spread_indices])))

    stats = {
        "annualized_return_ci_lower": float(np.nanpercentile(annualized_returns, 2.5)),
        "annualized_return_ci_upper": float(np.nanpercentile(annualized_returns, 97.5)),
        "sharpe_ci_lower": float(np.nanpercentile(sharpes, 2.5)),
        "sharpe_ci_upper": float(np.nanpercentile(sharpes, 97.5)),
    }
    if spread_means:
        stats["quantile_spread_ci_lower"] = float(np.nanpercentile(spread_means, 2.5))
        stats["quantile_spread_ci_upper"] = float(np.nanpercentile(spread_means, 97.5))
    return stats


def _make_random_placebo(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    placebo = predictions.copy()
    rng = np.random.default_rng(seed)
    placebo["prediction"] = placebo.groupby("date")["prediction"].transform(
        lambda x: pd.Series(rng.standard_normal(len(x)), index=x.index)
    )
    return placebo


def _make_label_shuffled_placebo(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    placebo = predictions.copy()
    rng = np.random.default_rng(seed)
    placebo["prediction"] = placebo.groupby("date")["target"].transform(
        lambda x: pd.Series(rng.permutation(x.to_numpy()), index=x.index)
    )
    return placebo


def _rolling_ic_series(predictions: pd.DataFrame, window: int = 504) -> pd.DataFrame:
    _, ic_by_date = compute_regression_metrics(predictions)
    rolling = ic_by_date.copy()
    rolling["rolling_mean_ic"] = rolling["information_coefficient"].rolling(window).mean()
    return rolling


def _normal_approx_two_sided_pvalue(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return float("nan")
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def _write_phase6_markdown(results: pd.DataFrame, output_path: Path) -> None:
    def to_markdown(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for col in display.columns:
            if pd.api.types.is_numeric_dtype(display[col]):
                if col == "horizon":
                    display[col] = display[col].map(lambda x: f"{int(x)}")
                else:
                    display[col] = display[col].map(lambda x: f"{x:.4f}")
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    significance = (
        results[results["validation_type"] == "ic_significance"]
        .pivot_table(index=["model", "horizon"], columns="metric", values="value", aggfunc="first")
        .reset_index()
    )
    placebo = (
        results[results["validation_type"] == "placebo"]
        .pivot_table(index=["model", "horizon", "variant"], columns="metric", values="value", aggfunc="first")
        .reset_index()
        .assign(
            variant_order=lambda df: df["variant"].map({"real": 0, "random_ranking": 1, "label_shuffled": 2}).fillna(99)
        )
        .sort_values(["model", "horizon", "variant_order"])
    )
    cost = (
        results[results["validation_type"] == "cost_sweep"]
        .pivot_table(index=["model", "horizon", "variant"], columns="metric", values="value", aggfunc="first")
        .reset_index()
        .rename(columns={"variant": "cost_bps"})
        .assign(cost_bps=lambda df: df["cost_bps"].astype(int))
        .sort_values(["model", "horizon", "cost_bps"])
    )
    bootstrap = (
        results[results["validation_type"] == "bootstrap_ci"]
        .pivot_table(index=["model", "horizon"], columns="metric", values="value", aggfunc="first")
        .reset_index()
    )

    sections = [
        "# Phase 6 Validation Summary",
        "",
        "## IC Significance",
        "",
        to_markdown(significance[["model", "horizon", "mean_ic", "std_ic", "t_stat", "p_value", "n_dates"]]),
        "",
        "## Bootstrap Confidence Intervals",
        "",
        to_markdown(
            bootstrap[
                [
                    "model",
                    "horizon",
                    "annualized_return_ci_lower",
                    "annualized_return_ci_upper",
                    "sharpe_ci_lower",
                    "sharpe_ci_upper",
                    "quantile_spread_ci_lower",
                    "quantile_spread_ci_upper",
                ]
            ]
        ),
        "",
        "## Placebo Comparison",
        "",
        to_markdown(placebo[["model", "horizon", "variant", "mean_ic", "quantile_spread", "annualized_return", "sharpe", "turnover"]]),
        "",
        "## Cost Sweep",
        "",
        to_markdown(cost[["model", "horizon", "cost_bps", "annualized_return", "sharpe", "turnover"]]),
        "",
    ]
    output_path.write_text("\n".join(sections) + "\n")


def _save_phase6_placebo_figure(results: pd.DataFrame) -> Path | None:
    placebo = results[results["validation_type"] == "placebo"].copy()
    if placebo.empty:
        return None
    placebo_metrics = (
        placebo.pivot_table(index=["model", "horizon", "variant"], columns="metric", values="value", aggfunc="first").reset_index()
    )
    if placebo_metrics.empty:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "phase6_placebo_comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ic_plot = placebo_metrics.pivot(index=["model", "horizon"], columns="variant", values="mean_ic")
    sharpe_plot = placebo_metrics.pivot(index=["model", "horizon"], columns="variant", values="sharpe")
    ic_plot.plot(kind="bar", ax=axes[0])
    sharpe_plot.plot(kind="bar", ax=axes[1])
    axes[0].set_title("Mean IC: Real vs Placebo")
    axes[0].set_xlabel("Model / Horizon")
    axes[0].set_ylabel("Mean IC")
    axes[1].set_title("Sharpe: Real vs Placebo")
    axes[1].set_xlabel("Model / Horizon")
    axes[1].set_ylabel("Sharpe")
    for ax in axes:
        ax.legend(title="Series", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_phase6_rolling_ic_figure(rolling_frames: list[pd.DataFrame]) -> Path | None:
    if not rolling_frames:
        return None
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "phase6_rolling_ic.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in rolling_frames:
        ax.plot(frame["date"], frame["rolling_mean_ic"], linewidth=2, label=frame["label"].iloc[0])
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Phase 6 Rolling 2-Year Mean IC")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Mean IC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_phase6_validation_study(args: argparse.Namespace) -> Path:
    market_data = get_market_data(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_ticker=args.benchmark_ticker,
    )
    benchmark_returns = build_benchmark_returns(market_data, benchmark_ticker=args.benchmark_ticker)
    rows: list[dict[str, object]] = []
    rolling_frames: list[pd.DataFrame] = []

    for idx, candidate in enumerate(_phase5_reference_candidates()):
        predictions = _load_phase5_predictions(candidate)
        regression_metrics, ic_by_date = compute_regression_metrics(predictions)
        _, quantile_spread = compute_quantile_analysis(predictions, n_quantiles=10)
        daily_returns, backtest_metrics = _run_reference_backtest(
            predictions=predictions,
            benchmark_returns=benchmark_returns,
            transaction_cost_bps=5.0,
        )

        mean_ic = regression_metrics["daily_information_coefficient_mean"]
        std_ic = regression_metrics["daily_information_coefficient_std"]
        n_dates = len(ic_by_date)
        t_stat = mean_ic / (std_ic / np.sqrt(n_dates)) if std_ic > 0 and n_dates > 1 else float("nan")
        p_value = _normal_approx_two_sided_pvalue(t_stat)
        for metric, value in {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "t_stat": t_stat,
            "p_value": p_value,
            "n_dates": float(n_dates),
        }.items():
            rows.append(
                {
                    "model": candidate["model"],
                    "horizon": candidate["horizon"],
                    "validation_type": "ic_significance",
                    "variant": "real",
                    "metric": metric,
                    "value": value,
                }
            )

        bootstrap_stats = _bootstrap_portfolio_statistics(daily_returns, quantile_spread)
        for metric, value in bootstrap_stats.items():
            rows.append(
                {
                    "model": candidate["model"],
                    "horizon": candidate["horizon"],
                    "validation_type": "bootstrap_ci",
                    "variant": "real",
                    "metric": metric,
                    "value": value,
                }
            )

        placebo_variants = {
            "real": predictions,
            "random_ranking": _make_random_placebo(predictions, seed=BOOTSTRAP_SEED + idx),
            "label_shuffled": _make_label_shuffled_placebo(predictions, seed=BOOTSTRAP_SEED + 100 + idx),
        }
        for variant, placebo_predictions in placebo_variants.items():
            placebo_metrics, _ = compute_regression_metrics(placebo_predictions)
            _, placebo_spread = compute_quantile_analysis(placebo_predictions, n_quantiles=10)
            placebo_daily_returns, placebo_backtest_metrics = _run_reference_backtest(
                predictions=placebo_predictions,
                benchmark_returns=benchmark_returns,
                transaction_cost_bps=5.0,
            )
            summary_metrics = {
                "mean_ic": placebo_metrics["daily_information_coefficient_mean"],
                "quantile_spread": float(placebo_spread["spread_return"].mean()) if not placebo_spread.empty else float("nan"),
                "annualized_return": placebo_backtest_metrics["annualized_return"],
                "sharpe": placebo_backtest_metrics["sharpe_ratio"],
                "turnover": placebo_backtest_metrics["average_turnover"],
            }
            for metric, value in summary_metrics.items():
                rows.append(
                    {
                        "model": candidate["model"],
                        "horizon": candidate["horizon"],
                        "validation_type": "placebo",
                        "variant": variant,
                        "metric": metric,
                        "value": value,
                    }
                )

        for cost_bps in [0, 2, 5, 10, 20]:
            _, cost_metrics = _run_reference_backtest(
                predictions=predictions,
                benchmark_returns=benchmark_returns,
                transaction_cost_bps=float(cost_bps),
            )
            for metric, value in {
                "annualized_return": cost_metrics["annualized_return"],
                "sharpe": cost_metrics["sharpe_ratio"],
                "turnover": cost_metrics["average_turnover"],
            }.items():
                rows.append(
                    {
                        "model": candidate["model"],
                        "horizon": candidate["horizon"],
                        "validation_type": "cost_sweep",
                        "variant": str(cost_bps),
                        "metric": metric,
                        "value": value,
                    }
                )

        rolling = _rolling_ic_series(predictions, window=504)
        rolling["model"] = candidate["model"]
        rolling["horizon"] = candidate["horizon"]
        rolling["label"] = f"{candidate['model']}_h{candidate['horizon']}"
        rolling_frames.append(rolling)
        for _, row in rolling.dropna(subset=["rolling_mean_ic"]).iterrows():
            rows.append(
                {
                    "model": candidate["model"],
                    "horizon": candidate["horizon"],
                    "validation_type": "rolling_ic",
                    "variant": str(row["date"].date()),
                    "metric": "rolling_mean_ic",
                    "value": float(row["rolling_mean_ic"]),
                }
            )

    results = pd.DataFrame(rows).sort_values(["model", "horizon", "validation_type", "variant", "metric"]).reset_index(drop=True)
    csv_path = METRICS_DIR / "phase6_validation_results.csv"
    md_path = METRICS_DIR / "phase6_validation_summary.md"
    results.to_csv(csv_path, index=False)
    _write_phase6_markdown(results, md_path)
    _save_phase6_placebo_figure(results)
    _save_phase6_rolling_ic_figure(rolling_frames)
    return csv_path


def _safe_efficiency_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= SMALL_DENOMINATOR:
        return float("nan")
    return float(numerator / denominator)


def _write_signal_gap_markdown(results: pd.DataFrame, output_path: Path) -> None:
    display = results.copy()
    numeric_cols = [
        "mean_IC",
        "quantile_spread",
        "net_return",
        "Sharpe",
        "turnover",
        "max_drawdown",
        "ranking_efficiency",
        "execution_efficiency",
        "total_efficiency",
    ]
    for col in numeric_cols:
        display[col] = display[col].map(lambda x: "NaN" if pd.isna(x) else f"{x:.4f}")

    headers = list(display.columns)
    lines = [
        "# Signal-to-Implementation Decomposition",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    best_rank = results.sort_values("ranking_efficiency", ascending=False).iloc[0]
    best_exec = results.sort_values("execution_efficiency", ascending=False).iloc[0]
    best_total = results.sort_values("total_efficiency", ascending=False).iloc[0]

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            f"- Ranking efficiency is highest for `{best_rank['model']} {int(best_rank['horizon'])}d`, meaning it converts IC into quantile spread most effectively in this small final set.",
            f"- Execution efficiency is highest for `{best_exec['model']} {int(best_exec['horizon'])}d`, meaning it preserves the largest share of spread into net return.",
            f"- Total efficiency is highest for `{best_total['model']} {int(best_total['horizon'])}d`, but all total-efficiency values remain economically small in absolute terms because full-sample net returns are very small.",
            "- These ratios are descriptive, not causal. They summarize where the observed attenuation occurs, but they do not prove a unique mechanism.",
            "- Ratios become unstable when denominators are near zero; this framework returns `NaN` in that case rather than forcing an interpretation.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def _save_signal_gap_stage_decay_figure(results: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "signal_to_implementation_stage_decay.png"
    labels = [f"{row['model']}_h{int(row['horizon'])}" for _, row in results.iterrows()]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, results["mean_IC"], width=width, label="Mean IC")
    ax.bar(x, results["quantile_spread"], width=width, label="Quantile Spread")
    ax.bar(x + width, results["net_return"], width=width, label="Net Return")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Signal-to-Implementation Stage Decay")
    ax.set_ylabel("Metric Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_signal_gap_scatter_figure(results: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "signal_to_implementation_ic_vs_return.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in results.iterrows():
        label = f"{row['model']}_h{int(row['horizon'])}"
        ax.scatter(row["mean_IC"], row["net_return"], s=90)
        ax.annotate(label, (row["mean_IC"], row["net_return"]), xytext=(6, 4), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_title("Mean IC Versus Net Return")
    ax.set_xlabel("Mean IC")
    ax.set_ylabel("Net Annualized Return")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_signal_to_implementation_gap(args: argparse.Namespace) -> Path:
    phase4_path = METRICS_DIR / "phase4_feature_comparison.csv"
    if not phase4_path.exists():
        raise FileNotFoundError(f"Missing Phase 4 artifact: {phase4_path}")

    phase4 = pd.read_csv(phase4_path)
    reference = phase4[phase4["feature_set"] == "extended"].copy()
    reference = reference[
        reference.apply(
            lambda row: (row["model"], int(row["horizon"])) in {(c["model"], int(c["horizon"])) for c in _phase5_reference_candidates()},
            axis=1,
        )
    ].copy()

    results = (
        reference.rename(columns={"IC": "mean_IC"})[
            ["model", "horizon", "mean_IC", "quantile_spread", "net_return", "Sharpe", "turnover", "max_drawdown"]
        ]
        .sort_values(["model", "horizon"])
        .reset_index(drop=True)
    )
    results["ranking_efficiency"] = results.apply(
        lambda row: _safe_efficiency_ratio(row["quantile_spread"], row["mean_IC"]),
        axis=1,
    )
    results["execution_efficiency"] = results.apply(
        lambda row: _safe_efficiency_ratio(row["net_return"], row["quantile_spread"]),
        axis=1,
    )
    results["total_efficiency"] = results.apply(
        lambda row: _safe_efficiency_ratio(row["net_return"], row["mean_IC"]),
        axis=1,
    )

    csv_path = METRICS_DIR / "signal_to_implementation_decomposition.csv"
    md_path = METRICS_DIR / "signal_to_implementation_decomposition.md"
    results.to_csv(csv_path, index=False)
    _write_signal_gap_markdown(results, md_path)
    _save_signal_gap_stage_decay_figure(results)
    _save_signal_gap_scatter_figure(results)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 research studies.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    horizon_parser = subparsers.add_parser("horizon-study")
    horizon_parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    horizon_parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    horizon_parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    horizon_parser.add_argument("--end-date", default=None)
    horizon_parser.add_argument("--target-mode", choices=["forward_return", "cross_sectional_rank"], default="cross_sectional_rank")
    horizon_parser.add_argument("--models", nargs="+", default=["ridge", "tree", "mlp"])
    horizon_parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    horizon_parser.add_argument("--test-size", type=int, default=63)
    horizon_parser.add_argument("--min-train-size", type=int, default=252)
    horizon_parser.add_argument("--transaction-cost-bps", type=float, default=5.0)

    ranking_parser = subparsers.add_parser("ranking-study")
    ranking_parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ranking_parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    ranking_parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    ranking_parser.add_argument("--end-date", default=None)
    ranking_parser.add_argument("--models", nargs="+", default=["tree", "ranker"])
    ranking_parser.add_argument("--horizons", nargs="+", type=int, default=[10, 20])
    ranking_parser.add_argument("--test-size", type=int, default=63)
    ranking_parser.add_argument("--min-train-size", type=int, default=252)
    ranking_parser.add_argument("--transaction-cost-bps", type=float, default=5.0)

    portfolio_parser = subparsers.add_parser("portfolio-study")
    portfolio_parser.add_argument("--prediction-file", required=True)
    portfolio_parser.add_argument("--benchmark-file", required=True)
    portfolio_parser.add_argument("--transaction-cost-bps", type=float, default=5.0)

    phase3_parser = subparsers.add_parser("phase3-portfolio-study")
    phase3_parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    phase3_parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    phase3_parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    phase3_parser.add_argument("--end-date", default=None)
    phase3_parser.add_argument("--test-size", type=int, default=63)
    phase3_parser.add_argument("--min-train-size", type=int, default=252)
    phase3_parser.add_argument("--transaction-cost-bps", type=float, default=5.0)

    phase4_parser = subparsers.add_parser("phase4-feature-study")
    phase4_parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    phase4_parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    phase4_parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    phase4_parser.add_argument("--end-date", default=None)
    phase4_parser.add_argument("--test-size", type=int, default=63)
    phase4_parser.add_argument("--min-train-size", type=int, default=252)
    phase4_parser.add_argument("--transaction-cost-bps", type=float, default=5.0)

    phase5_parser = subparsers.add_parser("phase5-robustness-study")
    phase5_parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    phase5_parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    phase5_parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    phase5_parser.add_argument("--end-date", default=None)

    phase6_parser = subparsers.add_parser("phase6-validation-study")
    phase6_parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    phase6_parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    phase6_parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    phase6_parser.add_argument("--end-date", default=None)

    signal_gap_parser = subparsers.add_parser("signal-gap-study")
    signal_gap_parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    signal_gap_parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    signal_gap_parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    signal_gap_parser.add_argument("--end-date", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "horizon-study":
        path = run_horizon_study(args)
    elif args.command == "ranking-study":
        path = run_ranking_study(args)
    elif args.command == "phase3-portfolio-study":
        path = run_phase3_portfolio_study(args)
    elif args.command == "phase4-feature-study":
        path = run_phase4_feature_study(args)
    elif args.command == "phase5-robustness-study":
        path = run_phase5_robustness_study(args)
    elif args.command == "phase6-validation-study":
        path = run_phase6_validation_study(args)
    elif args.command == "signal-gap-study":
        path = run_signal_to_implementation_gap(args)
    else:
        path = run_portfolio_study(args)
    print(f"Saved study output to {path}")


if __name__ == "__main__":
    main()
