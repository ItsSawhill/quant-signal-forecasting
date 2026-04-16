from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio import compute_turnover


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"


def run_backtest(
    portfolio_frame: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
    benchmark_returns: pd.DataFrame | None = None,
    rolling_window: int = 63,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a daily long/short backtest using next-day realized returns."""
    frame = portfolio_frame.copy().sort_values(["date", "asset"])
    frame["asset_pnl"] = frame["position"] * frame["next_day_return"]
    daily_returns = frame.groupby("date", as_index=False)["asset_pnl"].sum().rename(columns={"asset_pnl": "gross_return"})

    turnover = compute_turnover(frame[["date", "asset", "position"]])
    daily_returns = daily_returns.merge(turnover, on="date", how="left")
    daily_returns["turnover"] = daily_returns["turnover"].fillna(0.0)
    daily_returns["transaction_cost"] = daily_returns["turnover"] * (transaction_cost_bps / 10000.0)
    daily_returns["net_return"] = daily_returns["gross_return"] - daily_returns["transaction_cost"]
    daily_returns["equity_curve"] = (1.0 + daily_returns["net_return"]).cumprod()

    if benchmark_returns is not None and not benchmark_returns.empty:
        daily_returns = daily_returns.merge(benchmark_returns, on="date", how="left")
        daily_returns["benchmark_next_day_return"] = daily_returns["benchmark_next_day_return"].fillna(0.0)
        daily_returns["benchmark_equity_curve"] = (1.0 + daily_returns["benchmark_next_day_return"]).cumprod()

    daily_returns["rolling_sharpe_63d"] = (
        daily_returns["net_return"].rolling(rolling_window).mean()
        / daily_returns["net_return"].rolling(rolling_window).std(ddof=0).replace(0, np.nan)
    ) * np.sqrt(252)

    n_days = len(daily_returns)
    ending_equity = float(daily_returns["equity_curve"].iloc[-1]) if n_days else 1.0
    annualized_return = ending_equity ** (252 / n_days) - 1.0 if n_days else np.nan
    annualized_volatility = daily_returns["net_return"].std(ddof=0) * np.sqrt(252)
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
    rolling_max = daily_returns["equity_curve"].cummax()
    max_drawdown = (daily_returns["equity_curve"] / rolling_max - 1.0).min()

    metrics = {
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "average_turnover": float(daily_returns["turnover"].mean()),
        "rolling_sharpe_63d_mean": float(daily_returns["rolling_sharpe_63d"].mean(skipna=True)),
    }

    if "benchmark_next_day_return" in daily_returns.columns:
        benchmark_equity = float(daily_returns["benchmark_equity_curve"].iloc[-1]) if n_days else 1.0
        benchmark_annualized_return = benchmark_equity ** (252 / n_days) - 1.0 if n_days else np.nan
        metrics["benchmark_annualized_return"] = float(benchmark_annualized_return)
        metrics["excess_annualized_return_vs_benchmark"] = float(annualized_return - benchmark_annualized_return)

    return daily_returns, metrics


def save_backtest_outputs(daily_returns: pd.DataFrame, metrics: dict[str, float], prefix: str) -> tuple[Path, Path, Path]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    returns_path = METRICS_DIR / f"{prefix}_portfolio_returns.csv"
    metrics_path = METRICS_DIR / f"{prefix}_backtest_metrics.csv"
    figure_path = FIGURES_DIR / f"{prefix}_equity_curve.png"

    daily_returns.to_csv(returns_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_returns["date"], daily_returns["equity_curve"], label="Strategy", linewidth=2)
    if "benchmark_equity_curve" in daily_returns.columns:
        ax.plot(daily_returns["date"], daily_returns["benchmark_equity_curve"], label="SPY Benchmark", alpha=0.8)
        ax.legend()
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return returns_path, metrics_path, figure_path
