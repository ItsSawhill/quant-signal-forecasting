# v2 Baseline Freeze

## Inspection Summary

### Current Targets

- `forward_return_{5,10,20}d`
- `forward_return_binary_{5,10,20}d`
- `forward_return_rank_{5,10,20}d`

Default research path:

- task: `regression`
- target mode: `cross_sectional_rank`
- horizon: `5`

### Default Universe

- 58 liquid U.S. large-cap equities
- benchmark: `SPY`
- default start date: `2018-01-01`
- cached local data currently spans `2018-01-02` through `2026-04-16`

### Current Features

- trailing returns: `1d`, `5d`, `10d`, `20d`
- rolling volatility: `5d`, `20d`
- momentum: `10d`, `20d`
- moving-average gaps: `10d`, `20d`
- rolling volume z-score
- rolling Sharpe proxy
- drawdown from 20-day high
- downside volatility
- rolling beta to `SPY`
- rolling correlation to `SPY`
- volatility-adjusted momentum
- market-relative daily return
- cross-sectional z-scores for selected factor-style features

### Current Models

- `ridge`
- `tree`
- `mlp`
- `ranker` is partially present in code, but it is not part of the verified v2 benchmark

### Current Portfolio Construction Rules

- portfolio mode default: `quantile`
- weight scheme default: `signal`
- long top decile / short bottom decile
- gross exposure: `1.0`
- turnover cap: `0.5`
- holding horizon default: `5`
- rebalance frequency default: `holding_horizon`
- one-day implementation lag before returns are applied

### Current Evaluation Metrics

- RMSE
- MAE
- correlation
- Spearman rank correlation
- IC by date
- mean and volatility of daily IC
- bucket returns
- quantile returns
- quantile spread returns
- annualized gross return
- annualized net return
- annualized volatility
- Sharpe ratio
- max drawdown
- average turnover
- benchmark annualized return
- excess annualized return vs benchmark

## What Is Already Implemented Versus Missing

### Already Implemented

- market data ingestion
- feature engineering
- forward return and rank targets
- ridge / tree / mlp models
- walk-forward validation
- portfolio construction and backtesting
- leakage / alignment audit
- benchmark-aware reporting
- multi-horizon labels in the current codebase
- CLI horizon selection in `src/train.py`

### Missing Or Incomplete Before This Pass

- a reproducible cached-data path for local reruns without a live `yfinance` download
- a frozen benchmark artifact tied to regenerated files
- a clean horizon-study comparison artifact across the current model family
- complete per-run artifact generation from the research script

## Official v2 Benchmark

Reference command:

```bash
env MPLCONFIGDIR=/tmp/mpl python src/train.py --model ridge --task regression --target-mode cross_sectional_rank --horizon 5
```

Reference artifact set:

- `outputs/predictions/ridge_regression_cross_sectional_rank_h5_predictions.csv`
- `outputs/metrics/ridge_regression_cross_sectional_rank_h5_metrics_summary.csv`
- `outputs/metrics/ridge_regression_cross_sectional_rank_h5_ic_by_date.csv`
- `outputs/metrics/ridge_regression_cross_sectional_rank_h5_quantile_returns.csv`
- `outputs/metrics/ridge_regression_cross_sectional_rank_h5_quantile_spread_returns.csv`
- `outputs/metrics/ridge_regression_cross_sectional_rank_h5_portfolio_returns.csv`
- `outputs/metrics/ridge_regression_cross_sectional_rank_h5_backtest_metrics.csv`

Benchmark specification:

- universe: default 58-stock large-cap universe from `src/data_loader.py`
- start date: `2018-01-01`
- target mode: `cross_sectional_rank`
- horizon: `5`
- feature set: `FEATURE_COLUMNS` from `src/features.py`
- model list for baseline family: `ridge`, `tree`, `mlp`
- portfolio defaults: quantile, signal-weighted, top/bottom decile, gross exposure `1.0`, max turnover `0.5`, holding horizon `5`
- transaction cost assumption: `5 bps`

Observed v2 benchmark metrics from the regenerated run:

- mean daily IC: `0.0087`
- IC volatility: `0.2516`
- Spearman rank correlation: `0.0113`
- annualized return: `-0.0105`
- Sharpe ratio: `-0.1160`
- max drawdown: `-0.3469`
- average turnover: `0.0323`
- benchmark annualized return: `0.1538`
- top quantile mean realized return: `0.00537`
- bottom quantile mean realized return: `0.00383`

## Why Freeze The Benchmark First

Freezing the benchmark matters because the repo already has several moving parts that can change results materially:

- the target definition
- the forecast horizon
- the portfolio construction rule
- the turnover cap
- the transaction cost assumption

Without a frozen reference, later improvements are not attributable. A stronger IC could come from a different horizon rather than a better model. A better backtest could come from looser turnover rather than a better signal. Freezing the v2 benchmark first keeps the v3 work technically honest.

## Why Not Jump Straight To RL, Larger Nets, Or More Indicators

That would be premature for this repo.

- RL adds objective and simulation complexity before the supervised ranking baseline is fully characterized.
- Larger neural nets increase compute cost and reduce interpretability even though the current evidence suggests horizon choice is a larger lever.
- Adding many more indicators would blur attribution and make the research less defensible.

The next disciplined step is to test whether a better horizon improves implementability under the same baseline models and the same backtest rules.
