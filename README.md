# quant-signal-forecasting

`quant-signal-forecasting` is a portfolio-ready quant research project for cross-sectional equity signal modeling. v2 extends the original ETF demo into a more realistic research workflow with a larger equity universe, rank-based targets, richer factor-style features, configurable long/short portfolio construction, and benchmark-aware evaluation.

## Overview

This repository is designed to demonstrate research process rather than marketable alpha. The emphasis is on:

- leakage-aware feature and label construction
- chronological walk-forward validation
- cross-sectional ranking rather than point forecast marketing
- realistic interpretation of weak but plausible signal evidence
- clear artifacts for review: metrics, notebook analysis, and benchmark-aware backtests

## Key Takeaways

- v2 moves the project from ETF forecasting into a more relevant cross-sectional equity research setup.
- Cross-sectional rank targets are more aligned with quant portfolio construction than raw return regression alone.
- The latest verified v2 run shows weak but plausible signal quality: slightly positive mean IC and directionally sensible quantile spreads, but negative after-cost performance versus `SPY`.
- This is a research framework, not a deployable strategy, and the README is written to reflect that directly.

## What Changed In v2

Compared with v1, the project now supports:

- a larger liquid U.S. equity universe instead of only sector/index ETFs
- cross-sectional ranking targets in addition to raw forward-return targets
- richer research features including rolling beta, rolling correlation, downside volatility, volatility-adjusted momentum, and cross-sectional z-scores
- quantile portfolios, signal-weighted portfolios, configurable gross exposure, and turnover controls
- less overlapping portfolio construction via configurable rebalance frequency
- stronger evaluation artifacts including IC by date, quantile return spreads, rolling Sharpe, and benchmark comparison

## Research Question

Can simple tabular models learn a useful cross-sectional signal from price-and-volume features across a liquid U.S. equity universe, and is a rank-based learning target more useful for long/short research than forecasting raw forward returns directly?

## Why Ranking Matters

For cross-sectional equity research, the exact return forecast is often less important than the relative ordering of names on each date. Rank-based targets are more aligned with how many quant portfolios are actually built:

- signals are used to sort names into long and short baskets
- small forecast misspecification matters less if relative ordering is preserved
- information coefficient and quantile spread are often more informative than raw error alone

v2 therefore supports both raw forward-return targets and cross-sectional rank targets, with `cross_sectional_rank` as the default research path.

## Default v2 Setup

- Universe: 58 liquid U.S. large-cap equities
- Benchmark: `SPY`
- Default history: `2018-01-01` to present
- Label horizon: next 5 trading days
- Default target: cross-sectional rank of next 5-day forward return
- Default model: `ridge`
- Default portfolio: signal-weighted long top decile / short bottom decile
- Default controls: one-day implementation lag, turnover cap, transaction costs, benchmark-aware reporting

## Methodology

### Universe

The v2 universe is a laptop-friendly basket of liquid U.S. large-cap stocks across technology, healthcare, financials, consumer, industrials, and energy. The benchmark `SPY` is downloaded alongside the universe for relative features and benchmark comparison.

### Features

The feature set now includes both time-series and cross-sectional signals:

- trailing returns: `1d`, `5d`, `10d`, `20d`
- rolling volatility: `5d`, `20d`
- momentum and moving-average gaps
- rolling volume z-score
- rolling Sharpe proxy
- drawdown from 20-day high
- downside volatility
- rolling beta to `SPY`
- rolling correlation to `SPY`
- volatility-adjusted momentum
- cross-sectional z-scored versions of selected features

### Targets

Supported targets:

- `forward_return`: raw next 5-day return
- `cross_sectional_rank`: daily cross-sectional rank of next 5-day return
- `forward_return_binary`: classification label for positive next 5-day return

### Validation And Leakage Controls

The core chronology controls from v1 remain in place:

- features use trailing windows only
- model features are shifted by one day before training
- train/validation/test splits are expanding and chronological
- positions are lagged by one day before PnL application
- the written leakage review remains available in [outputs/metrics/leakage_alignment_audit.md](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/leakage_alignment_audit.md)

## Key Results From The Verified v2 Run

The latest verified v2 run used:

```bash
python src/train.py --model ridge --task regression --target-mode cross_sectional_rank
```

Actual outputs from that run:

- prediction metrics in `outputs/metrics/ridge_regression_cross_sectional_rank_metrics_summary.csv`
- backtest metrics in `outputs/metrics/ridge_regression_cross_sectional_rank_backtest_metrics.csv`
- IC series in `outputs/metrics/ridge_regression_cross_sectional_rank_ic_by_date.csv`
- quantile return summaries in `outputs/metrics/ridge_regression_cross_sectional_rank_quantile_returns.csv`
- quantile spread series in `outputs/metrics/ridge_regression_cross_sectional_rank_quantile_spread_returns.csv`

Headline metrics from the verified run:

| Metric | Value |
| --- | ---: |
| RMSE | 0.2891 |
| MAE | 0.2501 |
| Target Correlation | 0.0146 |
| Spearman Rank Correlation | 0.0113 |
| Mean Daily IC | 0.0088 |
| Annualized Return | -0.0105 |
| Annualized Volatility | 0.0904 |
| Sharpe Ratio | -0.1160 |
| Max Drawdown | -0.3469 |
| Average Turnover | 0.0323 |
| Benchmark Annualized Return (`SPY`) | 0.1538 |

Quantile results are directionally encouraging:

- mean realized return increases from the bottom decile to the top decile
- top decile mean return: `0.00537`
- bottom decile mean return: `0.00383`

But the after-cost portfolio result is still negative in this default configuration, which is the correct interpretation to keep front and center.

## Interpretation

What v2 improves:

- the research setup is much closer to a real cross-sectional equity signal pipeline
- rank-based targets are more aligned with long/short portfolio construction
- the new feature set better reflects common quant factor engineering patterns
- the backtest now uses next-day realized returns with held positions instead of booking only overlapping horizon returns

What the latest results say:

- the signal is weak but not random: mean IC is slightly positive and quantile returns are directionally monotonic
- the default ridge rank model is not yet strong enough to overcome costs and beat the benchmark
- v2 is therefore a stronger research project even though the default backtest outcome is not flattering

That is a stronger portfolio outcome than an unrealistically strong backtest claim, because it shows research depth and technical honesty.

## Portfolio Construction In v2

Portfolio construction now supports:

- `topk` or `quantile` portfolio selection
- `equal` or `signal` weighting
- configurable gross exposure
- configurable turnover cap
- configurable rebalance frequency

The default v2 run uses a signal-weighted quantile portfolio with a turnover cap, which is more realistic than the original overlapping equal-weight approximation.

## Limitations

Remaining realism caveats are explicit:

- this is still a research-grade backtest, not an execution-grade simulator
- `yfinance` is acceptable for research demos but not institutional data engineering
- benchmark comparison versus `SPY` is informative, but not a like-for-like comparison against a market-neutral strategy
- no borrow fees, slippage model, or path-level execution engine is included
- no hyperparameter search or nested research protocol is included

## How To Run

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the default v2 experiment:

```bash
python src/train.py --model ridge --task regression --target-mode cross_sectional_rank
```

Run raw forward-return regression instead of rank-based learning:

```bash
python src/train.py --model ridge --task regression --target-mode forward_return
```

Try an alternative portfolio configuration:

```bash
python src/train.py --model ridge --task regression --target-mode cross_sectional_rank --portfolio-mode topk --top-k 10 --weight-scheme equal --max-turnover 0.25
```

## Key Files

- [src/train.py](/Users/sahil/Desktop/quant-signal-forecasting/src/train.py)
- [src/features.py](/Users/sahil/Desktop/quant-signal-forecasting/src/features.py)
- [src/labels.py](/Users/sahil/Desktop/quant-signal-forecasting/src/labels.py)
- [src/portfolio.py](/Users/sahil/Desktop/quant-signal-forecasting/src/portfolio.py)
- [src/backtest.py](/Users/sahil/Desktop/quant-signal-forecasting/src/backtest.py)
- [outputs/metrics/ridge_regression_cross_sectional_rank_metrics_summary.csv](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/ridge_regression_cross_sectional_rank_metrics_summary.csv)
- [outputs/metrics/ridge_regression_cross_sectional_rank_backtest_metrics.csv](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/ridge_regression_cross_sectional_rank_backtest_metrics.csv)
- [outputs/metrics/ridge_regression_cross_sectional_rank_ic_by_date.csv](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/ridge_regression_cross_sectional_rank_ic_by_date.csv)
- [outputs/metrics/ridge_regression_cross_sectional_rank_quantile_returns.csv](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/ridge_regression_cross_sectional_rank_quantile_returns.csv)
- [outputs/metrics/ridge_regression_cross_sectional_rank_quantile_spread_returns.csv](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/ridge_regression_cross_sectional_rank_quantile_spread_returns.csv)

## Highlights

- Built an end-to-end cross-sectional equity signal research pipeline spanning data ingestion, feature engineering, rank-target labeling, walk-forward training, portfolio construction, and benchmark-aware backtesting.
- Extended a simple ETF forecasting demo into a more realistic quant research workflow with cross-sectional ranking targets, factor-style features, and configurable long/short portfolio mechanics.
- Evaluated weak but plausible signal quality using information coefficient, quantile spread analysis, and benchmark comparison while explicitly documenting leakage controls and realism caveats.

## What Still Belongs In v3

If this were taken beyond portfolio/demo scope, the next steps would be:

- more robust universe maintenance and delisting handling
- sector/industry neutralization
- benchmark-relative and residual return targets
- rolling hyperparameter search
- a cleaner separation between signal evaluation and execution simulation
