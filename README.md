# quant-signal-forecasting

`quant-signal-forecasting` is a compact quant research project for cross-sectional ETF return forecasting and long/short portfolio backtesting. It demonstrates a full research loop: market data ingestion, leakage-aware feature engineering, forward-return labeling, walk-forward model evaluation, signal ranking, and transaction-cost-aware backtesting.

## Overview

This repository is designed as a portfolio-quality research sample rather than a production trading system. The focus is on showing sound research process:

- chronological validation instead of random splits
- explicit leakage and alignment review
- comparable baseline and nonlinear models
- saved artifacts for model review and backtest inspection

## Key Takeaways

- `ridge` currently outperforms `tree` and `mlp` on this ETF setup.
- The predictive edge is modest, which is more credible than a dramatic backtest.
- The backtest is research-grade, not execution-grade: it is useful for signal evaluation, but it is still a simplified trading approximation.

## Research Question

Can a lightweight set of price-and-volume features generate a usable cross-sectional signal across major sector and index ETFs, and how do simple linear, tree-based, and neural baselines compare under walk-forward evaluation?

## Data, Features, And Targets

- Data source: `yfinance`
- Universe: `SPY`, `QQQ`, `IWM`, `XLF`, `XLK`, `XLV`, `XLE`, `XLI`, `XLP`, `XLY`, `XLU`
- Frequency: daily adjusted OHLCV
- Default history: `2015-01-01` to present

Feature set:

- trailing returns: `1d`, `5d`, `10d`, `20d`
- rolling volatility: `5d`, `20d`
- momentum: `10d`, `20d`
- moving-average gaps: `10d`, `20d`
- rolling volume z-score
- rolling Sharpe proxy
- drawdown from 20-day high
- optional market-relative return vs `SPY`

Targets:

- regression: next 5-day forward return
- classification: next 5-day return greater than zero

## Validation Design

The current implementation was explicitly reviewed for lookahead bias.

- Features are built from trailing windows and then shifted by one day before model use.
- Labels are forward returns built after feature construction.
- Train, validation, and test windows are chronological and expanding.
- Portfolio positions are lagged by one day before return application.
- Transaction costs are applied using realized turnover.

Leakage review summary:

- No explicit lookahead bug was found in the current pipeline.
- The main realism caveat is execution approximation, not leakage.

See [outputs/metrics/leakage_alignment_audit.md](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/leakage_alignment_audit.md) for the written audit.

## Models

- `ridge`: linear baseline using `Ridge`
- `tree`: `XGBoost` when available, otherwise sklearn gradient boosting
- `mlp`: PyTorch MLP with early stopping and checkpointing, otherwise sklearn MLP fallback

All reported results below come from actual walk-forward regression runs saved in `outputs/`.

## Key Results

On this setup, `ridge` is the strongest model overall. Its signal is still weak in absolute terms, but it is the most stable of the three and the only model with positive after-cost backtest performance in the full run.

| Model | RMSE | MAE | Corr | Spearman | Mean IC | Ann. Return | Ann. Vol | Sharpe | Max DD | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.0287 | 0.0201 | 0.0243 | 0.0612 | 0.0071 | 0.2001 | 0.2652 | 0.7544 | -0.7730 | 0.4185 |
| Tree | 0.0295 | 0.0204 | 0.0027 | 0.0509 | 0.0022 | -0.0480 | 0.2269 | -0.2117 | -0.7492 | 0.5136 |
| MLP | 0.0298 | 0.0206 | 0.0391 | 0.0367 | -0.0068 | -0.0854 | 0.2223 | -0.3840 | -0.7987 | 0.5308 |

Source artifacts:

- [outputs/metrics/model_comparison_regression.csv](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/model_comparison_regression.csv)
- [outputs/metrics/model_comparison_regression.md](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/model_comparison_regression.md)

## Model Comparison

What stands out from the comparison:

- `ridge` has the best RMSE, MAE, rank correlation, mean IC, and Sharpe ratio.
- `tree` and `mlp` do not beat the linear baseline out of sample on this feature set.
- The more flexible models carry higher turnover and still fail to convert into better after-cost outcomes.

## Interpretation

- The signal appears real enough to study, but not strong enough to overstate.
- The best model here is the simplest one, which is a useful research result in itself.
- Large drawdowns across all three models are a reminder that weak cross-sectional predictability does not automatically translate into a robust tradable strategy.
- No result in the full default-history run looks suspiciously strong.

## Portfolio-Facing Artifacts

Tracked artifacts that support the narrative:

- [outputs/figures/ridge_regression_equity_curve.png](/Users/sahil/Desktop/quant-signal-forecasting/outputs/figures/ridge_regression_equity_curve.png)
- [outputs/figures/ridge_regression_predicted_vs_realized.png](/Users/sahil/Desktop/quant-signal-forecasting/outputs/figures/ridge_regression_predicted_vs_realized.png)
- [outputs/metrics/model_comparison_regression.csv](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/model_comparison_regression.csv)
- [outputs/metrics/leakage_alignment_audit.md](/Users/sahil/Desktop/quant-signal-forecasting/outputs/metrics/leakage_alignment_audit.md)
- [notebooks/model_review.ipynb](/Users/sahil/Desktop/quant-signal-forecasting/notebooks/model_review.ipynb)

The working tree may also contain additional generated figures and metrics from local runs that are intentionally not tracked in git.

## Limitations

- `yfinance` is convenient for research, not a production-grade market data source.
- The ETF universe is fixed and small, so this is not a full universe-selection study.
- The backtest is research-grade, not execution-grade.
- Returns are evaluated with a simplified holding approximation rather than a path-accurate execution simulator.
- Transaction costs are linear turnover costs only and do not include slippage, borrow costs, or financing.
- No hyperparameter search or nested walk-forward tuning is included.

## How To Run

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the regression models:

```bash
python src/train.py --model ridge --task regression
python src/train.py --model tree --task regression
python src/train.py --model mlp --task regression
```

Review saved outputs:

```bash
jupyter notebook notebooks/model_review.ipynb
```

## Resume-Ready Highlights

- Built an end-to-end quant research pipeline for ETF return forecasting, including data ingestion, feature engineering, forward-return labeling, walk-forward model evaluation, and portfolio backtesting.
- Implemented leakage-aware validation and documented alignment assumptions through a dedicated audit of features, labels, prediction timing, and portfolio construction.
- Compared linear, tree-based, and neural baselines on a common research setup and showed that a simple ridge model outperformed more flexible alternatives out of sample.
