# quant-signal-forecasting

`quant-signal-forecasting` is a compact quant research project for cross-sectional ETF signal forecasting and long/short portfolio backtesting. It downloads daily market data, engineers leakage-aware features, trains chronological forecasting models, converts predictions into ranked portfolio signals, and saves research artifacts for review.

## Why This Repo Exists

The goal is not to claim production-ready alpha. The goal is to show a credible research workflow:

- data ingestion from a public source
- leakage-aware feature and label construction
- walk-forward model evaluation
- cross-sectional signal ranking
- transaction-cost-aware backtesting
- artifact generation for review and comparison

The implementation is intentionally lightweight enough to run on a laptop while still reflecting the structure of a real quant research exercise.

## Data

- Source: `yfinance`
- Universe: `SPY`, `QQQ`, `IWM`, `XLF`, `XLK`, `XLV`, `XLE`, `XLI`, `XLP`, `XLY`, `XLU`
- Frequency: daily adjusted OHLCV
- Default start date: `2015-01-01`

Downloaded data is saved to `data/market_data.csv`. The joined modeling dataset is saved to `data/modeling_dataset.csv`.

## Features And Labels

Features are engineered per asset using trailing information only:

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

## Models

- `ridge`: `Ridge` / `LogisticRegression`
- `tree`: `XGBoost` if available, otherwise sklearn gradient boosting
- `mlp`: PyTorch MLP with early stopping and checkpointing, otherwise sklearn MLP fallback

All training uses chronological expanding-window evaluation. There is no random train/test split.

## Leakage And Alignment Check

The current code path was reviewed specifically for lookahead bias.

- Feature engineering is based on trailing windows and then shifted by one day in [src/features.py](/Users/sahil/Desktop/quant-signal-forecasting/src/features.py), so model inputs at date `t` only use information available through `t-1`.
- Labels in [src/labels.py](/Users/sahil/Desktop/quant-signal-forecasting/src/labels.py) use `close[t+5] / close[t] - 1`, which is a standard forward-return target.
- Walk-forward prediction in [src/train.py](/Users/sahil/Desktop/quant-signal-forecasting/src/train.py) keeps validation and test periods strictly after training periods.
- Portfolio weights in [src/portfolio.py](/Users/sahil/Desktop/quant-signal-forecasting/src/portfolio.py) are shifted by one day before being applied, so positions are implemented with a lag.
- Backtest PnL in [src/backtest.py](/Users/sahil/Desktop/quant-signal-forecasting/src/backtest.py) uses lagged positions times forward returns and applies linear turnover costs.

Conclusion:

- I did not find an explicit lookahead bug in the current pipeline.
- The main remaining realism caveat is not leakage but approximation: the strategy uses overlapping 5-day forward returns together with daily rebalancing and rolling average holdings. That is acceptable for a lightweight research project, but it is still a simplified execution model rather than a path-accurate portfolio simulator.

## Key Results

The following results are from actual generated outputs in `outputs/metrics/model_comparison_regression.csv` after running the full regression pipeline for all three models on the default ETF universe.

| Model | RMSE | MAE | Corr | Spearman | Mean IC | Ann. Return | Ann. Vol | Sharpe | Max DD | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.0287 | 0.0201 | 0.0243 | 0.0612 | 0.0071 | 0.2001 | 0.2652 | 0.7544 | -0.7730 | 0.4185 |
| Tree | 0.0295 | 0.0204 | 0.0027 | 0.0509 | 0.0022 | -0.0480 | 0.2269 | -0.2117 | -0.7492 | 0.5136 |
| MLP | 0.0298 | 0.0206 | 0.0391 | 0.0367 | -0.0068 | -0.0854 | 0.2223 | -0.3840 | -0.7987 | 0.5308 |

## Brief Interpretation

- `ridge` is the strongest of the three models on this run. The edge is modest rather than dramatic, which is more believable for a simple feature set on liquid ETFs.
- `tree` and `mlp` do not improve on the linear baseline out of sample and both are negative after transaction costs.
- The drawdowns are large across all models. That weakens the practical attractiveness of the strategy even when prediction metrics are mildly positive.
- No full-sample result looks suspiciously strong. Earlier short-window sanity checks looked much better, but the full default-history run with compounded annualization is materially more conservative.

## Model Comparison Artifacts

Generated artifacts include:

- `outputs/metrics/model_comparison_regression.csv`
- `outputs/metrics/model_comparison_regression.md`
- `outputs/predictions/*_regression_predictions.csv`
- `outputs/metrics/*_regression_metrics_summary.csv`
- `outputs/metrics/*_regression_backtest_metrics.csv`
- `outputs/metrics/*_regression_portfolio_returns.csv`
- `outputs/figures/*_regression_equity_curve.png`
- `outputs/figures/*_regression_predicted_vs_realized.png`
- `outputs/figures/tree_regression_feature_importance.png`

## Limitations

- `yfinance` is convenient for research but not a production-grade market data source.
- The ETF universe is fixed and hand-selected, so this is not a full universe-selection study.
- The cost model is simple linear turnover cost and does not include slippage, borrow costs, financing, or market impact.
- The backtest uses forward returns instead of full path-level holdings and fills, so it is a research approximation rather than a full execution simulator.
- No hyperparameter search or nested walk-forward tuning is included.
- The current repo is aimed at credibility and clarity, not maximizing performance.

## Exact Commands To Reproduce Outputs

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the three regression models:

```bash
python src/train.py --model ridge --task regression
python src/train.py --model tree --task regression
python src/train.py --model mlp --task regression
```

Open the notebook for result review:

```bash
jupyter notebook notebooks/model_review.ipynb
```

## Repository Layout

```text
quant-signal-forecasting/
├── data/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── labels.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── portfolio.py
│   └── backtest.py
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── README.md
└── requirements.txt
```

## Future Improvements

- add path-accurate holding-period simulation instead of return-based approximation
- expand the feature set with regime, factor, and macro features
- add rolling hyperparameter search with a stricter research protocol
- separate signal evaluation from portfolio design more cleanly
- add benchmark comparisons and factor-neutralization checks
