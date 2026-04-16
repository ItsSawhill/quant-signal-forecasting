# quant-signal-forecasting

`quant-signal-forecasting` is a research-style Python project for cross-sectional market signal forecasting and portfolio backtesting. It downloads ETF market data, engineers predictive features with strict no-lookahead alignment, trains baseline and nonlinear models, converts forecasts into long/short portfolio signals, and saves artifacts for downstream analysis.

## Motivation

This repo is designed to resemble a compact quant research workflow:

- ingest liquid market data
- turn price and volume history into predictive features
- define forward-return labels
- evaluate models using time-aware validation
- translate model scores into tradable portfolio signals
- backtest a simple market-neutral strategy with transaction costs

The defaults are intentionally lightweight enough to run on a laptop while still reflecting the structure of a real research pipeline.

## Data Source

The project uses `yfinance` to download adjusted daily OHLCV data for a default ETF universe:

- `SPY`, `QQQ`, `IWM`
- `XLF`, `XLK`, `XLV`, `XLE`
- `XLI`, `XLP`, `XLY`, `XLU`

The default date range is `2015-01-01` through today. Downloaded and cleaned market data is saved to `data/market_data.csv`.

## Feature Set

Features are built per asset using only information available up to each date:

- trailing returns: `1d`, `5d`, `10d`, `20d`
- rolling volatility: `5d`, `20d`
- momentum: `10d`, `20d`
- moving-average gap vs `10d` and `20d` averages
- rolling volume z-score
- rolling Sharpe proxy
- drawdown from rolling `20d` high
- optional market-relative return vs `SPY`

## Labels

Two label modes are supported:

- regression: next `5`-day forward return
- classification: whether next `5`-day forward return is positive

Labels are aligned by date and asset and saved alongside the modeling dataset.

## Modeling Approach

The training pipeline supports three model families:

- linear baseline: `Ridge` or `LogisticRegression`
- tree model: `XGBoost` if installed, otherwise sklearn gradient boosting
- neural net: PyTorch MLP with early stopping and checkpointing when `torch` is available, otherwise sklearn MLP fallback

Each model is trained using a time-based split. Predictions are generated in a walk-forward / expanding-window manner so the test slice always occurs after the training slice.

## Validation Design

The repo avoids leakage by:

- building features from trailing windows only
- shifting labels forward by the holding horizon
- using expanding-window training with chronological test blocks
- applying portfolio positions with a one-day implementation lag

This design keeps the backtest closer to how a research forecast would actually be deployed.

## Backtest Logic

Portfolio construction ranks assets each day by model prediction:

- long top `3`
- short bottom `3`
- equal-weight long/short
- configurable transaction costs
- configurable holding horizon

The backtest computes:

- gross and net daily returns
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- turnover

## Outputs

Running the pipeline writes artifacts to:

- `outputs/predictions/`: out-of-sample predictions
- `outputs/metrics/`: evaluation metrics, backtest stats, portfolio returns, positions
- `outputs/figures/`: equity curve, predicted-vs-realized plot, tree feature importances when available

## Repo Structure

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

## How To Run

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the default regression pipeline.

```bash
python src/train.py
```

3. Run an alternative model or classification task.

```bash
python src/train.py --model tree --task regression
python src/train.py --model mlp --task classification
```

## Useful Commands

```bash
python src/train.py --tickers SPY QQQ IWM XLF XLK XLV XLE XLI XLP XLY XLU
python src/train.py --start-date 2018-01-01 --horizon 5 --top-k 3 --transaction-cost-bps 5
python src/train.py --model tree --test-size 63 --min-train-size 252
```

## Notes

- `xgboost` is used automatically when installed; otherwise the tree model falls back to sklearn gradient boosting.
- The neural net uses a PyTorch MLP with early stopping and checkpointing when `torch` is available. If not, sklearn MLP is used so the pipeline remains runnable.
- Output files are overwritten on each run for the selected configuration.

## Future Improvements

- add richer macro, factor, or alternative data inputs
- support multi-horizon and multi-task targets
- add probability calibration and position sizing
- introduce more realistic slippage and borrow-cost modeling
- expand to rolling hyperparameter search
- add notebook-based experiment tracking and report generation
