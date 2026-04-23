# quant-signal-forecasting

`quant-signal-forecasting` is a research-oriented Python repository for cross-sectional equity signal forecasting, ranking-aware modeling, portfolio construction, and implementation-aware validation.

The project studies a central question:

> Can direct cross-sectional ranking across multiple horizons produce more implementable equity signals than standard regression baselines after turnover and transaction costs?

The final conclusion is deliberately conservative. The repository finds evidence of a weak cross-sectional ranking signal, especially in information-coefficient and quantile-spread metrics, but the signal is not robust or economically meaningful enough to support a tradable-alpha claim in the current setup.

## Project Scope

The repository is designed as a reproducible research framework, not a production trading system. It includes:

- market data ingestion from `yfinance`
- leakage-aware feature engineering
- forward-return and cross-sectional rank target construction
- ridge, tree, MLP, and ranking-aware model options
- walk-forward validation
- configurable long/short portfolio construction
- transaction-cost-aware backtesting
- robustness, placebo, and signal-to-implementation validation studies
- a full paper draft and phase-by-phase research notes

## Main Findings

- Ranking-aware learning improves cross-sectional ordering quality relative to tree regression baselines, but it is not a universal after-cost winner.
- Portfolio construction choices explain more variation in realized performance than the remaining model differences.
- Small market-relative feature additions slightly improve signal metrics, but do not materially improve implemented returns.
- Robustness checks show strong regime dependence: results are negative in early subperiods and positive only in the most recent regime.
- Statistical validation supports a weak signal in IC terms, but portfolio-level bootstrap intervals remain too wide to justify an economic alpha claim.
- The signal-to-implementation decomposition shows that predictive structure is attenuated between ranking quality and realized net return.

## Repository Structure

```text
quant-signal-forecasting/
├── docs/                 # Phase notes and research documentation
├── notebooks/            # Review notebook
├── outputs/
│   ├── figures/          # Selected research figures
│   └── metrics/          # Selected metrics and summary tables
├── reports/              # Research paper draft and report artifacts
├── src/                  # Core pipeline and research commands
├── README.md
└── requirements.txt
```

## Data And Universe

The default research universe is a laptop-friendly basket of 58 liquid U.S. large-cap equities, with `SPY` used as the market benchmark and relative-feature reference. The default research window begins on `2018-01-01`.

Data is sourced from `yfinance`, which is appropriate for a reproducible research demonstration but not sufficient for institutional-grade trading research. The project notes survivorship, data-quality, and execution-model limitations explicitly.

## Features

The baseline feature set includes:

- trailing returns over `1d`, `5d`, `10d`, and `20d`
- rolling volatility over `5d` and `20d`
- momentum and moving-average gaps
- rolling volume z-scores
- rolling Sharpe proxy
- drawdown from a 20-day high
- downside volatility
- rolling beta and correlation to `SPY`
- volatility-adjusted momentum
- cross-sectional z-scored variants of selected features

Phase 4 adds a small controlled set of market-relative features:

- residual return vs `SPY`
- rolling excess return vs `SPY`
- beta-adjusted momentum

## Targets

Supported target modes include:

- `forward_return`: raw forward return
- `cross_sectional_rank`: centered cross-sectional percentile rank of forward return
- `forward_return_binary`: positive-forward-return classification target

The main research path uses `cross_sectional_rank` because it is aligned with long/short equity ranking and quantile portfolio construction.

## Models

The implemented model options are:

- `ridge`: linear regression/classification baseline
- `tree`: tree-based nonlinear baseline
- `mlp`: lightweight tabular MLP path
- `ranker`: ranking-aware learner for cross-sectional rank targets

The project intentionally avoids adding larger architectures such as transformers or reinforcement-learning agents because later phases show that the main bottleneck is implementation and robustness, not model capacity.

## Validation And Backtesting

The pipeline uses chronological walk-forward validation. Features are based on trailing windows, model inputs are shifted to avoid lookahead, and positions are lagged before applying next-day returns.

Portfolio construction supports:

- quantile and top-k selection
- equal and signal weighting
- configurable gross exposure
- configurable rebalance frequency
- optional turnover cap
- linear transaction costs

Backtests report annualized return, volatility, Sharpe ratio, max drawdown, turnover, benchmark comparison, and return per unit turnover where applicable.

## Research Phases

The final project is organized into six research phases:

| Phase | Purpose | Summary |
| --- | --- | --- |
| Phase 0 | Baseline freeze | Freezes the v2 benchmark before further changes. |
| Phase 1 | Multi-horizon study | Tests `5d`, `10d`, and `20d` rank targets. |
| Phase 2 | Ranking-aware learning | Compares tree regression with a ranking-aware learner. |
| Phase 3 | Portfolio construction sensitivity | Tests selection, weighting, rebalance, and turnover controls. |
| Phase 4 | Controlled feature research | Adds a small market-relative feature extension. |
| Phase 5 | Robustness analysis | Tests subperiods, transaction costs, and small portfolio variations. |
| Phase 6 | Statistical validation | Adds IC significance, bootstrap, placebo, cost-sweep, and rolling-stability checks. |

The project also includes a final signal-to-implementation decomposition that summarizes how signal quality is attenuated from IC to quantile spread to realized net return.

## Key Artifacts

- Phase notes: `docs/`
- Paper draft: `reports/quant_signal_paper_draft.md`
- Baseline freeze: `docs/v3_baseline_freeze.md`
- Phase 6 validation note: `docs/v3_phase6_validation_note.md`
- Signal-to-implementation note: `docs/v3_signal_to_implementation_gap_note.md`
- Signal decomposition table: `outputs/metrics/signal_to_implementation_decomposition.md`
- Portfolio sensitivity figure: `outputs/figures/phase3_portfolio_sensitivity_best_configs.png`
- Subperiod robustness figure: `outputs/figures/phase5_subperiod_sharpe.png`
- Signal decomposition figure: `outputs/figures/signal_to_implementation_stage_decay.png`

## How To Run

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the default pipeline:

```bash
python src/train.py --model ridge --task regression --target-mode cross_sectional_rank
```

Run the multi-horizon study:

```bash
python src/research.py horizon-study --target-mode cross_sectional_rank --models ridge tree mlp --horizons 5 10 20
```

Run the ranking-aware study:

```bash
python src/research.py ranking-study
```

Run the final validation layer:

```bash
python src/research.py phase6-validation-study
```

Run the signal-to-implementation decomposition:

```bash
python src/research.py signal-gap-study
```

## Limitations

- Data comes from `yfinance`; this is not institutional-grade data.
- The universe is static and may contain survivorship bias.
- The backtest uses simplified linear transaction costs and does not model borrow fees, slippage, or execution queues.
- Benchmark comparison to `SPY` is informative but not a complete risk-matched comparison.
- The final positive behavior is regime-dependent and not robust across all subperiods.

## Final Interpretation

This repository does not claim to discover deployable alpha. Its contribution is a controlled, reproducible research study showing that weak cross-sectional signals can pass prediction-oriented checks while still failing implementation, robustness, and economic significance tests.
