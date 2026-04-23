# v3 Phase 2: Ranking-Aware Learning

## What Was Tested

Phase 2 tested whether a ranking-aware learner improves cross-sectional signal quality and implementability relative to the existing regression baseline.

Scoped experiment grid:

- models: `tree`, `ranker`
- horizons: `10d`, `20d`
- target: `cross_sectional_rank`
- portfolio defaults: quantile, signal-weighted, top/bottom decile, gross exposure `1.0`, max turnover `0.5`
- transaction costs: `5 bps`

The ranking-aware model is `XGBRanker` with a pairwise ranking objective. `LightGBM` was not available in the current environment, so no new dependency was added.

Primary comparison artifact:

- `outputs/metrics/ranking_vs_regression_comparison.csv`

## Why Ranking-Aware Learning Matters

The existing pipeline already uses a rank-derived target, but it still trains `ridge`, `tree`, and `mlp` with regression losses. That means the model is asked to predict a numeric proxy for rank rather than directly optimize ordering.

For a cross-sectional equity signal, that distinction matters because portfolio construction only cares about relative ordering on each date:

- which names rise to the top bucket
- which names fall to the bottom bucket
- whether that ordering survives costs and turnover

Ranking-aware learning is therefore the cleanest Phase 2 extension of the current research question.

## Results

| Model | Horizon | Mean IC | IC Vol | Quantile Spread | Ann. Return | Sharpe | Turnover | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tree | 10d | 0.0221 | 0.1991 | 0.0023 | -0.0103 | -0.1112 | 0.0382 | -0.3302 |
| ranker | 10d | 0.0285 | 0.2308 | 0.0027 | -0.0042 | -0.0479 | 0.0136 | -0.3110 |
| tree | 20d | 0.0102 | 0.2032 | -0.0018 | 0.0026 | 0.0286 | 0.0338 | -0.3212 |
| ranker | 20d | 0.0308 | 0.2253 | 0.0026 | -0.0050 | -0.0579 | 0.0125 | -0.3032 |

## Key Findings

### 1. Ranking-aware learning improved signal quality

The `ranker` beat `tree` on both horizons for:

- mean IC
- quantile spread

At `10d`:

- mean IC improved from `0.0221` to `0.0285`
- quantile spread improved from `0.0023` to `0.0027`

At `20d`:

- mean IC improved from `0.0102` to `0.0308`
- quantile spread improved from `-0.0018` to `0.0026`

### 2. Ranking-aware learning materially reduced turnover

The `ranker` also reduced turnover sharply:

- `10d`: `0.0382` to `0.0136`
- `20d`: `0.0338` to `0.0125`

That is a meaningful implementability improvement because it directly lowers transaction-cost pressure.

### 3. After-cost portfolio performance improved at `10d`, but not at `20d`

At `10d`, the `ranker` improved the backtest versus `tree`:

- annualized return improved from `-0.0103` to `-0.0042`
- Sharpe improved from `-0.1112` to `-0.0479`
- max drawdown improved from `-0.3302` to `-0.3110`

At `20d`, the result was mixed:

- `ranker` improved IC, spread, turnover, and drawdown
- but `tree 20d` still had the better after-cost return and Sharpe

That means ranking-aware learning improved signal quality more consistently than it improved realized portfolio performance.

## Did Ranking Improve Implementability?

Partially.

What improved consistently:

- IC
- quantile spread
- turnover
- drawdown

What did not improve consistently:

- annualized return
- Sharpe

So the honest Phase 2 conclusion is:

- ranking-aware learning improved the quality of the cross-sectional ordering
- it likely made the signal cleaner and cheaper to trade
- but it did not produce a clean win on after-cost PnL across both horizons

## Why Ranking-Aware Learning Instead Of Deeper Models?

This was the correct next step because it changes the objective without changing the rest of the research stack.

- We did not use transformers because the repo still does not show evidence that model capacity is the main bottleneck.
- We did not use RL because the project is still evaluating supervised signal quality, not a fully specified execution/control problem.
- We used ranking-aware learning because the portfolio is built from relative ordering, so the training objective should be tested before adding model complexity.

That makes Phase 2 a more defensible research move than jumping to a larger architecture.

## Limitations

- The ranking model is `XGBRanker` only; no LightGBM comparison was run because `lightgbm` is not installed.
- The ranker uses the existing centered percentile-rank target rather than a redesigned discrete relevance target.
- Results are still based on the same research backtest assumptions as earlier phases.
- The Phase 2 grid was intentionally narrow, so this is evidence for the chosen `10d/20d` setups, not a universal statement across all horizons.
