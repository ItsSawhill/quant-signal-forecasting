# v3 Phase 5: Robustness and Validation

## What Was Tested

Phase 5 tested whether the final candidate configurations were robust to:

- time-period splits
- transaction cost assumptions
- small, realistic portfolio-rule variations

Reference configurations:

- `ranker 10d`
- `ranker 20d`
- `tree 20d`

Frozen reference portfolio template:

- decile selection
- equal weight
- rebalance every 5 trading days
- turnover cap on

Signal definition:

- Phase 4 extended feature set

Subperiods were evaluated on exact date ranges:

- `2018-01-01` to `2019-12-31`
- `2020-01-01` to `2021-12-31`
- `2022-01-01` to the available data end in `2026`

Artifacts:

- `outputs/metrics/phase5_robustness_results.csv`
- `outputs/metrics/phase5_robustness_results.md`
- `outputs/figures/phase5_subperiod_sharpe.png`

## Subperiod Stability Findings

The results were not stable across time.

All three reference configurations showed the same broad pattern:

- strongly negative in `2018-2020`
- strongly negative in `2020-2022`
- clearly positive in `2022-2026`

Examples:

- `ranker 10d`: Sharpe `-1.4205`, `-0.9442`, `0.8507`
- `ranker 20d`: Sharpe `-1.4205`, `-0.9442`, `0.8191`
- `tree 20d`: Sharpe `-1.4205`, `-0.9442`, `0.8191`

Interpretation:

- the strategy behavior is regime-dependent
- the positive full-sample result is driven by the later subperiod
- robustness across time is weak

One additional observation matters:

- IC still remained positive in most subperiods
- but positive IC did not guarantee positive PnL

That means the underlying cross-sectional ordering may be real, but the translation into returns is fragile.

## Cost Sensitivity Findings

Cost sensitivity was mild under the frozen reference portfolio because turnover was already extremely low.

Examples:

- `ranker 10d`: net return `0.0026` at `2 bps`, `0.0025` at `5 bps`, `0.0024` at `10 bps`
- `ranker 20d`: net return `0.0008` at `2 bps`, `0.0007` at `5 bps`, `0.0007` at `10 bps`
- `tree 20d`: net return `0.0008` at `2 bps`, `0.0007` at `5 bps`, `0.0007` at `10 bps`

Interpretation:

- once turnover was compressed by portfolio construction, costs stopped being the main source of fragility
- that is not evidence of strong alpha
- it is evidence that the surviving configurations trade very little

## Portfolio Sensitivity Findings

The small portfolio variations still mattered more than the remaining model differences.

Main patterns:

- decile outperformed quintile
- equal weight outperformed signal weight
- rebalance every `5` days outperformed every `10` days

Examples:

- `ranker 10d`, decile/equal/5d: Sharpe `0.0293`
- `ranker 10d`, quintile/signal/5d: Sharpe `-0.1775`
- `tree 20d`, decile/equal/5d: Sharpe `0.0086`
- `tree 20d`, quintile/signal/10d: Sharpe `-0.2685`

Average patterns across the portfolio-variation slice:

- decile mean Sharpe: `-0.0119`
- quintile mean Sharpe: `-0.1604`
- equal mean Sharpe: `-0.0349`
- signal mean Sharpe: `-0.1316`

This confirms the Phase 3 conclusion rather than overturning it:

- portfolio construction remains a larger lever than the remaining model differences

## Are The Results Robust Or Fragile?

Overall: fragile.

Reasons:

- the sign of performance changed across subperiods
- the positive full-sample outcomes were small
- IC and quantile separation did not translate reliably into returns
- multiple model variants collapsed to nearly identical PnL under the frozen low-turnover portfolio template

The strategy is therefore not robust enough to describe as a tradable result.

## Final Conclusion

### Is there a real signal?

Probably a weak one.

The repo repeatedly found:

- slightly positive IC
- directionally improved quantile spreads in several configurations
- some persistence under ranking-aware learning and market-relative features

So the cross-sectional ordering is not obviously random.

### Is it tradable?

Not convincingly.

The best configurations produced only small positive full-sample after-cost results, and those gains were not stable across subperiods. That is not enough to claim a robust tradable signal.

### What is the main bottleneck?

The main bottleneck is converting weak signal quality into stable after-cost portfolio performance.

Notably:

- model changes moved IC somewhat
- feature changes moved IC only slightly
- portfolio construction changed outcomes the most
- even after that, regime stability was weak

### What mattered more: model, features, or portfolio construction?

Portfolio construction mattered more than model class or the small Phase 4 feature extension.

That is the clearest project-level conclusion.

## Why We Stop Here

- Adding more models would not solve the main problem identified by Phase 5, which is fragility across regimes and weak translation from signal to tradable PnL.
- Adding more features would risk turning the project into an undisciplined search without first solving the robustness problem.
- Robustness is the right final step because a research pipeline is only defensible if its findings survive reasonable changes in time period, cost, and execution rules.

To pursue real alpha, the project would need materially stronger foundations:

- better data quality and corporate-action handling
- cleaner residual or sector-neutral targets
- sector or industry normalization
- more realistic execution assumptions
- larger and more diverse universes
- stronger out-of-sample robustness

Without those, adding complexity would mostly increase research noise rather than signal quality.
