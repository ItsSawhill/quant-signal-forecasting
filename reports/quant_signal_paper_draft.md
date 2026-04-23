# From Prediction to Implementation: Why Weak Cross-Sectional Signals Fail to Produce Robust Alpha

## Abstract
This paper studies whether simple cross-sectional equity signals can be improved into implementable long-short strategies once turnover, transaction costs, and portfolio construction are treated as first-class research objects rather than afterthoughts. Starting from a frozen v2 benchmark built on a 58-stock large-cap U.S. universe with `SPY` as the benchmark, the project proceeds through six controlled phases: a multi-horizon target study, ranking-aware learning, portfolio construction sensitivity, a small market-relative feature extension, robustness analysis, and a final signal-to-implementation decomposition. The empirical evidence suggests that a weak but plausible cross-sectional signal exists. Daily information coefficients are often slightly positive, quantile spreads are frequently directionally sensible, and ranking-aware learning improves ordering quality and reduces turnover. However, these improvements do not reliably translate into robust after-cost performance. Horizon effects are weaker and more model-dependent than initially expected, ranking-aware learning is not a universal after-cost win, portfolio construction choices explain more variation than the remaining model or small feature differences, and full-sample positive outcomes are not stable across subperiods. The decomposition framework then shows where the remaining edge is lost: some predictive structure survives into ranking metrics, but much of it is attenuated before it becomes realized return. The final conclusion is not that the repository discovers deployable alpha, but that it demonstrates, through controlled experiments, why weak signals often fail to survive implementation constraints.

## 1. Introduction
Cross-sectional equity research is often framed as a modeling contest: improve forecasts, improve rankings, and portfolio performance will follow. That framing is incomplete. In practice, the path from statistical prediction to implementable alpha is mediated by portfolio construction, turnover, costs, and regime dependence. A model can improve ordering quality without producing a better after-cost strategy, and a signal can look plausible in aggregate while remaining fragile to realistic variations in assumptions.

This gap between prediction and implementation motivates the present study. The repository was developed as a staged research framework rather than a production trading system. It evolved from an ETF-oriented baseline into a laptop-friendly cross-sectional equity pipeline with explicit leakage controls, walk-forward validation, rank-based targets, configurable portfolio construction, benchmark-aware evaluation, and a final robustness stage. The central problem is not only whether a signal exists, but whether any observed signal survives the sequence of design choices required to trade it.

That problem matters for both research and practice. For research, it distinguishes technically honest evidence from backtests that overstate the meaning of weak statistical edges. For practice, it highlights where many promising modeling results fail: not necessarily in-sample prediction, but the translation of those predictions into portfolios that can survive costs and changing market regimes. This paper therefore studies the full path from signal definition to implementation and asks where, in that chain, weak cross-sectional signals break down.

## 2. Research Question and Motivation
The central research question is:

Can direct cross-sectional ranking across multiple horizons improve implementable signal quality relative to regression-style baselines after turnover and transaction costs?

The project is motivated by a simple tension. In cross-sectional equity research, relative ordering matters more than point forecasts, yet implementation depends on how that ordering is mapped into a portfolio. If ordering quality improves but portfolio outcomes do not, then the limiting factor is not only prediction. The research program is therefore designed to isolate four possible bottlenecks:

- target and horizon specification
- training objective
- portfolio construction
- robustness to realistic assumptions

## 3. Contributions
This study makes six contributions.

1. It develops a controlled multi-phase experimental framework that changes one major research axis at a time: horizon, learning objective, portfolio construction, feature set, and robustness assumptions.
2. It shows that a weak but plausible cross-sectional signal can exist without supporting a robust claim of tradable alpha.
3. It demonstrates that portfolio construction dominates the remaining model differences in explaining after-cost performance variation.
4. It shows that modest feature and ranking improvements do not automatically translate into better portfolio returns.
5. It identifies regime instability, rather than model capacity, as the most important bottleneck to credible implementation in the current research setup.
6. It introduces a simple signal-to-implementation decomposition that quantifies how predictive structure is attenuated across ranking, portfolio construction, and execution.

The project’s contribution is therefore methodological as much as empirical: it documents why weak signals fail to survive implementation constraints.

## 4. Related Framing / Motivation
For cross-sectional equity research, the exact point forecast of a stock’s return is often less useful than the relative ordering of names on a given date. Many practical long-short portfolios are built by sorting the universe into long and short baskets, and their behavior is therefore more naturally evaluated through ranking metrics such as the information coefficient and quantile-spread returns than through raw regression error alone.

This framing motivates three design choices in the repository. First, the default target is a centered cross-sectional rank of future returns rather than only raw return regression. Second, the modeling workflow is evaluated jointly with portfolio construction, rather than treating execution as a separate downstream concern. Third, realism caveats are kept explicit throughout: the system uses `yfinance`, a simplified linear transaction-cost model, and a research backtest rather than a production execution engine.

## 5. Data and Universe
The universe is a laptop-friendly set of 58 liquid U.S. large-cap equities spanning major sectors, with `SPY` used as the benchmark for relative features and benchmark-aware evaluation. The default research window begins on `2018-01-01`, and the cached local data used in the project spans `2018-01-02` through `2026-04-16` according to the frozen benchmark note in [docs/v3_baseline_freeze.md](docs/v3_baseline_freeze.md).

This universe design is a compromise between realism and reproducibility. It is large enough to make cross-sectional ranking meaningful, but small enough to remain runnable on a laptop. It also carries obvious limitations: survivorship is not fully addressed, the universe is not maintained through institutional-quality membership logic, and the data source is appropriate for research demonstration but not for production-grade trading research.

## 6. Feature Engineering
The baseline feature set combines time-series and cross-sectional descriptors derived from price and volume data. Time-series features include trailing returns over `1d`, `5d`, `10d`, and `20d`; rolling volatility; moving-average gaps; a rolling volume z-score; a Sharpe proxy; drawdown from a 20-day high; downside volatility; and volatility-adjusted momentum. Benchmark-relative features include rolling beta and rolling correlation to `SPY`, plus a one-day market-relative return. Selected features are also normalized cross-sectionally with per-date z-scores.

Phase 4 adds only a small, controlled market-relative extension rather than a broad technical indicator expansion. The added features are `residual_return_vs_spy_1d`, `excess_return_vs_spy_5d`, `excess_return_vs_spy_20d`, and `beta_adjusted_momentum_20d`, documented in [docs/v3_phase4_feature_note.md](docs/v3_phase4_feature_note.md). The purpose of these features is not to increase dimensionality indiscriminately, but to isolate stock-specific strength from broad market movement using economically interpretable transformations.

## 7. Target Design
The repository supports three target types across multiple horizons:

- forward return
- forward return sign
- centered cross-sectional rank of forward return

The default research path is `cross_sectional_rank`. For each date and chosen horizon, future asset returns are ranked cross-sectionally within the universe, converted into percentile ranks, and then centered at zero by subtracting `0.5`. This makes the target directly relevant to how long-short portfolios are constructed while retaining compatibility with the broader modeling framework.

This design choice is central to the paper. The project does not ask only whether returns are predictable; it asks whether relative ordering can be learned well enough to matter under a practical portfolio construction process.

## 8. Modeling Framework
The model family is intentionally simple. Baseline learners include ridge regression, a tree-based model, and a lightweight multilayer perceptron. These are not meant to be the best conceivable architectures; they are chosen because they are easy to validate, cheap to run, and sufficient for a staged research design.

Phase 2 introduces a ranking-aware learner via `XGBRanker` with a pairwise ranking objective, compared only against the strongest surviving tree baseline from Phase 1. This preserves the rest of the pipeline while changing the training objective from point estimation to ordering. No transformer, reinforcement-learning, or deep sequence model is added, because the project’s later results do not support the claim that model capacity is the main bottleneck.

## 9. Validation and Backtesting Design
Validation is expanding-window and chronological. Features are constructed from trailing windows, then shifted by one day before modeling use. Train, validation, and test splits are walk-forward and time-ordered. Positions are lagged by one day before returns are applied. This chronology is summarized in the leakage audit at [outputs/metrics/leakage_alignment_audit.md](outputs/metrics/leakage_alignment_audit.md).

Portfolio construction supports quantile or top-k selection, equal or signal weighting, configurable rebalance frequency, gross exposure control, and an optional turnover cap. Backtests apply next-day realized returns to lagged positions, include linear transaction costs, and report both gross and net returns. These choices are still research-grade approximations: there is no borrow-cost model, no slippage model, and no execution engine.

## 10. Phase-by-Phase Experimental Results
### 10.1 Summary of Phase Findings

| Phase | Change Introduced | Effect on Signal (IC / Spread) | Effect on Portfolio (Sharpe / Return) | Key Insight |
| --- | --- | --- | --- | --- |
| Phase 0 | v2 baseline freeze with 5-day ridge rank benchmark | Mean IC `0.0087`; top quantile mean return `0.00537` vs bottom `0.00383` | Annualized return `-0.0105`; Sharpe `-0.1160` | Weak signal can look plausible while remaining unprofitable after costs |
| Phase 1 | Multi-horizon comparison across `5d`, `10d`, `20d` | Horizon effects were mixed; best mean IC `0.0224` for `mlp 5d`; `tree 10d` reached IC `0.0221` | Only `tree 20d` had positive annualized return `0.0026` and Sharpe `0.0286` | Better signal metrics did not map cleanly into better implementability |
| Phase 2 | Ranking-aware learning with `XGBRanker` | `ranker 10d` improved IC from `0.0221` to `0.0285`; `ranker 20d` improved IC from `0.0102` to `0.0308` | `ranker 10d` improved Sharpe from `-0.1112` to `-0.0479`; `ranker 20d` did not beat `tree 20d` after costs | Better ordering quality and lower turnover are not universal after-cost wins |
| Phase 3 | Portfolio construction sensitivity | Signal metrics fixed by model; portfolio mapping drove outcome dispersion | `ranker 10d` Sharpe ranged from `0.0293` to `-0.2648`; `tree 20d` from `0.0287` to `-0.4188` | Portfolio construction mattered more than remaining model differences |
| Phase 4 | Small market-relative feature extension | Slight spread improvement for all candidates; e.g. `ranker 20d` spread `0.0026` to `0.0028` | Net return and Sharpe unchanged across all three tested candidates | Slightly stronger signals did not improve implementable returns under the fixed best portfolio |
| Phase 5 | Robustness across subperiods, costs, and small portfolio variations | IC remained positive in most subperiods | Subperiod Sharpe flipped from strongly negative to positive late in sample; full-sample gains stayed small | Regime instability, not model capacity, is the dominant bottleneck |
| Phase 6 | Signal-to-implementation decomposition | Rankers preserved positive IC-to-spread mapping; `tree 20d` did not | Even the best total-efficiency case remained economically weak | Predictive structure is attenuated mainly after ranking, before robust realized return |

### 10.2 Phase 0: Baseline Freeze
The v2 benchmark was frozen using the default ridge rank configuration over a 5-day horizon. The frozen reference metrics are reported in [docs/v3_baseline_freeze.md](docs/v3_baseline_freeze.md). Representative values are:

- mean daily IC: `0.0087`
- Spearman rank correlation: `0.0113`
- annualized return: `-0.0105`
- Sharpe: `-0.1160`
- max drawdown: `-0.3469`
- average turnover: `0.0323`

The benchmark is weak but plausible: quantile returns are directionally sensible, yet the after-cost result remains negative. This frozen result anchors the interpretation of every later phase.

### 10.3 Phase 1: Multi-Horizon Study
Phase 1 compares `ridge`, `tree`, and `mlp` over `5d`, `10d`, and `20d` cross-sectional rank targets, with the summary table saved at [outputs/metrics/horizon_study_cross_sectional_rank.csv](outputs/metrics/horizon_study_cross_sectional_rank.csv). A bug in early long-horizon labels was identified and fixed before the final results were frozen, which is itself an important methodological point: small implementation errors can materially distort horizon conclusions.

Representative Phase 1 results are:

- `ridge 5d`: mean IC `0.0087`, Sharpe `-0.1160`
- `tree 10d`: mean IC `0.0221`, Sharpe `-0.1112`
- `tree 20d`: mean IC `0.0102`, annualized return `0.0026`, Sharpe `0.0286`
- `mlp 5d`: mean IC `0.0224`, Sharpe `-0.1182`

The horizon effect is weaker and more model-dependent than initially expected. There is no clean “medium horizon dominates” conclusion. The strongest after-cost result in the grid is `tree 20d`, but it does not coincide with the highest information coefficient. This phase establishes a recurring theme of the project: better signal metrics do not automatically translate into better implementable performance.

### 10.4 Phase 2: Ranking-Aware Learning
Phase 2 compares the tree baseline with a ranking-aware learner using `XGBRanker`, with the summary table saved at [outputs/metrics/ranking_vs_regression_comparison.csv](outputs/metrics/ranking_vs_regression_comparison.csv). Results are:

- `tree 10d`: IC `0.0221`, quantile spread `0.0023`, annualized return `-0.0103`, Sharpe `-0.1112`, turnover `0.0382`
- `ranker 10d`: IC `0.0285`, quantile spread `0.0027`, annualized return `-0.0042`, Sharpe `-0.0479`, turnover `0.0136`
- `tree 20d`: IC `0.0102`, quantile spread `-0.0018`, annualized return `0.0026`, Sharpe `0.0286`, turnover `0.0338`
- `ranker 20d`: IC `0.0308`, quantile spread `0.0026`, annualized return `-0.0050`, Sharpe `-0.0579`, turnover `0.0125`

Ranking-aware learning improves ordering quality and materially reduces turnover at both horizons. At `10d` it also improves the after-cost backtest relative to the tree baseline, but at `20d` it does not produce a universal after-cost win. The lesson is that optimizing ranking can improve signal quality without guaranteeing superior realized portfolio performance.

### 10.5 Phase 3: Portfolio Construction Sensitivity
Phase 3 re-evaluates the strongest surviving candidates under controlled portfolio variations. The full comparison is in [outputs/metrics/phase3_portfolio_sensitivity.csv](outputs/metrics/phase3_portfolio_sensitivity.csv), with a visual summary in [outputs/figures/phase3_portfolio_sensitivity_best_configs.png](outputs/figures/phase3_portfolio_sensitivity_best_configs.png).

This phase yields the strongest project-level insight. Within a single candidate, portfolio-rule choices produce larger swings than the remaining model differences. For example:

- `ranker 10d` Sharpe ranges from `0.0293` to `-0.2648`
- `ranker 20d` Sharpe ranges from `0.0086` to `-0.2706`
- `tree 20d` Sharpe ranges from `0.0287` to `-0.4188`

The best overall Phase 3 configuration by Sharpe is `ranker 10d` with decile selection, equal weighting, rebalance every 5 days, and turnover cap on, producing net return `0.0025`, Sharpe `0.0293`, and turnover `0.0006`. The strongest `tree 20d` configuration by net return remains competitive, but the broader result is clear: portfolio construction matters more than the remaining model differences. Deciles outperform quintiles and top-k, equal weighting outperforms signal weighting on average, and rebalancing every five days is the best compromise in the studied set.

Figure 1 is central to this conclusion. [Insert Figure: outputs/figures/phase3_portfolio_sensitivity_best_configs.png] The figure shows that even within a tightly filtered candidate set, modest changes in selection cutoff, weighting, and rebalance rules produce visibly different equity-curve paths. Its importance is not that any one configuration looks strong in absolute terms, but that the performance spread induced by portfolio rules exceeds the spread remaining across the best surviving model candidates.

### 10.6 Phase 4: Controlled Feature Research
Phase 4 holds the best Phase 3 portfolio template fixed and compares the baseline feature set with the controlled market-relative extension. The summary table is [outputs/metrics/phase4_feature_comparison.csv](outputs/metrics/phase4_feature_comparison.csv).

Representative results are:

- `ranker 10d`: IC `0.0285` to `0.0284`, quantile spread `0.0027` to `0.0029`, no change in net return or Sharpe
- `ranker 20d`: IC `0.0308` to `0.0310`, quantile spread `0.0026` to `0.0028`, no change in net return or Sharpe
- `tree 20d`: IC `0.0102` to `0.0105`, quantile spread `-0.0018` to `-0.0014`, no change in net return or Sharpe

The new features slightly improve signal-quality metrics, especially quantile spread, but those gains do not translate into better after-cost portfolio performance under the frozen low-turnover portfolio template. This phase reinforces another recurring project lesson: modest signal improvements are not sufficient when the implementation layer compresses downstream performance differences.

### 10.7 Phase 5: Robustness Analysis
Phase 5 tests the final candidate configurations across subperiods, transaction-cost assumptions, and small portfolio-rule variations. The main table is [outputs/metrics/phase5_robustness_results.csv](outputs/metrics/phase5_robustness_results.csv), with a subperiod figure at [outputs/figures/phase5_subperiod_sharpe.png](outputs/figures/phase5_subperiod_sharpe.png).

The strongest conclusion of the entire project comes from this phase: the results are not robust.

Across all three final candidates:

- subperiod performance is strongly negative in `2018-2020`
- strongly negative again in `2020-2022`
- positive only in `2022-2026`

Examples include:

- `ranker 10d`: Sharpe `-1.4205`, `-0.9442`, `0.8507`
- `ranker 20d`: Sharpe `-1.4205`, `-0.9442`, `0.8191`
- `tree 20d`: Sharpe `-1.4205`, `-0.9442`, `0.8191`

Figure 2 summarizes this result directly. [Insert Figure: outputs/figures/phase5_subperiod_sharpe.png] The figure matters because it collapses the final project claim into a single visual: the same general strategy family is negative in early and middle subperiods and positive only in the latest regime. That pattern is inconsistent with a stable claim of tradable alpha.

Cost sensitivity is mild, but primarily because the best surviving portfolio template trades very little. For `ranker 10d`, annualized net return changes only from `0.0026` at `2 bps` to `0.0024` at `10 bps`. This is evidence of low turnover, not evidence of strong alpha. Small portfolio-rule variations continue to matter more than the remaining model differences: decile beats quintile, equal beats signal, and 5-day rebalance beats 10-day rebalance on average.

### 10.8 Phase 6: Signal-to-Implementation Decomposition
Phase 6 reframes the earlier empirical results as a stage-by-stage attenuation problem:

`Signal -> Ranking -> Portfolio -> Execution -> Return`

The decomposition uses only existing project metrics and defines three descriptive ratios:

- ranking efficiency = quantile spread / mean IC
- execution efficiency = net annualized return / quantile spread
- total efficiency = net annualized return / mean IC

The summary artifact is [outputs/metrics/signal_to_implementation_decomposition.csv](outputs/metrics/signal_to_implementation_decomposition.csv), with a readable table at [outputs/metrics/signal_to_implementation_decomposition.md](outputs/metrics/signal_to_implementation_decomposition.md).

The final values are:

- `ranker 10d`: mean IC `0.0284`, quantile spread `0.0029`, net return `0.0025`, ranking efficiency `0.1022`, execution efficiency `0.8679`, total efficiency `0.0887`
- `ranker 20d`: mean IC `0.0310`, quantile spread `0.0028`, net return `0.0007`, ranking efficiency `0.0913`, execution efficiency `0.2624`, total efficiency `0.0240`
- `tree 20d`: mean IC `0.0105`, quantile spread `-0.0014`, net return `0.0007`, ranking efficiency `-0.1305`, execution efficiency `-0.5417`, total efficiency `0.0707`

The ratios should be read descriptively rather than causally, but they clarify where the project’s weak edge is lost. The ranker configurations preserve a positive mapping from IC into quantile spread, while `tree 20d` already fails at that stage. The larger attenuation occurs later: even when ranking quality is positive, the surviving spread becomes a very small net return once the final portfolio and execution layer are imposed. This is precisely the project’s central thesis in compact form.

Figure 3 makes that decay visible. [Insert Figure: outputs/figures/signal_to_implementation_stage_decay.png] The figure places mean IC, quantile spread, and net return side by side for the final reference configurations. Its purpose is not scale comparability in a strict statistical sense, but visual intuition: the later-stage realized metric is far smaller than the earlier-stage signal metric in every case.

## 11. Discussion
The central result of the paper is not that the models failed to find any signal. The more interesting result is that they found a signal that was too weak, too compressible, and too regime-dependent to support a robust implementation claim.

The first implication concerns model choice. Ranking-aware learning improved cross-sectional ordering quality and materially lowered turnover, but it was not a universal after-cost winner. This is important because it suggests that the residual problem is not well described as “the model is too simple.” If better ranking objectives improve IC and quantile spread but do not consistently improve realized returns, then the bottleneck lies downstream from pure prediction.

The second implication concerns feature research. Phase 4 added a carefully constrained set of market-relative features and achieved small improvements in signal-quality metrics. Yet portfolio results were unchanged under the frozen best Phase 3 template. That is not evidence that features do not matter. Rather, it shows that small increases in cross-sectional separation are insufficient when the portfolio mapping is already designed to suppress turnover and the remaining edge is weak. In other words, signal quality improved at the margin, but not enough to move realized outcomes.

The third implication concerns turnover and costs. Much of the later apparent improvement came from portfolios that traded very little. This mattered in two ways. First, low-turnover equal-weight decile portfolios often outperformed more elaborate alternatives, implying that score magnitudes were less useful than simple ordering once costs were considered. Second, once turnover was compressed far enough, cost sensitivity became mild. That should not be misread as evidence of robust profitability; it is instead evidence that the strategy survives by trading very little rather than by generating a large pre-cost edge.

The fourth and most important implication concerns regime instability. Phase 5 shows that positive full-sample results are largely driven by the most recent subperiod, while earlier periods are strongly negative. This is the strongest evidence against a deployment claim. A research process can tolerate weak average performance if the behavior is stable and interpretable. It cannot credibly tolerate a sign flip across regimes while still presenting the full-sample result as a robust strategy. The regime dependence seen here suggests that the project is closer to identifying a weak, context-dependent cross-sectional ordering than to uncovering persistent alpha.

The signal-to-implementation decomposition added in Phase 6 makes this structural point more explicit. Weak predictive structure survives into IC and, in the stronger ranker cases, into positive quantile spread. But the mapping from that spread into realized annualized return is small enough that modest changes in portfolio rules or regime are sufficient to dominate the remaining model differences. In that sense, the project’s novelty is not only that it documents failure, but that it quantifies where the attenuation occurs.

Taken together, these points explain why better models did not help enough, why feature gains did not translate into returns, why turnover and costs dominate the surviving configurations, and why regime instability is the critical barrier. The project’s main lesson is therefore structural: the difficulty is not only finding a signal, but ensuring that the signal is strong enough to survive implementation and stable enough to survive time.

## 12. Limitations
This study is research-grade, not production-grade.

First, the data source is `yfinance`, which is acceptable for a demonstrative research workflow but not for institutional data engineering. Survivorship and universe-maintenance issues remain.

Second, the backtest is simplified. It includes linear transaction costs and lagged positions, but does not model slippage, financing, borrow costs, or path-level execution. Benchmark comparison against `SPY` is informative but not a clean like-for-like comparison against a market-neutral or risk-matched portfolio.

Third, the universe is intentionally small and laptop-friendly. That makes the project reproducible, but also limits the scope of any performance claim.

Fourth, the feature and model expansions are intentionally narrow. This makes attribution cleaner, but also means the project is not an exhaustive search over possible alpha sources.

Finally, the strongest full-sample outcomes are small in absolute terms and unstable across time. This is the main reason the project stops short of any deployment claim.

## 13. Conclusion
Yes, there is likely a weak cross-sectional signal in this universe. No, it is not convincingly tradable in the project’s current form.

The evidence supports four direct answers. First, the signal appears real enough to generate slightly positive IC and directionally sensible quantile spreads in several configurations. Second, those gains are too small and too unstable to support a robust after-cost alpha claim. Third, the main bottleneck is not remaining model complexity but the conversion of weak predictive structure into stable implemented returns. Fourth, the project shows that portfolio construction and regime stability matter more than the remaining model and small feature differences.

What this project proves is not the existence of deployable alpha. It proves, through controlled experiments, that weak cross-sectional signals can survive prediction tests and still fail under implementation constraints. More specifically, it introduces a simple signal-to-implementation decomposition framework that quantifies how predictive structure is attenuated across ranking, portfolio construction, and execution, and uses that framework to explain why weak cross-sectional signals fail to produce robust alpha. That is the core research result.

## 14. Future Work
Future work should not begin with larger models. The current evidence does not support the claim that model capacity is the main bottleneck.

More credible next steps would include:

- improved data quality, corporate-action handling, and universe construction
- sector and industry normalization
- cleaner residual or benchmark-relative target design
- more realistic execution assumptions
- broader robustness testing across universes and market regimes
- nested research protocols to reduce overfitting to design choices

Only after these foundations improve would larger architectures or more complex objectives be justified.

## 15. Figures and Tables Used
The paper relies primarily on three figures.

Figure 1:

- [outputs/figures/phase3_portfolio_sensitivity_best_configs.png](outputs/figures/phase3_portfolio_sensitivity_best_configs.png)
- used to illustrate that portfolio construction choices produced larger differences than the remaining candidate model differences

Figure 2:

- [outputs/figures/phase5_subperiod_sharpe.png](outputs/figures/phase5_subperiod_sharpe.png)
- used to show directly that the surviving configurations were not stable across subperiods

Figure 3:

- [outputs/figures/signal_to_implementation_stage_decay.png](outputs/figures/signal_to_implementation_stage_decay.png)
- used to visualize the attenuation from mean IC to quantile spread to net return in the final frozen configurations

Figure 4:

- [outputs/figures/ranker_regression_cross_sectional_rank_h10_extended_equity_curve.png](outputs/figures/ranker_regression_cross_sectional_rank_h10_extended_equity_curve.png)
- used as a representative equity-curve example for the best Phase 3 / Phase 4 configuration

Key tables and metrics artifacts used:

- [docs/v3_baseline_freeze.md](docs/v3_baseline_freeze.md)
- [outputs/metrics/horizon_study_cross_sectional_rank.csv](outputs/metrics/horizon_study_cross_sectional_rank.csv)
- [outputs/metrics/ranking_vs_regression_comparison.csv](outputs/metrics/ranking_vs_regression_comparison.csv)
- [outputs/metrics/phase3_portfolio_sensitivity.csv](outputs/metrics/phase3_portfolio_sensitivity.csv)
- [outputs/metrics/phase4_feature_comparison.csv](outputs/metrics/phase4_feature_comparison.csv)
- [outputs/metrics/phase5_robustness_results.csv](outputs/metrics/phase5_robustness_results.csv)
- [outputs/metrics/signal_to_implementation_decomposition.csv](outputs/metrics/signal_to_implementation_decomposition.csv)
- [outputs/metrics/leakage_alignment_audit.md](outputs/metrics/leakage_alignment_audit.md)
