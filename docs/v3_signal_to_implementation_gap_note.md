# Signal-to-Implementation Gap Note

## What The Framework Is

This note introduces a simple signal-to-implementation decomposition for the frozen final reference configurations:

- `ranker 10d`
- `ranker 20d`
- `tree 20d`

All three use the final project template:

- extended feature set
- decile selection
- equal weighting
- rebalance every 5 trading days
- turnover cap on

The framework is:

`Signal -> Ranking -> Portfolio -> Execution -> Return`

The purpose is to quantify, descriptively, how predictive structure attenuates as it moves from cross-sectional prediction quality to realized portfolio outcomes.

## Why It Matters

Earlier phases already established the main empirical pattern of the project:

- ranking-aware learning improved IC
- portfolio construction dominated remaining model differences
- feature gains did not translate into better returns
- robustness remained weak across subperiods

Those findings suggest that the main open question is no longer whether one more model can improve a signal metric. The main open question is where the signal is being lost between prediction and implementable return.

That is the purpose of this framework.

## Stage Definitions

The decomposition uses only metrics already produced by the pipeline.

### 1. Signal Strength

- mean IC

This is the project’s basic measure of cross-sectional ordering quality.

### 2. Ranking Realization

- quantile spread

This measures whether better-ranked names actually separate from worse-ranked names in realized returns.

### 3. Implemented Return

- net annualized return

This is the realized portfolio outcome after the portfolio construction layer and transaction costs.

### 4. Execution Friction Context

- turnover
- Sharpe
- max drawdown

These metrics do not enter the ratios directly, but they provide context for whether any surviving return is economically meaningful.

## Derived Decomposition Metrics

The framework uses three simple ratios.

### Ranking Efficiency

`ranking_efficiency = quantile_spread / mean_IC`

This measures how much realized spread is extracted per unit of average IC.

### Execution Efficiency

`execution_efficiency = net_return / quantile_spread`

This measures how much of the ranking spread survives through the portfolio and execution layer into net annualized return.

### Total Efficiency

`total_efficiency = net_return / mean_IC`

This measures how much net return survives per unit of signal strength.

If a denominator is too close to zero, the ratio is treated as undefined rather than forced into an unstable interpretation.

## What The Decomposition Shows

The final decomposition table is saved at [outputs/metrics/signal_to_implementation_decomposition.csv](outputs/metrics/signal_to_implementation_decomposition.csv).

Key values are:

| Model | Horizon | Mean IC | Quantile Spread | Net Return | Ranking Eff. | Execution Eff. | Total Eff. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ranker | 10d | 0.0284 | 0.0029 | 0.0025 | 0.1022 | 0.8679 | 0.0887 |
| ranker | 20d | 0.0310 | 0.0028 | 0.0007 | 0.0913 | 0.2624 | 0.0240 |
| tree | 20d | 0.0105 | -0.0014 | 0.0007 | -0.1305 | -0.5417 | 0.0707 |

## Where Predictive Power Is Being Lost

The first loss point is between signal quality and ranking realization.

- `ranker 10d` and `ranker 20d` both preserve a positive mapping from IC into quantile spread.
- `tree 20d` does not. It has positive mean IC but negative quantile spread, which means that its average cross-sectional correlation is not translating into a useful top-minus-bottom portfolio sort.

The second, and more important, loss point is between ranking realization and implemented return.

- `ranker 10d` has the best execution efficiency at `0.8679`, but that still converts only a very small spread into a very small net annualized return in absolute terms.
- `ranker 20d` retains much less of its spread at the execution stage: `0.2624`.
- This is consistent with earlier phases showing that better ranking quality alone does not guarantee better implementation.

The total-efficiency view makes the project’s main result easier to state:

- `ranker 10d` is the strongest final configuration in this decomposition.
- even that best case produces only `0.0025` annualized net return and `0.0293` Sharpe
- the final economic result remains weak despite positive ranking ratios

So the main loss appears after signal generation, especially in the translation from ranking structure into economically meaningful realized return.

## How This Connects To Earlier Phases

The decomposition is consistent with the project’s earlier findings.

### Phase 2

Ranking-aware learning improved IC and reduced turnover. The decomposition confirms that the ranker variants preserve the signal-to-ranking mapping better than the tree baseline.

### Phase 3

Portfolio construction dominated outcomes. The decomposition helps explain why: even when signal metrics are reasonable, the mapping from spread to realized return is small and fragile.

### Phase 4

Feature gains improved ranking metrics only slightly. The decomposition makes clear why those gains did not matter much: the downstream execution stage is already compressing most of the remaining edge.

### Phase 5 and Phase 6

Robustness remained weak, and the portfolio template became extremely low turnover. The decomposition should therefore be read as a summary of attenuation in the frozen final setup, not as proof that the same ratios would hold under different portfolio choices or regimes.

## Honest Limitations

- This framework is descriptive rather than causal.
- The ratios summarize attenuation, but they do not identify a unique mechanism for why performance is lost.
- Quantile spread and annualized return are not directly comparable on a structural economic basis; the ratio is useful as a compact diagnostic, not as a deep theoretical object.
- The framework inherits all earlier project limitations around data quality, survivorship, benchmark handling, and simplified execution.
- The final reference template is extremely low turnover, which compresses portfolio-level differences and may understate some economically relevant distinctions.

## Novelty Statement

This project does not claim to discover deployable alpha. Its more credible contribution is methodological:

> We introduce a simple signal-to-implementation decomposition framework that quantifies how predictive structure is attenuated across ranking, portfolio construction, and execution, and use it to explain why weak cross-sectional signals fail to produce robust alpha.

## Why This Is A Useful Contribution

This is more valuable right now than adding another model because the unresolved problem is not lack of model variety. The unresolved problem is understanding why modest improvements in signal metrics repeatedly failed to survive implementation.

The decomposition improves the paper in three ways:

- it gives a compact language for the prediction-to-implementation gap that the earlier phases kept revealing
- it makes the project’s main result more publishable because it converts a sequence of repo experiments into a coherent analytical framework
- it clarifies what can and cannot be claimed

What it does prove:

- weak predictive structure can survive into ranking metrics
- that structure can still be heavily attenuated before it becomes realized return

What it does not prove:

- a causal attribution of exactly why each unit of signal is lost
- the existence of robust, tradable alpha
