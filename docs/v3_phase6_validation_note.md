# Phase 6 Validation Note

## What Was Tested

Phase 6 added a narrow validation layer on top of the frozen final reference configurations from Phase 5:

- `ranker 10d`
- `ranker 20d`
- `tree 20d`

All runs used the same final portfolio template:

- decile selection
- equal weighting
- rebalance every 5 trading days
- turnover cap on
- extended feature set

The purpose was not to improve performance, but to test whether the observed weak signal is distinguishable from noise and whether the economic results survive stronger validation.

## Why These Checks Matter

Earlier phases already showed three things:

1. signal quality metrics such as IC can improve without producing strong after-cost returns
2. portfolio construction matters more than the remaining model differences
3. the positive behavior is concentrated in the most recent subperiod rather than spread evenly across the sample

Those findings make additional modeling less important than validation. If the signal is not statistically distinct from noise, or if the net returns are too small to survive basic placebo and uncertainty checks, then stronger claims are not defensible.

## IC Significance Results

The strongest statistical evidence appears in the daily IC series, not in the portfolio PnL.

| Model | Horizon | Mean IC | IC Std | t-stat | p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| ranker | 10d | 0.0284 | 0.2307 | 5.2188 | <0.0001 |
| ranker | 20d | 0.0310 | 0.2243 | 5.8534 | <0.0001 |
| tree | 20d | 0.0105 | 0.2069 | 2.1506 | 0.0315 |

Interpretation:

- The two ranker variants have daily IC means that are statistically above zero under a simple t-test.
- `tree 20d` is weaker and only marginally significant.
- This supports the claim that a weak cross-sectional ordering signal likely exists.
- It does **not** imply that the signal is economically strong or implementable.

## Bootstrap Results

Bootstrap confidence intervals were estimated with reproducible daily resampling. These intervals were used as a rough uncertainty check rather than a production-grade inference framework.

### Annualized Return 95% CI

- `ranker 10d`: `[-0.0540, 0.0791]`
- `ranker 20d`: `[-0.0589, 0.0648]`
- `tree 20d`: `[-0.0589, 0.0648]`

### Sharpe 95% CI

- `ranker 10d`: `[-0.6224, 0.9223]`
- `ranker 20d`: `[-0.6867, 0.7321]`
- `tree 20d`: `[-0.6867, 0.7321]`

### Quantile Spread 95% CI

- `ranker 10d`: `[0.0007, 0.0051]`
- `ranker 20d`: `[-0.0001, 0.0061]`
- `tree 20d`: `[-0.0039, 0.0016]`

Interpretation:

- The portfolio-level return and Sharpe intervals are wide and all include zero.
- `ranker 10d` has the cleanest spread result: its bootstrap interval stays positive.
- `ranker 20d` is borderline on spread.
- `tree 20d` does not show a convincing positive spread interval.

This is the central Phase 6 result: the signal is more defensible in ranking metrics than in implementable portfolio metrics.

## Placebo Comparison Results

Two placebo baselines were tested:

1. random cross-sectional ranking by date
2. label-shuffled scores by date, preserving label shape while destroying the predictive mapping

### Mean IC Comparison

- `ranker 10d`: real `0.0284`, random `-0.0018`, label-shuffled `-0.0024`
- `ranker 20d`: real `0.0310`, random `-0.0026`, label-shuffled `0.0008`
- `tree 20d`: real `0.0105`, random `0.0002`, label-shuffled `0.0012`

### Quantile Spread Comparison

- `ranker 10d`: real `0.0029`, random `-0.0004`, label-shuffled `-0.0008`
- `ranker 20d`: real `0.0028`, random `0.0003`, label-shuffled `0.0011`
- `tree 20d`: real `-0.0014`, random `-0.0001`, label-shuffled `-0.0016`

Interpretation:

- The real ranker signals are clearly separated from placebo in IC and, for `ranker 10d`, also in quantile spread.
- `tree 20d` remains weak even relative to placebo.

However, the portfolio metrics under the frozen Phase 5 template are nearly identical across real and placebo variants:

- `ranker 10d`: annualized return `0.0025`, Sharpe `0.0293`
- `ranker 20d`: annualized return `0.0007`, Sharpe `0.0086`
- `tree 20d`: annualized return `0.0007`, Sharpe `0.0086`

This is not evidence that the placebo signals are good. It is evidence that the final low-turnover portfolio template is so inert that weak differences in ranking quality do not move portfolio outcomes much at all.

## Cost Sweep Findings

The cost sweep covered `0`, `2`, `5`, `10`, and `20` bps.

### Ranker 10d

- annualized return: `0.0026` at `0` bps, `0.0023` at `20` bps
- Sharpe: `0.0301` at `0` bps, `0.0268` at `20` bps
- turnover: `0.0006`

### Ranker 20d and Tree 20d

- annualized return: `0.0008` at `0` bps, `0.0005` at `20` bps
- Sharpe: `0.0095` at `0` bps, `0.0062` at `20` bps
- turnover: `0.0006`

Interpretation:

- Cost sensitivity is mild, but mainly because turnover is extremely low.
- That is not a strong robustness result. It means the portfolio hardly trades.
- The strategy is therefore not meaningfully profitable even before strong friction is applied.

## Rolling Stability Findings

Phase 6 added a rolling 2-year mean IC view.

Rolling 2-year mean IC ranges:

- `ranker 10d`: min `-0.0031`, median `0.0300`, max `0.0526`
- `ranker 20d`: min `-0.0340`, median `0.0376`, max `0.0691`
- `tree 20d`: min `-0.0326`, median `0.0120`, max `0.0430`

Interpretation:

- The IC is not persistently positive across all rolling windows.
- Each configuration goes through windows where rolling IC is near zero or negative.
- The stronger positive behavior is concentrated in later windows, which is consistent with the Phase 5 subperiod result that performance is regime-dependent.

## Is the Signal Distinguishable From Noise?

The honest answer is: **partially, but only weakly and mainly in cross-sectional ranking metrics**.

- The two ranker configurations are statistically distinguishable from noise in average IC.
- `ranker 10d` also shows the cleanest positive quantile-spread evidence.
- `tree 20d` is much less convincing.
- The signal does not survive the stronger economic test well: return and Sharpe confidence intervals remain wide and include zero.

So the project can support the claim that a weak cross-sectional signal is plausible. It cannot support a strong claim of robust alpha.

## Is the Strategy Economically Meaningful?

No, not in its current form.

- The best full-sample annualized return in the frozen final setup is only about `0.25%`.
- Full-sample Sharpe is only `0.0293` for the best candidate.
- Phase 5 already showed strongly negative subperiod Sharpe in `2018-2020` and `2020-2022`, with positive behavior concentrated in `2022-2026`.
- Phase 6 shows that bootstrap uncertainty around returns and Sharpe is still too wide to justify an economic claim.

The economic conclusion remains the same as earlier phases: there may be a signal, but it is not convincingly tradable.

## Honest Limitations

- The IC significance check uses a simple t-statistic and normal-approximation p-value, not a full serial-correlation-robust inference framework.
- The bootstrap uses simple daily resampling, which ignores time dependence and should be interpreted cautiously.
- The placebo backtests inherit the final low-turnover portfolio template, which compresses portfolio-level differences and makes the economic comparison conservative but also somewhat blunt.
- Data quality, survivorship risk, and execution realism remain limited, as already documented in earlier phases.

## Why Validation Is The Right Final Phase

Validation is the right final step because the unresolved question is no longer whether one more model or one more feature can improve a weak metric. The unresolved question is whether the project’s observed effects are real enough, stable enough, and economically meaningful enough to survive scrutiny.

Placebo tests and significance checks sharpen the thesis in a useful way:

- they strengthen the claim that the ranker signals are not pure random noise in IC terms
- they weaken any stronger claim that the project has found robust tradable alpha

After Phase 6, the defensible claim is narrow but credible:

> A weak cross-sectional signal likely exists, especially in ranking metrics, but it does not translate into robust or economically meaningful alpha under the current data, validation, and implementation setup.
