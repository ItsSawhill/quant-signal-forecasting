# v3 Phase 3: Portfolio Construction Sensitivity

## What Was Tested

Phase 3 asked whether portfolio construction choices explain more of the remaining performance variation than model class itself.

The study was intentionally limited to the strongest Phase 2 candidates:

- `ranker 10d`
- `ranker 20d`
- `tree 20d`

For each candidate prediction set, the study varied:

- selection method: decile, quintile, top-k
- weighting scheme: equal, signal
- rebalance frequency: daily, every 5 days, every 10 days
- turnover control: cap on, cap off

Primary artifacts:

- `outputs/metrics/phase3_portfolio_sensitivity.csv`
- `outputs/metrics/phase3_portfolio_sensitivity.md`
- `outputs/figures/phase3_portfolio_sensitivity_best_configs.png`

## Why Portfolio Construction Was The Right Next Step

By the end of Phase 2, the remaining model differences were modest and mixed:

- the `ranker` improved IC and turnover
- the `tree 20d` baseline still had the strongest after-cost return

That made portfolio construction the natural next bottleneck to test. The signal was already being mapped into trades through cutoff rules, weighting, rebalance timing, and turnover control. Those choices directly determine whether a weak but plausible cross-sectional signal survives costs.

## What Was Learned

### 1. Portfolio rules mattered more than the remaining model differences

Within each candidate, the portfolio-rule spread was large:

- `ranker 10d` Sharpe ranged from `0.0293` to `-0.2648`
- `ranker 20d` Sharpe ranged from `0.0086` to `-0.2706`
- `tree 20d` Sharpe ranged from `0.0287` to `-0.4188`

That intra-model swing is much larger than the gap between the best candidate configurations:

- best `ranker 10d`: Sharpe `0.0293`
- best `tree 20d`: Sharpe `0.0287`
- best `ranker 20d`: Sharpe `0.0086`

So the Phase 3 result is clear: portfolio construction choices explained more of the final net outcome than the remaining model-class differences.

### 2. Selection cutoff was the strongest design lever

Across the full Phase 3 table:

- decile portfolios had the best average net return and best average Sharpe
- quintile and top-k variants were consistently worse on average

Average Sharpe by selection method:

- decile: `-0.0288`
- quintile: `-0.1690`
- top-k: `-0.2000`

That is the strongest single Phase 3 effect.

### 3. Equal weighting beat signal weighting on average

This was the biggest surprise relative to the earlier default configuration.

Average results:

- equal weight: mean Sharpe `-0.0823`
- signal weight: mean Sharpe `-0.1830`

The reason is visible in turnover:

- equal weight average turnover: `0.0006`
- signal weight average turnover: `0.0381`

So most of the portfolio improvement came from suppressing turnover rather than extracting more value from score magnitudes.

### 4. Rebalancing every 5 days was the best compromise

Average Sharpe by rebalance frequency:

- daily: `-0.1680`
- every 5 days: `-0.1092`
- every 10 days: `-0.1206`

Average turnover followed the same pattern:

- daily: `0.0406`
- every 5 days: `0.0110`
- every 10 days: `0.0063`

The `5d` schedule was the best compromise in this study: it reduced turnover sharply versus daily rebalancing without giving up as much signal freshness as `10d`.

### 5. Turnover caps mattered less than expected once the portfolio was already low-turnover

Average results with cap on versus off were very close:

- cap off mean Sharpe: `-0.1312`
- cap on mean Sharpe: `-0.1340`

That suggests the larger gains came from the portfolio rule itself, especially equal weighting and slower rebalancing, rather than from the explicit turnover cap.

## Best Configurations

Best overall configuration by Sharpe:

- `ranker 10d`, decile, equal weight, rebalance every 5 days, turnover cap on
- net return: `0.0025`
- Sharpe: `0.0293`
- turnover: `0.0006`

Best `tree 20d` configuration:

- decile, signal weight, rebalance every 5 days, turnover cap off
- net return: `0.0026`
- Sharpe: `0.0287`
- turnover: `0.0339`

Interpretation:

- `tree 20d` still produced the highest net return among the best candidate configs
- `ranker 10d` produced the best Sharpe with much lower turnover

So the answer depends on what “best” means:

- if the focus is raw after-cost return, `tree 20d` remained competitive
- if the focus is cleaner implementability with minimal turnover, `ranker 10d` looked better

## Did Gains Come From Lower Turnover, Better Weighting, Or Rebalance Changes?

Mostly from lower turnover.

The clearest winning pattern was:

- decile selection
- equal weighting
- every-5-day rebalancing

That combination consistently pushed turnover toward zero relative to the signal-weighted alternatives, while keeping enough cross-sectional separation to preserve modest positive outcomes.

So the Phase 3 result is not that a more complicated portfolio did better. It is that a simpler and lower-turnover portfolio often did better.

## Why Portfolio Construction Before Bigger Models?

- We are not moving to RL because the current bottleneck is still mapping a weak signal into tradable positions, not solving a sequential control problem with a realistic execution simulator.
- We are not adding transformers because the remaining model differences are already smaller than the portfolio-rule differences observed in this phase.
- Portfolio construction is the correct next bottleneck because it directly determines turnover, concentration, and after-cost survival of the existing signals.

Phase 3 confirmed that this was the right sequencing decision.

## Honest Limitations

- The study only used three candidate signals from earlier phases, not the whole model universe.
- The best results are still weak in absolute terms.
- This remains a research backtest with linear transaction costs, not a production execution model.
- Some low-turnover equal-weight configurations produced nearly identical results across models, which means portfolio rules can compress model differences enough that the underlying signal distinction becomes harder to see.
