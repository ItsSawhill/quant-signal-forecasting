# v3 Phase 4: Controlled Feature Extension

## What Was Added

Phase 4 added a small, market-relative feature extension to the existing pipeline. No broad indicator expansion was introduced.

Added features:

- `residual_return_vs_spy_1d`
- `excess_return_vs_spy_5d`
- `excess_return_vs_spy_20d`
- `beta_adjusted_momentum_20d`

Why these were chosen:

- they are economically motivated rather than arbitrary technical additions
- they try to isolate stock-specific strength from broad market movement
- they fit naturally with the repo’s cross-sectional equity framing
- they reuse the existing `SPY` beta and benchmark machinery already in the pipeline

## What Was Tested

The Phase 4 comparison was intentionally fixed to the best Phase 3 portfolio template:

- decile selection
- equal weight
- rebalance every 5 days
- turnover cap on

Candidate signals:

- `ranker 10d`
- `ranker 20d`
- `tree 20d`

Comparison:

- baseline feature set
- extended feature set

Artifacts:

- `outputs/metrics/phase4_feature_comparison.csv`
- `outputs/metrics/phase4_feature_comparison.md`

## Results

| Model | Horizon | Feature Set | Mean IC | IC Vol | Quantile Spread | Net Return | Sharpe | Turnover | Max Drawdown |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ranker | 10d | baseline | 0.0285 | 0.2308 | 0.0027 | 0.0025 | 0.0293 | 0.0006 | -0.2822 |
| ranker | 10d | extended | 0.0284 | 0.2307 | 0.0029 | 0.0025 | 0.0293 | 0.0006 | -0.2822 |
| ranker | 20d | baseline | 0.0308 | 0.2253 | 0.0026 | 0.0007 | 0.0086 | 0.0006 | -0.2822 |
| ranker | 20d | extended | 0.0310 | 0.2243 | 0.0028 | 0.0007 | 0.0086 | 0.0006 | -0.2822 |
| tree | 20d | baseline | 0.0102 | 0.2032 | -0.0018 | 0.0007 | 0.0086 | 0.0006 | -0.2822 |
| tree | 20d | extended | 0.0105 | 0.2069 | -0.0014 | 0.0007 | 0.0086 | 0.0006 | -0.2822 |

## Impact On Signal Strength

The feature extension improved signal-quality metrics slightly, but only slightly.

Observed deltas for the extended set:

- `ranker 10d`: IC `-0.0001`, quantile spread `+0.0002`
- `ranker 20d`: IC `+0.0002`, quantile spread `+0.0003`
- `tree 20d`: IC `+0.0003`, quantile spread `+0.0004`

Interpretation:

- quantile spread improved for all three candidate signals
- mean IC improved for two of the three candidates
- the improvements were real but small

## Did Stronger Signals Translate Into Better Portfolio Performance?

Not in this Phase 4 setup.

Across all three candidates:

- net return was unchanged
- Sharpe was unchanged
- turnover was unchanged
- max drawdown was unchanged

So the honest conclusion is:

- the added features modestly improved cross-sectional separation
- but those improvements did not translate into better after-cost portfolio performance under the fixed best-Phase-3 portfolio template

## Which Features Helped Most?

The Phase 4 run does not isolate individual feature contributions. The four new features were added as a small bundle, so this phase only supports a bundle-level conclusion.

The most plausible contributors are the two medium-horizon market-relative features:

- `excess_return_vs_spy_20d`
- `beta_adjusted_momentum_20d`

Those are the features most directly aligned with the horizons and model setups that held up best in earlier phases. But that is an inference, not a directly measured attribution result.

## Why Feature Research Instead Of Larger Models?

- We are not using transformers because the repo still has not shown evidence that model capacity is the main bottleneck.
- We are not using RL because the current task is still signal discovery and portfolio translation, not sequential policy learning with a realistic execution environment.
- Signal strength remains the relevant bottleneck because the current models are extracting only weak cross-sectional structure.
- Feature quality matters more than model complexity at this stage because a cleaner signal is more likely to survive costs than a larger model fit to the same weak inputs.

Phase 4 was therefore the correct next research step even though the observed gains were modest.

## Honest Limitations

- The new features were tested only under the fixed best-Phase-3 portfolio template.
- Because that template is already extremely low-turnover and simple, it compresses downstream portfolio differences and may hide small signal improvements.
- The feature bundle was small by design, so Phase 4 was a test of disciplined feature extension, not an exhaustive search.
- No individual feature attribution study was run in this phase.
