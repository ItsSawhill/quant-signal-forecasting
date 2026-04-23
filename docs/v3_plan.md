# v3 Plan

## Goal

Evolve the current repo from a solid project pipeline into a more defensible quant research framework, while keeping the work laptop-friendly and reproducible.

## Guiding Principle

Change one important axis at a time.

The repo already has enough moving parts that uncontrolled iteration would make results hard to attribute. v3 should therefore proceed as a sequence of explicit studies, each measured against the frozen v2 benchmark.

## Scoped Roadmap

### Phase 0: Baseline Freeze

- freeze the official v2 benchmark
- document the benchmark specification and regenerated reference metrics
- ensure local reruns work from cached data

### Phase 1: Multi-Horizon Target Study

- support `5d`, `10d`, and `20d` horizons for both forward-return and cross-sectional-rank targets
- make horizon configurable from the CLI
- generate comparable artifact sets per model and horizon
- determine whether horizon choice matters more than model complexity

### Phase 2: Ranking-Aware Learning

- add at least one learner that directly optimizes ranking
- start with the strongest Phase 1 horizon rather than searching all horizons again
- compare it against the frozen ridge/tree/mlp baseline under the same backtest assumptions

### Phase 3: Portfolio Construction Sensitivity

- compare quantile, quintile, and top-k constructions
- compare equal versus signal weighting
- compare rebalance frequency and turnover-control settings
- identify whether implementation choices dominate modest forecast differences

### Phase 4: Controlled Feature Research

- add a small number of interpretable, benchmark-relative, or residual-style features
- avoid a large indicator dump
- keep attribution clean enough for research writeups

### Phase 5: Robustness

- check subperiod stability
- check cost sensitivity
- check whether conclusions survive mild portfolio-rule variations

### Phase 6: Statistical Validation And Paper-Facing Outputs

- add significance, placebo, bootstrap, cost-sweep, and rolling-stability checks
- write concise research notes with hypotheses, setup, results, and limitations
- separate confirmed findings from speculative future work

## Why Not RL Or Larger Nets Yet

- RL is hard to justify before the supervised signal and execution baselines are characterized.
- Larger models increase compute cost and complexity without evidence that model capacity is the current bottleneck.
- More random indicators would weaken attribution and make the research less defensible.

The Phase 1 result supports this sequencing: horizon choice moved outcomes more than model family did.
