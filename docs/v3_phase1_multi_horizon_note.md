# v3 Phase 1: Multi-Horizon Target Study

## Scope

Phase 1 asks whether horizon choice matters more than model complexity for the current cross-sectional ranking setup.

Compared runs:

- models: `ridge`, `tree`, `mlp`
- target mode: `cross_sectional_rank`
- horizons: `5`, `10`, `20`
- portfolio defaults: quantile, signal-weighted, top/bottom decile, gross exposure `1.0`, max turnover `0.5`
- transaction costs: `5 bps`

Reference commands:

```bash
env MPLCONFIGDIR=/tmp/mpl python src/research.py horizon-study --target-mode cross_sectional_rank --models ridge tree mlp --horizons 5 10 20
env MPLCONFIGDIR=/tmp/mpl python src/train.py --model mlp --task regression --target-mode cross_sectional_rank --horizon 5
env MPLCONFIGDIR=/tmp/mpl python src/train.py --model mlp --task regression --target-mode cross_sectional_rank --horizon 10
env MPLCONFIGDIR=/tmp/mpl python src/train.py --model mlp --task regression --target-mode cross_sectional_rank --horizon 20
```

Primary comparison artifact:

- `outputs/metrics/horizon_study_cross_sectional_rank.csv`

Per-run artifacts include:

- metrics summary
- IC by date
- quantile returns
- quantile spread returns
- portfolio returns
- backtest summary

## Important Correction

During Phase 1 implementation, a bug in `src/labels.py` corrupted the original `10d` and `20d` labels by reusing a stale grouped object after merge operations. That inflated the early long-horizon results and made them unusable. The bug was fixed before the final results below were frozen.

## Results

| Model | Horizon | Mean IC | IC Vol | Quantile Spread | Turnover | Ann. Return | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge | 5d | 0.0087 | 0.2516 | 0.0015 | 0.0323 | -0.0105 | -0.1160 | -0.3469 |
| ridge | 10d | 0.0041 | 0.2466 | 0.0008 | 0.0308 | -0.0122 | -0.1367 | -0.3430 |
| ridge | 20d | 0.0033 | 0.2499 | -0.0021 | 0.0249 | -0.0148 | -0.1652 | -0.3370 |
| tree | 5d | 0.0153 | 0.1997 | 0.0014 | 0.0385 | -0.0238 | -0.2568 | -0.3467 |
| tree | 10d | 0.0221 | 0.1991 | 0.0023 | 0.0382 | -0.0103 | -0.1112 | -0.3302 |
| tree | 20d | 0.0102 | 0.2032 | -0.0018 | 0.0338 | 0.0026 | 0.0286 | -0.3212 |
| mlp | 5d | 0.0224 | 0.1993 | 0.0027 | 0.0296 | -0.0104 | -0.1182 | -0.3210 |
| mlp | 10d | 0.0146 | 0.2072 | 0.0015 | 0.0294 | -0.0227 | -0.2555 | -0.3411 |
| mlp | 20d | 0.0194 | 0.1910 | 0.0021 | 0.0291 | -0.0158 | -0.1784 | -0.3352 |

## Main Findings

### 1. Horizon mattered, but less than the corrupted first pass suggested

After the label fix, the horizon effect is mixed rather than decisive.

- `tree` improved from mean IC `0.0153` at `5d` to `0.0221` at `10d`
- `ridge` was best at `5d` and weakened at longer horizons
- `mlp` was also best at `5d`, with `20d` second and `10d` weakest

That means there is no clean “medium horizon wins” result across the current model family.

### 2. The strongest implementability result came from `tree 20d`, not from the highest-IC setup

The only positive after-cost result in the grid was:

- `tree 20d`: annualized return `0.0026`, Sharpe `0.0286`, turnover `0.0338`

But that did not coincide with the best ranking metrics. The best mean IC was `mlp 5d` at `0.0224`, with `tree 10d` close behind at `0.0221`.

So the current evidence says:

- higher IC did not translate cleanly into better implementability
- longer horizons may help turnover
- the link between ranking quality and after-cost PnL remains weak in this repo

### 3. Longer horizons reduced turnover modestly

Turnover generally fell as horizon increased:

- `ridge`: `0.0323` at `5d` to `0.0249` at `20d`
- `tree`: `0.0385` at `5d` to `0.0338` at `20d`
- `mlp`: `0.0296` at `5d` to `0.0291` at `20d`

That is directionally consistent with the implementability thesis, even though the net return benefit remained weak.

### 4. The model-complexity story is still limited

The `mlp` did not earn a clear practical advantage over simpler models.

- best mean IC in the grid: `mlp 5d` at `0.0224`
- best net return in the grid: `tree 20d` at `0.0026`
- best Sharpe in the grid: `tree 20d` at `0.0286`

Those are still weak research results rather than evidence for a stronger model class.

## Why Testing Horizons Was Still The Right Next Step

Testing horizons remained the right Phase 1 move because horizon directly changes:

- turnover pressure
- signal persistence
- overlap in realized returns
- the gap between cross-sectional sorting quality and after-cost portfolio performance

That is still more central to the research question than jumping to RL or much larger neural models.

## Honest Interpretation

- all runs underperformed `SPY`
- only `tree 20d` produced a slightly positive after-cost annualized return, and it was still very weak
- the repo now has a correct multi-horizon research path, which is the real Phase 1 deliverable
- the evidence for “medium horizons are more implementable” is weak and model-dependent

## Laptop-Friendly Note

The `mlp` path was changed to use a lightweight scikit-learn baseline by default, with the older PyTorch version available only if `USE_TORCH_MLP=1` is set. This keeps the repo runnable locally, but the scikit-learn `mlp` still emits convergence warnings at `max_iter=20`, so its results should be treated as a lightweight baseline rather than a tuned neural benchmark.
