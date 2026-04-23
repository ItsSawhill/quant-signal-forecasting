# Phase 6 Validation Summary

## IC Significance

| model | horizon | mean_ic | std_ic | t_stat | p_value | n_dates |
| --- | --- | --- | --- | --- | --- | --- |
| ranker | 10 | 0.0284 | 0.2307 | 5.2188 | 0.0000 | 1800.0000 |
| ranker | 20 | 0.0310 | 0.2243 | 5.8534 | 0.0000 | 1790.0000 |
| tree | 20 | 0.0105 | 0.2069 | 2.1506 | 0.0315 | 1790.0000 |

## Bootstrap Confidence Intervals

| model | horizon | annualized_return_ci_lower | annualized_return_ci_upper | sharpe_ci_lower | sharpe_ci_upper | quantile_spread_ci_lower | quantile_spread_ci_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ranker | 10 | -0.0540 | 0.0791 | -0.6224 | 0.9223 | 0.0007 | 0.0051 |
| ranker | 20 | -0.0589 | 0.0648 | -0.6867 | 0.7321 | -0.0001 | 0.0061 |
| tree | 20 | -0.0589 | 0.0648 | -0.6867 | 0.7321 | -0.0039 | 0.0016 |

## Placebo Comparison

| model | horizon | variant | mean_ic | quantile_spread | annualized_return | sharpe | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ranker | 10 | real | 0.0284 | 0.0029 | 0.0025 | 0.0293 | 0.0006 |
| ranker | 10 | random_ranking | -0.0018 | -0.0004 | 0.0025 | 0.0293 | 0.0006 |
| ranker | 10 | label_shuffled | -0.0024 | -0.0008 | 0.0025 | 0.0293 | 0.0006 |
| ranker | 20 | real | 0.0310 | 0.0028 | 0.0007 | 0.0086 | 0.0006 |
| ranker | 20 | random_ranking | -0.0026 | 0.0003 | 0.0007 | 0.0086 | 0.0006 |
| ranker | 20 | label_shuffled | 0.0008 | 0.0011 | 0.0007 | 0.0086 | 0.0006 |
| tree | 20 | real | 0.0105 | -0.0014 | 0.0007 | 0.0086 | 0.0006 |
| tree | 20 | random_ranking | 0.0002 | -0.0001 | 0.0007 | 0.0086 | 0.0006 |
| tree | 20 | label_shuffled | 0.0012 | -0.0016 | 0.0007 | 0.0086 | 0.0006 |

## Cost Sweep

| model | horizon | cost_bps | annualized_return | sharpe | turnover |
| --- | --- | --- | --- | --- | --- |
| ranker | 10 | 0.0000 | 0.0026 | 0.0301 | 0.0006 |
| ranker | 10 | 2.0000 | 0.0026 | 0.0298 | 0.0006 |
| ranker | 10 | 5.0000 | 0.0025 | 0.0293 | 0.0006 |
| ranker | 10 | 10.0000 | 0.0024 | 0.0285 | 0.0006 |
| ranker | 10 | 20.0000 | 0.0023 | 0.0268 | 0.0006 |
| ranker | 20 | 0.0000 | 0.0008 | 0.0095 | 0.0006 |
| ranker | 20 | 2.0000 | 0.0008 | 0.0091 | 0.0006 |
| ranker | 20 | 5.0000 | 0.0007 | 0.0086 | 0.0006 |
| ranker | 20 | 10.0000 | 0.0007 | 0.0078 | 0.0006 |
| ranker | 20 | 20.0000 | 0.0005 | 0.0062 | 0.0006 |
| tree | 20 | 0.0000 | 0.0008 | 0.0095 | 0.0006 |
| tree | 20 | 2.0000 | 0.0008 | 0.0091 | 0.0006 |
| tree | 20 | 5.0000 | 0.0007 | 0.0086 | 0.0006 |
| tree | 20 | 10.0000 | 0.0007 | 0.0078 | 0.0006 |
| tree | 20 | 20.0000 | 0.0005 | 0.0062 | 0.0006 |

