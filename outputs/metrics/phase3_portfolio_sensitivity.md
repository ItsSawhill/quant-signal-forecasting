# Phase 3 Portfolio Sensitivity

## Best Configuration By Candidate

| model | horizon | selection_method | weighting_scheme | rebalance_frequency | turnover_cap | gross_return | net_return | sharpe | max_drawdown | turnover | annualized_volatility | mean_IC | quantile_spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ranker | 10 | decile | equal | 5 | on | 0.0026 | 0.0025 | 0.0293 | -0.2822 | 0.0006 | 0.0859 | 0.0285 | 0.0027 |
| ranker | 20 | decile | equal | 5 | on | 0.0008 | 0.0007 | 0.0086 | -0.2822 | 0.0006 | 0.0860 | 0.0308 | 0.0026 |
| tree | 20 | decile | signal | 5 | off | 0.0069 | 0.0026 | 0.0287 | -0.3212 | 0.0339 | 0.0904 | 0.0102 | -0.0018 |

## Top 15 Configurations By Sharpe

| model | horizon | selection_method | weighting_scheme | rebalance_frequency | turnover_cap | gross_return | net_return | sharpe | turnover | mean_IC | quantile_spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ranker | 10 | decile | equal | 5 | on | 0.0026 | 0.0025 | 0.0293 | 0.0006 | 0.0285 | 0.0027 |
| tree | 20 | decile | signal | 5 | off | 0.0069 | 0.0026 | 0.0287 | 0.0339 | 0.0102 | -0.0018 |
| tree | 20 | decile | signal | 5 | on | 0.0069 | 0.0026 | 0.0286 | 0.0338 | 0.0102 | -0.0018 |
| ranker | 10 | decile | equal | 5 | off | 0.0024 | 0.0023 | 0.0267 | 0.0006 | 0.0285 | 0.0027 |
| ranker | 10 | decile | equal | 10 | off | 0.0024 | 0.0023 | 0.0267 | 0.0006 | 0.0285 | 0.0027 |
| ranker | 10 | decile | equal | 1 | off | 0.0024 | 0.0023 | 0.0267 | 0.0006 | 0.0285 | 0.0027 |
| ranker | 10 | decile | equal | 1 | on | 0.0023 | 0.0023 | 0.0264 | 0.0006 | 0.0285 | 0.0027 |
| ranker | 10 | decile | equal | 10 | on | 0.0018 | 0.0018 | 0.0204 | 0.0006 | 0.0285 | 0.0027 |
| ranker | 20 | decile | equal | 5 | on | 0.0008 | 0.0007 | 0.0086 | 0.0006 | 0.0308 | 0.0026 |
| tree | 20 | decile | equal | 5 | on | 0.0008 | 0.0007 | 0.0086 | 0.0006 | 0.0102 | -0.0018 |
| tree | 20 | decile | equal | 1 | off | 0.0006 | 0.0005 | 0.0061 | 0.0006 | 0.0102 | -0.0018 |
| tree | 20 | decile | equal | 10 | off | 0.0006 | 0.0005 | 0.0061 | 0.0006 | 0.0102 | -0.0018 |
| tree | 20 | decile | equal | 5 | off | 0.0006 | 0.0005 | 0.0061 | 0.0006 | 0.0102 | -0.0018 |
| ranker | 20 | decile | equal | 10 | off | 0.0006 | 0.0005 | 0.0061 | 0.0006 | 0.0308 | 0.0026 |
| ranker | 20 | decile | equal | 5 | off | 0.0006 | 0.0005 | 0.0061 | 0.0006 | 0.0308 | 0.0026 |
