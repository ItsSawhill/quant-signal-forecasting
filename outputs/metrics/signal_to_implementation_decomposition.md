# Signal-to-Implementation Decomposition

| model | horizon | mean_IC | quantile_spread | net_return | Sharpe | turnover | max_drawdown | ranking_efficiency | execution_efficiency | total_efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ranker | 10 | 0.0284 | 0.0029 | 0.0025 | 0.0293 | 0.0006 | -0.2822 | 0.1022 | 0.8679 | 0.0887 |
| ranker | 20 | 0.0310 | 0.0028 | 0.0007 | 0.0086 | 0.0006 | -0.2822 | 0.0913 | 0.2624 | 0.0240 |
| tree | 20 | 0.0105 | -0.0014 | 0.0007 | 0.0086 | 0.0006 | -0.2822 | -0.1305 | -0.5417 | 0.0707 |

## Interpretation Notes

- Ranking efficiency is highest for `ranker 10d`, meaning it converts IC into quantile spread most effectively in this small final set.
- Execution efficiency is highest for `ranker 10d`, meaning it preserves the largest share of spread into net return.
- Total efficiency is highest for `ranker 10d`, but all total-efficiency values remain economically small in absolute terms because full-sample net returns are very small.
- These ratios are descriptive, not causal. They summarize where the observed attenuation occurs, but they do not prove a unique mechanism.
- Ratios become unstable when denominators are near zero; this framework returns `NaN` in that case rather than forcing an interpretation.
