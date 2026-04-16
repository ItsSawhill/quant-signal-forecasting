# Leakage and Alignment Audit

Reviewed files:

- `src/features.py`
- `src/labels.py`
- `src/train.py`
- `src/portfolio.py`
- `src/backtest.py`

Findings:

- Feature windows are trailing and then shifted by one day before modeling use.
- Labels are forward 5-day returns, created after feature engineering.
- Walk-forward splits are chronological and expanding.
- Portfolio positions are shifted by one day before return application.
- Backtest returns include transaction costs based on turnover.

Residual caveats:

- The backtest is a simplified research approximation because it uses overlapping forward returns rather than a path-level execution simulator.
- Transaction costs are linear and do not model slippage, financing, or borrow costs.
- Data comes from `yfinance`, which is acceptable for research but not institutional-grade.

Conclusion:

- No explicit lookahead bias was found in the current implementation.
- Reported backtest results should still be interpreted as research outputs, not deployable production estimates.
