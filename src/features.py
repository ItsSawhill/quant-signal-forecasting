from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def create_features(price_data: pd.DataFrame, include_market_relative: bool = True) -> pd.DataFrame:
    """Create no-lookahead features from daily price history."""
    frame = price_data.copy().sort_values(["asset", "date"])
    grouped = frame.groupby("asset", group_keys=False)

    frame["ret_1d"] = grouped["close"].pct_change(1)
    frame["ret_5d"] = grouped["close"].pct_change(5)
    frame["ret_10d"] = grouped["close"].pct_change(10)
    frame["ret_20d"] = grouped["close"].pct_change(20)

    daily_ret = grouped["close"].pct_change()
    frame["vol_5d"] = daily_ret.groupby(frame["asset"]).transform(lambda s: s.rolling(5).std(ddof=0))
    frame["vol_20d"] = daily_ret.groupby(frame["asset"]).transform(lambda s: s.rolling(20).std(ddof=0))

    frame["momentum_10d"] = frame["ret_10d"]
    frame["momentum_20d"] = frame["ret_20d"]

    ma_10 = grouped["close"].transform(lambda s: s.rolling(10).mean())
    ma_20 = grouped["close"].transform(lambda s: s.rolling(20).mean())
    frame["ma_gap_10d"] = frame["close"] / ma_10 - 1.0
    frame["ma_gap_20d"] = frame["close"] / ma_20 - 1.0

    frame["volume_zscore_20d"] = grouped["volume"].transform(lambda s: _rolling_zscore(s, 20))
    mean_20 = daily_ret.groupby(frame["asset"]).transform(lambda s: s.rolling(20).mean())
    frame["sharpe_proxy_20d"] = mean_20 / frame["vol_20d"].replace(0, np.nan)

    high_20 = grouped["close"].transform(lambda s: s.rolling(20).max())
    frame["drawdown_20d_high"] = frame["close"] / high_20 - 1.0

    if include_market_relative:
        spy_returns = frame.loc[frame["asset"] == "SPY", ["date", "ret_1d"]].rename(columns={"ret_1d": "spy_ret_1d"})
        frame = frame.merge(spy_returns, on="date", how="left")
        frame["market_relative_ret_1d"] = frame["ret_1d"] - frame["spy_ret_1d"]

    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vol_5d",
        "vol_20d",
        "momentum_10d",
        "momentum_20d",
        "ma_gap_10d",
        "ma_gap_20d",
        "volume_zscore_20d",
        "sharpe_proxy_20d",
        "drawdown_20d_high",
    ]
    if include_market_relative:
        feature_cols.append("market_relative_ret_1d")

    frame = frame.sort_values(["date", "asset"]).reset_index(drop=True)
    frame[feature_cols] = frame.groupby("asset")[feature_cols].shift(1)
    return frame


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "momentum_10d",
    "momentum_20d",
    "ma_gap_10d",
    "ma_gap_20d",
    "volume_zscore_20d",
    "sharpe_proxy_20d",
    "drawdown_20d_high",
    "market_relative_ret_1d",
]
