from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def create_features(
    price_data: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    include_market_relative: bool = True,
    include_phase4_features: bool = False,
) -> pd.DataFrame:
    """Create no-lookahead time-series and cross-sectional features."""
    frame = price_data.copy().sort_values(["asset", "date"])
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    grouped = frame.groupby("asset", group_keys=False)

    frame["daily_return"] = grouped["close"].pct_change(1)
    frame["ret_1d"] = frame["daily_return"]
    frame["ret_5d"] = grouped["close"].pct_change(5)
    frame["ret_10d"] = grouped["close"].pct_change(10)
    frame["ret_20d"] = grouped["close"].pct_change(20)

    frame["vol_5d"] = grouped["daily_return"].transform(lambda s: s.rolling(5).std(ddof=0))
    frame["vol_20d"] = grouped["daily_return"].transform(lambda s: s.rolling(20).std(ddof=0))
    frame["momentum_10d"] = frame["ret_10d"]
    frame["momentum_20d"] = frame["ret_20d"]

    ma_10 = grouped["close"].transform(lambda s: s.rolling(10).mean())
    ma_20 = grouped["close"].transform(lambda s: s.rolling(20).mean())
    frame["ma_gap_10d"] = frame["close"] / ma_10 - 1.0
    frame["ma_gap_20d"] = frame["close"] / ma_20 - 1.0

    frame["volume_zscore_20d"] = grouped["volume"].transform(lambda s: _rolling_zscore(s, 20))
    mean_20 = grouped["daily_return"].transform(lambda s: s.rolling(20).mean())
    frame["sharpe_proxy_20d"] = mean_20 / frame["vol_20d"].replace(0, np.nan)

    high_20 = grouped["close"].transform(lambda s: s.rolling(20).max())
    frame["drawdown_20d_high"] = frame["close"] / high_20 - 1.0
    frame["downside_vol_20d"] = grouped["daily_return"].transform(lambda s: s.clip(upper=0).rolling(20).std(ddof=0))
    frame["vol_adjusted_momentum_20d"] = frame["ret_20d"] / frame["vol_20d"].replace(0, np.nan)

    benchmark_only = frame.loc[frame["asset"] == benchmark_ticker, ["date", "close", "daily_return"]].copy().sort_values("date")
    benchmark_returns = benchmark_only.rename(columns={"daily_return": "benchmark_return_1d"})
    benchmark_returns["benchmark_return_5d"] = benchmark_returns["close"].pct_change(5)
    benchmark_returns["benchmark_return_20d"] = benchmark_returns["close"].pct_change(20)
    benchmark_returns["benchmark_var_20d"] = benchmark_returns["benchmark_return_1d"].rolling(20).var(ddof=0)
    frame = frame.merge(
        benchmark_returns[["date", "benchmark_return_1d", "benchmark_return_5d", "benchmark_return_20d", "benchmark_var_20d"]],
        on="date",
        how="left",
    )
    frame["return_x_benchmark"] = frame["daily_return"] * frame["benchmark_return_1d"]
    grouped = frame.groupby("asset", group_keys=False)

    asset_mean = grouped["daily_return"].transform(lambda s: s.rolling(20).mean())
    benchmark_mean = grouped["benchmark_return_1d"].transform(lambda s: s.rolling(20).mean())
    cross_mean = grouped["return_x_benchmark"].transform(lambda s: s.rolling(20).mean())
    covariance = cross_mean - asset_mean * benchmark_mean

    frame["beta_to_spy_20d"] = covariance / frame["benchmark_var_20d"].replace(0, np.nan)
    frame["corr_to_spy_20d"] = grouped.apply(
        lambda df: df["daily_return"].rolling(20).corr(df["benchmark_return_1d"])
    ).reset_index(level=0, drop=True)

    if include_market_relative:
        frame["market_relative_ret_1d"] = frame["ret_1d"] - frame["benchmark_return_1d"]

    if include_phase4_features:
        # Residualized daily return strips out the beta-scaled market move.
        frame["residual_return_vs_spy_1d"] = frame["ret_1d"] - frame["beta_to_spy_20d"] * frame["benchmark_return_1d"]
        # Excess-return features compare stock momentum directly against SPY over the same horizon.
        frame["excess_return_vs_spy_5d"] = frame["ret_5d"] - frame["benchmark_return_5d"]
        frame["excess_return_vs_spy_20d"] = frame["ret_20d"] - frame["benchmark_return_20d"]
        # Beta-adjusted momentum keeps long-horizon strength but removes estimated market exposure.
        frame["beta_adjusted_momentum_20d"] = frame["ret_20d"] - frame["beta_to_spy_20d"] * frame["benchmark_return_20d"]

    raw_feature_cols = [
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
        "downside_vol_20d",
        "vol_adjusted_momentum_20d",
        "beta_to_spy_20d",
        "corr_to_spy_20d",
    ]
    if include_market_relative:
        raw_feature_cols.append("market_relative_ret_1d")
    if include_phase4_features:
        raw_feature_cols.extend(
            [
                "residual_return_vs_spy_1d",
                "excess_return_vs_spy_5d",
                "excess_return_vs_spy_20d",
                "beta_adjusted_momentum_20d",
            ]
        )

    cross_sectional_base = [
        "ret_5d",
        "ret_20d",
        "momentum_20d",
        "ma_gap_20d",
        "vol_20d",
        "volume_zscore_20d",
        "beta_to_spy_20d",
        "corr_to_spy_20d",
        "vol_adjusted_momentum_20d",
    ]
    cross_sectional_cols = []
    for col in cross_sectional_base:
        xs_col = f"{col}_xs_z"
        frame[xs_col] = frame.groupby("date")[col].transform(_cross_sectional_zscore)
        cross_sectional_cols.append(xs_col)

    feature_cols = raw_feature_cols + cross_sectional_cols
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
    "downside_vol_20d",
    "vol_adjusted_momentum_20d",
    "beta_to_spy_20d",
    "corr_to_spy_20d",
    "market_relative_ret_1d",
    "ret_5d_xs_z",
    "ret_20d_xs_z",
    "momentum_20d_xs_z",
    "ma_gap_20d_xs_z",
    "vol_20d_xs_z",
    "volume_zscore_20d_xs_z",
    "beta_to_spy_20d_xs_z",
    "corr_to_spy_20d_xs_z",
    "vol_adjusted_momentum_20d_xs_z",
]


EXTENDED_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "residual_return_vs_spy_1d",
    "excess_return_vs_spy_5d",
    "excess_return_vs_spy_20d",
    "beta_adjusted_momentum_20d",
]
