from __future__ import annotations

import pandas as pd


def build_positions(
    predictions: pd.DataFrame,
    top_k: int = 3,
    holding_horizon: int = 5,
) -> pd.DataFrame:
    """Convert model scores into one-day-lagged long/short positions."""
    frame = predictions.copy().sort_values(["date", "asset"])
    daily_positions = []

    for date, group in frame.groupby("date"):
        group = group.copy().sort_values("prediction")
        group["signal_weight"] = 0.0
        if len(group) >= 2:
            shorts = group.head(min(top_k, len(group) // 2)).index
            longs = group.tail(min(top_k, len(group) // 2)).index
            if len(shorts) > 0:
                group.loc[shorts, "signal_weight"] = -1.0 / len(shorts)
            if len(longs) > 0:
                group.loc[longs, "signal_weight"] = 1.0 / len(longs)
        daily_positions.append(group[["date", "asset", "signal_weight"]])

    positions = pd.concat(daily_positions, ignore_index=True) if daily_positions else pd.DataFrame(columns=["date", "asset", "signal_weight"])
    positions = positions.sort_values(["asset", "date"])
    positions["position"] = (
        positions.groupby("asset")["signal_weight"]
        .transform(lambda s: s.rolling(holding_horizon, min_periods=1).mean())
        .shift(1)
        .fillna(0.0)
    )
    return positions[["date", "asset", "position"]]


def compute_turnover(positions: pd.DataFrame) -> pd.DataFrame:
    frame = positions.copy().sort_values(["asset", "date"])
    frame["position_change"] = frame.groupby("asset")["position"].diff().abs().fillna(frame["position"].abs())
    return frame.groupby("date", as_index=False)["position_change"].sum().rename(columns={"position_change": "turnover"})


def attach_positions(predictions: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.merge(positions, on=["date", "asset"], how="left")
    frame["position"] = frame["position"].fillna(0.0)
    return frame
