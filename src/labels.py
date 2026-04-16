from __future__ import annotations

import pandas as pd


def create_labels(
    feature_frame: pd.DataFrame,
    horizon: int = 5,
    ranking_universe: list[str] | None = None,
) -> pd.DataFrame:
    """Create forward returns, next-day returns, and rank-based targets."""
    frame = feature_frame.copy().sort_values(["asset", "date"])
    grouped = frame.groupby("asset", group_keys=False)
    frame["forward_return"] = grouped["close"].shift(-horizon) / frame["close"] - 1.0
    frame["next_day_return"] = grouped["close"].shift(-1) / frame["close"] - 1.0
    frame["forward_return_binary"] = (frame["forward_return"] > 0).astype(int)

    if ranking_universe is not None:
        rank_mask = frame["asset"].isin(ranking_universe)
    else:
        rank_mask = pd.Series(True, index=frame.index)

    rank_frame = frame.loc[rank_mask, ["date", "asset", "forward_return"]].copy()
    rank_frame["forward_return_rank_pct"] = rank_frame.groupby("date")["forward_return"].rank(method="average", pct=True)
    rank_frame["forward_return_rank"] = rank_frame["forward_return_rank_pct"] - 0.5
    frame = frame.merge(
        rank_frame[["date", "asset", "forward_return_rank_pct", "forward_return_rank"]],
        on=["date", "asset"],
        how="left",
    )
    return frame
