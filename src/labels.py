from __future__ import annotations

import numpy as np
import pandas as pd


def create_labels(
    feature_frame: pd.DataFrame,
    horizon: int = 5,
    horizons: tuple[int, ...] = (5, 10, 20),
    ranking_universe: list[str] | None = None,
) -> pd.DataFrame:
    """Create forward returns, next-day returns, and rank-based targets across horizons."""
    frame = feature_frame.copy().sort_values(["asset", "date"])
    grouped = frame.groupby("asset", group_keys=False)
    frame["next_day_return"] = grouped["close"].shift(-1) / frame["close"] - 1.0

    if ranking_universe is not None:
        rank_mask = frame["asset"].isin(ranking_universe)
    else:
        rank_mask = pd.Series(True, index=frame.index)

    for h in horizons:
        forward_col = f"forward_return_{h}d"
        binary_col = f"forward_return_binary_{h}d"
        rank_pct_col = f"forward_return_rank_pct_{h}d"
        rank_col = f"forward_return_rank_{h}d"

        frame[forward_col] = grouped["close"].shift(-h) / frame["close"] - 1.0
        frame[binary_col] = (frame[forward_col] > 0).astype(int)
        frame[rank_pct_col] = np.nan
        frame.loc[rank_mask, rank_pct_col] = (
            frame.loc[rank_mask].groupby("date")[forward_col].rank(method="average", pct=True)
        )
        frame[rank_pct_col] = frame[rank_pct_col].astype(float)
        frame[rank_col] = frame[rank_pct_col] - 0.5

    selected_forward_col = f"forward_return_{horizon}d"
    selected_binary_col = f"forward_return_binary_{horizon}d"
    selected_rank_pct_col = f"forward_return_rank_pct_{horizon}d"
    selected_rank_col = f"forward_return_rank_{horizon}d"

    frame["forward_return"] = frame[selected_forward_col]
    frame["forward_return_binary"] = frame[selected_binary_col]
    frame["forward_return_rank_pct"] = frame[selected_rank_pct_col]
    frame["forward_return_rank"] = frame[selected_rank_col]
    return frame
