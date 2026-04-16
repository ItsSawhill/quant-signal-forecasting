from __future__ import annotations

import pandas as pd


def create_labels(feature_frame: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create forward returns and binary labels aligned by asset and date."""
    frame = feature_frame.copy().sort_values(["asset", "date"])
    grouped = frame.groupby("asset", group_keys=False)
    frame["forward_return"] = grouped["close"].shift(-horizon) / frame["close"] - 1.0
    frame["forward_return_binary"] = (frame["forward_return"] > 0).astype(int)
    return frame
