from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_long_short(long_scores: pd.Series, short_scores: pd.Series, gross_exposure: float) -> pd.Series:
    weights = pd.Series(0.0, index=long_scores.index.union(short_scores.index))
    half_gross = gross_exposure / 2.0

    if not long_scores.empty:
        if long_scores.sum() <= 0:
            weights.loc[long_scores.index] = half_gross / len(long_scores)
        else:
            weights.loc[long_scores.index] = half_gross * (long_scores / long_scores.sum())

    if not short_scores.empty:
        if short_scores.sum() <= 0:
            weights.loc[short_scores.index] = -(half_gross / len(short_scores))
        else:
            weights.loc[short_scores.index] = -half_gross * (short_scores / short_scores.sum())
    return weights


def _signal_weighted_scores(group: pd.DataFrame, selected_assets: pd.Index, side: str) -> pd.Series:
    subset = group.loc[selected_assets, "prediction"]
    median_prediction = group["prediction"].median()
    if side == "long":
        scores = (subset - median_prediction).clip(lower=0)
    else:
        scores = (median_prediction - subset).clip(lower=0)
    if float(scores.sum()) == 0.0:
        scores = pd.Series(1.0, index=selected_assets)
    return scores


def _build_target_weights_for_date(
    group: pd.DataFrame,
    portfolio_mode: str,
    weight_scheme: str,
    top_k: int,
    quantile: float,
    gross_exposure: float,
) -> pd.Series:
    group = group.sort_values("prediction")
    n_assets = len(group)
    if n_assets < 2:
        return pd.Series(0.0, index=group["asset"])

    if portfolio_mode == "quantile":
        n_selected = max(1, int(np.floor(n_assets * quantile)))
    else:
        n_selected = min(top_k, n_assets // 2)

    short_assets = group.head(n_selected)["asset"]
    long_assets = group.tail(n_selected)["asset"]

    if weight_scheme == "signal":
        long_scores = _signal_weighted_scores(group.set_index("asset"), long_assets, side="long")
        short_scores = _signal_weighted_scores(group.set_index("asset"), short_assets, side="short")
    else:
        long_scores = pd.Series(1.0, index=long_assets)
        short_scores = pd.Series(1.0, index=short_assets)

    weights = _normalize_long_short(long_scores, short_scores, gross_exposure=gross_exposure)
    return weights.reindex(group["asset"]).fillna(0.0)


def build_positions(
    predictions: pd.DataFrame,
    top_k: int = 3,
    holding_horizon: int = 5,
    portfolio_mode: str = "quantile",
    weight_scheme: str = "signal",
    quantile: float = 0.1,
    gross_exposure: float = 1.0,
    max_turnover: float | None = None,
    rebalance_frequency: int | None = None,
) -> pd.DataFrame:
    """Convert model scores into lagged positions with configurable rebalancing."""
    frame = predictions.copy().sort_values(["date", "asset"])
    assets = sorted(frame["asset"].unique())
    dates = sorted(frame["date"].unique())
    rebalance_every = rebalance_frequency or holding_horizon

    current_target = pd.Series(0.0, index=assets)
    rows = []

    for date_idx, date in enumerate(dates):
        group = frame.loc[frame["date"] == date, ["asset", "prediction"]].copy()
        group = group[group["asset"].isin(assets)]
        if date_idx % rebalance_every == 0:
            desired = _build_target_weights_for_date(
                group=group,
                portfolio_mode=portfolio_mode,
                weight_scheme=weight_scheme,
                top_k=top_k,
                quantile=quantile,
                gross_exposure=gross_exposure,
            )
            desired.index = group["asset"].values
            desired = desired.reindex(assets).fillna(0.0)

            if max_turnover is not None:
                delta = desired - current_target
                turnover = float(delta.abs().sum())
                if turnover > max_turnover and turnover > 0:
                    desired = current_target + delta * (max_turnover / turnover)
            current_target = desired

        rows.append(
            pd.DataFrame(
                {
                    "date": date,
                    "asset": assets,
                    "target_position": current_target.values,
                }
            )
        )

    positions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "asset", "target_position"])
    positions = positions.sort_values(["asset", "date"]).reset_index(drop=True)
    positions["position"] = positions.groupby("asset")["target_position"].shift(1).fillna(0.0)
    return positions[["date", "asset", "target_position", "position"]]


def compute_turnover(positions: pd.DataFrame) -> pd.DataFrame:
    frame = positions.copy().sort_values(["asset", "date"])
    frame["position_change"] = frame.groupby("asset")["position"].diff().abs().fillna(frame["position"].abs())
    return frame.groupby("date", as_index=False)["position_change"].sum().rename(columns={"position_change": "turnover"})


def attach_positions(predictions: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.merge(positions, on=["date", "asset"], how="left")
    for col in ["target_position", "position"]:
        frame[col] = frame[col].fillna(0.0)
    return frame
