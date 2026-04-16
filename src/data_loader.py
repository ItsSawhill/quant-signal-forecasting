from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU"]


def download_market_data(
    tickers: Iterable[str] | None = None,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Download adjusted OHLCV data and save a cleaned long-form dataset."""
    symbols = list(tickers or DEFAULT_TICKERS)
    target_path = output_path or DATA_DIR / "market_data.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    raw = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise ValueError("No data was returned from yfinance. Check the ticker list or date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        long_frames = []
        for ticker in symbols:
            if ticker not in raw.columns.get_level_values(0):
                continue
            frame = raw[ticker].copy()
            frame["asset"] = ticker
            frame = frame.reset_index().rename(columns=str.lower)
            long_frames.append(frame)
        if not long_frames:
            raise ValueError("Downloaded data did not contain any requested tickers.")
        data = pd.concat(long_frames, ignore_index=True)
    else:
        data = raw.reset_index().rename(columns=str.lower)
        data["asset"] = symbols[0]

    rename_map = {"adj close": "close", "date": "date", "capital gains": "capital_gains", "stock splits": "stock_splits"}
    data = data.rename(columns=rename_map)
    keep_cols = [col for col in ["date", "asset", "open", "high", "low", "close", "volume"] if col in data.columns]
    data = data[keep_cols].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["asset", "date"]).dropna(subset=["close"])
    data["volume"] = data["volume"].fillna(0.0)
    data.to_csv(target_path, index=False)
    return data


def load_market_data(path: Path | None = None) -> pd.DataFrame:
    """Load cached market data."""
    target_path = path or DATA_DIR / "market_data.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"Market data file not found: {target_path}")
    data = pd.read_csv(target_path, parse_dates=["date"])
    return data.sort_values(["asset", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    frame = download_market_data()
    print(frame.head())
