from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BENCHMARK_TICKER = "SPY"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "V",
    "MA",
    "UNH",
    "XOM",
    "JNJ",
    "WMT",
    "PG",
    "HD",
    "COST",
    "ABBV",
    "CVX",
    "BAC",
    "KO",
    "PEP",
    "AVGO",
    "MRK",
    "TMO",
    "ORCL",
    "MCD",
    "CRM",
    "ADBE",
    "ACN",
    "LIN",
    "AMD",
    "CSCO",
    "NFLX",
    "QCOM",
    "ABT",
    "DHR",
    "WFC",
    "TXN",
    "INTC",
    "PM",
    "GE",
    "IBM",
    "CAT",
    "GS",
    "DIS",
    "AMGN",
    "NOW",
    "ISRG",
    "BKNG",
    "SPGI",
    "HON",
    "INTU",
    "BLK",
    "SYK",
    "C",
    "PLD",
    "MDT",
]


def _filter_cached_market_data(
    data: pd.DataFrame,
    tickers: Iterable[str] | None,
    start_date: str,
    end_date: str | None,
    benchmark_ticker: str,
) -> pd.DataFrame:
    universe = list(dict.fromkeys(tickers or DEFAULT_TICKERS))
    symbols = set(universe if benchmark_ticker in universe else universe + [benchmark_ticker])
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["asset"].isin(symbols)].copy()
    frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        raise ValueError("Cached market data is empty after applying the requested filters.")

    available_assets = set(frame["asset"].unique())
    missing_assets = symbols - available_assets
    if missing_assets:
        missing = ", ".join(sorted(missing_assets))
        raise ValueError(f"Cached market data is missing requested assets: {missing}")
    return frame.sort_values(["asset", "date"]).reset_index(drop=True)


def download_market_data(
    tickers: Iterable[str] | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    output_path: Path | None = None,
    benchmark_ticker: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Download adjusted OHLCV data for the requested universe and benchmark."""
    universe = list(dict.fromkeys(tickers or DEFAULT_TICKERS))
    symbols = universe if benchmark_ticker in universe else universe + [benchmark_ticker]
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
        available_tickers = set(raw.columns.get_level_values(0))
        for ticker in symbols:
            if ticker not in available_tickers:
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
    data = data.loc[:, ~data.columns.duplicated()].copy()
    keep_cols = [col for col in ["date", "asset", "open", "high", "low", "close", "volume"] if col in data.columns]
    data = data[keep_cols].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["asset", "date"]).dropna(subset=["close"])
    data["volume"] = data["volume"].fillna(0.0)
    if benchmark_ticker not in set(data["asset"]):
        raise ValueError(f"Benchmark ticker {benchmark_ticker} was not available in the downloaded data.")
    data.to_csv(target_path, index=False)
    return data


def load_market_data(path: Path | None = None) -> pd.DataFrame:
    """Load cached market data."""
    target_path = path or DATA_DIR / "market_data.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"Market data file not found: {target_path}")
    data = pd.read_csv(target_path, parse_dates=["date"])
    return data.sort_values(["asset", "date"]).reset_index(drop=True)


def get_market_data(
    tickers: Iterable[str] | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    output_path: Path | None = None,
    benchmark_ticker: str = BENCHMARK_TICKER,
    prefer_cache: bool = True,
) -> pd.DataFrame:
    """Load cached data when available, otherwise download fresh data."""
    target_path = output_path or DATA_DIR / "market_data.csv"
    if prefer_cache and target_path.exists():
        cached = load_market_data(target_path)
        return _filter_cached_market_data(
            cached,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            benchmark_ticker=benchmark_ticker,
        )

    try:
        return download_market_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_path=target_path,
            benchmark_ticker=benchmark_ticker,
        )
    except Exception:
        if prefer_cache and target_path.exists():
            cached = load_market_data(target_path)
            return _filter_cached_market_data(
                cached,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                benchmark_ticker=benchmark_ticker,
            )
        raise


if __name__ == "__main__":
    frame = download_market_data()
    print(frame.head())
