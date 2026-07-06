"""
Data extraction and cleaning module.

Fetches historical price data for TSLA, BND, and SPY from Yahoo Finance
via the yfinance library, and provides utilities for cleaning and
combining the resulting DataFrames.
"""

import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"


def fetch_asset_data(tickers=TICKERS, start=START_DATE, end=END_DATE, cache_dir="data/raw"):
    """
    Fetch historical OHLCV data for a list of tickers from Yahoo Finance.

    Results are cached to disk as CSV files so repeated runs do not
    re-hit the network. Returns a dict of {ticker: DataFrame}.
    """
    if not tickers:
        raise ValueError("fetch_asset_data requires at least one ticker symbol")

    os.makedirs(cache_dir, exist_ok=True)
    data = {}
    for ticker in tickers:
        cache_path = os.path.join(cache_dir, f"{ticker}.csv")
        if os.path.exists(cache_path):
            logger.info("Loading cached data for %s from %s", ticker, cache_path)
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            except Exception as exc:
                logger.warning(
                    "Failed to read cached data for %s from %s; falling back to download: %s",
                    ticker,
                    cache_path,
                    exc,
                )
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        if df.empty:
            logger.info("Fetching %s from Yahoo Finance (%s to %s)", ticker, start, end)
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download historical data for {ticker} from Yahoo Finance "
                    f"for the range {start} to {end}"
                ) from exc

            if df.empty:
                raise RuntimeError(
                    f"Yahoo Finance returned no rows for {ticker} between {start} and {end}"
                )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            try:
                df.to_csv(cache_path)
            except Exception as exc:
                logger.warning("Failed to cache %s data to %s: %s", ticker, cache_path, exc)
        data[ticker] = df
    return data


def clean_asset_data(data: dict) -> dict:
    """
    Clean each asset's DataFrame:
      - ensure DatetimeIndex sorted ascending
      - forward-fill missing trading days introduced by reindexing
      - drop fully empty rows
      - enforce numeric dtypes
    """
    if not data:
        raise ValueError("clean_asset_data requires a non-empty data dictionary")

    cleaned = {}
    business_days = None
    for ticker, df in data.items():
        if df is None or df.empty:
            raise ValueError(f"clean_asset_data received empty data for ticker '{ticker}'")

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        cleaned[ticker] = df
        if business_days is None:
            business_days = df.index

    if not cleaned:
        raise ValueError("clean_asset_data could not clean any assets")

    full_range = pd.date_range(
        start=min(df.index.min() for df in cleaned.values()),
        end=max(df.index.max() for df in cleaned.values()),
        freq="B",
    )

    for ticker, df in cleaned.items():
        missing_before = full_range.difference(df.index)
        df = df.reindex(full_range)
        df = df.ffill().bfill()
        cleaned[ticker] = df
        logger.info(f"{ticker}: reindexed to {len(full_range)} business days, "
                     f"filled {len(missing_before)} missing trading days")
    return cleaned


def combine_asset_data(data: dict, field="Adj Close") -> pd.DataFrame:
    """Combine a single field (default Adj Close) from each asset into one wide DataFrame."""
    missing_field = [ticker for ticker, df in data.items() if field not in df.columns]
    if missing_field:
        raise KeyError(f"Cannot combine field '{field}' because it is missing for: {', '.join(missing_field)}")

    combined = pd.DataFrame({ticker: df[field] for ticker, df in data.items()})
    combined.index.name = "Date"
    return combined


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily percentage returns from a price DataFrame."""
    return prices.pct_change().dropna()


def data_quality_report(raw: dict, cleaned: dict) -> pd.DataFrame:
    """Summarize missing values and dtype fixes applied during cleaning."""
    rows = []
    for ticker in raw:
        raw_df = raw[ticker]
        clean_df = cleaned[ticker]
        rows.append({
            "ticker": ticker,
            "raw_rows": len(raw_df),
            "raw_missing_values": int(raw_df.isna().sum().sum()),
            "cleaned_rows": len(clean_df),
            "cleaned_missing_values": int(clean_df.isna().sum().sum()),
            "date_range": f"{clean_df.index.min().date()} to {clean_df.index.max().date()}",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw = fetch_asset_data()
    cleaned = clean_asset_data(raw)
    report = data_quality_report(raw, cleaned)
    print(report.to_string(index=False))
