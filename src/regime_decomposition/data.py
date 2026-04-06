from __future__ import annotations

from pathlib import Path
from time import sleep

import pandas as pd
import yfinance as yf

from regime_decomposition.config import DATA_RAW_DIR, MarketDataConfig


def download_market_data(config: MarketDataConfig, output_path: Path | None = None) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance.

    The returned frame has a DatetimeIndex and two-level columns:
    first level is ticker, second level is field.
    """

    frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for ticker in config.tickers:
        ticker_data = _download_single_ticker(ticker=ticker, config=config)
        if ticker_data.empty:
            failed.append(ticker)
        else:
            frames[ticker] = ticker_data

    if failed:
        raise RuntimeError(f"Failed to download required tickers: {failed}")
    if not frames:
        raise RuntimeError("Downloaded market data is empty. Check ticker symbols or network access.")

    data = pd.concat(frames, axis=1)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    if output_path is None:
        suffix_end = config.end or "latest"
        output_path = DATA_RAW_DIR / f"market_data_{config.start}_{suffix_end}.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path)
    return data


def load_or_download_market_data(
    config: MarketDataConfig,
    output_path: Path | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load cached market data when available, otherwise download it."""

    if output_path is None:
        suffix_end = config.end or "latest"
        output_path = DATA_RAW_DIR / f"market_data_{config.start}_{suffix_end}.parquet"

    if output_path.exists() and not force_download:
        return pd.read_parquet(output_path)

    return download_market_data(config=config, output_path=output_path)


def _download_single_ticker(ticker: str, config: MarketDataConfig, attempts: int = 3) -> pd.DataFrame:
    """Download one ticker with light retry logic.

    Sequential downloads are slower than a threaded batch call but avoid flaky
    yfinance cache locking and make failed symbols explicit.
    """

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            data = yf.download(
                tickers=ticker,
                start=config.start,
                end=config.end,
                auto_adjust=config.auto_adjust,
                progress=False,
                threads=False,
            )
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data = _normalize_yfinance_columns(data, (ticker,)).xs(ticker, axis=1, level=0)
                return data.dropna(how="all")
        except Exception as exc:  # pragma: no cover - network/client failure path
            last_error = exc

        if attempt < attempts:
            sleep(1.5 * attempt)

    if last_error is not None:
        print(f"Failed to download {ticker}: {last_error}")
    return pd.DataFrame()


def _normalize_yfinance_columns(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Return columns in the canonical (ticker, field) order."""

    if not isinstance(raw.columns, pd.MultiIndex):
        if len(tickers) != 1:
            raise ValueError("Expected MultiIndex columns for multi-ticker yfinance download.")
        raw = pd.concat({tickers[0]: raw}, axis=1)

    first_level = set(raw.columns.get_level_values(0))
    second_level = set(raw.columns.get_level_values(1))

    if first_level.intersection(tickers):
        data = raw.copy()
    elif second_level.intersection(tickers):
        data = raw.swaplevel(axis=1)
    else:
        raise ValueError("Could not infer yfinance column layout.")

    data = data.sort_index(axis=1, level=[0, 1])
    return data
