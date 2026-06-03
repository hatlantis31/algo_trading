"""
Fetch OHLCV data from IBKR (live), yfinance, or local disk cache.

Data is stored as Parquet in data/<ticker>_<period>_<interval>.parquet
(gitignored – never committed to the repo).

Typical workflow
----------------
1. First run on your machine:
       df = get_data("AAPL")          # downloads from yfinance, saves to data/
2. Subsequent runs:
       df = get_data("AAPL")          # loads from disk instantly
3. Force refresh:
       df = get_data("AAPL", refresh=True)
4. With IBKR live:
       df = get_data("SHEL", source="ibkr", ibkr_connection=conn)
"""

import logging
import os
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    safe = ticker.replace("/", "_").replace(".", "_")
    return DATA_DIR / f"{safe}_{period}_{interval}.parquet"


def _load_cache(path: Path) -> pd.DataFrame | None:
    # Prefer Parquet, but fall back to a CSV sibling if that's what we wrote.
    if path.exists():
        logger.info("Loading from cache: %s", path.name)
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        logger.info("Loading from cache: %s", csv_path.name)
        return pd.read_csv(csv_path, index_col="date", parse_dates=True)
    return None


def _save_cache(df: pd.DataFrame, path: Path):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Parquet is faster/smaller, but needs pyarrow or fastparquet. If neither is
    # installed, fall back to CSV so caching works with zero extra dependencies.
    try:
        df.to_parquet(path)
        logger.info("Saved to cache: %s", path.name)
    except ImportError:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path)
        logger.info("pyarrow/fastparquet not found — saved CSV cache: %s", csv_path.name)


# ── Sources ───────────────────────────────────────────────────────────────────

def fetch_yfinance(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    refresh: bool = False,
) -> pd.DataFrame:
    """Download via yfinance, with transparent disk caching."""
    cache = _cache_path(ticker, period, interval)
    if not refresh:
        cached = _load_cache(cache)
        if cached is not None:
            return cached

    logger.info("Downloading %s from yfinance …", ticker)
    raw = yf.download(ticker, period=period, interval=interval,
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r}")

    df = raw.copy()
    # Newer yfinance returns MultiIndex columns like ('Close', 'CRH.L').
    # Flatten to just the OHLCV field name (the first level).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    _save_cache(df, cache)
    return df


def fetch_ibkr(
    symbol: str,
    connection,
    exchange: str = "SMART",
    currency: str = "EUR",
    duration: str = "1 Y",
    bar_size: str = "1 day",
    refresh: bool = False,
) -> pd.DataFrame:
    cache = _cache_path(symbol, duration.replace(" ", ""), bar_size.replace(" ", ""))
    if not refresh:
        cached = _load_cache(cache)
        if cached is not None:
            return cached

    df = connection.get_historical_data(
        symbol=symbol, exchange=exchange, currency=currency,
        duration=duration, bar_size=bar_size,
    )
    if not df.empty:
        _save_cache(df, cache)
    return df


# ── Unified entry-point ───────────────────────────────────────────────────────

def get_data(
    ticker: str,
    source: str = "yfinance",
    period: str = "2y",
    interval: str = "1d",
    refresh: bool = False,
    ibkr_connection=None,
    **ibkr_kwargs,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    ticker   : e.g. "AAPL", "CRH.L", "SHEL.L"
    source   : "yfinance" (default) or "ibkr"
    period   : yfinance period string, e.g. "1y", "2y", "5y"
    interval : bar size, e.g. "1d", "1h", "5m"
    refresh  : force re-download even if cache exists
    """
    if source == "yfinance":
        return fetch_yfinance(ticker, period=period, interval=interval, refresh=refresh)
    elif source == "ibkr":
        if ibkr_connection is None:
            raise ValueError("ibkr_connection required when source='ibkr'")
        return fetch_ibkr(ticker, ibkr_connection, refresh=refresh, **ibkr_kwargs)
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'yfinance' or 'ibkr'.")


def list_cached() -> list[str]:
    """Show all locally cached data files."""
    if not DATA_DIR.exists():
        return []
    return [f.name for f in DATA_DIR.glob("*.parquet")]
