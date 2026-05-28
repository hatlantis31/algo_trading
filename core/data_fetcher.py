"""
Fetch OHLCV data either from IBKR (live) or yfinance (offline/backtest).
Returns a standardised DataFrame with columns: open, high, low, close, volume.
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_yfinance(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download data via yfinance – useful for offline backtesting."""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_ibkr(
    symbol: str,
    connection,           # IBKRConnection instance
    exchange: str = "SMART",
    currency: str = "EUR",
    duration: str = "1 Y",
    bar_size: str = "1 day",
) -> pd.DataFrame:
    return connection.get_historical_data(
        symbol=symbol,
        exchange=exchange,
        currency=currency,
        duration=duration,
        bar_size=bar_size,
    )


def get_data(
    ticker: str,
    source: str = "yfinance",
    period: str = "2y",
    interval: str = "1d",
    ibkr_connection=None,
    **ibkr_kwargs,
) -> pd.DataFrame:
    """
    Unified entry-point.
    source = "yfinance"  -> no broker needed, great for backtesting
    source = "ibkr"      -> requires a live IBKRConnection
    """
    if source == "yfinance":
        return fetch_yfinance(ticker, period=period, interval=interval)
    elif source == "ibkr":
        if ibkr_connection is None:
            raise ValueError("ibkr_connection required when source='ibkr'")
        return fetch_ibkr(ticker, ibkr_connection, **ibkr_kwargs)
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'yfinance' or 'ibkr'.")
