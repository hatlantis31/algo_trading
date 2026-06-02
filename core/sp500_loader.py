"""
Loader for the bundled real S&P 500 5-year daily dataset
(data/sp500_5yr.csv – 505 stocks, 2013-02 to 2018-02).

This is REAL market data, used for backtesting when live/yfinance access
is unavailable. Download once with download_sp500():

    from core.sp500_loader import download_sp500, load_sp500, get_panel
    download_sp500()                      # one-time, ~29 MB
    df = load_sp500("AAPL")               # single ticker OHLCV
    panel = get_panel(["AAPL","MSFT"])    # wide close-price panel
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "sp500_5yr.csv"
URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"

_CACHE: pd.DataFrame | None = None


def download_sp500(force: bool = False) -> Path:
    """Download the dataset to data/ (gitignored). One-time ~29 MB."""
    if CSV_PATH.exists() and not force:
        logger.info("Already present: %s", CSV_PATH)
        return CSV_PATH
    import urllib.request
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading S&P 500 5yr dataset …")
    urllib.request.urlretrieve(URL, CSV_PATH)
    logger.info("Saved: %s", CSV_PATH)
    return CSV_PATH


def _load_raw() -> pd.DataFrame:
    global _CACHE
    if _CACHE is None:
        if not CSV_PATH.exists():
            download_sp500()
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
        df = df.rename(columns={"Name": "ticker"})
        _CACHE = df
    return _CACHE


def available_tickers() -> list[str]:
    return sorted(_load_raw()["ticker"].unique().tolist())


def load_sp500(ticker: str) -> pd.DataFrame:
    """Single-ticker OHLCV DataFrame with DatetimeIndex (same schema as get_data)."""
    raw = _load_raw()
    sub = raw[raw["ticker"] == ticker].copy()
    if sub.empty:
        raise KeyError(f"{ticker!r} not in dataset. See available_tickers().")
    sub = sub.set_index("date").sort_index()
    return sub[["open", "high", "low", "close", "volume"]].dropna()


def get_panel(tickers: list[str] | None = None, field: str = "close") -> pd.DataFrame:
    """
    Wide panel: rows = dates, columns = tickers, values = chosen field.
    Used for cross-sectional strategies. Drops tickers with incomplete history.
    """
    raw = _load_raw()
    if tickers is not None:
        raw = raw[raw["ticker"].isin(tickers)]
    panel = raw.pivot(index="date", columns="ticker", values=field).sort_index()
    # Keep only tickers with full history (no gaps from listings/delistings)
    full = panel.columns[panel.notna().all()]
    return panel[full]


def liquid_universe(n: int = 100) -> list[str]:
    """Return the n most-traded tickers (by median dollar volume) with full history."""
    raw = _load_raw()
    raw = raw.assign(dollar_vol=raw["close"] * raw["volume"])
    panel = raw.pivot(index="date", columns="ticker", values="close")
    full = set(panel.columns[panel.notna().all()])
    med = (
        raw[raw["ticker"].isin(full)]
        .groupby("ticker")["dollar_vol"].median()
        .sort_values(ascending=False)
    )
    return med.head(n).index.tolist()
