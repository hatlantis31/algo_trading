"""
Fundamental data loader — real S&P 500 valuation/quality ratios.

Source: the open `datasets/s-and-p-500-companies-financials` dataset on GitHub
(reachable from any locked-down environment — no API key, no registration).
It is a SINGLE cross-sectional snapshot (≈ Feb 2018), giving per-ticker:
Price/Earnings, Dividend Yield, EPS, Market Cap, EBITDA, Price/Sales, Price/Book,
Sector (GICS sub-industry).

From these we derive the classic value/quality factors:
    earnings_yield  = 1 / (P/E)          (E/P — Basu value)
    book_to_price   = 1 / (P/B)          (B/P — Fama-French HML)
    sales_yield     = 1 / (P/S)          (S/P — robust value)
    ebitda_yield    = EBITDA / MarketCap (cash-flow value / quality)
    dividend_yield  = as reported
    size            = -log(MarketCap)    (Banz small-cap premium; sign so high=small)

────────────────────────────────────────────────────────────────────────────────
HONEST LIMITATION (read this before backtesting):

This is ONE snapshot dated ≈ Feb 2018 — the SAME date as the END of the bundled
price panel (`sp500_5yr.csv`, 2013-02 → 2018-02). Using it across the whole
2013-2018 window is therefore LOOK-AHEAD biased: you'd be ranking stocks in 2013
by their 2018 valuations.

  • For a *current/live* cross-sectional ranking it is point-in-time-correct.
  • For *historical backtests* the bias is modest for SLOW-moving ratios
    (B/P, S/P rank persistence over a few years is high) and severe for
    fast-moving ones — never trust an earnings-surprise-style signal from a
    snapshot.

For a clean, point-in-time historical backtest you need time-series fundamentals.
This environment can only reach GitHub, so do that ingestion on YOUR machine with
`scripts/ingest_fundamentals_local.py` (SimFin free tier + yfinance — see that
file for the exact registration steps).
────────────────────────────────────────────────────────────────────────────────
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "sp500_fundamentals.csv"
URL = ("https://raw.githubusercontent.com/datasets/"
       "s-and-p-500-companies-financials/main/data/constituents-financials.csv")

# Approximate snapshot date (matches the END of the bundled price panel).
SNAPSHOT_DATE = pd.Timestamp("2018-02-07")

_CACHE: pd.DataFrame | None = None


def download_fundamentals(force: bool = False) -> Path:
    """Download the fundamentals snapshot to data/ (gitignored). ~70 KB, no key."""
    if CSV_PATH.exists() and not force:
        logger.info("Already present: %s", CSV_PATH)
        return CSV_PATH
    import urllib.request
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading S&P 500 fundamentals snapshot …")
    urllib.request.urlretrieve(URL, CSV_PATH)
    logger.info("Saved: %s", CSV_PATH)
    return CSV_PATH


def _safe_recip(s: pd.Series) -> pd.Series:
    """1/x with non-positive and zero denominators sent to NaN (drop, don't fake)."""
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(x > 0)
    return 1.0 / x


def load_fundamentals(refresh: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame indexed by ticker with raw + derived fundamental factors.

    Columns: earnings_yield, book_to_price, sales_yield, ebitda_yield,
             dividend_yield, size, sector, market_cap, pe, eps.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    if not CSV_PATH.exists() or refresh:
        download_fundamentals(force=refresh)

    raw = pd.read_csv(CSV_PATH)
    raw = raw.rename(columns={
        "Symbol": "ticker",
        "Sector": "sector",
        "Price/Earnings": "pe",
        "Earnings/Share": "eps",
        "Dividend Yield": "dividend_yield",
        "Market Cap": "market_cap",
        "EBITDA": "ebitda",
        "Price/Sales": "ps",
        "Price/Book": "pb",
    })
    raw = raw.set_index("ticker")

    out = pd.DataFrame(index=raw.index)
    out["earnings_yield"] = _safe_recip(raw["pe"])
    out["book_to_price"] = _safe_recip(raw["pb"])
    out["sales_yield"] = _safe_recip(raw["ps"])
    mcap = pd.to_numeric(raw["market_cap"], errors="coerce")
    out["ebitda_yield"] = pd.to_numeric(raw["ebitda"], errors="coerce") / mcap.where(mcap > 0)
    out["dividend_yield"] = pd.to_numeric(raw["dividend_yield"], errors="coerce")
    out["size"] = -np.log(mcap.where(mcap > 0))   # high score = small cap
    out["sector"] = raw["sector"]
    out["market_cap"] = mcap
    out["pe"] = pd.to_numeric(raw["pe"], errors="coerce")
    out["eps"] = pd.to_numeric(raw["eps"], errors="coerce")

    _CACHE = out
    return out


# Derived columns that are usable as cross-sectional alpha (higher = cheaper/better)
FUNDAMENTAL_FACTOR_COLS = [
    "earnings_yield", "book_to_price", "sales_yield",
    "ebitda_yield", "dividend_yield", "size",
]


def fundamental_factors(tickers: list[str] | None = None,
                        cols: list[str] | None = None,
                        refresh: bool = False) -> pd.DataFrame:
    """
    Wide factor frame (rows = tickers, cols = fundamental factors), aligned to a
    ticker list. RAW values (un-standardized) — standardize cross-sectionally at
    use time so it composes with the price-factor pipeline.
    """
    f = load_fundamentals(refresh=refresh)
    cols = cols or FUNDAMENTAL_FACTOR_COLS
    out = f[cols].copy()
    if tickers is not None:
        out = out.reindex(tickers)
    return out


def sector_map(tickers: list[str] | None = None, refresh: bool = False) -> pd.Series:
    """Ticker → GICS sub-industry, for sector-neutralization."""
    s = load_fundamentals(refresh=refresh)["sector"]
    return s.reindex(tickers) if tickers is not None else s


def available_tickers() -> list[str]:
    return sorted(load_fundamentals().index.tolist())
