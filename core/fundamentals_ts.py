"""
FundamentalsTimeSeries — point-in-time fundamentals with live-price recompute.

Wraps the parquet produced by scripts/ingest_fundamentals_local.py (+ optional
downsample). The frame has a MultiIndex (date, ticker) where each row holds the
most recently PUBLISHED statement figures as of that date — a quarterly
step-function sampled at month-end.

Two access patterns:

    asof(date)                    → static factor frame as of `date`
                                    (yields frozen at their last sample date)
    asof(date, prices=row)        → same, but valuation yields RECOMPUTED with
                                    the live price row, decoupling fundamental
                                    frequency (quarterly) from trade frequency
                                    (daily / weekly):
                                    earnings_yield = net_income / (price*shares)

Both are strictly point-in-time: only rows with sample date <= `date` are used,
and the underlying ingestion aligned figures by PUBLISH date, so nothing is
visible before the market saw it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATH = DATA_DIR / "fundamentals_timeseries.parquet"

# derived factor columns the strategy consumes (matches fundamentals_loader)
FACTOR_COLS = ["earnings_yield", "book_to_price", "sales_yield",
               "ebitda_yield", "dividend_yield", "size"]

# raw numerators needed for live recompute (newer ingestion runs include these)
RAW_COLS = ["net_income", "total_equity", "revenue",
            "operating_income", "dividends_paid", "shares"]


class FundamentalsTimeSeries:
    def __init__(self, path: str | Path = PARQUET_PATH, max_staleness_days: int = 200):
        """max_staleness_days: drop figures older than this at lookup time —
        a name that stopped reporting (delisting, acquisition) must not keep
        trading on fossil fundamentals."""
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.MultiIndex):
            df = df.set_index(["date", "ticker"])
        df = df.sort_index()
        self.df = df
        self.dates = df.index.get_level_values("date").unique().sort_values()
        self.max_staleness = pd.Timedelta(days=max_staleness_days)
        self.has_raw = all(c in df.columns for c in ("net_income", "shares"))

    def asof(self, date, prices: pd.Series = None) -> pd.DataFrame:
        """
        Factor frame (rows=tickers, cols=FACTOR_COLS) as of `date`.

        If `prices` (ticker → live close) is given and raw numerators are
        available, valuation yields are recomputed against the live market cap;
        otherwise the frozen sample-date yields are returned.
        """
        date = pd.Timestamp(date)
        valid = self.dates[self.dates <= date]
        if len(valid) == 0:
            return pd.DataFrame(columns=FACTOR_COLS)
        # take each ticker's latest row at or before `date`, then drop stale ones
        hist = self.df.loc[self.df.index.get_level_values("date") <= date]
        snap = hist.groupby(level="ticker").tail(1).reset_index(level="date")
        fresh = snap["date"] >= date - self.max_staleness
        snap = snap.loc[fresh].drop(columns="date")

        if prices is None or not self.has_raw:
            return snap.reindex(columns=FACTOR_COLS)

        px = prices.reindex(snap.index)
        mcap = (px * snap["shares"]).where(lambda m: m > 0)
        out = pd.DataFrame(index=snap.index)
        out["earnings_yield"] = snap["net_income"] / mcap
        out["book_to_price"]  = snap["total_equity"] / mcap
        out["sales_yield"]    = snap["revenue"] / mcap
        out["ebitda_yield"]   = snap["operating_income"] / mcap
        out["dividend_yield"] = snap["dividends_paid"].abs() / mcap
        out["size"]           = -np.log(mcap)
        # fall back to frozen yields where live recompute failed (missing price/shares)
        frozen = snap.reindex(columns=FACTOR_COLS)
        return out.where(out.notna(), frozen).replace([np.inf, -np.inf], np.nan)

    def coverage(self) -> pd.Series:
        """Tickers with data per sample date — quick health check."""
        return self.df.groupby(level="date").size()


def load_fundamentals_ts(path: str | Path = PARQUET_PATH) -> FundamentalsTimeSeries | None:
    """Convenience: return the time-series if the parquet exists, else None."""
    return FundamentalsTimeSeries(path) if Path(path).exists() else None
