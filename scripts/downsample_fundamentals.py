"""
downsample_fundamentals.py — shrink the daily fundamentals parquet to month-end.

The ingestion writes one row per (trading-day, ticker), but fundamentals only
change when a new statement is published (quarterly). Daily granularity is ~98%
redundant and bloats the file past GitHub's comfort zone.

This collapses to one row per (month-end, ticker) by taking the last published
figures in each month — point-in-time safe (the values are step-functions that
only change on publish dates) and exactly the frequency the strategy rebalances
at ("ME"). Result is small enough for plain git — no LFS needed.

    python scripts/downsample_fundamentals.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PATH = DATA_DIR / "fundamentals_timeseries.parquet"


def main():
    df = pd.read_parquet(PATH)
    before = len(df)

    # MultiIndex (date, ticker) -> flat
    flat = df.reset_index()
    flat["date"] = pd.to_datetime(flat["date"])
    flat = flat.sort_values("date")

    # last published values within each calendar month, per ticker
    flat["ym"] = flat["date"].dt.to_period("M")
    me = flat.groupby(["ticker", "ym"], as_index=False).last()
    me["date"] = me["ym"].dt.to_timestamp("M")        # stamp at month-end
    me = (me.drop(columns="ym")
            .set_index(["date", "ticker"])
            .sort_index()
            .dropna(how="all"))

    me.to_parquet(PATH)
    size_mb = PATH.stat().st_size / 1e6
    print(f"Downsampled {before:,} -> {len(me):,} rows  "
          f"({me.index.get_level_values('ticker').nunique():,} tickers, "
          f"{me.index.get_level_values('date').nunique()} months)")
    print(f"File: {PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
