"""
export_price_panel.py — export a daily close-price panel from the SimFin
share-prices file already downloaded by ingest_fundamentals_local.py.

Run LOCALLY after the ingestion (no new download needed):

    python scripts/export_price_panel.py

Writes two small parquets (committable — gitignore exceptions added):

    data/price_panel_daily.parquet    daily adjusted closes, top-N liquid names
    data/volume_panel_daily.parquet   matching dollar-volume panel

Together with data/fundamentals_timeseries.parquet this lets the FULL
HedgeFundStrategy pipeline (daily momentum/reversal/vol factors + point-in-time
fundamentals + weekly rebalancing) run on 2020-2025 — a window that includes
the 2022 bear market, with no survivorship bias (SimFin keeps delisted names).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SIMFIN_DIR = DATA_DIR / "simfin"
N_TICKERS = 600          # most-liquid names — keeps files small enough for git

OUT_PRICES = DATA_DIR / "price_panel_daily.parquet"
OUT_VOLUME = DATA_DIR / "volume_panel_daily.parquet"


def find_shareprices() -> Path:
    hits = list(SIMFIN_DIR.rglob("us-shareprices-daily*.csv"))
    if not hits:
        raise SystemExit(
            f"No us-shareprices-daily*.csv under {SIMFIN_DIR} — "
            "run scripts/ingest_fundamentals_local.py first.")
    return hits[0]


def main():
    path = find_shareprices()
    print(f"Reading {path} (large file, ~1-2 min) …")
    df = pd.read_csv(path, sep=";", usecols=["Ticker", "Date", "Close",
                                             "Adj. Close", "Volume"],
                     parse_dates=["Date"])
    df = df.rename(columns={"Ticker": "ticker", "Date": "date",
                            "Adj. Close": "adj_close",
                            "Close": "close", "Volume": "volume"})

    # rank by median dollar volume over the full window → liquid universe
    df["dollar_vol"] = df["close"] * df["volume"]
    liquid = (df.groupby("ticker")["dollar_vol"].median()
                .sort_values(ascending=False).head(N_TICKERS).index)
    sub = df[df["ticker"].isin(liquid)]

    prices = sub.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    volume = sub.pivot(index="date", columns="ticker", values="volume").sort_index()

    # keep names with at least one year of data (delisted names stay — that is
    # the survivorship-bias-free point of this dataset)
    enough = prices.columns[prices.notna().sum() >= 252]
    prices, volume = prices[enough], volume[enough]

    prices.astype(np.float32).to_parquet(OUT_PRICES)
    volume.astype(np.float32).to_parquet(OUT_VOLUME)
    for p in (OUT_PRICES, OUT_VOLUME):
        print(f"Wrote {p}  ({p.stat().st_size/1e6:.1f} MB)")
    print(f"Panel: {prices.shape[0]} days x {prices.shape[1]} tickers, "
          f"{prices.index.min().date()} -> {prices.index.max().date()}")
    print("\nNow commit and push:")
    print("  git add data/price_panel_daily.parquet data/volume_panel_daily.parquet")
    print('  git commit -m "Add daily price/volume panels (SimFin, survivorship-bias-free)"')
    print("  git push origin claude/awesome-shannon-jE7rR")


if __name__ == "__main__":
    main()
