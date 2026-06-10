"""
ingest_fundamentals_local.py — build POINT-IN-TIME fundamentals on YOUR machine.

Run this LOCALLY (not in the sandbox). The remote sandbox can only reach GitHub,
so the in-repo `core.fundamentals_loader` can only get a single 2018 snapshot.
This script pulls a proper time-series of fundamentals so historical backtests
are free of look-ahead bias.

Output: data/fundamentals_timeseries.parquet  (gitignored), with a MultiIndex
(date, ticker) and the same derived columns the model expects
(earnings_yield, book_to_price, sales_yield, ebitda_yield, dividend_yield, size).
Load it and pass per-rebalance-date slices into HedgeFundStrategy(fundamentals=…).

══════════════════════════════════════════════════════════════════════════════
TWO FREE SOURCES — pick one (or run both and merge). Registration steps:

  OPTION A — SimFin (best: true point-in-time statements, includes delisted names)
  ───────────────────────────────────────────────────────────────────────────
    1. Create a free account:  https://app.simfin.com/login/register
    2. Copy your API key:      https://app.simfin.com/api/v2/documentation/
                               (free tier covers US fundamentals for research)
    3. pip install simfin
    4. export SIMFIN_API_KEY=xxxxxxxx     (or paste below)
    Cost: €0. Free tier is rate-limited but fine for a one-time bulk pull.

  OPTION B — yfinance (quick, no key, but only ~4 yrs of quarterly statements
                       and survivorship-biased — current tickers only)
  ───────────────────────────────────────────────────────────────────────────
    1. pip install yfinance
    2. no registration required.
    Cost: €0.

  OPTION C — paid, if you want decades + delisted (your "<€10" budget):
  ───────────────────────────────────────────────────────────────────────────
    • SimFin Pro+ is ~$23/mo — above budget; the FREE tier is what to use.
    • Sharadar SF1 (via Nasdaq Data Link) is the gold standard but ~$50/mo.
    • Verdict: at a <€10 budget, the FREE SimFin tier (Option A) is the move.
      There is no reputable sub-€10 point-in-time fundamentals feed worth buying.
══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "fundamentals_timeseries.parquet"


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column if present, else NaN Series (never crashes on missing cols)."""
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw statement fields → the model's standard factor columns."""
    out = pd.DataFrame(index=df.index)
    mcap = df["market_cap"].where(df["market_cap"] > 0)
    out["earnings_yield"] = _safe_col(df, "net_income") / mcap
    out["book_to_price"]  = _safe_col(df, "total_equity") / mcap
    out["sales_yield"]    = _safe_col(df, "revenue") / mcap
    # operating_income is EBIT — good proxy for EBITDA, always available in SimFin
    out["ebitda_yield"]   = _safe_col(df, "operating_income") / mcap
    out["dividend_yield"] = _safe_col(df, "dividends_paid").abs() / mcap
    out["size"]           = -np.log(mcap)
    return out


def from_simfin(api_key: str | None = None) -> pd.DataFrame:
    """Point-in-time US fundamentals via SimFin (free tier)."""
    import simfin as sf
    sf.set_api_key(api_key or os.environ.get("SIMFIN_API_KEY", "free"))
    sf.set_data_dir(str(DATA_DIR / "simfin"))

    income  = sf.load_income(variant="quarterly",  market="us")
    balance = sf.load_balance(variant="quarterly", market="us")
    cashflow = sf.load_cashflow(variant="quarterly", market="us")
    shares  = sf.load_shareprices(variant="daily", market="us")

    def prep(df, col_map):
        """Reset index, rename columns, keep only what we need."""
        d = df.reset_index()
        d = d.rename(columns=col_map)
        keep = ["date", "ticker"] + [v for v in col_map.values() if v not in ("date", "ticker")]
        return d[[c for c in keep if c in d.columns]].sort_values("date")

    inc = prep(income, {
        "Ticker": "ticker", "Publish Date": "date",
        "Net Income": "net_income",
        "Revenue": "revenue",
        "Operating Income (Loss)": "operating_income",
    })
    bal = prep(balance, {
        "Ticker": "ticker", "Publish Date": "date",
        "Total Equity": "total_equity",
    })
    cf = prep(cashflow, {
        "Ticker": "ticker", "Publish Date": "date",
        "Dividends Paid": "dividends_paid",
    })
    px = (shares.reset_index()
          .rename(columns={"Ticker": "ticker", "Date": "date",
                           "Close": "close", "Shares Outstanding": "shares"})
          .sort_values("date"))
    px["market_cap"] = px["close"] * px["shares"]

    # merge_asof: for each price-bar date, pull the most-recently PUBLISHED
    # fundamental figures — this is point-in-time correct (no look-ahead).
    fund = pd.merge_asof(px, inc, on="date", by="ticker")
    fund = pd.merge_asof(fund.sort_values("date"), bal, on="date", by="ticker")
    fund = pd.merge_asof(fund.sort_values("date"), cf,  on="date", by="ticker")

    fund = fund.set_index(["date", "ticker"])
    return _derive(fund)


def from_yfinance(tickers: list[str]) -> pd.DataFrame:
    """Quick fundamentals via yfinance (current tickers only; ~4 yrs quarterly)."""
    import yfinance as yf
    frames = []
    for t in tickers:
        tk = yf.Ticker(t)
        try:
            fin = tk.quarterly_financials.T
            bs = tk.quarterly_balance_sheet.T
            info = tk.fast_info
            mcap = getattr(info, "market_cap", np.nan)
        except Exception:
            continue
        if fin.empty:
            continue
        rows = pd.DataFrame(index=fin.index)
        rows["net_income"] = fin.get("Net Income")
        rows["revenue"] = fin.get("Total Revenue")
        rows["total_equity"] = bs.get("Stockholders Equity") if not bs.empty else np.nan
        rows["market_cap"] = mcap
        rows["ticker"] = t
        rows = rows.reset_index().rename(columns={"index": "date"})
        frames.append(rows)
    if not frames:
        raise RuntimeError("yfinance returned nothing — check tickers / network.")
    fund = pd.concat(frames).set_index(["date", "ticker"])
    return _derive(fund)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = os.environ.get("FUND_SOURCE", "simfin").lower()

    if source == "simfin":
        df = from_simfin()
    elif source == "yfinance":
        # edit this universe to match your price panel
        from core.sp500_loader import liquid_universe
        df = from_yfinance(liquid_universe(100))
    else:
        raise SystemExit(f"Unknown FUND_SOURCE={source!r} (use 'simfin' or 'yfinance').")

    df = df.replace([np.inf, -np.inf], np.nan).sort_index()
    df.to_parquet(OUT_PATH)
    print(f"Wrote {len(df):,} rows → {OUT_PATH}")
    print("Load with:  pd.read_parquet('data/fundamentals_timeseries.parquet')")
    print("Then per rebalance date d:  fundamentals=df.xs(d, level='date')  (use asof for gaps)")


if __name__ == "__main__":
    main()
