"""
validate_full_pipeline.py — the final, one-command strategy validation.

Two modes, selected automatically:

  FULL (daily)    if data/price_panel_daily.parquet exists (scripts/
                  export_price_panel.py): HedgeFundStrategy with daily
                  momentum/reversal/vol factors + point-in-time fundamentals
                  (live-price recompute) + weekly rebalancing + RiskManager
                  circuit breakers, evaluated with CPCV + Deflated Sharpe.

  MONTHLY (proxy) otherwise: fundamentals-only IC-weighted long-short on the
                  top-500 by market cap, value-spread conditioned exposure
                  (Asness 2000), net of costs — using market-cap-change proxy
                  returns. This reproduces the validated result:
                  Sharpe ~+0.5 net, +15.6% in the 2022 bear, market-neutral.

    python scripts/validate_full_pipeline.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FUND_PATH = DATA_DIR / "fundamentals_timeseries.parquet"
PRICE_PATH = DATA_DIR / "price_panel_daily.parquet"
VOLUME_PATH = DATA_DIR / "volume_panel_daily.parquet"

N_TRIALS = 12          # research trials run against this dataset (keep honest!)
COST = 0.0010          # 10 bps per unit turnover


def zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return s * 0
    s = s.clip(s.mean() - 3 * s.std(), s.mean() + 3 * s.std())
    return (s - s.mean()) / s.std()


def report(returns: pd.Series, periods_per_year: int, label: str):
    from backtesting.cpcv import CombinatorialPurgedCV
    from backtesting.evaluation import deflated_sharpe_ratio

    r = returns.dropna()
    ann = np.sqrt(periods_per_year)
    sharpe = r.mean() / r.std() * ann
    eq = (1 + r).cumprod()
    print(f"\n=== {label} ===")
    print(f"  periods            : {len(r)}  ({r.index[0].date()} -> {r.index[-1].date()})")
    print(f"  ann. return        : {r.mean() * periods_per_year:+.1%}")
    print(f"  Sharpe (net)       : {sharpe:+.2f}")
    print(f"  max drawdown       : {(eq / eq.cummax() - 1).min():+.1%}")
    print(f"  t-stat             : {r.mean() / r.sem():+.2f}")
    print(f"  by year (%)        : "
          f"{(r.groupby(r.index.year).sum() * 100).round(1).to_dict()}")
    cv = CombinatorialPurgedCV(6, 2, embargo=1 if periods_per_year <= 12 else 21)
    s = cv.evaluate(r, periods_per_year=periods_per_year).summary()
    print(f"  CPCV               : mean={s.get('sharpe_mean')}  "
          f"std={s.get('sharpe_std')}  paths+={s.get('pct_paths_positive')}")
    dsr = deflated_sharpe_ratio(r, n_trials=N_TRIALS)
    print(f"  Deflated Sharpe    : {dsr['dsr']:.3f}  ({dsr['interpretation']})")


def monthly_validation():
    """Fundamentals-only, value-spread-conditioned — runs on the committed parquet."""
    df = pd.read_parquet(FUND_PATH).reset_index()
    df["mcap"] = np.exp(-df["size"])
    dates = sorted(df["date"].unique())
    factors = ["earnings_yield", "book_to_price", "sales_yield",
               "ebitda_yield", "dividend_yield", "size"]
    mcap = df.pivot(index="date", columns="ticker", values="mcap")
    fwd = (mcap.shift(-1) / mcap - 1).clip(-0.95, 3.0)

    snaps, ic_hist, spread_hist = {}, {f: {} for f in factors}, {}
    for d in dates[:-1]:
        snap = df[df["date"] == d].set_index("ticker")
        uni = snap.dropna(subset=["earnings_yield"]).nlargest(500, "mcap")
        Z = pd.DataFrame({f: zscore(uni[f]) for f in factors})
        snaps[d] = Z
        ey = uni["earnings_yield"].dropna().sort_values(ascending=False)
        k = int(len(ey) * 0.2)
        spread_hist[d] = ey.iloc[:k].median() - ey.iloc[-k:].median()
        fr = fwd.loc[d].reindex(Z.index)
        for f in factors:
            pair = pd.concat([Z[f], fr], axis=1).dropna()
            if len(pair) > 50:
                ic, _ = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                ic_hist[f][d] = ic

    ic_df = pd.DataFrame(ic_hist)
    spread = pd.Series(spread_hist)
    rets, prev_w = {}, pd.Series(dtype=float)
    for d in dates[:-1]:
        Z = snaps[d]
        fr = fwd.loc[d].reindex(Z.index)
        past = ic_df.loc[ic_df.index < d].tail(12)
        if len(past) < 6:
            continue
        alpha = (Z * past.mean()).sum(axis=1).dropna()
        k = int(len(alpha) * 0.2)
        srt = alpha.sort_values(ascending=False)
        w = pd.Series(0.0, index=alpha.index)
        w[srt.index[:k]] = 0.5 / k
        w[srt.index[-k:]] = -0.5 / k
        past_spread = spread.loc[spread.index < d]
        if len(past_spread) >= 6:           # Asness value-spread timing, 0.5x-1.5x
            w = w * (0.5 + (past_spread < spread[d]).mean())
        turn = w.subtract(prev_w, fill_value=0.0).abs().sum()
        rets[d] = (w * fr.reindex(w.index)).sum() - turn * COST
        prev_w = w

    r = pd.Series(rets)
    r.index = pd.to_datetime(r.index)
    report(r, 12, "MONTHLY VALIDATION (fundamentals-only, mcap-proxy returns)")
    print("\n  NOTE: momentum/reversal need the daily panel — run "
          "scripts/export_price_panel.py and re-run this script.")


def full_validation():
    """Full daily pipeline: price factors + PIT fundamentals, weekly rebalance."""
    from core.fundamentals_ts import FundamentalsTimeSeries
    from core.risk_manager import RiskManager
    from strategies.hedge_fund_strategy import HedgeFundStrategy
    from backtesting.portfolio_engine import PortfolioEngine

    # non-positive prices (e.g. SIVB recorded at 0 after its 2023 collapse)
    # would create infinite returns — treat as missing
    prices = pd.read_parquet(PRICE_PATH).astype(float)
    prices = prices.where(prices > 0)
    volume = pd.read_parquet(VOLUME_PATH).astype(float)
    ts = FundamentalsTimeSeries(FUND_PATH)

    # liquid names with enough history for 12-1 momentum
    enough = prices.columns[prices.notna().sum() >= 300]
    prices, volume = prices[enough], volume[enough]
    print(f"Panel: {prices.shape[0]} days x {prices.shape[1]} tickers, "
          f"{prices.index.min().date()} -> {prices.index.max().date()}")

    # VALIDATED configuration (see README / notebook 11): EQUAL-weight factor
    # blend, MONTHLY rebalance. Empirically on 2020-2025:
    #   equal + monthly:        Sharpe +0.50, maxDD -9.7%, 87% CPCV paths > 0
    #   ic_weighted + monthly:  +0.09  (16 factors x 12 IC samples = noise;
    #                           DeMiguel 2009: 1/N beats estimated weights)
    #   anything weekly:        negative — 4x the costs, no weekly-scale signal
    hf = HedgeFundStrategy(volume_panel=volume, fundamentals_ts=ts,
                           alpha_combination="equal", construction="decile",
                           gross_leverage=1.0)
    eng = PortfolioEngine("ME", cost_bps=10, risk_manager=RiskManager())
    res = eng.run(prices, hf.weights)
    report(res.returns, 252, "FULL PIPELINE (daily factors + PIT fundamentals, monthly)")
    if getattr(res, "halt_log", None):
        print(f"  risk halts         : {len(res.halt_log)}")


if __name__ == "__main__":
    if not FUND_PATH.exists():
        raise SystemExit("data/fundamentals_timeseries.parquet missing — "
                         "run scripts/ingest_fundamentals_local.py first.")
    if PRICE_PATH.exists() and VOLUME_PATH.exists():
        full_validation()
    else:
        monthly_validation()
