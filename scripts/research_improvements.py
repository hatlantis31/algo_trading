"""
research_improvements.py — systematic A/B testing of MarketBeatingStrategy variants.

Each variant tests ONE well-grounded idea from the literature:

  baseline       : current MarketBeatingStrategy (top 20%, EW within, monthly)
  rank_buffer    : hold winners until they fall out of top 2*top_q (turnover ↓;
                   Jegadeesh-Titman implementation practice)
  riskadj_mom    : momentum / realized vol ranking (Barroso & Santa-Clara 2015)
  quality_tilt   : blend PIT earnings_yield into the composite (momentum+quality)
  score_weighted : weight ∝ score rank within selection (more in best names)
  conc_10/15/30  : concentration sweep around top_q=0.20
  volscale       : Daniel-Moskowitz momentum-crash protection — scale exposure
                   by target_vol / trailing market vol (capped at 1)

All run with the same engine settings (monthly, 10 bps). Reports a ranked table
vs the equal-weight benchmark. N_TRIALS counted honestly for DSR downstream.

    python scripts/research_improvements.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FUND_PATH = DATA_DIR / "fundamentals_timeseries.parquet"
PRICE_PATH = DATA_DIR / "price_panel_daily.parquet"
VOLUME_PATH = DATA_DIR / "volume_panel_daily.parquet"
COST = 10


def metrics_row(returns: pd.Series, bm: pd.Series = None) -> dict:
    r = returns.dropna()
    ann = np.sqrt(252)
    eq = (1 + r).cumprod()
    out = {
        "ann_ret": r.mean() * 252,
        "sharpe": r.mean() / r.std() * ann,
        "max_dd": (eq / eq.cummax() - 1).min(),
        "final_eq": eq.iloc[-1],
    }
    if bm is not None:
        b = bm.reindex(r.index).dropna()
        ex = r.reindex(b.index) - b
        out["alpha"] = ex.mean() * 252
        out["ir"] = ex.mean() / ex.std() * ann if ex.std() > 0 else 0.0
    by_year = r.groupby(r.index.year).sum()
    out["worst_year"] = by_year.min()
    return out


def main():
    print("Loading data...")
    prices = pd.read_parquet(PRICE_PATH).astype(float)
    prices = prices.where(prices > 0)
    volume = pd.read_parquet(VOLUME_PATH).astype(float)
    enough = prices.columns[prices.notna().sum() >= 300]
    prices, volume = prices[enough], volume[enough]

    from core.fundamentals_ts import FundamentalsTimeSeries
    fund_ts = FundamentalsTimeSeries(FUND_PATH) if FUND_PATH.exists() else None

    from backtesting.portfolio_engine import PortfolioEngine
    from strategies.cross_sectional import EqualWeightBenchmark
    from strategies.long_only_factor import MarketBeatingStrategy

    eng = lambda: PortfolioEngine(rebalance="ME", cost_bps=COST)

    print("Running benchmark...")
    bm = eng().run(prices, EqualWeightBenchmark().weights).returns

    variants = {
        "baseline (top20, EW)": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0),
        "rank_buffer": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0, sell_rank_buffer=2.0),
        "riskadj_mom": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0, risk_adjust_momentum=True),
        "quality_tilt": MarketBeatingStrategy(
            volume_panel=volume, fundamentals_ts=fund_ts, regime_scale=1.0),
        "score_weighted": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0, weighting="score"),
        "conc_top10": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0, top_q=0.10),
        "conc_top15": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0, top_q=0.15),
        "conc_top30": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0, top_q=0.30),
        "volscale (DM crash prot.)": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0, vol_scale_target=0.20),
        "rank_buffer + riskadj": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0,
            sell_rank_buffer=2.0, risk_adjust_momentum=True),
        "rank_buffer + volscale": MarketBeatingStrategy(
            volume_panel=volume, regime_scale=1.0,
            sell_rank_buffer=2.0, vol_scale_target=0.20),
    }

    rows = {"EW benchmark": metrics_row(bm)}
    for name, strat in variants.items():
        print(f"Running {name}...")
        try:
            res = eng().run(prices, strat.weights)
            rows[name] = metrics_row(res.returns, bm)
            rows[name]["turnover"] = res.turnover
        except Exception as e:
            print(f"  FAILED: {e}")

    df = pd.DataFrame(rows).T
    df = df.sort_values("sharpe", ascending=False)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print("\n" + "=" * 100)
    print(f"{'variant':<28} {'annRet':>8} {'sharpe':>7} {'maxDD':>8} "
          f"{'alpha':>8} {'IR':>6} {'worstYr':>8} {'turnover':>8} {'finalEq':>8}")
    print("-" * 100)
    for name, r in df.iterrows():
        print(f"{name:<28} {r['ann_ret']:>+8.1%} {r['sharpe']:>+7.2f} "
              f"{r['max_dd']:>+8.1%} "
              f"{r.get('alpha', float('nan')):>+8.1%} {r.get('ir', float('nan')):>+6.2f} "
              f"{r['worst_year']:>+8.1%} "
              f"{r.get('turnover', float('nan')):>8.3f} {r['final_eq']:>8.2f}")


if __name__ == "__main__":
    main()
