"""
validate_market_beating.py — prove the long-only momentum factor strategy beats the market.

Runs three portfolios side by side:
  1. Equal-weight benchmark    (hold all stocks, monthly rebalance)
  2. V1 baseline               (raw momentum, top 20%, no buffer)
  3. V2 improved (primary)     (risk-adjusted momentum [Barroso-Santa-Clara] +
                                sell-rank buffer [buy top-k, sell at 2k])

The primary strategy to beat the market is #3 — the current defaults of
MarketBeatingStrategy.

    python scripts/validate_market_beating.py
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

# Honest trial count: 5 initial design iterations + 16 improvement variants
# tested in scripts/research_improvements.py
N_TRIALS = 21
COST = 10   # bps per unit turnover


def report(returns: pd.Series, label: str, benchmark_returns: pd.Series = None):
    from backtesting.cpcv import CombinatorialPurgedCV
    from backtesting.evaluation import deflated_sharpe_ratio

    r = returns.dropna()
    ann = np.sqrt(252)
    sharpe = r.mean() / r.std() * ann
    eq = (1 + r).cumprod()
    ann_ret = r.mean() * 252
    max_dd = (eq / eq.cummax() - 1).min()

    print(f"\n=== {label} ===")
    print(f"  periods       : {len(r)}  ({r.index[0].date()} -> {r.index[-1].date()})")
    print(f"  ann. return   : {ann_ret:+.1%}")
    print(f"  Sharpe        : {sharpe:+.2f}")
    print(f"  max drawdown  : {max_dd:+.1%}")
    print(f"  t-stat        : {r.mean() / r.sem():+.2f}")
    print(f"  final equity  : {eq.iloc[-1]:.2f}x")
    by_year = r.groupby(r.index.year).sum()
    print(f"  by year (%)   : {(by_year * 100).round(1).to_dict()}")

    if benchmark_returns is not None:
        b = benchmark_returns.reindex(r.index).dropna()
        excess = r.reindex(b.index) - b
        alpha_ann = excess.mean() * 252
        te = excess.std() * np.sqrt(252)
        ir = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        bm_eq = (1 + b).cumprod()
        beat_rate = (r.reindex(b.index) > b).mean()
        beta = np.cov(r.reindex(b.index).fillna(0), b.fillna(0))[0, 1] / max(b.var(), 1e-12)
        print(f"  --- vs benchmark ---")
        print(f"  ann. alpha    : {alpha_ann:+.1%}")
        print(f"  beta          : {beta:.2f}")
        print(f"  info ratio    : {ir:+.2f}")
        print(f"  tracking err  : {te:.1%}")
        print(f"  beat rate     : {beat_rate:.0%} of days")
        print(f"  benchmark eq  : {bm_eq.iloc[-1]:.2f}x  vs  strategy eq: {eq.iloc[-1]:.2f}x")

    cv = CombinatorialPurgedCV(6, 2, embargo=21)
    try:
        s = cv.evaluate(r, periods_per_year=252).summary()
        print(f"  CPCV          : mean={s.get('sharpe_mean')}  "
              f"std={s.get('sharpe_std')}  paths+={s.get('pct_paths_positive')}")
    except Exception:
        pass
    try:
        dsr = deflated_sharpe_ratio(r, n_trials=N_TRIALS)
        print(f"  Deflated SR   : {dsr['dsr']:.3f}  ({dsr['interpretation']})")
    except Exception:
        pass

    return sharpe, ann_ret, max_dd


def main():
    if not PRICE_PATH.exists():
        raise SystemExit(
            "data/price_panel_daily.parquet missing — run scripts/export_price_panel.py"
        )

    print("Loading data...")
    prices = pd.read_parquet(PRICE_PATH).astype(float)
    prices = prices.where(prices > 0)
    volume = pd.read_parquet(VOLUME_PATH).astype(float) if VOLUME_PATH.exists() else None

    enough = prices.columns[prices.notna().sum() >= 300]
    prices = prices[enough]
    if volume is not None:
        volume = volume[enough]

    print(f"Panel: {prices.shape[0]} days × {prices.shape[1]} tickers  "
          f"({prices.index.min().date()} → {prices.index.max().date()})")

    fund_ts = None
    if FUND_PATH.exists():
        from core.fundamentals_ts import FundamentalsTimeSeries
        fund_ts = FundamentalsTimeSeries(FUND_PATH)
        print("Fundamentals: loaded (point-in-time)")
    else:
        print("Fundamentals: not found, using price/volume only")

    from backtesting.portfolio_engine import PortfolioEngine
    from strategies.cross_sectional import EqualWeightBenchmark
    from strategies.long_only_factor import MarketBeatingStrategy

    # ── 1. Equal-weight benchmark ──────────────────────────────────────────
    print("\nRunning benchmark (equal-weight all stocks)...")
    bm = PortfolioEngine(rebalance="ME", cost_bps=COST).run(
        prices, EqualWeightBenchmark().weights, name="benchmark"
    )

    # ── 2. Previous baseline: raw momentum, no buffer ──────────────────────
    print("Running v1 baseline (raw momentum, no rank buffer)...")
    strat_v1 = MarketBeatingStrategy(
        volume_panel=volume,
        sell_rank_buffer=None, risk_adjust_momentum=False,
        name="MomentumV1",
    )
    res_v1 = PortfolioEngine(rebalance="ME", cost_bps=COST).run(
        prices, strat_v1.weights, name="momentum_v1"
    )

    # ── 3. Improved (primary): risk-adjusted momentum + sell-rank buffer ───
    print("Running improved strategy (risk-adj momentum + rank buffer)...")
    strat_v2 = MarketBeatingStrategy(volume_panel=volume, name="MarketBeaterV2")
    res_v2 = PortfolioEngine(rebalance="ME", cost_bps=COST).run(
        prices, strat_v2.weights, name="market_beater_v2"
    )

    bm_rets = bm.returns

    sharpe_bm,  ret_bm,  dd_bm  = report(bm.returns, "1. EQUAL-WEIGHT BENCHMARK")
    sharpe_v1,  ret_v1,  dd_v1  = report(
        res_v1.returns, "2. V1 BASELINE (raw momentum, no buffer)", bm_rets
    )
    sharpe_nf,  ret_nf,  dd_nf  = report(
        res_v2.returns,
        "3. IMPROVED (primary — risk-adj momentum + sell-rank buffer)", bm_rets
    )

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 62)
    rows = [
        ("Equal-weight benchmark",            ret_bm,  sharpe_bm, dd_bm),
        ("V1: raw momentum",                  ret_v1,  sharpe_v1, dd_v1),
        ("V2: risk-adj mom + rank buffer",    ret_nf,  sharpe_nf, dd_nf),
    ]
    print(f"{'Strategy':<36}  {'Ann Ret':>8}  {'Sharpe':>7}  {'Max DD':>8}")
    print("-" * 63)
    for name, r, s, d in rows:
        marker = " ◄" if name != "Equal-weight benchmark" and r > ret_bm else ""
        print(f"{name:<36}  {r:>+8.1%}  {s:>+7.2f}  {d:>+8.1%}{marker}")

    beats_return = ret_nf > ret_bm
    beats_sharpe = sharpe_nf > sharpe_bm
    beats_dd     = dd_nf > dd_bm    # less negative = better

    print(f"\nVERDICT (primary strategy vs benchmark):")
    markers = {True: "[PASS]", False: "[FAIL]"}
    print(f"  {markers[beats_return]} Higher annual return:  {ret_nf:+.1%}  vs {ret_bm:+.1%}")
    print(f"  {markers[beats_sharpe]} Higher Sharpe ratio:  {sharpe_nf:+.2f}  vs {sharpe_bm:+.2f}")
    print(f"  {markers[beats_dd]} Smaller max drawdown: {dd_nf:+.1%}  vs {dd_bm:+.1%}")

    wins = sum([beats_return, beats_sharpe, beats_dd])
    print(f"\n  Wins: {wins}/3 vs equal-weight benchmark")
    if wins == 3:
        print("  RESULT: strategy BEATS the market on ALL metrics.")
    elif wins >= 2:
        print("  RESULT: strategy BEATS the market on most metrics.")
    else:
        print("  RESULT: strategy does NOT beat the market on most metrics.")


if __name__ == "__main__":
    main()
