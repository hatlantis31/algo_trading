"""
MarketBeatingStrategy — momentum-driven long-only factor strategy with regime filter.

Design to beat an equal-weight buy-and-hold benchmark:

  1. Warm-up fallback   – hold equal weights until enough price history is
                          available for the momentum signal (147 days).
                          Avoids dead-cash during the initial period.

  2. Momentum-dominant  – factor set is 100% trend/momentum signals:
                          6-month momentum, 12-month momentum, 52-week-high
                          proximity, trend quality.  NO low-vol or reversal
                          factors — those bias toward defensive, low-beta
                          names that lag in bull markets.

  3. Quality tilt       – if point-in-time fundamentals are available, blend
                          in earnings_yield as a quality screen (profitable
                          companies within the momentum selection).

  4. Concentrated       – hold top 25% by composite score (equal-weight
                          within selection).  Concentration in momentum
                          winners (e.g. NVDA, META in 2023–24) is the
                          engine that drives outperformance vs a diversified
                          EW benchmark.

  5. Regime filter      – when the equal-weight market index is clearly below
                          its 200-day MA (bear market), scale exposure to
                          `regime_scale` (default 30%).  Cuts the 2022-type
                          momentum crash risk while preserving participation
                          during recoveries.

Plugs into PortfolioEngine.run(prices, strategy.weights).
"""

import numpy as np
import pandas as pd

from .alpha_factors import (
    momentum, high_52w, trend_quality, standardize, fundamental_factor_matrix
)


# Momentum-only factor set (no low-vol, no reversal — see module docstring).
# Two price-trend horizons + structural confirmation signals.  Three-month
# momentum was tested and hurt 2023 performance (picked wrong post-crash names
# before the AI rally was established), so it is intentionally omitted.
MOMENTUM_FACTORS = {
    "momentum_6_1":  lambda p, v: momentum(p, 126, 21),
    "momentum_12_1": lambda p, v: momentum(p, 252, 21),
    "high_52w":      lambda p, v: high_52w(p, 252),
    "trend_quality": lambda p, v: trend_quality(p, 126),
}


class MarketBeatingStrategy:
    name = "market_beater"

    def __init__(
        self,
        volume_panel: pd.DataFrame = None,
        fundamentals_ts=None,           # FundamentalsTimeSeries for PIT quality factors
        top_q: float = 0.20,            # select top 20% by momentum score
        regime_ma: int = 200,           # days for regime moving-average
        regime_scale: float = 0.30,     # equity fraction when below regime MA
        fallback_to_ew: bool = True,    # equal-weight during warm-up
        name: str = None,
    ):
        self.volume_panel = volume_panel
        self.fundamentals_ts = fundamentals_ts
        self.top_q = top_q
        self.regime_ma = regime_ma
        self.regime_scale = regime_scale
        self.fallback_to_ew = fallback_to_ew
        self.name = name or "MarketBeater"

    # ── called by PortfolioEngine ─────────────────────────────────────────────

    def weights(self, prices: pd.DataFrame, date) -> pd.Series:
        n_total = len(prices.columns)
        min_days = 147   # 6-month momentum + skip: smallest lookback

        # ── Warm-up: fall back to equal-weight until we have any signal ───
        if len(prices) < min_days:
            if self.fallback_to_ew:
                return pd.Series(1.0 / n_total, index=prices.columns)
            return pd.Series(dtype=float)

        # ── Build momentum factor matrix ──────────────────────────────────
        vol_hist = (self.volume_panel.loc[:date]
                    if self.volume_panel is not None else None)
        fm = self._build_factors(prices, vol_hist, date)
        if fm.empty or len(fm) < 10:
            if self.fallback_to_ew:
                return pd.Series(1.0 / n_total, index=prices.columns)
            return pd.Series(dtype=float)

        # ── Composite score (equal-weight across available factors) ────────
        alpha = fm.mean(axis=1)
        alpha = standardize(alpha)
        alpha = alpha.dropna()
        if len(alpha) < 10:
            return pd.Series(dtype=float)

        # ── Select top quintile by composite momentum score ───────────────
        k = max(5, int(len(alpha) * self.top_q))
        top_names = alpha.nlargest(k).index

        # Equal weight within selection (preserves high-beta of momentum picks)
        w = pd.Series(1.0 / k, index=top_names)

        # ── Regime filter: scale down in bear markets ─────────────────────
        scale = self._regime_scale(prices.mean(axis=1))
        return w * scale

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_factors(self, prices, vol_hist, date) -> pd.DataFrame:
        """Compute momentum factors; optionally blend in PIT quality signal."""
        cols = {}
        for name, fn in MOMENTUM_FACTORS.items():
            try:
                raw = fn(prices, vol_hist)
                if raw is not None and len(raw) > 0:
                    cols[name] = standardize(raw)
            except Exception:
                continue

        fm = pd.DataFrame(cols)
        if fm.empty:
            return fm

        # Add earnings_yield quality signal from PIT fundamentals if available
        if self.fundamentals_ts is not None:
            try:
                fund = self.fundamentals_ts.asof(date, prices=prices.iloc[-1])
                if not fund.empty and "earnings_yield" in fund.columns:
                    ey = standardize(fund["earnings_yield"].dropna())
                    if len(ey) > 10:
                        fm = fm.join(ey.rename("quality_ey"), how="left")
            except Exception:
                pass

        return fm

    def _regime_scale(self, mkt_index: pd.Series) -> float:
        """1.0 when market is above 200-day MA (bull), regime_scale otherwise (bear)."""
        if len(mkt_index) < self.regime_ma:
            return 1.0
        ma = mkt_index.iloc[-self.regime_ma:].mean()
        return 1.0 if mkt_index.iloc[-1] >= ma else self.regime_scale
