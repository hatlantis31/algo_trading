"""
Cross-sectional (rank-based) equity factor strategies.

These are the canonical building blocks of professional systematic equity
trading. Instead of timing a single asset, they rank the WHOLE universe each
rebalance date and build a diversified portfolio of the best names.

Each factor exposes a `weights(prices, date)` method compatible with
PortfolioEngine.run(). Weights sum to 1 (long-only) or 0 (dollar-neutral
long-short, gross = 1).

References
----------
- Jegadeesh & Titman (1993): 12-1 month momentum
- Lehmann (1990), Jegadeesh (1990): short-term reversal
- Ang, Hodrick, Xing, Zhang (2006): low-volatility anomaly
- Grinold & Kahn, "Active Portfolio Management": z-score / IC weighting
"""

import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.std() == 0 or len(s) < 2:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std()


def _to_weights(signal: pd.Series, top_q: float, long_short: bool) -> pd.Series:
    """
    Convert a cross-sectional signal (higher = more attractive) into weights.
    long_short=False : long-only equal-weight top quantile, sums to 1.
    long_short=True  : long top quantile, short bottom quantile, dollar-neutral.
    """
    signal = signal.dropna()
    if signal.empty:
        return signal
    n = len(signal)
    k = max(1, int(n * top_q))
    ranked = signal.sort_values(ascending=False)

    weights = pd.Series(0.0, index=signal.index)
    longs = ranked.index[:k]
    weights[longs] = 1.0 / k

    if long_short:
        shorts = ranked.index[-k:]
        weights[shorts] = -1.0 / k
    return weights


class CrossSectionalMomentum:
    """
    12-1 momentum: rank by return over [lookback] days, skipping the most recent
    [skip] days (to avoid 1-month reversal). Long winners, optionally short losers.
    """
    name = "xs_momentum"

    def __init__(self, lookback: int = 252, skip: int = 21,
                 top_q: float = 0.2, long_short: bool = False):
        self.lookback = lookback
        self.skip = skip
        self.top_q = top_q
        self.long_short = long_short

    def weights(self, prices: pd.DataFrame, date) -> pd.Series:
        if len(prices) < self.lookback + self.skip:
            return pd.Series(dtype=float)
        p_now = prices.iloc[-1 - self.skip]
        p_then = prices.iloc[-1 - self.skip - self.lookback + 1]
        mom = p_now / p_then - 1
        return _to_weights(mom.dropna(), self.top_q, self.long_short)


class ShortTermReversal:
    """Short-term reversal: buy recent losers, sell recent winners (1-week)."""
    name = "xs_reversal"

    def __init__(self, lookback: int = 5, top_q: float = 0.2, long_short: bool = True):
        self.lookback = lookback
        self.top_q = top_q
        self.long_short = long_short

    def weights(self, prices: pd.DataFrame, date) -> pd.Series:
        if len(prices) < self.lookback + 1:
            return pd.Series(dtype=float)
        recent_ret = prices.iloc[-1] / prices.iloc[-1 - self.lookback] - 1
        # reversal: negative of recent return is the signal
        return _to_weights((-recent_ret).dropna(), self.top_q, self.long_short)


class LowVolatility:
    """Low-volatility anomaly: long the lowest-volatility names (long-only)."""
    name = "xs_lowvol"

    def __init__(self, lookback: int = 126, top_q: float = 0.2):
        self.lookback = lookback
        self.top_q = top_q

    def weights(self, prices: pd.DataFrame, date) -> pd.Series:
        if len(prices) < self.lookback + 1:
            return pd.Series(dtype=float)
        rets = prices.pct_change().iloc[-self.lookback:]
        vol = rets.std()
        # signal = -vol so lowest vol ranks highest
        return _to_weights((-vol).dropna(), self.top_q, long_short=False)


class MultiFactor:
    """
    Combine standardized factor signals (momentum + low-vol + reversal) by
    equal-weight z-score blending, then build a long-only top-quantile book.
    This is the simple version of Grinold-Kahn signal combination.
    """
    name = "xs_multifactor"

    def __init__(self, top_q: float = 0.2, long_short: bool = False,
                 mom_lb: int = 252, mom_skip: int = 21,
                 vol_lb: int = 126, rev_lb: int = 5,
                 weights_mix=(1.0, 0.5, 0.5)):
        self.top_q = top_q
        self.long_short = long_short
        self.mom_lb, self.mom_skip = mom_lb, mom_skip
        self.vol_lb, self.rev_lb = vol_lb, rev_lb
        self.w_mom, self.w_vol, self.w_rev = weights_mix

    def weights(self, prices: pd.DataFrame, date) -> pd.Series:
        need = max(self.mom_lb + self.mom_skip, self.vol_lb, self.rev_lb) + 1
        if len(prices) < need:
            return pd.Series(dtype=float)

        # Momentum (12-1)
        p_now = prices.iloc[-1 - self.mom_skip]
        p_then = prices.iloc[-1 - self.mom_skip - self.mom_lb + 1]
        mom = _zscore(p_now / p_then - 1)

        # Low-vol (negative vol → higher z = lower vol)
        vol = prices.pct_change().iloc[-self.vol_lb:].std()
        lowvol = _zscore(-vol)

        # Short-term reversal (negative recent return)
        rev = _zscore(-(prices.iloc[-1] / prices.iloc[-1 - self.rev_lb] - 1))

        combined = (self.w_mom * mom).add(self.w_vol * lowvol, fill_value=0) \
                                     .add(self.w_rev * rev, fill_value=0)
        return _to_weights(combined.dropna(), self.top_q, self.long_short)


class EqualWeightBenchmark:
    """Buy-and-hold equal-weight the whole universe (benchmark)."""
    name = "equal_weight"

    def weights(self, prices: pd.DataFrame, date) -> pd.Series:
        cols = prices.columns
        return pd.Series(1.0 / len(cols), index=cols)
