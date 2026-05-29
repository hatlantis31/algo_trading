"""
Regime detector based on price vs N-day moving average.

Returns one of three regimes per bar:
  +1  = bull  (price > MA, trend up)
   0  = neutral (within threshold band)
  -1  = bear  (price < MA, trend down)

Can be used standalone or as a wrapper around any BaseStrategy.
"""

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy


def detect_regime(
    df: pd.DataFrame,
    ma_window: int = 200,
    band_pct: float = 0.01,
) -> pd.Series:
    """
    Vectorised regime series aligned to df.index.

    Parameters
    ----------
    ma_window : lookback for the trend MA (default 200 days)
    band_pct  : ±% neutral band around MA to avoid flip-flopping
                e.g. 0.01 → no signal if price is within 1% of MA
    """
    close = df["close"]
    ma = close.rolling(ma_window).mean()
    ratio = close / ma - 1          # +ve = above, -ve = below

    regime = pd.Series(0, index=df.index, dtype=int)
    regime[ratio >  band_pct] =  1
    regime[ratio < -band_pct] = -1
    return regime


class RegimeFilteredStrategy(BaseStrategy):
    """
    Wraps any BaseStrategy and gates its signals through a regime filter.

    Behaviour
    ---------
    bull  (+1) : pass through the inner strategy's signal unchanged
    bear  (-1) : go to cash (signal = 0), or optionally invert to short
    neutral(0) : go to cash (signal = 0)

    Parameters
    ----------
    inner_strategy : any BaseStrategy instance
    ma_window      : MA lookback for regime detection (default 200)
    band_pct       : neutral band (default 1%)
    allow_short    : if True, flip to -1 in bear regime instead of 0
    """
    name = "regime_filtered"
    param_grid = {}

    def __init__(
        self,
        inner_strategy: BaseStrategy,
        ma_window: int = 200,
        band_pct: float = 0.01,
        allow_short: bool = False,
    ):
        super().__init__()
        self.inner = inner_strategy
        self.ma_window = ma_window
        self.band_pct = band_pct
        self.allow_short = allow_short
        self.name = f"regime({self.inner.name})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raw = self.inner.generate_signals(df)
        regime = detect_regime(df, ma_window=self.ma_window, band_pct=self.band_pct)

        signal = pd.Series(0, index=df.index)
        bull_mask = regime == 1
        signal[bull_mask] = raw[bull_mask]

        if self.allow_short:
            bear_mask = regime == -1
            signal[bear_mask] = -1

        return signal

    def __repr__(self):
        return (
            f"RegimeFilteredStrategy(inner={self.inner}, "
            f"ma={self.ma_window}, band={self.band_pct:.1%}, "
            f"short={self.allow_short})"
        )
