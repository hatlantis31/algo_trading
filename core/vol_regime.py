"""
Volatility-regime detector — 2-state (calm / turbulent).

Better than the 200-day MA for regime detection (Man Group; research shows it
leads the MA signal by 2-4 weeks). Uses a simple but robust approach:

  1. Compute rolling realised volatility (e.g. 21-day)
  2. Compute a long-run volatility baseline (e.g. 252-day)
  3. Regime = TURBULENT when short vol > threshold × long vol
  4. Add hysteresis (separate entry/exit thresholds) to prevent flip-flopping

Reference: Man Group "Volatility Regimes and the Systematic Investor" (2022);
AQR Alternative Thinking "Key Design Choices in Long/Short Equity" (2023).

Usage
-----
    detector = VolatilityRegimeDetector()
    regime = detector.detect(df_close_prices)   # Series: 'calm' or 'turbulent'

    # In strategy:
    filtered = RegimeFilteredStrategy(
        inner=MultiFactor(),
        regime_func=detector.detect,
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CALM      = 1
TURBULENT = -1


def detect_vol_regime(
    prices: pd.Series | pd.DataFrame,
    short_window: int = 21,
    long_window: int = 252,
    entry_threshold: float = 1.25,   # turbulent when short_vol > 1.25 × long_vol
    exit_threshold: float = 1.10,    # resume calm when short_vol < 1.10 × long_vol
) -> pd.Series:
    """
    Returns a Series of {1=calm, -1=turbulent} aligned to prices.index.
    Uses hysteresis to avoid rapid switching.

    Parameters
    ----------
    prices           : close-price Series or DataFrame (uses mean vol across cols)
    short_window     : realised vol lookback for current regime signal (21 days)
    long_window      : baseline vol lookback (252 days)
    entry_threshold  : enter turbulent when ratio > this
    exit_threshold   : exit turbulent (resume calm) when ratio < this
    """
    if isinstance(prices, pd.DataFrame):
        rets = prices.pct_change().mean(axis=1)
    else:
        rets = prices.pct_change()

    short_vol = rets.rolling(short_window).std() * np.sqrt(252)
    long_vol  = rets.rolling(long_window).std()  * np.sqrt(252)

    ratio = (short_vol / long_vol.replace(0, np.nan)).fillna(1.0)

    # Apply hysteresis: state is sticky (requires threshold crossing to change)
    state = pd.Series(CALM, index=prices.index, dtype=int)
    current = CALM
    for i, (idx, r) in enumerate(ratio.items()):
        if current == CALM and r > entry_threshold:
            current = TURBULENT
        elif current == TURBULENT and r < exit_threshold:
            current = CALM
        state.iloc[i] = current

    return state


class VolatilityRegimeDetector:
    """Stateful wrapper around detect_vol_regime; also tracks regime history."""

    def __init__(
        self,
        short_window: int = 21,
        long_window: int = 252,
        entry_threshold: float = 1.25,
        exit_threshold: float = 1.10,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def detect(self, prices) -> pd.Series:
        return detect_vol_regime(
            prices,
            short_window=self.short_window,
            long_window=self.long_window,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
        )

    def regime_stats(self, prices) -> dict:
        regime = self.detect(prices)
        n = len(regime)
        n_turb = (regime == TURBULENT).sum()
        n_calm = (regime == CALM).sum()

        # Count transitions
        transitions = (regime != regime.shift()).sum() - 1
        avg_turb_len = n_turb / max(transitions / 2, 1)

        return {
            "pct_calm":      round(n_calm / n, 3),
            "pct_turbulent": round(n_turb / n, 3),
            "n_turbulent_episodes": int(transitions // 2),
            "avg_turbulent_duration_days": round(float(avg_turb_len), 1),
        }

    def plot(self, prices, figsize=(14, 5)):
        """Price chart with turbulent periods shaded in red."""
        regime = self.detect(prices)
        if isinstance(prices, pd.DataFrame):
            close = prices.mean(axis=1)
        else:
            close = prices

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(close.index, close.values, lw=1, color="grey", label="Price / index")

        turb = regime == TURBULENT
        for i in range(len(regime)):
            if turb.iloc[i]:
                ax.axvspan(regime.index[i], regime.index[min(i+1, len(regime)-1)],
                           alpha=0.15, color="red")
        from matplotlib.patches import Patch
        ax.legend(handles=[
            plt.Line2D([0], [0], color="grey", label="Price"),
            Patch(facecolor="red", alpha=0.3, label="Turbulent regime"),
        ])
        ax.set_title("Volatility regime detection (21d vs 252d realised vol)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return self.regime_stats(prices)
