"""
AlphaDecayMonitor — tracks live Information Coefficient and fires alerts
when the strategy's signal stops working.

Background (Lee 2025, arXiv:2512.11913; arXiv:2603.13252 "When Alpha Breaks")
-------------------------------------------------------------------------------
Factor alpha decays hyperbolically: α(t) = K / (1 + λ·t). Post-2015 crowding
accelerated decay. ML cross-sectional rankers show two distinct failure modes:
  - Position-level: individual signal quality degrades gradually
  - Strategy-level: the whole ranker becomes unreliable (regime break)

The best early-warning indicator for both is a collapsing rolling IC.
AQR/Man Group consensus: pause a strategy when rolling 60-day IC hits zero.

Usage
-----
    monitor = AlphaDecayMonitor(window=60, ic_halt_threshold=0.0)

    # At each rebalance date (live or backtest):
    monitor.update(signal_series, realised_fwd_returns)
    if monitor.should_halt():
        print("Alpha decay alert — pausing strategy")
    monitor.plot()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class AlphaDecayMonitor:
    def __init__(
        self,
        window: int = 60,                   # rolling window for IC (in observations)
        ic_halt_threshold: float = 0.0,     # halt when rolling IC drops below this
        ic_warn_threshold: float = 0.01,    # warn when below this
        min_obs_to_judge: int = 12,         # don't alert before enough data
    ):
        self.window = window
        self.ic_halt_threshold = ic_halt_threshold
        self.ic_warn_threshold = ic_warn_threshold
        self.min_obs_to_judge = min_obs_to_judge

        self._ic_series: list[float] = []
        self._dates: list = []
        self.alert_log: list[str] = []

    # ── Core update ──────────────────────────────────────────────────────────

    def update(
        self,
        signal: pd.Series,
        fwd_returns: pd.Series,
        date=None,
    ) -> float | None:
        """
        Compute cross-sectional IC for this rebalance date and update the monitor.

        signal      : cross-sectional factor scores (one value per stock)
        fwd_returns : realised returns over the forward period (same index)
        Returns     : IC for this date, or None if insufficient data.
        """
        pair = pd.concat([signal.rename("sig"), fwd_returns.rename("ret")], axis=1).dropna()
        if len(pair) < 5:
            return None

        ic, _ = stats.spearmanr(pair["sig"], pair["ret"])
        if np.isnan(ic):
            return None

        self._ic_series.append(float(ic))
        self._dates.append(date or len(self._ic_series))
        self._check_alerts()
        return float(ic)

    def rolling_ic(self) -> pd.Series:
        s = pd.Series(self._ic_series, index=self._dates)
        return s.rolling(self.window, min_periods=1).mean()

    def should_halt(self) -> bool:
        """Returns True if the rolling IC has collapsed below halt threshold."""
        ric = self.rolling_ic()
        if len(ric) < self.min_obs_to_judge:
            return False
        return float(ric.iloc[-1]) < self.ic_halt_threshold

    def should_warn(self) -> bool:
        ric = self.rolling_ic()
        if len(ric) < self.min_obs_to_judge:
            return False
        return float(ric.iloc[-1]) < self.ic_warn_threshold

    def current_ic(self) -> float | None:
        return self._ic_series[-1] if self._ic_series else None

    def current_rolling_ic(self) -> float | None:
        ric = self.rolling_ic()
        return float(ric.iloc[-1]) if len(ric) else None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _check_alerts(self):
        if len(self._ic_series) < self.min_obs_to_judge:
            return
        ric = self.rolling_ic().iloc[-1]
        date = self._dates[-1]
        if ric < self.ic_halt_threshold:
            msg = f"{date}  HALT: rolling IC {ric:.4f} < {self.ic_halt_threshold:.4f}"
            if not self.alert_log or self.alert_log[-1] != msg:
                self.alert_log.append(msg)
        elif ric < self.ic_warn_threshold:
            msg = f"{date}  WARN: rolling IC {ric:.4f} < {self.ic_warn_threshold:.4f}"
            if not self.alert_log or self.alert_log[-1] != msg:
                self.alert_log.append(msg)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        ric = self.rolling_ic()
        ics = np.array(self._ic_series)
        return {
            "n_observations":       len(ics),
            "ic_mean":              round(float(ics.mean()), 4) if len(ics) else None,
            "ic_std":               round(float(ics.std()),  4) if len(ics) else None,
            "ic_hit_rate":          round(float((ics > 0).mean()), 4) if len(ics) else None,
            "rolling_ic_current":   round(float(ric.iloc[-1]), 4) if len(ric) else None,
            "rolling_ic_min":       round(float(ric.min()), 4) if len(ric) else None,
            "should_halt":          self.should_halt(),
            "alert_count":          len(self.alert_log),
        }

    def plot(self, figsize=(14, 4)):
        if not self._ic_series:
            print("No IC data yet.")
            return
        ric = self.rolling_ic()
        raw = pd.Series(self._ic_series, index=self._dates)

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(raw.index, raw.values, alpha=0.3, color="steelblue", label="IC per period")
        ax.plot(ric.index, ric.values, color="steelblue", lw=2, label=f"Rolling {self.window}-obs IC")
        ax.axhline(self.ic_halt_threshold, ls="--", color="red",    lw=1, label=f"Halt ({self.ic_halt_threshold})")
        ax.axhline(self.ic_warn_threshold, ls="--", color="orange", lw=1, label=f"Warn ({self.ic_warn_threshold})")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title("Alpha Decay Monitor — rolling Information Coefficient")
        ax.set_ylabel("IC (Spearman rank correlation)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ── Hyperbolic alpha decay fit (Lee 2025) ────────────────────────────────

    @staticmethod
    def fit_decay_curve(ic_series: pd.Series) -> dict:
        """
        Fit α(t) = K / (1 + λ·t) to a series of ICs.
        Returns K (initial alpha), λ (decay rate), half-life in observations.
        From Lee (2025) arXiv:2512.11913.
        """
        from scipy.optimize import curve_fit

        def hyperbolic(t, K, lam):
            return K / (1 + lam * t)

        t = np.arange(len(ic_series))
        y = ic_series.values
        try:
            popt, _ = curve_fit(hyperbolic, t, y, p0=[y[0], 0.01], maxfev=5000)
            K, lam = popt
            half_life = (1 / lam - 1) if lam > 0 else np.inf
            return {"K_initial_ic": round(K, 4), "lambda_decay": round(lam, 6),
                    "half_life_obs": round(half_life, 1)}
        except Exception:
            return {"K_initial_ic": None, "lambda_decay": None, "half_life_obs": None}
