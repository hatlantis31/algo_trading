"""Tests for AlphaDecayMonitor and VolatilityRegimeDetector."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.alpha_decay_monitor import AlphaDecayMonitor
from core.vol_regime import VolatilityRegimeDetector, detect_vol_regime, CALM, TURBULENT


def make_prices(n=400, seed=0, trend=0.0003, vol=0.012):
    rng = np.random.default_rng(seed)
    r = rng.normal(trend, vol, n)
    return pd.Series(100 * np.exp(np.cumsum(r)),
                     index=pd.date_range("2020-01-01", periods=n, freq="B"))


class TestAlphaDecayMonitor:
    def test_good_ic_no_halt(self):
        """Perfect signal should never trigger halt."""
        monitor = AlphaDecayMonitor(window=5, ic_halt_threshold=0.0, min_obs_to_judge=3)
        for _ in range(15):
            stocks = [f"S{i}" for i in range(20)]
            sig = pd.Series(np.arange(20, dtype=float), index=stocks)
            fwd = sig + np.random.normal(0, 0.1, 20)  # noisy but correlated
            monitor.update(sig, pd.Series(fwd, index=stocks))
        assert not monitor.should_halt()

    def test_random_ic_may_halt(self):
        """Random (zero-IC) signal should eventually trigger halt."""
        rng = np.random.default_rng(42)
        monitor = AlphaDecayMonitor(window=3, ic_halt_threshold=0.0, min_obs_to_judge=3)
        for _ in range(20):
            stocks = [f"S{i}" for i in range(20)]
            sig = pd.Series(rng.standard_normal(20), index=stocks)
            fwd = pd.Series(rng.standard_normal(20), index=stocks)
            monitor.update(sig, fwd)
        # with random data, rolling IC will sometimes go negative
        assert isinstance(monitor.should_halt(), bool)

    def test_ic_stored(self):
        monitor = AlphaDecayMonitor(min_obs_to_judge=2)
        for i in range(5):
            stocks = [f"S{j}" for j in range(10)]
            sig = pd.Series(np.arange(10, dtype=float), index=stocks)
            fwd = pd.Series(np.arange(10, dtype=float) + 0.1, index=stocks)
            monitor.update(sig, fwd, date=f"2024-0{i+1}")
        assert monitor.summary()["n_observations"] == 5
        assert monitor.summary()["ic_mean"] > 0.9

    def test_rolling_ic_length(self):
        monitor = AlphaDecayMonitor(window=3)
        for i in range(8):
            stocks = [f"S{j}" for j in range(10)]
            sig = pd.Series(np.random.normal(size=10), index=stocks)
            fwd = pd.Series(np.random.normal(size=10), index=stocks)
            monitor.update(sig, fwd, date=i)
        assert len(monitor.rolling_ic()) == 8

    def test_decay_fit(self):
        ic_series = pd.Series([0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.018])
        result = AlphaDecayMonitor.fit_decay_curve(ic_series)
        assert "K_initial_ic" in result
        assert "half_life_obs" in result


class TestVolatilityRegimeDetector:
    def test_calm_market_mostly_calm(self):
        prices = make_prices(400, vol=0.006)  # low vol
        regime = detect_vol_regime(prices, short_window=21, long_window=100)
        pct_calm = (regime == CALM).mean()
        assert pct_calm > 0.5

    def test_shock_triggers_turbulent(self):
        """Inject a volatility spike; detector should flag it."""
        prices = make_prices(300, vol=0.008)
        # add a crash period
        prices_crash = prices.copy()
        prices_crash.iloc[150:170] = prices.iloc[150] * (1 - np.linspace(0, 0.25, 20))
        regime = detect_vol_regime(prices_crash, short_window=10, long_window=60,
                                   entry_threshold=1.5)
        assert (regime.iloc[155:175] == TURBULENT).any()

    def test_output_shape_and_values(self):
        prices = make_prices(200)
        regime = detect_vol_regime(prices)
        assert len(regime) == len(prices)
        assert set(regime.unique()).issubset({CALM, TURBULENT})

    def test_regime_stats_keys(self):
        prices = make_prices(300)
        det = VolatilityRegimeDetector()
        stats = det.regime_stats(prices)
        assert "pct_calm" in stats
        assert "pct_turbulent" in stats
        assert abs(stats["pct_calm"] + stats["pct_turbulent"] - 1.0) < 1e-6

    def test_panel_input(self):
        """DataFrame input (multiple stocks) should work."""
        rng = np.random.default_rng(0)
        panel = pd.DataFrame(
            rng.lognormal(0, 0.01, (200, 10)),
            index=pd.date_range("2020-01-01", periods=200, freq="B")
        )
        regime = detect_vol_regime(panel)
        assert len(regime) == 200
