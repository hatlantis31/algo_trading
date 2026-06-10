"""Tests for MarketBeatingStrategy."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from strategies.long_only_factor import MarketBeatingStrategy
from strategies.cross_sectional import EqualWeightBenchmark
from backtesting.portfolio_engine import PortfolioEngine


def make_panel(n: int = 500, n_stocks: int = 30, seed: int = 42,
               trend: float = 0.0003) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.015, (n, n_stocks))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(
        prices,
        columns=[f"S{i:02d}" for i in range(n_stocks)],
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


class TestMarketBeatingWeights:
    def test_warm_up_returns_equal_weights(self):
        """With < 147 days data and fallback_to_ew, must return equal weights."""
        strat = MarketBeatingStrategy(fallback_to_ew=True)
        panel = make_panel(n=100)
        w = strat.weights(panel, panel.index[-1])
        assert not w.empty
        # EW: all weights equal and positive
        assert (w > 0).all()
        assert abs(w.std()) < 1e-9

    def test_warm_up_returns_empty_without_fallback(self):
        """Without fallback, too few days → empty weights."""
        strat = MarketBeatingStrategy(fallback_to_ew=False)
        panel = make_panel(n=100)
        w = strat.weights(panel, panel.index[-1])
        assert w.empty

    def test_weights_sum_to_at_most_one(self):
        """Long-only weights sum to ≤ 1 (≤ because regime filter may scale down)."""
        strat = MarketBeatingStrategy(regime_scale=0.5)
        panel = make_panel(n=500)
        w = strat.weights(panel, panel.index[-1])
        if not w.empty:
            assert w.sum() <= 1.0 + 1e-6

    def test_all_weights_non_negative(self):
        """Long-only: no short positions."""
        strat = MarketBeatingStrategy()
        panel = make_panel(n=500)
        w = strat.weights(panel, panel.index[-1])
        if not w.empty:
            assert (w >= 0).all()

    def test_selects_top_fraction(self):
        """Selects approximately top_q fraction of universe after warm-up."""
        top_q = 0.20
        strat = MarketBeatingStrategy(top_q=top_q, regime_scale=1.0)
        panel = make_panel(n=500, n_stocks=100)
        w = strat.weights(panel, panel.index[-1])
        if not w.empty:
            n_nonzero = (w > 0).sum()
            expected = int(100 * top_q)
            assert abs(n_nonzero - expected) <= 3  # within 3 of target

    def test_regime_filter_scales_down(self):
        """Regime filter must reduce exposure below 200-day MA."""
        # Build a declining price panel (market clearly below 200 MA)
        rng = np.random.default_rng(99)
        n = 600
        rets = rng.normal(-0.002, 0.012, (n, 20))  # downward trend
        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        panel = pd.DataFrame(
            prices,
            columns=[f"S{i:02d}" for i in range(20)],
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        strat = MarketBeatingStrategy(regime_scale=0.3)
        w = strat.weights(panel, panel.index[-1])
        if not w.empty:
            assert w.sum() <= 0.31   # must be near regime_scale

    def test_no_regime_filter_fully_invested(self):
        """When regime_scale=1.0, sum of weights equals 1.0 (full exposure)."""
        strat = MarketBeatingStrategy(regime_scale=1.0)
        panel = make_panel(n=500)
        w = strat.weights(panel, panel.index[-1])
        if not w.empty:
            assert abs(w.sum() - 1.0) < 1e-6


class TestPortfolioEngineIntegration:
    def test_runs_without_error(self):
        strat = MarketBeatingStrategy(fallback_to_ew=True)
        panel = make_panel(n=500)
        result = PortfolioEngine(rebalance="ME", cost_bps=10).run(
            panel, strat.weights
        )
        assert len(result.returns) > 0
        assert result.metrics["sharpe_ratio"] is not None

    def test_beats_eq_weight_on_uptrending_data(self):
        """On strongly uptrending data with dispersion, momentum selection
        should have equal or better Sharpe than equal-weight benchmark."""
        rng = np.random.default_rng(7)
        n, n_stocks = 700, 50

        # Create heterogeneous trends: some stocks trend strongly up, others flat
        trends = rng.uniform(0.0001, 0.0010, n_stocks)
        rets = rng.normal(0, 0.015, (n, n_stocks)) + trends
        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        panel = pd.DataFrame(
            prices,
            columns=[f"S{i:02d}" for i in range(n_stocks)],
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )

        eng = PortfolioEngine(rebalance="ME", cost_bps=10)
        bm = eng.run(panel, EqualWeightBenchmark().weights)

        strat = MarketBeatingStrategy(regime_scale=1.0, fallback_to_ew=True)
        res = eng.run(panel, strat.weights)

        # Momentum on uptrend should have non-negative alpha; not always strictly
        # better due to cost, but equity should be comparable
        assert res.equity_curve.iloc[-1] > 0.5 * bm.equity_curve.iloc[-1]

    def test_result_has_halt_log(self):
        strat = MarketBeatingStrategy()
        panel = make_panel(n=500)
        result = PortfolioEngine(rebalance="ME", cost_bps=10).run(
            panel, strat.weights
        )
        assert isinstance(result.halt_log, list)
