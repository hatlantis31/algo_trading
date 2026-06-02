"""
Tests for the cross-sectional portfolio engine, factor strategies, and
the professional evaluation suite.
Run: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from backtesting.portfolio_engine import PortfolioEngine
from backtesting.evaluation import (
    information_coefficient, t_stat_of_returns,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio, drawdown_duration,
)
from strategies.cross_sectional import (
    CrossSectionalMomentum, ShortTermReversal, LowVolatility,
    MultiFactor, EqualWeightBenchmark, _to_weights,
)


def make_panel(n_days=400, n_stocks=30, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, (n_days, n_stocks))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"S{i:02d}" for i in range(n_stocks)]
    return pd.DataFrame(prices, columns=cols,
                        index=pd.date_range("2020-01-01", periods=n_days, freq="B"))


class TestWeights:
    def test_long_only_sums_to_one(self):
        sig = pd.Series(np.arange(20), index=[f"S{i}" for i in range(20)])
        w = _to_weights(sig, top_q=0.2, long_short=False)
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= 0).all()

    def test_long_short_dollar_neutral(self):
        sig = pd.Series(np.arange(20), index=[f"S{i}" for i in range(20)])
        w = _to_weights(sig, top_q=0.2, long_short=True)
        assert abs(w.sum()) < 1e-9          # dollar neutral
        assert abs(w.abs().sum() - 2.0) < 1e-9  # gross ≈ 2 (1 long + 1 short)


class TestPortfolioEngine:
    def test_runs_and_shapes(self):
        panel = make_panel()
        eng = PortfolioEngine(rebalance="ME", cost_bps=10)
        res = eng.run(panel, CrossSectionalMomentum(long_short=False).weights)
        assert len(res.equity_curve) == len(panel)
        assert "sharpe_ratio" in res.metrics

    def test_no_lookahead(self):
        """Truncating the panel must not change earlier weights."""
        panel = make_panel(400)
        eng = PortfolioEngine(rebalance="ME", cost_bps=0)
        strat = CrossSectionalMomentum(long_short=False)
        full = eng.run(panel, strat.weights)
        part = eng.run(panel.iloc[:300], strat.weights)
        # Equity path over the shared window should match closely
        shared = full.returns.iloc[:280]
        shared2 = part.returns.iloc[:280]
        assert np.allclose(shared.values, shared2.values, atol=1e-9)

    def test_costs_reduce_returns(self):
        panel = make_panel()
        strat = MultiFactor(long_short=True)
        cheap = PortfolioEngine(rebalance="ME", cost_bps=0).run(panel, strat.weights)
        dear  = PortfolioEngine(rebalance="ME", cost_bps=50).run(panel, strat.weights)
        assert cheap.metrics["total_return"] >= dear.metrics["total_return"]

    def test_vol_targeting_controls_vol(self):
        panel = make_panel()
        strat = MultiFactor(long_short=False)
        vt = PortfolioEngine(rebalance="ME", cost_bps=0, vol_target=0.10).run(panel, strat.weights)
        # realised annual vol should be in a sane band around the 10% target
        realised = vt.returns.std() * np.sqrt(252)
        assert 0.03 < realised < 0.30


class TestEvaluation:
    def test_ic_perfect_signal(self):
        """If signal == forward return, IC should be ~1."""
        panel = make_panel(100, 20)
        fwd = panel.pct_change().shift(-1)
        ic = information_coefficient(fwd, fwd)  # signal identical to outcome
        assert ic["ic_mean"] > 0.95

    def test_ic_random_signal_near_zero(self):
        panel = make_panel(200, 25)
        fwd = panel.pct_change().shift(-1)
        rng = np.random.default_rng(1)
        noise = pd.DataFrame(rng.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
        ic = information_coefficient(noise, fwd)
        assert abs(ic["ic_mean"]) < 0.1

    def test_t_stat(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.001, 0.01, 500))
        out = t_stat_of_returns(r)
        assert "t_stat" in out and "p_value" in out

    def test_psr_bounds(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.001, 0.01, 500))
        psr = probabilistic_sharpe_ratio(r, 0.0)
        assert 0.0 <= psr <= 1.0

    def test_deflated_sharpe_more_trials_lower(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0008, 0.01, 750))
        dsr_few  = deflated_sharpe_ratio(r, n_trials=1)["dsr"]
        dsr_many = deflated_sharpe_ratio(r, n_trials=200)["dsr"]
        assert dsr_many <= dsr_few   # more trials → harder to be significant

    def test_drawdown_duration(self):
        eq = pd.Series([1, 1.1, 1.05, 0.9, 0.95, 1.2])
        dd = drawdown_duration(eq)
        assert dd["max_dd"] < 0
        assert dd["longest_underwater_obs"] >= 1
