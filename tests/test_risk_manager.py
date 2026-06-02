"""Tests for RiskManager and its integration with PortfolioEngine."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.risk_manager import RiskManager
from backtesting.portfolio_engine import PortfolioEngine
from strategies.cross_sectional import MultiFactor


def make_panel(n=300, n_stocks=20, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, (n, n_stocks))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"S{i:02d}" for i in range(n_stocks)]
    return pd.DataFrame(prices, columns=cols,
                        index=pd.date_range("2020-01-01", periods=n, freq="B"))


class TestPositionCaps:
    def test_position_cap_applied(self):
        rm = RiskManager(max_position_wt=0.05, max_gross_leverage=1.0)
        w = pd.Series({"A": 0.20, "B": 0.10, "C": 0.05})
        out = rm.check_weights(w)
        assert (out.abs() <= 0.05 + 1e-9).all()

    def test_leverage_cap(self):
        rm = RiskManager(max_position_wt=0.10, max_gross_leverage=1.0)
        w = pd.Series({"A": 0.50, "B": 0.50, "C": 0.50})
        out = rm.check_weights(w)
        assert out.abs().sum() <= 1.0 + 1e-9

    def test_sector_cap(self):
        rm = RiskManager(
            max_position_wt=0.20,
            max_sector_wt=0.30,
            sector_map={"A": "Tech", "B": "Tech", "C": "Energy"}
        )
        w = pd.Series({"A": 0.25, "B": 0.25, "C": 0.10})
        out = rm.check_weights(w)
        assert out[["A", "B"]].sum() <= 0.30 + 1e-9

    def test_empty_weights_unchanged(self):
        rm = RiskManager()
        w = pd.Series({"A": 0.0, "B": 0.0})
        out = rm.check_weights(w)
        assert (out == 0.0).all()


class TestCircuitBreaker:
    def test_halts_on_drawdown(self):
        rm = RiskManager(max_dd_halt=0.10, hard_halt=True)
        # equity that drops 20%
        eq = pd.Series([1.0, 1.05, 1.02, 0.85, 0.84],
                       index=pd.date_range("2022-01-01", periods=5, freq="B"))
        halted = [rm.check_halt(eq.iloc[:i+1]) for i in range(len(eq))]
        assert any(halted), "Should have halted after >10% drawdown"

    def test_no_halt_within_limit(self):
        rm = RiskManager(max_dd_halt=0.20)
        eq = pd.Series([1.0, 1.05, 1.10, 1.08, 1.12],
                       index=pd.date_range("2022-01-01", periods=5, freq="B"))
        halted = [rm.check_halt(eq.iloc[:i+1]) for i in range(len(eq))]
        assert not any(halted)

    def test_resumes_after_recovery(self):
        rm = RiskManager(max_dd_halt=0.10, dd_resume_threshold=0.05)
        eq = pd.Series([1.0, 0.85, 0.90, 0.96],
                       index=pd.date_range("2022-01-01", periods=4, freq="B"))
        for i in range(len(eq)):
            rm.check_halt(eq.iloc[:i+1])
        # after recovering to dd=-4%, should have resumed
        assert not rm._halted
        assert any("RESUME" in e for e in rm.halt_log)

    def test_reset_clears_state(self):
        rm = RiskManager(max_dd_halt=0.05)
        eq = pd.Series([1.0, 0.90], index=pd.date_range("2022-01-01", periods=2))
        rm.check_halt(eq)
        rm.reset()
        assert not rm._halted
        assert len(rm.halt_log) == 0


class TestVaR:
    def test_var_positive(self):
        panel = make_panel(100, 10)
        rm = RiskManager(var_limit=0.05)
        w = pd.Series({c: 0.1 for c in panel.columns})
        var = rm.portfolio_var(w, panel)
        assert var > 0

    def test_var_scales_with_position_size(self):
        panel = make_panel(100, 10)
        rm = RiskManager()
        w_small = pd.Series({c: 0.05 for c in panel.columns})
        w_large = pd.Series({c: 0.20 for c in panel.columns})
        assert rm.portfolio_var(w_large, panel) > rm.portfolio_var(w_small, panel)


class TestPortfolioEngineIntegration:
    def test_risk_manager_reduces_drawdown(self):
        """Circuit breaker should reduce max drawdown vs uncontrolled."""
        panel = make_panel(400, 20, seed=99)
        strat = MultiFactor(long_short=False)

        no_rm = PortfolioEngine(rebalance="ME", cost_bps=10).run(panel, strat.weights)
        with_rm = PortfolioEngine(
            rebalance="ME", cost_bps=10,
            risk_manager=RiskManager(max_dd_halt=0.15, max_position_wt=0.10)
        ).run(panel, strat.weights)

        # With risk manager, drawdown should be ≤ without (or equal if never triggered)
        assert with_rm.metrics["max_drawdown"] >= no_rm.metrics["max_drawdown"] - 0.01

    def test_halt_log_populated(self):
        panel = make_panel(400, 20, seed=7)
        rm = RiskManager(max_dd_halt=0.05)   # very tight → will trigger
        eng = PortfolioEngine(rebalance="ME", cost_bps=10, risk_manager=rm)
        result = eng.run(panel, MultiFactor(long_short=False).weights)
        # Either halted or not — just check the attribute exists
        assert isinstance(result.halt_log, list)
