"""
Tests for the backtest engine and metrics.
Run: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from backtesting import BacktestEngine, compute_metrics
from strategies import MACrossStrategy, RSIStrategy


def make_ohlcv(n=500, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1_000_000},
        index=pd.date_range("2021-01-01", periods=n, freq="B"),
    )


class TestMetrics:
    def test_zero_returns(self):
        r = pd.Series(np.zeros(252))
        m = compute_metrics(r)
        assert m["total_return"] == 0.0
        assert m["max_drawdown"] == 0.0

    def test_positive_sharpe(self):
        rng = np.random.default_rng(0)
        r = pd.Series(np.abs(rng.normal(0.001, 0.005, 252)))
        m = compute_metrics(r)
        assert m["sharpe_ratio"] > 0

    def test_keys_present(self):
        r = pd.Series(np.random.normal(0, 0.01, 252))
        m = compute_metrics(r)
        for key in ["total_return", "ann_return", "sharpe_ratio",
                    "max_drawdown", "win_rate", "num_trades"]:
            assert key in m


class TestBacktestEngine:
    def test_basic_run(self):
        engine = BacktestEngine(initial_capital=10_000)
        df = make_ohlcv()
        result = engine.run(MACrossStrategy(), df, ticker="TEST")
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(df)
        assert result.metrics["sharpe_ratio"] is not None

    def test_equity_starts_at_capital(self):
        engine = BacktestEngine(initial_capital=10_000)
        df = make_ohlcv()
        result = engine.run(MACrossStrategy(), df)
        assert abs(result.equity_curve.iloc[0] - 10_000) < 200  # first bar can move

    def test_walk_forward_splits(self):
        engine = BacktestEngine()
        df = make_ohlcv(500)
        results = engine.run_walk_forward(RSIStrategy(), df, n_splits=5)
        assert len(results) == 5

    def test_commission_reduces_returns(self):
        df = make_ohlcv()
        strat = MACrossStrategy()
        r_no_cost  = BacktestEngine(commission=0.0, slippage=0.0).run(strat, df).metrics
        r_with_cost = BacktestEngine(commission=0.01, slippage=0.005).run(strat, df).metrics
        assert r_no_cost["total_return"] >= r_with_cost["total_return"]
