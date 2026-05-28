"""
Unit tests for all strategies.
Run: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from strategies import STRATEGY_REGISTRY, BaseStrategy


def make_ohlcv(n=300, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   close * (1 + rng.uniform(0, 0.01, n)),
        "low":    close * (1 - rng.uniform(0, 0.01, n)),
        "close":  close,
        "volume": rng.integers(100_000, 1_000_000, n),
    }, index=pd.date_range("2022-01-01", periods=n, freq="B"))
    return df


@pytest.mark.parametrize("name,cls", STRATEGY_REGISTRY.items())
def test_strategy_output_shape(name, cls):
    df = make_ohlcv()
    strat = cls()
    signals = strat.generate_signals(df)
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(df)


@pytest.mark.parametrize("name,cls", STRATEGY_REGISTRY.items())
def test_signal_values(name, cls):
    df = make_ohlcv()
    signals = cls().generate_signals(df)
    assert set(signals.dropna().unique()).issubset({-1.0, 0.0, 1.0})


@pytest.mark.parametrize("name,cls", STRATEGY_REGISTRY.items())
def test_no_lookahead(name, cls):
    """Signals must not depend on data after their timestamp (basic check)."""
    df = make_ohlcv(300)
    df_short = df.iloc[:150]
    strat = cls()
    sig_full  = strat.generate_signals(df)
    sig_short = strat.generate_signals(df_short)
    # First 150 signals should be identical
    pd.testing.assert_series_equal(
        sig_full.iloc[:150].reset_index(drop=True),
        sig_short.reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize("name,cls", STRATEGY_REGISTRY.items())
def test_custom_params(name, cls):
    df = make_ohlcv()
    strat = cls(**{k: v for k, v in cls.param_grid.items()})
    signals = strat.generate_signals(df)
    assert not signals.isnull().all()
