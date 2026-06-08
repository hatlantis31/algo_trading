"""Tests for the hedge-fund pipeline: alpha factors, risk model, optimizer,
ML alpha, CPCV, and the integrated HedgeFundStrategy."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from strategies.alpha_factors import (
    momentum, reversal, low_volatility, standardize, neutralize,
    build_factor_matrix, fundamental_factor_matrix, ic_weighted_alpha, ALL_FACTORS,
)
from backtesting.risk_model import RiskModel
from backtesting.optimizer import mean_variance_neutral, realized_neutrality, long_only_optimized
from backtesting.cpcv import CombinatorialPurgedCV
from strategies.hedge_fund_strategy import HedgeFundStrategy
from backtesting.portfolio_engine import PortfolioEngine


def make_panels(n=400, n_stocks=40, seed=0):
    """Synthetic panel with a shared market factor so the covariance is non-trivial
    (correlated names) — needed for the risk model / optimizer to be exercised."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.010, (n, 1))
    idio = rng.normal(0, 0.010, (n, n_stocks))
    betas = rng.uniform(0.5, 1.5, n_stocks)
    rets = market * betas + idio
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                          columns=[f"S{i:02d}" for i in range(n_stocks)],
                          index=pd.date_range("2020-01-01", periods=n, freq="B"))
    volume = pd.DataFrame(rng.integers(1e6, 1e7, (n, n_stocks)),
                          columns=prices.columns, index=prices.index)
    return prices, volume


class TestAlphaFactors:
    def test_momentum_shape(self):
        p, v = make_panels()
        m = momentum(p, 252, 21)
        assert len(m) == p.shape[1]

    def test_standardize_zero_mean_unit_std(self):
        s = pd.Series([1, 2, 3, 4, 5, 100], dtype=float)
        z = standardize(s)
        assert abs(z.mean()) < 1e-9
        assert abs(z.std() - 1.0) < 0.2  # winsorization affects std slightly

    def test_neutralize_removes_beta(self):
        rng = np.random.default_rng(0)
        beta = pd.Series(rng.normal(1, 0.3, 50), index=[f"S{i}" for i in range(50)])
        alpha = 2 * beta + rng.normal(0, 0.1, 50)  # alpha highly correlated with beta
        resid = neutralize(alpha, beta=beta)
        # residual should be ~uncorrelated with beta
        assert abs(np.corrcoef(resid, beta.reindex(resid.index))[0, 1]) < 0.1

    def test_factor_matrix_columns(self):
        p, v = make_panels()
        fm = build_factor_matrix(p, v)
        assert fm.shape[1] >= 5
        assert not fm.empty


def make_fundamentals(tickers, seed=7):
    """Synthetic static value/quality frame matching the loader's schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "earnings_yield": rng.uniform(0.01, 0.10, len(tickers)),
        "book_to_price":  rng.uniform(0.10, 1.50, len(tickers)),
        "sales_yield":    rng.uniform(0.05, 3.00, len(tickers)),
        "ebitda_yield":   rng.uniform(0.02, 0.30, len(tickers)),
        "dividend_yield": rng.uniform(0.00, 0.06, len(tickers)),
        "size":          -rng.uniform(22, 29, len(tickers)),
    }, index=list(tickers))


class TestFundamentals:
    def test_fundamental_matrix_standardized(self):
        p, _ = make_panels()
        fund = make_fundamentals(p.columns)
        fm = fundamental_factor_matrix(fund, list(p.columns))
        assert not fm.empty
        # each column cross-sectionally standardized → ~zero mean
        assert fm.mean().abs().max() < 1e-6
        assert fm.shape[1] == 6

    def test_build_matrix_appends_fundamentals(self):
        p, v = make_panels()
        base = build_factor_matrix(p, v)
        withf = build_factor_matrix(p, v, fundamentals=make_fundamentals(p.columns))
        # fundamentals add fnd_* columns without dropping the price factors
        assert withf.shape[1] > base.shape[1]
        assert any(c.startswith("fnd_") for c in withf.columns)

    def test_strategy_accepts_fundamentals(self):
        p, v = make_panels(400, 40)
        fund = make_fundamentals(p.columns)
        hf = HedgeFundStrategy(volume_panel=v, fundamentals=fund,
                               alpha_combination="ic_weighted", construction="decile")
        res = PortfolioEngine("ME", cost_bps=10).run(p, hf.weights)
        assert res.equity_curve is not None

    def test_missing_fundamentals_are_safe(self):
        """Tickers absent from the fundamentals frame must not crash the pipeline."""
        p, v = make_panels(400, 40)
        fund = make_fundamentals(list(p.columns)[:20])   # only half covered
        fm = build_factor_matrix(p, v, fundamentals=fund)
        assert any(c.startswith("fnd_") for c in fm.columns)


class TestRiskModel:
    def test_cov_well_conditioned(self):
        p, _ = make_panels()
        rm = RiskModel(lookback=252).fit(p)
        # LW shrinkage should keep condition number reasonable
        assert rm.condition_number() < 1e5
        assert rm.cov.shape[0] == p.shape[1]

    def test_betas_near_one_on_average(self):
        p, _ = make_panels()
        rm = RiskModel().fit(p)
        # average beta to equal-weight market should be ~1
        assert 0.7 < rm.betas.mean() < 1.3

    def test_precision_inverts_cov(self):
        p, _ = make_panels()
        rm = RiskModel().fit(p)
        prod = rm.cov.values @ rm.precision().values
        assert np.allclose(prod, np.eye(len(rm.tickers)), atol=1e-6)


class TestOptimizer:
    def test_market_neutral_constraints(self):
        p, _ = make_panels()
        rm = RiskModel().fit(p)
        rng = np.random.default_rng(1)
        alpha = pd.Series(rng.normal(size=len(rm.tickers)), index=rm.tickers)
        w = mean_variance_neutral(alpha, rm.cov, rm.betas,
                                  gross_leverage=1.0, position_cap=0.1)
        neut = realized_neutrality(w, rm.betas)
        assert abs(neut["net_exposure"]) < 1e-6        # dollar neutral
        assert abs(neut["portfolio_beta"]) < 0.05      # beta neutral
        assert abs(neut["gross_leverage"] - 1.0) < 0.01

    def test_position_cap_respected(self):
        p, _ = make_panels()
        rm = RiskModel().fit(p)
        rng = np.random.default_rng(2)
        alpha = pd.Series(rng.normal(size=len(rm.tickers)), index=rm.tickers)
        w = mean_variance_neutral(alpha, rm.cov, rm.betas, position_cap=0.03)
        assert w.abs().max() <= 0.03 + 1e-6

    def test_shrinkage_changes_weights(self):
        p, _ = make_panels()
        rm = RiskModel().fit(p)
        rng = np.random.default_rng(3)
        alpha = pd.Series(rng.normal(size=len(rm.tickers)), index=rm.tickers)
        w0 = mean_variance_neutral(alpha, rm.cov, rm.betas, shrinkage=0.0)
        w1 = mean_variance_neutral(alpha, rm.cov, rm.betas, shrinkage=1.0)
        assert not np.allclose(w0.values, w1.values)

    def test_long_only_sums_to_one(self):
        p, _ = make_panels()
        rm = RiskModel().fit(p)
        rng = np.random.default_rng(4)
        alpha = pd.Series(rng.normal(size=len(rm.tickers)), index=rm.tickers)
        w = long_only_optimized(alpha, rm.cov, top_q=0.3)
        assert (w >= 0).all()
        assert abs(w.sum() - 1.0) < 1e-6


class TestCPCV:
    def test_n_paths(self):
        cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2)
        assert cv.n_paths() == 15   # C(6,2)

    def test_splits_disjoint(self):
        cv = CombinatorialPurgedCV(n_splits=5, n_test_groups=1, embargo=5)
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        for train, test in cv.split(idx):
            assert len(set(train) & set(test)) == 0   # no overlap

    def test_evaluate_returns_paths(self):
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.0005, 0.01, 500),
                         index=pd.date_range("2020-01-01", periods=500, freq="B"))
        cv = CombinatorialPurgedCV(6, 2, 21)
        res = cv.evaluate(rets)
        assert res.summary()["n_paths"] > 0


class TestHedgeFundStrategy:
    def test_integration_market_neutral(self):
        p, v = make_panels(400, 40)
        hf = HedgeFundStrategy(volume_panel=v, alpha_combination="equal",
                               construction="optimizer", gross_leverage=1.0)
        res = PortfolioEngine("ME", cost_bps=10).run(p, hf.weights)
        assert len(res.equity_curve) == len(p)
        # market-neutral book should have low net exposure on average
        avg_net = res.weights.sum(axis=1).abs().mean()
        assert avg_net < 0.1

    def test_decile_construction(self):
        p, v = make_panels(400, 40)
        hf = HedgeFundStrategy(volume_panel=v, alpha_combination="equal",
                               construction="decile", gross_leverage=1.0)
        res = PortfolioEngine("ME", cost_bps=10).run(p, hf.weights)
        assert "sharpe_ratio" in res.metrics

    def test_ic_weighted_runs(self):
        p, v = make_panels(400, 40)
        hf = HedgeFundStrategy(volume_panel=v, alpha_combination="ic_weighted",
                               construction="decile")
        res = PortfolioEngine("ME", cost_bps=10).run(p, hf.weights)
        assert res.equity_curve is not None

    def test_no_lookahead(self):
        """Truncating the panel must not change earlier weights."""
        p, v = make_panels(400, 40)
        hf1 = HedgeFundStrategy(volume_panel=v, alpha_combination="equal", construction="decile")
        hf2 = HedgeFundStrategy(volume_panel=v, alpha_combination="equal", construction="decile")
        full = PortfolioEngine("ME", cost_bps=0).run(p, hf1.weights)
        part = PortfolioEngine("ME", cost_bps=0).run(p.iloc[:300], hf2.weights)
        a = full.returns.iloc[:280].values
        b = part.returns.iloc[:280].values
        assert np.allclose(a, b, atol=1e-8)
