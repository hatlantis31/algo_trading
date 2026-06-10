"""
Cross-sectional portfolio backtest engine.

Unlike BacktestEngine (single asset, time-series signal), this engine takes a
*panel* of prices and a function that produces target weights for every asset
on each rebalance date. This is how professional equity quant works: rank the
whole universe, hold a diversified book, rebalance periodically.

Supports:
  - periodic rebalancing (e.g. monthly)
  - long-only or long-short (dollar-neutral) books
  - volatility targeting (scale gross exposure to a target annual vol)
  - inverse-vol position sizing
  - transaction costs on turnover
  - RiskManager integration (position caps, circuit breakers, VaR, beta)

Usage
-----
    from core.risk_manager import RiskManager
    rm = RiskManager(max_position_wt=0.05, max_dd_halt=0.15)
    engine = PortfolioEngine(rebalance="ME", cost_bps=10, risk_manager=rm)
    result = engine.run(price_panel, weight_func)
    print(result.metrics)
    print(result.halt_log)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .metrics import compute_metrics


@dataclass
class PortfolioResult:
    name: str
    equity_curve: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    turnover: float = 0.0
    halt_log: list = field(default_factory=list)   # circuit-breaker events

    def summary(self) -> pd.Series:
        s = pd.Series(self.metrics, name=self.name)
        s["avg_turnover"] = round(self.turnover, 4)
        s["halt_events"] = len(self.halt_log)
        return s


class PortfolioEngine:
    def __init__(
        self,
        rebalance: str = "ME",
        cost_bps: float = 10.0,
        vol_target: Optional[float] = None,
        vol_lookback: int = 60,
        max_leverage: float = 2.0,
        inverse_vol: bool = False,
        risk_manager=None,              # RiskManager instance (optional)
    ):
        self.rebalance = rebalance
        self.cost_bps = cost_bps / 10_000.0
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage
        self.inverse_vol = inverse_vol
        self.risk_manager = risk_manager

    def run(self, prices: pd.DataFrame, weight_func, name: str = "portfolio") -> PortfolioResult:
        """
        prices      : wide DataFrame, rows=dates, cols=tickers (close prices)
        weight_func : callable(prices_up_to_date, date) -> pd.Series of target weights.
                      Uses data up to and INCLUDING rebalance date (applied next day).
        """
        prices = prices.sort_index()
        daily_ret = prices.pct_change().fillna(0.0)
        mkt_ret = daily_ret.mean(axis=1)   # equal-weight market proxy for beta calc

        if self.risk_manager is not None:
            self.risk_manager.reset()

        rebal_dates = prices.resample(self.rebalance).last().index
        rebal_dates = [d for d in rebal_dates if d in prices.index]

        # ── Build target weight matrix ────────────────────────────────────
        weight_rows = {}
        for d in rebal_dates:
            hist = prices.loc[:d]
            w = weight_func(hist, d)
            w = w.reindex(prices.columns).fillna(0.0)

            # Inverse-vol tilt (risk-weight, preserve gross exposure)
            if self.inverse_vol and w.abs().sum() > 0:
                vol = hist.pct_change().iloc[-self.vol_lookback:].std()
                inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
                tilted = (w * inv.reindex(w.index)).fillna(0.0)
                if tilted.abs().sum() > 0:
                    tilted *= w.abs().sum() / tilted.abs().sum()
                    w = tilted

            # Risk manager: position caps, sector caps, leverage, VaR, beta
            if self.risk_manager is not None:
                w = self.risk_manager.check_weights(
                    w, hist, mkt_ret.loc[:d]
                )

            weight_rows[d] = w

        target_w = pd.DataFrame(weight_rows).T.reindex(prices.index).ffill().fillna(0.0)

        # Apply weights starting the day AFTER they are decided (no look-ahead)
        applied_w = target_w.shift(1).fillna(0.0)

        # ── Circuit breaker: zero weights on halted days ───────────────────
        if self.risk_manager is not None:
            equity_running = pd.Series(1.0, index=prices.index, dtype=float)
            halted_mask = pd.Series(False, index=prices.index)
            for i, d in enumerate(prices.index):
                if i == 0:
                    continue
                prev_eq = equity_running.iloc[:i]
                halted_mask.iloc[i] = self.risk_manager.check_halt(prev_eq, d)
                # Update running equity (approximate, used only for circuit breaker).
                # A halted book is in cash: bar return is 0, so the breaker
                # tracks the ACTUAL book, not a hypothetical still-invested one.
                if halted_mask.iloc[i]:
                    bar_ret = 0.0
                else:
                    w_row = applied_w.iloc[i]
                    r_row = daily_ret.iloc[i]
                    bar_ret = (w_row * r_row).sum()
                equity_running.iloc[i] = equity_running.iloc[i - 1] * (1 + bar_ret)

            # Zero out weights on halted days
            applied_w[halted_mask] = 0.0

        # ── Volatility targeting ──────────────────────────────────────────
        if self.vol_target is not None:
            port_ret_raw = (applied_w * daily_ret).sum(axis=1)
            realised_vol = port_ret_raw.rolling(self.vol_lookback).std() * np.sqrt(252)
            scale = (self.vol_target / realised_vol).clip(upper=self.max_leverage)
            scale = scale.shift(1).fillna(1.0).replace([np.inf, -np.inf], 1.0)
            applied_w = applied_w.mul(scale, axis=0)

        # ── P&L ───────────────────────────────────────────────────────────
        gross_ret = (applied_w * daily_ret).sum(axis=1)
        turnover_series = applied_w.diff().abs().sum(axis=1).fillna(0.0)
        cost = turnover_series * self.cost_bps
        net_ret = gross_ret - cost

        equity = (1 + net_ret).cumprod()
        metrics = compute_metrics(net_ret)

        n_rebal = max(len(rebal_dates), 1)
        avg_turnover = turnover_series.sum() / n_rebal / 2.0

        halt_log = self.risk_manager.halt_log.copy() if self.risk_manager else []

        return PortfolioResult(
            name=name,
            equity_curve=equity,
            returns=net_ret,
            weights=applied_w,
            metrics=metrics,
            turnover=float(avg_turnover) if not np.isnan(avg_turnover) else 0.0,
            halt_log=halt_log,
        )
