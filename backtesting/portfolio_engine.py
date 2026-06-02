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
  - transaction costs on turnover

Usage
-----
    engine = PortfolioEngine(rebalance="M", cost_bps=10)
    result = engine.run(price_panel, weight_func)
    print(result.metrics)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .metrics import compute_metrics


@dataclass
class PortfolioResult:
    name: str
    equity_curve: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    turnover: float = 0.0

    def summary(self) -> pd.Series:
        s = pd.Series(self.metrics, name=self.name)
        s["avg_turnover"] = round(self.turnover, 4)
        return s


class PortfolioEngine:
    def __init__(
        self,
        rebalance: str = "ME",          # pandas offset: 'ME'=month-end, 'W-FRI', etc.
        cost_bps: float = 10.0,         # round-trip cost in basis points of traded notional
        vol_target: float | None = None,  # e.g. 0.10 for 10% annualised; None = off
        vol_lookback: int = 60,
        max_leverage: float = 2.0,
        inverse_vol: bool = False,      # risk-weight positions by 1/realised-vol
    ):
        self.rebalance = rebalance
        self.cost_bps = cost_bps / 10_000.0
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage
        self.inverse_vol = inverse_vol

    def run(self, prices: pd.DataFrame, weight_func, name: str = "portfolio") -> PortfolioResult:
        """
        prices      : wide DataFrame, rows=dates, cols=tickers (close prices)
        weight_func : callable(prices_up_to_date, date) -> pd.Series of target
                      weights indexed by ticker (NaN/missing = 0).
                      Only called on rebalance dates; uses data up to and
                      INCLUDING that date (weights applied next day → no look-ahead).
        """
        prices = prices.sort_index()
        daily_ret = prices.pct_change().fillna(0.0)

        rebal_dates = prices.resample(self.rebalance).last().index
        rebal_dates = [d for d in rebal_dates if d in prices.index]

        # Build target weight matrix on rebalance dates, forward-fill between
        weight_rows = {}
        for d in rebal_dates:
            hist = prices.loc[:d]
            w = weight_func(hist, d)
            w = w.reindex(prices.columns).fillna(0.0)

            # Inverse-volatility tilt: scale each position by 1/realised-vol, then
            # renormalise so gross exposure is unchanged (risk- not dollar-weighting).
            if self.inverse_vol and w.abs().sum() > 0:
                vol = hist.pct_change().iloc[-self.vol_lookback:].std()
                inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
                tilted = w * inv.reindex(w.index)
                tilted = tilted.fillna(0.0)
                if tilted.abs().sum() > 0:
                    tilted *= w.abs().sum() / tilted.abs().sum()
                    w = tilted
            weight_rows[d] = w
        target_w = pd.DataFrame(weight_rows).T.reindex(prices.index).ffill().fillna(0.0)

        # Apply weights starting the day AFTER they are decided (no look-ahead)
        applied_w = target_w.shift(1).fillna(0.0)

        # Volatility targeting: scale gross exposure
        if self.vol_target is not None:
            port_ret_raw = (applied_w * daily_ret).sum(axis=1)
            realised_vol = port_ret_raw.rolling(self.vol_lookback).std() * np.sqrt(252)
            scale = (self.vol_target / realised_vol).clip(upper=self.max_leverage)
            scale = scale.shift(1).fillna(1.0).replace([np.inf, -np.inf], 1.0)
            applied_w = applied_w.mul(scale, axis=0)

        # Gross portfolio return
        gross_ret = (applied_w * daily_ret).sum(axis=1)

        # Transaction costs on turnover (sum of abs weight changes)
        turnover_series = applied_w.diff().abs().sum(axis=1).fillna(0.0)
        cost = turnover_series * self.cost_bps
        net_ret = gross_ret - cost

        equity = (1 + net_ret).cumprod()
        metrics = compute_metrics(net_ret)

        # Average one-side turnover per rebalance period (standard definition):
        # total two-side turnover amortised over the number of rebalances, halved.
        n_rebal = max(len(rebal_dates), 1)
        avg_turnover = turnover_series.sum() / n_rebal / 2.0

        return PortfolioResult(
            name=name,
            equity_curve=equity,
            returns=net_ret,
            weights=applied_w,
            metrics=metrics,
            turnover=float(avg_turnover) if not np.isnan(avg_turnover) else 0.0,
        )
