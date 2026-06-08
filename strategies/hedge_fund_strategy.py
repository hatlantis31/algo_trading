"""
HedgeFundStrategy — full systematic equity market-neutral pipeline.

This is the "as close as possible to a real quant fund" strategy. On each
rebalance date it runs the complete institutional pipeline:

    price/volume  →  alpha factors  →  combine (ML or linear)  →  neutralize
                  →  risk model (LW covariance + betas)  →  MV optimizer
                  →  dollar- & beta-neutral, position-capped weights

Plugs into PortfolioEngine.run(panel, strategy.weights). The engine then adds
volatility targeting, transaction costs, and the RiskManager circuit breakers
on top — exactly the layering a fund uses (alpha → construction → risk overlay).

Two modes:
    use_ml=True   → HistGradientBoosting cross-sectional ranker (Gu-Kelly-Xiu)
    use_ml=False  → equal-weight z-score factor blend (robust baseline)
"""

import numpy as np
import pandas as pd

from .alpha_factors import (build_factor_matrix, standardize, neutralize,
                            ic_weighted_alpha, ALL_FACTORS)
from .ml_alpha import MLAlphaModel
from backtesting.risk_model import RiskModel
from backtesting.optimizer import mean_variance_neutral, long_only_optimized


class HedgeFundStrategy:
    def __init__(
        self,
        volume_panel: pd.DataFrame = None,
        alpha_combination: str = "ic_weighted",   # 'equal' | 'ic_weighted' | 'ml'
        market_neutral: bool = True,
        construction: str = "optimizer",           # 'optimizer' | 'decile'
        gross_leverage: float = 1.0,
        position_cap: float = 0.04,
        risk_lookback: int = 252,
        horizon: int = 21,
        opt_shrinkage: float = 0.5,
        factor_set: dict = None,
        sector_map: dict = None,
        fundamentals: pd.DataFrame = None,
        ml_model: MLAlphaModel = None,
        name: str = None,
    ):
        self.volume_panel = volume_panel
        self.fundamentals = fundamentals
        self.alpha_combination = alpha_combination
        self.market_neutral = market_neutral
        self.construction = construction
        self.gross_leverage = gross_leverage
        self.position_cap = position_cap
        self.risk_lookback = risk_lookback
        self.opt_shrinkage = opt_shrinkage
        self.horizon = horizon
        self.factor_set = factor_set or ALL_FACTORS
        self.sector_map = pd.Series(sector_map) if sector_map else None
        self.ml_model = ml_model or MLAlphaModel(horizon=horizon)

        self.name = name or f"HedgeFund({alpha_combination},{'MN' if market_neutral else 'LO'})"
        self._feature_cache: dict = {}
        self._last_alpha: pd.Series | None = None

    # ── Main weight function (called by PortfolioEngine) ─────────────────────

    def weights(self, prices: pd.DataFrame, date) -> pd.Series:
        # 1. Alpha factors at this date (cache for ML training)
        vol_hist = self.volume_panel.loc[:date] if self.volume_panel is not None else None
        fm = build_factor_matrix(prices, vol_hist, factors=self.factor_set,
                                 fundamentals=self.fundamentals)
        self._feature_cache[date] = fm
        if fm.empty or len(fm) < 10:
            return pd.Series(dtype=float)

        # 2. Risk model (shrunk covariance + betas)
        rm = RiskModel(lookback=self.risk_lookback).fit(prices)
        if rm.cov is None:
            return pd.Series(dtype=float)

        # 3. Combine factors into one alpha
        if self.alpha_combination == "ml":
            alpha = self.ml_model.predict_alpha(self._feature_cache, prices, date)
            if alpha.empty:
                alpha = fm.mean(axis=1)
        elif self.alpha_combination == "ic_weighted":
            alpha = ic_weighted_alpha(self._feature_cache, prices, date,
                                      horizon=self.horizon)
            if alpha.empty:
                alpha = fm.mean(axis=1)
        else:  # 'equal'
            alpha = fm.mean(axis=1)
        alpha = standardize(alpha)

        # 4. Neutralize alpha vs beta (and sector if available)
        sectors = self.sector_map.reindex(alpha.index) if self.sector_map is not None else None
        alpha = neutralize(alpha, beta=rm.betas, sectors=sectors)
        self._last_alpha = alpha

        # 5. Portfolio construction → weights
        if not self.market_neutral:
            return long_only_optimized(alpha, rm.cov, top_q=0.3,
                                       position_cap=self.position_cap)
        if self.construction == "decile":
            return self._decile_weights(alpha)
        return mean_variance_neutral(
            alpha, rm.cov, rm.betas,
            gross_leverage=self.gross_leverage,
            position_cap=self.position_cap,
            shrinkage=self.opt_shrinkage,
        )

    def _decile_weights(self, alpha: pd.Series, top_q: float = 0.2) -> pd.Series:
        """Simple long-short decile spread (dollar-neutral), gross = gross_leverage."""
        a = alpha.dropna()
        if len(a) < 20:
            return pd.Series(dtype=float)
        k = max(1, int(len(a) * top_q))
        ranked = a.sort_values(ascending=False)
        w = pd.Series(0.0, index=a.index)
        w[ranked.index[:k]] = (self.gross_leverage / 2) / k
        w[ranked.index[-k:]] = -(self.gross_leverage / 2) / k
        return w

    def reset_cache(self):
        self._feature_cache.clear()
        self._last_alpha = None
