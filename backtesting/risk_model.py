"""
Risk model — the covariance and beta estimates a portfolio optimizer needs.

Sample covariance on ~150 stocks with ~250 days of data is noisy and often
ill-conditioned (you can't invert it reliably). Ledoit-Wolf shrinkage pulls the
sample covariance toward a structured target, producing a well-conditioned,
invertible matrix. This is standard practice at quant funds.

Reference: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix".
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class RiskModel:
    def __init__(self, lookback: int = 252, min_obs: int = 60):
        self.lookback = lookback
        self.min_obs = min_obs
        self._cov: pd.DataFrame | None = None
        self._betas: pd.Series | None = None
        self._tickers: list | None = None

    def fit(self, prices: pd.DataFrame) -> "RiskModel":
        """Estimate shrunk covariance and market betas from a price panel."""
        rets = prices.pct_change().iloc[-self.lookback:].dropna(how="all")
        # Real panels have NaN from listings/delistings — dropping whole rows
        # would collapse the frame. Keep tickers that are still trading (data in
        # the last week) with sufficient history, then zero-fill sporadic gaps
        # (a zero return is a neutral, standard imputation for risk estimation).
        alive = rets.columns[rets.iloc[-5:].notna().any()]
        good = [c for c in alive if rets[c].notna().sum() >= self.min_obs]
        rets = rets[good].fillna(0.0)
        if rets.shape[0] < self.min_obs or rets.shape[1] < 2:
            self._cov, self._betas, self._tickers = None, None, list(good)
            return self

        # ── Ledoit-Wolf shrinkage covariance (daily) ───────────────────────
        lw = LedoitWolf().fit(rets.values)
        self._cov = pd.DataFrame(lw.covariance_, index=rets.columns, columns=rets.columns)

        # ── Market betas (vs equal-weight market) ──────────────────────────
        mkt = rets.mean(axis=1)
        mkt_var = mkt.var()
        betas = {}
        for col in rets.columns:
            betas[col] = np.cov(rets[col], mkt)[0, 1] / mkt_var if mkt_var > 0 else 1.0
        self._betas = pd.Series(betas)
        self._tickers = list(rets.columns)
        return self

    @property
    def cov(self) -> pd.DataFrame:
        return self._cov

    @property
    def betas(self) -> pd.Series:
        return self._betas

    @property
    def tickers(self) -> list:
        return self._tickers or []

    def precision(self) -> pd.DataFrame | None:
        """Inverse covariance (precision matrix), needed for MV optimization."""
        if self._cov is None:
            return None
        prec = np.linalg.inv(self._cov.values)
        return pd.DataFrame(prec, index=self._cov.index, columns=self._cov.columns)

    def annualized_vol(self) -> pd.Series:
        """Per-stock annualized volatility from the diagonal."""
        if self._cov is None:
            return pd.Series(dtype=float)
        return pd.Series(np.sqrt(np.diag(self._cov.values)) * np.sqrt(252),
                         index=self._cov.index)

    def portfolio_vol(self, weights: pd.Series) -> float:
        """Annualized portfolio volatility for a given weight vector."""
        if self._cov is None:
            return np.nan
        w = weights.reindex(self._cov.index).fillna(0.0).values
        daily_var = w @ self._cov.values @ w
        return float(np.sqrt(max(daily_var, 0)) * np.sqrt(252))

    def condition_number(self) -> float:
        """Condition number of the covariance — lower = better conditioned.
        Sample cov is often >1e6 (near-singular); LW shrinkage keeps it manageable."""
        if self._cov is None:
            return np.nan
        return float(np.linalg.cond(self._cov.values))
