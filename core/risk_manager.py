"""
RiskManager — portfolio-level and position-level risk controls.

Sits between signal generation and order execution. Call check_weights()
on every rebalance date; the portfolio engine calls check_halt() on every
bar to decide whether trading is suspended.

Controls implemented
--------------------
1. Position cap           — no single position > max_position_wt
2. Sector cap             — no sector > max_sector_wt (requires sector_map)
3. Gross leverage cap     — total |weights| ≤ max_gross_leverage
4. Drawdown circuit breaker — halt all trading when portfolio DD > max_dd_halt;
                              resume only when it recovers past dd_resume_threshold
5. Daily loss limit       — halt if intraday/daily PnL < max_daily_loss
6. VaR check (historical) — warn/cap if 1-day 95% VaR > var_limit
7. Beta cap               — scale down positions if portfolio beta > max_beta

Usage in backtesting
--------------------
    rm = RiskManager(max_position_wt=0.05, max_dd_halt=0.15)
    engine = PortfolioEngine(risk_manager=rm, ...)
    result = engine.run(panel, strat.weights)

Usage before live order
-----------------------
    rm = RiskManager(...)
    safe_weights = rm.check_weights(raw_weights, prices_history)
    if rm.check_halt(equity_series):
        print("HALT — circuit breaker active, do not trade")
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        # ── Position limits ───────────────────────────────────────────────
        max_position_wt: float = 0.05,     # 5% max single name (long or short)
        max_gross_leverage: float = 1.0,   # 1.0 = long-only, no leverage
        max_sector_wt: float = 0.30,       # 30% max any sector
        sector_map: Optional[dict] = None, # {ticker: sector}

        # ── Portfolio-level circuit breakers ─────────────────────────────
        max_dd_halt: float = 0.15,         # halt at 15% drawdown from peak
        dd_resume_threshold: float = 0.05, # resume only when DD recovers to 5%
        max_daily_loss: float = 0.03,      # halt if single-day loss > 3%

        # ── Risk metrics ─────────────────────────────────────────────────
        var_limit: float = 0.02,           # 95% 1-day historical VaR limit
        var_lookback: int = 252,
        max_beta: float = 1.5,             # portfolio beta to broad market

        # ── Behaviour ────────────────────────────────────────────────────
        hard_halt: bool = True,            # True = go to cash on breach; False = warn only
    ):
        self.max_position_wt     = max_position_wt
        self.max_gross_leverage  = max_gross_leverage
        self.max_sector_wt       = max_sector_wt
        self.sector_map          = sector_map or {}
        self.max_dd_halt         = max_dd_halt
        self.dd_resume_threshold = dd_resume_threshold
        self.max_daily_loss      = max_daily_loss
        self.var_limit           = var_limit
        self.var_lookback        = var_lookback
        self.max_beta            = max_beta
        self.hard_halt           = hard_halt

        self._halted = False            # circuit-breaker state (stateful)
        self.halt_log: list[str] = []   # history of halt/resume events

    # ── Main entry points ────────────────────────────────────────────────────

    def check_weights(
        self,
        weights: pd.Series,
        prices_history: Optional[pd.DataFrame] = None,
        market_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Apply all position-level controls to raw target weights.
        Returns adjusted weights (never raises — safe to call in any state).
        """
        w = weights.copy().fillna(0.0)

        w = self._cap_positions(w)
        w = self._cap_sector(w)
        w = self._cap_leverage(w)

        if prices_history is not None:
            w = self._check_var(w, prices_history)

        if market_returns is not None:
            w = self._cap_beta(w, prices_history, market_returns)

        return w

    def check_halt(
        self,
        equity_curve: pd.Series,
        current_date=None,
    ) -> bool:
        """
        Returns True if trading should be HALTED (go to cash immediately).
        Updates internal circuit-breaker state.
        Should be called on every bar.
        """
        if len(equity_curve) < 2:
            return False

        peak = equity_curve.cummax().iloc[-1]
        current = equity_curve.iloc[-1]
        dd = current / peak - 1

        daily_ret = equity_curve.pct_change().iloc[-1]
        date_str = str(current_date or equity_curve.index[-1])[:10]

        # ── Check for halt conditions ────────────────────────────────────
        if not self._halted:
            if dd < -self.max_dd_halt:
                msg = f"{date_str}  HALT: drawdown {dd:.1%} exceeded limit {-self.max_dd_halt:.1%}"
                self._halted = True
                self.halt_log.append(msg)
                logger.warning(msg)

            elif not np.isnan(daily_ret) and daily_ret < -self.max_daily_loss:
                msg = f"{date_str}  HALT: daily loss {daily_ret:.1%} exceeded limit {-self.max_daily_loss:.1%}"
                self._halted = True
                self.halt_log.append(msg)
                logger.warning(msg)

        # ── Check for resume ─────────────────────────────────────────────
        elif self._halted and dd > -self.dd_resume_threshold:
            msg = f"{date_str}  RESUME: drawdown recovered to {dd:.1%}"
            self._halted = False
            self.halt_log.append(msg)
            logger.info(msg)

        return self._halted

    def reset(self):
        """Reset circuit-breaker state (call before each new backtest run)."""
        self._halted = False
        self.halt_log.clear()

    # ── VaR ─────────────────────────────────────────────────────────────────

    def portfolio_var(
        self,
        weights: pd.Series,
        prices_history: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """Historical simulation 1-day VaR at given confidence level."""
        rets = prices_history.pct_change().iloc[-self.var_lookback:]
        w = weights.reindex(rets.columns).fillna(0.0)
        port_rets = (rets * w).sum(axis=1).dropna()
        if len(port_rets) < 20:
            return np.nan
        return float(-np.percentile(port_rets, (1 - confidence) * 100))

    # ── Internal controls ────────────────────────────────────────────────────

    def _cap_positions(self, w: pd.Series) -> pd.Series:
        cap = self.max_position_wt
        clipped = w.clip(lower=-cap, upper=cap)
        breaches = (w.abs() > cap).sum()
        if breaches:
            logger.debug("Position cap: clipped %d positions to ±%.0f%%", breaches, cap*100)
        return clipped

    def _cap_sector(self, w: pd.Series) -> pd.Series:
        if not self.sector_map:
            return w
        w = w.copy()
        sectors = pd.Series(self.sector_map).reindex(w.index)
        for sec, grp in sectors.groupby(sectors):
            tickers = grp.index
            sec_wt = w[tickers].sum()
            if sec_wt > self.max_sector_wt:
                scale = self.max_sector_wt / sec_wt
                w[tickers] *= scale
                logger.debug("Sector cap: %s scaled by %.2f", sec, scale)
        return w

    def _cap_leverage(self, w: pd.Series) -> pd.Series:
        gross = w.abs().sum()
        if gross > self.max_gross_leverage and gross > 0:
            w = w * (self.max_gross_leverage / gross)
            logger.debug("Leverage cap: scaled from %.2f to %.2f", gross, self.max_gross_leverage)
        return w

    def _check_var(self, w: pd.Series, prices_history: pd.DataFrame) -> pd.Series:
        var = self.portfolio_var(w, prices_history)
        if np.isnan(var):
            return w
        if var > self.var_limit:
            scale = self.var_limit / var
            if self.hard_halt:
                w = w * scale
                logger.warning("VaR breach: %.1f%% > limit %.1f%%. Scaled by %.2f.",
                               var*100, self.var_limit*100, scale)
            else:
                logger.warning("VaR warning: portfolio 95%% VaR = %.1f%% > limit %.1f%%",
                               var*100, self.var_limit*100)
        return w

    def _cap_beta(
        self,
        w: pd.Series,
        prices_history: pd.DataFrame,
        market_returns: pd.Series,
    ) -> pd.Series:
        rets = prices_history.pct_change().iloc[-self.var_lookback:]
        mkt = market_returns.reindex(rets.index).dropna()
        betas = {}
        for col in rets.columns:
            s = rets[col].reindex(mkt.index).dropna()
            if len(s) > 20:
                b = np.cov(s, mkt.loc[s.index])[0, 1] / np.var(mkt.loc[s.index])
                betas[col] = b
        beta_s = pd.Series(betas)
        w_aligned = w.reindex(beta_s.index).fillna(0.0)
        port_beta = (w_aligned * beta_s).sum()
        if abs(port_beta) > self.max_beta:
            scale = self.max_beta / abs(port_beta)
            w = w * scale
            logger.warning("Beta cap: portfolio beta %.2f > %.2f. Scaled by %.2f.",
                           port_beta, self.max_beta, scale)
        return w

    # ── Summary ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "max_position_wt":    self.max_position_wt,
            "max_gross_leverage": self.max_gross_leverage,
            "max_sector_wt":      self.max_sector_wt,
            "max_dd_halt":        self.max_dd_halt,
            "dd_resume_threshold":self.dd_resume_threshold,
            "max_daily_loss":     self.max_daily_loss,
            "var_limit":          self.var_limit,
            "max_beta":           self.max_beta,
            "currently_halted":   self._halted,
            "halt_events":        len(self.halt_log),
        }
