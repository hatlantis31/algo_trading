"""
Vectorised backtest engine.

Usage
-----
    engine = BacktestEngine(initial_capital=10_000, commission=0.001)
    result = engine.run(strategy, df)
    print(result.metrics)
    result.plot()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from .metrics import compute_metrics
from config.settings import BACKTEST_INITIAL_CAPITAL, BACKTEST_COMMISSION, BACKTEST_SLIPPAGE


@dataclass
class BacktestResult:
    strategy_name: str
    ticker: str
    equity_curve: pd.Series
    returns: pd.Series
    signals: pd.Series
    metrics: dict = field(default_factory=dict)

    def plot(self, figsize=(14, 5)):
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        self.equity_curve.plot(ax=axes[0], title=f"{self.strategy_name} – {self.ticker}", color="steelblue")
        axes[0].set_ylabel("Portfolio value (€)")
        axes[0].grid(True, alpha=0.3)

        # drawdown
        cum_max = self.equity_curve.cummax()
        dd = (self.equity_curve / cum_max - 1) * 100
        dd.plot(ax=axes[1], color="tomato", title="Drawdown %")
        axes[1].fill_between(dd.index, dd, 0, alpha=0.3, color="tomato")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def summary(self) -> pd.Series:
        return pd.Series(self.metrics, name=self.strategy_name)


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        commission: float = BACKTEST_COMMISSION,
        slippage: float = BACKTEST_SLIPPAGE,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self, strategy, df: pd.DataFrame, ticker: str = "asset") -> BacktestResult:
        """
        Vectorised single-asset backtest.
        strategy : instance of BaseStrategy
        df       : OHLCV DataFrame with DatetimeIndex
        """
        signals = strategy.generate_signals(df)
        # Position is the signal shifted by 1 day (execute next open)
        position = signals.shift(1).fillna(0)

        close = df["close"]
        daily_return = close.pct_change().fillna(0)

        # Strategy gross return
        strat_return = position * daily_return

        # Apply round-trip costs on position changes
        trade_flag = position.diff().abs().clip(upper=1)
        cost = trade_flag * (self.commission + self.slippage)
        net_return = strat_return - cost

        equity = self.initial_capital * (1 + net_return).cumprod()
        metrics = compute_metrics(net_return)

        return BacktestResult(
            strategy_name=strategy.name,
            ticker=ticker,
            equity_curve=equity,
            returns=net_return,
            signals=signals,
            metrics=metrics,
        )

    def run_walk_forward(
        self,
        strategy,
        df: pd.DataFrame,
        n_splits: int = 5,
        ticker: str = "asset",
    ) -> list[BacktestResult]:
        """Split df into n_splits folds; train is ignored (signal-based), test each fold."""
        size = len(df) // n_splits
        results = []
        for i in range(n_splits):
            fold_df = df.iloc[i * size: (i + 1) * size]
            result = self.run(strategy, fold_df, ticker=f"{ticker}_fold{i+1}")
            results.append(result)
        return results
