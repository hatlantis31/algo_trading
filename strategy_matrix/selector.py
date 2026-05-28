"""
StrategyMatrix
==============
Runs every strategy in the registry against every ticker you provide,
then builds a 2-D performance matrix (rows = strategies, columns = tickers).

Usage
-----
    matrix = StrategyMatrix(tickers=["AAPL", "MSFT"], source="yfinance")
    matrix.run()
    matrix.show("sharpe_ratio")        # heatmap
    best = matrix.select_best("sharpe_ratio", top_n=3)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from core.data_fetcher import get_data
from backtesting.backtest_engine import BacktestEngine
from strategies import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


class StrategyMatrix:
    def __init__(
        self,
        tickers: list[str],
        strategies: dict | None = None,
        source: str = "yfinance",
        period: str = "2y",
        initial_capital: float = 10_000.0,
        commission: float = 0.001,
        ibkr_connection=None,
        **data_kwargs,
    ):
        self.tickers = tickers
        self.strategies = strategies or {
            name: cls() for name, cls in STRATEGY_REGISTRY.items()
        }
        self.source = source
        self.period = period
        self.initial_capital = initial_capital
        self.commission = commission
        self.ibkr_connection = ibkr_connection
        self.data_kwargs = data_kwargs

        self._results: dict[str, dict[str, object]] = {}  # [strategy][ticker] -> BacktestResult
        self._matrix: dict[str, pd.DataFrame] = {}        # [metric] -> DataFrame

    # ── Core ────────────────────────────────────────────────────────────────

    def run(self):
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
        )
        data_cache: dict[str, pd.DataFrame] = {}

        for ticker in self.tickers:
            logger.info("Fetching data for %s …", ticker)
            try:
                data_cache[ticker] = get_data(
                    ticker,
                    source=self.source,
                    period=self.period,
                    ibkr_connection=self.ibkr_connection,
                    **self.data_kwargs,
                )
            except Exception as exc:
                logger.warning("Could not fetch %s: %s", ticker, exc)

        for strat_name, strategy in self.strategies.items():
            self._results[strat_name] = {}
            for ticker in self.tickers:
                if ticker not in data_cache:
                    continue
                try:
                    result = engine.run(strategy, data_cache[ticker], ticker=ticker)
                    self._results[strat_name][ticker] = result
                except Exception as exc:
                    logger.warning("Strategy %s on %s failed: %s", strat_name, ticker, exc)

        self._build_matrices()
        logger.info("Matrix complete: %d strategies × %d tickers", len(self.strategies), len(self.tickers))

    def _build_matrices(self):
        metrics = [
            "total_return", "ann_return", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "win_rate",
        ]
        for metric in metrics:
            rows = {}
            for strat_name, ticker_results in self._results.items():
                rows[strat_name] = {
                    ticker: result.metrics.get(metric, np.nan)
                    for ticker, result in ticker_results.items()
                }
            self._matrix[metric] = pd.DataFrame(rows).T  # rows=strategies, cols=tickers

    # ── Visualisation ───────────────────────────────────────────────────────

    def show(
        self,
        metric: str = "sharpe_ratio",
        figsize: tuple = (12, 6),
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = "RdYlGn",
    ):
        if not self._matrix:
            raise RuntimeError("Call .run() first.")
        df = self._matrix[metric]
        plt.figure(figsize=figsize)
        sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap, linewidths=0.5)
        plt.title(f"Strategy × Ticker matrix  –  {metric}")
        plt.tight_layout()
        plt.show()

    def summary_table(self, metric: str = "sharpe_ratio") -> pd.DataFrame:
        return self._matrix.get(metric, pd.DataFrame())

    # ── Selection ───────────────────────────────────────────────────────────

    def select_best(
        self,
        metric: str = "sharpe_ratio",
        top_n: int = 3,
        higher_is_better: bool = True,
    ) -> pd.DataFrame:
        """
        Returns the top_n (strategy, ticker) pairs ranked by the metric.
        """
        df = self._matrix.get(metric, pd.DataFrame())
        stacked = df.stack().reset_index()
        stacked.columns = ["strategy", "ticker", metric]
        stacked = stacked.dropna()
        return stacked.sort_values(metric, ascending=not higher_is_better).head(top_n)

    def select_for_ticker(self, ticker: str, metric: str = "sharpe_ratio") -> pd.Series:
        """Return all strategy scores for a given ticker, sorted best first."""
        df = self._matrix.get(metric, pd.DataFrame())
        if ticker not in df.columns:
            raise KeyError(f"Ticker {ticker!r} not in matrix.")
        return df[ticker].sort_values(ascending=False)

    def full_report(self) -> pd.DataFrame:
        """Wide table: one row per (strategy, ticker) with all metrics."""
        frames = []
        for strat_name, ticker_results in self._results.items():
            for ticker, result in ticker_results.items():
                row = {"strategy": strat_name, "ticker": ticker}
                row.update(result.metrics)
                frames.append(row)
        return pd.DataFrame(frames).set_index(["strategy", "ticker"])
