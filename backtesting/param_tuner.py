"""
Grid-search parameter tuner for any BaseStrategy.

Usage
-----
    from backtesting.param_tuner import ParamTuner
    from strategies import MACrossStrategy

    grid = {"fast": [5, 10, 20], "slow": [30, 50, 100, 200]}
    tuner = ParamTuner(MACrossStrategy, grid, df, metric="sharpe_ratio")
    tuner.run()
    print(tuner.best_params())
    tuner.results_table()
    tuner.plot_heatmap("fast", "slow")
"""

import itertools
import warnings
from typing import Type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from strategies.base_strategy import BaseStrategy
from backtesting.backtest_engine import BacktestEngine
from config.settings import BACKTEST_INITIAL_CAPITAL, BACKTEST_COMMISSION, BACKTEST_SLIPPAGE


class ParamTuner:
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        param_grid: dict,
        df: pd.DataFrame,
        metric: str = "sharpe_ratio",
        higher_is_better: bool = True,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        commission: float = BACKTEST_COMMISSION,
        slippage: float = BACKTEST_SLIPPAGE,
        walk_forward_splits: int = 0,   # 0 = no WF, >0 = WF folds
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.df = df
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.engine = BacktestEngine(initial_capital, commission, slippage)
        self.wf_splits = walk_forward_splits
        self._records: list[dict] = []

    # ── Core ────────────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> "ParamTuner":
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(itertools.product(*values))

        if verbose:
            print(f"Grid search: {len(combos)} combinations × "
                  f"{'WF ' + str(self.wf_splits) + ' folds' if self.wf_splits else 'full backtest'}")

        self._records.clear()
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                strategy = self.strategy_class(**params)
            except Exception:
                continue

            if self.wf_splits > 0:
                folds = self.engine.run_walk_forward(strategy, self.df, n_splits=self.wf_splits)
                fold_metrics = [f.metrics[self.metric] for f in folds]
                score = float(np.mean(fold_metrics))
                score_std = float(np.std(fold_metrics))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self.engine.run(strategy, self.df)
                score = result.metrics.get(self.metric, np.nan)
                score_std = np.nan
                all_metrics = result.metrics

            record = {**params, self.metric: score, f"{self.metric}_std": score_std}
            if self.wf_splits == 0:
                record.update({k: v for k, v in all_metrics.items() if k != self.metric})
            self._records.append(record)

        if verbose:
            best = self.best_params()
            print(f"Best params: {best}  →  {self.metric}={self._best_score():.4f}")
        return self

    # ── Accessors ────────────────────────────────────────────────────────────

    def results_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self._records)
        return df.sort_values(self.metric, ascending=not self.higher_is_better).reset_index(drop=True)

    def best_params(self) -> dict:
        if not self._records:
            raise RuntimeError("Call .run() first.")
        df = pd.DataFrame(self._records)
        idx = df[self.metric].idxmax() if self.higher_is_better else df[self.metric].idxmin()
        keys = list(self.param_grid.keys())
        return {k: df.loc[idx, k] for k in keys}

    def _best_score(self) -> float:
        df = pd.DataFrame(self._records)
        return df[self.metric].max() if self.higher_is_better else df[self.metric].min()

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_heatmap(
        self,
        x_param: str,
        y_param: str,
        figsize: tuple = (10, 7),
        cmap: str = "RdYlGn",
        annot: bool = True,
    ):
        """2-D heatmap for two parameters (all others held at best value)."""
        if not self._records:
            raise RuntimeError("Call .run() first.")

        best = self.best_params()
        df = pd.DataFrame(self._records)

        # Fix all other params at best value
        other_params = [k for k in self.param_grid if k not in (x_param, y_param)]
        mask = pd.Series(True, index=df.index)
        for k in other_params:
            mask &= df[k] == best[k]
        df = df[mask]

        pivot = df.pivot(index=y_param, columns=x_param, values=self.metric)
        pivot = pivot.sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot, annot=annot, fmt=".2f", cmap=cmap,
            linewidths=0.4, ax=ax,
            cbar_kws={"label": self.metric},
        )
        ax.set_title(
            f"{self.strategy_class.__name__}  –  {self.metric} heatmap\n"
            f"({', '.join(f'{k}={best[k]}' for k in other_params) or 'no fixed params'})"
        )
        plt.tight_layout()
        plt.show()

    def plot_top_n(self, n: int = 10, figsize: tuple = (12, 5)):
        """Bar chart of top-N parameter combos."""
        top = self.results_table().head(n)
        labels = [
            " | ".join(f"{k}={row[k]}" for k in self.param_grid)
            for _, row in top.iterrows()
        ]
        vals = top[self.metric].values

        fig, ax = plt.subplots(figsize=figsize)
        colours = ["steelblue" if v >= 0 else "tomato" for v in vals]
        ax.barh(range(len(vals)), vals, color=colours)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(self.metric)
        ax.set_title(f"Top {n} parameter combinations – {self.strategy_class.__name__}")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()
