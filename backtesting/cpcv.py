"""
Combinatorial Purged Cross-Validation (CPCV) — López de Prado, AFML (2018).

Plain walk-forward gives ONE backtest path and leaks information across the
train/test boundary. CPCV fixes both problems:

  1. Split the timeline into N groups.
  2. Test on every combination of k groups; train on the rest.
  3. PURGE training samples whose label window overlaps a test group.
  4. EMBARGO a gap after each test group to prevent serial-correlation leakage.

This produces a DISTRIBUTION of backtest paths (not a single fragile number),
so you can see how stable the strategy is and compute a Deflated Sharpe across
the paths. Lower probability of backtest overfitting than walk-forward.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import comb


@dataclass
class CPCVResult:
    path_sharpes: list = field(default_factory=list)
    path_returns: list = field(default_factory=list)
    n_splits: int = 0
    n_test_groups: int = 0

    def summary(self) -> dict:
        s = np.array(self.path_sharpes)
        if len(s) == 0:
            return {}
        return {
            "n_paths":        len(s),
            "sharpe_mean":    round(float(s.mean()), 3),
            "sharpe_std":     round(float(s.std()), 3),
            "sharpe_min":     round(float(s.min()), 3),
            "sharpe_max":     round(float(s.max()), 3),
            "pct_paths_positive": round(float((s > 0).mean()), 3),
            # probability of backtest overfitting proxy: fraction of paths
            # whose Sharpe is below the median of a zero-skill distribution
            "sharpe_consistency": round(float((s > 0).mean()), 3),
        }


class CombinatorialPurgedCV:
    def __init__(self, n_splits: int = 6, n_test_groups: int = 2, embargo: int = 21):
        """
        n_splits       : number of contiguous time groups (N)
        n_test_groups  : groups held out for testing each path (k)
        embargo        : bars to skip after each test group
        Number of paths = C(N, k).
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo = embargo

    def n_paths(self) -> int:
        return int(comb(self.n_splits, self.n_test_groups))

    def split(self, index: pd.DatetimeIndex):
        """
        Yields (train_idx, test_idx) integer-position arrays for each combination,
        with purging + embargo applied.
        """
        n = len(index)
        group_bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        groups = [np.arange(group_bounds[i], group_bounds[i + 1])
                  for i in range(self.n_splits)]

        for test_combo in itertools.combinations(range(self.n_splits), self.n_test_groups):
            test_idx = np.concatenate([groups[g] for g in test_combo])
            test_set = set(test_idx.tolist())

            # Build train set = all groups not in test, with purge + embargo
            train_idx = []
            for g in range(self.n_splits):
                if g in test_combo:
                    continue
                for i in groups[g]:
                    # purge: drop train points within embargo of any test point
                    lo, hi = i - self.embargo, i + self.embargo
                    if any((j >= lo and j <= hi) for j in
                           range(max(0, lo), min(n, hi + 1)) if j in test_set):
                        continue
                    train_idx.append(i)
            yield np.array(sorted(train_idx)), np.array(sorted(test_idx))

    def evaluate(self, returns: pd.Series, periods_per_year: int = 252,
                 min_obs: int = None) -> CPCVResult:
        """
        Given a strategy's return series, compute the Sharpe on each CPCV
        test path. (Strategy is assumed already generated; this measures how
        consistent its performance is across different held-out periods.)

        periods_per_year : annualization base — 252 for daily, 12 for monthly.
        min_obs          : minimum test observations per path (default: 20 for
                           daily, 8 for coarser frequencies).
        """
        if min_obs is None:
            min_obs = 20 if periods_per_year >= 252 else 8
        result = CPCVResult(n_splits=self.n_splits, n_test_groups=self.n_test_groups)
        idx = returns.index
        for _, test_idx in self.split(idx):
            test_rets = returns.iloc[test_idx].dropna()
            if len(test_rets) < min_obs or test_rets.std() == 0:
                continue
            sharpe = test_rets.mean() / test_rets.std() * np.sqrt(periods_per_year)
            result.path_sharpes.append(float(sharpe))
            result.path_returns.append(test_rets)
        return result
