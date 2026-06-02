"""
Practical trading diagnostics:
  - live_feedback_horizon : how long before live results are statistically meaningful
  - overfitting_risk      : degrees-of-freedom check on a backtest
  - weekly_data_from_daily: resample daily OHLCV to weekly (for weekly strategies)

These answer the practical questions:
  "How long until I know the strategy is working?"
  "Am I overfitting my backtest?"
"""

import numpy as np
import pandas as pd
from scipy import stats


def live_feedback_horizon(
    expected_sharpe: float,
    rebalance: str = "weekly",
    confidence: float = 0.95,
) -> dict:
    """
    Given an expected Sharpe ratio and rebalance frequency, tells you:
      - How many observations you need before results are statistically meaningful
      - How many months of live trading that translates to
      - What Sharpe you need to observe (at p=0.05) to be confident the edge is real

    expected_sharpe : annualised Sharpe you expect (be conservative — use deflated SR)
    rebalance       : 'daily', 'weekly', 'monthly'
    confidence      : e.g. 0.95 = 95% confidence (one-sided)
    """
    obs_per_year = {"daily": 252, "weekly": 52, "monthly": 12}[rebalance]
    sr_per_obs = expected_sharpe / np.sqrt(obs_per_year)
    z = stats.norm.ppf(confidence)                     # z-score for confidence level
    n_needed = int(np.ceil((z / sr_per_obs) ** 2))     # N such that SR*sqrt(N) > z
    months_needed = n_needed / obs_per_year * 12

    # Minimum observable Sharpe (annualised) to reject null at this N and confidence
    min_sr_obs = z / np.sqrt(n_needed) * np.sqrt(obs_per_year)

    return {
        "rebalance":          rebalance,
        "expected_ann_sharpe": round(expected_sharpe, 2),
        "obs_needed":         n_needed,
        "months_of_live_trading": round(months_needed, 1),
        "years_of_live_trading":  round(months_needed / 12, 1),
        "min_observable_sharpe":  round(min_sr_obs, 2),
        "note": (
            f"With {rebalance} rebalancing and an expected Sharpe of "
            f"{expected_sharpe:.2f}, you need ~{months_needed:.0f} months of "
            f"live trading before the result is statistically meaningful at "
            f"{int(confidence*100)}% confidence."
        ),
    }


def overfitting_risk(
    n_observations: int,
    n_free_params: int,
    n_trials_searched: int = 1,
) -> dict:
    """
    Flags overfitting risk based on:
      1. Degrees of freedom ratio (obs / params) — below 30 is risky
      2. Number of strategy variants tried (multiple-testing problem)

    n_observations     : independent observations in the backtest
                         (e.g. 60 months for 5yr monthly)
    n_free_params      : parameters that were tuned (e.g. fast+slow window = 2)
    n_trials_searched  : how many param combinations were tried in grid search

    Returns a risk rating and the Bonferroni-corrected significance threshold.
    """
    dof_ratio = n_observations / max(n_free_params, 1)
    # Bonferroni-corrected p-value threshold (to maintain overall 5% error rate)
    bonferroni_p = 0.05 / max(n_trials_searched, 1)
    bonferroni_t = stats.norm.ppf(1 - bonferroni_p)   # required t-stat

    if dof_ratio < 10:
        dof_risk = "HIGH — fewer than 10 obs per parameter. Results likely spurious."
    elif dof_ratio < 30:
        dof_risk = "MEDIUM — 10-30 obs per parameter. Walk-forward validation essential."
    else:
        dof_risk = "LOW — >30 obs per parameter. Still validate out-of-sample."

    if n_trials_searched > 50:
        trial_risk = f"HIGH — searched {n_trials_searched} combinations. Use Deflated Sharpe."
    elif n_trials_searched > 10:
        trial_risk = f"MEDIUM — searched {n_trials_searched} combinations. Be cautious."
    else:
        trial_risk = f"LOW — {n_trials_searched} trial(s). Standard validation sufficient."

    return {
        "n_observations":     n_observations,
        "n_free_params":      n_free_params,
        "dof_ratio":          round(dof_ratio, 1),
        "dof_risk":           dof_risk,
        "n_trials_searched":  n_trials_searched,
        "trial_risk":         trial_risk,
        "required_t_stat_bonferroni": round(bonferroni_t, 2),
        "rule_of_thumb":      (
            "Keep n_free_params ≤ n_observations/30. "
            "A monthly strategy with 5yr data (60 obs) should have ≤2 free params."
        ),
    }


def weekly_from_daily(df: pd.DataFrame, day: str = "FRI") -> pd.DataFrame:
    """
    Convert daily OHLCV to weekly bars (last trading day of each week).
    Standard for weekly-rebalancing strategies.
    """
    rule = f"W-{day}"
    weekly = pd.DataFrame({
        "open":   df["open"].resample(rule).first(),
        "high":   df["high"].resample(rule).max(),
        "low":    df["low"].resample(rule).min(),
        "close":  df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum(),
    }).dropna()
    return weekly


def print_feedback_table():
    """
    Print how long you need to live-trade before results are significant.

    Key insight: calendar time needed is INDEPENDENT of rebalancing frequency.
    The formula collapses to T_years = (z / SR_annualised)² — frequency cancels.
    What weekly rebalancing DOES give you:
      - More observation points (easier to spot a broken strategy early)
      - More frequent portfolio review (practical control)
    What it does NOT give you:
      - Faster statistical proof (same calendar time required)
      - Better results if your signals are designed for monthly horizons
    """
    print(f"\n{'='*72}")
    print("LIVE TRADING: MONTHS NEEDED BEFORE RESULTS ARE STATISTICALLY REAL")
    print(f"{'='*72}")
    print("(same calendar time regardless of frequency — see note below)\n")
    print(f"{'Expected Sharpe':>20} | {'Months needed':>15} | {'Years':>8}")
    print("-"*52)
    for sr in [0.3, 0.5, 0.75, 1.0, 1.5]:
        h = live_feedback_horizon(sr, "weekly")
        print(f"{sr:>20.2f} | {h['months_of_live_trading']:>15.0f} | {h['years_of_live_trading']:>8.1f}")
    print("\nNote: calendar time = (z/SR)² in years — frequency cancels out.")
    print("A Sharpe of 0.5 requires ~11 years of ANY frequency to reach t-stat>2.")
    print("Use realistic (deflated) Sharpe, not the backtest number.")
