"""
Professional backtest-evaluation tools — the part that separates a real edge
from a grid-search artifact.

Implements:
  - information_coefficient : rank correlation of signal vs forward return (skill)
  - t_stat_of_returns       : statistical significance of mean return
  - probabilistic_sharpe    : P(true Sharpe > benchmark)         (Bailey & Lopez de Prado)
  - deflated_sharpe_ratio   : PSR adjusted for number of trials  (Bailey & Lopez de Prado)
  - drawdown_duration       : longest underwater stretch

References
----------
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
- Grinold & Kahn, "Active Portfolio Management" (IC, breadth)
- Harvey, Liu & Zhu (2016): with multiple testing, demand t-stat > ~3
"""

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649


def information_coefficient(
    signal_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> dict:
    """
    Cross-sectional IC: each date, rank-correlate the signal with the NEXT
    period's return across stocks. A daily IC mean of 0.02-0.05 is genuinely good.

    signal_panel    : rows=dates, cols=tickers, values=signal (higher=more bullish)
    forward_returns : same shape, the return realised AFTER the signal date
    """
    ics = []
    for date in signal_panel.index:
        if date not in forward_returns.index:
            continue
        s = signal_panel.loc[date]
        f = forward_returns.loc[date]
        pair = pd.concat([s, f], axis=1).dropna()
        if len(pair) < 5:
            continue
        if method == "spearman":
            ic, _ = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        else:
            ic = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if not np.isnan(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "n": 0}
    return {
        "ic_mean": round(float(ics.mean()), 4),
        "ic_std": round(float(ics.std()), 4),
        "ic_ir": round(float(ics.mean() / ics.std()), 4) if ics.std() > 0 else np.nan,
        "hit_rate": round(float((ics > 0).mean()), 4),
        "n": len(ics),
    }


def t_stat_of_returns(returns: pd.Series) -> dict:
    """t-statistic of the mean daily return. |t|>2 is suggestive, >3 after
    multiple testing (Harvey-Liu-Zhu)."""
    r = returns.dropna()
    n = len(r)
    if n < 2 or r.std() == 0:
        return {"t_stat": np.nan, "p_value": np.nan, "n": n}
    t = r.mean() / (r.std() / np.sqrt(n))
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return {"t_stat": round(float(t), 3), "p_value": round(float(p), 5), "n": n}


def _sharpe_per_obs(returns: pd.Series) -> float:
    r = returns.dropna()
    return r.mean() / r.std() if r.std() > 0 else 0.0


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """
    PSR = P(true Sharpe > sr_benchmark), accounting for track-record length,
    skew and kurtosis. sr_benchmark is per-observation Sharpe (0 = "better than
    random"). Returns a probability in [0, 1].
    """
    r = returns.dropna()
    n = len(r)
    if n < 3 or r.std() == 0:
        return np.nan
    sr = _sharpe_per_obs(r)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # non-excess
    denom = np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr ** 2)
    psr = stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom)
    return round(float(psr), 4)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpe_std: float | None = None,
) -> dict:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado): the probability the strategy's
    Sharpe is genuinely > 0 AFTER accounting for the number of configurations tried.

    n_trials         : how many strategy/parameter variants you tested (be honest!)
    trial_sharpe_std : std-dev of per-observation Sharpe across those trials.
                       If None, a conservative default of 0.5/sqrt(n) heuristic is used.

    Returns dsr (probability) and the deflated benchmark sr0 it had to beat.
    """
    r = returns.dropna()
    n = len(r)
    if n < 3 or r.std() == 0:
        return {"dsr": np.nan, "sr0": np.nan}

    if trial_sharpe_std is None:
        # Heuristic: variability of Sharpe estimates ~ 1/sqrt(n) per obs
        trial_sharpe_std = 1.0 / np.sqrt(n)

    N = max(int(n_trials), 1)
    # Expected maximum of N independent N(0, trial_sharpe_std) Sharpe estimates
    z1 = stats.norm.ppf(1 - 1.0 / N)
    z2 = stats.norm.ppf(1 - 1.0 / (N * np.e))
    sr0 = trial_sharpe_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)

    dsr = probabilistic_sharpe_ratio(r, sr_benchmark=sr0)
    return {
        "dsr": dsr,
        "sr0": round(float(sr0), 4),
        "n_trials": N,
        "interpretation": (
            "strong (>0.95)" if (dsr or 0) > 0.95
            else "weak/insignificant" if (dsr or 0) < 0.90
            else "borderline"
        ),
    }


def drawdown_duration(equity_curve: pd.Series) -> dict:
    """Longest underwater period (in observations) and current drawdown."""
    eq = equity_curve.dropna()
    running_max = eq.cummax()
    underwater = eq < running_max
    # length of the longest consecutive underwater run
    longest, current = 0, 0
    for u in underwater:
        current = current + 1 if u else 0
        longest = max(longest, current)
    dd = (eq / running_max - 1)
    return {
        "max_dd": round(float(dd.min()), 4),
        "longest_underwater_obs": int(longest),
        "current_dd": round(float(dd.iloc[-1]), 4),
    }


def full_evaluation(result, n_trials: int = 1) -> pd.Series:
    """Convenience: bundle the headline + professional metrics for a result object
    that has .returns and .equity_curve (BacktestResult or PortfolioResult)."""
    out = dict(result.metrics)
    out.update(t_stat_of_returns(result.returns))
    out["psr_vs_0"] = probabilistic_sharpe_ratio(result.returns, 0.0)
    dsr = deflated_sharpe_ratio(result.returns, n_trials=n_trials)
    out["deflated_sharpe"] = dsr["dsr"]
    out["dsr_benchmark_sr0"] = dsr["sr0"]
    out.update(drawdown_duration(result.equity_curve))
    return pd.Series(out, name=getattr(result, "name", getattr(result, "strategy_name", "strategy")))
