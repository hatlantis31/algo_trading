"""
Performance metric calculations given a returns Series.
"""

import numpy as np
import pandas as pd


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Parameters
    ----------
    returns : daily strategy returns (arithmetic, not log)
    risk_free_rate : annual risk-free rate (e.g. 0.03 for 3%)

    Returns
    -------
    dict of scalar performance metrics
    """
    ann = 252
    r = returns.dropna()

    total_return   = (1 + r).prod() - 1
    ann_return     = (1 + total_return) ** (ann / max(len(r), 1)) - 1
    ann_vol        = r.std() * np.sqrt(ann)
    daily_rf       = (1 + risk_free_rate) ** (1 / ann) - 1
    excess         = r - daily_rf
    sharpe         = excess.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else 0.0

    cum             = (1 + r).cumprod()
    rolling_max     = cum.cummax()
    drawdown        = cum / rolling_max - 1
    max_drawdown    = drawdown.min()

    downside        = r[r < 0].std() * np.sqrt(ann)
    sortino         = (ann_return - risk_free_rate) / downside if downside > 0 else np.nan

    calmar          = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    win_rate        = (r > 0).mean()
    trades          = (r != 0).sum()

    return {
        "total_return":  round(total_return, 4),
        "ann_return":    round(ann_return, 4),
        "ann_volatility": round(ann_vol, 4),
        "sharpe_ratio":  round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio":  round(calmar, 4),
        "max_drawdown":  round(max_drawdown, 4),
        "win_rate":      round(win_rate, 4),
        "num_trades":    int(trades),
    }
