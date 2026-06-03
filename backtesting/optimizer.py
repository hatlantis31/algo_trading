"""
Mean-variance portfolio optimizer with market-neutral constraints.

Solves the constrained problem a market-neutral equity fund actually optimizes:

    maximize    αᵀw  −  (λ/2) wᵀΣw
    subject to  1ᵀw = 0          (dollar neutral)
                βᵀw = 0          (beta neutral)
                |wᵢ| ≤ cap       (position limits)
                Σ|wᵢ| = gross    (target gross leverage)

The equality-constrained part has a closed-form Lagrangian solution (fast, robust,
no iterative solver needed):

    w = (1/λ) Σ⁻¹ (α − Aᵀμ),   μ = (A Σ⁻¹ Aᵀ)⁻¹ A Σ⁻¹ α,   A = [1ᵀ; βᵀ]

Position caps and gross-leverage scaling are applied afterwards. This is the
"characteristic portfolio" of Grinold & Kahn, Active Portfolio Management.
"""

import numpy as np
import pandas as pd


def mean_variance_neutral(
    alpha: pd.Series,
    cov: pd.DataFrame,
    betas: pd.Series,
    gross_leverage: float = 1.0,
    position_cap: float = 0.05,
    beta_neutral: bool = True,
    dollar_neutral: bool = True,
    risk_aversion: float = 1.0,
    shrinkage: float = 0.5,
) -> pd.Series:
    """
    Returns optimal market-neutral weights (sum ≈ 0, gross ≈ gross_leverage).

    alpha          : expected-return scores per ticker (the combined alpha)
    cov            : covariance matrix (from RiskModel, shrunk)
    betas          : market beta per ticker
    gross_leverage : target sum of |weights| (1.0 = 100% gross, i.e. 50% long/50% short)
    position_cap   : max |weight| per name
    shrinkage      : 0..1 blend of the precision matrix toward (scaled) identity.
                     Raw mean-variance (shrinkage=1) is fragile — Σ⁻¹ amplifies
                     estimation error and naive MV underperforms equal-weight out
                     of sample (DeMiguel, Garlappi & Uppal 2009). shrinkage=0 gives
                     alpha-proportional (robust) weights; 0.5 is a balanced default.
    """
    # Align everything to the covariance universe
    tickers = cov.index
    a = alpha.reindex(tickers).fillna(0.0).values
    b = betas.reindex(tickers).fillna(1.0).values
    Sigma = cov.values
    n = len(tickers)

    try:
        prec_full = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        prec_full = np.linalg.pinv(Sigma)

    # Shrink precision toward scaled identity (robust optimization).
    # At shrinkage=0 → w ∝ α (robust); at shrinkage=1 → full mean-variance.
    avg_var = np.mean(np.diag(Sigma))
    identity_prec = np.eye(n) / avg_var if avg_var > 0 else np.eye(n)
    prec = shrinkage * prec_full + (1 - shrinkage) * identity_prec

    # Build constraint matrix A (rows = active constraints)
    rows = []
    if dollar_neutral:
        rows.append(np.ones(n))
    if beta_neutral:
        rows.append(b)

    if rows:
        A = np.vstack(rows)                      # (k, n)
        # μ = (A Σ⁻¹ Aᵀ)⁻¹ A Σ⁻¹ α
        A_prec = A @ prec                        # (k, n)
        M = A_prec @ A.T                         # (k, k)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        mu = M_inv @ (A_prec @ a)                # (k,)
        w = (1.0 / risk_aversion) * prec @ (a - A.T @ mu)
    else:
        w = (1.0 / risk_aversion) * prec @ a

    w = pd.Series(w, index=tickers)

    # Restore dollar-neutrality, scale to target gross, then enforce the position
    # cap LAST so it is a hard limit (gross may end slightly below target if the
    # cap binds — the cap is a risk constraint and takes priority).
    if dollar_neutral:
        w = w - w.mean()
    gross = w.abs().sum()
    if gross > 0:
        w = w * (gross_leverage / gross)
    w = w.clip(-position_cap, position_cap)

    return w


def long_only_optimized(
    alpha: pd.Series,
    cov: pd.DataFrame,
    top_q: float = 0.3,
    position_cap: float = 0.05,
    risk_aversion: float = 1.0,
) -> pd.Series:
    """
    Long-only variant: mean-variance over the top-quantile names, weights sum to 1.
    Useful as a benchmark vs the market-neutral book.
    """
    tickers = cov.index
    a = alpha.reindex(tickers).dropna()
    if a.empty:
        return pd.Series(dtype=float)

    k = max(1, int(len(a) * top_q))
    selected = a.sort_values(ascending=False).index[:k]

    sub_cov = cov.loc[selected, selected]
    sub_alpha = a.loc[selected].values
    try:
        prec = np.linalg.inv(sub_cov.values)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(sub_cov.values)

    w = (1.0 / risk_aversion) * prec @ sub_alpha
    w = np.clip(w, 0, None)                      # long-only
    w = pd.Series(w, index=selected)
    if w.sum() > 0:
        w = w / w.sum()
    w = w.clip(upper=position_cap)
    if w.sum() > 0:
        w = w / w.sum()
    return w.reindex(tickers).fillna(0.0)


def realized_neutrality(weights: pd.Series, betas: pd.Series) -> dict:
    """Diagnostic: check how dollar- and beta-neutral the weights actually are."""
    w = weights.fillna(0.0)
    b = betas.reindex(w.index).fillna(1.0)
    return {
        "net_exposure":  round(float(w.sum()), 4),        # ≈0 = dollar neutral
        "gross_leverage": round(float(w.abs().sum()), 4),
        "portfolio_beta": round(float((w * b).sum()), 4),  # ≈0 = beta neutral
        "n_long":  int((w > 1e-6).sum()),
        "n_short": int((w < -1e-6).sum()),
        "max_position": round(float(w.abs().max()), 4),
    }
