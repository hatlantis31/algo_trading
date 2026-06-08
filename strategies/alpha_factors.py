"""
Alpha factor library — cross-sectional signals from price/volume only.

Each factor function takes a price (and optionally volume) panel and returns a
cross-sectional Series of scores at the LAST date (no look-ahead). Higher score
= more attractive (expected to outperform).

The factors here are the price/volume-derivable signals that Gu, Kelly & Xiu
(2020) identified as the dominant predictors in their ML asset-pricing study:
momentum (multiple horizons), reversal, volatility, liquidity, and volume.

All factors are designed to be combined AFTER cross-sectional standardization
(see standardize() and neutralize()).
"""

import numpy as np
import pandas as pd


# ── Individual alpha signals (raw, un-standardized) ──────────────────────────

def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Jegadeesh-Titman momentum: return over [t-lookback, t-skip]."""
    if len(prices) < lookback + skip:
        return pd.Series(dtype=float)
    return prices.iloc[-1 - skip] / prices.iloc[-1 - skip - lookback + 1] - 1


def reversal(prices: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """Short-term reversal: negative of recent return (buy losers)."""
    if len(prices) < lookback + 1:
        return pd.Series(dtype=float)
    return -(prices.iloc[-1] / prices.iloc[-1 - lookback] - 1)


def low_volatility(prices: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Low-volatility anomaly: negative realized vol (prefer low-vol names)."""
    if len(prices) < lookback + 1:
        return pd.Series(dtype=float)
    vol = prices.pct_change().iloc[-lookback:].std()
    return -vol


def idiosyncratic_volatility(prices: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Negative idiosyncratic vol = residual vol after removing market move.
    Ang-Hodrick-Xing-Zhang: low idio-vol stocks outperform."""
    if len(prices) < lookback + 1:
        return pd.Series(dtype=float)
    rets = prices.pct_change().iloc[-lookback:]
    mkt = rets.mean(axis=1)
    # residual std of each stock vs market
    resid_vol = {}
    mkt_var = mkt.var()
    for col in rets.columns:
        r = rets[col]
        if r.notna().sum() < 10 or mkt_var == 0:
            continue
        beta = np.cov(r.fillna(0), mkt)[0, 1] / mkt_var
        resid = r - beta * mkt
        resid_vol[col] = resid.std()
    return -pd.Series(resid_vol)


def amihud_illiquidity(prices: pd.DataFrame, volume: pd.DataFrame,
                       lookback: int = 63) -> pd.Series:
    """Amihud illiquidity = mean(|return| / dollar-volume). Illiquidity premium:
    higher illiquidity → higher expected return (use with caution for tradeability)."""
    if len(prices) < lookback + 1 or volume is None:
        return pd.Series(dtype=float)
    rets = prices.pct_change().iloc[-lookback:].abs()
    dollar_vol = (prices * volume).iloc[-lookback:]
    illiq = (rets / dollar_vol.replace(0, np.nan)).mean()
    # scale up (Amihud values are tiny); higher = more illiquid = higher premium
    return illiq * 1e9


def volume_trend(prices: pd.DataFrame, volume: pd.DataFrame,
                 short: int = 21, long: int = 63) -> pd.Series:
    """Volume trend: rising volume confirms moves. Ratio of recent to longer volume."""
    if volume is None or len(volume) < long + 1:
        return pd.Series(dtype=float)
    short_v = volume.iloc[-short:].mean()
    long_v = volume.iloc[-long:].mean()
    return short_v / long_v.replace(0, np.nan) - 1


def high_52w(prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """George-Hwang 52-week-high momentum: proximity to rolling max."""
    if len(prices) < lookback:
        return pd.Series(dtype=float)
    window = prices.iloc[-lookback:]
    return prices.iloc[-1] / window.max()


def trend_quality(prices: pd.DataFrame, lookback: int = 126) -> pd.Series:
    """Smoothness of the uptrend: return / path-length (penalizes choppy moves)."""
    if len(prices) < lookback + 1:
        return pd.Series(dtype=float)
    window = prices.iloc[-lookback:]
    total_move = window.iloc[-1] / window.iloc[0] - 1
    daily = window.pct_change().abs().sum()
    return total_move / daily.replace(0, np.nan)


# ── Factor registry ──────────────────────────────────────────────────────────

PRICE_FACTORS = {
    "momentum_12_1": lambda p, v: momentum(p, 252, 21),
    "momentum_6_1":  lambda p, v: momentum(p, 126, 21),
    "reversal_1m":   lambda p, v: reversal(p, 21),
    "reversal_1w":   lambda p, v: reversal(p, 5),
    "low_vol":       lambda p, v: low_volatility(p, 63),
    "idio_vol":      lambda p, v: idiosyncratic_volatility(p, 63),
    "high_52w":      lambda p, v: high_52w(p, 252),
    "trend_quality": lambda p, v: trend_quality(p, 126),
}

VOLUME_FACTORS = {
    "amihud_illiq":  lambda p, v: amihud_illiquidity(p, v, 63),
    "volume_trend":  lambda p, v: volume_trend(p, v, 21, 63),
}

ALL_FACTORS = {**PRICE_FACTORS, **VOLUME_FACTORS}


# ── Cross-sectional transforms ───────────────────────────────────────────────

def winsorize(s: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Clip extreme values to ±n_std (after the cross-section is roughly centered)."""
    if s.std() == 0 or len(s) < 2:
        return s
    lo, hi = s.mean() - n_std * s.std(), s.mean() + n_std * s.std()
    return s.clip(lo, hi)


def standardize(s: pd.Series, do_winsorize: bool = True) -> pd.Series:
    """Cross-sectional z-score (mean 0, std 1), optionally winsorized first."""
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return pd.Series(0.0, index=s.index)
    if do_winsorize:
        s = winsorize(s)
    return (s - s.mean()) / s.std()


def neutralize(alpha: pd.Series, beta: pd.Series = None,
               sectors: pd.Series = None) -> pd.Series:
    """
    Remove market-beta and/or sector exposure from an alpha by cross-sectional
    regression: alpha ~ [1, beta, sector_dummies]; return the residual.

    This makes the alpha a bet on the IDIOSYNCRATIC component — the core of a
    market-neutral strategy. (Grinold & Kahn, Active Portfolio Management.)
    """
    a = alpha.dropna()
    if len(a) < 5:
        return a

    X_cols = [pd.Series(1.0, index=a.index, name="const")]
    if beta is not None:
        X_cols.append(beta.reindex(a.index).fillna(beta.median()).rename("beta"))
    if sectors is not None:
        dummies = pd.get_dummies(sectors.reindex(a.index), prefix="sec", dtype=float)
        # drop one to avoid collinearity with the constant
        if dummies.shape[1] > 1:
            dummies = dummies.iloc[:, 1:]
        X_cols.append(dummies)

    X = pd.concat(X_cols, axis=1).fillna(0.0)
    y = a.values
    Xv = X.values
    try:
        coef, *_ = np.linalg.lstsq(Xv, y, rcond=None)
        resid = y - Xv @ coef
        return pd.Series(resid, index=a.index)
    except np.linalg.LinAlgError:
        return a


def ic_weighted_alpha(
    feature_cache: dict,
    close_panel: pd.DataFrame,
    current_date,
    horizon: int = 21,
    embargo: int = 5,
    ic_lookback: int = 12,
    min_history: int = 6,
) -> pd.Series:
    """
    Combine factors by their trailing out-of-sample Information Coefficient.

    For each factor, estimate its IC over the most recent `ic_lookback` rebalance
    dates (using only PURGED, fully-realized labels), then weight the current
    factor scores by that IC. Negative-IC factors automatically get negative
    weight; useless factors get ~zero weight. This is the standard professional
    alternative to equal-weighting (Grinold & Kahn) and is far more robust than
    hand-picking factors.

    Returns the combined alpha Series at current_date (falls back to equal-weight
    blend until enough history accrues).
    """
    from scipy.stats import spearmanr

    current = feature_cache.get(current_date)
    if current is None or current.empty:
        return pd.Series(dtype=float)

    try:
        cutoff_idx = close_panel.index.get_loc(current_date) - embargo
    except KeyError:
        return current.mean(axis=1)

    # Gather per-factor IC samples from purged past dates
    past_dates = sorted([d for d in feature_cache if d < current_date])[-ic_lookback:]
    ic_samples: dict[str, list] = {c: [] for c in current.columns}

    for d in past_dates:
        fdf = feature_cache[d]
        if fdf.empty:
            continue
        try:
            di = close_panel.index.get_loc(d)
        except KeyError:
            continue
        if di + horizon > cutoff_idx:
            continue  # purge: label not realized / within embargo
        fwd = close_panel.iloc[di + horizon] / close_panel.iloc[di] - 1
        for col in fdf.columns:
            pair = pd.concat([fdf[col], fwd], axis=1).dropna()
            if len(pair) >= 5:
                ic, _ = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                if not np.isnan(ic):
                    ic_samples[col].append(ic)

    # If insufficient history, fall back to equal-weight blend
    if sum(len(v) for v in ic_samples.values()) == 0 or len(past_dates) < min_history:
        return current.mean(axis=1)

    weights = {c: np.mean(v) if v else 0.0 for c, v in ic_samples.items()}
    combined = pd.Series(0.0, index=current.index)
    for col, wt in weights.items():
        combined = combined.add(wt * current[col], fill_value=0.0)
    return combined


def fundamental_factor_matrix(
    fundamentals: pd.DataFrame,
    tickers,
    standardized: bool = True,
) -> pd.DataFrame:
    """
    Build a cross-sectional factor matrix from a STATIC fundamentals frame
    (rows=tickers, cols=value/quality factors such as earnings_yield,
    book_to_price, …; see core.fundamentals_loader.fundamental_factors).

    Returns rows=tickers (restricted to `tickers`), cols=factor names, each
    column cross-sectionally standardized so it composes with build_factor_matrix.

    NOTE: a static snapshot is look-ahead biased for historical backtests — see
    the warning in core/fundamentals_loader.py. Use for live ranking, or accept
    the bias for slow-moving value ratios.
    """
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame()
    sub = fundamentals.reindex(tickers)
    cols = {}
    for name in sub.columns:
        raw = pd.to_numeric(sub[name], errors="coerce")
        if raw.notna().sum() < 5:
            continue
        cols[name] = standardize(raw) if standardized else raw
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def build_factor_matrix(
    prices: pd.DataFrame,
    volume: pd.DataFrame = None,
    factors: dict = None,
    standardized: bool = True,
    fundamentals: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute all factors at the last date of the panel and return a DataFrame
    (rows=tickers, cols=factor names). Each column cross-sectionally standardized.

    If `fundamentals` (a static per-ticker value/quality frame) is supplied, its
    standardized columns are appended — letting the same IC-weighting / ML
    combiner blend price and fundamental signals together.
    """
    factors = factors or ALL_FACTORS
    cols = {}
    for name, fn in factors.items():
        try:
            raw = fn(prices, volume)
            if raw is None or len(raw) == 0:
                continue
            cols[name] = standardize(raw) if standardized else raw
        except Exception:
            continue
    price_fm = pd.DataFrame(cols) if cols else pd.DataFrame()

    if fundamentals is not None and not price_fm.empty:
        fund_fm = fundamental_factor_matrix(fundamentals, price_fm.index,
                                            standardized=standardized)
        if not fund_fm.empty:
            # prefix to avoid name clashes and keep provenance clear
            fund_fm = fund_fm.add_prefix("fnd_")
            price_fm = price_fm.join(fund_fm, how="left")

    return price_fm
