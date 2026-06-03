"""
ML alpha combination — a cross-sectional return ranker (Gu, Kelly & Xiu 2020).

Instead of equal-weighting factor z-scores, train a gradient-boosting model to
predict each stock's forward relative return from its factor exposures. The
model learns NONLINEAR INTERACTIONS between factors that a linear blend misses
(e.g. "momentum only works when volatility is low").

Uses sklearn's HistGradientBoostingRegressor (the same histogram-based gradient
boosting algorithm as LightGBM). Trained with strict PURGING: at any prediction
date, the model only sees samples whose forward-return labels were fully realized
and embargoed BEFORE that date — no look-ahead.

Heavy regularization (shallow trees, high min-samples) per the 2023-2026 consensus
that ML in equities overfits easily given the low signal-to-noise.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class MLAlphaModel:
    def __init__(
        self,
        horizon: int = 21,          # forward-return label horizon (days)
        embargo: int = 21,          # gap between train labels and prediction date
        min_train_samples: int = 500,
        retrain_every: int = 3,     # retrain every N rebalance dates (speed)
        max_depth: int = 3,         # shallow trees → less overfitting
        learning_rate: float = 0.05,
        max_iter: int = 150,
        min_samples_leaf: int = 50,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ):
        self.horizon = horizon
        self.embargo = embargo
        self.min_train_samples = min_train_samples
        self.retrain_every = retrain_every
        self.model_params = dict(
            max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
            min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
            random_state=random_state,
        )
        self._model: HistGradientBoostingRegressor | None = None
        self._feature_names: list | None = None
        self._calls_since_train = 0

    def predict_alpha(
        self,
        feature_cache: dict,        # {date: factor_df (tickers × factors)}
        close_panel: pd.DataFrame,  # full close panel (for computing labels)
        current_date,
    ) -> pd.Series:
        """
        Train (or reuse) the ranker on all purged historical samples, then predict
        cross-sectional alpha scores for `current_date`.

        Returns a Series of predicted forward returns per ticker (the alpha).
        Falls back to equal-weight factor blend if not enough training data yet.
        """
        if current_date not in feature_cache:
            return pd.Series(dtype=float)

        current_features = feature_cache[current_date]
        if current_features.empty:
            return pd.Series(dtype=float)

        # ── Assemble purged training set ───────────────────────────────────
        X_list, y_list = [], []
        label_cutoff_idx = close_panel.index.get_loc(current_date) - self.embargo

        for date, fdf in feature_cache.items():
            if date >= current_date or fdf.empty:
                continue
            # label window must end before (current_date - embargo)
            try:
                di = close_panel.index.get_loc(date)
            except KeyError:
                continue
            label_end_idx = di + self.horizon
            if label_end_idx > label_cutoff_idx:
                continue  # PURGE: label overlaps embargo/prediction window

            # forward return label for this past date
            fwd = (close_panel.iloc[label_end_idx] / close_panel.iloc[di] - 1)
            # cross-sectionally demean → predict RELATIVE performance
            common = fdf.index.intersection(fwd.dropna().index)
            if len(common) < 5:
                continue
            y = fwd.loc[common]
            y = y - y.mean()
            X_list.append(fdf.loc[common])
            y_list.append(y)

        if not X_list:
            # not enough history → fall back to equal-weight factor blend
            return current_features.mean(axis=1)

        X_train = pd.concat(X_list).fillna(0.0)
        y_train = pd.concat(y_list)

        if len(X_train) < self.min_train_samples:
            return current_features.mean(axis=1)

        # ── Train (periodically, for speed) ────────────────────────────────
        need_train = (
            self._model is None
            or self._calls_since_train >= self.retrain_every
            or self._feature_names != list(current_features.columns)
        )
        if need_train:
            self._feature_names = list(X_train.columns)
            self._model = HistGradientBoostingRegressor(**self.model_params)
            self._model.fit(X_train.values, y_train.values)
            self._calls_since_train = 0
        else:
            self._calls_since_train += 1

        # ── Predict for current date ───────────────────────────────────────
        X_pred = current_features.reindex(columns=self._feature_names).fillna(0.0)
        preds = self._model.predict(X_pred.values)
        return pd.Series(preds, index=current_features.index)

    def feature_importance(self) -> pd.Series | None:
        """Permutation-free proxy: not directly available for HGB; returns None.
        Use sklearn.inspection.permutation_importance externally if needed."""
        return None
