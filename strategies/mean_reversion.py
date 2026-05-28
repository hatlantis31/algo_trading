import pandas as pd
from .base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Z-score mean reversion.  Trade when price deviates significantly from its rolling mean.
    """
    name = "mean_reversion"
    param_grid = {"lookback": 30, "z_entry": 1.5, "z_exit": 0.0}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lookback = self.params["lookback"]
        z_entry  = self.params["z_entry"]
        close    = df["close"]
        mean     = close.rolling(lookback).mean()
        std      = close.rolling(lookback).std()
        zscore   = (close - mean) / std
        signal = pd.Series(0, index=df.index)
        signal[zscore < -z_entry] = 1
        signal[zscore >  z_entry] = -1
        return signal.fillna(0)
