import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


class RSIStrategy(BaseStrategy):
    """
    RSI mean-reversion.
    Long when RSI crosses up through oversold, short when it crosses down through overbought.
    """
    name = "rsi"
    param_grid = {"period": 14, "oversold": 30, "overbought": 70}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        period    = self.params["period"]
        oversold  = self.params["oversold"]
        overbought = self.params["overbought"]
        rsi = _rsi(df["close"], period)
        signal = pd.Series(0, index=df.index)
        signal[rsi < oversold]  = 1
        signal[rsi > overbought] = -1
        return signal.fillna(0)
