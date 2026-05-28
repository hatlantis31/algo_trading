import pandas as pd
from .base_strategy import BaseStrategy


class MACrossStrategy(BaseStrategy):
    """
    Dual SMA crossover.  Long when fast MA > slow MA, flat otherwise.
    """
    name = "ma_cross"
    param_grid = {"fast": 20, "slow": 50}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = self.params["fast"]
        slow = self.params["slow"]
        close = df["close"]
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        signal = pd.Series(0, index=df.index)
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma < slow_ma] = -1
        return signal.fillna(0)
