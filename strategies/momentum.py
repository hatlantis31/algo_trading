import pandas as pd
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Price momentum: long when N-day return is positive, short when negative.
    """
    name = "momentum"
    param_grid = {"lookback": 20, "threshold": 0.0}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lookback  = self.params["lookback"]
        threshold = self.params["threshold"]
        ret = df["close"].pct_change(lookback)
        signal = pd.Series(0, index=df.index)
        signal[ret > threshold]  = 1
        signal[ret < -threshold] = -1
        return signal.fillna(0)
