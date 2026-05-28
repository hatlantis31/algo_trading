import pandas as pd
from .base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Mean-reversion using Bollinger Bands.
    Long when price touches lower band, short when it touches upper band.
    """
    name = "bollinger"
    param_grid = {"period": 20, "std_dev": 2.0}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        period  = self.params["period"]
        std_dev = self.params["std_dev"]
        close   = df["close"]
        mid     = close.rolling(period).mean()
        std     = close.rolling(period).std()
        upper   = mid + std_dev * std
        lower   = mid - std_dev * std
        signal = pd.Series(0, index=df.index)
        signal[close <= lower] = 1
        signal[close >= upper] = -1
        return signal.fillna(0)
