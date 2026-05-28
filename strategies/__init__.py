from .base_strategy import BaseStrategy
from .moving_average import MACrossStrategy
from .rsi_strategy import RSIStrategy
from .bollinger_bands import BollingerBandsStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy

STRATEGY_REGISTRY = {
    "ma_cross":      MACrossStrategy,
    "rsi":           RSIStrategy,
    "bollinger":     BollingerBandsStrategy,
    "momentum":      MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
}
