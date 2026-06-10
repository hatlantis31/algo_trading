from .base_strategy import BaseStrategy
from .moving_average import MACrossStrategy
from .rsi_strategy import RSIStrategy
from .bollinger_bands import BollingerBandsStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy

# Single-asset, time-series strategies (per-ticker buy/sell signals)
STRATEGY_REGISTRY = {
    "ma_cross":      MACrossStrategy,
    "rsi":           RSIStrategy,
    "bollinger":     BollingerBandsStrategy,
    "momentum":      MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
}

# Cross-sectional, portfolio-level factor strategies (rank the universe, hold a
# diversified book). These are the professional/SOTA approach — use with
# backtesting.PortfolioEngine, not the single-asset BacktestEngine.
from .cross_sectional import (
    CrossSectionalMomentum, ShortTermReversal, LowVolatility,
    MultiFactor, EqualWeightBenchmark,
)

CROSS_SECTIONAL_REGISTRY = {
    "xs_momentum":   CrossSectionalMomentum,
    "xs_reversal":   ShortTermReversal,
    "xs_lowvol":     LowVolatility,
    "xs_multifactor": MultiFactor,
    "equal_weight":  EqualWeightBenchmark,
}

# Market-beating long-only strategy: multi-horizon momentum selection
# with equal-weight warm-up fallback.  Beats equal-weight benchmark
# on return, Sharpe, and max drawdown (see scripts/validate_market_beating.py).
from .long_only_factor import MarketBeatingStrategy
CROSS_SECTIONAL_REGISTRY["market_beater"] = MarketBeatingStrategy
