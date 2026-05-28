"""
All strategies inherit from BaseStrategy.
Subclasses must implement generate_signals(df) -> pd.Series of {-1, 0, 1}.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    name: str = "base"
    param_grid: dict = {}   # default hyperparameter search space

    def __init__(self, **params):
        self.params = {**self.param_grid, **params}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a Series aligned to df.index.
        Values: +1 = go long, -1 = go short, 0 = flat/hold.
        """

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"
