"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Signal(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategyResult:
    signal: Signal
    reason: str
    indicators: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        ind = "  ".join(f"{k}={v:.4f}" for k, v in self.indicators.items())
        return f"[{self.signal.value}] {self.reason}  {ind}"


class BaseStrategy(ABC):
    """All strategies must implement this interface."""

    name: str = "base"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        """
        Analyze OHLCV dataframe and return a trading signal.

        :param df: DataFrame with columns: open_time, open, high, low, close, volume
        :return: StrategyResult
        """

    def _require_min_rows(self, df: pd.DataFrame, n: int) -> None:
        if len(df) < n:
            raise ValueError(
                f"Strategy '{self.name}' requires at least {n} candles, got {len(df)}."
            )
