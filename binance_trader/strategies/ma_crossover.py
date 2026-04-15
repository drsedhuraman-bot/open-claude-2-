"""Moving-Average Crossover strategy.

BUY  when fast MA crosses above slow MA (golden cross).
SELL when fast MA crosses below slow MA (death cross).
HOLD otherwise.
"""

from __future__ import annotations

import pandas as pd

import config
from .base import BaseStrategy, Signal, StrategyResult


class MACrossoverStrategy(BaseStrategy):
    name = "ma_crossover"

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.fast = int(self.params.get("fast_period", config.MA_FAST_PERIOD))
        self.slow = int(self.params.get("slow_period", config.MA_SLOW_PERIOD))
        if self.fast >= self.slow:
            raise ValueError(f"fast_period ({self.fast}) must be < slow_period ({self.slow})")

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        self._require_min_rows(df, self.slow + 2)

        close = df["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()

        # Current and previous bar values
        fast_now  = fast_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2]
        slow_now  = slow_ma.iloc[-1]
        slow_prev = slow_ma.iloc[-2]

        indicators = {
            f"MA{self.fast}": fast_now,
            f"MA{self.slow}": slow_now,
            "price": close.iloc[-1],
        }

        # Golden cross: fast crosses above slow
        if fast_prev <= slow_prev and fast_now > slow_now:
            return StrategyResult(
                signal=Signal.BUY,
                reason=f"Golden cross: MA{self.fast} crossed above MA{self.slow}",
                indicators=indicators,
            )

        # Death cross: fast crosses below slow
        if fast_prev >= slow_prev and fast_now < slow_now:
            return StrategyResult(
                signal=Signal.SELL,
                reason=f"Death cross: MA{self.fast} crossed below MA{self.slow}",
                indicators=indicators,
            )

        trend = "bullish" if fast_now > slow_now else "bearish"
        return StrategyResult(
            signal=Signal.HOLD,
            reason=f"No crossover — trend is {trend}",
            indicators=indicators,
        )
