"""RSI (Relative Strength Index) strategy.

BUY  when RSI crosses above oversold threshold (default 30).
SELL when RSI crosses below overbought threshold (default 70).
HOLD otherwise.
"""

from __future__ import annotations

import pandas as pd

import config
from .base import BaseStrategy, Signal, StrategyResult


def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


class RSIStrategy(BaseStrategy):
    name = "rsi"

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.period     = int(self.params.get("rsi_period",     config.RSI_PERIOD))
        self.oversold   = float(self.params.get("rsi_oversold",   config.RSI_OVERSOLD))
        self.overbought = float(self.params.get("rsi_overbought", config.RSI_OVERBOUGHT))

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        self._require_min_rows(df, self.period + 2)

        rsi = _calc_rsi(df["close"], self.period)
        rsi_now  = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-2]

        indicators = {
            "RSI": rsi_now,
            "oversold":   self.oversold,
            "overbought": self.overbought,
            "price": df["close"].iloc[-1],
        }

        # Crossed above oversold → BUY signal
        if rsi_prev <= self.oversold < rsi_now:
            return StrategyResult(
                signal=Signal.BUY,
                reason=f"RSI({self.period}) crossed above oversold ({self.oversold})",
                indicators=indicators,
            )

        # Crossed below overbought → SELL signal
        if rsi_prev >= self.overbought > rsi_now:
            return StrategyResult(
                signal=Signal.SELL,
                reason=f"RSI({self.period}) crossed below overbought ({self.overbought})",
                indicators=indicators,
            )

        zone = (
            "oversold"   if rsi_now < self.oversold  else
            "overbought" if rsi_now > self.overbought else
            "neutral"
        )
        return StrategyResult(
            signal=Signal.HOLD,
            reason=f"RSI({self.period})={rsi_now:.1f} — zone: {zone}",
            indicators=indicators,
        )
