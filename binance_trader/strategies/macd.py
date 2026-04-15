"""MACD (Moving Average Convergence Divergence) strategy.

BUY  when MACD line crosses above signal line (bullish crossover).
SELL when MACD line crosses below signal line (bearish crossover).
HOLD otherwise.
"""

from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, StrategyResult


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


class MACDStrategy(BaseStrategy):
    name = "macd"

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.fast_span   = int(self.params.get("macd_fast",   12))
        self.slow_span   = int(self.params.get("macd_slow",   26))
        self.signal_span = int(self.params.get("macd_signal",  9))

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        self._require_min_rows(df, self.slow_span + self.signal_span + 2)

        close      = df["close"]
        macd_line  = _ema(close, self.fast_span) - _ema(close, self.slow_span)
        signal_line = _ema(macd_line, self.signal_span)
        histogram  = macd_line - signal_line

        macd_now   = macd_line.iloc[-1]
        macd_prev  = macd_line.iloc[-2]
        sig_now    = signal_line.iloc[-1]
        sig_prev   = signal_line.iloc[-2]
        hist_now   = histogram.iloc[-1]

        indicators = {
            "MACD":      macd_now,
            "Signal":    sig_now,
            "Histogram": hist_now,
            "price":     close.iloc[-1],
        }

        # Bullish crossover: MACD crossed above signal
        if macd_prev <= sig_prev and macd_now > sig_now:
            return StrategyResult(
                signal=Signal.BUY,
                reason="MACD bullish crossover (MACD crossed above signal line)",
                indicators=indicators,
            )

        # Bearish crossover: MACD crossed below signal
        if macd_prev >= sig_prev and macd_now < sig_now:
            return StrategyResult(
                signal=Signal.SELL,
                reason="MACD bearish crossover (MACD crossed below signal line)",
                indicators=indicators,
            )

        trend = "bullish" if macd_now > sig_now else "bearish"
        return StrategyResult(
            signal=Signal.HOLD,
            reason=f"MACD no crossover — trend is {trend}  hist={hist_now:.4f}",
            indicators=indicators,
        )
