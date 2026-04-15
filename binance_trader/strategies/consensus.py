"""Consensus strategy — runs multiple strategies and votes.

A BUY/SELL signal is only fired when the majority (or all, configurable)
of the constituent strategies agree.  This dramatically reduces false signals.

Example usage:
    strat = ConsensusStrategy(params={
        "strategies": ["ma_crossover", "rsi", "macd"],
        "threshold": 0.67,      # 67% must agree (2-of-3)
    })
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, StrategyResult
from .ma_crossover import MACrossoverStrategy
from .rsi import RSIStrategy
from .macd import MACDStrategy

_ALL_STRATS: dict[str, type[BaseStrategy]] = {
    "ma_crossover": MACrossoverStrategy,
    "rsi":          RSIStrategy,
    "macd":         MACDStrategy,
}


class ConsensusStrategy(BaseStrategy):
    """Meta-strategy: weighted vote across multiple sub-strategies."""

    name = "consensus"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        strategy_names: list[str] = self.params.get("strategies", ["ma_crossover", "rsi", "macd"])
        self.threshold: float     = float(self.params.get("threshold", 0.67))

        self._strategies: list[BaseStrategy] = []
        for sname in strategy_names:
            cls = _ALL_STRATS.get(sname)
            if cls is None:
                raise ValueError(f"Unknown strategy in consensus: '{sname}'")
            self._strategies.append(cls(params))

        if not self._strategies:
            raise ValueError("ConsensusStrategy requires at least one sub-strategy.")

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        results: list[StrategyResult] = []
        for strat in self._strategies:
            try:
                results.append(strat.analyze(df))
            except ValueError:
                # Not enough data for this sub-strategy — treat as HOLD
                from .base import StrategyResult as SR
                results.append(SR(signal=Signal.HOLD, reason="insufficient data"))

        total  = len(results)
        buys   = sum(1 for r in results if r.signal == Signal.BUY)
        sells  = sum(1 for r in results if r.signal == Signal.SELL)

        buy_ratio  = buys  / total
        sell_ratio = sells / total

        # Collect all indicators
        combined_indicators: dict[str, float] = {}
        sub_signals = []
        for r, s in zip(results, self._strategies):
            sub_signals.append(f"{s.name}={r.signal.value}")
            for k, v in r.indicators.items():
                combined_indicators[f"{s.name}.{k}"] = v

        combined_indicators["buy_votes"]  = float(buys)
        combined_indicators["sell_votes"] = float(sells)
        combined_indicators["hold_votes"] = float(total - buys - sells)

        vote_summary = " | ".join(sub_signals)

        if buy_ratio >= self.threshold:
            return StrategyResult(
                signal=Signal.BUY,
                reason=f"Consensus BUY ({buys}/{total} agree)  [{vote_summary}]",
                indicators=combined_indicators,
            )
        if sell_ratio >= self.threshold:
            return StrategyResult(
                signal=Signal.SELL,
                reason=f"Consensus SELL ({sells}/{total} agree)  [{vote_summary}]",
                indicators=combined_indicators,
            )
        return StrategyResult(
            signal=Signal.HOLD,
            reason=f"No consensus (buy={buys} sell={sells} hold={total-buys-sells}/{total})  [{vote_summary}]",
            indicators=combined_indicators,
        )
