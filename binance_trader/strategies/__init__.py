"""Trading strategies package."""

from .base import BaseStrategy, Signal, StrategyResult
from .ma_crossover import MACrossoverStrategy
from .rsi import RSIStrategy
from .macd import MACDStrategy
from .consensus import ConsensusStrategy
from .scalp import ScalpStrategy, ScalpEngine, PRESETS as SCALP_PRESETS

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "ma_crossover": MACrossoverStrategy,
    "rsi":          RSIStrategy,
    "macd":         MACDStrategy,
    "consensus":    ConsensusStrategy,
    "scalp":        ScalpStrategy,
}

__all__ = [
    "BaseStrategy", "Signal", "StrategyResult",
    "MACrossoverStrategy", "RSIStrategy", "MACDStrategy", "ConsensusStrategy",
    "ScalpStrategy", "ScalpEngine", "SCALP_PRESETS",
    "STRATEGIES",
]
