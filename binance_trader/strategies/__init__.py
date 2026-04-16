from .base import BaseStrategy, Signal, StrategyResult
from .ma_crossover import MACrossoverStrategy
from .consensus import ConsensusStrategy

# These imports assume the files rsi.py, macd.py, and scalp.py exist in the folder
try:
    from .rsi import RSIStrategy
    from .macd import MACDStrategy
    from .scalp import ScalpStrategy, ScalpEngine, PRESETS as SCALP_PRESETS
except ImportError:
    pass

STRATEGIES = {
    "ma_crossover": MACrossoverStrategy,
    "consensus":    ConsensusStrategy,
}

# Dynamically add others if they were successfully imported
if 'RSIStrategy' in locals():
    STRATEGIES["rsi"] = RSIStrategy
if 'MACDStrategy' in locals():
    STRATEGIES["macd"] = MACDStrategy
if 'ScalpStrategy' in locals():
    STRATEGIES["scalp"] = ScalpStrategy