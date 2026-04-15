"""Configuration management for Binance Trader."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _getenv(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _getenv_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _getenv_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _getenv_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


# ── API ────────────────────────────────────────────────────────────────────────
API_KEY    = _getenv("BINANCE_API_KEY")
API_SECRET = _getenv("BINANCE_API_SECRET")
USE_TESTNET = _getenv_bool("USE_TESTNET", True)

TESTNET_BASE_URL = "https://testnet.binance.vision/api"
TESTNET_WS_URL   = "wss://testnet.binance.vision/ws"

# ── Bot ────────────────────────────────────────────────────────────────────────
BOT_SYMBOL          = _getenv("BOT_SYMBOL", "BTCUSDT")
BOT_INTERVAL        = _getenv("BOT_INTERVAL", "1h")
BOT_STRATEGY        = _getenv("BOT_STRATEGY", "ma_crossover")
BOT_TRADE_AMOUNT    = _getenv_float("BOT_TRADE_AMOUNT", 10.0)
BOT_MAX_OPEN_TRADES = _getenv_int("BOT_MAX_OPEN_TRADES", 3)
BOT_STOP_LOSS_PCT   = _getenv_float("BOT_STOP_LOSS_PCT", 2.0)
BOT_TAKE_PROFIT_PCT = _getenv_float("BOT_TAKE_PROFIT_PCT", 4.0)

# ── MA Crossover ───────────────────────────────────────────────────────────────
MA_FAST_PERIOD = _getenv_int("MA_FAST_PERIOD", 9)
MA_SLOW_PERIOD = _getenv_int("MA_SLOW_PERIOD", 21)

# ── RSI ────────────────────────────────────────────────────────────────────────
RSI_PERIOD     = _getenv_int("RSI_PERIOD", 14)
RSI_OVERSOLD   = _getenv_float("RSI_OVERSOLD", 30.0)
RSI_OVERBOUGHT = _getenv_float("RSI_OVERBOUGHT", 70.0)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = _getenv("LOG_LEVEL", "INFO")
LOG_FILE  = _getenv("LOG_FILE", "trader.log")

VALID_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]


def setup_logging() -> logging.Logger:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    fmt   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if LOG_FILE:
        handlers.append(logging.FileHandler(LOG_FILE))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    return logging.getLogger("binance_trader")


def validate() -> list[str]:
    """Return a list of validation errors (empty = OK)."""
    errors: list[str] = []
    if not API_KEY or API_KEY == "your_api_key_here":
        errors.append("BINANCE_API_KEY is not set. Copy .env.example to .env and fill in your keys.")
    if not API_SECRET or API_SECRET == "your_api_secret_here":
        errors.append("BINANCE_API_SECRET is not set.")
    if BOT_INTERVAL not in VALID_INTERVALS:
        errors.append(f"BOT_INTERVAL '{BOT_INTERVAL}' is invalid. Choose from: {VALID_INTERVALS}")
    return errors
