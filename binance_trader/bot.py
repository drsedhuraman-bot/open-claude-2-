"""Automated trading bot engine."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

import config
from client import BinanceClient, Order, Position
from strategies import STRATEGIES, BaseStrategy, Signal

logger = logging.getLogger("binance_trader.bot")


# ── State ──────────────────────────────────────────────────────────────────────

class BotState(Enum):
    IDLE    = "IDLE"
    RUNNING = "RUNNING"
    PAUSED  = "PAUSED"
    STOPPED = "STOPPED"


@dataclass
class TradeRecord:
    symbol: str
    side: str
    qty: float
    price: float
    reason: str
    time: datetime = field(default_factory=datetime.utcnow)
    pnl_pct: float = 0.0
    order_id: int  = 0


@dataclass
class BotStats:
    total_trades:   int   = 0
    winning_trades: int   = 0
    losing_trades:  int   = 0
    total_pnl_pct:  float = 0.0
    start_time: datetime  = field(default_factory=datetime.utcnow)

    @property
    def win_rate(self) -> float:
        if not self.total_trades:
            return 0.0
        return self.winning_trades / self.total_trades * 100

    @property
    def uptime_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()


# ── Bot ────────────────────────────────────────────────────────────────────────

class TradingBot:
    """
    Polls Binance for new candles on a given interval, runs the chosen
    strategy, and places market orders with optional stop-loss / take-profit.
    """

    # Interval string → seconds between polls (poll slightly after close)
    _INTERVAL_SECONDS: dict[str, int] = {
        "1m": 62,  "3m": 183, "5m": 303, "15m": 903,
        "30m": 1803, "1h": 3603, "2h": 7203, "4h": 14403,
        "6h": 21603, "8h": 28803, "12h": 43203,
        "1d": 86403, "3d": 259203, "1w": 604803,
    }

    def __init__(
        self,
        client: BinanceClient,
        symbol: str | None = None,
        interval: str | None = None,
        strategy_name: str | None = None,
        on_signal: Callable[[TradeRecord], None] | None = None,
    ) -> None:
        self._client   = client
        self.symbol    = (symbol or config.BOT_SYMBOL).upper()
        self.interval  = interval or config.BOT_INTERVAL
        self.state     = BotState.IDLE
        self.stats     = BotStats()
        self.trades: list[TradeRecord] = []
        self.positions: dict[str, Position] = {}
        self._on_signal = on_signal
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        strat_cls = STRATEGIES.get(strategy_name or config.BOT_STRATEGY)
        if strat_cls is None:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(STRATEGIES.keys())}"
            )
        self.strategy: BaseStrategy = strat_cls()
        logger.info(
            "Bot initialized: symbol=%s  interval=%s  strategy=%s",
            self.symbol, self.interval, self.strategy.name,
        )

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.state == BotState.RUNNING:
            logger.warning("Bot is already running.")
            return
        self._stop_event.clear()
        self.stats = BotStats()
        self.state = BotState.RUNNING
        self._thread = threading.Thread(target=self._loop, daemon=True, name="bot-loop")
        self._thread.start()
        logger.info("Bot started.")

    def stop(self) -> None:
        self._stop_event.set()
        self.state = BotState.STOPPED
        logger.info("Bot stop requested.")

    def pause(self) -> None:
        if self.state == BotState.RUNNING:
            self.state = BotState.PAUSED
            logger.info("Bot paused.")

    def resume(self) -> None:
        if self.state == BotState.PAUSED:
            self.state = BotState.RUNNING
            logger.info("Bot resumed.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        sleep_secs = self._INTERVAL_SECONDS.get(self.interval, 60)
        while not self._stop_event.is_set():
            if self.state == BotState.RUNNING:
                try:
                    self._tick()
                except Exception as exc:
                    logger.error("Bot tick error: %s", exc, exc_info=True)
            # Sleep in small increments so stop/pause responds quickly
            for _ in range(sleep_secs):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _tick(self) -> None:
        df = self._client.get_klines(self.symbol, self.interval, limit=200)
        result = self.strategy.analyze(df)
        logger.info("Strategy signal: %s", result)

        current_price = df["close"].iloc[-1]
        self._check_sl_tp(current_price)

        if result.signal == Signal.BUY:
            self._handle_buy(current_price, result.reason)
        elif result.signal == Signal.SELL:
            self._handle_sell(current_price, result.reason)

    # ── Trade execution ───────────────────────────────────────────────────────

    def _handle_buy(self, price: float, reason: str) -> None:
        if self.symbol in self.positions:
            logger.debug("Already in position for %s, skipping BUY.", self.symbol)
            return
        if len(self.positions) >= config.BOT_MAX_OPEN_TRADES:
            logger.info("Max open trades (%d) reached.", config.BOT_MAX_OPEN_TRADES)
            return

        qty = config.BOT_TRADE_AMOUNT / price
        try:
            order = self._client.market_buy(self.symbol, qty)
        except Exception as e:
            logger.error("BUY order failed: %s", e)
            return

        sl = price * (1 - config.BOT_STOP_LOSS_PCT   / 100)
        tp = price * (1 + config.BOT_TAKE_PROFIT_PCT / 100)

        self.positions[self.symbol] = Position(
            symbol=self.symbol,
            entry_price=price,
            qty=order.executed_qty or qty,
            stop_loss=sl,
            take_profit=tp,
        )

        record = TradeRecord(
            symbol=self.symbol, side="BUY", qty=qty, price=price,
            reason=reason, order_id=order.order_id,
        )
        self._record_trade(record)

    def _handle_sell(self, price: float, reason: str) -> None:
        pos = self.positions.get(self.symbol)
        if not pos:
            logger.debug("No position for %s, skipping SELL.", self.symbol)
            return

        try:
            order = self._client.market_sell(self.symbol, pos.qty)
        except Exception as e:
            logger.error("SELL order failed: %s", e)
            return

        pnl = (price - pos.entry_price) / pos.entry_price * 100
        record = TradeRecord(
            symbol=self.symbol, side="SELL", qty=pos.qty, price=price,
            reason=reason, pnl_pct=pnl, order_id=order.order_id,
        )
        self._record_trade(record)
        self.stats.total_pnl_pct += pnl
        if pnl >= 0:
            self.stats.winning_trades += 1
        else:
            self.stats.losing_trades += 1
        del self.positions[self.symbol]

    def _check_sl_tp(self, current_price: float) -> None:
        pos = self.positions.get(self.symbol)
        if not pos:
            return
        if pos.stop_loss and current_price <= pos.stop_loss:
            logger.warning("Stop-loss triggered at %.4f (entry=%.4f)", current_price, pos.entry_price)
            self._handle_sell(current_price, f"Stop-loss hit at {current_price:.4f}")
        elif pos.take_profit and current_price >= pos.take_profit:
            logger.info("Take-profit triggered at %.4f (entry=%.4f)", current_price, pos.entry_price)
            self._handle_sell(current_price, f"Take-profit hit at {current_price:.4f}")

    def _record_trade(self, record: TradeRecord) -> None:
        self.trades.append(record)
        self.stats.total_trades += 1
        logger.info("Trade recorded: %s", record)
        if self._on_signal:
            try:
                self._on_signal(record)
            except Exception:
                pass

    # ── Info ──────────────────────────────────────────────────────────────────

    def status_dict(self) -> dict:
        pos = self.positions.get(self.symbol)
        return {
            "state":          self.state.value,
            "symbol":         self.symbol,
            "interval":       self.interval,
            "strategy":       self.strategy.name,
            "open_position":  pos is not None,
            "entry_price":    pos.entry_price if pos else None,
            "stop_loss":      pos.stop_loss   if pos else None,
            "take_profit":    pos.take_profit if pos else None,
            "total_trades":   self.stats.total_trades,
            "win_rate":       f"{self.stats.win_rate:.1f}%",
            "total_pnl":      f"{self.stats.total_pnl_pct:.2f}%",
            "uptime":         f"{self.stats.uptime_seconds:.0f}s",
        }
