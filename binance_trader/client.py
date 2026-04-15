"""Binance API client wrapper with testnet support."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

import config

logger = logging.getLogger("binance_trader.client")


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class Ticker:
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    change_pct_24h: float
    high_24h: float
    low_24h: float


@dataclass
class Balance:
    asset: str
    free: float
    locked: float

    @property
    def total(self) -> float:
        return self.free + self.locked


@dataclass
class Order:
    order_id: int
    symbol: str
    side: str        # BUY / SELL
    order_type: str  # LIMIT / MARKET / STOP_LOSS_LIMIT
    status: str
    price: float
    qty: float
    executed_qty: float
    time: datetime
    stop_price: float = 0.0

    @property
    def fill_pct(self) -> float:
        return (self.executed_qty / self.qty * 100) if self.qty else 0.0


@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: float
    side: str = "LONG"
    stop_loss: float = 0.0
    take_profit: float = 0.0
    open_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def pnl_pct(self, current_price: float = 0.0) -> float:
        if not self.entry_price or not current_price:
            return 0.0
        if self.side == "LONG":
            return (current_price - self.entry_price) / self.entry_price * 100
        return (self.entry_price - current_price) / self.entry_price * 100


# ── Client ─────────────────────────────────────────────────────────────────────

class BinanceClient:
    """Thread-safe Binance REST API wrapper."""

    def __init__(self) -> None:
        kwargs: dict[str, Any] = {
            "api_key": config.API_KEY,
            "api_secret": config.API_SECRET,
            "tld": "com",
        }
        if config.USE_TESTNET:
            kwargs["testnet"] = True
            logger.info("Using Binance TESTNET")
        else:
            logger.info("Using Binance MAINNET")

        self._client = Client(**kwargs)
        self._exchange_info: dict[str, Any] = {}

    # ── Market data ───────────────────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> Ticker:
        try:
            t = self._client.get_ticker(symbol=symbol.upper())
            return Ticker(
                symbol=t["symbol"],
                price=float(t["lastPrice"]),
                bid=float(t["bidPrice"]),
                ask=float(t["askPrice"]),
                volume_24h=float(t["volume"]),
                change_pct_24h=float(t["priceChangePercent"]),
                high_24h=float(t["highPrice"]),
                low_24h=float(t["lowPrice"]),
            )
        except BinanceAPIException as e:
            logger.error("get_ticker(%s) failed: %s", symbol, e)
            raise

    def get_price(self, symbol: str) -> float:
        return float(self._client.get_symbol_ticker(symbol=symbol.upper())["price"])

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Return OHLCV dataframe sorted oldest→newest."""
        raw = self._client.get_klines(
            symbol=symbol.upper(),
            interval=interval,
            limit=limit,
        )
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        return df.sort_values("open_time").reset_index(drop=True)

    def get_orderbook(self, symbol: str, limit: int = 10) -> dict[str, list]:
        book = self._client.get_order_book(symbol=symbol.upper(), limit=limit)
        return {
            "bids": [[float(p), float(q)] for p, q in book["bids"]],
            "asks": [[float(p), float(q)] for p, q in book["asks"]],
        }

    def get_top_symbols(self, quote_asset: str = "USDT", top_n: int = 20) -> list[str]:
        tickers = self._client.get_ticker()
        filtered = [t for t in tickers if t["symbol"].endswith(quote_asset)]
        filtered.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
        return [t["symbol"] for t in filtered[:top_n]]

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balances(self, non_zero_only: bool = True) -> list[Balance]:
        info = self._client.get_account()
        balances = [
            Balance(
                asset=b["asset"],
                free=float(b["free"]),
                locked=float(b["locked"]),
            )
            for b in info["balances"]
        ]
        if non_zero_only:
            balances = [b for b in balances if b.total > 0]
        return balances

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        kwargs = {}
        if symbol:
            kwargs["symbol"] = symbol.upper()
        raw = self._client.get_open_orders(**kwargs)
        return [self._parse_order(o) for o in raw]

    def get_order_history(self, symbol: str, limit: int = 20) -> list[Order]:
        raw = self._client.get_all_orders(symbol=symbol.upper(), limit=limit)
        return [self._parse_order(o) for o in raw]

    # ── Trading ───────────────────────────────────────────────────────────────

    def market_buy(self, symbol: str, quantity: float) -> Order:
        return self._place_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            order_type=Client.ORDER_TYPE_MARKET,
            quantity=quantity,
        )

    def market_sell(self, symbol: str, quantity: float) -> Order:
        return self._place_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            order_type=Client.ORDER_TYPE_MARKET,
            quantity=quantity,
        )

    def limit_buy(self, symbol: str, quantity: float, price: float) -> Order:
        return self._place_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            order_type=Client.ORDER_TYPE_LIMIT,
            quantity=quantity,
            price=price,
            timeInForce=Client.TIME_IN_FORCE_GTC,
        )

    def limit_sell(self, symbol: str, quantity: float, price: float) -> Order:
        return self._place_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            order_type=Client.ORDER_TYPE_LIMIT,
            quantity=quantity,
            price=price,
            timeInForce=Client.TIME_IN_FORCE_GTC,
        )

    def stop_loss_sell(self, symbol: str, quantity: float, stop_price: float, limit_price: float) -> Order:
        return self._place_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            order_type=Client.ORDER_TYPE_STOP_LOSS_LIMIT,
            quantity=quantity,
            price=limit_price,
            stopPrice=stop_price,
            timeInForce=Client.TIME_IN_FORCE_GTC,
        )

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        try:
            self._client.cancel_order(symbol=symbol.upper(), orderId=order_id)
            logger.info("Cancelled order %d for %s", order_id, symbol)
            return True
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error("cancel_order failed: %s", e)
            return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _place_order(self, symbol: str, side: str, order_type: str, quantity: float, **kwargs) -> Order:
        try:
            qty_str = self._format_quantity(symbol, quantity)
            resp = self._client.create_order(
                symbol=symbol.upper(),
                side=side,
                type=order_type,
                quantity=qty_str,
                **kwargs,
            )
            logger.info("Order placed: %s %s %s qty=%s", side, order_type, symbol, qty_str)
            return self._parse_order(resp)
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error("_place_order failed: %s", e)
            raise

    def _parse_order(self, raw: dict) -> Order:
        return Order(
            order_id=raw["orderId"],
            symbol=raw["symbol"],
            side=raw["side"],
            order_type=raw["type"],
            status=raw["status"],
            price=float(raw.get("price") or 0),
            qty=float(raw.get("origQty") or raw.get("quantity") or 0),
            executed_qty=float(raw.get("executedQty") or 0),
            time=datetime.fromtimestamp(raw["time"] / 1000) if "time" in raw else datetime.utcnow(),
            stop_price=float(raw.get("stopPrice") or 0),
        )

    def _format_quantity(self, symbol: str, quantity: float) -> str:
        if not self._exchange_info:
            self._exchange_info = self._client.get_exchange_info()
        for s in self._exchange_info["symbols"]:
            if s["symbol"] == symbol.upper():
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        step = f["stepSize"].rstrip("0").rstrip(".")
                        decimals = len(step.split(".")[-1]) if "." in step else 0
                        return f"{quantity:.{decimals}f}"
        return str(quantity)

    def server_time(self) -> datetime:
        ts = self._client.get_server_time()["serverTime"]
        return datetime.fromtimestamp(ts / 1000)

    def ping(self) -> bool:
        try:
            self._client.ping()
            return True
        except Exception:
            return False
