"""
MetaTrader5 / Forex-Metals-Indices client.

Live mode  (Windows + MT5 terminal open):
    Uses the official MetaTrader5 Python package — real bid/ask, live trading.

Fallback mode (any OS, no MT5):
    Uses yfinance for OHLCV data — signals, backtesting, paper analysis.
    Trading methods raise NotImplementedError (need MT5 terminal).

Interface mirrors BinanceClient so the rest of the codebase is broker-agnostic.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger("binance_trader.mt5")

# ── Symbol catalogue ───────────────────────────────────────────────────────────

SYMBOL_GROUPS: dict[str, list[str]] = {
    "FX Majors":  ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"],
    "FX Minors":  ["EURGBP","EURJPY","GBPJPY","EURCHF","AUDJPY","CADJPY","EURAUD","GBPAUD","NZDJPY","EURCAD"],
    "Metals":     ["XAUUSD","XAGUSD","XPTUSD","XPDUSD"],
    "Indices":    ["US30","NAS100","SPX500","GER40","UK100","JP225","AUS200","HK50"],
    "Energies":   ["USOIL","UKOIL","NATGAS"],
    "Crypto-FX":  ["BTCUSD","ETHUSD","LTCUSD","XRPUSD","SOLUSD","BNBUSD"],
}

ALL_FX_SYMBOLS: list[str] = [s for syms in SYMBOL_GROUPS.values() for s in syms]

# MT5 timeframe integer mapping
_MT5_TF_VAL: dict[str, int] = {
    "1m": 1,   "2m": 2,   "3m": 3,   "5m": 5,
    "10m": 10, "15m": 15, "30m": 30, "1h": 60,
    "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440,"1w": 10080,
}

# yfinance interval / period
_YF_INT: dict[str, str] = {
    "1m":"1m",  "5m":"5m",  "15m":"15m", "30m":"30m",
    "1h":"1h",  "4h":"1h",  "1d":"1d",   "1w":"1wk",
}
_YF_PER: dict[str, str] = {
    "1m":"5d",  "5m":"20d", "15m":"40d", "30m":"40d",
    "1h":"60d", "4h":"60d", "1d":"2y",   "1w":"5y",
}


def _to_yf(symbol: str) -> str:
    """Map MT5-style symbol → yfinance ticker."""
    s = symbol.upper().strip()

    # Metals (spot rates via futures)
    _metals = {
        "XAUUSD": "GC=F",   # Gold
        "XAGUSD": "SI=F",   # Silver
        "XPTUSD": "PL=F",   # Platinum
        "XPDUSD": "PA=F",   # Palladium
    }
    if s in _metals:
        return _metals[s]

    # Standard 6-char forex pair → append =X
    if len(s) == 6 and s.isalpha():
        return s + "=X"

    _idx = {
        "US30":"^DJI","NAS100":"^NDX","SPX500":"^GSPC",
        "GER40":"^GDAXI","UK100":"^FTSE","JP225":"^N225",
        "AUS200":"^AXJO","HK50":"^HSI",
    }
    if s in _idx:
        return _idx[s]

    _ene = {"USOIL":"CL=F","UKOIL":"BZ=F","NATGAS":"NG=F"}
    if s in _ene:
        return _ene[s]

    _cr = {
        "BTCUSD":"BTC-USD","ETHUSD":"ETH-USD","LTCUSD":"LTC-USD",
        "XRPUSD":"XRP-USD","SOLUSD":"SOL-USD","BNBUSD":"BNB-USD",
    }
    if s in _cr:
        return _cr[s]

    return s


# ── Shared data models (mirror BinanceClient's types) ─────────────────────────

@dataclass
class FxTicker:
    symbol:         str
    price:          float
    bid:            float
    ask:            float
    volume_24h:     float
    change_pct_24h: float
    high_24h:       float
    low_24h:        float
    spread_pips:    float = 0.0


@dataclass
class FxBalance:
    asset:  str
    free:   float
    locked: float = 0.0

    @property
    def total(self) -> float:
        return self.free + self.locked


@dataclass
class FxOrder:
    order_id:     int
    symbol:       str
    side:         str        # BUY | SELL
    order_type:   str        # MARKET | LIMIT | STOP
    status:       str
    price:        float
    qty:          float      # lots for FX
    executed_qty: float
    time:         datetime
    stop_price:   float = 0.0

    @property
    def fill_pct(self) -> float:
        return (self.executed_qty / self.qty * 100) if self.qty else 0.0


# ── MT5 Client ─────────────────────────────────────────────────────────────────

class MT5Client:
    """
    Broker-agnostic forex/metals/indices client.

    Attributes:
        is_live: True when connected to a real MT5 terminal.
    """

    def __init__(
        self,
        login:    int = 0,
        password: str = "",
        server:   str = "",
    ) -> None:
        self._login    = int(login) if login else 0
        self._password = password
        self._server   = server
        self._mt5      = None
        self._live     = False
        self._acct:    dict[str, Any] = {}

        if platform.system() == "Windows":
            self._try_mt5()
        else:
            logger.info("MT5Client: Linux/Mac — using yfinance data fallback (no live trading)")

    # ── Connection ─────────────────────────────────────────────────────────────

    def _try_mt5(self) -> None:
        try:
            import MetaTrader5 as mt5
            kw: dict[str, Any] = {}
            if self._login:
                kw.update(login=self._login, password=self._password, server=self._server)
            if mt5.initialize(**kw):
                self._mt5  = mt5
                self._live = True
                info = mt5.account_info()
                if info:
                    self._acct = info._asdict()
                logger.info(
                    "MT5 connected  server=%s  login=%s  balance=%.2f %s",
                    self._acct.get("server","?"), self._acct.get("login","?"),
                    self._acct.get("balance",0), self._acct.get("currency",""),
                )
            else:
                logger.warning("MT5 initialize failed: %s", mt5.last_error())
        except ImportError:
            logger.info("MetaTrader5 package not found — install on Windows: pip install MetaTrader5")

    def connect(self, login: int, password: str, server: str) -> bool:
        """Re-connect / switch account credentials."""
        self._login    = int(login)
        self._password = password
        self._server   = server
        if self._mt5:
            self._mt5.shutdown()
            self._live = False
        self._try_mt5()
        return self._live

    def disconnect(self) -> None:
        if self._mt5:
            self._mt5.shutdown()
        self._live = False

    @property
    def is_live(self) -> bool:
        return self._live

    def ping(self) -> bool:
        if self._live:
            return self._mt5 and self._mt5.terminal_info() is not None
        try:
            import yfinance as yf
            df = yf.download("EURUSD=X", period="1d", interval="1h",
                             progress=False, auto_adjust=True)
            return not df.empty
        except Exception:
            return False

    def status(self) -> dict:
        if self._live:
            info = self._mt5.terminal_info()
            return {
                "connected": True,
                "mode": "live",
                "server": self._acct.get("server",""),
                "login": self._acct.get("login",""),
                "balance": self._acct.get("balance",0),
                "equity": self._acct.get("equity",0),
                "currency": self._acct.get("currency",""),
                "leverage": self._acct.get("leverage",0),
                "company": self._acct.get("company",""),
            }
        return {"connected": True, "mode": "yfinance-fallback",
                "note": "Data-only mode. Connect MT5 terminal on Windows for live trading."}

    # ── Market data ────────────────────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> FxTicker:
        if self._live:
            return self._mt5_ticker(symbol)
        return self._yf_ticker(symbol)

    def _mt5_ticker(self, symbol: str) -> FxTicker:
        sym  = symbol.upper()
        tick = self._mt5.symbol_info_tick(sym)
        info = self._mt5.symbol_info(sym)
        if tick is None:
            raise ValueError(f"MT5: no tick for {symbol}")
        price = (tick.bid + tick.ask) / 2
        high  = float(getattr(info, "session_price_settlement", tick.ask) or tick.ask)
        low   = float(getattr(info, "session_price_buy", tick.bid) or tick.bid)
        digits = getattr(info, "digits", 5)
        point  = getattr(info, "point", 0.00001)
        spread = round((tick.ask - tick.bid) / point, 1)
        return FxTicker(
            symbol=symbol, price=price,
            bid=tick.bid, ask=tick.ask,
            volume_24h=float(getattr(tick, "volume_real", 0) or 0),
            change_pct_24h=0.0,
            high_24h=high, low_24h=low,
            spread_pips=spread,
        )

    def _yf_ticker(self, symbol: str) -> FxTicker:
        import yfinance as yf
        yf_sym = _to_yf(symbol)
        hist   = yf.download(yf_sym, period="2d", interval="1h",
                              progress=False, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
        if hist.empty:
            raise ValueError(f"yfinance: no data for {symbol} ({yf_sym})")
        price = float(hist["Close"].iloc[-1])
        chg   = float((hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                      / hist["Close"].iloc[0] * 100)
        return FxTicker(
            symbol=symbol, price=price,
            bid=round(price * 0.9999, 6), ask=round(price * 1.0001, 6),
            volume_24h=float(hist["Volume"].sum()),
            change_pct_24h=round(chg, 3),
            high_24h=float(hist["High"].max()),
            low_24h=float(hist["Low"].min()),
        )

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        if self._live:
            return self._mt5_klines(symbol, interval, limit)
        return self._yf_klines(symbol, interval, limit)

    def _mt5_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        tf_val = _MT5_TF_VAL.get(interval, 60)
        tf_map = {
            1:    self._mt5.TIMEFRAME_M1,
            5:    self._mt5.TIMEFRAME_M5,
            15:   self._mt5.TIMEFRAME_M15,
            30:   self._mt5.TIMEFRAME_M30,
            60:   self._mt5.TIMEFRAME_H1,
            240:  self._mt5.TIMEFRAME_H4,
            1440: self._mt5.TIMEFRAME_D1,
            10080:self._mt5.TIMEFRAME_W1,
        }
        tf    = tf_map.get(tf_val, self._mt5.TIMEFRAME_H1)
        rates = self._mt5.copy_rates_from_pos(symbol.upper(), tf, 0, limit)
        if rates is None or len(rates) == 0:
            raise ValueError(f"MT5: no rates for {symbol} {interval}")
        df = pd.DataFrame(rates)
        df["open_time"]  = pd.to_datetime(df["time"], unit="s")
        df["close_time"] = df["open_time"] + pd.Timedelta(minutes=tf_val)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        for c in ("open","high","low","close","volume"):
            df[c] = df[c].astype(float)
        return (df[["open_time","open","high","low","close","volume","close_time"]]
                .sort_values("open_time").reset_index(drop=True))

    def _yf_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        import yfinance as yf
        yf_sym = _to_yf(symbol)
        yf_int = _YF_INT.get(interval, "1h")
        period = _YF_PER.get(interval, "60d")

        hist = yf.download(yf_sym, period=period, interval=yf_int,
                           progress=False, auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
        if hist.empty:
            raise ValueError(f"yfinance: no data for {symbol} ({yf_sym})")

        hist = hist.reset_index()
        ts   = "Datetime" if "Datetime" in hist.columns else "Date"
        hist["open_time"]  = pd.to_datetime(hist[ts])
        hist["close_time"] = hist["open_time"]
        hist.rename(columns={
            "Open":"open","High":"high","Low":"low",
            "Close":"close","Volume":"volume"
        }, inplace=True)

        # Resample 1h → 4h
        if interval == "4h":
            hist = (hist.set_index("open_time")
                    .resample("4h").agg({"open":"first","high":"max",
                                         "low":"min","close":"last","volume":"sum"})
                    .dropna().reset_index())
            hist["close_time"] = hist["open_time"]

        for c in ("open","high","low","close","volume"):
            hist[c] = pd.to_numeric(hist[c], errors="coerce").astype(float)

        return (hist[["open_time","open","high","low","close","volume","close_time"]]
                .sort_values("open_time").tail(limit).reset_index(drop=True))

    def get_top_symbols(self, group: str = "FX Majors", n: int = 20) -> list[str]:
        return SYMBOL_GROUPS.get(group, ALL_FX_SYMBOLS)[:n]

    # ── Account ────────────────────────────────────────────────────────────────

    def get_balances(self) -> list[FxBalance]:
        if not self._live:
            return [FxBalance(asset="USD (fallback)", free=10000.0)]
        info = self._mt5.account_info()
        if not info:
            return []
        return [
            FxBalance(asset=info.currency,   free=info.balance),
            FxBalance(asset="Equity",        free=info.equity),
            FxBalance(asset="Free Margin",   free=info.margin_free),
            FxBalance(asset="Used Margin",   free=info.margin),
        ]

    def get_open_orders(self, symbol: str | None = None) -> list[FxOrder]:
        if not self._live:
            return []
        pos = (self._mt5.positions_get(symbol=symbol.upper())
               if symbol else self._mt5.positions_get())
        if not pos:
            return []
        return [self._parse_pos(p) for p in pos]

    def _parse_pos(self, p) -> FxOrder:
        return FxOrder(
            order_id=p.ticket, symbol=p.symbol,
            side="BUY" if p.type == 0 else "SELL",
            order_type="MARKET", status="OPEN",
            price=p.price_open, qty=p.volume,
            executed_qty=p.volume,
            time=datetime.fromtimestamp(p.time),
        )

    # ── Trading ────────────────────────────────────────────────────────────────

    def market_buy(self, symbol: str, volume: float) -> FxOrder:
        if not self._live:
            raise NotImplementedError(
                "MT5 terminal not connected. Live trading requires Windows + MT5 terminal open.")
        return self._send_order(symbol, volume, "BUY")

    def market_sell(self, symbol: str, volume: float) -> FxOrder:
        if not self._live:
            raise NotImplementedError("MT5 terminal not connected.")
        return self._send_order(symbol, volume, "SELL")

    def _send_order(self, symbol: str, volume: float, side: str) -> FxOrder:
        sym  = symbol.upper()
        tick = self._mt5.symbol_info_tick(sym)
        price = tick.ask if side == "BUY" else tick.bid
        otype = self._mt5.ORDER_TYPE_BUY if side == "BUY" else self._mt5.ORDER_TYPE_SELL
        req  = {
            "action":       self._mt5.TRADE_ACTION_DEAL,
            "symbol":       sym,
            "volume":       round(float(volume), 2),
            "type":         otype,
            "price":        price,
            "deviation":    20,
            "magic":        234000,
            "comment":      "scalp-bot",
            "type_time":    self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(req)
        if result.retcode != self._mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"MT5 order failed: {result.comment} (code {result.retcode})")
        return FxOrder(
            order_id=result.order, symbol=sym, side=side,
            order_type="MARKET", status="FILLED",
            price=result.price, qty=volume, executed_qty=volume,
            time=datetime.utcnow(),
        )

    def close_position(self, ticket: int) -> bool:
        if not self._live:
            raise NotImplementedError("MT5 terminal not connected.")
        positions = self._mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        pos   = positions[0]
        tick  = self._mt5.symbol_info_tick(pos.symbol)
        side  = self._mt5.ORDER_TYPE_SELL if pos.type == 0 else self._mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == 0 else tick.ask
        req   = {
            "action":       self._mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         side,
            "position":     ticket,
            "price":        price,
            "deviation":    20,
            "magic":        234000,
            "comment":      "scalp-bot close",
            "type_time":    self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(req)
        return result.retcode == self._mt5.TRADE_RETCODE_DONE

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel a pending order (not a position)."""
        if not self._live:
            return False
        req = {
            "action":  self._mt5.TRADE_ACTION_REMOVE,
            "order":   order_id,
            "comment": "cancel",
        }
        result = self._mt5.order_send(req)
        return result.retcode == self._mt5.TRADE_RETCODE_DONE
