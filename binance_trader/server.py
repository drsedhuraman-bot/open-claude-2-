"""
FastAPI web server — REST + WebSocket + autonomous scalp-bot engine.

Run:  python server.py
Then open: http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from bot import BotState, TradingBot, TradeRecord
from client import BinanceClient
from strategies import STRATEGIES, ScalpEngine, SCALP_PRESETS
from strategies.base import Signal

config.setup_logging()
logger = logging.getLogger("binance_trader.server")

# ── Core globals ───────────────────────────────────────────────────────────────

client:     BinanceClient | None = None
bot:        TradingBot    | None = None
ws_clients: list[WebSocket] = []

# ── Scalp-bot globals ──────────────────────────────────────────────────────────

scalp_cfg: dict = {
    "symbol":             config.BOT_SYMBOL,
    "interval":           "5m",
    "preset":             "Standard",
    "auto_trade":         False,
    "amount_usdt":        50.0,
    "sl_atr_mult":        1.0,
    "t1_atr_mult":        1.5,
    "t2_atr_mult":        3.0,
    "max_daily_trades":   10,
    "max_daily_loss_pct": 5.0,
    "cooldown_secs":      60,
    "scan_secs":          30,
}

scalp_pos: dict = {
    "in_position": False,
    "entry_price": 0.0,
    "entry_qty":   0.0,
    "stop_price":  0.0,
    "t1_price":    0.0,
    "t2_price":    0.0,
    "entry_time":  None,
}

scalp_stats: dict = {
    "running":         False,
    "daily_trades":    0,
    "daily_pnl_pct":   0.0,
    "last_trade_time": None,
    "last_scan_time":  None,
    "last_verdict":    "HOLD",
    "last_bull_score": 0.0,
    "last_bear_score": 0.0,
    "total_trades":    0,
    "total_pnl_pct":   0.0,
    "day_key":         "",
}

scalp_trades: list[dict] = []
_scalp_task:  asyncio.Task | None = None


# ── Lifecycle ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, bot
    errors = config.validate()
    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
    else:
        client = BinanceClient()
        if client.ping():
            logger.info("Binance connected  testnet=%s", config.USE_TESTNET)
            _init_bot(config.BOT_SYMBOL, config.BOT_INTERVAL, config.BOT_STRATEGY)
        else:
            logger.error("Binance ping failed")
    yield
    # Cleanup
    scalp_stats["running"] = False


app = FastAPI(title="Binance Trader", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _init_bot(symbol: str, interval: str, strategy: str) -> None:
    global bot
    def _on_trade(t: TradeRecord) -> None:
        asyncio.run(_broadcast({
            "type":   "trade",
            "side":   t.side,
            "symbol": t.symbol,
            "price":  t.price,
            "qty":    t.qty,
            "reason": t.reason,
            "pnl":    t.pnl_pct,
            "time":   t.time.strftime("%H:%M:%S"),
        }))
    bot = TradingBot(
        client=client, symbol=symbol, interval=interval,
        strategy_name=strategy, on_signal=_on_trade,
    )
    logger.info("Bot initialised  symbol=%s  strategy=%s", symbol, strategy)


async def _broadcast(data: dict) -> None:
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)


def _err(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg}, status_code=status)


def _ok(data: Any = None) -> dict:
    return {"ok": True, "data": data}


def _fmt_time(dt: datetime | None) -> str | None:
    return dt.strftime("%H:%M:%S") if dt else None


# ── Scalp-bot engine ───────────────────────────────────────────────────────────

async def _scalp_tick() -> None:
    """One scan cycle: fetch data → engine → optional auto-trade → broadcast."""
    global scalp_pos, scalp_stats, scalp_trades

    # Day rollover
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if scalp_stats["day_key"] != today:
        scalp_stats["day_key"]         = today
        scalp_stats["daily_trades"]    = 0
        scalp_stats["daily_pnl_pct"]   = 0.0
        scalp_stats["last_trade_time"] = None

    symbol   = scalp_cfg["symbol"]
    interval = scalp_cfg["interval"]

    df            = client.get_klines(symbol, interval, limit=220)
    current_price = float(df["close"].iloc[-1])

    # ── Check open position for stop / T1 / T2 ────────────────────────────────
    if scalp_pos["in_position"]:
        entry = scalp_pos["entry_price"]
        qty   = scalp_pos["entry_qty"]
        stop  = scalp_pos["stop_price"]
        t1    = scalp_pos["t1_price"]
        t2    = scalp_pos["t2_price"]
        reason_out = None

        if stop and current_price <= stop:
            reason_out = "Stop Loss"
        elif t2 and current_price >= t2:
            reason_out = "T2 Target"
        elif t1 and current_price >= t1:
            reason_out = "T1 Target"

        if reason_out:
            pnl = (current_price - entry) / entry * 100
            try:
                if scalp_cfg["auto_trade"]:
                    client.market_sell(symbol, qty)
                scalp_pos["in_position"] = False
                scalp_stats["daily_trades"]  += 1
                scalp_stats["total_trades"]  += 1
                scalp_stats["daily_pnl_pct"] += pnl
                scalp_stats["total_pnl_pct"] += pnl
                scalp_stats["last_trade_time"] = datetime.utcnow()
                t = {
                    "time": datetime.utcnow().strftime("%H:%M:%S"),
                    "side": "SELL", "symbol": symbol,
                    "entry": entry, "exit": current_price,
                    "pnl_pct": round(pnl, 3), "reason": reason_out,
                }
                scalp_trades.append(t)
                if len(scalp_trades) > 100:
                    scalp_trades.pop(0)
                await _broadcast({"type": "scalp_trade", **t})
                await _broadcast({"type": "log",
                    "msg": f"[SCALP] {reason_out} SELL {symbol} @ {current_price:.4f}  PnL={pnl:+.2f}%"})
            except Exception as e:
                logger.error("Scalp close failed: %s", e)

    # ── Run engine ─────────────────────────────────────────────────────────────
    engine = ScalpEngine.from_preset(scalp_cfg["preset"], **{
        k: scalp_cfg[k] for k in ("sl_atr_mult", "t1_atr_mult", "t2_atr_mult")
    })
    result = engine.analyze(df)

    scalp_stats["last_scan_time"]  = datetime.utcnow().strftime("%H:%M:%S")
    scalp_stats["last_verdict"]    = result.verdict
    scalp_stats["last_bull_score"] = result.bull_score
    scalp_stats["last_bear_score"] = result.bear_score

    conditions = [
        {"name": c.name, "label": c.label, "detail": c.detail,
         "bull_pts": c.bull_pts, "bear_pts": c.bear_pts}
        for c in result.conditions
    ]

    pos_payload = {**scalp_pos, "entry_time": _fmt_time(scalp_pos.get("entry_time"))}
    stats_payload = {
        **scalp_stats,
        "last_trade_time": _fmt_time(scalp_stats.get("last_trade_time")),
    }

    await _broadcast({
        "type": "scalp_result",
        "symbol":       symbol,    "interval":     interval,
        "ltp":          result.entry,
        "verdict":      result.verdict,
        "bull_score":   result.bull_score, "bear_score":  result.bear_score,
        "bull_pass":    result.bull_pass,  "bear_pass":   result.bear_pass,
        "entry":        result.entry,      "stop":        result.stop,
        "t1":           result.t1,         "t2":          result.t2,
        "risk_reward":  result.risk_reward,"risk_pts":    result.risk_pts,
        "vwap":         result.vwap,       "atr":         result.atr,
        "pivot":        result.pivot,      "r1":          result.r1,
        "s1":           result.s1,         "r2":          result.r2,
        "s2":           result.s2,
        "rsi":          result.rsi,        "stoch_k":     result.stoch_k,
        "stoch_d":      result.stoch_d,    "macd_hist":   result.macd_hist,
        "bb_pct":       result.bb_pct,     "volume_ratio":result.volume_ratio,
        "conditions":   conditions,
        "position":     pos_payload,
        "stats":        stats_payload,
    })

    # ── Auto-trade: entry ──────────────────────────────────────────────────────
    if (scalp_cfg["auto_trade"]
            and not scalp_pos["in_position"]
            and result.signal == Signal.BUY):

        cb_ok = True
        msg   = ""
        if scalp_stats["daily_trades"] >= scalp_cfg["max_daily_trades"]:
            cb_ok, msg = False, "max daily trades reached"
        elif (scalp_stats["daily_pnl_pct"] < -scalp_cfg["max_daily_loss_pct"]):
            cb_ok, msg = False, "daily loss limit hit"
        elif scalp_stats["last_trade_time"]:
            elapsed = (datetime.utcnow() - scalp_stats["last_trade_time"]).total_seconds()
            if elapsed < scalp_cfg["cooldown_secs"]:
                cb_ok, msg = False, f"cooldown ({scalp_cfg['cooldown_secs']-elapsed:.0f}s left)"

        if not cb_ok:
            logger.info("Scalp auto-trade blocked: %s", msg)
            await _broadcast({"type": "log", "msg": f"[SCALP] Skip BUY — {msg}"})
        else:
            qty = scalp_cfg["amount_usdt"] / current_price
            try:
                order = client.market_buy(symbol, qty)
                scalp_pos.update({
                    "in_position": True,
                    "entry_price": current_price,
                    "entry_qty":   qty,
                    "stop_price":  result.stop,
                    "t1_price":    result.t1,
                    "t2_price":    result.t2,
                    "entry_time":  datetime.utcnow(),
                })
                scalp_stats["last_trade_time"] = datetime.utcnow()
                t = {
                    "time": datetime.utcnow().strftime("%H:%M:%S"),
                    "side": "BUY", "symbol": symbol,
                    "entry": current_price, "exit": 0,
                    "pnl_pct": 0.0, "reason": f"Auto scalp bull={result.bull_score}",
                }
                scalp_trades.append(t)
                await _broadcast({"type": "scalp_trade", **t})
                await _broadcast({"type": "log",
                    "msg": f"[SCALP AUTO] BUY {symbol} @ {current_price:.4f}  "
                           f"Stop={result.stop:.4f}  T1={result.t1:.4f}  "
                           f"T2={result.t2:.4f}  order={order.order_id}"})
            except Exception as e:
                logger.error("Scalp auto-buy failed: %s", e)
                await _broadcast({"type": "log", "msg": f"[SCALP] BUY FAILED: {e}"})


async def _scalp_loop() -> None:
    """Background asyncio task running the scalp scan loop."""
    logger.info("Scalp auto-scan started  interval=%s", scalp_cfg["scan_secs"])
    while scalp_stats["running"]:
        try:
            if client:
                await _scalp_tick()
        except Exception as e:
            logger.error("Scalp tick error: %s", e, exc_info=True)
            await _broadcast({"type": "log", "msg": f"[SCALP] Error: {e}"})
        await asyncio.sleep(scalp_cfg["scan_secs"])
    logger.info("Scalp auto-scan stopped")


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse("static/index.html")


# ── Market data ────────────────────────────────────────────────────────────────

@app.get("/api/ticker/{symbol}")
def get_ticker(symbol: str):
    if not client:
        return _err("Not connected")
    try:
        t = client.get_ticker(symbol.upper())
        return _ok({
            "symbol": t.symbol, "price":  t.price,
            "change": t.change_pct_24h,
            "high":   t.high_24h, "low": t.low_24h,
            "volume": t.volume_24h, "bid": t.bid, "ask": t.ask,
        })
    except Exception as e:
        return _err(str(e))


@app.get("/api/klines/{symbol}/{interval}")
def get_klines(symbol: str, interval: str, limit: int = 100):
    if not client:
        return _err("Not connected")
    try:
        df = client.get_klines(symbol.upper(), interval, limit=limit)
        return _ok(df[["open_time", "open", "high", "low", "close", "volume"]]
                   .assign(open_time=df["open_time"].dt.strftime("%Y-%m-%d %H:%M"))
                   .to_dict(orient="records"))
    except Exception as e:
        return _err(str(e))


@app.get("/api/orderbook/{symbol}")
def get_orderbook(symbol: str, depth: int = 10):
    if not client:
        return _err("Not connected")
    try:
        return _ok(client.get_orderbook(symbol.upper(), limit=depth))
    except Exception as e:
        return _err(str(e))


@app.get("/api/top-symbols")
def top_symbols(quote: str = "USDT", n: int = 20):
    if not client:
        return _err("Not connected")
    try:
        return _ok(client.get_top_symbols(quote, n))
    except Exception as e:
        return _err(str(e))


# ── Account ────────────────────────────────────────────────────────────────────

@app.get("/api/balances")
def get_balances():
    if not client:
        return _err("Not connected")
    try:
        balances = client.get_balances()
        return _ok([{"asset": b.asset, "free": b.free, "locked": b.locked, "total": b.total}
                    for b in balances])
    except Exception as e:
        return _err(str(e))


@app.get("/api/orders/open")
def open_orders(symbol: str = ""):
    if not client:
        return _err("Not connected")
    try:
        orders = client.get_open_orders(symbol.upper() if symbol else None)
        return _ok([{
            "id":     o.order_id, "symbol": o.symbol,
            "side":   o.side,     "type":   o.order_type,
            "price":  o.price,    "qty":    o.qty,
            "filled": o.fill_pct, "status": o.status,
        } for o in orders])
    except Exception as e:
        return _err(str(e))


# ── Trading ────────────────────────────────────────────────────────────────────

@app.post("/api/order/market")
async def place_market_order(body: dict):
    if not client:
        return _err("Not connected")
    sym  = body.get("symbol", "").upper()
    side = body.get("side", "").upper()
    qty  = body.get("qty")
    if not sym or side not in ("BUY", "SELL") or not qty:
        return _err("symbol, side (BUY|SELL), qty required")
    try:
        o = client.market_buy(sym, float(qty)) if side == "BUY" else client.market_sell(sym, float(qty))
        await _broadcast({"type": "log", "msg": f"Market {side} {sym} qty={qty}  status={o.status}"})
        return _ok({"order_id": o.order_id, "status": o.status})
    except Exception as e:
        return _err(str(e))


@app.post("/api/order/limit")
async def place_limit_order(body: dict):
    if not client:
        return _err("Not connected")
    sym   = body.get("symbol", "").upper()
    side  = body.get("side", "").upper()
    qty   = body.get("qty")
    price = body.get("price")
    if not sym or side not in ("BUY", "SELL") or not qty or not price:
        return _err("symbol, side, qty, price required")
    try:
        o = (client.limit_buy(sym, float(qty), float(price)) if side == "BUY"
             else client.limit_sell(sym, float(qty), float(price)))
        await _broadcast({"type": "log", "msg": f"Limit {side} {sym} @ {price}  status={o.status}"})
        return _ok({"order_id": o.order_id, "status": o.status})
    except Exception as e:
        return _err(str(e))


@app.delete("/api/order/{symbol}/{order_id}")
async def cancel_order(symbol: str, order_id: int):
    if not client:
        return _err("Not connected")
    ok = client.cancel_order(symbol.upper(), order_id)
    if ok:
        await _broadcast({"type": "log", "msg": f"Cancelled order {order_id} on {symbol}"})
    return _ok(ok)


# ── Legacy Bot ─────────────────────────────────────────────────────────────────

@app.get("/api/bot/status")
def bot_status():
    if not bot:
        return _ok({"state": "NOT_CREATED"})
    return _ok(bot.status_dict())


@app.get("/api/bot/trades")
def bot_trades():
    if not bot:
        return _ok([])
    return _ok([{
        "time": t.time.strftime("%H:%M:%S"), "side": t.side,
        "price": t.price, "qty": t.qty, "pnl": t.pnl_pct, "reason": t.reason,
    } for t in bot.trades[-50:]])


@app.post("/api/bot/start")
async def bot_start(body: dict):
    global bot
    if not client:
        return _err("Not connected")
    symbol   = body.get("symbol",   config.BOT_SYMBOL).upper()
    interval = body.get("interval", config.BOT_INTERVAL)
    strategy = body.get("strategy", config.BOT_STRATEGY)
    try:
        config.BOT_TRADE_AMOUNT    = float(body.get("amount",     config.BOT_TRADE_AMOUNT))
        config.BOT_STOP_LOSS_PCT   = float(body.get("sl",         config.BOT_STOP_LOSS_PCT))
        config.BOT_TAKE_PROFIT_PCT = float(body.get("tp",         config.BOT_TAKE_PROFIT_PCT))
        config.BOT_MAX_OPEN_TRADES = int(body.get("max_trades",   config.BOT_MAX_OPEN_TRADES))
    except (ValueError, TypeError):
        pass
    if bot and bot.state == BotState.RUNNING:
        bot.stop()
    _init_bot(symbol, interval, strategy)
    bot.start()
    await _broadcast({"type": "log", "msg": f"Bot started  {symbol}  {strategy}  {interval}"})
    return _ok(bot.status_dict())


@app.post("/api/bot/stop")
async def bot_stop():
    if not bot:
        return _err("Bot not running")
    bot.stop()
    await _broadcast({"type": "log", "msg": "Bot stopped"})
    return _ok(bot.status_dict())


@app.post("/api/bot/pause")
async def bot_pause():
    if not bot:
        return _err("Bot not running")
    bot.pause()
    return _ok(bot.status_dict())


@app.post("/api/bot/resume")
async def bot_resume():
    if not bot:
        return _err("Bot not running")
    bot.resume()
    return _ok(bot.status_dict())


# ── Scalp Bot ──────────────────────────────────────────────────────────────────

@app.post("/api/scalp/bot/start")
async def scalp_bot_start(body: dict):
    global _scalp_task
    if not client:
        return _err("Not connected")

    # Apply config from request
    for k in ("symbol", "interval", "preset"):
        if k in body:
            scalp_cfg[k] = str(body[k]).upper() if k == "symbol" else body[k]
    for k in ("amount_usdt", "sl_atr_mult", "t1_atr_mult", "t2_atr_mult",
              "max_daily_loss_pct", "cooldown_secs", "scan_secs"):
        if k in body:
            try:
                scalp_cfg[k] = float(body[k])
            except (ValueError, TypeError):
                pass
    for k in ("max_daily_trades",):
        if k in body:
            try:
                scalp_cfg[k] = int(body[k])
            except (ValueError, TypeError):
                pass
    scalp_cfg["auto_trade"] = bool(body.get("auto_trade", scalp_cfg["auto_trade"]))

    if scalp_stats["running"]:
        scalp_stats["running"] = False
        if _scalp_task and not _scalp_task.done():
            _scalp_task.cancel()
        await asyncio.sleep(0.1)

    scalp_stats["running"] = True
    _scalp_task = asyncio.create_task(_scalp_loop())
    mode = "AUTO-TRADE" if scalp_cfg["auto_trade"] else "SCAN-ONLY"
    await _broadcast({"type": "log",
        "msg": f"[SCALP] Bot started  {scalp_cfg['symbol']}  {scalp_cfg['preset']}  {scalp_cfg['interval']}  {mode}"})
    return _ok({**scalp_cfg, **scalp_stats})


@app.post("/api/scalp/bot/stop")
async def scalp_bot_stop():
    global _scalp_task
    scalp_stats["running"] = False
    if _scalp_task and not _scalp_task.done():
        _scalp_task.cancel()
    scalp_pos["in_position"] = False
    await _broadcast({"type": "log", "msg": "[SCALP] Bot stopped"})
    return _ok({**scalp_stats})


@app.get("/api/scalp/bot/status")
def scalp_bot_status():
    return _ok({
        "config":  scalp_cfg,
        "stats":   {**scalp_stats, "last_trade_time": _fmt_time(scalp_stats.get("last_trade_time"))},
        "position": {**scalp_pos, "entry_time": _fmt_time(scalp_pos.get("entry_time"))},
        "trades":  scalp_trades[-20:],
    })


@app.post("/api/scalp/position/close")
async def scalp_close_position():
    if not scalp_pos["in_position"]:
        return _err("No open position")
    if not client:
        return _err("Not connected")
    symbol = scalp_cfg["symbol"]
    qty    = scalp_pos["entry_qty"]
    entry  = scalp_pos["entry_price"]
    try:
        t = client.get_ticker(symbol)
        cur = t.price
        order = client.market_sell(symbol, qty)
        pnl = (cur - entry) / entry * 100
        scalp_pos["in_position"] = False
        scalp_stats["daily_trades"]  += 1
        scalp_stats["total_trades"]  += 1
        scalp_stats["daily_pnl_pct"] += pnl
        scalp_stats["total_pnl_pct"] += pnl
        scalp_stats["last_trade_time"] = datetime.utcnow()
        rec = {
            "time": datetime.utcnow().strftime("%H:%M:%S"),
            "side": "SELL", "symbol": symbol,
            "entry": entry, "exit": cur,
            "pnl_pct": round(pnl, 3), "reason": "Manual close",
        }
        scalp_trades.append(rec)
        await _broadcast({"type": "scalp_trade", **rec})
        await _broadcast({"type": "log", "msg": f"[SCALP] Manual close {symbol} @ {cur:.4f}  PnL={pnl:+.2f}%"})
        return _ok({"order_id": order.order_id, "pnl_pct": round(pnl, 3)})
    except Exception as e:
        return _err(str(e))


# ── One-shot scalp analyze ─────────────────────────────────────────────────────

@app.post("/api/scalp/analyze")
def scalp_analyze(body: dict):
    if not client:
        return _err("Not connected")
    symbol   = body.get("symbol", config.BOT_SYMBOL).upper()
    interval = body.get("interval", "5m")
    limit    = int(body.get("limit", 200))
    preset   = body.get("preset", "Standard")
    params   = {k: body[k] for k in ("sl_atr_mult","t1_atr_mult","t2_atr_mult") if k in body}
    try:
        df     = client.get_klines(symbol, interval, limit=limit)
        engine = ScalpEngine.from_preset(preset, **params)
        r      = engine.analyze(df)
        return _ok({
            "symbol": symbol, "interval": interval, "preset": preset,
            "ltp": r.entry, "verdict": r.verdict,
            "bull_score": r.bull_score, "bear_score": r.bear_score,
            "bull_pass": r.bull_pass, "bear_pass": r.bear_pass,
            "entry": r.entry, "stop": r.stop, "t1": r.t1, "t2": r.t2,
            "risk_reward": r.risk_reward, "risk_pts": r.risk_pts,
            "vwap": r.vwap, "atr": r.atr,
            "pivot": r.pivot, "r1": r.r1, "s1": r.s1, "r2": r.r2, "s2": r.s2,
            "rsi": r.rsi, "stoch_k": r.stoch_k, "stoch_d": r.stoch_d,
            "macd_hist": r.macd_hist, "bb_pct": r.bb_pct, "volume_ratio": r.volume_ratio,
            "conditions": [{"name": c.name, "label": c.label, "detail": c.detail,
                            "bull_pts": c.bull_pts, "bear_pts": c.bear_pts}
                           for c in r.conditions],
        })
    except Exception as e:
        logger.exception("Scalp analyze failed")
        return _err(str(e))


@app.get("/api/scalp/presets")
def scalp_presets():
    return _ok(list(SCALP_PRESETS.keys()))


@app.get("/api/scalp/trades")
def scalp_trade_history():
    return _ok(scalp_trades[-50:])


# ── Backtest ───────────────────────────────────────────────────────────────────

@app.post("/api/backtest")
def run_backtest(body: dict):
    if not client:
        return _err("Not connected")
    symbol   = body.get("symbol",   config.BOT_SYMBOL).upper()
    interval = body.get("interval", config.BOT_INTERVAL)
    strategy = body.get("strategy", config.BOT_STRATEGY)
    days     = int(body.get("days",   90))
    equity   = float(body.get("equity", 1000))
    sl_pct   = float(body.get("sl",   config.BOT_STOP_LOSS_PCT))
    tp_pct   = float(body.get("tp",   config.BOT_TAKE_PROFIT_PCT))
    try:
        from backtest import Backtester
        limit  = min(1000, days * 24)
        df     = client.get_klines(symbol, interval, limit=limit)
        strat  = STRATEGIES[strategy]()
        bt     = Backtester(strategy=strat, initial_usdt=equity,
                            stop_loss_pct=sl_pct, take_profit_pct=tp_pct)
        result = bt.run(df, symbol=symbol, interval=interval)
        return _ok({
            "symbol":   result.symbol, "interval":  result.interval,
            "strategy": result.strategy,
            "start":    result.start_date.strftime("%Y-%m-%d"),
            "end":      result.end_date.strftime("%Y-%m-%d"),
            "initial":  result.initial_usdt, "final": result.final_usdt,
            "pnl_pct":  result.total_pnl_pct, "pnl_usdt": result.total_pnl_usdt,
            "drawdown": result.max_drawdown_pct,
            "total_trades": result.total_trades,
            "wins": result.winning_trades, "losses": result.losing_trades,
            "win_rate": result.win_rate, "avg_pnl": result.avg_pnl_pct,
            "best": result.best_trade_pct, "worst": result.worst_trade_pct,
            "trades": [{
                "entry_time":  t.entry_time.strftime("%Y-%m-%d %H:%M"),
                "exit_time":   t.exit_time.strftime("%Y-%m-%d %H:%M"),
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "pnl_pct": t.pnl_pct, "pnl_usdt": t.pnl_usdt,
                "reason_out": t.reason_out,
            } for t in result.trades[-50:]],
        })
    except Exception as e:
        logger.exception("Backtest failed")
        return _err(str(e))


# ── Meta ───────────────────────────────────────────────────────────────────────

@app.get("/api/strategies")
def list_strategies():
    return _ok(list(STRATEGIES.keys()))


@app.get("/api/intervals")
def list_intervals():
    return _ok(config.VALID_INTERVALS)


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    logger.info("WS client connected  total=%d", len(ws_clients))
    try:
        while True:
            await asyncio.sleep(3)
            if client:
                try:
                    sym = scalp_cfg["symbol"] if scalp_stats["running"] else (bot.symbol if bot else config.BOT_SYMBOL)
                    t   = client.get_ticker(sym)
                    await ws.send_text(json.dumps({
                        "type":   "ticker",
                        "symbol": t.symbol, "price":  t.price,
                        "change": t.change_pct_24h,
                        "high":   t.high_24h, "low": t.low_24h,
                        "volume": t.volume_24h,
                    }))
                    if bot:
                        await ws.send_text(json.dumps({
                            "type": "bot_status", **bot.status_dict(),
                        }))
                    # Push scalp bot heartbeat
                    await ws.send_text(json.dumps({
                        "type": "scalp_heartbeat",
                        "running":      scalp_stats["running"],
                        "in_position":  scalp_pos["in_position"],
                        "daily_trades": scalp_stats["daily_trades"],
                        "daily_pnl":    scalp_stats["daily_pnl_pct"],
                        "last_scan":    scalp_stats["last_scan_time"],
                        "last_verdict": scalp_stats["last_verdict"],
                        "auto_trade":   scalp_cfg["auto_trade"],
                    }))
                except Exception:
                    pass
    except WebSocketDisconnect:
        ws_clients.remove(ws)
        logger.info("WS client disconnected  total=%d", len(ws_clients))


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
