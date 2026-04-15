"""
Binance Trading Terminal — full interactive TUI built with Textual.

Tabs:
  1  Dashboard   — live price, portfolio, open orders, bot status
  2  Trade       — place / cancel orders
  3  Bot         — start/stop/configure the trading bot
  4  Backtest    — run walk-forward backtest on historical data
  5  Log         — live activity log
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label,
    Log, ProgressBar, Select, Static, Switch, TabbedContent, TabPane,
)

import config
from bot import BotState, TradingBot, TradeRecord
from client import BinanceClient
from strategies import STRATEGIES

logger = logging.getLogger("binance_trader.terminal")

# ── Colour helpers ─────────────────────────────────────────────────────────────

def _pct_markup(val: float, decimals: int = 2) -> str:
    color = "green" if val >= 0 else "red"
    return f"[{color}]{val:+.{decimals}f}%[/{color}]"


def _price(val: float) -> str:
    return f"${val:,.4f}" if val < 1000 else f"${val:,.2f}"


# ── Confirm dialog ─────────────────────────────────────────────────────────────

class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss_false", "Cancel")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm-box"):
            yield Label(self._message, id="confirm-msg")
            with Horizontal(id="confirm-btns"):
                yield Button("Confirm", variant="success", id="yes")
                yield Button("Cancel",  variant="error",   id="no")

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_dismiss_false(self) -> None:
        self.dismiss(False)


# ── Dashboard tab ──────────────────────────────────────────────────────────────

class DashboardTab(Container):
    """Auto-refreshing price / portfolio / orders panel."""

    symbol: reactive[str] = reactive(config.BOT_SYMBOL)

    def compose(self) -> ComposeResult:
        with Horizontal(id="dash-top"):
            with Vertical(id="ticker-panel", classes="panel"):
                yield Label("── Ticker ──", classes="panel-title")
                yield Static("", id="ticker-body")
            with Vertical(id="portfolio-panel", classes="panel"):
                yield Label("── Portfolio ──", classes="panel-title")
                yield DataTable(id="balance-table", show_cursor=False)
            with Vertical(id="bot-status-panel", classes="panel"):
                yield Label("── Bot Status ──", classes="panel-title")
                yield Static("", id="bot-status-body")
        with Vertical(id="orders-panel", classes="panel"):
            yield Label("── Open Orders ──", classes="panel-title")
            yield DataTable(id="orders-table", show_cursor=False)

    def on_mount(self) -> None:
        bal = self.query_one("#balance-table", DataTable)
        bal.add_columns("Asset", "Free", "Locked", "Total")
        ord_tbl = self.query_one("#orders-table", DataTable)
        ord_tbl.add_columns("ID", "Symbol", "Side", "Type", "Price", "Qty", "Filled%", "Status")

    def refresh_data(self, client: BinanceClient, bot: TradingBot | None) -> None:
        """Called by the app timer — runs in worker thread."""
        try:
            ticker = client.get_ticker(self.symbol)
            chg_c  = "green" if ticker.change_pct_24h >= 0 else "red"
            self.query_one("#ticker-body", Static).update(
                f"Symbol : [bold]{ticker.symbol}[/bold]\n"
                f"Price  : [bold yellow]{_price(ticker.price)}[/bold yellow]\n"
                f"24h    : [{chg_c}]{ticker.change_pct_24h:+.2f}%[/{chg_c}]\n"
                f"High   : {_price(ticker.high_24h)}\n"
                f"Low    : {_price(ticker.low_24h)}\n"
                f"Volume : {ticker.volume_24h:,.2f}\n"
                f"Bid/Ask: {_price(ticker.bid)} / {_price(ticker.ask)}"
            )
        except Exception as e:
            self.query_one("#ticker-body", Static).update(f"[red]{e}[/red]")

        try:
            balances = client.get_balances()
            tbl = self.query_one("#balance-table", DataTable)
            tbl.clear()
            for b in balances[:12]:
                tbl.add_row(b.asset, f"{b.free:.6f}", f"{b.locked:.6f}", f"{b.total:.6f}")
        except Exception:
            pass

        try:
            orders = client.get_open_orders(self.symbol)
            tbl = self.query_one("#orders-table", DataTable)
            tbl.clear()
            for o in orders[:10]:
                side_m = f"[green]{o.side}[/green]" if o.side == "BUY" else f"[red]{o.side}[/red]"
                tbl.add_row(
                    str(o.order_id), o.symbol, side_m, o.order_type,
                    _price(o.price) if o.price else "MARKET",
                    f"{o.qty:.6f}", f"{o.fill_pct:.1f}%", o.status,
                )
            if not orders:
                tbl.add_row("—", "—", "—", "—", "—", "—", "—", "—")
        except Exception:
            pass

        if bot:
            s = bot.status_dict()
            sc = {"RUNNING": "green", "PAUSED": "yellow", "STOPPED": "red", "IDLE": "dim"}.get(s["state"], "white")
            pos_line = (
                f"Entry  : {s['entry_price']:.4f}\n"
                f"SL/TP  : {s['stop_loss']:.4f} / {s['take_profit']:.4f}\n"
                if s["open_position"] else ""
            )
            self.query_one("#bot-status-body", Static).update(
                f"State    : [{sc}]{s['state']}[/{sc}]\n"
                f"Symbol   : {s['symbol']}\n"
                f"Strategy : {s['strategy']}\n"
                f"Interval : {s['interval']}\n"
                f"Position : {'[green]OPEN[/green]' if s['open_position'] else '[dim]NONE[/dim]'}\n"
                f"{pos_line}"
                f"Trades   : {s['total_trades']}\n"
                f"Win Rate : {s['win_rate']}\n"
                f"Total PnL: {s['total_pnl']}\n"
                f"Uptime   : {s['uptime']}"
            )
        else:
            self.query_one("#bot-status-body", Static).update("[dim]Bot not started[/dim]")


# ── Trade tab ─────────────────────────────────────────────────────────────────

class TradeTab(Container):
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="place-order-panel", classes="panel"):
                yield Label("── Place Order ──", classes="panel-title")
                yield Label("Symbol")
                yield Input(value=config.BOT_SYMBOL, id="trade-symbol", placeholder="e.g. BTCUSDT")
                yield Label("Side")
                yield Select(
                    [("BUY", "BUY"), ("SELL", "SELL")],
                    value="BUY", id="trade-side",
                )
                yield Label("Order Type")
                yield Select(
                    [("MARKET", "MARKET"), ("LIMIT", "LIMIT")],
                    value="MARKET", id="trade-type",
                )
                yield Label("Quantity (base asset)")
                yield Input(placeholder="e.g. 0.001", id="trade-qty")
                yield Label("Limit Price (leave blank for MARKET)", id="limit-price-label")
                yield Input(placeholder="e.g. 74000", id="trade-price")
                yield Static("", id="current-price-label")
                with Horizontal():
                    yield Button("Fetch Price", id="fetch-price-btn", variant="default")
                    yield Button("Place Order", id="place-order-btn", variant="success")

            with Vertical(id="cancel-order-panel", classes="panel"):
                yield Label("── Cancel Order ──", classes="panel-title")
                yield Label("Symbol")
                yield Input(value=config.BOT_SYMBOL, id="cancel-symbol", placeholder="e.g. BTCUSDT")
                yield Button("Load Open Orders", id="load-orders-btn", variant="default")
                yield DataTable(id="cancel-orders-table")
                yield Label("Order ID to cancel")
                yield Input(placeholder="Order ID", id="cancel-order-id")
                yield Button("Cancel Order", id="cancel-order-btn", variant="error")
                yield Static("", id="cancel-status")

    def on_mount(self) -> None:
        tbl = self.query_one("#cancel-orders-table", DataTable)
        tbl.add_columns("ID", "Side", "Type", "Price", "Qty", "Status")
        tbl.cursor_type = "row"


# ── Bot tab ───────────────────────────────────────────────────────────────────

class BotTab(Container):
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="bot-config-panel", classes="panel"):
                yield Label("── Bot Configuration ──", classes="panel-title")
                yield Label("Symbol")
                yield Input(value=config.BOT_SYMBOL, id="bot-symbol")
                yield Label("Interval")
                yield Select(
                    [(i, i) for i in config.VALID_INTERVALS],
                    value=config.BOT_INTERVAL, id="bot-interval",
                )
                yield Label("Strategy")
                yield Select(
                    [(k, k) for k in STRATEGIES.keys()],
                    value=config.BOT_STRATEGY, id="bot-strategy",
                )
                yield Label("Trade Amount (USDT)")
                yield Input(value=str(config.BOT_TRADE_AMOUNT), id="bot-amount")
                yield Label("Stop Loss %")
                yield Input(value=str(config.BOT_STOP_LOSS_PCT), id="bot-sl")
                yield Label("Take Profit %")
                yield Input(value=str(config.BOT_TAKE_PROFIT_PCT), id="bot-tp")
                yield Label("Max Open Trades")
                yield Input(value=str(config.BOT_MAX_OPEN_TRADES), id="bot-max")

            with Vertical(id="bot-control-panel", classes="panel"):
                yield Label("── Bot Control ──", classes="panel-title")
                yield Static("", id="bot-state-display")
                with Horizontal(id="bot-btns"):
                    yield Button("▶  Start",  id="bot-start",  variant="success")
                    yield Button("⏸  Pause",  id="bot-pause",  variant="warning")
                    yield Button("▶  Resume", id="bot-resume", variant="primary")
                    yield Button("⏹  Stop",   id="bot-stop",   variant="error")
                yield Label("── Recent Bot Trades ──", classes="panel-title")
                yield DataTable(id="bot-trades-table", show_cursor=False)

    def on_mount(self) -> None:
        tbl = self.query_one("#bot-trades-table", DataTable)
        tbl.add_columns("Time", "Side", "Price", "Qty", "PnL%", "Reason")

    def refresh_trades(self, bot: TradingBot | None) -> None:
        if not bot:
            return
        s  = bot.status_dict()
        sc = {"RUNNING": "green", "PAUSED": "yellow", "STOPPED": "red", "IDLE": "dim"}.get(s["state"], "white")
        self.query_one("#bot-state-display", Static).update(
            f"State: [{sc}]{s['state']}[/{sc}]   "
            f"Trades: {s['total_trades']}   "
            f"Win: {s['win_rate']}   "
            f"PnL: {s['total_pnl']}   "
            f"Uptime: {s['uptime']}"
        )
        tbl = self.query_one("#bot-trades-table", DataTable)
        tbl.clear()
        for t in bot.trades[-20:][::-1]:
            sc2 = "green" if t.side == "BUY" else "red"
            pnl = f"{t.pnl_pct:+.2f}%" if t.side == "SELL" else "—"
            pc  = "green" if t.pnl_pct >= 0 else "red"
            tbl.add_row(
                t.time.strftime("%H:%M:%S"),
                f"[{sc2}]{t.side}[/{sc2}]",
                _price(t.price),
                f"{t.qty:.6f}",
                f"[{pc}]{pnl}[/{pc}]",
                t.reason[:45],
            )


# ── Backtest tab ──────────────────────────────────────────────────────────────

class BacktestTab(Container):
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="bt-config-panel", classes="panel"):
                yield Label("── Backtest Settings ──", classes="panel-title")
                yield Label("Symbol")
                yield Input(value=config.BOT_SYMBOL, id="bt-symbol")
                yield Label("Interval")
                yield Select(
                    [(i, i) for i in config.VALID_INTERVALS],
                    value=config.BOT_INTERVAL, id="bt-interval",
                )
                yield Label("Strategy")
                yield Select(
                    [(k, k) for k in STRATEGIES.keys()],
                    value=config.BOT_STRATEGY, id="bt-strategy",
                )
                yield Label("Lookback Days")
                yield Input(value="90", id="bt-days")
                yield Label("Initial USDT")
                yield Input(value="1000", id="bt-equity")
                yield Label("Stop Loss %  (0 = off)")
                yield Input(value=str(config.BOT_STOP_LOSS_PCT), id="bt-sl")
                yield Label("Take Profit %  (0 = off)")
                yield Input(value=str(config.BOT_TAKE_PROFIT_PCT), id="bt-tp")
                yield Button("▶  Run Backtest", id="bt-run-btn", variant="success")
                yield Static("", id="bt-status")

            with Vertical(id="bt-results-panel", classes="panel"):
                yield Label("── Results ──", classes="panel-title")
                yield Static("", id="bt-summary")
                yield DataTable(id="bt-trades-table", show_cursor=False)

    def on_mount(self) -> None:
        tbl = self.query_one("#bt-trades-table", DataTable)
        tbl.add_columns("Entry", "Exit", "Entry$", "Exit$", "PnL%", "PnL$", "Reason")


# ── Main App ──────────────────────────────────────────────────────────────────

class BinanceTerminal(App):
    """Full interactive Binance trading terminal."""

    CSS = """
    Screen { background: #0d1117; }

    Header { background: #1f2937; color: #f9c74f; }
    Footer { background: #1f2937; }

    .panel {
        border: solid #374151;
        padding: 1 2;
        margin: 1;
        height: auto;
    }
    .panel-title {
        color: #60a5fa;
        text-style: bold;
        margin-bottom: 1;
    }

    #dash-top { height: auto; }
    #ticker-panel    { width: 1fr; min-height: 12; }
    #portfolio-panel { width: 2fr; }
    #bot-status-panel{ width: 1fr; }
    #orders-panel    { height: auto; }

    #place-order-panel  { width: 1fr; }
    #cancel-order-panel { width: 1fr; }

    #bot-config-panel  { width: 1fr; }
    #bot-control-panel { width: 2fr; }

    #bt-config-panel  { width: 1fr; }
    #bt-results-panel { width: 2fr; }

    #bot-btns { height: auto; margin: 1 0; }
    #confirm-btns { height: auto; margin-top: 1; }

    #confirm-box {
        background: #1f2937;
        border: double #f9c74f;
        padding: 2 4;
        width: 60;
        height: auto;
        margin: 10 20;
    }
    #confirm-msg { margin-bottom: 1; }

    Button { margin: 0 1; }
    Input  { margin-bottom: 1; }
    Select { margin-bottom: 1; }
    Label  { color: #9ca3af; margin-bottom: 0; }

    DataTable { height: auto; max-height: 20; }
    Log { height: 30; }
    """

    BINDINGS = [
        Binding("q", "quit",            "Quit"),
        Binding("r", "refresh",         "Refresh"),
        Binding("1", "switch_tab('dashboard')", "Dashboard"),
        Binding("2", "switch_tab('trade')",     "Trade"),
        Binding("3", "switch_tab('bot')",       "Bot"),
        Binding("4", "switch_tab('backtest')",  "Backtest"),
        Binding("5", "switch_tab('activitylog')","Log"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.client: BinanceClient | None = None
        self.bot:    TradingBot    | None = None
        self._refresh_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="tabs", initial="dashboard"):
            with TabPane("Dashboard [1]", id="dashboard"):
                yield DashboardTab(id="dash-tab")
            with TabPane("Trade [2]", id="trade"):
                yield TradeTab(id="trade-tab")
            with TabPane("Bot [3]", id="bot"):
                yield BotTab(id="bot-tab")
            with TabPane("Backtest [4]", id="backtest"):
                yield BacktestTab(id="bt-tab")
            with TabPane("Log [5]", id="activitylog"):
                yield Log(id="activity-log", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Binance Trading Terminal"
        self.sub_title = f"{'TESTNET' if config.USE_TESTNET else 'LIVE'}  ·  {config.BOT_SYMBOL}"
        self._connect()
        self._refresh_timer = self.set_interval(5, self._auto_refresh)

    # ── Connection ─────────────────────────────────────────────────────────────

    @work(thread=True)
    def _connect(self) -> None:
        self._log("Connecting to Binance…")
        try:
            self.client = BinanceClient()
            if not self.client.ping():
                raise ConnectionError("Ping failed.")
            self._log(f"[green]Connected[/green]  server time: {self.client.server_time()}")
            self._create_bot(config.BOT_SYMBOL, config.BOT_INTERVAL, config.BOT_STRATEGY)
        except Exception as e:
            self._log(f"[red]Connection failed: {e}[/red]")

    def _create_bot(self, symbol: str, interval: str, strategy: str) -> None:
        def _on_trade(t: TradeRecord) -> None:
            sc = "green" if t.side == "BUY" else "red"
            self._log(
                f"[{sc}]{t.side}[/{sc}] {t.symbol} @ {_price(t.price)} "
                f"qty={t.qty:.6f}  {t.reason}"
            )
        self.bot = TradingBot(
            client=self.client,
            symbol=symbol,
            interval=interval,
            strategy_name=strategy,
            on_signal=_on_trade,
        )
        self._log(f"Bot ready  symbol={symbol}  interval={interval}  strategy={strategy}")

    # ── Auto-refresh ───────────────────────────────────────────────────────────

    @work(thread=True)
    def _auto_refresh(self) -> None:
        if not self.client:
            return
        dash = self.query_one("#dash-tab", DashboardTab)
        dash.refresh_data(self.client, self.bot)
        bot_tab = self.query_one("#bot-tab", BotTab)
        bot_tab.refresh_trades(self.bot)

    def action_refresh(self) -> None:
        self._auto_refresh()
        self._log("Manual refresh")

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab_id

    # ── Trade tab handlers ────────────────────────────────────────────────────

    @on(Button.Pressed, "#fetch-price-btn")
    def _fetch_price(self) -> None:
        if not self.client:
            return
        sym = self.query_one("#trade-symbol", Input).value.upper()

        @work(thread=True)
        async def _do():
            try:
                p = self.client.get_price(sym)
                self.query_one("#current-price-label", Static).update(
                    f"[yellow]Current price: {_price(p)}[/yellow]"
                )
            except Exception as e:
                self.query_one("#current-price-label", Static).update(f"[red]{e}[/red]")
        _do(self)

    @on(Button.Pressed, "#place-order-btn")
    async def _place_order(self) -> None:
        if not self.client:
            self._log("[red]Not connected.[/red]")
            return
        sym        = self.query_one("#trade-symbol",  Input).value.upper()
        side_sel   = self.query_one("#trade-side",    Select)
        type_sel   = self.query_one("#trade-type",    Select)
        qty_in     = self.query_one("#trade-qty",     Input).value
        price_in   = self.query_one("#trade-price",   Input).value

        side  = str(side_sel.value)
        otype = str(type_sel.value)

        try:
            qty = float(qty_in)
        except ValueError:
            self._log("[red]Invalid quantity.[/red]")
            return

        lp = None
        if otype == "LIMIT":
            try:
                lp = float(price_in)
            except ValueError:
                self._log("[red]Invalid limit price.[/red]")
                return

        msg = f"{side} {qty} {sym} @ {'MARKET' if otype == 'MARKET' else lp}"
        confirmed = await self.push_screen_wait(ConfirmScreen(f"Confirm order:\n{msg}"))
        if not confirmed:
            return

        @work(thread=True)
        async def _do():
            try:
                if otype == "MARKET":
                    o = self.client.market_buy(sym, qty) if side == "BUY" else self.client.market_sell(sym, qty)
                else:
                    o = self.client.limit_buy(sym, qty, lp) if side == "BUY" else self.client.limit_sell(sym, qty, lp)
                self._log(f"[green]Order placed  ID={o.order_id}  status={o.status}[/green]")
            except Exception as e:
                self._log(f"[red]Order failed: {e}[/red]")
        _do(self)

    @on(Button.Pressed, "#load-orders-btn")
    @work(thread=True)
    def _load_open_orders(self) -> None:
        if not self.client:
            return
        sym = self.query_one("#cancel-symbol", Input).value.upper()
        try:
            orders = self.client.get_open_orders(sym)
            tbl    = self.query_one("#cancel-orders-table", DataTable)
            tbl.clear()
            for o in orders:
                sc = "green" if o.side == "BUY" else "red"
                tbl.add_row(
                    str(o.order_id),
                    f"[{sc}]{o.side}[/{sc}]",
                    o.order_type,
                    _price(o.price) if o.price else "MARKET",
                    f"{o.qty:.6f}",
                    o.status,
                )
            if not orders:
                tbl.add_row("—", "—", "—", "—", "—", "No open orders")
            self._log(f"Loaded {len(orders)} open orders for {sym}")
        except Exception as e:
            self._log(f"[red]{e}[/red]")

    @on(Button.Pressed, "#cancel-order-btn")
    async def _cancel_order(self) -> None:
        if not self.client:
            return
        sym     = self.query_one("#cancel-symbol",   Input).value.upper()
        oid_str = self.query_one("#cancel-order-id", Input).value
        try:
            oid = int(oid_str)
        except ValueError:
            self.query_one("#cancel-status", Static).update("[red]Invalid order ID.[/red]")
            return

        confirmed = await self.push_screen_wait(ConfirmScreen(f"Cancel order {oid} on {sym}?"))
        if not confirmed:
            return

        @work(thread=True)
        async def _do():
            ok = self.client.cancel_order(sym, oid)
            self.query_one("#cancel-status", Static).update(
                "[green]Cancelled.[/green]" if ok else "[red]Failed.[/red]"
            )
            self._log(f"Cancel order {oid} → {'OK' if ok else 'FAILED'}")
        _do(self)

    # ── Bot tab handlers ──────────────────────────────────────────────────────

    @on(Button.Pressed, "#bot-start")
    async def _bot_start(self) -> None:
        if not self.client:
            self._log("[red]Not connected.[/red]")
            return

        sym      = self.query_one("#bot-symbol",   Input).value.upper()
        interval = str(self.query_one("#bot-interval", Select).value)
        strategy = str(self.query_one("#bot-strategy", Select).value)

        try:
            config.BOT_TRADE_AMOUNT    = float(self.query_one("#bot-amount", Input).value)
            config.BOT_STOP_LOSS_PCT   = float(self.query_one("#bot-sl",     Input).value)
            config.BOT_TAKE_PROFIT_PCT = float(self.query_one("#bot-tp",     Input).value)
            config.BOT_MAX_OPEN_TRADES = int(self.query_one("#bot-max",      Input).value)
        except ValueError as e:
            self._log(f"[red]Invalid bot config: {e}[/red]")
            return

        confirmed = await self.push_screen_wait(
            ConfirmScreen(f"Start bot?\n{strategy} on {sym} [{interval}]")
        )
        if not confirmed:
            return

        if self.bot and self.bot.state == BotState.RUNNING:
            self.bot.stop()
        self._create_bot(sym, interval, strategy)
        self.bot.start()
        self._log(f"[green]Bot started  {sym}  {strategy}  {interval}[/green]")

    @on(Button.Pressed, "#bot-pause")
    def _bot_pause(self) -> None:
        if self.bot:
            self.bot.pause()
            self._log("[yellow]Bot paused.[/yellow]")

    @on(Button.Pressed, "#bot-resume")
    def _bot_resume(self) -> None:
        if self.bot:
            self.bot.resume()
            self._log("[green]Bot resumed.[/green]")

    @on(Button.Pressed, "#bot-stop")
    async def _bot_stop(self) -> None:
        if not self.bot:
            return
        confirmed = await self.push_screen_wait(ConfirmScreen("Stop the bot?"))
        if confirmed:
            self.bot.stop()
            self._log("[red]Bot stopped.[/red]")

    # ── Backtest handlers ─────────────────────────────────────────────────────

    @on(Button.Pressed, "#bt-run-btn")
    @work(thread=True)
    def _run_backtest(self) -> None:
        if not self.client:
            self._log("[red]Not connected.[/red]")
            return

        sym      = self.query_one("#bt-symbol",   Input).value.upper()
        interval = str(self.query_one("#bt-interval", Select).value)
        strategy = str(self.query_one("#bt-strategy", Select).value)

        try:
            days   = int(float(self.query_one("#bt-days",   Input).value))
            equity = float(self.query_one("#bt-equity", Input).value)
            sl_pct = float(self.query_one("#bt-sl",     Input).value)
            tp_pct = float(self.query_one("#bt-tp",     Input).value)
        except ValueError as e:
            self._log(f"[red]Invalid backtest config: {e}[/red]")
            return

        self.query_one("#bt-status", Static).update("[yellow]Fetching candles…[/yellow]")
        self._log(f"Backtest: {sym} {interval} {strategy} {days}d")

        try:
            from backtest import Backtester
            limit = min(1000, days * 24)
            df    = self.client.get_klines(sym, interval, limit=limit)
            self.query_one("#bt-status", Static).update(
                f"[yellow]Running on {len(df)} candles…[/yellow]"
            )
            strat  = STRATEGIES[strategy]()
            bt     = Backtester(strategy=strat, initial_usdt=equity,
                                stop_loss_pct=sl_pct, take_profit_pct=tp_pct)
            result = bt.run(df, symbol=sym, interval=interval)

            pnl_c = "green" if result.total_pnl_pct >= 0 else "red"
            self.query_one("#bt-summary", Static).update(
                f"Period   : {result.start_date:%Y-%m-%d} → {result.end_date:%Y-%m-%d}\n"
                f"Equity   : ${result.initial_usdt:,.2f} → ${result.final_usdt:,.2f}\n"
                f"Total PnL: [{pnl_c}]{result.total_pnl_pct:+.2f}%  (${result.total_pnl_usdt:+,.2f})[/{pnl_c}]\n"
                f"Drawdown : {result.max_drawdown_pct:.2f}%\n"
                f"Trades   : {result.total_trades}  "
                f"Win [green]{result.winning_trades}[/green] / "
                f"Loss [red]{result.losing_trades}[/red]  "
                f"({result.win_rate:.1f}% win rate)\n"
                f"Avg PnL  : {result.avg_pnl_pct:+.2f}%   "
                f"Best [green]{result.best_trade_pct:+.2f}%[/green]   "
                f"Worst [red]{result.worst_trade_pct:+.2f}%[/red]"
            )

            tbl = self.query_one("#bt-trades-table", DataTable)
            tbl.clear()
            for t in result.trades[-30:]:
                pc = "green" if t.pnl_pct >= 0 else "red"
                tbl.add_row(
                    f"{t.entry_time:%m-%d %H:%M}",
                    f"{t.exit_time:%m-%d %H:%M}",
                    f"{t.entry_price:.2f}",
                    f"{t.exit_price:.2f}",
                    f"[{pc}]{t.pnl_pct:+.2f}%[/{pc}]",
                    f"[{pc}]{t.pnl_usdt:+.2f}[/{pc}]",
                    t.reason_out[:40],
                )
            self.query_one("#bt-status", Static).update("[green]Done.[/green]")
            self._log(f"[green]Backtest complete: {result.total_trades} trades  PnL={result.total_pnl_pct:+.2f}%[/green]")
        except Exception as e:
            self.query_one("#bt-status", Static).update(f"[red]Error: {e}[/red]")
            self._log(f"[red]Backtest error: {e}[/red]")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            self.query_one("#activity-log", Log).write_line(f"[dim]{ts}[/dim]  {msg}")
        except Exception:
            pass

    def action_quit(self) -> None:
        if self.bot and self.bot.state == BotState.RUNNING:
            self.bot.stop()
        self.exit()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_terminal() -> None:
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}")
        return
    BinanceTerminal().run()
