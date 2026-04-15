"""
Backtester — replay historical OHLCV data through any strategy.

Usage (from terminal menu or CLI):
    python backtest.py --symbol BTCUSDT --interval 1h --strategy macd --days 90
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

import config
from client import BinanceClient
from strategies import STRATEGIES, BaseStrategy, Signal, StrategyResult

logger  = logging.getLogger("binance_trader.backtest")
console = Console()


# ── Result models ──────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    entry_time:  datetime
    exit_time:   datetime
    side:        str
    entry_price: float
    exit_price:  float
    qty:         float
    pnl_pct:     float
    pnl_usdt:    float
    reason_in:   str
    reason_out:  str


@dataclass
class BacktestResult:
    symbol:        str
    interval:      str
    strategy:      str
    start_date:    datetime
    end_date:      datetime
    initial_usdt:  float
    final_usdt:    float
    trades:        list[BacktestTrade] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl_usdt > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl_usdt <= 0)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades * 100 if self.total_trades else 0.0

    @property
    def total_pnl_usdt(self) -> float:
        return self.final_usdt - self.initial_usdt

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl_usdt / self.initial_usdt * 100 if self.initial_usdt else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum peak-to-trough drawdown."""
        if not self.trades:
            return 0.0
        equity = self.initial_usdt
        peak   = equity
        max_dd = 0.0
        for t in self.trades:
            equity += t.pnl_usdt
            peak    = max(peak, equity)
            dd      = (peak - equity) / peak * 100
            max_dd  = max(max_dd, dd)
        return max_dd

    @property
    def avg_pnl_pct(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl_pct for t in self.trades) / len(self.trades)

    @property
    def best_trade_pct(self) -> float:
        return max((t.pnl_pct for t in self.trades), default=0.0)

    @property
    def worst_trade_pct(self) -> float:
        return min((t.pnl_pct for t in self.trades), default=0.0)


# ── Engine ─────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Walk-forward backtester.

    At each candle close we feed the strategy ALL candles up to that point
    (rolling window) and act on the returned signal.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_usdt: float        = 1000.0,
        trade_amount_usdt: float   = None,   # None = use full equity
        stop_loss_pct: float       = 0.0,    # 0 = disabled
        take_profit_pct: float     = 0.0,    # 0 = disabled
        fee_pct: float             = 0.1,    # 0.1% maker/taker
        min_candles: int           = 50,     # warm-up candles fed before trading
    ) -> None:
        self.strategy         = strategy
        self.initial_usdt     = initial_usdt
        self.trade_amount_usdt = trade_amount_usdt
        self.sl_pct           = stop_loss_pct
        self.tp_pct           = take_profit_pct
        self.fee_pct          = fee_pct
        self.min_candles      = min_candles

    def run(self, df: pd.DataFrame, symbol: str = "UNKNOWN", interval: str = "?") -> BacktestResult:
        """
        :param df: Full OHLCV dataframe (oldest first).
        :param symbol: Label for the result.
        :param interval: Label for the result.
        """
        result = BacktestResult(
            symbol=symbol,
            interval=interval,
            strategy=self.strategy.name,
            start_date=df["open_time"].iloc[0],
            end_date=df["open_time"].iloc[-1],
            initial_usdt=self.initial_usdt,
            final_usdt=self.initial_usdt,
        )

        equity          = self.initial_usdt
        in_position     = False
        entry_price     = 0.0
        entry_qty       = 0.0
        entry_time: datetime | None = None
        entry_reason    = ""
        sl_price        = 0.0
        tp_price        = 0.0

        for i in range(self.min_candles, len(df)):
            window = df.iloc[: i + 1]
            candle = df.iloc[i]
            close  = float(candle["close"])
            ts     = candle["open_time"]

            # ── Check SL / TP ──────────────────────────────────────────────
            if in_position:
                triggered = None
                if self.sl_pct and close <= sl_price:
                    triggered = ("STOP_LOSS", sl_price)
                elif self.tp_pct and close >= tp_price:
                    triggered = ("TAKE_PROFIT", tp_price)

                if triggered:
                    reason, exit_px = triggered
                    trade, equity = self._close_trade(
                        result, entry_time, ts, entry_price, exit_px,
                        entry_qty, equity, entry_reason, reason,
                    )
                    result.trades.append(trade)
                    in_position = False
                    continue

            # ── Strategy signal ────────────────────────────────────────────
            try:
                sig_result = self.strategy.analyze(window)
            except ValueError:
                continue

            if sig_result.signal == Signal.BUY and not in_position:
                trade_usdt = self.trade_amount_usdt or equity
                trade_usdt = min(trade_usdt, equity)
                fee        = trade_usdt * self.fee_pct / 100
                entry_qty  = (trade_usdt - fee) / close
                entry_price = close
                entry_time  = ts
                entry_reason = sig_result.reason
                sl_price = close * (1 - self.sl_pct   / 100) if self.sl_pct  else 0.0
                tp_price = close * (1 + self.tp_pct   / 100) if self.tp_pct  else 0.0
                in_position = True

            elif sig_result.signal == Signal.SELL and in_position:
                trade, equity = self._close_trade(
                    result, entry_time, ts, entry_price, close,
                    entry_qty, equity, entry_reason, sig_result.reason,
                )
                result.trades.append(trade)
                in_position = False

        # ── Close any open position at end ─────────────────────────────────
        if in_position:
            last_close = float(df["close"].iloc[-1])
            last_ts    = df["open_time"].iloc[-1]
            trade, equity = self._close_trade(
                result, entry_time, last_ts, entry_price, last_close,
                entry_qty, equity, entry_reason, "End of backtest",
            )
            result.trades.append(trade)

        result.final_usdt = equity
        return result

    def _close_trade(
        self,
        result: BacktestResult,
        entry_time,
        exit_time,
        entry_price: float,
        exit_price:  float,
        qty:         float,
        equity:      float,
        reason_in:   str,
        reason_out:  str,
    ):
        gross_usdt = qty * exit_price
        fee        = gross_usdt * self.fee_pct / 100
        net_usdt   = gross_usdt - fee
        pnl_usdt   = net_usdt - (qty * entry_price)
        pnl_pct    = (exit_price - entry_price) / entry_price * 100
        new_equity = equity + pnl_usdt

        trade = BacktestTrade(
            entry_time=entry_time,
            exit_time=exit_time,
            side="LONG",
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            pnl_pct=pnl_pct,
            pnl_usdt=pnl_usdt,
            reason_in=reason_in,
            reason_out=reason_out,
        )
        return trade, new_equity


# ── Display ────────────────────────────────────────────────────────────────────

def print_backtest_result(r: BacktestResult) -> None:
    pnl_color = "green" if r.total_pnl_pct >= 0 else "red"

    # Summary panel
    summary = Table(title=f"Backtest: {r.symbol}  [{r.interval}]  strategy={r.strategy}",
                    box=box.ROUNDED, show_header=False)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value")

    rows = [
        ("Period",         f"{r.start_date:%Y-%m-%d}  →  {r.end_date:%Y-%m-%d}"),
        ("Initial equity", f"${r.initial_usdt:,.2f}"),
        ("Final equity",   f"${r.final_usdt:,.2f}"),
        ("Total PnL",      f"[{pnl_color}]{r.total_pnl_pct:+.2f}%  (${r.total_pnl_usdt:+,.2f})[/]"),
        ("Max Drawdown",   f"{r.max_drawdown_pct:.2f}%"),
        ("Total trades",   str(r.total_trades)),
        ("Win / Loss",     f"[green]{r.winning_trades}[/] / [red]{r.losing_trades}[/]"),
        ("Win rate",       f"{r.win_rate:.1f}%"),
        ("Avg trade PnL",  f"{r.avg_pnl_pct:+.2f}%"),
        ("Best trade",     f"[green]{r.best_trade_pct:+.2f}%[/]"),
        ("Worst trade",    f"[red]{r.worst_trade_pct:+.2f}%[/]"),
    ]
    for k, v in rows:
        summary.add_row(k, v)
    console.print(summary)

    if not r.trades:
        return

    # Trades table (last 20)
    tbl = Table(title="Trades", box=box.SIMPLE, show_header=True)
    tbl.add_column("Entry time")
    tbl.add_column("Exit time")
    tbl.add_column("Entry $", justify="right")
    tbl.add_column("Exit $",  justify="right")
    tbl.add_column("PnL %",   justify="right")
    tbl.add_column("PnL $",   justify="right")
    tbl.add_column("Reason out")

    for t in r.trades[-20:]:
        pnl_col = "green" if t.pnl_pct >= 0 else "red"
        tbl.add_row(
            f"{t.entry_time:%Y-%m-%d %H:%M}",
            f"{t.exit_time:%Y-%m-%d %H:%M}",
            f"{t.entry_price:.2f}",
            f"{t.exit_price:.2f}",
            f"[{pnl_col}]{t.pnl_pct:+.2f}%[/]",
            f"[{pnl_col}]{t.pnl_usdt:+.2f}[/]",
            t.reason_out[:50],
        )
    console.print(tbl)


# ── CLI entry point ────────────────────────────────────────────────────────────

def run_backtest_menu(client: BinanceClient) -> None:
    """Interactive backtest runner called from the terminal menu."""
    from rich.prompt import Prompt

    symbol   = Prompt.ask("Symbol",   default=config.BOT_SYMBOL).upper()
    interval = Prompt.ask("Interval", default=config.BOT_INTERVAL)
    strategy_name = Prompt.ask("Strategy", choices=list(STRATEGIES.keys()), default=config.BOT_STRATEGY)
    days     = int(Prompt.ask("Lookback days", default="90"))
    equity   = float(Prompt.ask("Initial USDT", default="1000"))
    sl_pct   = float(Prompt.ask("Stop-loss %  (0=off)", default=str(config.BOT_STOP_LOSS_PCT)))
    tp_pct   = float(Prompt.ask("Take-profit % (0=off)", default=str(config.BOT_TAKE_PROFIT_PCT)))

    limit    = min(1000, days * 24)   # rough candle count
    console.print(f"Fetching {limit} candles for {symbol} [{interval}]…")
    df = client.get_klines(symbol, interval, limit=limit)
    console.print(f"Got {len(df)} candles from {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")

    strategy = STRATEGIES[strategy_name]()
    bt = Backtester(
        strategy=strategy,
        initial_usdt=equity,
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
    )
    result = bt.run(df, symbol=symbol, interval=interval)
    print_backtest_result(result)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest a trading strategy on Binance historical data")
    ap.add_argument("--symbol",   default=config.BOT_SYMBOL)
    ap.add_argument("--interval", default=config.BOT_INTERVAL)
    ap.add_argument("--strategy", default=config.BOT_STRATEGY, choices=list(STRATEGIES.keys()))
    ap.add_argument("--days",     type=int,   default=90)
    ap.add_argument("--equity",   type=float, default=1000.0)
    ap.add_argument("--sl",       type=float, default=config.BOT_STOP_LOSS_PCT,   help="Stop-loss %")
    ap.add_argument("--tp",       type=float, default=config.BOT_TAKE_PROFIT_PCT, help="Take-profit %")
    args = ap.parse_args()

    config.setup_logging()
    errors = config.validate()
    if errors:
        for e in errors:
            console.print(f"[red]{e}[/red]")
        return

    client   = BinanceClient()
    limit    = min(1000, args.days * 24)
    console.print(f"Fetching {limit} candles for {args.symbol} [{args.interval}]…")
    df = client.get_klines(args.symbol, args.interval, limit=limit)

    strategy = STRATEGIES[args.strategy]()
    bt = Backtester(
        strategy=strategy,
        initial_usdt=args.equity,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
    )
    result = bt.run(df, symbol=args.symbol, interval=args.interval)
    print_backtest_result(result)


if __name__ == "__main__":
    main()
