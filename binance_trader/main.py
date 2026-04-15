#!/usr/bin/env python3
"""
Binance Trading Terminal — entry point.

Usage:
    python main.py              # interactive terminal + bot
    python main.py --dashboard  # jump straight to dashboard (uses .env defaults)
    python main.py --bot        # headless bot only (no menu)
"""

from __future__ import annotations

import argparse
import sys

import config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance Trading Terminal + Bot")
    p.add_argument("--dashboard", action="store_true",
                   help="Open live dashboard immediately (uses .env defaults)")
    p.add_argument("--bot",       action="store_true",
                   help="Run bot headlessly — no interactive menu")
    p.add_argument("--symbol",    default=None, help="Override BOT_SYMBOL")
    p.add_argument("--interval",  default=None, help="Override BOT_INTERVAL")
    p.add_argument("--strategy",  default=None, help="Override BOT_STRATEGY (ma_crossover | rsi)")
    return p.parse_args()


def _headless_bot(symbol: str | None, interval: str | None, strategy: str | None) -> None:
    """Run the bot without an interactive terminal — useful in a tmux/screen session."""
    import signal
    import time
    from rich.console import Console
    from client import BinanceClient
    from bot import TradingBot, TradeRecord

    console = Console()
    logger  = config.setup_logging()

    errors = config.validate()
    if errors:
        for e in errors:
            console.print(f"[red]{e}[/red]")
        sys.exit(1)

    client = BinanceClient()
    if not client.ping():
        console.print("[red]Cannot reach Binance API.[/red]")
        sys.exit(1)

    def _on_trade(t: TradeRecord) -> None:
        side_col = "green" if t.side == "BUY" else "red"
        console.print(
            f"[bold]{t.time:%Y-%m-%d %H:%M:%S}[/bold]  "
            f"[{side_col}]{t.side}[/]  {t.symbol}  "
            f"@ {t.price:.4f}  qty={t.qty:.6f}  {t.reason}"
        )

    bot = TradingBot(
        client=client,
        symbol=symbol,
        interval=interval,
        strategy_name=strategy,
        on_signal=_on_trade,
    )
    bot.start()
    console.print(f"[green]Headless bot running. Symbol={bot.symbol}  Strategy={bot.strategy.name}[/green]")
    console.print("Press Ctrl-C to stop.")

    def _handle_sigterm(*_):
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        while True:
            time.sleep(60)
            s = bot.status_dict()
            console.print(
                f"  [dim]{s['state']}  trades={s['total_trades']}  "
                f"win={s['win_rate']}  pnl={s['total_pnl']}  uptime={s['uptime']}[/dim]"
            )
    except KeyboardInterrupt:
        bot.stop()
        console.print("[yellow]Bot stopped.[/yellow]")


def main() -> None:
    args = _parse_args()

    # Override .env settings from CLI flags
    if args.symbol:
        config.BOT_SYMBOL   = args.symbol.upper()
    if args.interval:
        config.BOT_INTERVAL = args.interval
    if args.strategy:
        config.BOT_STRATEGY = args.strategy

    config.setup_logging()

    if args.bot:
        _headless_bot(args.symbol, args.interval, args.strategy)
        return

    # Interactive terminal (default)
    from terminal import run_terminal
    run_terminal()


if __name__ == "__main__":
    main()
