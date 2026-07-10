"""Account snapshot and open position info."""

import MetaTrader5 as mt5
import pandas as pd
from config import SYMBOL, MAGIC


def snapshot() -> dict:
    info = mt5.account_info()
    return {
        "balance":      info.balance,
        "equity":       info.equity,
        "profit":       info.profit,
        "margin":       info.margin,
        "margin_free":  info.margin_free,
        "margin_level": info.margin_level,
        "currency":     info.currency,
    }


def print_snapshot():
    s = snapshot()
    print("\n=== ACCOUNT ===")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:<14}: {v:.2f}")
        else:
            print(f"  {k:<14}: {v}")


def open_positions(symbol: str = SYMBOL) -> pd.DataFrame:
    """Return all open positions for the given symbol."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return pd.DataFrame()
    rows = []
    for p in positions:
        rows.append({
            "ticket":   p.ticket,
            "type":     "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume":   p.volume,
            "open_price": p.price_open,
            "current_price": p.price_current,
            "sl":       p.sl,
            "tp":       p.tp,
            "profit":   p.profit,
            "magic":    p.magic,
            "comment":  p.comment,
        })
    return pd.DataFrame(rows)


def bot_positions(symbol: str = SYMBOL) -> pd.DataFrame:
    """Open positions placed by this bot (matched by MAGIC number)."""
    df = open_positions(symbol)
    if df.empty:
        return df
    return df[df["magic"] == MAGIC].reset_index(drop=True)
