"""
04_rollover_close.py -- Close all US500.pro positions before rollover.

Market-sells every open US500.pro position. Designed for rollover day
(next: Sep 16, 2026) — close before 22:55 Polish time to avoid the
quarterly swap charge.

**Dry-run by default.** Pass --execute to actually close positions.

Usage:
  python 04_rollover_close.py              # show what would close
  python 04_rollover_close.py --execute    # close all positions for real
"""

import sys
import argparse
import MetaTrader5 as mt5

from mt5_client import connect, disconnect

SYMBOL = "US500.pro"
MAGIC = 20250101


def get_positions():
    raw = mt5.positions_get()
    if raw is None or not len(raw):
        return []
    return [p for p in raw if p.symbol == SYMBOL]


def close_position(position) -> bool:
    """Market-close a single position."""
    # Determine close direction
    if position.type == 0:  # long -> sell to close
        close_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).bid
    else:  # short -> buy to close
        close_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(SYMBOL).ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       position.volume,
        "type":         close_type,
        "position":     position.ticket,
        "price":        price,
        "deviation":    5,
        "magic":        MAGIC,
        "comment":      "rollover close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        print(f"    ERROR: order_send returned None. {mt5.last_error()}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"    FAILED: ticket={position.ticket}, "
              f"retcode={result.retcode}, comment={result.comment}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Close all US500.pro positions before rollover."
    )
    parser.add_argument("--execute", action="store_true",
                        help="Actually close positions (default: dry-run)")
    args = parser.parse_args()

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        info = mt5.account_info()
        positions = get_positions()

        if not positions:
            print("  No US500.pro positions found. Nothing to close.")
            return

        total_lots = sum(p.volume for p in positions)
        total_profit = sum(p.profit for p in positions)
        total_swap = sum(p.swap for p in positions)

        print(f"\n{'='*70}")
        print(f"  ROLLOVER CLOSE -- US500.pro")
        print(f"{'='*70}")
        print(f"  Positions:    {len(positions)}")
        print(f"  Total lots:   {total_lots:.3f}")
        print(f"  Unrealized:   {total_profit:+,.2f} {info.currency}")
        print(f"  Swap:         {total_swap:+,.2f} {info.currency}")
        print(f"  Net:          {total_profit + total_swap:+,.2f} {info.currency}")
        print(f"{'='*70}")

        print(f"\n  {'Ticket':>12}  {'Type':>5}  {'Lots':>7}  "
              f"{'Entry':>8}  {'Current':>8}  {'Profit':>10}  {'Swap':>8}")
        print(f"  {'-'*12}  {'-'*5}  {'-'*7}  "
              f"{'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
        for p in positions:
            ptype = "LONG" if p.type == 0 else "SHORT"
            print(f"  {p.ticket:>12}  {ptype:>5}  {p.volume:>7.3f}  "
                  f"{p.price_open:>8.1f}  {p.price_current:>8.1f}  "
                  f"{p.profit:>+10.2f}  {p.swap:>+8.2f}")

        if not args.execute:
            print(f"\n  DRY RUN -- no positions were closed.")
            print(f"  Re-run with --execute to close all {len(positions)} positions.")
            return

        print(f"\n  CLOSING {len(positions)} positions ({total_lots:.3f} lots)...")
        closed, errors = 0, 0
        for p in positions:
            ok = close_position(p)
            if ok:
                closed += 1
                print(f"    OK: closed ticket {p.ticket} "
                      f"({p.volume:.3f}L @ {p.price_open:.1f})")
            else:
                errors += 1

        print(f"\n  Done: {closed} closed, {errors} errors.")
        if errors:
            print(f"  WARNING: {errors} positions failed to close. Check MT5.")

    finally:
        disconnect()


if __name__ == "__main__":
    main()
