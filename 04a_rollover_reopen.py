"""
04a_rollover_reopen.py -- Reopen position after rollover via buy-limit ladder.

Places buy-limit orders starting just below --price, stepping down 1 pt
per order, each for --chunk lots, until --lots total is covered.

Example: --price 7899 --lots 0.260 --chunk 0.050
  0.050L @ 7899
  0.050L @ 7898
  0.050L @ 7897
  0.050L @ 7896
  0.050L @ 7895
  0.010L @ 7894   (remainder)

**Dry-run by default.** Pass --execute to place orders.

Usage:
  python 04a_rollover_reopen.py --price 7899 --lots 0.260                # dry-run
  python 04a_rollover_reopen.py --price 7899 --lots 0.260 --execute      # place orders
  python 04a_rollover_reopen.py --price 7899 --lots 0.260 --chunk 0.030  # smaller chunks
  python 04a_rollover_reopen.py --price 7899 --lots 0.260 --step 2       # 2-pt spacing
"""

import sys
import argparse
import MetaTrader5 as mt5

from mt5_client import connect, disconnect

SYMBOL = "US500.pro"
MAGIC = 20250101


def build_orders(price, total_lots, chunk, step):
    """Build a list of (price, volume) tuples for the reopen ladder."""
    orders = []
    remaining = round(total_lots, 3)
    p = price

    while remaining > 0.0005:  # lot step is 0.001
        vol = min(chunk, remaining)
        vol = round(vol, 3)
        if vol < 0.001:
            break
        orders.append((p, vol))
        remaining = round(remaining - vol, 3)
        p -= step

    return orders


def place_buy_limit(price, lots) -> bool:
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print(f"    ERROR: Cannot get tick data.")
        return False

    if price >= tick.ask:
        print(f"    SKIP: Price {price} >= ask {tick.ask} (would fill immediately).")
        return False

    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       SYMBOL,
        "volume":       lots,
        "type":         mt5.ORDER_TYPE_BUY_LIMIT,
        "price":        price,
        "deviation":    0,
        "magic":        MAGIC,
        "comment":      "rollover reopen",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    if result is None:
        print(f"    ERROR: order_send returned None. {mt5.last_error()}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"    FAILED: retcode={result.retcode}, comment={result.comment}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Reopen US500.pro position after rollover via buy-limit ladder."
    )
    parser.add_argument("--price", type=float, required=True,
                        help="Starting price for the top buy-limit order")
    parser.add_argument("--lots", type=float, required=True,
                        help="Total lots to reopen (e.g. 0.260)")
    parser.add_argument("--chunk", type=float, default=0.050,
                        help="Lots per order (default 0.050)")
    parser.add_argument("--step", type=float, default=1.0,
                        help="Price step between orders in pts (default 1)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually place orders (default: dry-run)")
    args = parser.parse_args()

    orders = build_orders(args.price, args.lots, args.chunk, args.step)

    if not orders:
        print("No orders to place (check --lots and --chunk).")
        sys.exit(1)

    total = sum(v for _, v in orders)
    price_range = f"{orders[0][0]:.1f} -> {orders[-1][0]:.1f}"

    print(f"\n{'='*60}")
    print(f"  ROLLOVER REOPEN -- US500.pro")
    print(f"{'='*60}")
    print(f"  Orders:     {len(orders)}")
    print(f"  Total lots: {total:.3f}")
    print(f"  Chunk:      {args.chunk:.3f}L per order")
    print(f"  Prices:     {price_range} (step {args.step} pt)")
    print(f"{'='*60}")

    print(f"\n  {'#':>3}  {'Price':>8}  {'Lots':>7}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*7}")
    for i, (p, v) in enumerate(orders):
        print(f"  {i+1:>3}  {p:>8.1f}  {v:>7.3f}")

    print(f"  {'':>3}  {'':>8}  {'-'*7}")
    print(f"  {'':>3}  {'Total':>8}  {total:>7.3f}")

    if not args.execute:
        print(f"\n  DRY RUN -- no orders were placed.")
        print(f"  Re-run with --execute to place {len(orders)} buy-limit orders.")
        return

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        print(f"\n  PLACING {len(orders)} buy-limit orders...")
        placed, errors = 0, 0
        for p, v in orders:
            ok = place_buy_limit(p, v)
            if ok:
                placed += 1
                print(f"    OK: {v:.3f}L @ {p:.1f}")
            else:
                errors += 1

        print(f"\n  Done: {placed} placed, {errors} errors.")
        if errors:
            print(f"  WARNING: {errors} orders failed. Check MT5.")

    finally:
        disconnect()


if __name__ == "__main__":
    main()
