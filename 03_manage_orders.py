"""
03_manage_orders.py -- Place and cancel pending orders for US500.pro.

Actions:
  place   -- Place a single buy-limit order
  cancel  -- Cancel a pending order by ticket number
  list    -- List all pending orders

Usage:
  python 03_manage_orders.py list
  python 03_manage_orders.py place 6969
  python 03_manage_orders.py place 7000 --lots 0.002
  python 03_manage_orders.py cancel 149818230

Safety: only pending orders (buy-limit). NEVER places market orders.
"""

import sys
import argparse
import MetaTrader5 as mt5
from mt5_client import connect, disconnect

SYMBOL     = "US500.pro"
MAGIC      = 20250101
LOT_DEFAULT = 0.001
SLIPPAGE   = 0


def list_orders():
    orders = mt5.orders_get(symbol=SYMBOL)
    if orders is None or len(orders) == 0:
        print("  No pending orders.")
        return
    type_names = {2: "BUY_LIMIT", 3: "SELL_LIMIT",
                  4: "BUY_STOP", 5: "SELL_STOP"}
    print(f"\n  {'Ticket':>12}  {'Type':>12}  {'Lots':>6}  {'Price':>8}  {'Comment'}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*20}")
    for o in sorted(orders, key=lambda x: x.price_open):
        tname = type_names.get(o.type, str(o.type))
        print(f"  {o.ticket:>12}  {tname:>12}  {o.volume_current:>6.3f}  "
              f"{o.price_open:>8.1f}  {o.comment}")
    print(f"\n  Total: {len(orders)} orders, "
          f"{sum(o.volume_current for o in orders):.3f} lots")


def place_buy_limit(price: float, lots: float):
    """Place a BUY LIMIT pending order. Never a market order."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("ERROR: Cannot get tick data.")
        return False

    if price >= tick.ask:
        print(f"ERROR: Price {price} >= ask {tick.ask}. "
              f"BUY LIMIT must be below current ask price.")
        return False

    request = {
        "action":    mt5.TRADE_ACTION_PENDING,
        "symbol":    SYMBOL,
        "volume":    lots,
        "type":      mt5.ORDER_TYPE_BUY_LIMIT,
        "price":     price,
        "sl":        0.0,
        "tp":        0.0,
        "deviation": SLIPPAGE,
        "magic":     MAGIC,
        "comment":   "ladder",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    print(f"  Sending BUY LIMIT: {lots:.3f}L @ {price:.1f} ...")
    result = mt5.order_send(request)

    if result is None:
        print(f"  ERROR: order_send returned None. {mt5.last_error()}")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  FAILED: retcode={result.retcode}, comment={result.comment}")
        return False

    print(f"  OK: ticket={result.order}, placed BUY LIMIT {lots:.3f}L @ {price:.1f}")
    return True


def cancel_order(ticket: int):
    """Cancel a pending order by ticket number."""
    # Verify the order exists and is pending
    orders = mt5.orders_get(ticket=ticket)
    if orders is None or len(orders) == 0:
        print(f"  ERROR: No pending order with ticket {ticket}.")
        return False

    order = orders[0]
    print(f"  Cancelling: ticket={ticket}, "
          f"{order.volume_current:.3f}L @ {order.price_open:.1f}")

    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order":  ticket,
    }

    result = mt5.order_send(request)

    if result is None:
        print(f"  ERROR: order_send returned None. {mt5.last_error()}")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  FAILED: retcode={result.retcode}, comment={result.comment}")
        return False

    print(f"  OK: order {ticket} cancelled.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Place and cancel pending orders for US500.pro."
    )
    sub = parser.add_subparsers(dest="action")

    sub.add_parser("list", help="List all pending orders")

    p_place = sub.add_parser("place", help="Place a buy-limit order")
    p_place.add_argument("price", type=float, help="Limit price")
    p_place.add_argument("--lots", type=float, default=LOT_DEFAULT,
                         help=f"Lot size (default {LOT_DEFAULT})")

    p_cancel = sub.add_parser("cancel", help="Cancel a pending order")
    p_cancel.add_argument("ticket", type=int, help="Order ticket to cancel")

    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        sys.exit(1)

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        if args.action == "list":
            list_orders()
        elif args.action == "place":
            place_buy_limit(args.price, args.lots)
        elif args.action == "cancel":
            cancel_order(args.ticket)
    finally:
        disconnect()


if __name__ == "__main__":
    main()
