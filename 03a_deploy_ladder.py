"""
03a_deploy_ladder.py -- Deploy the proposed ladder from 02_support_levels.

Compares current pending orders against the proposed ladder and computes
the minimal diff (cancel stale orders, place missing ones).

**Dry-run by default** -- prints the plan but touches nothing.
Pass --execute to actually cancel/place orders on the live account.

Usage:
  python 03a_deploy_ladder.py                  # dry-run: show plan
  python 03a_deploy_ladder.py --execute        # deploy for real
  python 03a_deploy_ladder.py --grid-step 25   # forward args to ladder proposal
  python 03a_deploy_ladder.py --skip-cancel    # only place missing, don't cancel

Reads proposed ladder from 02_support_levels.propose_ladder().
Audit trail saved to data/ladder_deployed.csv after each --execute run.
"""

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

from mt5_client import connect, disconnect
from import_02 import get_pending_orders, detect_supports, propose_ladder, fetch_d1

_ROOT = Path(__file__).parent
_DATA = _ROOT / "data"

SYMBOL = "US500.pro"
MAGIC = 20250101
LOT = 0.001
PRICE_TOLERANCE = 0.5  # orders within this many pts are considered the same


def diff_ladder(current: pd.DataFrame, proposed: pd.DataFrame):
    """
    Compare current pending orders vs proposed ladder.
    Returns (to_cancel, to_place) DataFrames.

    Matching is by price (within PRICE_TOLERANCE pts).
    """
    if current.empty:
        return pd.DataFrame(), proposed.copy()
    if proposed.empty:
        return current.copy(), pd.DataFrame()

    current = current.copy()
    proposed = proposed.copy()
    current["matched"] = False
    proposed["matched"] = False

    # Match current orders to proposed by price
    for i, cur in current.iterrows():
        for j, prop in proposed.iterrows():
            if proposed.at[j, "matched"]:
                continue
            if abs(cur["price"] - prop["price"]) <= PRICE_TOLERANCE:
                current.at[i, "matched"] = True
                proposed.at[j, "matched"] = True
                break

    to_cancel = current[~current["matched"]].drop(columns=["matched"])
    to_place = proposed[~proposed["matched"]].drop(columns=["matched"])

    return to_cancel.reset_index(drop=True), to_place.reset_index(drop=True)


def print_plan(current, proposed, to_cancel, to_place, current_price):
    print(f"\n{'='*70}")
    print(f"  DEPLOYMENT PLAN -- US500.pro @ {current_price:,.1f}")
    print(f"{'='*70}")
    print(f"  Current orders:  {len(current):>4}  ({current['volume'].sum():.3f}L)"
          if not current.empty else "  Current orders:     0  (0.000L)")
    print(f"  Proposed ladder: {len(proposed):>4}  ({proposed['volume'].sum():.3f}L)"
          if not proposed.empty else "  Proposed ladder:    0  (0.000L)")
    print(f"  ─────────────────────────────────")

    matched = len(current) - len(to_cancel) if not current.empty else 0
    print(f"  Keep (matched):  {matched:>4}")
    print(f"  Cancel (stale):  {len(to_cancel):>4}")
    print(f"  Place (new):     {len(to_place):>4}")

    if not to_cancel.empty:
        print(f"\n  CANCEL ({len(to_cancel)} orders):")
        print(f"  {'Ticket':>12}  {'Price':>8}  {'Lots':>6}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*6}")
        for _, o in to_cancel.iterrows():
            ticket_s = f"{o['ticket']:.0f}" if "ticket" in o.index else "?"
            print(f"  {ticket_s:>12}  {o['price']:>8.1f}  {o['volume']:>6.3f}")

    if not to_place.empty:
        print(f"\n  PLACE ({len(to_place)} orders):")
        print(f"  {'Price':>8}  {'Lots':>6}  {'Source':>14}")
        print(f"  {'-'*8}  {'-'*6}  {'-'*14}")
        for _, r in to_place.iterrows():
            src = r.get("source", "?")
            print(f"  {r['price']:>8.1f}  {r['volume']:>6.3f}  {src:>14}")

    print(f"{'='*70}")


def place_buy_limit(price: float, lots: float) -> bool:
    """Place a BUY LIMIT pending order. Never a market order."""
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
        "sl":           0.0,
        "tp":           0.0,
        "deviation":    0,
        "magic":        MAGIC,
        "comment":      "ladder",
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


def cancel_order(ticket: int) -> bool:
    """Cancel a pending order by ticket number."""
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order":  ticket,
    }
    result = mt5.order_send(request)
    if result is None:
        print(f"    ERROR: order_send returned None. {mt5.last_error()}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"    FAILED: retcode={result.retcode}, comment={result.comment}")
        return False
    return True


def execute_plan(to_cancel, to_place):
    """Execute the deployment plan: cancel stale orders, then place new ones."""
    cancelled, placed, errors = 0, 0, 0

    if not to_cancel.empty:
        print(f"\n  Cancelling {len(to_cancel)} orders...")
        for _, o in to_cancel.iterrows():
            ticket = int(o["ticket"])
            ok = cancel_order(ticket)
            if ok:
                cancelled += 1
                print(f"    OK: cancelled {ticket} @ {o['price']:.1f}")
            else:
                errors += 1

    if not to_place.empty:
        print(f"\n  Placing {len(to_place)} orders...")
        for _, r in to_place.iterrows():
            ok = place_buy_limit(r["price"], r["volume"])
            if ok:
                placed += 1
                print(f"    OK: placed {r['volume']:.3f}L @ {r['price']:.1f}")
            else:
                errors += 1

    print(f"\n  Done: {cancelled} cancelled, {placed} placed, {errors} errors.")
    return cancelled, placed, errors


def save_audit_trail(proposed, current_price):
    """Save deployed ladder to data/ladder_deployed.csv."""
    _DATA.mkdir(parents=True, exist_ok=True)
    out = proposed.copy()
    out["deployed_at"] = datetime.now(timezone.utc).isoformat()
    out["current_price"] = current_price
    path = _DATA / "ladder_deployed.csv"
    out.to_csv(path, index=False)
    print(f"  Audit trail saved -> data/ladder_deployed.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy the proposed support-based ladder for US500.pro."
    )
    parser.add_argument("--execute", action="store_true",
                        help="Actually cancel/place orders (default: dry-run only)")
    parser.add_argument("--skip-cancel", action="store_true",
                        help="Only place missing orders, don't cancel existing ones")
    parser.add_argument("--days", type=int, default=250,
                        help="D1 bars for support analysis (default 250)")
    parser.add_argument("--zone-width", type=float, default=20.0,
                        help="Zone merge width (default 20 pts)")
    parser.add_argument("--n-zones", type=int, default=12,
                        help="Number of top zones for clusters (default 12)")
    parser.add_argument("--grid-step", type=float, default=25.0,
                        help="Background grid spacing (default 25 pts)")
    parser.add_argument("--cluster-step", type=float, default=5.0,
                        help="Cluster rung spacing (default 5 pts)")
    parser.add_argument("--floor", type=float, default=6300.0,
                        help="Lowest price to cover (default 6300)")
    args = parser.parse_args()

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        # Fetch data and current state
        print(f"Fetching D1 data ({args.days} bars)...")
        d1 = fetch_d1(args.days)
        if d1.empty:
            print("ERROR: No D1 data available.")
            sys.exit(1)

        tick = mt5.symbol_info_tick(SYMBOL)
        current_price = tick.bid
        print(f"  Current price: {current_price:.1f}")

        current = get_pending_orders()
        print(f"  Current pending: {len(current)} orders"
              + (f" ({current['volume'].sum():.3f}L)" if not current.empty else ""))

        # Build proposed ladder
        print(f"Detecting supports and building ladder...")
        zones = detect_supports(d1, current_price, zone_width=args.zone_width)
        if zones.empty:
            print("ERROR: No support zones detected.")
            sys.exit(1)

        proposed = propose_ladder(
            zones, current_price,
            n_zones=args.n_zones,
            grid_step=args.grid_step,
            cluster_step=args.cluster_step,
            floor=args.floor,
        )
        print(f"  Proposed: {len(proposed)} orders, {proposed['volume'].sum():.3f}L")

        # Compute diff
        to_cancel, to_place = diff_ladder(current, proposed)
        if args.skip_cancel:
            to_cancel = pd.DataFrame()

        print_plan(current, proposed, to_cancel, to_place, current_price)

        if to_cancel.empty and to_place.empty:
            print("\n  Nothing to do -- current orders match the proposed ladder.")
            return

        if not args.execute:
            print("\n  DRY RUN -- no orders were changed.")
            print("  Re-run with --execute to deploy for real.")
            return

        # Execute
        print("\n  EXECUTING deployment...")
        cancelled, placed, errors = execute_plan(to_cancel, to_place)

        if errors == 0:
            save_audit_trail(proposed, current_price)
        else:
            print(f"  WARNING: {errors} errors occurred. "
                  f"Review with: python 03_manage_orders.py list")

    finally:
        disconnect()


if __name__ == "__main__":
    main()
