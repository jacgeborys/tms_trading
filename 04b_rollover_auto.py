"""
04b_rollover_auto.py -- Unattended rollover close + reopen (Windows task "TMS rollover close").

The task runs Mon-Fri at 22:50 local time. On any day that is not an OANDA
rollover day (Wednesday before the 3rd Friday of Mar/Jun/Sep/Dec) it logs one
line and exits. On rollover day:

  1. checks MT5 server time is inside the close window (22:40-22:57),
     otherwise aborts (never closes at the wrong moment)
  2. cancels all pending US500.pro orders, so nothing fills before the cutoff
  3. closes all positions via 04_rollover_close.py --execute
  4. waits until 22:57 server time and places buy-limits for the same total
     lots, REOPEN_OFFSETS points below the closing ask; they fill when trading
     resumes (~00:00) if the price opens at or below them
  5. from the next trading day after the rollover week, 03b's daily re-fit
     replaces any reopen order that never filled with the normal ladder

The swap is charged on positions open after 22:59 Polish time (confirmed by
OANDA). Log: logs/04b_rollover_auto.log.

Usage:
  python 04b_rollover_auto.py                          # what the task runs
  python 04b_rollover_auto.py --dry-run --date 2026-12-16 --no-time-check
                                                       # rehearse on any day
"""

import sys
import time
import argparse
import importlib.util
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent
_LOG = _ROOT / "logs" / "04b_rollover_auto.log"

CLOSE_WINDOW = ((22, 40), (22, 57))   # server time, inclusive
REOPEN_AT = (22, 57)                  # server time to place reopen limits
REOPEN_OFFSETS = [1, 2, 3, 4, 5]      # points below the closing ask
LOT_STEP = 0.001


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), _ROOT / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_lots(lots, n):
    """Split lots into up to n chunks of whole LOT_STEP units, larger first."""
    units = int(round(lots / LOT_STEP))
    n = max(1, min(n, units))
    base, rest = divmod(units, n)
    return [round((base + (1 if i < rest else 0)) * LOT_STEP, 3) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
    ap.add_argument("--date", help="pretend today is YYYY-MM-DD (testing)")
    ap.add_argument("--no-time-check", action="store_true",
                    help="skip the server-time window check and the wait (testing)")
    args = ap.parse_args()

    today = date.fromisoformat(args.date) if args.date else date.today()
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(_LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    print(f"\n{'#'*70}\n# {datetime.now():%Y-%m-%d %H:%M:%S}  "
          f"{'DRY RUN' if args.dry_run else 'EXECUTE'}  (date {today})\n{'#'*70}")

    rollover_date = _load("03b_daily_refit.py").rollover_date
    if today.month not in (3, 6, 9, 12) or today != rollover_date(today.year, today.month):
        print("Not a rollover day -- nothing to do.")
        return

    import MetaTrader5 as mt5
    from config import SYMBOL
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()} -- ABORTED. Close manually before 22:59!")
        return
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        offset = tick.time - time.time()            # server clock - local clock (s)
        server = datetime.utcfromtimestamp(time.time() + offset)
        print(f"Server time {server:%H:%M:%S}, local {datetime.now():%H:%M:%S}")
        lo, hi = CLOSE_WINDOW
        if not args.no_time_check and not (lo <= (server.hour, server.minute) <= hi):
            print(f"Server time outside {lo[0]}:{lo[1]:02d}-{hi[0]}:{hi[1]:02d} -- ABORTED, "
                  f"nothing changed. Close manually before 22:59 if still in time.")
            return

        positions = [p for p in (mt5.positions_get(symbol=SYMBOL) or [])]
        orders = [o for o in (mt5.orders_get(symbol=SYMBOL) or [])]
        lots = round(sum(p.volume for p in positions), 3)
        print(f"Open: {len(positions)} positions, {lots:.3f} lots | "
              f"pending: {len(orders)} orders")

        # 1. cancel pending orders
        for o in orders:
            if args.dry_run:
                print(f"  would cancel {o.ticket} @ {o.price_open}")
                continue
            r = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
            ok = r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
            print(f"  {'OK' if ok else 'FAILED'}: cancel {o.ticket} @ {o.price_open}"
                  + ("" if ok else f" ({getattr(r, 'comment', mt5.last_error())})"))

        if not positions:
            print("No positions to close.")
            return

        # 2. close positions (04_rollover_close.py has the retry logic)
        close_ask = mt5.symbol_info_tick(SYMBOL).ask
        if args.dry_run:
            print(f"  would close {lots:.3f} lots at ~{close_ask:.1f} (ask)")
        else:
            mt5.shutdown()
            subprocess.run([sys.executable, str(_ROOT / "04_rollover_close.py"), "--execute"],
                           cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT)
            mt5.initialize()
            close_ask = mt5.symbol_info_tick(SYMBOL).ask
            left = mt5.positions_get(symbol=SYMBOL) or []
            if left:
                print(f"WARNING: {len(left)} positions still open -- close manually before 22:59!")
            lots = round(lots - sum(p.volume for p in left), 3)

        # 3. wait until REOPEN_AT server time, then place reopen limits
        if not args.no_time_check and not args.dry_run:
            while True:
                s = datetime.utcfromtimestamp(time.time() + offset)
                if (s.hour, s.minute) >= REOPEN_AT:
                    break
                time.sleep(5)
            close_ask = mt5.symbol_info_tick(SYMBOL).ask
        chunks = split_lots(lots, len(REOPEN_OFFSETS))
        print(f"Reopen {lots:.3f} lots below ask {close_ask:.1f}:")
        place = _load("04a_rollover_reopen.py").place_buy_limit
        for off, vol in zip(REOPEN_OFFSETS, chunks):
            price = round(close_ask - off, 1)
            if args.dry_run:
                print(f"  would place {vol:.3f}L @ {price}")
            else:
                ok = place(price, vol)
                print(f"  {'OK' if ok else 'FAILED'}: buy-limit {vol:.3f}L @ {price}")
        print("Done. 03b's daily re-fit takes over after the rollover week.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
