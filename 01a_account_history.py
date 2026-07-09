"""
01a_account_history.py — Account balance, equity and margin health.

Reconstructs the account equity curve since each position was opened,
using cached M5 price data.  Produces a stacked band chart showing:

  ┌──────────────────────────────────────────────────┐
  │  equity (line)  — balance + unrealised P&L       │
  │ ░░░░ Free margin  (equity − margin used)         │
  │ ████ Margin used  (capital locked by broker)     │
  └──────────────────────────────────────────────────┘

Also shows:
  - Drawdown from equity peak
  - Margin level %  (with 100 % / 200 % danger zones)
  - Deposit / withdrawal markers from deal history

Usage:
  python 01a_account_history.py

Outputs:
  results/charts/01a_account_history.png
"""

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import MetaTrader5 as mt5

from mt5_client import connect, disconnect
import cache

DEAL_TYPE_BALANCE = 2
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


# ── Data fetching ──────────────────────────────────────────────────────────────

DEAL_TYPE_NAMES = {
    0:  "BUY",
    1:  "SELL",
    2:  "BALANCE",
    3:  "CREDIT",
    4:  "CHARGE",
    5:  "CORRECTION",
    6:  "BONUS",
    7:  "COMMISSION",
    8:  "COMMISSION_DAILY",
    9:  "COMMISSION_MONTHLY",
    10: "COMMISSION_AGENT_DAILY",
    11: "COMMISSION_AGENT_MONTHLY",
    12: "INTEREST",
    13: "BUY_CANCELED",
    14: "SELL_CANCELED",
    15: "DIVIDEND",
    16: "DIVIDEND_FRANKED",
    17: "TAX",
}

DEAL_ENTRY_NAMES = {0: "IN", 1: "OUT", 2: "INOUT", 3: "OUT_BY"}


def fetch_deal_history():
    """
    Fetch all account deals and split into two DataFrames:
      bal_events  — deposits / withdrawals / adjustments (non-trade deal types)
      trade_deals — closed positions (DEAL_ENTRY_OUT / DEAL_ENTRY_INOUT)

    Also returns:
      open_deals  — position-open legs (DEAL_ENTRY_IN)
      all_deals   — every raw deal as a flat DataFrame for auditing
    """
    now = datetime.now(tz=timezone.utc)
    raw = mt5.history_deals_get(EPOCH, now)
    if raw is None or not len(raw):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bal_rows, trade_rows, open_rows, all_rows = [], [], [], []
    for d in raw:
        t = pd.Timestamp(d.time, unit="s", tz="UTC")
        all_rows.append({
            "time":        t,
            "ticket":      d.ticket,
            "order":       d.order,
            "position_id": d.position_id,
            "type":        d.type,
            "type_name":   DEAL_TYPE_NAMES.get(d.type, f"UNKNOWN({d.type})"),
            "entry":       d.entry,
            "entry_name":  DEAL_ENTRY_NAMES.get(d.entry, f"?({d.entry})"),
            "symbol":      d.symbol,
            "volume":      d.volume,
            "price":       d.price,
            "profit":      d.profit,
            "commission":  d.commission,
            "swap":        d.swap,
            "fee":         d.fee,
            "comment":     d.comment,
        })
        if d.type != 0 and d.type != 1:   # non-trade deal type → balance/adjustment event
            bal_rows.append({"time": t, "amount": d.profit,
                             "type": d.type,
                             "type_name": DEAL_TYPE_NAMES.get(d.type, f"UNKNOWN({d.type})"),
                             "comment": d.comment})
        elif d.entry in (1, 2):            # DEAL_ENTRY_OUT / INOUT — closed trade
            trade_rows.append({
                "time":        t,
                "position_id": d.position_id,
                "symbol":      d.symbol,
                "volume":      d.volume,
                "profit":      d.profit,
                "commission":  d.commission,
                "swap":        d.swap,
                "fee":         d.fee,
                "net":         d.profit + d.commission + d.swap + d.fee,
            })
        elif d.entry == 0:                 # DEAL_ENTRY_IN — position opened
            open_rows.append({
                "time":        t,
                "position_id": d.position_id,
                "symbol":      d.symbol,
                "volume":      d.volume,
                "price_open":  d.price,
                "direction":   1.0 if d.type == 0 else -1.0,
            })

    def _df(rows):
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    return _df(bal_rows), _df(trade_rows), _df(open_rows), _df(all_rows)


def get_open_positions() -> pd.DataFrame:
    raw = mt5.positions_get()
    if raw is None or not len(raw):
        return pd.DataFrame()
    rows = []
    for p in raw:
        rows.append({
            "ticket":      p.ticket,
            "position_id": p.identifier,   # matches position_id in deal history
            "symbol":      p.symbol,
            "type":        p.type,         # 0=buy 1=sell
            "volume":      p.volume,
            "price_open":  p.price_open,
            "time_open":   pd.Timestamp(p.time, unit="s", tz="UTC"),
            "profit":      p.profit,
            "swap":        p.swap,
        })
    return pd.DataFrame(rows)


# ── Equity reconstruction ──────────────────────────────────────────────────────

def reconstruct_equity(positions: pd.DataFrame, bal_events: pd.DataFrame,
                       trade_deals: pd.DataFrame, open_deals: pd.DataFrame,
                       margin_used: float, leverage: int,
                       t_start: pd.Timestamp = None,
                       rollover_costs: dict = None) -> pd.DataFrame:
    """
    Build a per-bar time series of equity / free_margin / margin_level
    from the earliest open position to now, using cached M5 price data.

    Balance is reconstructed as a step function from deposit/withdrawal
    events — so top-ups during a drawdown are reflected correctly.

    Margin is recalculated each bar as:
        sum(current_price × volume × contract_size / leverage)
    This matches how OANDA computes margin requirements (on current price,
    not entry price), so the margin level accurately reflects how close
    to a stop-out the account was at every point in time.

    Returns an empty DataFrame if no cached data is available.
    """
    # t_start: start of the chart window (from --months), not the position open time
    # This lets us show a flat equity = balance before any position was opened
    if t_start is None:
        t_start = positions["time_open"].min()

    # Load price data for every symbol — prefer M5, fall back to H1 when M5
    # doesn't reach back to t_start (MT5 only stores ~8.5 months of M5 locally)
    # Normalise t_start to tz-naive so all comparisons below are consistent
    if t_start is not None and t_start.tzinfo is not None:
        t_start = t_start.tz_localize(None)

    symbol_data = {}
    for sym in positions["symbol"].unique():
        for tf in ("M5", "H1"):
            df = cache.load_ohlcv(sym, tf)
            if df is None:
                continue
            # Normalise to tz-naive UTC — avoids mixed-tz comparison bugs
            if df["time"].dt.tz is not None:
                df["time"] = df["time"].dt.tz_localize(None)
            # Only use this TF if its earliest bar covers t_start
            if t_start is not None and df["time"].iloc[0] > t_start:
                print(f"  {tf} starts {str(df['time'].iloc[0])[:10]} — too late, trying H1")
                continue
            df = df[df["time"] >= t_start].reset_index(drop=True) if t_start is not None else df
            if len(df) == 0:
                continue
            symbol_data[sym] = df
            print(f"  Using {tf} data for {sym}  ({len(df):,} bars from "
                  f"{str(df['time'].iloc[0])[:10]})")
            break
        if sym not in symbol_data:
            print(f"  WARNING: No cached data for {sym} — run 01_fetch_data.py first")

    if not symbol_data:
        return pd.DataFrame()

    # Master time axis from the first symbol
    master = symbol_data[list(symbol_data.keys())[0]]
    times  = master["time"].values
    n      = len(times)

    def _step_series(events: pd.DataFrame, amount_col: str) -> np.ndarray:
        """Build a cumulative step-function array over `times` from event rows."""
        out = np.zeros(n)
        if events.empty:
            return out
        ev_times   = events["time"].dt.tz_localize(None).values
        ev_amounts = events[amount_col].values
        running, ev_idx = 0.0, 0
        for i, t in enumerate(times):
            while ev_idx < len(ev_times) and ev_times[ev_idx] <= t:
                running  += ev_amounts[ev_idx]
                ev_idx   += 1
            out[i] = running
        return out

    # ── Deposited capital (cash in/out) and realised trading P&L ─────────────
    deposited = _step_series(bal_events, "amount")   # steps on deposits/withdrawals
    realised  = _step_series(trade_deals, "net")     # steps on every closed trade
    balance   = deposited + realised

    # ── Per-bar unrealised P&L and margin ────────────────────────────────────
    total_upnl      = np.zeros(n)
    total_swap_upnl = np.zeros(n)   # swap portion only — for no-swap equity line
    total_margin    = np.zeros(n)

    # margin_scale: converts price × vol × cs / leverage (USD) → account currency (PLN).
    # Calibrated using the CURRENT close price (≈ current market price) so that the
    # scale factor represents the effective USD/PLN rate rather than an entry-price artefact.
    # Historical margin then floats correctly with market price, matching how OANDA
    # calculates margin requirements on the current price at each point in time.
    formula_margin_now = 0.0
    for _, pos in positions.iterrows():
        si = mt5.symbol_info(pos["symbol"])
        contract_size = si.trade_contract_size if si else 50.0
        sym = pos["symbol"]
        last_close = (float(symbol_data[sym]["close"].iloc[-1])
                      if sym in symbol_data else float(pos["price_open"]))
        formula_margin_now += last_close * pos["volume"] * contract_size / leverage
    margin_scale = (margin_used / formula_margin_now) if formula_margin_now > 0 else 1.0

    for _, pos in positions.iterrows():
        sym = pos["symbol"]
        if sym not in symbol_data:
            continue

        si            = mt5.symbol_info(sym)
        contract_size = si.trade_contract_size if si else 50.0
        direction     = 1.0 if pos["type"] == 0 else -1.0

        price_df  = symbol_data[sym]
        closes    = price_df["close"].values
        bar_times = price_df["time"].values

        pos_open = pos["time_open"]
        if pos_open.tzinfo is not None:
            pos_open = pos_open.tz_localize(None)
        active = bar_times >= np.datetime64(pos_open)

        # Option C: anchor upnl to actual PLN profit from MT5.
        # p.profit is the price-movement P&L only (swap is separate in p.swap).
        # We scale it by the price-movement ratio so that at the last bar
        # price_upnl == p.profit exactly, and intermediate bars are proportional.
        # USDPLN cancels out — no exchange rate data needed.
        profit_pln = float(pos["profit"])
        last_close  = float(closes[-1])
        delta       = last_close - pos["price_open"]
        if abs(delta) > 0.1:
            price_ratio = (closes - pos["price_open"]) / delta
            upnl = np.where(active, profit_pln * price_ratio, 0.0)
        else:
            upnl = np.zeros(n)
        total_upnl += upnl

        # Step swap in at each rollover date the position lived through.
        # This produces visible equity drops on rollover dates rather than a
        # flat offset from position open.  Costs are back-calculated per-rollover
        # and scaled to match the actual accumulated swap exactly.
        swap_pln = float(pos.get("swap", 0.0))
        if swap_pln != 0.0:
            pos_open_tz = pos["time_open"]
            if pos_open_tz.tzinfo is None:
                pos_open_tz = pos_open_tz.tz_localize("UTC")
            applicable = sorted(
                [(r, c) for r, c in (rollover_costs or {}).items() if r > pos_open_tz]
            )
            if applicable:
                vol         = float(pos["volume"])
                model_total = sum(c * (vol / 0.001) for _, c in applicable)
                scale       = swap_pln / model_total if abs(model_total) > 1e-6 else 1.0
                for rdate, cost_per_001 in applicable:
                    # tz_convert(None): convert UTC→naive UTC (safe for tz-aware Timestamps)
                    r_naive  = np.datetime64(rdate.tz_convert(None))
                    cost_pln = cost_per_001 * (vol / 0.001) * scale
                    step     = np.where(bar_times >= r_naive, cost_pln, 0.0)
                    total_upnl      += step
                    total_swap_upnl += step
            else:
                # No rollover cost data — fall back to flat offset
                flat = np.where(active, swap_pln, 0.0)
                total_upnl      += flat
                total_swap_upnl += flat

        # Margin floats per bar with the close price — matches OANDA's current-price
        # margin requirement.  When market dips, required margin falls too.
        pos_margin_arr = closes * pos["volume"] * contract_size / leverage * margin_scale
        total_margin += np.where(active, pos_margin_arr, 0.0)

    # ── Historical (closed) positions — margin + P&L reconstruction ─────────
    # For each closed position we reconstruct BOTH margin and unrealised P&L
    # over its open period.  Without the P&L, equity = balance (just cash) while
    # margin reflects a real position — making margin > equity look wrong.
    if not open_deals.empty and not trade_deals.empty:
        close_by_id = trade_deals.set_index("position_id")["time"].to_dict() \
                      if "position_id" in trade_deals.columns else {}
        # Pre-populate with currently open position IDs so partially-closed
        # positions (which appear in both current loop and open_deals) are not
        # double-counted in the historical reconstruction.
        seen_pids = set()
        if "position_id" in positions.columns:
            seen_pids.update(
                int(pid) for pid in positions["position_id"].dropna()
            )

        for _, od in open_deals.iterrows():
            pid = od["position_id"]
            if pid not in close_by_id or pid in seen_pids:
                continue   # still open (handled above) or already processed
            seen_pids.add(pid)

            t_open  = od["time"]
            t_close = close_by_id[pid]
            if t_open.tzinfo is not None:
                t_open  = t_open.tz_localize(None)
            if t_close.tzinfo is not None:
                t_close = t_close.tz_localize(None)

            sym = od["symbol"]
            si  = mt5.symbol_info(sym)
            contract_size = si.trade_contract_size if si else 50.0

            master_times = symbol_data[list(symbol_data.keys())[0]]["time"].values
            window = ((master_times >= np.datetime64(t_open)) &
                      (master_times <  np.datetime64(t_close)))

            # Per-bar margin using close price (not fixed entry price)
            if sym in symbol_data:
                sym_closes_all = symbol_data[sym]["close"].values
                hist_margin_arr = (sym_closes_all * od["volume"] * contract_size
                                   / leverage * margin_scale)
                total_margin += np.where(window, hist_margin_arr, 0.0)
            else:
                hist_margin = (od["price_open"] * od["volume"] * contract_size
                               / leverage * margin_scale)
                total_margin += np.where(window, hist_margin, 0.0)

            # Option C: anchor historical upnl to actual net PLN from deal history.
            # net_pln = sum of all close-deal P&L for this position (already PLN).
            # Scale by price-movement ratio so upnl == net_pln at the close bar.
            if sym in symbol_data:
                sym_closes = symbol_data[sym]["close"].values
                net_pln    = float(
                    trade_deals[trade_deals["position_id"] == pid]["net"].sum()
                )
                # Price at the last bar inside the window (≈ close price)
                if window.any():
                    close_bar_price = float(sym_closes[window][-1])
                else:
                    close_bar_price = od["price_open"]
                delta = close_bar_price - od["price_open"]
                if abs(delta) > 0.1:
                    price_ratio = (sym_closes - od["price_open"]) / delta
                    hist_upnl   = np.where(window, net_pln * price_ratio, 0.0)
                else:
                    hist_upnl = np.zeros(n)
                total_upnl += hist_upnl

    equity       = balance + total_upnl
    free_margin  = equity - total_margin
    margin_level = np.where(total_margin > 0, equity / total_margin * 100.0, np.inf)

    return pd.DataFrame({
        "time":         times,
        "equity":       equity,
        "equity_no_swap": balance + total_upnl - total_swap_upnl,
        "balance":      balance,        # deposited + realised
        "deposited":    deposited,      # cash in only
        "realised":     realised,       # cumulative closed-trade P&L
        "upnl":         total_upnl,
        "swap_upnl":    total_swap_upnl,
        "margin_used":  total_margin,
        "free_margin":  free_margin,
        "margin_level": margin_level,
    })


def compute_drawdown(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak > 0, (equity - peak) / peak * 100.0, 0.0)
    return dd


# ── Console output ─────────────────────────────────────────────────────────────

def print_snapshot(info, positions: pd.DataFrame, bal_events: pd.DataFrame):
    acct = info
    print(f"\n{'═'*55}")
    print(f"  ACCOUNT SNAPSHOT  ({acct.currency})")
    print(f"{'═'*55}")
    print(f"  Balance        : {acct.balance:>12,.2f}")
    print(f"  Equity         : {acct.equity:>12,.2f}  "
          f"({'%+.2f' % (acct.equity - acct.balance)}  unrealised)")
    print(f"  Margin used    : {acct.margin:>12,.2f}")
    print(f"  Free margin    : {acct.margin_free:>12,.2f}")
    print(f"  Margin level   : {acct.margin_level:>11.1f} %")
    if len(bal_events):
        total_dep = bal_events[bal_events["amount"] > 0]["amount"].sum()
        total_wdr = bal_events[bal_events["amount"] < 0]["amount"].sum()
        print(f"  Total deposits : {total_dep:>12,.2f}  ({(bal_events['amount'] > 0).sum()} events)")
        if total_wdr:
            print(f"  Withdrawals    : {total_wdr:>12,.2f}  ({(bal_events['amount'] < 0).sum()} events)")
        print(f"\n  All non-trade events ({len(bal_events)} total):")
        print(f"  {'Date':>22}  {'Amount':>12}  {'Type':<20}  Comment")
        print(f"  {'─'*22}  {'─'*12}  {'─'*20}  {'─'*40}")
        for _, ev in bal_events.iterrows():
            comment  = str(ev.get("comment", "")) or ""
            typename = str(ev.get("type_name", ""))
            print(f"  {str(ev['time'])[:22]}  {ev['amount']:>12,.2f}  {typename:<20}  {comment}")
    if not positions.empty:
        print(f"\n  Open positions : {len(positions)}")
        print(f"  {'Dir':<4}  {'Vol':>6}  {'Symbol':<12}  {'Entry':>10}  "
              f"{'P&L':>10}  {'Swap':>10}  {'Net':>10}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*12}  {'─'*10}  "
              f"{'─'*10}  {'─'*10}  {'─'*10}")
        for _, p in positions.iterrows():
            direction = "BUY" if p["type"] == 0 else "SELL"
            swap = float(p.get("swap", 0.0))
            net  = float(p["profit"]) + swap
            print(f"  {direction:<4}  {p['volume']:>6}  {p['symbol']:<12}  "
                  f"{p['price_open']:>10.2f}  {p['profit']:>+10.2f}  "
                  f"{swap:>+10.2f}  {net:>+10.2f}")
    else:
        print(f"\n  No open positions.")
    print(f"{'═'*55}")


def print_swap_analysis(trade_deals: pd.DataFrame, currency: str = "PLN"):
    """Print a breakdown of swap charges from closed trade history."""
    if trade_deals.empty or "swap" not in trade_deals.columns:
        return
    total_swap = trade_deals["swap"].sum()
    nonzero = trade_deals[trade_deals["swap"] != 0].copy()
    print(f"\n{'═'*55}")
    print(f"  SWAP / ROLLOVER ANALYSIS  ({currency})")
    print(f"{'═'*55}")
    print(f"  Total swap charged (all closed trades) : {total_swap:>+10,.2f}")
    print(f"  Trades with non-zero swap              : {len(nonzero):>10}  "
          f"/ {len(trade_deals)} total")
    if nonzero.empty:
        print("  No swap charges found in closed trade history.")
        print(f"{'═'*55}")
        return

    # Monthly breakdown
    nonzero["month"] = nonzero["time"].dt.to_period("M")
    monthly = nonzero.groupby("month")["swap"].agg(["sum", "count"]).reset_index()
    monthly.columns = ["month", "swap_sum", "n_trades"]
    print(f"\n  Monthly swap breakdown:")
    print(f"  {'Month':<10}  {'Trades':>8}  {'Swap sum':>12}  {'Avg/trade':>12}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*12}")
    for _, row in monthly.iterrows():
        avg = row["swap_sum"] / row["n_trades"]
        print(f"  {str(row['month']):<10}  {int(row['n_trades']):>8}  "
              f"{row['swap_sum']:>+12,.2f}  {avg:>+12,.2f}")

    # Individual trades around March 15 and June 15 (±7 days)
    rollover_windows = [
        ("Mar rollover", 3, 15),
        ("Jun rollover", 6, 15),
        ("Sep rollover", 9, 15),
        ("Dec rollover", 12, 15),
    ]
    print(f"\n  Trades near quarterly rollover dates (±7 days):")
    print(f"  {'Date':>22}  {'Symbol':<12}  {'Vol':>6}  "
          f"{'Profit':>10}  {'Swap':>10}  {'Net':>10}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*6}  "
          f"{'─'*10}  {'─'*10}  {'─'*10}")
    found_any = False
    for label, month, day in rollover_windows:
        mask = (
            (nonzero["time"].dt.month == month) &
            (nonzero["time"].dt.day.between(day - 7, day + 7))
        )
        subset = nonzero[mask]
        if not subset.empty:
            print(f"  ── {label} ──")
            found_any = True
            for _, r in subset.iterrows():
                sym = str(r.get("symbol", ""))
                print(f"  {str(r['time'])[:22]}  {sym:<12}  {r['volume']:>6}  "
                      f"{r['profit']:>+10.2f}  {r['swap']:>+10.2f}  {r['net']:>+10.2f}")
    if not found_any:
        print("  None found near rollover windows.")
    print(f"{'═'*55}")


# ── Rollover model ─────────────────────────────────────────────────────────────

# Official OANDA TMS rollover dates for US500.pro (from broker PDF "Tabela rolowań 2026").
# Key: (year, month) → day-of-month.  OANDA rolls on the Wednesday before the
# 3rd Friday of the expiry month (i.e. 3rd Friday − 2 days).
_OANDA_US500_ROLLOVER_DAYS = {
    (2026, 3):  18,
    (2026, 6):  17,
    (2026, 9):  16,
    (2026, 12): 16,
}


def compute_rollover_dates(year_start: int = 2020, year_end: int = 2030):
    """
    Return US500.pro rollover dates as UTC Timestamps.

    Uses official OANDA broker table for 2026; for all other years uses
    'Wednesday before 3rd Friday' (= 3rd Friday − 2 days) as approximation.
    """
    import calendar
    dates = []
    for year in range(year_start, year_end + 1):
        for month in (3, 6, 9, 12):
            if (year, month) in _OANDA_US500_ROLLOVER_DAYS:
                day = _OANDA_US500_ROLLOVER_DAYS[(year, month)]
            else:
                fridays = [w[4] for w in calendar.monthcalendar(year, month) if w[4] != 0]
                day = fridays[2] - 2   # 3rd Friday − 2 = rollover Wednesday
            dates.append(pd.Timestamp(year, month, day, tz="UTC"))
    return sorted(dates)


def compute_rollover_costs(positions: pd.DataFrame) -> dict:
    """
    Back-calculate the cost per 0.001 lot for each historical rollover date.

    Logic:
      - Sort past rollover dates descending: R0 (most recent), R1, R2, …
      - Positions that lived through k rollovers have swap = sum(cost_R0 … cost_Rk-1)
      - Group positions by rollover count; take mean swap/0.001lot per group
      - Cost(Rk) = avg_swap(n=k+1) − avg_swap(n=k)

    Returns: {pd.Timestamp (UTC) → cost_per_0001_lot (float)}
    """
    if positions.empty or "swap" not in positions.columns \
            or "time_open" not in positions.columns:
        return {}

    now          = pd.Timestamp.now(tz="UTC")
    past_rollovers = [d for d in compute_rollover_dates() if d <= now]
    past_desc      = sorted(past_rollovers, reverse=True)

    rows = []
    for _, pos in positions.iterrows():
        t_open = pos["time_open"]
        if t_open.tzinfo is None:
            t_open = t_open.tz_localize("UTC")
        n   = sum(1 for r in past_rollovers if r > t_open)
        vol = float(pos["volume"])
        swap_per_001 = float(pos["swap"]) / (vol / 0.001) if vol > 0 else 0.0
        rows.append({"n_rollovers": n, "swap_per_001": swap_per_001, "volume": vol})

    df     = pd.DataFrame(rows)
    groups = df.groupby("n_rollovers")["swap_per_001"].mean().sort_index().to_dict()

    costs = {}
    for n in sorted(groups):
        if n == 0:
            continue
        cost = groups[n] - groups.get(n - 1, 0.0)
        if n - 1 < len(past_desc):
            costs[past_desc[n - 1]] = cost
    return costs


def print_rollover_model(positions: pd.DataFrame, currency: str = "PLN"):
    """Print rollover cost table, cumulative swap paid, and next-rollover projection."""
    if positions.empty or "swap" not in positions.columns \
            or "time_open" not in positions.columns:
        return

    now              = pd.Timestamp.now(tz="UTC")
    all_dates        = compute_rollover_dates(2020, 2030)
    past_rollovers   = [d for d in all_dates if d <= now]
    rollover_costs   = compute_rollover_costs(positions)

    # Rebuild grouping for the diagnostic table
    rows = []
    for _, pos in positions.iterrows():
        t_open = pos["time_open"]
        if t_open.tzinfo is None:
            t_open = t_open.tz_localize("UTC")
        n   = sum(1 for r in past_rollovers if r > t_open)
        vol = float(pos["volume"])
        swap_per_001 = float(pos["swap"]) / (vol / 0.001) if vol > 0 else 0.0
        rows.append({"n_rollovers": n, "swap_per_001": swap_per_001, "volume": vol})
    df     = pd.DataFrame(rows)
    groups = df.groupby("n_rollovers")["swap_per_001"].mean().sort_index().to_dict()

    print(f"\n{'═'*60}")
    print(f"  ROLLOVER MODEL  ({currency})")
    print(f"  Dates: official OANDA table for 2026; 'Wed before 3rd Fri' elsewhere")
    print(f"{'═'*60}")

    # ── Total swap already paid ───────────────────────────────────────────────
    total_swap = float(positions["swap"].sum())
    print(f"\n  Total swap charged to open positions : {total_swap:>+,.2f} {currency}")

    # ── Per-rollover cost table ───────────────────────────────────────────────
    print(f"\n  Historical rollover costs (back-calculated from open positions):")
    print(f"  {'Rollover date':<14}  {'Cost/0.001lot':>14}  {'Cost/0.01lot':>13}  "
          f"{'Cost/lot':>12}")
    print(f"  {'─'*14}  {'─'*14}  {'─'*13}  {'─'*12}")
    for rdate in sorted(rollover_costs):
        c = rollover_costs[rdate]
        print(f"  {str(rdate)[:10]:<14}  {c:>+14.2f}  {c*10:>+13.2f}  "
              f"{c*1000:>+12.2f}")

    # ── Summary by rollover count ─────────────────────────────────────────────
    print(f"\n  Positions grouped by # of rollovers survived:")
    print(f"  {'# Rollovers':>12}  {'Positions':>10}  {'Avg swap/0.001lot':>18}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*18}")
    for n in sorted(groups):
        n_pos = int((df["n_rollovers"] == n).sum())
        print(f"  {n:>12}  {n_pos:>10}  {groups[n]:>+18.2f}")

    # ── Next rollover projection ──────────────────────────────────────────────
    future = [d for d in all_dates if d > now]
    if future:
        next_r    = future[0]
        days_away = (next_r - now).days
        print(f"\n  Next rollover : {str(next_r)[:10]}  ({days_away} days away)")
        if rollover_costs:
            avg_cost  = sum(rollover_costs.values()) / len(rollover_costs)
            total_vol = df["volume"].sum()
            proj      = avg_cost * (total_vol / 0.001)
            print(f"  Avg cost/rollover/0.001lot : {avg_cost:>+.2f} {currency}")
            print(f"  Projected cost on {total_vol:.3f} lots  : "
                  f"{proj:>+,.2f} {currency}  (avg of {len(rollover_costs)} known rollovers)")

    print(f"{'═'*60}")


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot(ts: pd.DataFrame, bal_events: pd.DataFrame,
         info, currency: str,
         rollover_dates: list = None,
         margin_call_pct: float = 100.0,
         margin_so_pct: float = 50.0) -> plt.Figure:
    plt.style.use("dark_background")

    has_ts = not ts.empty

    # Downsample to H1 for plotting — 22k M5 bars → ~1800 points, much cleaner
    if has_ts and len(ts) > 500:
        ts_plot = (
            ts.set_index("time")
            .resample("1H")
            .agg({
                "equity":         "last",
                "equity_no_swap": "last",
                "balance":        "last",
                "deposited":      "last",
                "realised":       "last",
                "upnl":           "last",
                "swap_upnl":      "last",
                # Margin uses mean over the hour — smooths out intrabar noise
                # while keeping the value accurate as an hourly average.
                # Equity uses last (anchored to actual close).
                "margin_used":    "mean",
                "free_margin":    "mean",
                "margin_level":   "mean",
            })
            .dropna()
            .reset_index()
        )
    else:
        ts_plot = ts

    fig = plt.figure(figsize=(18, 12))
    n_rows = 3 if has_ts else 1
    gs = gridspec.GridSpec(n_rows, 1, figure=fig,
                           hspace=0.45,
                           height_ratios=[4, 1.5, 1.5] if has_ts else [1])

    title = (f"Account health — {info.login}  |  "
             f"Balance {info.balance:,.0f}  Equity {info.equity:,.0f}  "
             f"Margin level {info.margin_level:.0f}%  ({currency})")
    fig.suptitle(title, fontsize=11)

    # ── Panel 1: stacked bands ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    if has_ts:
        times       = ts_plot["time"].values
        equity      = ts_plot["equity"].values
        m_used      = ts_plot["margin_used"].values   # per-bar array — 0 before position opens, steps up after
        m_current   = float(ts_plot["margin_used"].iloc[-1])  # last bar = current

        # Band 1 — margin used (bottom, orange): steps up when position opens
        ax1.fill_between(times, 0, m_used,
                         color="#f08040", alpha=0.75,
                         label=f"Margin used  (current {m_current:,.0f})")

        # Band 2 — free margin (teal when positive, red when negative)
        ax1.fill_between(times, m_used, equity,
                         where=(equity >= m_used),
                         color="#26a69a", alpha=0.55, label="Free margin")
        ax1.fill_between(times, equity, m_used,
                         where=(equity < m_used),
                         color="#ef5350", alpha=0.70, label="Margin call zone")

        # Equity line
        ax1.plot(times, equity, color="#f0f0f0", lw=1.6, label="Equity", zorder=4)

        # Equity without swap — shows what equity would be if no rollover costs
        equity_ns = ts_plot["equity_no_swap"].values
        ax1.plot(times, equity_ns, color="#f0f0f0", lw=1.0, ls=":",
                 alpha=0.5, label="Equity (no swap)", zorder=3)

        # Balance — step function (rises on deposits, flat otherwise)
        ax1.step(times, ts_plot["balance"].values, where="post",
                 color="#aaaaaa", lw=1.2, ls="--",
                 label=f"Balance  (current {ts_plot['balance'].iloc[-1]:,.0f})")

        # Deposit / withdrawal markers
        for _, ev in bal_events.iterrows():
            color    = "#26a69a" if ev["amount"] > 0 else "#ef5350"
            marker   = "^" if ev["amount"] > 0 else "v"
            # ts["time"] is tz-naive (numpy datetime64); strip tz for comparison
            ev_naive = ev["time"].tz_localize(None) if ev["time"].tzinfo is not None \
                       else ev["time"]
            ax1.axvline(ev_naive, color=color, lw=0.8, ls=":", alpha=0.6)
            mask = ts["time"] <= ev_naive
            bal_at_event = float(ts.loc[mask, "balance"].iloc[-1]
                                 if mask.any() else ts["balance"].iloc[0])
            ax1.scatter([ev_naive], [bal_at_event], marker=marker,
                        color=color, s=70, zorder=5)

        # Rollover date markers — gold vertical lines across ax1 & ax2
        if rollover_dates:
            from matplotlib.transforms import blended_transform_factory
            t_start_np = times[0]
            t_end_np   = times[-1]
            for rdate in rollover_dates:
                r_np = np.datetime64(
                    rdate.tz_convert(None) if rdate.tzinfo else rdate
                )
                if not (t_start_np <= r_np <= t_end_np):
                    continue
                ax1.axvline(r_np, color="#f0c040", lw=1.2, ls="--", alpha=0.8, zorder=3)
                trans = blended_transform_factory(ax1.transData, ax1.transAxes)
                ax1.text(r_np, 0.99,
                         f" ↓ rollover\n   {str(rdate)[:10]}",
                         transform=trans, color="#f0c040",
                         fontsize=6, va="top", ha="left",
                         bbox=dict(fc="none", ec="none"))

        ax1.set_ylabel(f"Amount ({currency})")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"))
        ax1.legend(fontsize=8, ncol=5, loc="upper left")
        ax1.grid(True, alpha=0.18)
        plt.setp(ax1.get_xticklabels(), rotation=20, ha="right", fontsize=7)

    else:
        # No open positions — just show current numbers as text
        labels = ["Balance", "Equity", "Margin used", "Free margin"]
        values = [info.balance, info.equity, info.margin, info.margin_free]
        colors = ["#40a0f0", "#f0f0f0", "#f08040", "#26a69a"]
        bars   = ax1.bar(labels, values, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.01,
                     f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
        ax1.set_ylabel(f"Amount ({currency})")
        ax1.set_title("Current account snapshot (no open positions — no time series to reconstruct)")
        ax1.grid(True, axis="y", alpha=0.2)

    ax1.set_title("Equity vs margin — stacked band  (margin: orange, free margin: teal)")
    ax1.set_ylim(bottom=0)   # ensure margin band at bottom is always visible

    if not has_ts:
        plt.tight_layout()
        return fig

    # ── Panel 2: deposited vs realised P&L vs unrealised P&L ──────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    dep  = ts_plot["deposited"].values
    real = ts_plot["realised"].values
    eq   = ts_plot["equity"].values
    bal  = ts_plot["balance"].values   # dep + real

    # Layer 1 — deposited cash (blue, always at the bottom)
    ax2.fill_between(times, 0, dep,
                     color="#40a0f0", alpha=0.55, label="Deposited")

    # Layer 2 — realised P&L from closed trades (on top of deposited)
    ax2.fill_between(times, dep, bal,
                     where=(bal >= dep),
                     color="#26a69a", alpha=0.70, label="Realised P&L (profit)")
    ax2.fill_between(times, bal, dep,
                     where=(bal < dep),
                     color="#ef5350", alpha=0.70, label="Realised P&L (loss)")

    # Layer 3 — unrealised P&L from open positions (lightest, on top)
    ax2.fill_between(times, bal, eq,
                     where=(eq >= bal),
                     color="#80e080", alpha=0.40, label="Unrealised (profit)")
    ax2.fill_between(times, eq, bal,
                     where=(eq < bal),
                     color="#ff8080", alpha=0.40, label="Unrealised (loss)")

    ax2.plot(times, eq, color="#f0f0f0", lw=1.2, zorder=4, label="Equity")

    if rollover_dates:
        for rdate in rollover_dates:
            r_np = np.datetime64(rdate.tz_convert(None) if rdate.tzinfo else rdate)
            if times[0] <= r_np <= times[-1]:
                ax2.axvline(r_np, color="#f0c040", lw=1.2, ls="--", alpha=0.8, zorder=3)

    ax2.set_ylim(bottom=0)
    ax2.set_ylabel(f"Amount ({currency})")
    ax2.set_title("Capital composition: deposited  +  realised P&L  +  unrealised P&L")
    ax2.legend(fontsize=7, ncol=6)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.grid(True, alpha=0.18)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Panel 3: margin level % (only when positions are open) ────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Hide bars where no positions were open (margin_used == 0 → level = inf)
    pos_open_mask = ts_plot["margin_used"].values > 0
    times_ml = times[pos_open_mask]
    ml = np.clip(ts_plot["margin_level"].values[pos_open_mask], 0, 1000)

    if len(times_ml):
        ax3.plot(times_ml, ml, color="#f0c040", lw=1.2, label="Margin level")
        ax3.fill_between(times_ml, ml, 0, color="#f0c040", alpha=0.12)

    ax3.axhline(margin_so_pct,   color="#ef5350", lw=1.2, ls="--",
                label=f"Stop-out  ({margin_so_pct:.0f}%)")
    ax3.axhline(margin_call_pct, color="#f08040", lw=0.9, ls=":",
                label=f"Margin call ({margin_call_pct:.0f}%)")
    ax3.axhspan(0, margin_so_pct, color="#ef5350", alpha=0.10)
    cur_ml = float(info.margin_level)
    ax3.axhline(cur_ml, color="#ffffff", lw=0.7, ls="--", alpha=0.4,
                label=f"Current  ({cur_ml:.0f}%)")

    ax3.set_ylabel("Margin level (%)")
    ax3.set_title("Margin level — only shown when positions are open")
    ax3.legend(fontsize=7, ncol=4)
    ax3.grid(True, alpha=0.18)
    plt.setp(ax3.get_xticklabels(), rotation=20, ha="right", fontsize=7)

    plt.tight_layout()
    return fig


def print_margin_diagnostic(ts: pd.DataFrame, date_str: str = "2026-03"):
    """
    Print daily min/max equity, margin used and margin level for a date window.
    Useful for verifying the reconstruction against known account events.
    """
    if ts.empty:
        return
    # ts["time"] is tz-naive numpy datetime64 — convert to Period for filtering
    times_ser = pd.to_datetime(ts["time"])
    mask = times_ser.dt.to_period("M").astype(str) == date_str
    subset = ts[mask].copy()
    if subset.empty:
        print(f"  No reconstructed data found for '{date_str}'")
        return

    # Downsample to daily summary
    subset = subset.copy()
    subset["date"] = pd.to_datetime(subset["time"]).dt.floor("D")
    daily = subset.groupby("date").agg(
        equity_min=("equity",       "min"),
        equity_max=("equity",       "max"),
        margin_min=("margin_used",  "min"),
        margin_max=("margin_used",  "max"),
        ml_min    =("margin_level", "min"),
        ml_max    =("margin_level", "max"),
    ).reset_index()

    print(f"\n{'═'*70}")
    print(f"  MARGIN DIAGNOSTIC — {date_str}")
    print(f"{'═'*70}")
    print(f"  {'Date':<12}  {'Equity min':>12}  {'Equity max':>12}  "
          f"{'Margin min':>11}  {'ML min %':>10}  {'ML max %':>10}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*11}  {'─'*10}  {'─'*10}")
    for _, r in daily.iterrows():
        ml_min = r["ml_min"] if r["ml_min"] != np.inf else 9999
        ml_min = min(ml_min, 9999)
        print(f"  {str(r['date'])[:10]:<12}  {r['equity_min']:>12,.0f}  "
              f"{r['equity_max']:>12,.0f}  {r['margin_min']:>11,.0f}  "
              f"{ml_min:>10.1f}  {min(r['ml_max'], 9999):>10.1f}")
    print(f"{'═'*70}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6,
                        help="How many recent months to display (default: 4)")
    parser.add_argument("--all", dest="show_all", action="store_true",
                        help="Show from the earliest open position")
    args = parser.parse_args()

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        now       = pd.Timestamp.now(tz="UTC")
        info      = mt5.account_info()
        currency  = info.currency
        leverage  = info.leverage if info.leverage > 0 else 20

        print("Fetching deal history...")
        bal_events, trade_deals, open_deals, all_deals = fetch_deal_history()
        print(f"  {len(bal_events)} non-trade event(s)  (balance / adjustment / dividend / ...)")
        print(f"  {len(trade_deals)} closed trade deal(s)")
        print(f"  {len(open_deals)} position-open deal(s)")
        print(f"  {len(all_deals)} total raw deals in history")

        print("Fetching open positions...")
        positions = get_open_positions()

        print_snapshot(info, positions, bal_events)
        print_swap_analysis(trade_deals, currency)
        print_rollover_model(positions, currency)

        rollover_costs = compute_rollover_costs(positions)

        # ── Save deal history to CSV for offline inspection ───────────────────
        import os
        os.makedirs("results", exist_ok=True)
        if not all_deals.empty:
            all_deals.to_csv("results/01a_all_deals.csv", index=False)
            print(f"\n  Saved {len(all_deals)} raw deals      → results/01a_all_deals.csv")
        if not bal_events.empty:
            bal_events.to_csv("results/01a_balance_events.csv", index=False)
            print(f"  Saved {len(bal_events)} non-trade events → results/01a_balance_events.csv")
        if not trade_deals.empty:
            swap_nonzero = trade_deals[trade_deals["swap"] != 0]
            trade_deals.to_csv("results/01a_trade_deals.csv", index=False)
            print(f"  Saved {len(trade_deals)} trade deals     → results/01a_trade_deals.csv"
                  f"  ({len(swap_nonzero)} have non-zero swap)")

        ts = pd.DataFrame()
        if not positions.empty:
            print("\nReconstructing equity from cached M5 price data...")
            cutoff = None if args.show_all else (
                pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=args.months)
            )
            ts = reconstruct_equity(positions, bal_events, trade_deals, open_deals,
                                    info.margin, leverage, t_start=cutoff,
                                    rollover_costs=rollover_costs)
            if not ts.empty:
                print(f"  {len(ts):,} bars  "
                      f"({str(ts['time'].iloc[0])[:16]} UTC → now)")
            else:
                print("  Could not reconstruct — check that M5 data is cached "
                      "(run 01_fetch_data.py)")
        else:
            print("\nNo open positions — showing current snapshot only.")

        # Fetch actual broker stop-out thresholds from MT5 account info
        margin_so_pct   = float(getattr(info, "margin_so_so",   50.0))
        margin_call_pct = float(getattr(info, "margin_so_call", 100.0))
        print(f"\n  Broker thresholds: margin call {margin_call_pct:.0f}%  "
              f"/ stop-out {margin_so_pct:.0f}%")

        if not ts.empty:
            print_margin_diagnostic(ts, "2026-03")

        past_rollover_dates = [d for d in compute_rollover_dates() if d <= now]
        fig = plot(ts, bal_events, info, currency,
                   rollover_dates=past_rollover_dates,
                   margin_call_pct=margin_call_pct,
                   margin_so_pct=margin_so_pct)
        cache.save_chart(fig, "01a_account_history")
        plt.show()

    finally:
        disconnect()


if __name__ == "__main__":
    main()
