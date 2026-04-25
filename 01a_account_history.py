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

def fetch_deal_history():
    """
    Fetch all account deals and split into two DataFrames:
      bal_events  — deposits / withdrawals  (DEAL_TYPE_BALANCE)
      trade_deals — closed positions        (DEAL_ENTRY_OUT / DEAL_ENTRY_INOUT)
    """
    now = datetime.now(tz=timezone.utc)
    raw = mt5.history_deals_get(EPOCH, now)
    if raw is None or not len(raw):
        return pd.DataFrame(), pd.DataFrame()

    bal_rows, trade_rows, open_rows = [], [], []
    for d in raw:
        t = pd.Timestamp(d.time, unit="s", tz="UTC")
        if d.type == DEAL_TYPE_BALANCE:
            bal_rows.append({"time": t, "amount": d.profit, "comment": d.comment})
        elif d.entry in (1, 2):   # DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT — closed trade
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
        elif d.entry == 0:        # DEAL_ENTRY_IN — position opened
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

    return _df(bal_rows), _df(trade_rows), _df(open_rows)


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
                       t_start: pd.Timestamp = None) -> pd.DataFrame:
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
    total_upnl   = np.zeros(n)
    total_margin = np.zeros(n)

    # margin_scale: converts the raw formula (price × vol × cs / leverage) to the
    # actual account-currency margin.  Uses entry price so that the same scale
    # applies consistently to both current and historical upnl/margin values.
    formula_margin_now = 0.0
    for _, pos in positions.iterrows():
        si = mt5.symbol_info(pos["symbol"])
        contract_size = si.trade_contract_size if si else 50.0
        formula_margin_now += pos["price_open"] * pos["volume"] * contract_size / leverage
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

        # margin_scale ≈ PLN/USD — converts current-position upnl to account currency.
        # Historical closed-position upnl is left in raw index-point units (consistent
        # with how their margin is computed before the margin_scale factor).
        upnl = np.where(
            active,
            direction * (closes - pos["price_open"]) * pos["volume"] * contract_size * margin_scale,
            0.0,
        )
        total_upnl += upnl

        # Margin steps up when position opens; scaled to account currency
        pos_margin = pos["price_open"] * pos["volume"] * contract_size / leverage * margin_scale
        total_margin += np.where(active, pos_margin, 0.0)

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
            hist_margin   = (od["price_open"] * od["volume"] * contract_size
                             / leverage * margin_scale)

            master_times = symbol_data[list(symbol_data.keys())[0]]["time"].values
            window = ((master_times >= np.datetime64(t_open)) &
                      (master_times <  np.datetime64(t_close)))

            total_margin += np.where(window, hist_margin, 0.0)

            # Reconstruct unrealised P&L so equity is correct during this period
            if sym in symbol_data:
                closes    = symbol_data[sym]["close"].values
                direction = od["direction"]
                hist_upnl = np.where(
                    window,
                    direction * (closes - od["price_open"]) * od["volume"] * contract_size,
                    0.0,
                )
                total_upnl += hist_upnl

    equity       = balance + total_upnl
    free_margin  = equity - total_margin
    margin_level = np.where(total_margin > 0, equity / total_margin * 100.0, np.inf)

    return pd.DataFrame({
        "time":         times,
        "equity":       equity,
        "balance":      balance,        # deposited + realised
        "deposited":    deposited,      # cash in only
        "realised":     realised,       # cumulative closed-trade P&L
        "upnl":         total_upnl,
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
    if not positions.empty:
        print(f"\n  Open positions : {len(positions)}")
        for _, p in positions.iterrows():
            direction = "BUY " if p["type"] == 0 else "SELL"
            print(f"    {direction}  {p['volume']} lot  {p['symbol']}  "
                  f"@ {p['price_open']}  P&L {p['profit']:+.2f}")
    else:
        print(f"\n  No open positions.")
    print(f"{'═'*55}")


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot(ts: pd.DataFrame, bal_events: pd.DataFrame,
         info, currency: str) -> plt.Figure:
    plt.style.use("dark_background")

    has_ts = not ts.empty

    # Downsample to H1 for plotting — 22k M5 bars → ~1800 points, much cleaner
    if has_ts and len(ts) > 500:
        ts_plot = (
            ts.set_index("time")
            .resample("1H")
            .agg({
                "equity":       "last",
                "balance":      "last",
                "deposited":    "last",
                "realised":     "last",
                "upnl":         "last",
                "margin_used":  "last",
                "free_margin":  "last",
                "margin_level": "last",
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

    ax3.axhline(100, color="#ef5350", lw=1.0, ls="--", label="Stop out  (100%)")
    ax3.axhline(200, color="#f08040", lw=0.8, ls=":",  label="Warning   (200%)")
    ax3.axhspan(0, 100, color="#ef5350", alpha=0.08)
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
        info      = mt5.account_info()
        currency  = info.currency
        leverage  = info.leverage if info.leverage > 0 else 20

        print("Fetching deal history...")
        bal_events, trade_deals, open_deals = fetch_deal_history()
        print(f"  {len(bal_events)} deposit/withdrawal event(s)")
        print(f"  {len(trade_deals)} closed trade deal(s)")
        print(f"  {len(open_deals)} position-open deal(s)")

        print("Fetching open positions...")
        positions = get_open_positions()

        print_snapshot(info, positions, bal_events)

        ts = pd.DataFrame()
        if not positions.empty:
            print("\nReconstructing equity from cached M5 price data...")
            cutoff = None if args.show_all else (
                pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=args.months)
            )
            ts = reconstruct_equity(positions, bal_events, trade_deals, open_deals,
                                    info.margin, leverage, t_start=cutoff)
            if not ts.empty:
                print(f"  {len(ts):,} bars  "
                      f"({str(ts['time'].iloc[0])[:16]} UTC → now)")
            else:
                print("  Could not reconstruct — check that M5 data is cached "
                      "(run 01_fetch_data.py)")
        else:
            print("\nNo open positions — showing current snapshot only.")

        fig = plot(ts, bal_events, info, currency)
        cache.save_chart(fig, "01a_account_history")
        plt.show()

    finally:
        disconnect()


if __name__ == "__main__":
    main()
