"""
03a_deploy_ladder.py -- Deploy the proposed ladder from 02_support_levels.

Compares current pending orders against the proposed ladder and computes
the minimal diff (cancel stale orders, place missing ones).

**Dry-run by default** -- prints the plan but touches nothing.
Pass --execute to actually cancel/place orders on the live account.

Always produces a cascade chart showing:
  Left  -- pending order tower (volume at each price bin)
  Right -- equity & margin level curve as price drops through the ladder

Usage:
  python 03a_deploy_ladder.py                  # dry-run: show plan + chart
  python 03a_deploy_ladder.py --execute        # deploy for real
  python 03a_deploy_ladder.py --grid-step 25   # forward args to ladder proposal
  python 03a_deploy_ladder.py --skip-cancel    # only place missing, don't cancel
  python 03a_deploy_ladder.py --no-chart       # skip chart generation

Reads proposed ladder from 02_support_levels.propose_ladder().
Audit trail saved to data/ladder_deployed.csv after each --execute run.
"""

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import MetaTrader5 as mt5

from mt5_client import connect, disconnect
from import_02 import get_pending_orders, detect_supports, propose_ladder, fetch_d1

_ROOT   = Path(__file__).parent
_DATA   = _ROOT / "data"
_CHARTS = _ROOT / "results" / "charts"

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


CONTRACT_SIZE  = 50.0
INSTR_LEVERAGE = 20
SPREAD_PTS     = 0.7


def _save_chart(fig, name):
    _CHARTS.mkdir(parents=True, exist_ok=True)
    path = _CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved chart -> results/charts/{name}.png")


def get_account_state():
    """Pull live account state + existing positions for cascade math."""
    info = mt5.account_info()
    if info is None:
        return None, None, None

    raw = mt5.positions_get()
    if raw is None or not len(raw):
        positions = pd.DataFrame()
    else:
        rows = []
        for p in raw:
            if p.symbol != SYMBOL:
                continue
            rows.append({
                "volume":        p.volume,
                "direction":     1.0 if p.type == 0 else -1.0,
                "price_open":    p.price_open,
                "price_current": p.price_current,
                "profit":        p.profit,
            })
        positions = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Infer PLN/USD rate from existing positions
    pln_rate = 4.0
    if not positions.empty:
        rates = []
        for _, p in positions.iterrows():
            delta = p["price_current"] - p["price_open"]
            if abs(delta) > 2.0:
                r = p["profit"] / (p["direction"] * delta * p["volume"] * CONTRACT_SIZE)
                if 1.0 < r < 12.0:
                    rates.append(r)
        if rates:
            pln_rate = float(np.median(rates))

    R = pln_rate * CONTRACT_SIZE  # PLN per lot per point
    L = float((positions["volume"] * positions["direction"]).sum()) if not positions.empty else 0.0
    upnl = float(positions["profit"].sum()) if not positions.empty else 0.0
    margin = info.margin if info.margin else 0.0

    state = {
        "balance":  info.balance,
        "equity":   info.equity,
        "margin":   margin,
        "pln_rate": pln_rate,
        "R":        R,
        "L":        L,
        "upnl_now": upnl,
        "currency": info.currency,
    }
    return state, positions, info


def cascade_analysis(state, proposed, current_price):
    """
    For each price from current down to the lowest order, compute:
      - cumulative orders triggered and lots
      - equity with and without ladder orders
      - margin level and margin required by ladder orders
    """
    if proposed.empty or state is None:
        return pd.DataFrame()

    R = state["R"]
    order_prices = proposed["price"].values
    order_volumes = proposed["volume"].values

    # Evaluate at every order price + a few extra points
    test_prices = np.sort(np.unique(np.concatenate([
        order_prices,
        [current_price],
        np.arange(
            int(order_prices.min() // 25) * 25,
            current_price + 1,
            25,
        ),
    ])))[::-1]  # descending

    rows = []
    for P in test_prices:
        if P > current_price + 1:
            continue

        # Orders triggered: buy limits fill when price drops to their level
        triggered = order_prices >= P
        n_fired = int(triggered.sum())
        cum_lots = float(order_volumes[triggered].sum())

        # Existing positions P&L change
        upnl_existing = state["upnl_now"] + state["L"] * R * (P - current_price)
        margin_existing = max(state["margin"] * P / current_price, 1e-6)
        equity_no_ladder = state["balance"] + upnl_existing

        # Triggered orders P&L and margin
        upnl_ladder = 0.0
        margin_ladder = 0.0
        for op, ov in zip(order_prices[triggered], order_volumes[triggered]):
            upnl_ladder += ov * R * (P - op)
            margin_ladder += ov * R * P / INSTR_LEVERAGE

        equity = equity_no_ladder + upnl_ladder
        total_margin = max(margin_existing + margin_ladder, 1e-6)
        ml = equity / total_margin * 100

        rows.append({
            "price":           P,
            "drop":            current_price - P,
            "n_fired":         n_fired,
            "cum_lots":        cum_lots,
            "equity":          equity,
            "equity_no_ladder": equity_no_ladder,
            "margin_level":    ml,
            "margin_ladder":   margin_ladder,
            "total_lots":      state["L"] + cum_lots,
        })

    return pd.DataFrame(rows)


def print_cascade_summary(cascade, state, current_price):
    if cascade.empty:
        return
    print(f"\n{'='*90}")
    print(f"  CASCADE ANALYSIS -- what happens as price drops through the ladder")
    print(f"{'='*90}")
    print(f"  {'Price':>7}  {'Drop':>7}  {'Fired':>5}  {'Cum lots':>8}  "
          f"{'Eq (ladder)':>11}  {'Eq (no ldr)':>11}  {'ML%':>6}  "
          f"{'Margin req':>10}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*8}  "
          f"{'-'*11}  {'-'*11}  {'-'*6}  {'-'*10}")

    # Show a subset: every ~200 pts + key levels
    shown = set()
    for _, r in cascade.iterrows():
        drop = r["drop"]
        show = (drop == 0 or r["n_fired"] == 1
                or int(drop) % 200 < 30
                or r["margin_level"] < 200 and r["price"] not in shown
                or r["price"] == cascade["price"].min())
        if show and r["price"] not in shown:
            shown.add(r["price"])
            ml_sym = "+" if r["margin_level"] >= 200 else ("!" if r["margin_level"] >= 100 else "X")
            print(f"  {r['price']:>7,.0f}  {-r['drop']:>+7,.0f}  "
                  f"{r['n_fired']:>5.0f}  {r['cum_lots']:>8.3f}  "
                  f"{r['equity']:>11,.0f}  {r['equity_no_ladder']:>11,.0f}  "
                  f"{r['margin_level']:>5.0f}%{ml_sym}  "
                  f"{r['margin_ladder']:>10,.0f}")
    print(f"{'='*90}")


def _ml_color(ml):
    if ml <= 0:   return "#6a0000"
    if ml < 100:  return "#ef5350"
    if ml < 200:  return "#f08040"
    return "#26a69a"


def plot_cascade(cascade, proposed, state, current_price):
    """
    Left:   Pending order tower (volume per 25-pt price bin)
    Center: Two equity curves -- with ladder vs without (existing only)
    Right:  Cumulative lots triggered + cumulative margin required
    """
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.2, 2, 1.2], wspace=0.30)
    ax_tower = fig.add_subplot(gs[0])
    ax_eq    = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2])

    total_lots = proposed["volume"].sum()
    n_orders = len(proposed)
    max_margin = cascade["margin_ladder"].max() if not cascade.empty else 0

    fig.suptitle(
        f"Ladder Cascade -- US500.pro @ {current_price:,.1f}  |  "
        f"{n_orders} orders, {total_lots:.3f}L  |  "
        f"Equity {state['equity']:,.0f} {state['currency']}  |  "
        f"Max margin for ladder: {max_margin:,.0f} {state['currency']}",
        fontsize=11,
    )

    # ── Left: Order tower ──────────────────────────────────────────────────
    bin_size = 25.0
    bin_lo = np.floor(proposed["price"].min() / bin_size) * bin_size
    bin_hi = np.ceil(current_price / bin_size) * bin_size
    edges = np.arange(bin_lo, bin_hi + bin_size / 2, bin_size)

    bin_centers, bin_vols, bin_counts = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        center = (lo + hi) / 2
        mask = (proposed["price"] >= lo) & (proposed["price"] < hi)
        vol = float(proposed.loc[mask, "volume"].sum())
        cnt = int(mask.sum())
        if cnt > 0:
            bin_centers.append(center)
            bin_vols.append(vol)
            bin_counts.append(cnt)

    if bin_centers:
        max_vol = max(bin_vols)
        colors = ["#26a69a" if c < current_price else "#ef5350" for c in bin_centers]
        ax_tower.barh(bin_centers, bin_vols, height=bin_size * 0.85,
                      color=colors, alpha=0.85, edgecolor="#ffffff", linewidth=0.4,
                      zorder=3)
        for c, v, n in zip(bin_centers, bin_vols, bin_counts):
            ax_tower.text(v + max_vol * 0.03, c,
                          f"{v:.3f} ({n})",
                          va="center", ha="left", fontsize=6.5, color="#cccccc")

    ax_tower.axhline(current_price, color="#ffffff", lw=1.2, ls=":", alpha=0.7,
                     label=f"Current {current_price:,.0f}")
    ax_tower.set_xlabel("Lots in bin")
    ax_tower.set_ylabel("Price level")
    ax_tower.set_title(f"Pending order tower\n({bin_size:.0f}-pt bins)")
    ax_tower.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_tower.legend(fontsize=7.5, loc="lower right")
    ax_tower.grid(True, alpha=0.12, axis="y")

    if not cascade.empty:
        y_lo = cascade["price"].min() - 30
        y_hi = current_price + 30
        ax_tower.set_ylim(y_lo, y_hi)

    # ── Center: Two equity curves ─────────────────────────────────────────
    if not cascade.empty:
        prices = cascade["price"].values
        eq_with = cascade["equity"].values
        eq_without = cascade["equity_no_ladder"].values

        # Equity without ladder (existing positions only) -- grey dashed
        ax_eq.plot(prices, eq_without, color="#888888", lw=1.5, ls="--", zorder=3,
                   label="Existing positions only")

        # Equity with ladder -- blue solid, shaded area between the two
        ax_eq.plot(prices, eq_with, color="#5dade2", lw=2.0, zorder=4,
                   label="With ladder orders")
        ax_eq.fill_between(prices, eq_with, eq_without,
                           where=(eq_with < eq_without),
                           color="#ef5350", alpha=0.12, zorder=2,
                           label="Ladder cost (loss from triggered orders)")
        ax_eq.fill_between(prices, eq_with, eq_without,
                           where=(eq_with >= eq_without),
                           color="#26a69a", alpha=0.12, zorder=2)

        ax_eq.axhline(0, color="#ef5350", lw=1.0, ls="-", alpha=0.5,
                      label="Zero equity")
        ax_eq.axhline(state["balance"], color="#ffd700", lw=0.8, ls=":", alpha=0.4,
                      label=f"Balance ({state['balance']:,.0f})")

        # Annotate equity values at key drops
        last_annotated = current_price + 999
        for _, r in cascade.iterrows():
            if r["drop"] > 0 and int(r["drop"]) % 500 == 0 and last_annotated - r["price"] > 100:
                last_annotated = r["price"]
                ax_eq.annotate(
                    f"{r['equity']:,.0f}",
                    xy=(r["price"], r["equity"]),
                    fontsize=7, color="#5dade2", ha="center", va="bottom",
                )
                ax_eq.annotate(
                    f"{r['equity_no_ladder']:,.0f}",
                    xy=(r["price"], r["equity_no_ladder"]),
                    fontsize=7, color="#888888", ha="center", va="top",
                )

        ax_eq.set_xlabel("US500.pro price level")
        ax_eq.set_ylabel("Equity (PLN)")
        ax_eq.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax_eq.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax_eq.set_title("Equity as price drops: with ladder vs without")
        ax_eq.grid(True, alpha=0.12)
        ax_eq.axvline(current_price, color="#ffffff", lw=0.8, ls=":", alpha=0.5)
        ax_eq.set_xlim(current_price + 20, prices.min() - 20)
        ax_eq.legend(fontsize=7.5, loc="upper right")

    # ── Right: Cumulative lots + margin required ──────────────────────────
    if not cascade.empty:
        ax_right.plot(cascade["cum_lots"], cascade["price"],
                      color="#00e676", lw=2.0, label="Cum. lots triggered")
        ax_right.fill_betweenx(cascade["price"], 0, cascade["cum_lots"],
                               color="#00e676", alpha=0.15)

        ax_margin = ax_right.twiny()
        ax_margin.plot(cascade["margin_ladder"], cascade["price"],
                       color="#ffd700", lw=1.5, ls="--", label="Margin required (PLN)")
        ax_margin.fill_betweenx(cascade["price"], 0, cascade["margin_ladder"],
                                color="#ffd700", alpha=0.08)

        # Lock margin axis proportional to lots axis so divergence
        # honestly reflects the price effect on margin cost.
        # At current price, 1 lot costs R * P / leverage in margin.
        R = state["R"]
        margin_per_lot = R * current_price / INSTR_LEVERAGE
        max_lots = cascade["cum_lots"].max()
        ax_right.set_xlim(0, max_lots * 1.15)
        ax_margin.set_xlim(0, max_lots * 1.15 * margin_per_lot)

        ax_right.set_xlabel("Cumulative lots", color="#00e676")
        ax_right.tick_params(axis="x", labelcolor="#00e676")
        ax_margin.set_xlabel("Margin required (PLN)", color="#ffd700")
        ax_margin.tick_params(axis="x", labelcolor="#ffd700")

        ax_right.set_ylabel("Price level")
        ax_right.set_title("Lots triggered &\nmargin required")
        ax_right.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax_right.grid(True, alpha=0.12, axis="y")
        ax_right.axhline(current_price, color="#ffffff", lw=0.8, ls=":", alpha=0.5)
        ax_right.set_ylim(cascade["price"].min() - 30, current_price + 30)

        lines_a, labels_a = ax_right.get_legend_handles_labels()
        lines_b, labels_b = ax_margin.get_legend_handles_labels()
        ax_right.legend(lines_a + lines_b, labels_a + labels_b,
                        fontsize=7, loc="lower left")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    return fig


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
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip cascade chart generation")
    parser.add_argument("--skip-cancel", action="store_true",
                        help="Only place missing orders, don't cancel existing ones")
    parser.add_argument("--days", type=int, default=250,
                        help="D1 bars for support analysis (default 250)")
    parser.add_argument("--zone-width", type=float, default=20.0,
                        help="Zone merge width (default 20 pts)")
    parser.add_argument("--n-zones", type=int, default=12,
                        help="Number of top zones for clusters (default 12)")
    parser.add_argument("--grid-step", type=float, default=20.0,
                        help="Background grid spacing (default 20 pts)")
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

        # Cascade analysis (always, regardless of execute mode)
        if not args.no_chart:
            print("Building cascade analysis...")
            acct_state, positions, acct_info = get_account_state()
            if acct_state is not None:
                cascade = cascade_analysis(acct_state, proposed, current_price)
                print_cascade_summary(cascade, acct_state, current_price)

                fig = plot_cascade(cascade, proposed, acct_state, current_price)
                _save_chart(fig, "03a_ladder_cascade")
                plt.show()
            else:
                print("  WARNING: Could not fetch account state for cascade chart.")

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
