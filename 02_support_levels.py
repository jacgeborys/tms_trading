"""
02_support_levels.py -- Support-level detection for US500.pro ladder placement.

Identifies support zones from H1 price history using three complementary methods:
  1. Swing lows  -- local minima (price reversed upward after touching)
  2. Volume profile -- price bins with above-average tick volume (consolidation)
  3. Round numbers -- psychological levels (multiples of 50 and 100)

Zones are scored by touch count, recency, and volume, then plotted against
the current price chart with existing pending orders overlaid.

Usage:
  python 02_support_levels.py                     # default: last 12 months H1
  python 02_support_levels.py --months 6          # shorter window
  python 02_support_levels.py --months 24         # longer history
  python 02_support_levels.py --zone-width 15     # wider merge band (default 12)

Outputs:
  results/charts/02_support_levels.png
  results/02_support_levels.csv
"""

import sys
import argparse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import MetaTrader5 as mt5

from pathlib import Path
from mt5_client import connect, disconnect

_ROOT   = Path(__file__).parent
_CHARTS = _ROOT / "results" / "charts"
_RESULTS = _ROOT / "results"


def _save_chart(fig, name: str):
    _CHARTS.mkdir(parents=True, exist_ok=True)
    path = _CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved chart -> results/charts/{name}.png")


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_h1(months: int) -> pd.DataFrame:
    """Fetch H1 OHLCV from MT5 for the last `months` months."""
    mt5.symbol_select("US500.pro", True)
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=months * 30)
    rates = mt5.copy_rates_range("US500.pro", mt5.TIMEFRAME_H1, start, now)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df.reset_index(drop=True)


def fetch_d1_recent(days: int = 120) -> pd.DataFrame:
    """Fetch recent D1 bars for the chart panel."""
    mt5.symbol_select("US500.pro", True)
    rates = mt5.copy_rates_from_pos("US500.pro", mt5.TIMEFRAME_D1, 0, days)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df.reset_index(drop=True)


def get_pending_orders() -> pd.DataFrame:
    """Get all pending buy-limit orders for US500.pro."""
    orders = mt5.orders_get()
    if orders is None or len(orders) == 0:
        return pd.DataFrame()
    rows = []
    for o in orders:
        if o.symbol != "US500.pro":
            continue
        rows.append({
            "ticket":  o.ticket,
            "type":    o.type,
            "volume":  o.volume_current,
            "price":   o.price_open,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("price").reset_index(drop=True)


# ── Support detection ─────────────────────────────────────────────────────────

def find_swing_lows(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Find swing lows: bars where low[i] is the minimum in a +-window neighborhood.
    Returns DataFrame with columns: time, price, strength.
    """
    lows = df["low"].values
    n = len(lows)
    swings = []

    for i in range(window, n - window):
        lo = lows[i]
        neighborhood = lows[max(0, i - window):i + window + 1]
        if lo == neighborhood.min():
            # Strength: how far price bounced from this low (next window bars)
            future_high = lows[i:min(i + window * 2, n)].max() if i + 1 < n else lo
            bounce = future_high - lo
            swings.append({
                "time":     df["time"].iloc[i],
                "price":    lo,
                "bounce":   bounce,
            })

    return pd.DataFrame(swings) if swings else pd.DataFrame()


def volume_profile(df: pd.DataFrame, n_bins: int = 200,
                   price_lo: float = None, price_hi: float = None) -> pd.DataFrame:
    """
    Build a volume profile: total tick volume in each price bin.
    Returns DataFrame with columns: price_mid, volume, is_high.
    """
    if price_lo is None:
        price_lo = df["low"].min()
    if price_hi is None:
        price_hi = df["high"].max()

    bin_edges = np.linspace(price_lo, price_hi, n_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_counts = np.zeros(n_bins)

    # Distribute each bar's volume across the price bins it spans
    for _, row in df.iterrows():
        lo_idx = np.searchsorted(bin_edges, row["low"], side="right") - 1
        hi_idx = np.searchsorted(bin_edges, row["high"], side="left")
        lo_idx = max(0, lo_idx)
        hi_idx = min(n_bins - 1, hi_idx)
        n_spanned = hi_idx - lo_idx + 1
        if n_spanned > 0:
            vol_counts[lo_idx:hi_idx + 1] += row["volume"] / n_spanned

    vp = pd.DataFrame({"price_mid": bin_mids, "volume": vol_counts})
    # High-volume nodes: above 75th percentile
    threshold = np.percentile(vol_counts[vol_counts > 0], 75)
    vp["is_high"] = vp["volume"] >= threshold
    return vp


def cluster_supports(prices: np.ndarray, scores: np.ndarray,
                     zone_width: float = 12.0) -> pd.DataFrame:
    """
    Merge nearby support prices into zones.
    Returns DataFrame: zone_center, zone_lo, zone_hi, score, n_touches.
    """
    if len(prices) == 0:
        return pd.DataFrame()

    order = np.argsort(prices)
    prices = prices[order]
    scores = scores[order]

    zones = []
    i = 0
    while i < len(prices):
        # Collect all prices within zone_width of the first
        j = i
        while j < len(prices) and prices[j] - prices[i] <= zone_width:
            j += 1
        cluster_prices = prices[i:j]
        cluster_scores = scores[i:j]
        # Weighted center
        total_score = cluster_scores.sum()
        center = float(np.average(cluster_prices, weights=cluster_scores))
        zones.append({
            "zone_center": center,
            "zone_lo":     float(cluster_prices.min()),
            "zone_hi":     float(cluster_prices.max()),
            "score":       float(total_score),
            "n_touches":   len(cluster_prices),
        })
        i = j

    return pd.DataFrame(zones)


def detect_supports(h1: pd.DataFrame, current_price: float,
                    zone_width: float = 12.0) -> pd.DataFrame:
    """
    Run all detection methods and merge into scored support zones.
    Only returns zones below current price (relevant for buy-limit ladder).
    """
    now = h1["time"].iloc[-1]

    # 1. Swing lows
    swings = find_swing_lows(h1, window=10)
    swing_prices = []
    swing_scores = []
    if not swings.empty:
        swings = swings[swings["price"] < current_price]
        for _, s in swings.iterrows():
            # Score by bounce magnitude + recency
            age_days = (now - s["time"]).total_seconds() / 86400
            recency = max(0.1, 1.0 / (1.0 + age_days / 90))  # half-life ~90 days
            score = s["bounce"] * recency
            swing_prices.append(s["price"])
            swing_scores.append(score)

    # 2. Volume profile high-volume nodes
    vp = volume_profile(h1, n_bins=300,
                        price_lo=current_price * 0.85,
                        price_hi=current_price)
    hvn = vp[vp["is_high"]]
    vol_prices = hvn["price_mid"].values
    vol_scores = hvn["volume"].values
    # Normalize volume scores to similar range as swing scores
    if len(vol_scores) > 0 and vol_scores.max() > 0:
        vol_scores = vol_scores / vol_scores.max() * 30.0

    # 3. Round numbers (multiples of 50 below current price)
    round_prices = []
    round_scores = []
    base = int(current_price * 0.85) // 50 * 50
    while base < current_price:
        round_prices.append(float(base))
        # 100-multiples stronger than 50-multiples
        round_scores.append(5.0 if base % 100 == 0 else 2.0)
        base += 50

    # Combine all
    all_prices = np.array(swing_prices + list(vol_prices) + round_prices)
    all_scores = np.array(swing_scores + list(vol_scores) + round_scores)

    if len(all_prices) == 0:
        return pd.DataFrame()

    zones = cluster_supports(all_prices, all_scores, zone_width=zone_width)
    zones = zones.sort_values("score", ascending=False).reset_index(drop=True)

    # Add distance from current price
    zones["dist_pts"] = current_price - zones["zone_center"]
    zones["dist_pct"] = zones["dist_pts"] / current_price * 100

    return zones


# ── Chart ─────────────────────────────────────────────────────────────────────

def draw_candles(ax, df: pd.DataFrame):
    w = 0.65
    for i, row in df.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.bar(i, abs(c - o), bottom=min(o, c), width=w,
               color=color, alpha=0.9, linewidth=0)
        ax.plot([i, i], [l, h], color=color, lw=0.7, alpha=0.85, zorder=2)


def plot(zones: pd.DataFrame, d1: pd.DataFrame, vp: pd.DataFrame,
         pending: pd.DataFrame, current_price: float) -> plt.Figure:
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[3, 2],
                           width_ratios=[3, 1],
                           hspace=0.35, wspace=0.08)

    ax_price = fig.add_subplot(gs[0, 0])    # D1 candles + zones
    ax_vp    = fig.add_subplot(gs[0, 1], sharey=ax_price)  # volume profile
    ax_table = fig.add_subplot(gs[1, :])    # support table

    n_zones = len(zones)
    pending_total = f"{pending['volume'].sum():.3f}" if not pending.empty else "0"
    fig.suptitle(
        f"Support Levels -- US500.pro @ {current_price:,.1f}  |  "
        f"{n_zones} support zones detected  |  "
        f"{len(pending)} pending orders ({pending_total} lots)",
        fontsize=11,
    )

    # ── Panel 1: D1 candles with support zones ────────────────────────────────
    draw_candles(ax_price, d1)
    n = len(d1)

    # Current price
    ax_price.axhline(current_price, color="#ffffff", lw=1.0, ls=":",
                     alpha=0.7, zorder=4, label=f"Current {current_price:,.0f}")

    # Support zones as horizontal bands
    max_score = zones["score"].max() if not zones.empty else 1
    cmap = plt.cm.YlOrRd
    for rank, (_, z) in enumerate(zones.head(25).iterrows()):
        intensity = min(1.0, z["score"] / max_score)
        color = cmap(0.3 + intensity * 0.6)
        alpha = 0.08 + intensity * 0.15
        ax_price.axhspan(z["zone_lo"], z["zone_hi"],
                         color=color, alpha=alpha, zorder=1)
        ax_price.axhline(z["zone_center"], color=color, lw=0.8,
                         ls="-", alpha=0.5 + intensity * 0.3, zorder=3)
        # Label top-10 zones
        if rank < 10:
            ax_price.text(n + 1, z["zone_center"],
                          f"  S{rank+1} {z['zone_center']:,.0f}  "
                          f"({z['n_touches']}t, {z['score']:.0f})",
                          fontsize=6.5, color=color, va="center",
                          clip_on=False)

    # Pending orders as small markers on right edge
    if not pending.empty:
        for _, o in pending.iterrows():
            ax_price.plot([n - 3, n + 0.5], [o["price"], o["price"]],
                          color="#40c0f0", lw=0.6, alpha=0.6, zorder=5)
        ax_price.plot([], [], color="#40c0f0", lw=1.5, label="Pending orders")

    # X-axis ticks
    step = max(1, n // 12)
    idxs = list(range(0, n, step))
    labels = [d1["time"].iloc[i].strftime("%d %b") for i in idxs]
    ax_price.set_xticks(idxs)
    ax_price.set_xticklabels(labels, fontsize=7.5)
    ax_price.set_xlim(-1, n + 8)

    # Focus Y-axis on relevant range (around pending orders and zones)
    y_lo = min(
        zones["zone_lo"].min() if not zones.empty else current_price * 0.85,
        pending["price"].min() if not pending.empty else current_price * 0.85,
        d1["low"].min(),
    ) - 30
    y_hi = current_price + 50
    ax_price.set_ylim(y_lo, y_hi)
    ax_price.set_ylabel("US500.pro")
    ax_price.set_title(f"D1 price chart with support zones  ({len(d1)} bars)")
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_price.legend(fontsize=7.5, loc="lower right")
    ax_price.grid(True, alpha=0.12)

    # ── Side panel: Volume profile ────────────────────────────────────────────
    vp_vis = vp[(vp["price_mid"] >= y_lo) & (vp["price_mid"] <= y_hi)]
    bar_colors = ["#f08040" if h else "#555555" for h in vp_vis["is_high"]]
    ax_vp.barh(vp_vis["price_mid"], vp_vis["volume"],
               height=(vp_vis["price_mid"].diff().median() or 5) * 0.9,
               color=bar_colors, alpha=0.7)
    ax_vp.set_xlabel("Volume", fontsize=8)
    ax_vp.set_title("Volume profile", fontsize=9)
    ax_vp.tick_params(labelleft=False)
    ax_vp.grid(True, alpha=0.12, axis="x")

    # ── Panel 2: Support table ────────────────────────────────────────────────
    ax_table.axis("off")

    top = zones.head(20).copy()
    if top.empty:
        ax_table.text(0.5, 0.5, "No support zones detected",
                      ha="center", va="center", fontsize=14, color="#888888")
    else:
        # Check which zones have pending orders nearby
        top["has_order"] = False
        top["order_lots"] = 0.0
        if not pending.empty:
            for idx, z in top.iterrows():
                nearby = pending[
                    (pending["price"] >= z["zone_lo"] - 10) &
                    (pending["price"] <= z["zone_hi"] + 10)
                ]
                if not nearby.empty:
                    top.at[idx, "has_order"] = True
                    top.at[idx, "order_lots"] = nearby["volume"].sum()

        col_labels = ["Rank", "Support", "Zone range", "Dist (pts)",
                      "Dist (%)", "Touches", "Score", "Orders?"]
        table_data = []
        for rank, (_, z) in enumerate(top.iterrows()):
            order_str = (f"{z['order_lots']:.3f}L" if z["has_order"]
                         else "--")
            table_data.append([
                f"S{rank+1}",
                f"{z['zone_center']:,.1f}",
                f"{z['zone_lo']:,.0f} - {z['zone_hi']:,.0f}",
                f"-{z['dist_pts']:,.0f}",
                f"-{z['dist_pct']:.1f}%",
                f"{z['n_touches']:.0f}",
                f"{z['score']:.1f}",
                order_str,
            ])

        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        # Style header
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#333333")
            cell.set_text_props(color="#ffffff", fontweight="bold")

        # Color rows by whether they have orders
        for i in range(len(table_data)):
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                cell.set_facecolor("#1a1a2e")
                cell.set_text_props(color="#dddddd")
                if top.iloc[i]["has_order"]:
                    cell.set_facecolor("#1a2e1a")  # green tint = covered

        ax_table.set_title(
            "Top 20 support zones  "
            "(green rows = already have pending orders nearby)",
            fontsize=9, pad=15,
        )

    plt.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Support-level detection for US500.pro ladder placement."
    )
    parser.add_argument("--months", type=int, default=12,
                        help="Months of H1 history to analyze (default 12)")
    parser.add_argument("--zone-width", type=float, default=12.0,
                        help="Max width to merge nearby supports into one zone (default 12 pts)")
    parser.add_argument("--chart-days", type=int, default=120,
                        help="D1 bars to show in price chart (default 120)")
    args = parser.parse_args()

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        # Fetch data
        print(f"Fetching H1 data ({args.months} months)...")
        h1 = fetch_h1(args.months)
        if h1.empty:
            print("ERROR: No H1 data available.")
            sys.exit(1)
        print(f"  {len(h1):,} H1 bars loaded "
              f"({h1['time'].iloc[0].date()} -> {h1['time'].iloc[-1].date()})")

        print(f"Fetching D1 data (last {args.chart_days} bars)...")
        d1 = fetch_d1_recent(args.chart_days)
        if d1.empty:
            print("ERROR: No D1 data available.")
            sys.exit(1)
        print(f"  {len(d1)} D1 bars loaded")

        # Current price
        tick = mt5.symbol_info_tick("US500.pro")
        current_price = tick.bid
        print(f"  Current price: {current_price:.1f}")

        # Pending orders
        pending = get_pending_orders()
        if not pending.empty:
            print(f"  {len(pending)} pending orders "
                  f"({pending['volume'].sum():.3f} lots, "
                  f"{pending['price'].min():.0f} - {pending['price'].max():.0f})")

        # Detect supports
        print(f"Detecting support zones (zone width = {args.zone_width} pts)...")
        zones = detect_supports(h1, current_price, zone_width=args.zone_width)
        print(f"  {len(zones)} zones found")

        if zones.empty:
            print("No support zones detected.")
            sys.exit(0)

        # Console summary: top 15
        print(f"\n{'='*80}")
        print(f"  TOP SUPPORT ZONES -- US500.pro @ {current_price:,.1f}")
        print(f"{'='*80}")
        print(f"  {'Rank':>4}  {'Center':>8}  {'Zone':>17}  "
              f"{'Dist':>8}  {'Dist%':>6}  {'Touches':>7}  {'Score':>7}  {'Orders':>8}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*17}  "
              f"{'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}")

        for rank, (_, z) in enumerate(zones.head(15).iterrows()):
            # Check for nearby pending orders
            has_order = ""
            if not pending.empty:
                nearby = pending[
                    (pending["price"] >= z["zone_lo"] - 10) &
                    (pending["price"] <= z["zone_hi"] + 10)
                ]
                if not nearby.empty:
                    has_order = f"{nearby['volume'].sum():.3f}L"

            print(f"  S{rank+1:>3}  {z['zone_center']:>8,.1f}  "
                  f"{z['zone_lo']:>7,.0f} - {z['zone_hi']:>7,.0f}  "
                  f"{-z['dist_pts']:>+8,.0f}  {-z['dist_pct']:>5.1f}%  "
                  f"{z['n_touches']:>7.0f}  {z['score']:>7.1f}  "
                  f"{has_order:>8}")
        print(f"{'='*80}")

        # Volume profile for chart
        vp = volume_profile(h1, n_bins=300,
                            price_lo=current_price * 0.85,
                            price_hi=current_price)

        # Plot
        fig = plot(zones, d1, vp, pending, current_price)
        _save_chart(fig, "02_support_levels")

        # Save CSV
        _RESULTS.mkdir(parents=True, exist_ok=True)
        csv_path = _RESULTS / "02_support_levels.csv"
        zones.round(2).to_csv(csv_path, index=False)
        print(f"  Saved zones -> results/02_support_levels.csv")

        plt.show()

    finally:
        disconnect()


if __name__ == "__main__":
    main()
