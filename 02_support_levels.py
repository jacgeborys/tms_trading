"""
02_support_levels.py -- Support-level detection + ladder proposal for US500.pro.

Identifies support zones from H1 price history using three methods:
  1. Swing lows  -- local minima (price reversed upward after touching)
  2. Volume profile -- price bins with above-average tick volume (consolidation)
  3. Round numbers -- psychological levels (multiples of 50 and 100)

Then proposes a buy-limit ladder where:
  - Each support zone gets a cluster of orders distributed radially (bell shape)
  - Stronger supports get wider, heavier clusters ("thicker")
  - Deeper supports get more total volume (the lower the heavier)

Produces two charts:
  Chart 1 -- Market analysis: D1 candles + support zones + proposed ladder
  Chart 2 -- Comparison: current pending orders vs proposed ladder

Usage:
  python 02_support_levels.py                         # defaults
  python 02_support_levels.py --budget 0.100           # total lots to distribute
  python 02_support_levels.py --months 18              # longer lookback
  python 02_support_levels.py --n-zones 15             # more zones to cover
  python 02_support_levels.py --rung-step 5            # 5-pt spacing within clusters

Outputs:
  results/charts/02_support_levels.png        (market analysis + proposed ladder)
  results/charts/02_support_comparison.png    (current vs proposed)
  results/02_support_levels.csv               (zone list)
  results/02_proposed_ladder.csv              (order-by-order proposal)
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

_ROOT    = Path(__file__).parent
_CHARTS  = _ROOT / "results" / "charts"
_RESULTS = _ROOT / "results"

LOT_STEP = 0.001   # broker minimum lot increment
LOT_MIN  = 0.001


def _save_chart(fig, name: str):
    _CHARTS.mkdir(parents=True, exist_ok=True)
    path = _CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved chart -> results/charts/{name}.png")


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_h1(months: int) -> pd.DataFrame:
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
    mt5.symbol_select("US500.pro", True)
    rates = mt5.copy_rates_from_pos("US500.pro", mt5.TIMEFRAME_D1, 0, days)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df.reset_index(drop=True)


def get_pending_orders() -> pd.DataFrame:
    orders = mt5.orders_get()
    if orders is None or len(orders) == 0:
        return pd.DataFrame()
    rows = []
    for o in orders:
        if o.symbol != "US500.pro":
            continue
        rows.append({
            "ticket": o.ticket,
            "type":   o.type,
            "volume": o.volume_current,
            "price":  o.price_open,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("price").reset_index(drop=True)


# ── Support detection ─────────────────────────────────────────────────────────

def find_swing_lows(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    lows = df["low"].values
    n = len(lows)
    swings = []
    for i in range(window, n - window):
        lo = lows[i]
        neighborhood = lows[max(0, i - window):i + window + 1]
        if lo == neighborhood.min():
            future_high = lows[i:min(i + window * 2, n)].max() if i + 1 < n else lo
            bounce = future_high - lo
            swings.append({"time": df["time"].iloc[i], "price": lo, "bounce": bounce})
    return pd.DataFrame(swings) if swings else pd.DataFrame()


def volume_profile(df: pd.DataFrame, n_bins: int = 200,
                   price_lo: float = None, price_hi: float = None) -> pd.DataFrame:
    if price_lo is None:
        price_lo = df["low"].min()
    if price_hi is None:
        price_hi = df["high"].max()
    bin_edges = np.linspace(price_lo, price_hi, n_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_counts = np.zeros(n_bins)
    for _, row in df.iterrows():
        lo_idx = max(0, np.searchsorted(bin_edges, row["low"], side="right") - 1)
        hi_idx = min(n_bins - 1, np.searchsorted(bin_edges, row["high"], side="left"))
        n_spanned = hi_idx - lo_idx + 1
        if n_spanned > 0:
            vol_counts[lo_idx:hi_idx + 1] += row["volume"] / n_spanned
    vp = pd.DataFrame({"price_mid": bin_mids, "volume": vol_counts})
    threshold = np.percentile(vol_counts[vol_counts > 0], 75)
    vp["is_high"] = vp["volume"] >= threshold
    return vp


def cluster_supports(prices: np.ndarray, scores: np.ndarray,
                     zone_width: float = 12.0) -> pd.DataFrame:
    if len(prices) == 0:
        return pd.DataFrame()
    order = np.argsort(prices)
    prices, scores = prices[order], scores[order]
    zones = []
    i = 0
    while i < len(prices):
        j = i
        while j < len(prices) and prices[j] - prices[i] <= zone_width:
            j += 1
        cp, cs = prices[i:j], scores[i:j]
        zones.append({
            "zone_center": float(np.average(cp, weights=cs)),
            "zone_lo":     float(cp.min()),
            "zone_hi":     float(cp.max()),
            "score":       float(cs.sum()),
            "n_touches":   len(cp),
        })
        i = j
    return pd.DataFrame(zones)


def detect_supports(h1: pd.DataFrame, current_price: float,
                    zone_width: float = 12.0) -> pd.DataFrame:
    now = h1["time"].iloc[-1]

    # Swing lows
    swings = find_swing_lows(h1, window=10)
    swing_prices, swing_scores = [], []
    if not swings.empty:
        swings = swings[swings["price"] < current_price]
        for _, s in swings.iterrows():
            age_days = (now - s["time"]).total_seconds() / 86400
            recency = max(0.1, 1.0 / (1.0 + age_days / 90))
            swing_prices.append(s["price"])
            swing_scores.append(s["bounce"] * recency)

    # Volume profile high-volume nodes
    vp = volume_profile(h1, n_bins=300,
                        price_lo=current_price * 0.85, price_hi=current_price)
    hvn = vp[vp["is_high"]]
    vol_prices = hvn["price_mid"].values
    vol_scores = hvn["volume"].values
    if len(vol_scores) > 0 and vol_scores.max() > 0:
        vol_scores = vol_scores / vol_scores.max() * 30.0

    # Round numbers
    round_prices, round_scores = [], []
    base = int(current_price * 0.85) // 50 * 50
    while base < current_price:
        round_prices.append(float(base))
        round_scores.append(5.0 if base % 100 == 0 else 2.0)
        base += 50

    all_prices = np.array(swing_prices + list(vol_prices) + round_prices)
    all_scores = np.array(swing_scores + list(vol_scores) + round_scores)
    if len(all_prices) == 0:
        return pd.DataFrame()

    zones = cluster_supports(all_prices, all_scores, zone_width=zone_width)
    zones = zones.sort_values("score", ascending=False).reset_index(drop=True)
    zones["dist_pts"] = current_price - zones["zone_center"]
    zones["dist_pct"] = zones["dist_pts"] / current_price * 100
    return zones


# ── Ladder proposal ───────────────────────────────────────────────────────────

def propose_ladder(zones: pd.DataFrame, current_price: float,
                   budget: float, n_zones: int, rung_step: float,
                   depth_power: float = 1.5) -> pd.DataFrame:
    """
    Build a radial-distribution ladder across the top support zones.

    For each zone:
      - raw_weight = depth^depth_power * score  (deeper + stronger = heavier)
      - Cluster width (# rungs) scales with score (stronger = wider spread)
      - Within the cluster, lots follow a triangular (tent) distribution
        centered on zone_center

    Returns DataFrame with columns: price, volume, zone_rank, zone_center.
    """
    top = zones.head(n_zones).copy()
    if top.empty:
        return pd.DataFrame()

    max_dist = top["dist_pts"].max()
    max_score = top["score"].max()

    # Raw weight per zone: depth contribution * score contribution
    top["depth_w"] = (top["dist_pts"] / max_dist) ** depth_power
    top["raw_weight"] = top["depth_w"] * top["score"]
    total_weight = top["raw_weight"].sum()

    # Allocate budget across zones proportionally
    top["alloc_lots"] = top["raw_weight"] / total_weight * budget

    # Determine cluster half-width in rungs for each zone
    # Stronger zones get wider spread: 1..max_half rungs on each side
    max_half = 4   # up to +-4 rungs from center = 9 rungs at strongest zone
    top["score_norm"] = top["score"] / max_score
    top["half_rungs"] = np.clip(
        np.round(top["score_norm"] * max_half).astype(int), 0, max_half
    )

    all_rungs = []

    for rank, (_, z) in enumerate(top.iterrows()):
        center = z["zone_center"]
        alloc = z["alloc_lots"]
        half_n = int(z["half_rungs"])

        # Build triangular weights: peak at center, linear taper
        offsets = list(range(-half_n, half_n + 1))
        weights = np.array([half_n + 1 - abs(k) for k in offsets], dtype=float)
        weights /= weights.sum()

        # Distribute lots with rounding
        raw_lots = weights * alloc
        rounded = np.maximum(
            np.round(raw_lots / LOT_STEP) * LOT_STEP,
            0.0,
        )
        # Ensure at least the center rung gets LOT_MIN
        center_idx = half_n
        if rounded[center_idx] < LOT_MIN:
            rounded[center_idx] = LOT_MIN

        for i, off in enumerate(offsets):
            lot = rounded[i]
            if lot < LOT_MIN:
                continue
            price = round(center + off * rung_step, 1)
            if price >= current_price:
                continue
            all_rungs.append({
                "price":       price,
                "volume":      round(lot, 3),
                "zone_rank":   rank + 1,
                "zone_center": round(center, 1),
                "zone_score":  round(z["score"], 1),
            })

    if not all_rungs:
        return pd.DataFrame()

    ladder = pd.DataFrame(all_rungs).sort_values("price").reset_index(drop=True)

    # Merge duplicate prices (different zones can overlap)
    ladder = (ladder.groupby("price", as_index=False)
              .agg({"volume": "sum", "zone_rank": "min",
                    "zone_center": "first", "zone_score": "max"}))
    ladder["volume"] = np.maximum(
        np.round(ladder["volume"] / LOT_STEP) * LOT_STEP, LOT_MIN
    )
    return ladder.sort_values("price").reset_index(drop=True)


# ── Charting helpers ──────────────────────────────────────────────────────────

def draw_candles(ax, df: pd.DataFrame):
    w = 0.65
    for i, row in df.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.bar(i, abs(c - o), bottom=min(o, c), width=w,
               color=color, alpha=0.9, linewidth=0)
        ax.plot([i, i], [l, h], color=color, lw=0.7, alpha=0.85, zorder=2)


def _draw_support_zones(ax, zones, n_bars):
    """Draw support zone bands and labels on a price axis."""
    max_score = zones["score"].max() if not zones.empty else 1
    cmap = plt.cm.YlOrRd
    for rank, (_, z) in enumerate(zones.head(25).iterrows()):
        intensity = min(1.0, z["score"] / max_score)
        color = cmap(0.3 + intensity * 0.6)
        alpha = 0.08 + intensity * 0.15
        ax.axhspan(z["zone_lo"], z["zone_hi"],
                    color=color, alpha=alpha, zorder=1)
        ax.axhline(z["zone_center"], color=color, lw=0.8,
                    ls="-", alpha=0.5 + intensity * 0.3, zorder=3)
        if rank < 12:
            ax.text(n_bars + 1, z["zone_center"],
                    f"  S{rank+1} {z['zone_center']:,.0f}  "
                    f"({z['n_touches']}t, {z['score']:.0f})",
                    fontsize=6.5, color=color, va="center", clip_on=False)


def _draw_ladder_markers(ax, ladder, n_bars, color, label, side="right"):
    """Draw ladder orders as horizontal ticks, size proportional to volume."""
    if ladder.empty:
        return
    max_vol = ladder["volume"].max()
    for _, r in ladder.iterrows():
        # Tick length proportional to lot size
        tick_len = 2 + (r["volume"] / max_vol) * 6
        if side == "right":
            x0, x1 = n_bars - tick_len, n_bars + 0.5
        else:
            x0, x1 = -0.5, tick_len
        alpha = 0.4 + 0.5 * (r["volume"] / max_vol)
        ax.plot([x0, x1], [r["price"], r["price"]],
                color=color, lw=1.0 + r["volume"] / max_vol * 1.5,
                alpha=alpha, zorder=5)
    ax.plot([], [], color=color, lw=2, label=label)


def _price_ylim(zones, pending, ladder, current_price, d1):
    """Compute Y-axis limits that show all relevant price levels."""
    candidates = [current_price * 0.85]
    if not zones.empty:
        candidates.append(zones["zone_lo"].min())
    if not pending.empty:
        candidates.append(pending["price"].min())
    if not ladder.empty:
        candidates.append(ladder["price"].min())
    candidates.append(d1["low"].min())
    return min(candidates) - 30, current_price + 50


# ── Chart 1: Market analysis + proposed ladder ────────────────────────────────

def plot_market(zones, d1, vp, ladder, current_price) -> plt.Figure:
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[3, 2], width_ratios=[3, 1],
                           hspace=0.30, wspace=0.08)
    ax_price = fig.add_subplot(gs[0, 0])
    ax_vp    = fig.add_subplot(gs[0, 1], sharey=ax_price)
    ax_table = fig.add_subplot(gs[1, :])

    n_zones = len(zones)
    n_rungs = len(ladder) if not ladder.empty else 0
    total_lots = ladder["volume"].sum() if not ladder.empty else 0
    fig.suptitle(
        f"Market Analysis + Proposed Ladder -- US500.pro @ {current_price:,.1f}  |  "
        f"{n_zones} support zones  |  "
        f"Proposed: {n_rungs} orders, {total_lots:.3f} lots",
        fontsize=11,
    )

    # ── Price chart ───────────────────────────────────────────────────────────
    draw_candles(ax_price, d1)
    n = len(d1)

    ax_price.axhline(current_price, color="#ffffff", lw=1.0, ls=":",
                     alpha=0.7, zorder=4, label=f"Current {current_price:,.0f}")

    _draw_support_zones(ax_price, zones, n)
    _draw_ladder_markers(ax_price, ladder, n,
                         color="#00e676", label=f"Proposed ({total_lots:.3f}L)")

    step = max(1, n // 12)
    idxs = list(range(0, n, step))
    labels = [d1["time"].iloc[i].strftime("%d %b") for i in idxs]
    ax_price.set_xticks(idxs)
    ax_price.set_xticklabels(labels, fontsize=7.5)
    ax_price.set_xlim(-1, n + 8)

    y_lo, y_hi = _price_ylim(zones, pd.DataFrame(), ladder, current_price, d1)
    ax_price.set_ylim(y_lo, y_hi)
    ax_price.set_ylabel("US500.pro")
    ax_price.set_title(f"D1 price chart with support zones + proposed ladder  ({n} bars)")
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_price.legend(fontsize=7.5, loc="lower right")
    ax_price.grid(True, alpha=0.12)

    # ── Volume profile ────────────────────────────────────────────────────────
    vp_vis = vp[(vp["price_mid"] >= y_lo) & (vp["price_mid"] <= y_hi)]
    bar_colors = ["#f08040" if h else "#555555" for h in vp_vis["is_high"]]
    ax_vp.barh(vp_vis["price_mid"], vp_vis["volume"],
               height=(vp_vis["price_mid"].diff().median() or 5) * 0.9,
               color=bar_colors, alpha=0.7)
    ax_vp.set_xlabel("Volume", fontsize=8)
    ax_vp.set_title("Volume profile", fontsize=9)
    ax_vp.tick_params(labelleft=False)
    ax_vp.grid(True, alpha=0.12, axis="x")

    # ── Proposed ladder table ─────────────────────────────────────────────────
    ax_table.axis("off")

    if ladder.empty:
        ax_table.text(0.5, 0.5, "No ladder proposed",
                      ha="center", va="center", fontsize=14, color="#888888")
    else:
        # Group by zone for display
        by_zone = (ladder.groupby("zone_rank")
                   .agg({"price": ["min", "max", "count"],
                         "volume": "sum",
                         "zone_center": "first",
                         "zone_score": "first"})
                   .reset_index())
        by_zone.columns = ["rank", "price_lo", "price_hi", "n_rungs",
                           "total_lots", "center", "score"]
        by_zone["dist"] = current_price - by_zone["center"]

        col_labels = ["Zone", "Center", "Score", "Dist (pts)", "Dist (%)",
                      "Rungs", "Price range", "Total lots"]
        table_data = []
        for _, bz in by_zone.iterrows():
            table_data.append([
                f"S{bz['rank']:.0f}",
                f"{bz['center']:,.1f}",
                f"{bz['score']:.0f}",
                f"-{bz['dist']:,.0f}",
                f"-{bz['dist'] / current_price * 100:.1f}%",
                f"{bz['n_rungs']:.0f}",
                f"{bz['price_lo']:,.1f} - {bz['price_hi']:,.1f}",
                f"{bz['total_lots']:.3f}",
            ])

        table = ax_table.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#333333")
            table[0, j].set_text_props(color="#ffffff", fontweight="bold")
        for i in range(len(table_data)):
            # Color intensity by lot allocation
            lots = by_zone.iloc[i]["total_lots"]
            max_lots = by_zone["total_lots"].max()
            g = int(30 + 40 * (lots / max_lots))
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(f"#{26:02x}{g:02x}{26:02x}")
                table[i + 1, j].set_text_props(color="#dddddd")

        ax_table.set_title(
            "Proposed ladder by zone  (deeper + stronger = heavier allocation, "
            "brighter green = more lots)", fontsize=9, pad=15,
        )

    return fig


# ── Chart 2: Current pending vs proposed ──────────────────────────────────────

def plot_comparison(zones, d1, pending, ladder, current_price) -> plt.Figure:
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 2], hspace=0.30)
    ax_price = fig.add_subplot(gs[0])
    ax_dist  = fig.add_subplot(gs[1])

    pend_lots = pending["volume"].sum() if not pending.empty else 0
    prop_lots = ladder["volume"].sum() if not ladder.empty else 0
    fig.suptitle(
        f"Current vs Proposed Ladder -- US500.pro @ {current_price:,.1f}  |  "
        f"Current: {len(pending)} orders, {pend_lots:.3f}L  |  "
        f"Proposed: {len(ladder)} orders, {prop_lots:.3f}L",
        fontsize=11,
    )

    # ── Panel 1: Price chart with both ladders ────────────────────────────────
    draw_candles(ax_price, d1)
    n = len(d1)

    ax_price.axhline(current_price, color="#ffffff", lw=1.0, ls=":",
                     alpha=0.7, zorder=4, label=f"Current {current_price:,.0f}")
    _draw_support_zones(ax_price, zones, n)

    # Current = cyan on the left side
    _draw_ladder_markers(ax_price, pending, n,
                         color="#40c0f0", label=f"Current ({pend_lots:.3f}L)",
                         side="left")
    # Proposed = green on the right side
    _draw_ladder_markers(ax_price, ladder, n,
                         color="#00e676", label=f"Proposed ({prop_lots:.3f}L)",
                         side="right")

    step = max(1, n // 12)
    idxs = list(range(0, n, step))
    labels = [d1["time"].iloc[i].strftime("%d %b") for i in idxs]
    ax_price.set_xticks(idxs)
    ax_price.set_xticklabels(labels, fontsize=7.5)
    ax_price.set_xlim(-1, n + 8)

    y_lo, y_hi = _price_ylim(zones, pending, ladder, current_price, d1)
    ax_price.set_ylim(y_lo, y_hi)
    ax_price.set_ylabel("US500.pro")
    ax_price.set_title(
        "Current pending (cyan, left) vs Proposed ladder (green, right)")
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_price.legend(fontsize=8, loc="lower right")
    ax_price.grid(True, alpha=0.12)

    # ── Panel 2: Volume distribution by price ─────────────────────────────────
    # Horizontal bar chart: for each price bin, show current and proposed lots
    bin_step = 25.0
    price_lo = min(
        pending["price"].min() if not pending.empty else current_price,
        ladder["price"].min() if not ladder.empty else current_price,
    )
    price_hi = current_price
    bins = np.arange(
        (price_lo // bin_step) * bin_step,
        price_hi + bin_step,
        bin_step,
    )

    curr_dist = np.zeros(len(bins))
    prop_dist = np.zeros(len(bins))

    if not pending.empty:
        for _, o in pending.iterrows():
            idx = np.argmin(np.abs(bins - o["price"]))
            curr_dist[idx] += o["volume"]

    if not ladder.empty:
        for _, r in ladder.iterrows():
            idx = np.argmin(np.abs(bins - r["price"]))
            prop_dist[idx] += r["volume"]

    bar_h = bin_step * 0.4
    ax_dist.barh(bins - bar_h * 0.55, curr_dist, height=bar_h,
                 color="#40c0f0", alpha=0.7, label=f"Current ({pend_lots:.3f}L)",
                 zorder=3)
    ax_dist.barh(bins + bar_h * 0.55, prop_dist, height=bar_h,
                 color="#00e676", alpha=0.7, label=f"Proposed ({prop_lots:.3f}L)",
                 zorder=3)

    ax_dist.set_ylabel("Price level")
    ax_dist.set_xlabel("Lots")
    ax_dist.set_title(
        f"Volume distribution by price ({bin_step:.0f}-pt bins)  --  "
        f"cyan = current  |  green = proposed")
    ax_dist.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_dist.legend(fontsize=8, loc="lower right")
    ax_dist.grid(True, alpha=0.12, axis="x")
    ax_dist.axhline(current_price, color="#ffffff", lw=0.8, ls=":", alpha=0.5)

    return fig


# ── Console output ────────────────────────────────────────────────────────────

def print_zone_summary(zones, pending, current_price):
    print(f"\n{'='*85}")
    print(f"  TOP SUPPORT ZONES -- US500.pro @ {current_price:,.1f}")
    print(f"{'='*85}")
    print(f"  {'Rank':>4}  {'Center':>8}  {'Zone':>17}  "
          f"{'Dist':>8}  {'Dist%':>6}  {'Touches':>7}  {'Score':>7}  {'Orders':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*17}  "
          f"{'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}")

    for rank, (_, z) in enumerate(zones.head(15).iterrows()):
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
    print(f"{'='*85}")


def print_ladder_summary(ladder, current_price):
    if ladder.empty:
        print("\n  No ladder rungs proposed.")
        return
    print(f"\n{'='*70}")
    print(f"  PROPOSED LADDER -- {len(ladder)} orders, "
          f"{ladder['volume'].sum():.3f} lots total")
    print(f"{'='*70}")
    print(f"  {'Price':>8}  {'Lots':>6}  {'Dist':>8}  {'Zone':>6}  {'Zone center':>12}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*12}")
    for _, r in ladder.iterrows():
        dist = current_price - r["price"]
        print(f"  {r['price']:>8,.1f}  {r['volume']:>6.3f}  "
              f"{-dist:>+8,.0f}  S{r['zone_rank']:>4.0f}  "
              f"{r['zone_center']:>12,.1f}")
    print(f"{'='*70}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Support-level detection + ladder proposal for US500.pro."
    )
    parser.add_argument("--months", type=int, default=12,
                        help="Months of H1 history (default 12)")
    parser.add_argument("--zone-width", type=float, default=12.0,
                        help="Max width to merge nearby supports (default 12 pts)")
    parser.add_argument("--chart-days", type=int, default=120,
                        help="D1 bars in price chart (default 120)")
    parser.add_argument("--budget", type=float, default=0.100,
                        help="Total lots to distribute across ladder (default 0.100)")
    parser.add_argument("--n-zones", type=int, default=12,
                        help="Number of top zones to place orders at (default 12)")
    parser.add_argument("--rung-step", type=float, default=5.0,
                        help="Point spacing between rungs within a cluster (default 5)")
    parser.add_argument("--depth-power", type=float, default=1.5,
                        help="Exponent for depth weighting -- higher = more "
                             "weight to deep levels (default 1.5)")
    args = parser.parse_args()

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
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

        tick = mt5.symbol_info_tick("US500.pro")
        current_price = tick.bid
        print(f"  Current price: {current_price:.1f}")

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

        # Propose ladder
        print(f"Building ladder proposal "
              f"({args.budget:.3f}L across {args.n_zones} zones, "
              f"{args.rung_step}-pt step, depth^{args.depth_power})...")
        ladder = propose_ladder(
            zones, current_price,
            budget=args.budget,
            n_zones=args.n_zones,
            rung_step=args.rung_step,
            depth_power=args.depth_power,
        )
        print(f"  {len(ladder)} rungs proposed, "
              f"{ladder['volume'].sum():.3f} lots total")

        # Console summaries
        print_zone_summary(zones, pending, current_price)
        print_ladder_summary(ladder, current_price)

        # Volume profile
        vp = volume_profile(h1, n_bins=300,
                            price_lo=current_price * 0.85,
                            price_hi=current_price)

        # Chart 1: Market analysis + proposed ladder
        fig1 = plot_market(zones, d1, vp, ladder, current_price)
        _save_chart(fig1, "02_support_levels")

        # Chart 2: Current vs proposed
        fig2 = plot_comparison(zones, d1, pending, ladder, current_price)
        _save_chart(fig2, "02_support_comparison")

        # Save CSVs
        _RESULTS.mkdir(parents=True, exist_ok=True)
        zones.round(2).to_csv(_RESULTS / "02_support_levels.csv", index=False)
        print(f"  Saved zones -> results/02_support_levels.csv")
        if not ladder.empty:
            ladder.to_csv(_RESULTS / "02_proposed_ladder.csv", index=False)
            print(f"  Saved ladder -> results/02_proposed_ladder.csv")

        plt.show()

    finally:
        disconnect()


if __name__ == "__main__":
    main()
