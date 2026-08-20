"""
02_support_levels.py -- Support-level detection + ladder proposal for US500.pro.

Identifies support zones from D1 price history using three methods:
  1. Swing lows           -- daily bars where price reversed upward
  2. Broken resistance    -- former daily resistance now acting as support
  3. Round numbers        -- psychological levels (multiples of 50/100)

Then proposes a buy-limit ladder where:
  - A background grid of 0.001L orders every --grid-step pts catches sideways drift
  - At support zones, extra 0.001L orders cluster radially (stronger = more rungs)
  - All orders are 0.001L -- density varies, not lot size
  - Grid snaps to round multiples (stable across runs)

Produces two charts:
  Chart 1 -- Pure market analysis: D1 candles + support zones + D1 volume profile
  Chart 2 -- Proposed vs current: ladders overlaid + smoothed density profile

Usage:
  python 02_support_levels.py                         # defaults
  python 02_support_levels.py --days 250              # ~1 year of D1
  python 02_support_levels.py --n-zones 15            # more zones to cover
  python 02_support_levels.py --grid-step 25          # wider grid spacing
  python 02_support_levels.py --floor 6300            # lowest price to cover

Outputs:
  results/charts/02_support_levels.png        (pure market analysis)
  results/charts/02_support_comparison.png    (current vs proposed + density)
  results/02_support_levels.csv               (zone list)
  results/02_proposed_ladder.csv              (order-by-order proposal)
"""

import sys
import argparse
from datetime import datetime, timezone

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

LOT = 0.001


def _save_chart(fig, name: str):
    _CHARTS.mkdir(parents=True, exist_ok=True)
    path = _CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved chart -> results/charts/{name}.png")


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_d1(n_bars: int) -> pd.DataFrame:
    mt5.symbol_select("US500.pro", True)
    rates = mt5.copy_rates_from_pos("US500.pro", mt5.TIMEFRAME_D1, 0, n_bars)
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

def find_swing_lows(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Swing lows on D1: bar where low is the minimum in a +-window neighborhood."""
    lows = df["low"].values
    n = len(lows)
    swings = []
    for i in range(window, n - window):
        lo = lows[i]
        neighborhood = lows[max(0, i - window):i + window + 1]
        if lo == neighborhood.min():
            # Bounce strength: how far the high went in the next `window` bars
            future_high = df["high"].values[i:min(i + window * 2, n)].max()
            bounce = future_high - lo
            swings.append({
                "time":   df["time"].iloc[i],
                "price":  lo,
                "bounce": bounce,
                "bar_idx": i,
            })
    return pd.DataFrame(swings) if swings else pd.DataFrame()


def find_broken_resistance(df: pd.DataFrame, window: int = 5,
                           min_break_bars: int = 10) -> pd.DataFrame:
    """Swing highs on D1 that were later broken upward."""
    highs = df["high"].values
    closes = df["close"].values
    n = len(highs)
    broken = []
    for i in range(window, n - window):
        hi = highs[i]
        neighborhood = highs[max(0, i - window):i + window + 1]
        if hi != neighborhood.max():
            continue
        future = closes[i + window:]
        if len(future) < min_break_bars:
            continue
        bars_above = int((future > hi).sum())
        if bars_above >= min_break_bars:
            broken.append({
                "time":       df["time"].iloc[i],
                "price":      hi,
                "bars_above": bars_above,
            })
    return pd.DataFrame(broken) if broken else pd.DataFrame()


def cluster_supports(prices: np.ndarray, scores: np.ndarray,
                     zone_width: float = 20.0) -> pd.DataFrame:
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


def detect_supports(d1: pd.DataFrame, current_price: float,
                    zone_width: float = 20.0) -> pd.DataFrame:
    now = d1["time"].iloc[-1]
    all_prices, all_scores = [], []

    # 1. Swing lows
    swings = find_swing_lows(d1, window=5)
    if not swings.empty:
        swings = swings[swings["price"] < current_price]
        for _, s in swings.iterrows():
            age_days = (now - s["time"]).total_seconds() / 86400
            recency = max(0.1, 1.0 / (1.0 + age_days / 90))
            all_prices.append(s["price"])
            all_scores.append(s["bounce"] * recency)

    # 2. Broken resistance
    broken = find_broken_resistance(d1, window=5, min_break_bars=10)
    if not broken.empty:
        broken = broken[broken["price"] < current_price]
        for _, b in broken.iterrows():
            age_days = (now - b["time"]).total_seconds() / 86400
            recency = max(0.1, 1.0 / (1.0 + age_days / 120))
            all_prices.append(b["price"])
            all_scores.append(b["bars_above"] * recency * 0.5)

    # 3. Round numbers
    base = int(current_price * 0.82) // 50 * 50
    while base < current_price:
        all_prices.append(float(base))
        all_scores.append(5.0 if base % 100 == 0 else 2.0)
        base += 50

    if not all_prices:
        return pd.DataFrame()

    all_prices = np.array(all_prices)
    all_scores = np.array(all_scores)

    zones = cluster_supports(all_prices, all_scores, zone_width=zone_width)
    zones = zones.sort_values("score", ascending=False).reset_index(drop=True)
    zones["dist_pts"] = current_price - zones["zone_center"]
    zones["dist_pct"] = zones["dist_pts"] / current_price * 100
    return zones


# ── Volume profile (D1) ──────────────────────────────────────────────────────

def volume_profile_d1(df: pd.DataFrame, n_bins: int = 100) -> pd.DataFrame:
    price_lo = df["low"].min()
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


# ── Ladder proposal ───────────────────────────────────────────────────────────

def propose_ladder(zones: pd.DataFrame, current_price: float,
                   n_zones: int, grid_step: float, cluster_step: float,
                   floor: float) -> pd.DataFrame:
    """
    Two-layer ladder of 0.001L orders:

    Layer 1 -- Background grid at round multiples of grid_step.
    Layer 2 -- Support clusters at integer prices around zone centers.
    """
    top = zones.head(n_zones).copy()
    max_score = top["score"].max() if not top.empty else 1
    max_dist = top["dist_pts"].max() if not top.empty else 1

    # Layer 1: background grid
    grid_top = int(current_price // grid_step) * grid_step
    grid_prices = np.arange(grid_top, floor - 0.1, -grid_step)

    rungs = {}
    for p in grid_prices:
        p = float(p)
        rungs[p] = {"volume": LOT, "source": "grid", "zone_rank": 0,
                    "zone_center": 0.0, "zone_score": 0.0}

    # Layer 2: support clusters
    for rank, (_, z) in enumerate(top.iterrows()):
        center = round(z["zone_center"])
        score_norm = z["score"] / max_score
        depth_norm = z["dist_pts"] / max_dist

        half_n = int(round(1 + 3 * score_norm + 1 * depth_norm))
        half_n = min(half_n, 5)

        for k in range(-half_n, half_n + 1):
            p = float(center + k * int(cluster_step))
            if p >= current_price or p < floor:
                continue
            if p in rungs:
                rungs[p]["volume"] = round(rungs[p]["volume"] + LOT, 3)
                if rungs[p]["source"] == "grid":
                    rungs[p]["source"] = "grid+support"
                if z["score"] > rungs[p]["zone_score"]:
                    rungs[p]["zone_rank"] = rank + 1
                    rungs[p]["zone_center"] = center
                    rungs[p]["zone_score"] = round(z["score"], 1)
            else:
                rungs[p] = {
                    "volume":      LOT,
                    "source":      "support",
                    "zone_rank":   rank + 1,
                    "zone_center": center,
                    "zone_score":  round(z["score"], 1),
                }

    rows = [{"price": price, **info} for price, info in sorted(rungs.items())]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Ladder density profile ────────────────────────────────────────────────────

def ladder_density(ladder: pd.DataFrame, y_lo: float, y_hi: float,
                   bin_width: float = 5.0, smooth_sigma: float = 3.0):
    bins = np.arange(y_lo, y_hi + bin_width, bin_width)
    counts = np.zeros(len(bins))
    for _, r in ladder.iterrows():
        idx = np.argmin(np.abs(bins - r["price"]))
        counts[idx] += r["volume"]
    kernel_size = int(smooth_sigma * 4) * 2 + 1
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (x / smooth_sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts, kernel, mode="same")
    return bins, smoothed


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
    if ladder.empty:
        return
    max_vol = ladder["volume"].max()
    for _, r in ladder.iterrows():
        tick_len = 2 + (r["volume"] / max_vol) * 6
        if side == "right":
            x0, x1 = n_bars - tick_len, n_bars + 0.5
        else:
            x0, x1 = -0.5, tick_len
        alpha = 0.3 + 0.6 * (r["volume"] / max_vol)
        ax.plot([x0, x1], [r["price"], r["price"]],
                color=color, lw=0.8 + r["volume"] / max_vol * 1.8,
                alpha=alpha, zorder=5)
    ax.plot([], [], color=color, lw=2, label=label)


def _price_ylim(zones, pending, ladder, current_price, d1):
    candidates = [current_price * 0.82]
    if not zones.empty:
        candidates.append(zones["zone_lo"].min())
    if not pending.empty:
        candidates.append(pending["price"].min())
    if not ladder.empty:
        candidates.append(ladder["price"].min())
    candidates.append(d1["low"].min())
    return min(candidates) - 30, current_price + 50


# ── Chart 1: Pure market analysis ─────────────────────────────────────────────

def plot_market(zones, d1, vp, current_price) -> plt.Figure:
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[3, 2], width_ratios=[3, 1],
                           hspace=0.30, wspace=0.08)
    ax_price = fig.add_subplot(gs[0, 0])
    ax_vp    = fig.add_subplot(gs[0, 1], sharey=ax_price)
    ax_table = fig.add_subplot(gs[1, :])

    fig.suptitle(
        f"Market Analysis -- US500.pro @ {current_price:,.1f}  |  "
        f"{len(zones)} support zones  |  D1 data ({len(d1)} bars)",
        fontsize=11,
    )

    draw_candles(ax_price, d1)
    n = len(d1)
    ax_price.axhline(current_price, color="#ffffff", lw=1.0, ls=":",
                     alpha=0.7, zorder=4, label=f"Current {current_price:,.0f}")
    _draw_support_zones(ax_price, zones, n)

    step = max(1, n // 12)
    idxs = list(range(0, n, step))
    labels = [d1["time"].iloc[i].strftime("%d %b") for i in idxs]
    ax_price.set_xticks(idxs)
    ax_price.set_xticklabels(labels, fontsize=7.5)
    ax_price.set_xlim(-1, n + 8)

    y_lo = min(zones["zone_lo"].min() if not zones.empty else current_price * 0.82,
               d1["low"].min()) - 30
    y_hi = current_price + 50
    ax_price.set_ylim(y_lo, y_hi)
    ax_price.set_ylabel("US500.pro")
    ax_price.set_title(f"D1 price chart with support zones")
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_price.legend(fontsize=7.5, loc="lower right")
    ax_price.grid(True, alpha=0.12)

    # Volume profile (D1)
    vp_vis = vp[(vp["price_mid"] >= y_lo) & (vp["price_mid"] <= y_hi)]
    bar_colors = ["#f08040" if h else "#555555" for h in vp_vis["is_high"]]
    ax_vp.barh(vp_vis["price_mid"], vp_vis["volume"],
               height=(vp_vis["price_mid"].diff().median() or 10) * 0.9,
               color=bar_colors, alpha=0.7)
    ax_vp.set_xlabel("Volume", fontsize=8)
    ax_vp.set_title("Volume profile (D1)", fontsize=9)
    ax_vp.tick_params(labelleft=False)
    ax_vp.grid(True, alpha=0.12, axis="x")

    # Support table
    ax_table.axis("off")
    top = zones.head(20).copy()
    if not top.empty:
        col_labels = ["Rank", "Support", "Zone range", "Dist (pts)",
                      "Dist (%)", "Touches", "Score"]
        table_data = []
        for rank, (_, z) in enumerate(top.iterrows()):
            table_data.append([
                f"S{rank+1}",
                f"{z['zone_center']:,.1f}",
                f"{z['zone_lo']:,.0f} - {z['zone_hi']:,.0f}",
                f"-{z['dist_pts']:,.0f}",
                f"-{z['dist_pct']:.1f}%",
                f"{z['n_touches']:.0f}",
                f"{z['score']:.0f}",
            ])
        table = ax_table.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)
        max_score = top["score"].max()
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#333333")
            table[0, j].set_text_props(color="#ffffff", fontweight="bold")
        for i in range(len(table_data)):
            intensity = top.iloc[i]["score"] / max_score
            r_val = int(26 + 20 * intensity)
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(f"#{r_val:02x}{26:02x}{26:02x}")
                table[i + 1, j].set_text_props(color="#dddddd")
        ax_table.set_title("Top 20 support zones  (brighter = stronger score)",
                           fontsize=9, pad=15)

    return fig


# ── Chart 2: Current vs proposed + density ────────────────────────────────────

def plot_comparison(zones, d1, pending, ladder, current_price) -> plt.Figure:
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[3, 2], width_ratios=[3, 1],
                           hspace=0.30, wspace=0.08)
    ax_price = fig.add_subplot(gs[0, 0])
    ax_dens  = fig.add_subplot(gs[0, 1], sharey=ax_price)
    ax_dist  = fig.add_subplot(gs[1, :])

    pend_lots = pending["volume"].sum() if not pending.empty else 0
    prop_lots = ladder["volume"].sum() if not ladder.empty else 0

    fig.suptitle(
        f"Current vs Proposed Ladder -- US500.pro @ {current_price:,.1f}  |  "
        f"Current: {len(pending)} orders, {pend_lots:.3f}L  |  "
        f"Proposed: {len(ladder)} orders, {prop_lots:.3f}L",
        fontsize=11,
    )

    draw_candles(ax_price, d1)
    n = len(d1)
    ax_price.axhline(current_price, color="#ffffff", lw=1.0, ls=":",
                     alpha=0.7, zorder=4, label=f"Current {current_price:,.0f}")
    _draw_support_zones(ax_price, zones, n)
    _draw_ladder_markers(ax_price, pending, n,
                         color="#40c0f0", label=f"Current ({pend_lots:.3f}L)",
                         side="left")
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

    # Density sidebar
    if not ladder.empty:
        dens_prices, dens_smooth = ladder_density(ladder, y_lo, y_hi)
        ax_dens.fill_betweenx(dens_prices, 0, dens_smooth,
                              color="#00e676", alpha=0.4, zorder=2)
        ax_dens.plot(dens_smooth, dens_prices,
                     color="#00e676", lw=1.2, alpha=0.8, zorder=3)
    if not pending.empty:
        pend_prices, pend_smooth = ladder_density(pending, y_lo, y_hi)
        ax_dens.fill_betweenx(pend_prices, 0, pend_smooth,
                              color="#40c0f0", alpha=0.25, zorder=1)
        ax_dens.plot(pend_smooth, pend_prices,
                     color="#40c0f0", lw=1.0, alpha=0.6, zorder=2, ls="--")
    ax_dens.set_xlabel("Density (lots)", fontsize=8)
    ax_dens.set_title("Order density", fontsize=9)
    ax_dens.tick_params(labelleft=False)
    ax_dens.grid(True, alpha=0.12, axis="x")

    # Volume distribution
    bin_step = 25.0
    price_lo = min(
        pending["price"].min() if not pending.empty else current_price,
        ladder["price"].min() if not ladder.empty else current_price,
    )
    bins = np.arange((price_lo // bin_step) * bin_step,
                     current_price + bin_step, bin_step)
    curr_dist = np.zeros(len(bins))
    prop_dist = np.zeros(len(bins))
    if not pending.empty:
        for _, o in pending.iterrows():
            curr_dist[np.argmin(np.abs(bins - o["price"]))] += o["volume"]
    if not ladder.empty:
        for _, r in ladder.iterrows():
            prop_dist[np.argmin(np.abs(bins - r["price"]))] += r["volume"]

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
                (pending["price"] >= z["zone_lo"] - 15) &
                (pending["price"] <= z["zone_hi"] + 15)
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
    total_L   = ladder["volume"].sum()
    grid_only = ladder[ladder["source"] == "grid"]
    support   = ladder[ladder["source"] != "grid"]
    boosted   = ladder[ladder["volume"] > LOT]
    print(f"\n{'='*75}")
    print(f"  PROPOSED LADDER -- {len(ladder)} orders, {total_L:.3f} lots total")
    print(f"  Grid-only: {len(grid_only)} x 0.001L  |  "
          f"Support-boosted: {len(support)} orders  |  "
          f"Stacked (>0.001L): {len(boosted)} orders")
    print(f"{'='*75}")
    print(f"  {'Price':>8}  {'Lots':>6}  {'Dist':>8}  {'Source':>14}  "
          f"{'Zone':>6}  {'Zone center':>12}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*14}  {'-'*6}  {'-'*12}")
    for _, r in ladder.iterrows():
        dist = current_price - r["price"]
        zone_s = f"S{r['zone_rank']:.0f}" if r["zone_rank"] > 0 else "--"
        center_s = f"{r['zone_center']:,.0f}" if r["zone_rank"] > 0 else ""
        print(f"  {r['price']:>8,.1f}  {r['volume']:>6.3f}  "
              f"{-dist:>+8,.0f}  {r['source']:>14}  "
              f"{zone_s:>6}  {center_s:>12}")
    print(f"{'='*75}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Support-level detection + ladder proposal for US500.pro."
    )
    parser.add_argument("--days", type=int, default=250,
                        help="D1 bars for analysis and chart (default 250, ~1 year)")
    parser.add_argument("--zone-width", type=float, default=20.0,
                        help="Max width to merge nearby supports (default 20 pts)")
    parser.add_argument("--n-zones", type=int, default=12,
                        help="Number of top zones to place extra orders at (default 12)")
    parser.add_argument("--grid-step", type=float, default=20.0,
                        help="Background grid spacing in pts (default 20)")
    parser.add_argument("--cluster-step", type=float, default=5.0,
                        help="Spacing between rungs within a support cluster (default 5)")
    parser.add_argument("--floor", type=float, default=6300.0,
                        help="Lowest price to cover (default 6300)")
    args = parser.parse_args()

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        print(f"Fetching D1 data ({args.days} bars)...")
        d1 = fetch_d1(args.days)
        if d1.empty:
            print("ERROR: No D1 data available.")
            sys.exit(1)
        print(f"  {len(d1)} D1 bars loaded "
              f"({d1['time'].iloc[0].date()} -> {d1['time'].iloc[-1].date()})")

        tick = mt5.symbol_info_tick("US500.pro")
        current_price = tick.bid
        print(f"  Current price: {current_price:.1f}")

        pending = get_pending_orders()
        if not pending.empty:
            print(f"  {len(pending)} pending orders "
                  f"({pending['volume'].sum():.3f} lots, "
                  f"{pending['price'].min():.0f} - {pending['price'].max():.0f})")

        # Detect supports
        print(f"Detecting support zones (D1, zone width = {args.zone_width} pts)...")
        zones = detect_supports(d1, current_price, zone_width=args.zone_width)
        print(f"  {len(zones)} zones found")
        if zones.empty:
            print("No support zones detected.")
            sys.exit(0)

        # Propose ladder
        print(f"Building ladder proposal "
              f"(grid every {args.grid_step}pts, {args.n_zones} zones, "
              f"floor {args.floor:.0f})...")
        ladder = propose_ladder(
            zones, current_price,
            n_zones=args.n_zones,
            grid_step=args.grid_step,
            cluster_step=args.cluster_step,
            floor=args.floor,
        )
        print(f"  {len(ladder)} orders proposed, "
              f"{ladder['volume'].sum():.3f} lots total")

        # Console summaries
        print_zone_summary(zones, pending, current_price)
        print_ladder_summary(ladder, current_price)

        # Volume profile from same D1 data
        vp = volume_profile_d1(d1, n_bins=100)

        # Chart 1: Pure market analysis
        fig1 = plot_market(zones, d1, vp, current_price)
        _save_chart(fig1, "02_support_levels")

        # Chart 2: Current vs proposed + density
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
