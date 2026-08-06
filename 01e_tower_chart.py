"""
01e_tower_chart.py — Position tower: half-violin volume distribution by price.

Aggregates all open US500.pro positions into price bins and draws a
half-violin chart showing volume concentration at each price level.

A stable tower has heavy volume (wide bars) at low prices.
A top-heavy tower — wide bars at high prices — is fragile.

Key elements:
  - Left panel: half-violin of volume per price bin (bin width configurable)
  - Green bins = in profit (below current price), red = underwater
  - Gold dashed line = VWAP (volume-weighted average entry = "center of mass")
  - White dotted line = current price
  - Right panel: cumulative volume from bottom up (weight distribution)
  - Console: top-heaviness score, volume split, per-bin breakdown

Usage:
  python 01e_tower_chart.py                  # 20-pt bins (default)
  python 01e_tower_chart.py --bin 10         # 10-pt bins
  python 01e_tower_chart.py --bin 50         # 50-pt bins (wider view)

Outputs:
  results/charts/01e_tower_chart.png
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

_ROOT   = Path(__file__).parent
_CHARTS = _ROOT / "results" / "charts"

CONTRACT_SIZE = 50.0


def _save_chart(fig, name: str):
    _CHARTS.mkdir(parents=True, exist_ok=True)
    path = _CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved chart -> results/charts/{name}.png")


def get_positions() -> pd.DataFrame:
    raw = mt5.positions_get()
    if raw is None or not len(raw):
        return pd.DataFrame()
    rows = []
    for p in raw:
        rows.append({
            "ticket":        p.ticket,
            "symbol":        p.symbol,
            "type":          p.type,
            "direction":     1.0 if p.type == 0 else -1.0,
            "volume":        p.volume,
            "price_open":    p.price_open,
            "price_current": p.price_current,
            "profit":        p.profit,
            "swap":          p.swap,
            "time_open":     pd.Timestamp(p.time, unit="s", tz="UTC"),
        })
    return pd.DataFrame(rows).sort_values("price_open").reset_index(drop=True)


def vwap_entry(positions: pd.DataFrame) -> float:
    w = (positions["volume"] * positions["direction"]).sum()
    if abs(w) < 1e-9:
        return float(positions["price_open"].mean())
    return float(
        (positions["price_open"] * positions["volume"] * positions["direction"]).sum() / w
    )


def bin_positions(positions: pd.DataFrame, bin_size: float, current_price: float):
    """Aggregate positions into price bins. Returns DataFrame with bin_center, volume, count, color."""
    entries = positions["price_open"].values
    # Align bins to round numbers (e.g. 5500, 5520, 5540 for bin_size=20)
    bin_lo = np.floor(entries.min() / bin_size) * bin_size
    bin_hi = np.ceil(entries.max() / bin_size) * bin_size + bin_size
    edges = np.arange(bin_lo, bin_hi + bin_size / 2, bin_size)

    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        center = (lo + hi) / 2
        mask = (entries >= lo) & (entries < hi)
        vol = float(positions.loc[mask, "volume"].sum())
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        color = "#26a69a" if center <= current_price else "#ef5350"
        rows.append({
            "bin_lo": lo, "bin_hi": hi, "bin_center": center,
            "volume": vol, "count": cnt, "color": color,
        })
    return pd.DataFrame(rows)


def plot_tower(positions: pd.DataFrame, info, bin_size: float) -> plt.Figure:
    plt.style.use("dark_background")

    current_price = float(positions["price_current"].iloc[0])
    total_vol     = positions["volume"].sum()
    total_pl      = positions["profit"].sum()
    total_swap    = positions["swap"].sum()
    total_net     = total_pl + total_swap
    n_pos         = len(positions)
    ve            = vwap_entry(positions)

    entries = positions["price_open"].values
    volumes = positions["volume"].values

    # Volume above and below VWAP
    vol_below = positions.loc[positions["price_open"] <= ve, "volume"].sum()
    vol_above = positions.loc[positions["price_open"] >  ve, "volume"].sum()
    pct_above = vol_above / total_vol * 100 if total_vol > 0 else 0

    # Top-heaviness score
    price_min = entries.min()
    price_max = entries.max()
    price_range = price_max - price_min
    if price_range > 0:
        normalised = (entries - price_min) / price_range
        heaviness = float(np.average(normalised, weights=volumes) * 100)
    else:
        heaviness = 50.0

    # Bin the positions
    bins = bin_positions(positions, bin_size, current_price)

    # Console summary
    print(f"\n{'='*65}")
    print(f"  TOWER ANALYSIS — US500.pro  (bin = {bin_size:.0f} pts)")
    print(f"{'='*65}")
    print(f"  Positions     : {n_pos}")
    print(f"  Total volume  : {total_vol:.3f} lots")
    print(f"  VWAP (CoM)    : {ve:,.1f}")
    print(f"  Current price : {current_price:,.1f}")
    print(f"  Price range   : {price_min:,.1f} -- {price_max:,.1f}  ({price_range:.1f} pts)")
    print(f"  Vol below VWAP: {vol_below:.3f}  ({100 - pct_above:.0f}%)")
    print(f"  Vol above VWAP: {vol_above:.3f}  ({pct_above:.0f}%)")
    print(f"  Top-heaviness : {heaviness:.0f}/100  ", end="")
    if heaviness <= 35:
        print("(bottom-heavy -- stable)")
    elif heaviness <= 55:
        print("(balanced)")
    elif heaviness <= 70:
        print("(slightly top-heavy)")
    else:
        print("(top-heavy -- fragile!)")
    print(f"{'='*65}")

    # Per-bin breakdown
    print(f"\n  {'Bin range':>20}  {'Lots':>7}  {'Count':>5}  {'Bar':>20}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*5}  {'-'*20}")
    max_vol = bins["volume"].max() if not bins.empty else 1
    for _, b in bins.iterrows():
        bar_len = int(b["volume"] / max_vol * 20)
        bar_str = "#" * bar_len
        status = "  <-- underwater" if b["bin_center"] > current_price else ""
        print(f"  {b['bin_lo']:>8,.0f} - {b['bin_hi']:>8,.0f}  "
              f"{b['volume']:>7.3f}  {b['count']:>5}  {bar_str:<20}{status}")
    print()

    # ── Layout: tower (left) + cumulative volume (right) ──────────────────
    n_bins = len(bins)
    fig_h = max(10, n_bins * 0.6 + 4)
    fig = plt.figure(figsize=(14, fig_h))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.05)
    ax_tower = fig.add_subplot(gs[0])
    ax_cum   = fig.add_subplot(gs[1], sharey=ax_tower)

    sp = "+" if total_net >= 0 else ""
    fig.suptitle(
        f"Position Tower — US500.pro  |  {n_pos} positions  {total_vol:.3f} lots  |  "
        f"Net P&L {sp}{total_net:,.0f} {info.currency}  (swap {total_swap:+,.0f})  |  "
        f"Top-heaviness {heaviness:.0f}/100",
        fontsize=11,
    )

    # ── Half-violin tower (binned) ────────────────────────────────────────
    bin_centers = bins["bin_center"].values
    bin_vols    = bins["volume"].values
    bin_colors  = bins["color"].tolist()

    ax_tower.barh(
        bin_centers, bin_vols, height=bin_size * 0.85,
        color=bin_colors, alpha=0.85, edgecolor="#ffffff", linewidth=0.5,
        zorder=3,
    )

    # Volume + count labels on each bar
    for _, b in bins.iterrows():
        ax_tower.text(
            b["volume"] + max_vol * 0.02, b["bin_center"],
            f"{b['volume']:.3f}  ({b['count']})",
            va="center", ha="left", fontsize=7.5, color="#cccccc",
        )

    # VWAP = center of mass
    ax_tower.axhline(ve, color="#ffd700", lw=2.0, ls="--", alpha=0.9, zorder=5,
                     label=f"VWAP (CoM)  {ve:,.1f}")

    # Current price
    ax_tower.axhline(current_price, color="#ffffff", lw=1.5, ls=":", alpha=0.8, zorder=5,
                     label=f"Current  {current_price:,.1f}")

    # Shade the underwater zone
    y_lo = min(bin_centers.min() - bin_size * 2, current_price - bin_size * 2)
    y_hi = max(bin_centers.max() + bin_size * 2, current_price + bin_size * 2)
    ax_tower.fill_between(
        [0, max_vol * 1.5],
        [current_price, current_price], [y_hi, y_hi],
        color="#ef5350", alpha=0.04, zorder=1,
    )
    ax_tower.fill_between(
        [0, max_vol * 1.5],
        [y_lo, y_lo], [current_price, current_price],
        color="#26a69a", alpha=0.04, zorder=1,
    )

    ax_tower.set_xlim(0, max_vol * 1.5)
    ax_tower.set_ylim(y_lo, y_hi)
    ax_tower.set_xlabel("Total lots in bin")
    ax_tower.set_ylabel("Price level")
    ax_tower.set_title(f"Volume per {bin_size:.0f}-pt price bin  (half-violin)")
    ax_tower.legend(fontsize=9, loc="lower right")
    ax_tower.grid(True, alpha=0.12, axis="y")
    ax_tower.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    # ── Cumulative volume panel (right) ───────────────────────────────────
    # Sort positions by entry price for cumulative
    pos_sorted = positions.sort_values("price_open").reset_index(drop=True)
    sorted_entries = pos_sorted["price_open"].values
    sorted_vols    = pos_sorted["volume"].values
    cum_vol = np.cumsum(sorted_vols)

    ax_cum.fill_betweenx(sorted_entries, 0, cum_vol, step="mid",
                         color="#5dade2", alpha=0.3, zorder=2)
    ax_cum.step(cum_vol, sorted_entries, where="mid",
                color="#5dade2", lw=1.5, zorder=3)

    # Mark VWAP and current price
    ax_cum.axhline(ve, color="#ffd700", lw=1.5, ls="--", alpha=0.7)
    ax_cum.axhline(current_price, color="#ffffff", lw=1.0, ls=":", alpha=0.6)

    # Label total at top
    ax_cum.text(
        cum_vol[-1], sorted_entries[-1] + bin_size * 0.5,
        f"{cum_vol[-1]:.3f}\ntotal",
        ha="center", va="bottom", fontsize=8, color="#5dade2",
    )

    # Volume at VWAP level
    vol_at_vwap = float(np.interp(ve, sorted_entries, cum_vol))
    pct_below_vwap = vol_at_vwap / total_vol * 100
    ax_cum.plot(vol_at_vwap, ve, "o", color="#ffd700", ms=6, zorder=5)
    ax_cum.text(
        vol_at_vwap + total_vol * 0.03, ve,
        f"{pct_below_vwap:.0f}% below",
        va="center", ha="left", fontsize=8, color="#ffd700",
    )

    ax_cum.set_xlabel("Cumulative lots")
    ax_cum.set_title("Volume buildup")
    ax_cum.tick_params(labelleft=False)
    ax_cum.grid(True, alpha=0.12, axis="y")
    ax_cum.set_xlim(0, total_vol * 1.2)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Position tower — half-violin volume distribution for US500.pro."
    )
    parser.add_argument("--bin", type=float, default=20.0,
                        help="Price bin width in points (default 20)")
    args = parser.parse_args()

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        info = mt5.account_info()
        if info is None:
            print("ERROR: Could not fetch account info.")
            sys.exit(1)

        print("Fetching open positions...")
        positions = get_positions()

        if positions.empty:
            print("No open positions found.")
            sys.exit(0)

        positions = positions[positions["symbol"] == "US500.pro"].reset_index(drop=True)
        if positions.empty:
            print("No US500.pro positions found.")
            sys.exit(0)

        print(f"  {len(positions)} US500.pro position(s) found.")

        fig = plot_tower(positions, info, args.bin)
        _save_chart(fig, "01e_tower_chart")
        plt.show()

    finally:
        disconnect()


if __name__ == "__main__":
    main()
