"""
01e_tower_chart.py — Position tower: vertical stack of entries by price.

Visualises all open US500.pro positions as a vertical tower of horizontal
bars.  Each bar sits at its entry price (y-axis), and its width is
proportional to lot size (x-axis).

A stable tower has heavy positions (large lots) at the bottom (low prices).
A top-heavy tower — large lots concentrated at high prices — is fragile
and vulnerable to drawdowns.

Key elements:
  - Horizontal bars: one per position, width = lot size, y = entry price
  - Green bar = in profit (entry < current price), red = underwater
  - Gold dashed line = VWAP (volume-weighted average entry = "center of mass")
  - White dotted line = current price
  - Right margin: cumulative volume from bottom up (weight distribution)
  - Console: top-heaviness score and volume split above/below VWAP

Usage:
  python 01e_tower_chart.py

Outputs:
  results/charts/01e_tower_chart.png
"""

import sys
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


def plot_tower(positions: pd.DataFrame, info) -> plt.Figure:
    plt.style.use("dark_background")

    current_price = float(positions["price_current"].iloc[0])
    total_vol     = positions["volume"].sum()
    total_pl      = positions["profit"].sum()
    n_pos         = len(positions)
    ve            = vwap_entry(positions)

    # Sort by entry price (low to high = bottom to top)
    pos = positions.sort_values("price_open").reset_index(drop=True)

    entries = pos["price_open"].values
    volumes = pos["volume"].values

    # Volume above and below VWAP
    vol_below = pos.loc[pos["price_open"] <= ve, "volume"].sum()
    vol_above = pos.loc[pos["price_open"] >  ve, "volume"].sum()
    pct_above = vol_above / total_vol * 100 if total_vol > 0 else 0

    # Top-heaviness score: volume-weighted average percentile position
    # 0 = all weight at the bottom, 100 = all weight at the top
    price_min = entries.min()
    price_max = entries.max()
    price_range = price_max - price_min
    if price_range > 0:
        normalised = (entries - price_min) / price_range  # 0..1
        heaviness = float(np.average(normalised, weights=volumes) * 100)
    else:
        heaviness = 50.0

    # Console summary
    print(f"\n{'='*60}")
    print(f"  TOWER ANALYSIS — US500.pro")
    print(f"{'='*60}")
    print(f"  Positions     : {n_pos}")
    print(f"  Total volume  : {total_vol:.3f} lots")
    print(f"  VWAP (CoM)    : {ve:,.1f}")
    print(f"  Current price : {current_price:,.1f}")
    print(f"  Price range   : {price_min:,.1f} — {price_max:,.1f}  ({price_range:.1f} pts)")
    print(f"  Vol below VWAP: {vol_below:.3f}  ({100 - pct_above:.0f}%)")
    print(f"  Vol above VWAP: {vol_above:.3f}  ({pct_above:.0f}%)")
    print(f"  Top-heaviness : {heaviness:.0f}/100  ", end="")
    if heaviness <= 35:
        print("(bottom-heavy — stable)")
    elif heaviness <= 55:
        print("(balanced)")
    elif heaviness <= 70:
        print("(slightly top-heavy)")
    else:
        print("(top-heavy — fragile!)")
    print(f"{'='*60}")

    # ── Layout: tower (left) + cumulative volume (right) ──────────────────
    fig = plt.figure(figsize=(14, max(10, n_pos * 0.35 + 4)))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.05)
    ax_tower = fig.add_subplot(gs[0])
    ax_cum   = fig.add_subplot(gs[1], sharey=ax_tower)

    sp = "+" if total_pl >= 0 else ""
    fig.suptitle(
        f"Position Tower — US500.pro  |  {n_pos} positions  {total_vol:.3f} lots  |  "
        f"P&L {sp}{total_pl:,.0f} {info.currency}  |  "
        f"Top-heaviness {heaviness:.0f}/100",
        fontsize=11,
    )

    # ── Tower bars ────────────────────────────────────────────────────────
    # Bar height: proportional spacing so bars don't overlap
    # Use entry price as y-position, fixed bar height
    if n_pos > 1:
        min_gap = np.diff(entries).min()
        bar_h = max(min_gap * 0.7, price_range * 0.012) if min_gap > 0 else price_range * 0.02
    else:
        bar_h = 5.0

    colors = []
    for _, p in pos.iterrows():
        if p["price_open"] <= current_price:
            colors.append("#26a69a")  # in profit — green
        else:
            colors.append("#ef5350")  # underwater — red

    ax_tower.barh(
        entries, volumes, height=bar_h,
        color=colors, alpha=0.85, edgecolor="#ffffff", linewidth=0.5,
        zorder=3,
    )

    # Volume labels on each bar
    for i, (entry, vol) in enumerate(zip(entries, volumes)):
        ax_tower.text(
            vol + total_vol * 0.01, entry,
            f"{vol:.3f}",
            va="center", ha="left", fontsize=7.5, color="#cccccc",
        )

    # Price labels on y-axis side
    for entry in entries:
        ax_tower.text(
            -total_vol * 0.01, entry,
            f"{entry:,.1f}",
            va="center", ha="right", fontsize=7, color="#aaaaaa",
        )

    # VWAP = center of mass
    ax_tower.axhline(ve, color="#ffd700", lw=2.0, ls="--", alpha=0.9, zorder=5,
                     label=f"VWAP (CoM)  {ve:,.1f}")

    # Current price
    ax_tower.axhline(current_price, color="#ffffff", lw=1.5, ls=":", alpha=0.8, zorder=5,
                     label=f"Current  {current_price:,.1f}")

    # Shade the underwater zone (above current price)
    y_lo = min(entries.min() - price_range * 0.1, current_price - 20)
    y_hi = max(entries.max() + price_range * 0.1, current_price + 20)
    ax_tower.fill_between(
        [0, volumes.max() * 1.4],
        [current_price, current_price],
        [y_hi, y_hi],
        color="#ef5350", alpha=0.04, zorder=1,
    )
    ax_tower.fill_between(
        [0, volumes.max() * 1.4],
        [y_lo, y_lo],
        [current_price, current_price],
        color="#26a69a", alpha=0.04, zorder=1,
    )

    ax_tower.set_xlim(0, volumes.max() * 1.5)
    ax_tower.set_ylim(y_lo, y_hi)
    ax_tower.set_xlabel("Lot size")
    ax_tower.set_ylabel("Entry price")
    ax_tower.set_title("Each bar = one position at its entry price")
    ax_tower.legend(fontsize=9, loc="lower right")
    ax_tower.grid(True, alpha=0.12, axis="y")
    ax_tower.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    # ── Cumulative volume panel (right) ───────────────────────────────────
    # Shows how volume accumulates from lowest entry to highest
    cum_vol = np.cumsum(volumes)

    # Step plot: cumulative volume at each entry price
    ax_cum.fill_betweenx(entries, 0, cum_vol, step="mid",
                         color="#5dade2", alpha=0.3, zorder=2)
    ax_cum.step(cum_vol, entries, where="mid",
                color="#5dade2", lw=1.5, zorder=3)

    # Mark VWAP and current price on this panel too
    ax_cum.axhline(ve, color="#ffd700", lw=1.5, ls="--", alpha=0.7)
    ax_cum.axhline(current_price, color="#ffffff", lw=1.0, ls=":", alpha=0.6)

    # Label total at top
    ax_cum.text(
        cum_vol[-1], entries[-1] + bar_h,
        f"{cum_vol[-1]:.3f}\ntotal",
        ha="center", va="bottom", fontsize=8, color="#5dade2",
    )

    # Volume at VWAP level
    vol_at_vwap = float(np.interp(ve, entries, cum_vol))
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

        fig = plot_tower(positions, info)
        _save_chart(fig, "01e_tower_chart")
        plt.show()

    finally:
        disconnect()


if __name__ == "__main__":
    main()
