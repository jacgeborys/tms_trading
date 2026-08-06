"""
01c_ladder_calc.py — Buy-limit ladder safety calculator.

Given current account state (pulled live from MT5) and a proposed ladder
of buy-limit orders, shows exactly what happens at every rung-trigger depth.

  Panel 1 — Scenario bars
    One grouped bar pair per depth level (step = rung spacing):
      · Grey  = existing positions only
      · Colour = existing + proposed ladder  (green / orange / red by ML)
      · Orange tick = margin required at that price
      · +NR label = number of rungs fired at that level

  Panel 2 — Safety frontier
    ML at worst case vs number of rungs, for six spacing options.
    Instantly answers: "for 100-pt spacing I can safely add 5 rungs."
    Optional: overlay alternative lot sizes as dashed lines.

Usage:
  python 01c_ladder_calc.py                              # 10 × 15pts × 0.001L
  python 01c_ladder_calc.py --ladder 20,10,0.001         # 20 rungs, 10pt spacing
  python 01c_ladder_calc.py --ladder 8,50,0.005          # custom ladder
  python 01c_ladder_calc.py --lots 0.001,0.002,0.003     # compare lot sizes
  python 01c_ladder_calc.py --ladder 15,12.5,0.001 --min-ml 150

Outputs:
  results/charts/01c_ladder_calc.png
"""

import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import MetaTrader5 as mt5

from pathlib import Path
from mt5_client import connect, disconnect

_CHARTS = Path(__file__).parent / "results" / "charts"

def _save_chart(fig, name: str):
    _CHARTS.mkdir(parents=True, exist_ok=True)
    path = _CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved chart → results/charts/{name}.png")

CONTRACT_SIZE  = 50.0
INSTR_LEVERAGE = 20


# ── Data ──────────────────────────────────────────────────────────────────────

def get_us500_positions() -> pd.DataFrame:
    raw = mt5.positions_get()
    if raw is None or not len(raw):
        return pd.DataFrame()
    rows = []
    for p in raw:
        if p.symbol != "US500.pro":
            continue
        rows.append({
            "volume":        p.volume,
            "direction":     1.0 if p.type == 0 else -1.0,
            "price_open":    p.price_open,
            "price_current": p.price_current,
            "profit":        p.profit,
        })
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def derive_pln_rate(positions: pd.DataFrame) -> float:
    rates = []
    for _, p in positions.iterrows():
        delta = p["price_current"] - p["price_open"]
        if abs(delta) > 2.0:
            r = p["profit"] / (p["direction"] * delta * p["volume"] * CONTRACT_SIZE)
            if 1.0 < r < 12.0:
                rates.append(r)
    return float(np.median(rates)) if rates else 4.0


def build_state(positions: pd.DataFrame, info) -> dict:
    pln_rate = derive_pln_rate(positions)
    L        = float((positions["volume"] * positions["direction"]).sum())
    upnl_now = float(positions["profit"].sum())
    price    = float(positions["price_current"].mean())
    return {
        "balance":  info.balance,
        "equity":   info.equity,
        "margin":   info.margin,
        "ml":       info.margin_level,
        "currency": info.currency,
        "pln_rate": pln_rate,
        "R":        pln_rate * CONTRACT_SIZE,   # PLN per lot per point
        "L":        L,
        "upnl_now": upnl_now,
        "price":    price,
    }


# ── Core maths ────────────────────────────────────────────────────────────────

def _at_depth(state: dict, drop: float, n_fired: int, d: float, x: float):
    """
    Equity and ML at `drop` pts below current, with n_fired rungs triggered.

    Existing book:
      upnl(P) = upnl_now + L × R × (P − P0)
      margin  = current_margin × P / P0            (OANDA marks to market)

    Each fired rung k opened at P0 − k×d:
      upnl_k   = x × R × (P − (P0 − k×d))
      margin_k = x × R × P / leverage
    """
    P0  = state["price"]
    P   = P0 - drop
    R   = state["R"]

    # Existing
    upnl_ex  = state["upnl_now"] + state["L"] * R * (P - P0)
    margin_ex = max(state["margin"] * P / P0, 1e-6)
    equity_ex = state["balance"] + upnl_ex

    # Ladder
    upnl_ld = margin_ld = 0.0
    for k in range(1, int(n_fired) + 1):
        entry      = P0 - k * d
        upnl_ld   += x * R * (P - entry)
        margin_ld  += x * R * P / INSTR_LEVERAGE

    equity_ld = equity_ex + upnl_ld
    total_mg  = max(margin_ex + margin_ld, 1e-6)

    return (
        equity_ex,
        equity_ex / margin_ex * 100,
        equity_ld,
        equity_ld / total_mg * 100,
        total_mg,
    )


def scenario_table(state: dict, n: int, d: float, x: float) -> pd.DataFrame:
    """One row per rung-trigger point (step = d), plus two extra depths."""
    rows = []
    for k in range(0, n + 3):
        drop    = k * d
        n_fired = min(k, n)
        eq_ex, ml_ex, eq_ld, ml_ld, mg_ld = _at_depth(state, drop, n_fired, d, x)
        rows.append({
            "drop":    drop,
            "price":   state["price"] - drop,
            "n_fired": n_fired,
            "eq_ex":   eq_ex,
            "ml_ex":   ml_ex,
            "eq_ld":   eq_ld,
            "ml_ld":   ml_ld,
            "mg_ld":   mg_ld,
        })
    return pd.DataFrame(rows)


def worst_ml(state: dict, n: int, d: float, x: float) -> float:
    """ML at the worst case: all n rungs fired, price at n×d below current."""
    _, _, _, ml_ld, _ = _at_depth(state, n * d, n, d, x)
    return ml_ld


# ── Console output ─────────────────────────────────────────────────────────────

def print_summary(state: dict, sc: pd.DataFrame, n: int, d: float, x: float):
    wml  = worst_ml(state, n, d, x)
    safe = "✓ SAFE" if wml >= 150 else ("⚠ CAUTION" if wml >= 100 else "✗ MARGIN CALL")
    print(f"\n{'═'*72}")
    print(f"  LADDER CALC  |  {n} rungs  ·  {d:.0f} pts spacing  ·  {x:.4f} lots/rung")
    print(f"  Account: balance {state['balance']:,.0f}  equity {state['equity']:,.0f}  "
          f"ML {state['ml']:.0f}%  ({state['currency']})")
    print(f"  Existing: {state['L']:.4f} lots  |  PLN/USD: {state['pln_rate']:.3f}")
    print(f"  Verdict at deepest rung (−{n*d:.0f} pts): ML = {wml:.0f}%  →  {safe}")
    print(f"{'═'*72}")
    print(f"  {'Drop':>7}  {'Price':>7}  {'Fired':>5}  "
          f"{'Eq existing':>12}  {'ML':>6}  "
          f"{'Eq +ladder':>11}  {'ML':>6}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*5}  "
          f"{'-'*12}  {'-'*6}  {'-'*11}  {'-'*6}")
    for _, r in sc.iterrows():
        sym_ex = "✓" if r["ml_ex"] > 200 else ("⚠" if r["ml_ex"] > 100 else "✗")
        sym_ld = "✓" if r["ml_ld"] > 200 else ("⚠" if r["ml_ld"] > 100 else "✗")
        fired_s = f"{r['n_fired']:.0f}/{n}" if r["n_fired"] <= n else f"{n}/{n}+"
        print(f"  {-r['drop']:>+7.0f}  {r['price']:>7,.0f}  {fired_s:>5}  "
              f"{r['eq_ex']:>12,.0f}  {r['ml_ex']:>5.0f}%{sym_ex}  "
              f"{r['eq_ld']:>11,.0f}  {r['ml_ld']:>5.0f}%{sym_ld}")
    print(f"{'═'*72}")


# ── Chart ──────────────────────────────────────────────────────────────────────

def _ml_color(ml: float) -> str:
    if ml <= 0:  return "#6a0000"
    if ml < 100: return "#ef5350"
    if ml < 200: return "#f08040"
    return "#26a69a"


def plot(state: dict, sc: pd.DataFrame, n: int, d: float, x: float,
         lot_sizes: list, min_ml: float) -> plt.Figure:
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    wml  = worst_ml(state, n, d, x)
    safe = "SAFE" if wml >= 150 else ("CAUTION" if wml >= 100 else "MARGIN CALL")
    fig.suptitle(
        f"Ladder Calculator — US500.pro  |  "
        f"Proposed: {n} × {d:.0f}pts × {x:.4f}L  |  "
        f"Worst-case ML {wml:.0f}%  →  {safe}  |  "
        f"Equity {state['equity']:,.0f}  ML {state['ml']:.0f}%  ({state['currency']})",
        fontsize=10,
    )

    # ── Panel 1: ML% per depth ────────────────────────────────────────────────
    n_sc = len(sc)
    bw   = 0.65
    ml_max = max(sc["ml_ex"].max(), sc["ml_ld"].max(), 250.0)

    for i, (_, r) in enumerate(sc.iterrows()):
        # Main bar: ML% with proposed ladder
        ax1.bar(i, r["ml_ld"], width=bw,
                color=_ml_color(r["ml_ld"]), alpha=0.85, zorder=3)

        # ML% label above bar
        ax1.text(i, r["ml_ld"] + ml_max * 0.012,
                 f"{r['ml_ld']:.0f}%",
                 ha="center", va="bottom", fontsize=7, color="#cccccc")

        # Rungs fired label below x-axis
        if 0 < r["n_fired"] <= n:
            ax1.text(i, -ml_max * 0.04,
                     f"+{r['n_fired']:.0f}R",
                     ha="center", va="top", fontsize=6.5, color="#aaaaaa")

    # Grey step line: existing-only ML for reference
    xs = list(range(n_sc))
    ax1.step(xs, sc["ml_ex"].tolist(), where="mid",
             color="#888888", lw=1.2, ls="--", zorder=4,
             label="Existing only (no ladder)")

    # Threshold lines
    ax1.axhline(100, color="#ef5350", lw=1.2, ls="--", zorder=5,
                label="Margin call (100%)")
    ax1.axhline(150, color="#f08040", lw=0.9, ls=":",  zorder=5,
                label="Floor (150%)")
    ax1.axhline(200, color="#26a69a", lw=0.7, ls=":", alpha=0.5, zorder=5,
                label="Safe (200%)")

    ax1.set_xlim(-0.6, n_sc - 0.4)
    ax1.set_ylim(bottom=0, top=ml_max * 1.10)
    ax1.set_xticks(range(n_sc))
    ax1.set_xticklabels(
        [f"{-r['drop']:+.0f}pts\n{r['price']:,.0f}" for _, r in sc.iterrows()],
        fontsize=7.5,
    )
    ax1.set_ylabel("Margin level (%)")
    ax1.set_title(
        "Margin level at each depth  (with proposed ladder)\n"
        "colour = ML with ladder  ·  grey dashed = existing only  ·  +NR = rungs fired"
    )
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(True, alpha=0.12, axis="y")

    # ── Panel 2: Safety frontier ───────────────────────────────────────────────
    spacings  = [10, 12.5, 15, 17.5, 20, 25]
    sp_colors = ["#ef5350", "#f08040", "#ffd700", "#80e080", "#40c0f0", "#c040f0"]
    n_range   = list(range(1, 31))

    # Solid: six spacings at proposed lot size
    for sp, col in zip(spacings, sp_colors):
        mls = [worst_ml(state, nn, sp, x) for nn in n_range]
        ax2.plot(n_range, mls, color=col, lw=1.5, marker="o", markersize=3,
                 label=f"{sp} pts", zorder=4)

    # Dashed: alternative lot sizes at proposed spacing d
    dash_palette = ["#ffffff", "#88aaff", "#ffaaff", "#aaffaa"]
    for lot_x, dcol in zip([l for l in lot_sizes if l != x], dash_palette):
        mls = [worst_ml(state, nn, d, lot_x) for nn in n_range]
        ax2.plot(n_range, mls, color=dcol, lw=1.0, ls="--", marker="s", markersize=2,
                 label=f"{d:.0f}pts × {lot_x:.4f}L", zorder=3)

    # Reference lines
    ax2.axhline(100, color="#ef5350", lw=1.0, ls="--", label="Margin call (100%)")
    ax2.axhline(min_ml, color="#f08040", lw=0.8, ls=":",
                label=f"Floor ({min_ml:.0f}%)")
    ax2.axhline(200, color="#26a69a", lw=0.6, ls=":", alpha=0.5, label="Safe (200%)")

    # Mark proposed combination
    ax2.scatter([n], [wml], color="#ffffff", s=100, zorder=7,
                label=f"Proposed ({n} × {d:.0f}pts)")

    ax2.set_xlim(0.5, 30.5)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Number of rungs (n)")
    ax2.set_ylabel("Margin level at worst case (%)")
    ax2.set_title(
        f"Safety frontier — ML when all rungs have fired\n"
        f"(solid lines = {x:.4f} lots/rung across spacings)"
    )
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.grid(True, alpha=0.13)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    plt.tight_layout()
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Buy-limit ladder safety calculator for US500.pro."
    )
    parser.add_argument(
        "--ladder", default="10,15,0.001",
        help="n,d,x — rungs, spacing(pts), lots/rung  e.g. 20,10,0.001",
    )
    parser.add_argument(
        "--lots", default=None,
        help="Extra lot sizes to compare in panel 2 (comma-sep, e.g. 0.003,0.01)",
    )
    parser.add_argument(
        "--min-ml", type=float, default=150.0,
        help="Target minimum ML%% shown as floor line in panel 2 (default 150)",
    )
    args = parser.parse_args()

    try:
        parts = args.ladder.split(",")
        n, d, x = int(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        print("ERROR: --ladder must be  n,d,x  e.g.  --ladder 6,100,0.005")
        sys.exit(1)

    lot_sizes = [x]
    if args.lots:
        for v in args.lots.split(","):
            lv = float(v.strip())
            if lv not in lot_sizes:
                lot_sizes.append(lv)

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        info = mt5.account_info()
        if info is None:
            print("ERROR: Could not fetch account info.")
            sys.exit(1)

        positions = get_us500_positions()
        if positions.empty:
            print("No US500.pro positions found.")
            sys.exit(0)

        state = build_state(positions, info)
        print(f"  PLN/USD inferred: {state['pln_rate']:.4f}")
        print(f"  {len(positions)} positions  {state['L']:.4f} lots  "
              f"equity {state['equity']:,.0f}  ML {state['ml']:.0f}%")

        sc = scenario_table(state, n, d, x)
        print_summary(state, sc, n, d, x)

        fig = plot(state, sc, n, d, x, lot_sizes, args.min_ml)
        _save_chart(fig, "01c_ladder_calc")
        plt.show()

    finally:
        disconnect()


if __name__ == "__main__":
    main()
