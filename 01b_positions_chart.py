"""
01b_positions_chart.py — Live position map on actual price chart.

Three-panel FOMO dashboard for all currently open US500.pro positions:

  Panel 1 — D1 candlestick chart (default: 2026-01-01 → now)
    · Horizontal entry line per position (colour-coded, unique per ticket)
    · Volume-weighted average entry (VWAP of entries) — gold line
    · Vertical marker + triangle at the bar where each position was opened
    · Right-side annotation: lot size, P&L in PLN, time since open

  Panel 2 — Tower view (shares Y-axis / price axis with Panel 1)
    · One horizontal brick per position at its entry price
    · Brick width = lot size  →  shows the "weight" of each level
    · Current price and VWAP lines cut across the tower
    · Underwater bricks are visually distinguishable from profitable ones

  Panel 3 — P&L sensitivity curve  ("will the tower topple?")
    · X-axis: hypothetical US500 price  (sweeps from deep drawdown to +5%)
    · Y-axis: total unrealised P&L in account currency
    · Marks: current price, break-even price, margin-warning (200%),
      and margin-call price (100% level — the hard stop-out)
    · Left of margin-call is shaded as a danger zone

Usage:
  python 01b_positions_chart.py                        # D1, from 2026-01-01
  python 01b_positions_chart.py --since 2025-09-01     # earlier start
  python 01b_positions_chart.py --tf H1                # hourly bars
  python 01b_positions_chart.py --bars 576             # last 576 M5 bars (ignores --since)

Outputs:
  results/charts/01b_positions_chart.png
"""

import sys
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import MetaTrader5 as mt5

from mt5_client import connect, disconnect
import cache

CONTRACT_SIZE = 50.0

# Colour palette — one per position, cycles if >8 open
_PALETTE = [
    "#f0c040",  # gold
    "#40c0f0",  # sky blue
    "#e040a0",  # pink
    "#80f080",  # mint
    "#f08040",  # orange
    "#c040f0",  # violet
    "#40f0c0",  # teal
    "#f04040",  # coral
]


# ── Data fetching ─────────────────────────────────────────────────────────────

def get_positions() -> pd.DataFrame:
    raw = mt5.positions_get()
    if raw is None or not len(raw):
        return pd.DataFrame()
    rows = []
    for p in raw:
        rows.append({
            "ticket":        p.ticket,
            "symbol":        p.symbol,
            "type":          p.type,              # 0=buy  1=sell
            "direction":     1.0 if p.type == 0 else -1.0,
            "volume":        p.volume,
            "price_open":    p.price_open,
            "price_current": p.price_current,
            "sl":            p.sl,
            "tp":            p.tp,
            "profit":        p.profit,            # account currency (PLN)
            "swap":          p.swap,
            "time_open":     pd.Timestamp(p.time, unit="s", tz="UTC"),
            "comment":       p.comment,
        })
    return pd.DataFrame(rows).sort_values("time_open").reset_index(drop=True)


_TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def get_price_data(symbol: str, tf: str, n_bars: int,
                   since: datetime = None) -> pd.DataFrame:
    """
    Load price data.  Two modes:
      • since=<datetime>  — date-range fetch via copy_rates_range (always live)
      • since=None        — try cache first, fall back to last n_bars from MT5
    """
    if tf not in _TF_MAP:
        return pd.DataFrame()

    if since is not None:
        now = datetime.now(tz=timezone.utc)
        rates = mt5.copy_rates_range(symbol, _TF_MAP[tf], since, now)
        if rates is None or not len(rates):
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.reset_index(drop=True)

    # n_bars mode — cache first
    df = cache.load_ohlcv(symbol, tf)
    if df is not None and len(df) >= n_bars:
        df = df.tail(n_bars).reset_index(drop=True)
        return df

    rates = mt5.copy_rates_from_pos(symbol, _TF_MAP[tf], 0, n_bars)
    if rates is None or not len(rates):
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.reset_index(drop=True)


# ── Analytics ─────────────────────────────────────────────────────────────────

def vwap_entry(positions: pd.DataFrame) -> float:
    """Volume-weighted average entry price across all open positions."""
    w = (positions["volume"] * positions["direction"]).sum()
    if abs(w) < 1e-9:
        return float(positions["price_open"].mean())
    return float(
        (positions["price_open"] * positions["volume"] * positions["direction"]).sum() / w
    )


def derive_pln_rate(positions: pd.DataFrame, info) -> float:
    """
    Infer PLN-per-USD rate from the broker's reported margin.
    OANDA computes margin on current price:
        margin = pln_rate × cs × sum(vol × P_current) / leverage
    → pln_rate = margin × leverage / (cs × sum(vol × P_current))
    Falls back to 1.0 if margin data is unavailable.
    """
    leverage = info.leverage if info.leverage > 0 else 20
    if info.margin <= 0:
        return 1.0
    denom = CONTRACT_SIZE * (positions["volume"] * positions["price_current"]).sum()
    if denom <= 0:
        return 1.0
    return info.margin * leverage / denom


def sensitivity_curve(positions: pd.DataFrame, info, pln_rate: float,
                      n_points: int = 600):
    """
    Sweep a price range and compute equity + margin_level at every point.

    upnl(P)   = pln_rate × cs × Σ [ dir_i × (P − entry_i) × vol_i ]
    margin(P) = pln_rate × cs × Σ(vol_i) × P / leverage   (OANDA marks to market)
    equity(P) = balance + upnl(P)
    """
    entries  = positions["price_open"].values
    vols     = positions["volume"].values
    dirs     = positions["direction"].values
    balance  = info.balance
    leverage = info.leverage if info.leverage > 0 else 20

    current_price = float(positions["price_current"].mean())
    lo = min(entries.min(), current_price) * 0.90
    hi = current_price * 1.06
    prices = np.linspace(lo, hi, n_points)

    net_vol_dir = (dirs * vols).sum()
    net_cost    = (dirs * vols * entries).sum()
    upnl        = pln_rate * CONTRACT_SIZE * (net_vol_dir * prices - net_cost)

    equity = balance + upnl

    total_vol = vols.sum()
    margin    = pln_rate * CONTRACT_SIZE * total_vol * prices / leverage
    margin    = np.maximum(margin, 1e-6)  # avoid division by zero

    ml = np.clip(equity / margin * 100.0, 0, 5000)

    return prices, upnl, equity, margin, ml


def _find_crossing(xs: np.ndarray, ys: np.ndarray, level: float):
    """Return the x value where ys crosses `level` (linear interp). None if no crossing."""
    diff = ys - level
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if not len(sign_changes):
        return None
    i = sign_changes[0]
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    # linear interpolation
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(positions: pd.DataFrame, info, ve: float,
                  break_even: float, mc_price, warn_price):
    now = pd.Timestamp.now(tz="UTC")
    cur = float(positions["price_current"].iloc[0])
    print(f"\n{'═'*68}")
    print(f"  POSITION MAP — US500.pro  |  "
          f"Balance {info.balance:,.0f}  Equity {info.equity:,.0f}  "
          f"ML {info.margin_level:.0f}%  ({info.currency})")
    print(f"{'═'*68}")
    print(f"  {'Ticket':>10}  {'Dir':4}  {'Lots':>5}  "
          f"{'Entry':>8}  {'Current':>8}  {'P&L':>10}  {'Age':>10}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*5}  "
          f"{'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for _, p in positions.iterrows():
        direction = "BUY " if p["type"] == 0 else "SELL"
        age_h = (now - p["time_open"]).total_seconds() / 3600
        age_s = f"{age_h:.0f}h ago" if age_h < 48 else f"{age_h/24:.1f}d ago"
        sign  = "+" if p["profit"] >= 0 else ""
        print(f"  {p['ticket']:>10}  {direction}  {p['volume']:>5.3f}  "
              f"{p['price_open']:>8.1f}  {cur:>8.1f}  "
              f"{sign}{p['profit']:>9.2f}  {age_s:>10}")
    print(f"{'─'*68}")
    total_pl = positions["profit"].sum()
    sp = "+" if total_pl >= 0 else ""
    print(f"  {'TOTAL':>10}  {'':4}  {positions['volume'].sum():>5.3f}  "
          f"  VWAP {ve:>8.1f}  {sp}{total_pl:>9.2f}")
    print(f"\n  Break-even   : {break_even:>8.1f}  "
          f"(dist from current: {cur - break_even:+.1f} pts)")
    if warn_price:
        print(f"  ML 200% warn : {warn_price:>8.1f}  "
              f"(dist: {cur - warn_price:+.1f} pts  /  "
              f"{(warn_price - cur) / cur * 100:.1f}%)")
    if mc_price:
        print(f"  Margin call  : {mc_price:>8.1f}  "
              f"(dist: {cur - mc_price:+.1f} pts  /  "
              f"{(mc_price - cur) / cur * 100:.1f}%)")
    print(f"{'═'*68}")


# ── Candlestick drawing ────────────────────────────────────────────────────────

def draw_candles(ax, df: pd.DataFrame):
    w = 0.65
    for i, row in df.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.bar(i, abs(c - o), bottom=min(o, c), width=w,
               color=color, alpha=0.9, linewidth=0)
        ax.plot([i, i], [l, h], color=color, lw=0.7, alpha=0.85, zorder=2)


def time_ticks(df: pd.DataFrame, tf: str, max_ticks: int = 10):
    step   = max(1, len(df) // max_ticks)
    idxs   = list(range(0, len(df), step))
    fmt    = "%d %b" if tf in ("D1", "H4", "H1") else "%d %b\n%H:%M"
    labels = [df["time"].iloc[i].strftime(fmt) for i in idxs]
    return idxs, labels


# ── Main chart ────────────────────────────────────────────────────────────────

def plot(positions: pd.DataFrame, price_df: pd.DataFrame,
         info, tf: str) -> plt.Figure:
    plt.style.use("dark_background")

    leverage     = info.leverage if info.leverage > 0 else 20
    current_price = float(positions["price_current"].iloc[0])
    total_pl      = positions["profit"].sum()
    total_vol     = positions["volume"].sum()
    n_pos         = len(positions)
    ve            = vwap_entry(positions)
    rate          = derive_pln_rate(positions, info)
    now           = pd.Timestamp.now(tz="UTC")

    prices_s, upnl_s, equity_s, margin_s, ml_s = sensitivity_curve(
        positions, info, rate
    )

    # Key price levels
    break_even = _find_crossing(prices_s, upnl_s, 0.0)
    if break_even is None:
        break_even = ve  # fallback: VWAP of entries
    mc_price   = _find_crossing(prices_s, ml_s, 100.0)
    warn_price = _find_crossing(prices_s, ml_s, 200.0)

    print_summary(positions, info, ve, break_even, mc_price, warn_price)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 13))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[4, 1],
        height_ratios=[3, 2],
        hspace=0.38, wspace=0.04,
    )
    ax_price = fig.add_subplot(gs[0, 0])
    ax_tower = fig.add_subplot(gs[0, 1], sharey=ax_price)
    ax_sens  = fig.add_subplot(gs[1, :])

    sp = "+" if total_pl >= 0 else ""
    fig.suptitle(
        f"Position Map — US500.pro  |  {n_pos} open  {total_vol:.3f} lots  |  "
        f"P&L {sp}{total_pl:,.0f} {info.currency}  |  "
        f"Balance {info.balance:,.0f}  Equity {info.equity:,.0f}  "
        f"ML {info.margin_level:.0f}%",
        fontsize=11,
    )

    # ── Panel 1: Candlestick price chart ──────────────────────────────────────
    draw_candles(ax_price, price_df)
    n = len(price_df)

    # Normalise price_df times to tz-naive for comparison
    times_raw = price_df["time"]
    if times_raw.dt.tz is not None:
        times_naive = times_raw.dt.tz_localize(None)
    else:
        times_naive = times_raw

    for idx, (_, p) in enumerate(positions.iterrows()):
        color     = _PALETTE[idx % len(_PALETTE)]
        is_profit = p["profit"] >= 0
        ls        = "--" if is_profit else "-."

        # Entry price horizontal line
        ax_price.axhline(p["price_open"], color=color, lw=1.4, ls=ls, alpha=0.85, zorder=3)

        # Vertical marker at entry time
        t_open = p["time_open"]
        if t_open.tzinfo is not None:
            t_open = t_open.tz_localize(None)
        diff = (times_naive - t_open).abs()
        bar_idx = diff.idxmin()
        if diff[bar_idx] < pd.Timedelta(hours=4):
            ax_price.axvline(bar_idx, color=color, lw=0.8, ls=":", alpha=0.45)
            ax_price.scatter([bar_idx], [p["price_open"]],
                             color=color, s=65, zorder=6, marker="^",
                             edgecolors="#000000", linewidths=0.5)

        # Right-side annotation
        age_h   = (now - p["time_open"]).total_seconds() / 3600
        age_str = f"{age_h:.0f}h" if age_h < 48 else f"{age_h/24:.1f}d"
        sign    = "+" if p["profit"] >= 0 else ""
        label   = (f" {p['volume']:.3f}L @ {p['price_open']:.1f}"
                   f"  {sign}{p['profit']:.0f}  ({age_str})")
        ax_price.annotate(
            label,
            xy=(n - 1, p["price_open"]),
            xytext=(n + 1, p["price_open"]),
            fontsize=7.5, color=color, va="center",
            annotation_clip=False,
        )

    # VWAP of entries — gold anchor line
    ax_price.axhline(ve, color="#ffd700", lw=2.2, ls="-", alpha=0.9, zorder=4,
                     label=f"Avg entry  {ve:.1f}")

    # Current price
    ax_price.axhline(current_price, color="#ffffff", lw=1.0, ls=":",
                     alpha=0.75, zorder=4, label=f"Current  {current_price:.1f}")

    # Soft shading between avg entry and current
    lo_fill = min(ve, current_price)
    hi_fill = max(ve, current_price)
    fill_color = "#26a69a" if current_price >= ve else "#ef5350"
    ax_price.fill_between([-1, n + 1], [lo_fill] * 2, [hi_fill] * 2,
                           color=fill_color, alpha=0.06)

    # SL / TP lines if set
    for _, p in positions.iterrows():
        if p["sl"] > 0:
            ax_price.axhline(p["sl"], color="#ef5350", lw=0.7, ls=":", alpha=0.5)
        if p["tp"] > 0:
            ax_price.axhline(p["tp"], color="#26a69a", lw=0.7, ls=":", alpha=0.5)

    xt, xl = time_ticks(price_df, tf)
    ax_price.set_xticks(xt)
    ax_price.set_xticklabels(xl, fontsize=7)
    ax_price.set_xlim(-1, n + 1)
    ax_price.set_ylabel("US500.pro")
    first_bar = price_df["time"].iloc[0].strftime("%d %b %Y")
    ax_price.set_title(f"Price chart — {tf}  ({n} bars,  {first_bar} → now)")
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_price.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax_price.grid(True, alpha=0.12)

    # ── Panel 2: Tower view ───────────────────────────────────────────────────
    # Each brick: y centred on entry_price, height = visual unit, width = lot size
    price_range = price_df["high"].max() - price_df["low"].min()
    brick_h     = max(2.5, price_range * 0.012)   # at least 2.5 pts tall

    for idx, (_, p) in enumerate(positions.iterrows()):
        color     = _PALETTE[idx % len(_PALETTE)]
        is_profit = p["profit"] >= 0
        edge      = "#ffffff"
        alpha     = 0.82 if is_profit else 0.55

        rect = mpatches.Rectangle(
            (0, p["price_open"] - brick_h / 2),
            p["volume"], brick_h,
            facecolor=color, edgecolor=edge,
            linewidth=0.6, alpha=alpha, zorder=3,
        )
        ax_tower.add_patch(rect)

        # Lot label inside brick
        ax_tower.text(
            p["volume"] / 2, p["price_open"],
            f"{p['volume']:.2f}",
            ha="center", va="center",
            fontsize=6.5, color="#000000", fontweight="bold",
        )

    ax_tower.axhline(current_price, color="#ffffff", lw=1.4, ls="--",
                     alpha=0.85, zorder=5)
    ax_tower.axhline(ve, color="#ffd700", lw=1.8, ls="-", alpha=0.9, zorder=5)

    ax_tower.text(0.02, current_price, f" {current_price:.0f}",
                  transform=ax_tower.get_yaxis_transform(),
                  fontsize=7, color="#ffffff", va="center")
    ax_tower.text(0.02, ve, f" {ve:.0f}",
                  transform=ax_tower.get_yaxis_transform(),
                  fontsize=7, color="#ffd700", va="center")

    max_lot = positions["volume"].max()
    ax_tower.set_xlim(0, max_lot * 1.8)
    ax_tower.set_xlabel("Lots", fontsize=8)
    ax_tower.set_title("Tower\n(width = lots)", fontsize=8.5)
    ax_tower.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax_tower.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    plt.setp(ax_tower.get_yticklabels(), visible=False)
    ax_tower.grid(True, alpha=0.12, axis="x")

    # ── Panel 3: P&L sensitivity curve ────────────────────────────────────────
    ax_sens.axhline(0, color="#666666", lw=0.8, ls="-")

    ax_sens.fill_between(prices_s, 0, upnl_s,
                         where=(upnl_s >= 0), color="#26a69a", alpha=0.30,
                         label="Profit zone")
    ax_sens.fill_between(prices_s, 0, upnl_s,
                         where=(upnl_s < 0), color="#ef5350", alpha=0.30,
                         label="Loss zone")
    ax_sens.plot(prices_s, upnl_s, color="#f0f0f0", lw=1.6,
                 label="Total unrealised P&L")

    # Current price
    ax_sens.axvline(current_price, color="#ffffff", lw=1.3, ls=":",
                    label=f"Current  {current_price:,.0f}")
    ax_sens.scatter([current_price], [total_pl], color="#ffffff", s=70, zorder=7)

    # Break-even
    if prices_s[0] < break_even < prices_s[-1]:
        ax_sens.axvline(break_even, color="#26a69a", lw=1.3, ls="--",
                        label=f"Break-even  {break_even:,.1f}")

    # Margin warning (200%)
    if warn_price is not None and prices_s[0] < warn_price < prices_s[-1]:
        upnl_at_warn = float(np.interp(warn_price, prices_s, upnl_s))
        ax_sens.axvline(warn_price, color="#f08040", lw=1.1, ls=":",
                        label=f"ML 200%  {warn_price:,.0f}")
        ax_sens.scatter([warn_price], [upnl_at_warn], color="#f08040", s=70, zorder=7)

    # Margin call (100%)
    if mc_price is not None and prices_s[0] < mc_price < prices_s[-1]:
        upnl_at_mc = float(np.interp(mc_price, prices_s, upnl_s))
        ax_sens.axvline(mc_price, color="#ef5350", lw=1.5, ls="-",
                        label=f"Margin call  {mc_price:,.0f}")
        ax_sens.scatter([mc_price], [upnl_at_mc], color="#ef5350", s=80, zorder=7,
                        marker="X")
        ax_sens.axvspan(prices_s[0], mc_price, color="#ef5350", alpha=0.07)

    # Annotate current P&L on curve
    cur_sign = "+" if total_pl >= 0 else ""
    ax_sens.annotate(
        f" {cur_sign}{total_pl:,.0f} PLN",
        xy=(current_price, total_pl),
        xytext=(current_price + (prices_s[-1] - prices_s[0]) * 0.015, total_pl),
        fontsize=8, color="#f0f0f0", va="center", annotation_clip=False,
    )

    # Annotate distances to key levels
    if mc_price is not None:
        dist_mc = current_price - mc_price
        ax_sens.text(
            0.01, 0.04,
            f"Margin call {dist_mc:+.0f} pts  ({(mc_price - current_price) / current_price * 100:.1f}%)",
            transform=ax_sens.transAxes, fontsize=8, color="#ef5350",
        )
    if warn_price is not None:
        dist_w = current_price - warn_price
        ax_sens.text(
            0.01, 0.09,
            f"ML 200% warn {dist_w:+.0f} pts  ({(warn_price - current_price) / current_price * 100:.1f}%)",
            transform=ax_sens.transAxes, fontsize=8, color="#f08040",
        )

    ax_sens.set_xlabel("US500.pro price")
    ax_sens.set_ylabel(f"Unrealised P&L ({info.currency})")
    ax_sens.set_title(
        "P&L sensitivity — how far can it fall before the tower topples?"
    )
    ax_sens.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_sens.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_sens.legend(fontsize=8, ncol=6, loc="upper left")
    ax_sens.grid(True, alpha=0.14)

    plt.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live position map on actual US500.pro price chart."
    )
    parser.add_argument("--tf", default="D1",
                        help="Timeframe for price chart (M5/M15/H1/D1, default D1)")
    parser.add_argument("--since", default="2026-01-01",
                        help="Start date YYYY-MM-DD (default 2026-01-01)")
    parser.add_argument("--bars", type=int, default=0,
                        help="Fixed bar count — overrides --since when set")
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

        # Single-asset focus — US500.pro only
        positions = positions[positions["symbol"] == "US500.pro"].reset_index(drop=True)
        if positions.empty:
            print("No US500.pro positions found.")
            sys.exit(0)

        print(f"  {len(positions)} US500.pro position(s) found.")

        if args.bars > 0:
            since_dt = None
            print(f"Loading price data ({args.tf}, last {args.bars} bars)...")
        else:
            since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            print(f"Loading price data ({args.tf}, from {args.since})...")

        price_df = get_price_data("US500.pro", args.tf, args.bars, since=since_dt)
        if price_df.empty:
            print("ERROR: No price data. Is MT5 open? Try --bars 500 as fallback.")
            sys.exit(1)
        price_df = price_df.reset_index(drop=True)
        print(f"  {len(price_df)} bars loaded.")

        fig = plot(positions, price_df, info, args.tf)
        cache.save_chart(fig, "01b_positions_chart")
        plt.show()

    finally:
        disconnect()


if __name__ == "__main__":
    main()
