"""
01b_positions_chart.py — Live position map on actual price chart.

Three-panel FOMO dashboard for all currently open US500.pro positions:

  Panel 1 — D1 candlestick chart (default: 2026-01-01 → now)
    · Circles at (entry_time, entry_price), size ∝ lot size, one accent colour
    · Volume-weighted average entry (VWAP of entries) — gold line
    · White dotted line = current price
    · Gap text: how many pts / % above average entry

  Panel 2 — P&L sensitivity curve  ("will the tower topple?")
    · X-axis: hypothetical US500 price (trimmed to ±15% of current)
    · Y-axis: total unrealised P&L in account currency
    · Key levels annotated directly on their vertical lines:
      break-even, margin-warning (200%), margin-call (100%)
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
import MetaTrader5 as mt5

from mt5_client import connect, disconnect
import cache

CONTRACT_SIZE = 50.0
ENTRY_COLOR   = "#5bc8f5"   # single accent colour for all entry markers


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

    current_price = float(positions["price_current"].iloc[0])
    total_pl      = positions["profit"].sum()
    total_vol     = positions["volume"].sum()
    n_pos         = len(positions)
    ve            = vwap_entry(positions)
    rate          = derive_pln_rate(positions, info)

    prices_s, upnl_s, equity_s, margin_s, ml_s = sensitivity_curve(
        positions, info, rate
    )
    break_even = _find_crossing(prices_s, upnl_s, 0.0) or ve
    mc_price   = _find_crossing(prices_s, ml_s, 100.0)
    warn_price = _find_crossing(prices_s, ml_s, 200.0)

    print_summary(positions, info, ve, break_even, mc_price, warn_price)

    # ── Layout: 2 panels ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[3, 2], hspace=0.40)
    ax_price = fig.add_subplot(gs[0])
    ax_sens  = fig.add_subplot(gs[1])

    sp = "+" if total_pl >= 0 else ""
    fig.suptitle(
        f"Position Map — US500.pro  |  {n_pos} entries  {total_vol:.3f} lots  |  "
        f"P&L {sp}{total_pl:,.0f} {info.currency}  |  "
        f"Balance {info.balance:,.0f}  Equity {info.equity:,.0f}  "
        f"ML {info.margin_level:.0f}%",
        fontsize=11,
    )

    # ── Panel 1: Price chart with entry circles ───────────────────────────────
    draw_candles(ax_price, price_df)
    n = len(price_df)

    times_raw   = price_df["time"]
    times_naive = (times_raw.dt.tz_localize(None)
                   if times_raw.dt.tz is not None else times_raw)

    # Circle size proportional to lot size (visible even when all equal)
    min_vol  = positions["volume"].min()
    max_vol  = positions["volume"].max()
    vol_span = max_vol - min_vol if max_vol > min_vol else 1.0

    # More lenient time tolerance for wide timeframes (D1 has weekend gaps)
    time_tol = pd.Timedelta(days=5 if tf in ("D1", "H4") else 1)

    xs, ys, ss = [], [], []
    for _, p in positions.iterrows():
        t = p["time_open"]
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        diff = (times_naive - t).abs()
        bidx = diff.idxmin()
        if diff[bidx] > time_tol:
            continue
        xs.append(bidx)
        ys.append(p["price_open"])
        ss.append(60 + 280 * (p["volume"] - min_vol) / vol_span)

    ax_price.scatter(xs, ys, s=ss,
                     color=ENTRY_COLOR, alpha=0.55,
                     edgecolors="#ffffff", linewidths=0.4,
                     zorder=6, label=f"{n_pos} entries")

    # Gold VWAP entry
    ax_price.axhline(ve, color="#ffd700", lw=2.0, ls="-", alpha=0.9, zorder=4,
                     label=f"Avg entry  {ve:,.0f}")

    # White dotted current price
    ax_price.axhline(current_price, color="#ffffff", lw=1.0, ls=":",
                     alpha=0.75, zorder=4, label=f"Current  {current_price:,.0f}")

    # Soft fill between avg entry and current
    lo_fill  = min(ve, current_price)
    hi_fill  = max(ve, current_price)
    fill_col = "#26a69a" if current_price >= ve else "#ef5350"
    ax_price.fill_between([-1, n + 1], [lo_fill] * 2, [hi_fill] * 2,
                          color=fill_col, alpha=0.07)

    # Gap annotation top-left
    gap_pts = current_price - ve
    gap_pct = gap_pts / ve * 100
    sg      = "+" if gap_pts >= 0 else ""
    ax_price.text(
        0.012, 0.97,
        f"{sg}{gap_pts:.0f} pts  ({sg}{gap_pct:.1f}%)  above avg entry",
        transform=ax_price.transAxes, fontsize=8.5,
        color="#ffd700", va="top",
    )

    xt, xl = time_ticks(price_df, tf)
    ax_price.set_xticks(xt)
    ax_price.set_xticklabels(xl, fontsize=7.5)
    ax_price.set_xlim(-1, n + 1)
    ax_price.set_ylabel("US500.pro")
    first_bar = price_df["time"].iloc[0].strftime("%d %b %Y")
    ax_price.set_title(f"Price chart — {tf}  ({n} bars,  {first_bar} → now)")
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_price.legend(fontsize=8, loc="lower right", ncol=3)
    ax_price.grid(True, alpha=0.12)

    # ── Panel 2: P&L sensitivity ──────────────────────────────────────────────
    # Trim to sensible window: don't sweep more than ~15% below current
    lo_trim = max(current_price * 0.85, positions["price_open"].min() * 0.97)
    hi_trim = current_price * 1.05
    mask    = (prices_s >= lo_trim) & (prices_s <= hi_trim)
    p_trim  = prices_s[mask]
    u_trim  = upnl_s[mask]

    ax_sens.axhline(0, color="#555555", lw=0.8)
    ax_sens.fill_between(p_trim, 0, u_trim,
                         where=(u_trim >= 0), color="#26a69a", alpha=0.28)
    ax_sens.fill_between(p_trim, 0, u_trim,
                         where=(u_trim < 0), color="#ef5350", alpha=0.28)
    ax_sens.plot(p_trim, u_trim, color="#d8d8d8", lw=1.5)

    # Label a key vertical line with a boxed annotation near the x-axis
    # Uses blended transform: x in data coords, y in axes fraction — stays
    # at the bottom regardless of ylim.
    from matplotlib.transforms import blended_transform_factory as _btf
    _tx = _btf(ax_sens.transData, ax_sens.transAxes)

    def _vline(px, label_lines, color, ls="-", lw=1.2):
        ax_sens.axvline(px, color=color, lw=lw, ls=ls, alpha=0.85)
        ax_sens.text(px, 0.03, "\n".join(label_lines),
                     transform=_tx, ha="center", va="bottom",
                     fontsize=7.5, color=color,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="#111111", edgecolor=color,
                               alpha=0.85))

    # Current price
    pl_sign = "+" if total_pl >= 0 else ""
    ax_sens.axvline(current_price, color="#888888", lw=0.9, ls=":", alpha=0.7)
    ax_sens.scatter([current_price], [total_pl], color="#ffffff", s=65, zorder=7)
    ax_sens.annotate(
        f" {pl_sign}{total_pl:,.0f} {info.currency}",
        xy=(current_price, total_pl),
        xytext=(current_price + (hi_trim - lo_trim) * 0.02, total_pl),
        fontsize=9, color="#ffffff", va="center",
    )

    # Break-even
    if lo_trim < break_even < hi_trim:
        pct = (break_even - current_price) / current_price * 100
        _vline(break_even,
               [f"Break-even", f"{break_even:,.0f}", f"({pct:+.1f}%)"],
               "#26a69a", ls="--", lw=1.2)

    # ML 200% warning
    if warn_price and lo_trim < warn_price < hi_trim:
        pct = (warn_price - current_price) / current_price * 100
        _vline(warn_price,
               [f"ML 200%", f"{warn_price:,.0f}", f"({pct:+.1f}%)"],
               "#f08040", ls=":", lw=1.1)

    # Margin call (100%) — danger zone
    if mc_price and lo_trim < mc_price < hi_trim:
        pct = (mc_price - current_price) / current_price * 100
        ax_sens.axvspan(lo_trim, mc_price, color="#ef5350", alpha=0.06)
        _vline(mc_price,
               [f"Margin call", f"{mc_price:,.0f}", f"({pct:+.1f}%)"],
               "#ef5350", lw=1.3)

    ax_sens.set_xlim(lo_trim, hi_trim)
    ax_sens.set_xlabel("US500.pro price")
    ax_sens.set_ylabel(f"Unrealised P&L ({info.currency})")
    ax_sens.set_title("P&L vs price — how far can it fall before the tower topples?")
    ax_sens.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_sens.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_sens.grid(True, alpha=0.13)

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
