"""
05_trade_charts.py — Candle chart for every trade, showing entry and exit.

Each subplot = one trade:
  - 1 hour of M5 candles before entry  (context: what was happening before)
  - 2+ hours of M5 candles after entry (the full TP/SL lookahead window)
  - Blue line     = VWAP
  - Orange line   = EMA 50
  - White dashed  = entry bar
  - White dotted  = entry price
  - Green line    = TP level (entry + 1×ATR)
  - Red line      = SL level (entry − 1×ATR)
  - Dark green bg = trade WON (TP hit)
  - Dark red bg   = trade LOST (SL hit or timeout)

Output: one PNG per page of 16 trades → results/charts/trade_charts_*_p01.png etc.

Usage:
  python 05_trade_charts.py                            # default: power_hour_stop_trades
  python 05_trade_charts.py power_hour_market_trades   # market-entry mode
  python 05_trade_charts.py power_hour_limit_trades    # limit-entry mode
  python 05_trade_charts.py baseline_trades            # rule-based strategy
  python 05_trade_charts.py --max 32                   # show only first 32 trades
  python 05_trade_charts.py --wins                     # show only winning trades
  python 05_trade_charts.py --losses                   # show only losing trades
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cache
from indicators import add_all

# ── Layout ────────────────────────────────────────────────────────────────────
COLS           = 4
ROWS           = 4
PER_PAGE       = COLS * ROWS
CONTEXT_BEFORE = 12    # bars before entry  =  1 h on M5
CONTEXT_AFTER  = 28    # bars after entry   =  2 h 20 min (covers full 2 h window)
TP_MULT        = 1.0
SL_MULT        = 1.0


def _draw_candles(ax, opens, highs, lows, closes):
    """Draw OHLC candlesticks with coloured bodies and wicks."""
    for i in range(len(opens)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color  = "#26a69a" if c >= o else "#ef5350"
        body_h = max(abs(c - o), (h - l) * 0.015)   # minimum 1.5% of range
        ax.add_patch(mpatches.Rectangle(
            (i - 0.35, min(o, c)), 0.7, body_h,
            facecolor=color, edgecolor="none", zorder=3
        ))
        ax.plot([i, i], [l, h], color=color, lw=0.6, zorder=2)


def plot_page(df_full: pd.DataFrame, trades: pd.DataFrame,
              page: int, strategy_name: str) -> plt.Figure:
    """Render one page of PER_PAGE trade subplots."""
    plt.style.use("dark_background")

    batch = trades.iloc[page * PER_PAGE: (page + 1) * PER_PAGE]
    n = len(df_full)

    fig, axes = plt.subplots(ROWS, COLS, figsize=(20, 14))
    axes = axes.flatten()

    for ax_i, (_, trade) in enumerate(batch.iterrows()):
        ax = axes[ax_i]

        # Find entry bar by timestamp
        entry_time = pd.Timestamp(trade["time"])
        match = df_full[df_full["time"] == entry_time]
        if match.empty:
            ax.text(0.5, 0.5, "bar not found\nin cache",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.set_facecolor("#1a1a1a")
            continue

        bar_idx = match.index[0]
        start   = max(0, bar_idx - CONTEXT_BEFORE)
        end     = min(n, bar_idx + CONTEXT_AFTER + 1)
        entry_x = bar_idx - start          # entry bar's x in this subplot

        w = df_full.iloc[start:end]

        # Candlesticks
        _draw_candles(ax,
                      w["open"].values, w["high"].values,
                      w["low"].values,  w["close"].values)

        # VWAP and EMA50
        xs = range(len(w))
        ax.plot(xs, w["vwap"].values,   color="#2196F3", lw=0.9, alpha=0.80, zorder=4)
        ax.plot(xs, w["ema_50"].values, color="#FF9800", lw=0.8, alpha=0.70, zorder=4)

        # Entry / TP / SL
        entry_price = float(trade["entry"])
        atr         = float(trade["atr"])
        tp_price    = entry_price + TP_MULT * atr
        sl_price    = entry_price - SL_MULT * atr

        ax.axvline(entry_x,  color="#ffffff", lw=1.3, ls="--", alpha=0.65, zorder=5)
        ax.axhline(entry_price, color="#cccccc", lw=0.8, ls=":",  alpha=0.50, zorder=5)
        ax.axhline(tp_price,    color="#00e676", lw=1.4, ls="-",  alpha=0.85, zorder=5)
        ax.axhline(sl_price,    color="#ff1744", lw=1.4, ls="-",  alpha=0.85, zorder=5)

        # Shade risk/reward zone
        ax.axhspan(sl_price, tp_price, alpha=0.04, color="#ffffff", zorder=1)

        # Background = outcome
        won = (int(trade["label"]) == 1)
        ax.set_facecolor("#04140a" if won else "#140404")

        # X-axis: mark entry and every full UTC hour
        tick_xs, tick_labels = [], []
        for xi, (_, row) in enumerate(w.iterrows()):
            t = pd.Timestamp(row["time"])
            if t.minute == 0:
                tick_xs.append(xi)
                tick_labels.append(f"{t.hour:02d}:00")
        # Always mark entry
        tick_xs.append(entry_x)
        tick_labels.append("▶")
        ax.set_xticks(tick_xs)
        ax.set_xticklabels(tick_labels, fontsize=5)

        # Y limits: ensure TP and SL are always visible
        p_lo = min(w["low"].min(),  sl_price) - 0.4 * atr
        p_hi = max(w["high"].max(), tp_price) + 0.4 * atr
        ax.set_xlim(-0.5, len(w) - 0.5)
        ax.set_ylim(p_lo, p_hi)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.grid(True, alpha=0.08)

        # Title
        result_str = "WIN" if won else "LOSS"
        res_color  = "#00e676" if won else "#ff1744"
        pnl        = float(trade["pnl"])
        mode_str   = f"  [{trade['mode']}]" if "mode" in trade else ""
        ax.set_title(
            f"{entry_time.strftime('%b %d  %H:%M')} UTC{mode_str}   "
            f"{result_str}  {pnl:+.1f} pts   ATR={atr:.1f}",
            fontsize=6.5, color=res_color, pad=3
        )

    # Hide unused subplots
    for ax_i in range(len(batch), len(axes)):
        axes[ax_i].set_visible(False)

    # Page header
    valid    = batch[batch["label"] >= 0]
    wins     = int((valid["label"] == 1).sum())
    page_wr  = wins / len(valid) if len(valid) else 0
    trade_start = page * PER_PAGE + 1
    trade_end   = page * PER_PAGE + len(batch)

    fig.suptitle(
        f"{strategy_name}   —   trades {trade_start}–{trade_end}   "
        f"({wins}/{len(valid)} = {page_wr:.0%} WR on page {page + 1})\n"
        f"blue = VWAP   orange = EMA50   white dashed = entry   "
        f"green line = TP (+{TP_MULT}×ATR)   red line = SL (−{SL_MULT}×ATR)",
        fontsize=8, y=1.01
    )
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Trade-by-trade candle charts")
    parser.add_argument(
        "strategy", nargs="?", default="power_hour_stop_trades",
        help="Trade file name without .csv (default: power_hour_stop_trades)"
    )
    parser.add_argument("--max",    type=int,  default=0,
                        help="Max trades to chart (0 = all)")
    parser.add_argument("--wins",   action="store_true",
                        help="Show only winning trades")
    parser.add_argument("--losses", action="store_true",
                        help="Show only losing trades")
    args = parser.parse_args()

    # Load trade log
    trades_path = cache.RESULTS / f"{args.strategy}.csv"
    if not trades_path.exists():
        available = sorted(p.stem for p in cache.RESULTS.glob("*_trades.csv"))
        print(f"Not found: {trades_path.name}")
        print(f"Available trade files:")
        for a in available:
            print(f"  {a}")
        print(f"\nRun 04_backtest.py first to generate trade logs.")
        return

    trades = pd.read_csv(trades_path)
    trades["time"] = pd.to_datetime(trades["time"])
    trades = trades[trades["label"] >= 0].reset_index(drop=True)

    if args.wins:
        trades = trades[trades["label"] == 1].reset_index(drop=True)
        print("Filtering: wins only")
    elif args.losses:
        trades = trades[trades["label"] == 0].reset_index(drop=True)
        print("Filtering: losses only")

    if args.max > 0:
        trades = trades.iloc[:args.max]

    total = len(trades)
    wr    = (trades["label"] == 1).mean()
    print(f"Strategy : {args.strategy}")
    print(f"Trades   : {total:,}   Win rate: {wr:.1%}")

    # Load candle data with indicators
    print("Loading M5 data + indicators...")
    raw = cache.load_ohlcv("US500.pro", "M5")
    if raw is None:
        print("No cached data found. Run 01_fetch_data.py first.")
        return
    df = add_all(raw)
    print(f"  {len(df):,} bars loaded")

    # Plot and save one PNG per page
    n_pages = (total + PER_PAGE - 1) // PER_PAGE
    print(f"Drawing {n_pages} page(s) × {PER_PAGE} trades each...")

    saved = []
    for page in range(n_pages):
        fig  = plot_page(df, trades, page, args.strategy)
        name = f"07_trade_charts_{args.strategy}_p{page + 1:02d}"
        cache.save_chart(fig, name)
        saved.append(name)
        plt.close(fig)
        print(f"  Saved page {page + 1}/{n_pages}")

    # Show last page interactively
    fig = plot_page(df, trades, n_pages - 1, args.strategy)
    plt.show()

    print(f"\n{n_pages} chart(s) saved to results/charts/")
    print(f"First file: results/charts/{saved[0]}.png")


if __name__ == "__main__":
    main()
