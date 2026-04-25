"""
Power-hour strategy: Long-only, hour 22 UTC (17:00 ET).

Entry mechanics (four modes):
  'market'   : fill at close of signal bar + spread   (all power-hour bars)
  'limit'    : fill only if next bar's low <= close - offset*ATR
  'stop'     : fill only if next bar's high >= close + offset*ATR
  'filtered' : market entry, but ONLY when RSI was below 30 in the last 50 min
               (recent_oversold filter — the best combination from 05_strategy_search)

TP / SL remain ATR-based.  All costs included.

Outputs:
  - results/power_hour_<mode>_metrics.json
  - results/power_hour_<mode>_trades.csv
  - results/charts/power_hour_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cache
from indicators import add_all
from labeler import make_long_labels
from backtest import metrics, print_metrics, SPREAD_PTS

# ── Strategy parameters ───────────────────────────────────────────────────────
ENTRY_HOUR_UTC = 22          # Only consider bars starting at 22:xx UTC
ENTRY_MINUTE_END = 45        # Don't enter after 22:45 (need time to reach TP before close)
TP_MULT   = 1.0              # ATR multiples for TP  (match time_analysis params)
SL_MULT   = 1.0              # ATR multiples for SL
LOOKAHEAD = 24               # Max bars to reach TP/SL (2h on M5)
STOP_OFFSET  = 0.15          # Buy stop: entry bar high + 0.15×ATR above
LIMIT_OFFSET = 0.15          # Buy limit: entry bar low  - 0.15×ATR below


def session_filter(df: pd.DataFrame) -> np.ndarray:
    """True for bars in the entry window: 22:00–22:44 UTC."""
    h = df["time"].dt.hour.values
    m = df["time"].dt.minute.values
    return (h == ENTRY_HOUR_UTC) & (m < ENTRY_MINUTE_END)


def recent_oversold_filter(df: pd.DataFrame, lookback: int = 10) -> np.ndarray:
    """
    True when RSI was below 30 (oversold) within the last `lookback` bars (default=10, i.e. 50 min).
    This is the best secondary condition found by 05_strategy_search:
      time_power_hour + recent_oversold → 62.7% WR on 413 trades (vs 58.2% break-even).
    Logic: power-hour institutional buying + a recent dip-and-recovery = double tailwind.
    """
    rsi    = df["rsi_14"].values
    is_os  = rsi < 30          # True where RSI is currently oversold
    n      = len(rsi)
    result = np.zeros(n, dtype=bool)
    count  = 0                  # bars since last oversold event
    for i in range(n):
        if is_os[i]:
            count = 0
        else:
            count += 1
        result[i] = count < lookback
    return result


def simulate_market(
    df: pd.DataFrame,
    signal_mask: np.ndarray,
    long_labels: np.ndarray,
    tp_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    """Standard market order: fill at close + spread."""
    valid = signal_mask & (long_labels >= 0)
    idx   = np.where(valid)[0]
    if not len(idx):
        return pd.DataFrame()

    atrs = df["atr_14"].values[idx]
    labs = long_labels[idx]
    pnl  = np.where(labs == 1, tp_mult * atrs, -sl_mult * atrs) - SPREAD_PTS

    return pd.DataFrame({
        "bar_idx": idx,
        "time":    df["time"].values[idx],
        "entry":   df["close"].values[idx],
        "label":   labs,
        "atr":     atrs,
        "pnl":     pnl,
        "mode":    "market",
    })


def simulate_stop(
    df: pd.DataFrame,
    signal_mask: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    offset: float = STOP_OFFSET,
) -> pd.DataFrame:
    """
    Buy stop: enter only if the NEXT bar's high >= signal bar close + offset×ATR.
    Entry price = stop level (not next bar's open — conservative).
    Then look forward LOOKAHEAD bars from entry bar for TP/SL.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr_14"].values
    times  = df["time"].values
    n      = len(df)

    rows = []
    sig_idx = np.where(signal_mask)[0]

    for i in sig_idx:
        if i + 1 >= n:
            continue
        atr        = atrs[i]
        stop_price = closes[i] + offset * atr      # trigger level
        tp_price   = stop_price + tp_mult * atr
        sl_price   = stop_price - sl_mult * atr

        # Check if stop triggers on the very next bar
        if highs[i + 1] < stop_price:
            continue   # never triggered — no trade

        entry = stop_price   # filled at stop level

        # Scan forward from i+1 for TP or SL (recomputed from actual entry)
        label = 0
        for j in range(i + 1, min(i + 1 + LOOKAHEAD, n)):
            if highs[j] >= tp_price:
                label = 1
                break
            if lows[j] <= sl_price:
                label = 0
                break

        tp_pts = tp_mult * atr
        sl_pts = sl_mult * atr
        pnl    = (tp_pts if label == 1 else -sl_pts) - SPREAD_PTS

        rows.append({
            "bar_idx": i,
            "time":    times[i],
            "entry":   entry,
            "label":   label,
            "atr":     atr,
            "pnl":     pnl,
            "mode":    "stop",
        })

    return pd.DataFrame(rows)


def simulate_limit(
    df: pd.DataFrame,
    signal_mask: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    offset: float = LIMIT_OFFSET,
) -> pd.DataFrame:
    """
    Buy limit: enter only if the NEXT bar's low <= signal bar close - offset×ATR.
    Entry price = limit level.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr_14"].values
    times  = df["time"].values
    n      = len(df)

    rows = []
    sig_idx = np.where(signal_mask)[0]

    for i in sig_idx:
        if i + 1 >= n:
            continue
        atr          = atrs[i]
        limit_price  = closes[i] - offset * atr     # fill level
        tp_price     = limit_price + tp_mult * atr
        sl_price     = limit_price - sl_mult * atr

        # Check if limit fills on next bar
        if lows[i + 1] > limit_price:
            continue   # price never came down — no fill

        entry = limit_price

        label = 0
        for j in range(i + 1, min(i + 1 + LOOKAHEAD, n)):
            if highs[j] >= tp_price:
                label = 1
                break
            if lows[j] <= sl_price:
                label = 0
                break

        tp_pts = tp_mult * atr
        sl_pts = sl_mult * atr
        pnl    = (tp_pts if label == 1 else -sl_pts) - SPREAD_PTS

        rows.append({
            "bar_idx": i,
            "time":    times[i],
            "entry":   entry,
            "label":   label,
            "atr":     atr,
            "pnl":     pnl,
            "mode":    "limit",
        })

    return pd.DataFrame(rows)


def plot_comparison(results: dict):
    """Equity curves + trade distribution for all entry modes."""
    n_modes = len(results)
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, n_modes, figure=fig, hspace=0.4, wspace=0.35)

    colors = {
        "market":   "#f0c040",
        "stop":     "#40a0f0",
        "limit":    "#f08040",
        "filtered": "#80e080",   # green — the best mode
    }

    # Equity curves (top row, spanning full width)
    ax_eq = fig.add_subplot(gs[0, :])
    for mode, trades in results.items():
        if len(trades) == 0:
            continue
        cum   = trades["pnl"].cumsum()
        x     = trades["time"]
        color = colors.get(mode, "#aaaaaa")
        lw    = 2.2 if mode == "filtered" else 1.5
        ax_eq.plot(x, cum, color=color, lw=lw, label=f"{mode} (n={len(trades)})")
        ax_eq.fill_between(x, cum, 0, where=(cum >= 0), color=color, alpha=0.06)
    ax_eq.axhline(0, color="#888888", lw=0.8, ls="--")
    ax_eq.set_title(
        f"Power-hour strategy — equity curves  "
        f"(TP={TP_MULT}×ATR  SL={SL_MULT}×ATR  entry window 22:00–22:44 UTC)\n"
        f"'filtered' = power hour AND RSI was below 30 in last 50 min"
    )
    ax_eq.set_ylabel("Cumulative P&L (pts)")
    ax_eq.legend(fontsize=9)
    ax_eq.grid(True, alpha=0.2)
    plt.setp(ax_eq.get_xticklabels(), rotation=20, ha="right", fontsize=7)

    # P&L distribution per mode (bottom row)
    for col, (mode, trades) in enumerate(results.items()):
        ax    = fig.add_subplot(gs[1, col])
        color = colors.get(mode, "#aaaaaa")
        if len(trades) == 0:
            ax.set_title(f"{mode} — no trades")
            continue
        m = metrics(trades)
        ax.hist(trades["pnl"], bins=30, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(0, color="#ffffff", lw=0.8, ls="--")
        ax.axvline(trades["pnl"].mean(), color="#ff4040", lw=1.2, ls="-",
                   label=f"mean {trades['pnl'].mean():+.2f}")
        ax.set_title(
            f"{mode.upper()} entry\n"
            f"n={m['n_trades']}  WR={m['win_rate']:.1%}  "
            f"Sharpe={m['sharpe']:.2f}\n"
            f"PnL={m['total_pnl']:+.1f}pts",
            fontsize=8
        )
        ax.set_xlabel("Trade P&L (pts)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Entry mode comparison — Power Hour (22:00 UTC / 17:00 ET)", fontsize=11)
    return fig


def main():
    plt.style.use("dark_background")

    print("Loading data...")
    df = add_all(cache.load_ohlcv("US500.pro", "M5"))
    print(f"  {len(df):,} bars")

    # Signal: bars in the entry window
    sig = session_filter(df)
    print(f"\nEntry window bars (22:00–22:44 UTC): {sig.sum():,}")

    # Filtered signal: power hour AND RSI was recently oversold (< 30 in last 50 min)
    osd      = recent_oversold_filter(df, lookback=10)
    sig_filt = sig & osd
    print(f"Filtered bars (+ recent_oversold):   {sig_filt.sum():,}")

    # Pre-compute labels for market-mode comparison
    long_lbl = make_long_labels(df, tp_mult=TP_MULT, sl_mult=SL_MULT, lookahead=LOOKAHEAD)

    print(f"\nRunning four entry modes  (TP={TP_MULT}×ATR  SL={SL_MULT}×ATR)...")

    mkt      = simulate_market(df, sig,      long_lbl, TP_MULT, SL_MULT)
    stop     = simulate_stop(  df, sig,      TP_MULT,  SL_MULT)
    limit    = simulate_limit( df, sig,      TP_MULT,  SL_MULT)
    filtered = simulate_market(df, sig_filt, long_lbl, TP_MULT, SL_MULT)

    results = {"market": mkt, "stop": stop, "limit": limit, "filtered": filtered}

    for mode, trades in results.items():
        m = metrics(trades)
        base  = sig_filt.sum() if mode == "filtered" else sig.sum()
        fill_rate = len(trades) / max(base, 1)
        print_metrics(m, f"Power Hour — {mode.upper()} entry  (fill rate {fill_rate:.0%})")
        cache.save_metrics(m,      f"power_hour_{mode}_metrics")
        cache.save_trades(trades,  f"power_hour_{mode}_trades")

    fig = plot_comparison(results)
    cache.save_chart(fig, "06_power_hour_comparison")
    plt.show()

    # Summary
    print("\n" + "═" * 60)
    print("  POWER HOUR SUMMARY")
    print("═" * 60)
    for mode, trades in results.items():
        if not len(trades):
            continue
        m    = metrics(trades)
        base = sig_filt.sum() if mode == "filtered" else sig.sum()
        fr   = len(trades) / max(base, 1)
        flag = "  ← ABOVE BREAK-EVEN" if m["win_rate"] > 0.582 else ""
        print(f"  {mode:<10}: WR={m['win_rate']:.1%}  Sharpe={m['sharpe']:+.2f}  "
              f"fill={fr:.0%}  PnL={m['total_pnl']:+.1f}pts{flag}")
    print("═" * 60)
    print("\n  Saved → results/power_hour_*  +  results/charts/06_power_hour_comparison.png")


if __name__ == "__main__":
    main()
