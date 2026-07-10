"""
Power-hour explorer — CSV + candlestick chart.

WHAT THIS DOES
--------------
1. Loads M5 data and computes all indicators.
2. Filters to the 22:00–22:44 UTC window (last hour of US session = "power hour").
3. Builds a wide CSV where every row = one 5-min bar, every column = one
   indicator value, derived ratio, or boolean buy-signal flag, plus the
   trade outcome (did price hit TP before SL within 2 hours?).
4. Prints a quick filter win-rate table to the console.
5. Draws a candlestick chart of the last N trading days with:
     - M5 candles for the full visible session (18:00–23:30 UTC)
     - Power-hour window shaded in yellow
     - Entry bars marked: green triangle = win, red triangle = loss
     - VWAP and EMA50 overlaid

OUTPUTS
-------
  results/power_hour_bars.csv
  results/charts/power_hour_signals.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import cache
from indicators import add_all
from labeler import make_long_labels

# ── Parameters ────────────────────────────────────────────────────────────────
ENTRY_HOUR_UTC   = 22
ENTRY_MINUTE_END = 45      # bars at 22:00–22:44
TP_MULT          = 1.0
SL_MULT          = 1.0
LOOKAHEAD        = 24      # 2 h on M5
CHART_DAYS       = 10      # how many trading days to show on the candle chart
CHART_START_HOUR = 18      # show candles from this UTC hour each day
CHART_END_HOUR   = 23      # to this UTC hour (inclusive)


# ── Build the wide dataframe ──────────────────────────────────────────────────

def build(df_full: pd.DataFrame) -> pd.DataFrame:
    """Return a wide DataFrame of power-hour bars with all columns attached."""

    labels = make_long_labels(df_full, tp_mult=TP_MULT, sl_mult=SL_MULT,
                              lookahead=LOOKAHEAD)

    close  = df_full["close"]
    open_  = df_full["open"]
    high   = df_full["high"]
    low    = df_full["low"]
    atr    = df_full["atr_14"]
    vwap   = df_full["vwap"]

    # ATR-normalised distances from key levels
    df2 = df_full.copy()
    df2["dist_vwap_atr"]    = (close - vwap)              / atr
    df2["dist_ema9_atr"]    = (close - df_full["ema_9"])  / atr
    df2["dist_ema21_atr"]   = (close - df_full["ema_21"]) / atr
    df2["dist_ema50_atr"]   = (close - df_full["ema_50"]) / atr
    df2["dist_ema200_atr"]  = (close - df_full["ema_200"])/ atr

    # Volatility context
    df2["rvol_ratio"]       = df_full["rvol_20"] / df_full["rvol_60"]  # <1 = contracting
    df2["atr_pct"]          = atr / close * 100                         # ATR as % of price

    # Bar-level
    df2["bar_return_pct"]   = (close - open_) / open_ * 100
    df2["bar_range_atr"]    = (high - low) / atr
    df2["prev1_return_pct"] = close.pct_change(1) * 100
    df2["prev2_return_pct"] = close.pct_change(2) * 100

    # ── Boolean buy-signal filters ────────────────────────────────────────────
    # Each of these is a candidate entry condition.
    # In live trading the bot will evaluate every one of these at the
    # current bar and only enter if the required combination is True.
    df2["f_above_vwap"]      = (close > vwap).astype(int)
    df2["f_above_ema9"]      = (close > df_full["ema_9"]).astype(int)
    df2["f_above_ema21"]     = (close > df_full["ema_21"]).astype(int)
    df2["f_above_ema50"]     = (close > df_full["ema_50"]).astype(int)
    df2["f_above_ema200"]    = (close > df_full["ema_200"]).astype(int)
    df2["f_rsi_lt70"]        = (df_full["rsi_14"] < 70).astype(int)
    df2["f_rsi_lt65"]        = (df_full["rsi_14"] < 65).astype(int)
    df2["f_rsi_lt60"]        = (df_full["rsi_14"] < 60).astype(int)
    df2["f_rsi_gt40"]        = (df_full["rsi_14"] > 40).astype(int)
    df2["f_rsi_gt50"]        = (df_full["rsi_14"] > 50).astype(int)
    df2["f_macd_pos"]        = (df_full["macd_hist"] > 0).astype(int)
    df2["f_macd_rising"]     = (df_full["macd_hist"] > df_full["macd_hist"].shift(1)).astype(int)
    df2["f_bb_upper_half"]   = (df_full["bb_pct"] > 0.5).astype(int)
    df2["f_bb_lt80"]         = (df_full["bb_pct"] < 0.8).astype(int)
    df2["f_vol_contracting"] = (df_full["rvol_20"] < df_full["rvol_60"]).astype(int)
    df2["f_green_bar"]       = (close > open_).astype(int)

    df2["label"] = labels

    # ── Slice to entry window ─────────────────────────────────────────────────
    h    = df2["time"].dt.hour
    m    = df2["time"].dt.minute
    mask = (h == ENTRY_HOUR_UTC) & (m < ENTRY_MINUTE_END)
    ph   = df2[mask].copy()

    ph.insert(1, "date",           ph["time"].dt.date)
    ph.insert(2, "day_of_week",    ph["time"].dt.day_name().str[:3])
    ph.insert(3, "session_minute", ph["time"].dt.minute)

    float_cols = ph.select_dtypes("float64").columns
    ph[float_cols] = ph[float_cols].round(4)

    return ph.reset_index(drop=True)


# ── Candlestick chart ─────────────────────────────────────────────────────────

def _draw_candles(ax, df_day, x_offset=0):
    """Draw OHLC candlesticks on ax. df_day must have open/high/low/close."""
    for i, (_, row) in enumerate(df_day.iterrows()):
        x    = x_offset + i
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color  = "#26a69a" if c >= o else "#ef5350"   # green / red
        # Body
        body_y = min(o, c)
        body_h = abs(c - o) if abs(c - o) > 0 else 0.05
        ax.add_patch(mpatches.Rectangle(
            (x - 0.3, body_y), 0.6, body_h,
            color=color, zorder=3
        ))
        # Wick
        ax.plot([x, x], [l, h], color=color, lw=0.7, zorder=2)


def plot_signals(df_full: pd.DataFrame, ph: pd.DataFrame):
    """
    Multi-day candlestick chart.
    Shows the last CHART_DAYS trading days. Each row = one day.
    Time window per day: CHART_START_HOUR–CHART_END_HOUR UTC.
    Power-hour window shaded. Entry bars = coloured triangles.
    """
    plt.style.use("dark_background")

    # Get list of trading dates in descending order
    all_dates = sorted(df_full["time"].dt.date.unique())
    recent_dates = all_dates[-CHART_DAYS:]

    fig, axes = plt.subplots(
        CHART_DAYS, 1,
        figsize=(18, CHART_DAYS * 2.2),
        sharex=False
    )
    if CHART_DAYS == 1:
        axes = [axes]

    fig.suptitle(
        f"Power-hour entries — last {CHART_DAYS} trading days  "
        f"(22:00–22:44 UTC / 17:00–17:44 ET)\n"
        f"▲ green = win (TP hit)   ▼ red = loss (SL hit)   shaded = power-hour window",
        fontsize=10, y=1.01
    )

    for ax, date in zip(axes, recent_dates):
        # Candle data for this day in the visible window
        day_mask = (
            (df_full["time"].dt.date == date) &
            (df_full["time"].dt.hour >= CHART_START_HOUR) &
            (df_full["time"].dt.hour <= CHART_END_HOUR)
        )
        day_df = df_full[day_mask].reset_index(drop=True)
        if len(day_df) == 0:
            ax.set_visible(False)
            continue

        _draw_candles(ax, day_df)

        # VWAP and EMA50 lines
        xs = list(range(len(day_df)))
        ax.plot(xs, day_df["vwap"].values,   color="#2196F3", lw=1.0, alpha=0.8, label="VWAP")
        ax.plot(xs, day_df["ema_50"].values, color="#FF9800", lw=0.8, alpha=0.7, label="EMA50")

        # Shade the power-hour window
        ph_start_x = None
        ph_end_x   = None
        for i, row in day_df.iterrows():
            h, m = row["time"].hour, row["time"].minute
            if h == ENTRY_HOUR_UTC and m == 0:
                ph_start_x = i
            if h == ENTRY_HOUR_UTC and m == ENTRY_MINUTE_END - 5:
                ph_end_x = i + 1
        if ph_start_x is not None and ph_end_x is not None:
            ax.axvspan(ph_start_x - 0.5, ph_end_x + 0.5,
                       color="#f0c040", alpha=0.07, zorder=1, label="Power hour")

        # Entry signals for this day
        day_ph = ph[ph["date"] == date]
        for _, sig in day_ph.iterrows():
            if sig["label"] == -1:
                continue
            # Find x position of this bar in day_df
            bar_time = sig["time"]
            match = day_df[day_df["time"] == bar_time]
            if match.empty:
                continue
            xi    = match.index[0]
            price = sig["close"]
            atr   = sig["atr_14"]
            if sig["label"] == 1:   # win
                ax.plot(xi, price - 0.3 * atr, marker="^", color="#00e676",
                        markersize=7, zorder=5, linestyle="None")
            else:                    # loss
                ax.plot(xi, price + 0.3 * atr, marker="v", color="#ff1744",
                        markersize=7, zorder=5, linestyle="None")

        # X-axis: show only round hours as labels
        tick_xs    = []
        tick_labels = []
        for i, row in day_df.iterrows():
            if row["time"].minute == 0:
                tick_xs.append(i)
                tick_labels.append(f"{row['time'].hour:02d}:00")
        ax.set_xticks(tick_xs)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlim(-0.5, len(day_df) - 0.5)

        price_range = day_df["high"].max() - day_df["low"].min()
        ax.set_ylim(day_df["low"].min() - 0.1 * price_range,
                    day_df["high"].max() + 0.1 * price_range)
        ax.set_ylabel(str(date), fontsize=7, rotation=0, labelpad=50, va="center")
        ax.yaxis.set_tick_params(labelsize=6)
        ax.grid(True, alpha=0.1)

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading M5 data...")
    raw = cache.load_ohlcv("US500.pro", "M5")
    if raw is None:
        print("No cached data found. Run pipeline.py first to fetch from MT5.")
        return

    print("Computing indicators...")
    df = add_all(raw)
    print(f"  Full df: {len(df):,} bars")

    print("Building power-hour table...")
    ph = build(df)

    valid = ph[ph["label"] != -1]
    wins  = valid[valid["label"] == 1]
    total = len(valid)
    print(f"  Power-hour bars : {len(ph):,}")
    print(f"  Labelled trades : {total:,}")
    print(f"  Base win rate   : {len(wins)/total:.1%}  (need >58.2% to profit)")

    # Save CSV
    out_path = cache.RESULTS / "08_power_hour_bars.csv"
    ph.to_csv(out_path, index=False)
    print(f"\nCSV saved → {out_path}  ({len(ph.columns)} columns)")

    # Filter win-rate preview
    print("\n── Filter win-rate preview ─────────────────────────────────────────")
    print(f"  {'Filter':<25}  {'N':>5}  {'WR':>6}  {'vs base':>8}")
    base_wr = len(wins) / total
    filter_cols = [c for c in ph.columns if c.startswith("f_")]
    rows = []
    for col in filter_cols:
        sub = valid[valid[col] == 1]
        if len(sub) < 50:
            continue
        wr = sub["label"].mean()
        rows.append((col, len(sub), wr, wr - base_wr))
    rows.sort(key=lambda x: -x[3])
    for col, n, wr, delta in rows:
        bar = "█" * int(abs(delta) * 200)
        sign = "+" if delta >= 0 else ""
        print(f"  {col:<25}  {n:>5}  {wr:>6.1%}  {sign}{delta:>+.1%}  {bar}")

    # Candlestick chart
    print(f"\nDrawing candlestick chart (last {CHART_DAYS} trading days)...")
    fig = plot_signals(df, ph)
    cache.save_chart(fig, "08_power_hour_signals")
    plt.show()


if __name__ == "__main__":
    main()
