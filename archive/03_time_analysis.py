"""
Time-of-day statistical analysis.

Questions:
  1. Does win rate vary meaningfully by UTC hour? By day of week?
  2. Is there short-term momentum or mean reversion in returns?
  3. Which sessions have highest realised volatility?
  4. What does the average intraday price path look like (normalised)?

All stats use the full 50k-bar dataset loaded from cache.
Results saved to results/time_analysis.json + charts/time_analysis.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

import cache
from labeler import make_long_labels, make_short_labels

SYMBOL = "US500.pro"
TF     = "M5"

# Label params — use balanced 1:1 to make win-rate interpretation clean
TP_MULT   = 1.0
SL_MULT   = 1.0
LOOKAHEAD = 24    # 2h on M5

# US session reference lines (UTC)
US_OPEN  = 14.5   # 09:30 ET
US_CLOSE = 21.0   # 16:00 ET


def load_data() -> pd.DataFrame:
    df = cache.load_ohlcv(SYMBOL, TF)
    if df is None:
        raise RuntimeError("No cached data. Run pipeline.py first.")
    return df


def add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]       = df["time"].dt.hour + df["time"].dt.minute / 60
    df["hour_int"]   = df["time"].dt.hour
    df["dow"]        = df["time"].dt.dayofweek   # 0=Mon 4=Fri
    df["dow_name"]   = df["time"].dt.strftime("%a")
    df["log_ret"]    = np.log(df["close"] / df["close"].shift(1))
    df["fwd_ret_1"]  = df["log_ret"].shift(-1)    # next 1-bar return
    df["fwd_ret_6"]  = df["log_ret"].rolling(6).sum().shift(-6)   # next 30min
    df["fwd_ret_12"] = df["log_ret"].rolling(12).sum().shift(-12) # next 1h
    df["atr_14"]     = (df["high"] - df["low"]).ewm(span=14, adjust=False).mean()
    return df


def win_rate_by_hour(df: pd.DataFrame, long_lbl: np.ndarray, short_lbl: np.ndarray) -> pd.DataFrame:
    """Win rate for LONG and SHORT labels, grouped by UTC hour."""
    tmp = df[["hour_int"]].copy()
    tmp["long_lbl"]  = long_lbl
    tmp["short_lbl"] = short_lbl
    tmp = tmp[(tmp["long_lbl"] >= 0) & (tmp["short_lbl"] >= 0)]

    rows = []
    for h in sorted(tmp["hour_int"].unique()):
        mask = tmp["hour_int"] == h
        n    = mask.sum()
        if n < 30:
            continue
        rows.append({
            "hour":       h,
            "n_bars":     n,
            "long_wr":    tmp.loc[mask, "long_lbl"].mean(),
            "short_wr":   tmp.loc[mask, "short_lbl"].mean(),
        })
    return pd.DataFrame(rows).set_index("hour")


def win_rate_by_dow(df: pd.DataFrame, long_lbl: np.ndarray) -> pd.DataFrame:
    tmp = df[["dow", "dow_name"]].copy()
    tmp["long_lbl"] = long_lbl
    tmp = tmp[tmp["long_lbl"] >= 0]
    rows = []
    for d in sorted(tmp["dow"].unique()):
        mask = tmp["dow"] == d
        rows.append({
            "day":      tmp.loc[mask, "dow_name"].iloc[0],
            "n_bars":   mask.sum(),
            "long_wr":  tmp.loc[mask, "long_lbl"].mean(),
        })
    return pd.DataFrame(rows).set_index("day")


def autocorrelation(df: pd.DataFrame, max_lag: int = 20) -> pd.Series:
    """Return autocorrelation of log returns at lags 1..max_lag."""
    ret = df["log_ret"].dropna()
    return pd.Series(
        [ret.autocorr(lag=k) for k in range(1, max_lag + 1)],
        index=range(1, max_lag + 1),
    )


def avg_intraday_path(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average normalised intraday price path by UTC hour.
    Each day's price is normalised to 100 at midnight (first bar of that day).
    """
    tmp = df[["time", "close"]].copy()
    tmp["date"] = tmp["time"].dt.date
    tmp["hour"] = tmp["time"].dt.hour + tmp["time"].dt.minute / 60

    day_start = tmp.groupby("date")["close"].transform("first")
    tmp["norm"] = (tmp["close"] / day_start - 1) * 100   # % from day start

    return tmp.groupby(tmp["time"].dt.hour)["norm"].agg(["mean", "std", "count"])


def volatility_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["hour_int", "atr_14", "log_ret"]].copy()
    tmp["abs_ret"] = tmp["log_ret"].abs()
    return tmp.groupby("hour_int").agg(
        mean_atr=("atr_14",  "mean"),
        mean_abs_ret=("abs_ret", "mean"),
        n=("atr_14", "count"),
    )


def main():
    plt.style.use("dark_background")

    print("Loading cached data...")
    df = load_data()
    df = add_time_cols(df)
    print(f"  {len(df):,} bars  ({str(df['time'].iloc[0])[:10]} → {str(df['time'].iloc[-1])[:10]})")

    print("Computing labels (1:1 RR, 2h lookahead)...")
    long_lbl  = make_long_labels( df, tp_mult=TP_MULT, sl_mult=SL_MULT, lookahead=LOOKAHEAD)
    short_lbl = make_short_labels(df, tp_mult=TP_MULT, sl_mult=SL_MULT, lookahead=LOOKAHEAD)
    random_wr = SL_MULT / (TP_MULT + SL_MULT)
    print(f"  Random-walk baseline: {random_wr:.1%}")

    print("Computing statistics...")
    wr_hour = win_rate_by_hour(df, long_lbl, short_lbl)
    wr_dow  = win_rate_by_dow(df, long_lbl)
    acf     = autocorrelation(df, max_lag=20)
    intraday = avg_intraday_path(df)
    vol_hour = volatility_by_hour(df)

    # ── Print findings ────────────────────────────────────────────────────────
    print("\n── Win rate by UTC hour (LONG, 1:1 RR, 2h lookahead) ──")
    print(f"  {'Hour':>5}  {'N':>6}  {'Long WR':>8}  {'vs Random':>10}  {'Short WR':>9}")
    for h, row in wr_hour.iterrows():
        edge = row["long_wr"] - random_wr
        flag = " ◄" if abs(edge) > 0.03 else ""
        print(f"  {h:>5}  {int(row['n_bars']):>6}  {row['long_wr']:>8.1%}  "
              f"{edge:>+10.1%}  {row['short_wr']:>9.1%}{flag}")

    print("\n── Win rate by day of week (LONG) ──")
    for day, row in wr_dow.iterrows():
        edge = row["long_wr"] - random_wr
        flag = " ◄" if abs(edge) > 0.03 else ""
        print(f"  {day}: WR={row['long_wr']:.1%}  edge={edge:+.1%}{flag}")

    print("\n── Return autocorrelation (lags 1–20 bars = 5–100min) ──")
    sig = acf[acf.abs() > 0.02]
    if len(sig):
        for lag, val in sig.items():
            direction = "momentum" if val > 0 else "mean-reversion"
            print(f"  Lag {lag:>2} ({lag*5:>3}min): r={val:+.4f}  [{direction}]")
    else:
        print("  No autocorrelation above 0.02 at any lag — returns are near i.i.d.")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("Time-of-day structural analysis — US500.pro M5", fontsize=12)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Win rate by hour
    ax1 = fig.add_subplot(gs[0, :2])
    hours = wr_hour.index
    long_wrs  = wr_hour["long_wr"].values
    short_wrs = wr_hour["short_wr"].values
    x = np.arange(len(hours))
    w = 0.35
    ax1.bar(x - w/2, long_wrs  * 100, w, label="Long WR",  color="#40a0f0", alpha=0.85)
    ax1.bar(x + w/2, short_wrs * 100, w, label="Short WR", color="#f08040", alpha=0.85)
    ax1.axhline(random_wr * 100, color="#ffffff", lw=1.2, ls="--", label=f"Random ({random_wr:.0%})")
    ax1.axvline(x[np.argmin(abs(hours - US_OPEN))],  color="#26a69a", lw=1, ls=":", alpha=0.7, label="US open")
    ax1.axvline(x[np.argmin(abs(hours - US_CLOSE))], color="#ef5350", lw=1, ls=":", alpha=0.7, label="US close")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h:02d}h" for h in hours], fontsize=8)
    ax1.set_ylabel("Win rate (%)")
    ax1.set_title("Win rate by UTC hour  (TP=SL=1×ATR, 2h lookahead)")
    ax1.legend(fontsize=7, ncol=3)
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.set_ylim(35, 65)

    # 2. Win rate by day of week
    ax2 = fig.add_subplot(gs[0, 2])
    days = list(wr_dow.index)
    dwr  = wr_dow["long_wr"].values * 100
    colors = ["#26a69a" if v > random_wr * 100 else "#ef5350" for v in dwr]
    ax2.bar(days, dwr, color=colors, alpha=0.85)
    ax2.axhline(random_wr * 100, color="#ffffff", lw=1.2, ls="--")
    ax2.set_ylabel("Long win rate (%)")
    ax2.set_title("Win rate by day of week")
    ax2.set_ylim(35, 65)
    ax2.grid(True, axis="y", alpha=0.2)

    # 3. Volatility by hour (ATR)
    ax3 = fig.add_subplot(gs[1, :2])
    vh = vol_hour[vol_hour["n"] > 100]
    ax3.bar(vh.index, vh["mean_atr"], color="#c0c0ff", alpha=0.8, label="Mean ATR")
    ax3_r = ax3.twinx()
    ax3_r.plot(vh.index, vh["mean_abs_ret"] * 10000, color="#f0c040", lw=1.5, label="|Return| (bps)")
    ax3.axvline(US_OPEN,  color="#26a69a", lw=1, ls=":", alpha=0.7)
    ax3.axvline(US_CLOSE, color="#ef5350", lw=1, ls=":", alpha=0.7)
    ax3.set_xlabel("UTC Hour")
    ax3.set_ylabel("Mean ATR (pts)")
    ax3_r.set_ylabel("|Log return| (bps)")
    ax3.set_title("Volatility by UTC hour")
    ax3.legend(loc="upper left", fontsize=7)
    ax3_r.legend(loc="upper right", fontsize=7)
    ax3.grid(True, axis="y", alpha=0.2)

    # 4. Autocorrelation
    ax4 = fig.add_subplot(gs[1, 2])
    lags = acf.index
    ax4.bar(lags, acf.values, color=np.where(acf.values > 0, "#26a69a", "#ef5350"), alpha=0.8)
    ax4.axhline(0,     color="#888888", lw=0.8)
    ax4.axhline( 0.02, color="#ffffff", lw=0.6, ls="--", alpha=0.4)
    ax4.axhline(-0.02, color="#ffffff", lw=0.6, ls="--", alpha=0.4)
    ax4.set_xlabel("Lag (bars, 1 bar = 5min)")
    ax4.set_ylabel("Autocorrelation")
    ax4.set_title("Return autocorrelation")
    ax4.grid(True, axis="y", alpha=0.2)

    # 5. Average intraday path
    ax5 = fig.add_subplot(gs[2, :])
    mean_path = intraday["mean"]
    std_path  = intraday["std"]
    x_h = mean_path.index
    ax5.plot(x_h, mean_path, color="#40a0f0", lw=2, label="Mean price path (% from day start)")
    ax5.fill_between(x_h, mean_path - std_path * 0.5,
                          mean_path + std_path * 0.5,
                     color="#40a0f0", alpha=0.12, label="±0.5σ")
    ax5.axhline(0, color="#888888", lw=0.8, ls="--")
    ax5.axvline(int(US_OPEN),  color="#26a69a", lw=1, ls=":", alpha=0.7, label="US open  (14h UTC)")
    ax5.axvline(int(US_CLOSE), color="#ef5350", lw=1, ls=":", alpha=0.7, label="US close (21h UTC)")
    ax5.set_xlabel("UTC Hour")
    ax5.set_ylabel("% from day start")
    ax5.set_title("Average normalised intraday price path (all trading days)")
    ax5.legend(fontsize=7, ncol=4)
    ax5.grid(True, alpha=0.2)

    plt.tight_layout()
    cache.save_chart(fig, "03_time_analysis")
    plt.show()

    # ── Save stats ────────────────────────────────────────────────────────────
    summary = {
        "random_wr":      random_wr,
        "wr_by_hour":     wr_hour.reset_index().to_dict(orient="records"),
        "wr_by_dow":      wr_dow.reset_index().to_dict(orient="records"),
        "acf_max_abs":    float(acf.abs().max()),
        "acf_lag_max":    int(acf.abs().idxmax()),
        "best_long_hour": int(wr_hour["long_wr"].idxmax()),
        "worst_long_hour":int(wr_hour["long_wr"].idxmin()),
    }
    cache.save_metrics(summary, "03_time_analysis")
    print("\n  Saved → results/time_analysis.json")
    print("  Saved → results/charts/time_analysis.png")


if __name__ == "__main__":
    main()
