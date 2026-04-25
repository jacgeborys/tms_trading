"""
Exploration script: fetch live data, apply all indicators, plot everything.
Run standalone: python explore.py
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from mt5_client import connect, disconnect
from data import get_candles
from indicators import (
    add_all,
    align_instruments,
    calc_cross_corr,
    calc_relative_strength,
    risk_off_regime,
)
from config import SYMBOL

# ── Config ──────────────────────────────────────────────────────────────────
N_BARS      = 400        # candles to plot
TIMEFRAME   = "M5"
CORR_PERIOD = 60         # bars for rolling correlation (~5h on M5)

CROSS_SYMBOLS = {
    "US100": "US100.pro",
    "US30":  "US30.pro",
    "GOLD":  "GOLD.pro",
}
# ────────────────────────────────────────────────────────────────────────────


def fetch_all():
    """Fetch US500 + all cross instruments, same timeframe & length."""
    base = get_candles(n=N_BARS, symbol=SYMBOL, timeframe=TIMEFRAME)
    others = {}
    for name, sym in CROSS_SYMBOLS.items():
        others[name] = get_candles(n=N_BARS, symbol=sym, timeframe=TIMEFRAME)
    return base, others


def plot_price_panel(ax, df):
    """Price line + EMAs + Bollinger Bands + VWAP."""
    ax.plot(df["time"], df["close"],   color="#e8e8e8", lw=1,   label="Close", zorder=3)
    ax.plot(df["time"], df["ema_9"],   color="#f0c040", lw=1,   label="EMA 9",  alpha=0.85)
    ax.plot(df["time"], df["ema_21"],  color="#40a0f0", lw=1,   label="EMA 21", alpha=0.85)
    ax.plot(df["time"], df["ema_50"],  color="#f08040", lw=1.2, label="EMA 50", alpha=0.85)
    ax.plot(df["time"], df["ema_200"], color="#c040f0", lw=1.5, label="EMA 200",alpha=0.85)
    ax.plot(df["time"], df["vwap"],    color="#40f0a0", lw=1.2, ls="--", label="VWAP", alpha=0.9)
    ax.fill_between(df["time"], df["bb_upper"], df["bb_lower"],
                    color="#4080ff", alpha=0.08, label="BB 2σ")
    ax.plot(df["time"], df["bb_upper"], color="#4080ff", lw=0.6, alpha=0.5)
    ax.plot(df["time"], df["bb_lower"], color="#4080ff", lw=0.6, alpha=0.5)
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=7, ncol=4)
    ax.grid(True, alpha=0.2)


def plot_volume_panel(ax, df):
    """Tick volume bars, colour-coded by bar direction."""
    colors = np.where(df["close"] >= df["open"], "#26a69a", "#ef5350")
    ax.bar(df["time"], df["volume"], color=colors, width=0.003, alpha=0.8)
    ax.set_ylabel("Volume")
    ax.grid(True, alpha=0.2)


def plot_rsi_panel(ax, df):
    ax.plot(df["time"], df["rsi_14"], color="#f0c040", lw=1)
    ax.axhline(70, color="#ef5350", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(50, color="#888888", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(30, color="#26a69a", lw=0.8, ls="--", alpha=0.7)
    ax.fill_between(df["time"], df["rsi_14"], 70,
                    where=(df["rsi_14"] >= 70), color="#ef5350", alpha=0.15)
    ax.fill_between(df["time"], df["rsi_14"], 30,
                    where=(df["rsi_14"] <= 30), color="#26a69a", alpha=0.15)
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI 14")
    ax.grid(True, alpha=0.2)


def plot_macd_panel(ax, df):
    colors = np.where(df["macd_hist"] >= 0, "#26a69a", "#ef5350")
    ax.bar(df["time"], df["macd_hist"], color=colors, width=0.003, alpha=0.7)
    ax.plot(df["time"], df["macd"],        color="#f0c040", lw=1,   label="MACD")
    ax.plot(df["time"], df["macd_signal"], color="#ff8c00", lw=0.8, label="Signal")
    ax.axhline(0, color="#888888", lw=0.6, alpha=0.5)
    ax.set_ylabel("MACD")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.2)


def plot_atr_panel(ax, df):
    ax.plot(df["time"], df["atr_14"], color="#c0c0ff", lw=1, label="ATR 14")
    ax.set_ylabel("ATR")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.2)


def plot_rvol_panel(ax, df):
    ax.plot(df["time"], df["rvol_60"] * 100, color="#ff6060", lw=1,
            label="RVol 60-bar (~5h) [%]")
    ax.plot(df["time"], df["rvol_20"] * 100, color="#ffa0a0", lw=0.8, alpha=0.7,
            label="RVol 20-bar (~1.5h) [%]")
    ax.set_ylabel("Ann. RVol %")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.2)


def plot_cross_panel(ax, aligned, corr_nd, corr_gold):
    """Cross-instrument rolling correlation."""
    ax.plot(aligned["time"], corr_nd,   color="#40a0f0", lw=1, label="Corr US500 / US100")
    ax.plot(aligned["time"], corr_gold, color="#f0c040", lw=1, label="Corr US500 / GOLD")
    ax.axhline(0,    color="#888888", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(0.5,  color="#40a0f0", lw=0.5, ls=":", alpha=0.4)
    ax.axhline(-0.5, color="#f0c040", lw=0.5, ls=":", alpha=0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(f"Roll Corr ({CORR_PERIOD}b)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.2)


def plot_relative_panel(ax, aligned):
    """Normalized price performance (all indexed to 100 at bar 0)."""
    for col, color, label in [
        ("close_base",  "#e8e8e8", "US500"),
        ("close_US100", "#40a0f0", "US100"),
        ("close_US30",  "#f08040", "US30"),
        ("close_GOLD",  "#f0c040", "Gold"),
    ]:
        normed = aligned[col] / aligned[col].iloc[0] * 100
        ax.plot(aligned["time"], normed, color=color, lw=1, label=label)
    ax.set_ylabel("Norm. Price (100)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)


def main():
    if not connect():
        return

    try:
        print(f"Fetching {N_BARS} {TIMEFRAME} bars...")
        base_df, others = fetch_all()

        print("Computing indicators...")
        df = add_all(base_df)

        # Cross-instrument alignment
        aligned = align_instruments(base_df, others)

        corr_nd   = calc_cross_corr(aligned, "close_base", "close_US100", CORR_PERIOD)
        corr_gold = calc_cross_corr(aligned, "close_base", "close_GOLD",  CORR_PERIOD)
        rel_str   = calc_relative_strength(aligned, "close_base", "close_US100", CORR_PERIOD)
        regime    = risk_off_regime(aligned)

        n_risk_off = regime.sum()
        print(f"Risk-off bars in window: {n_risk_off} / {len(aligned)} "
              f"({n_risk_off/len(aligned)*100:.1f}%)")

        # ── Summary stats ────────────────────────────────────────────────────
        last = df.iloc[-1]
        print(f"\n── Last bar ({str(last['time'])[:16]}) ──────────────────")
        print(f"  Close   : {last['close']:.1f}")
        print(f"  EMA 9   : {last['ema_9']:.1f}   EMA 21: {last['ema_21']:.1f}   "
              f"EMA 50: {last['ema_50']:.1f}")
        print(f"  VWAP    : {last['vwap']:.1f}   ({'ABOVE' if last['close'] > last['vwap'] else 'BELOW'} vwap)")
        print(f"  RSI 14  : {last['rsi_14']:.1f}")
        print(f"  ATR 14  : {last['atr_14']:.2f} pts")
        print(f"  BB %B   : {last['bb_pct']:.2f}  (0=lower band, 1=upper band)")
        print(f"  MACD    : {last['macd']:.3f}  hist: {last['macd_hist']:.3f}")
        print(f"  RVol 60 : {last['rvol_60']*100:.1f}%  RVol 20: {last['rvol_20']*100:.1f}%")
        print(f"  Regime  : {'RISK-OFF' if regime.iloc[-1] else 'NORMAL'}")

        # ── Plot ─────────────────────────────────────────────────────────────
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            f"{SYMBOL}  |  {TIMEFRAME}  |  {N_BARS} bars  |  "
            f"last: {str(last['time'])[:16]}",
            fontsize=11, color="#cccccc",
        )

        gs = gridspec.GridSpec(
            8, 1, figure=fig,
            height_ratios=[3, 0.8, 1, 1, 1, 1, 1, 1],
            hspace=0.08,
        )

        axes = [fig.add_subplot(gs[i]) for i in range(8)]

        plot_price_panel(axes[0], df)
        plot_volume_panel(axes[1], df)
        plot_rsi_panel(axes[2], df)
        plot_macd_panel(axes[3], df)
        plot_atr_panel(axes[4], df)
        plot_rvol_panel(axes[5], df)
        plot_cross_panel(axes[6], aligned, corr_nd, corr_gold)
        plot_relative_panel(axes[7], aligned)

        # x-axis: only show on last panel
        fmt = mdates.DateFormatter("%m-%d %H:%M")
        for i, ax in enumerate(axes):
            ax.xaxis.set_major_formatter(fmt)
            if i < len(axes) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

        plt.tight_layout()
        plt.show()

    finally:
        disconnect()


if __name__ == "__main__":
    main()
