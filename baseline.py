"""
Rule-based baseline strategy.

LONG:  price > VWAP, price > EMA21, 40 < RSI < 65, MACD hist > 0, vol contracting
SHORT: price < VWAP, price < EMA21, 35 < RSI < 60, MACD hist < 0, vol contracting

Both restricted to US session (14:30–20:30 UTC).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest import simulate, metrics, print_metrics, SPREAD_PTS


def _us_session(df: pd.DataFrame) -> np.ndarray:
    mins = df["time"].dt.hour * 60 + df["time"].dt.minute
    return ((mins >= 870) & (mins < 1230)).values


def long_signals(df: pd.DataFrame) -> np.ndarray:
    """
    Mean-reversion LONG: price has pulled back to value area, momentum turning up.
      - US session
      - Price at or below VWAP (pulled back to institutional anchor)
      - Price above EMA50 (still in uptrend context)
      - RSI oversold recovery: 30-50 (bouncing from weak zone)
      - MACD histogram turning positive (momentum shift)
      - Vol contracting (quiet before move)
    """
    sess        = _us_session(df)
    near_vwap   = df["close"].values <= df["vwap"].values * 1.001
    above_ema50 = df["close"].values > df["ema_50"].values
    rsi_ok      = (df["rsi_14"].values > 30) & (df["rsi_14"].values < 52)
    macd_pos    = df["macd_hist"].values > 0
    vol_comp    = df["rvol_20"].values < df["rvol_60"].values
    return sess & near_vwap & above_ema50 & rsi_ok & macd_pos & vol_comp


def short_signals(df: pd.DataFrame) -> np.ndarray:
    """
    Mean-reversion SHORT: price extended above value, momentum turning down.
    """
    sess        = _us_session(df)
    above_vwap  = df["close"].values >= df["vwap"].values * 0.999
    below_ema50 = df["close"].values < df["ema_50"].values
    rsi_ok      = (df["rsi_14"].values > 48) & (df["rsi_14"].values < 70)
    macd_neg    = df["macd_hist"].values < 0
    vol_comp    = df["rvol_20"].values < df["rvol_60"].values
    return sess & above_vwap & below_ema50 & rsi_ok & macd_neg & vol_comp


def run(
    df: pd.DataFrame,
    long_labels: np.ndarray,
    short_labels: np.ndarray,
    tp_mult: float,
    sl_mult: float,
) -> tuple:
    atr   = df["atr_14"].values
    times = df["time"].values

    l_trades = simulate(long_signals(df),  long_labels,  tp_mult, sl_mult, atr, times)
    s_trades = simulate(short_signals(df), short_labels, tp_mult, sl_mult, atr, times)

    all_trades = (
        pd.concat([l_trades, s_trades])
        .sort_values("bar_idx")
        .reset_index(drop=True)
    )

    return all_trades, metrics(l_trades), metrics(s_trades), metrics(all_trades)


def plot_equity(trades: pd.DataFrame, ax=None, label: str = "Baseline"):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    if len(trades) == 0:
        ax.set_title(f"{label} — no trades")
        return ax

    cum = trades["pnl"].cumsum()
    x   = trades["time"] if "time" in trades.columns else trades["bar_idx"]

    ax.plot(x, cum, color="#40a0f0", lw=1.5, label=label)
    ax.fill_between(x, cum, 0, where=(cum >= 0), color="#26a69a", alpha=0.15)
    ax.fill_between(x, cum, 0, where=(cum <  0), color="#ef5350", alpha=0.15)
    ax.axhline(0, color="#888888", lw=0.7, ls="--")
    ax.set_ylabel("Cumulative P&L (pts)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    return ax
