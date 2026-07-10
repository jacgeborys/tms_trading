"""
Technical indicators. All functions operate on a candle DataFrame
(columns: time, open, high, low, close, volume) and return a Series or DataFrame.

Convention: use add_all() to stamp every indicator onto a copy of the candle df.
"""

import numpy as np
import pandas as pd
from typing import Dict


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ---------------------------------------------------------------------------
# Tier 1 — price-derived
# ---------------------------------------------------------------------------

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR (EMA-smoothed true range)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"]  - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(span=period, adjust=False).mean()
    avg_l = loss.ewm(span=period, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_bollinger(
    df: pd.DataFrame, period: int = 20, n_std: float = 2.0
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      bb_mid, bb_upper, bb_lower, bb_pct  (%B: 0=lower, 1=upper, 0.5=mid)
    """
    mid   = df["close"].rolling(period).mean()
    std   = df["close"].rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (df["close"] - lower) / (upper - lower)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_pct": pct_b},
        index=df.index,
    )


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP using tick volume, resets at each UTC calendar day.
    Typical institutional reference; price above VWAP = bullish bias.
    """
    tmp = df[["time", "close", "volume"]].copy()
    tmp["date"] = tmp["time"].dt.date
    tmp["pv"]   = tmp["close"] * tmp["volume"]
    cum_pv  = tmp.groupby("date")["pv"].cumsum()
    cum_vol = tmp.groupby("date")["volume"].cumsum()
    return cum_pv / cum_vol


def calc_realized_vol(
    df: pd.DataFrame,
    period: int = 60,
    bars_per_day: int = 270,
) -> pd.Series:
    """
    Rolling annualized realized volatility — our VIX proxy.

    period=60  →  5h lookback on M5 (short-term vol).
    period=20  →  ~1.5h — useful for intraday regime.
    bars_per_day=270: empirically measured M5 bars per trading day on US500.pro.
    Multiply by 252 trading days to annualize (matches VIX convention).
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    rolling_std = log_ret.rolling(period).std()
    return rolling_std * np.sqrt(bars_per_day * 252)


def calc_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: macd, macd_signal, macd_hist.
    Default params tuned for M5 (similar rhythm to daily MACD on equities).
    """
    fast_ema   = ema(df["close"], fast)
    slow_ema   = ema(df["close"], slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line,
         "macd_hist": macd_line - signal_line},
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Tier 2 — cross-instrument
# ---------------------------------------------------------------------------

def align_instruments(
    base_df: pd.DataFrame, other_dfs: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge multiple candle DataFrames on time, keeping only rows where
    the base instrument has a bar. Returns a df with close_<name> columns.
    """
    merged = base_df[["time", "close"]].rename(columns={"close": "close_base"})
    for name, df in other_dfs.items():
        tmp = df[["time", "close"]].rename(columns={"close": f"close_{name}"})
        merged = merged.merge(tmp, on="time", how="left")
    merged = merged.ffill()   # fill occasional gaps with last known price
    return merged.reset_index(drop=True)


def calc_cross_corr(
    aligned_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    period: int = 60,
) -> pd.Series:
    """Rolling Pearson correlation of log-returns between two price series."""
    ret_a = np.log(aligned_df[col_a] / aligned_df[col_a].shift(1))
    ret_b = np.log(aligned_df[col_b] / aligned_df[col_b].shift(1))
    return ret_a.rolling(period).corr(ret_b)


def calc_relative_strength(
    aligned_df: pd.DataFrame, col_a: str, col_b: str, period: int = 60
) -> pd.Series:
    """
    Rolling relative strength: z-score of (return_A - return_B).
    Positive → A outperforming B over the window.
    """
    ret_a = aligned_df[col_a].pct_change()
    ret_b = aligned_df[col_b].pct_change()
    diff  = ret_a - ret_b
    return (diff - diff.rolling(period).mean()) / diff.rolling(period).std()


def risk_off_regime(
    aligned_df: pd.DataFrame,
    spx_col: str = "close_base",
    gold_col: str = "close_GOLD",
    period: int = 60,
) -> pd.Series:
    """
    Boolean series: True when gold is strengthening while SPX weakens.
    Proxy for risk-off environment — typically bad for long entries.
    """
    spx_ret  = aligned_df[spx_col].pct_change(period)
    gold_ret = aligned_df[gold_col].pct_change(period)
    return (gold_ret > 0) & (spx_ret < 0)


# ---------------------------------------------------------------------------
# Convenience: stamp everything onto a candle df
# ---------------------------------------------------------------------------

def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all Tier 1 indicators to a candle DataFrame. Returns a new df.
    Requires columns: time, open, high, low, close, volume.
    """
    df = df.copy()

    # EMAs
    for p in [9, 21, 50, 200]:
        df[f"ema_{p}"] = ema(df["close"], p)

    # ATR
    df["atr_14"] = calc_atr(df, 14)

    # RSI
    df["rsi_14"] = calc_rsi(df, 14)

    # Bollinger Bands
    bb = calc_bollinger(df, 20)
    df = pd.concat([df, bb], axis=1)

    # VWAP
    df["vwap"] = calc_vwap(df)

    # MACD
    macd = calc_macd(df)
    df = pd.concat([df, macd], axis=1)

    # Realized vol — two windows
    df["rvol_20"] = calc_realized_vol(df, period=20)   # ~1.5h
    df["rvol_60"] = calc_realized_vol(df, period=60)   # ~5h

    return df
