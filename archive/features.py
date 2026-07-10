"""
Feature matrix builder.

All features are computed from data available at bar close — no look-ahead.
Time features use cyclical encoding so the model sees hour/day as continuous.
"""

import numpy as np
import pandas as pd
from indicators import add_all, align_instruments, calc_cross_corr, calc_relative_strength

CORR_PERIOD = 60   # bars for rolling cross-instrument correlation (~5h on M5)

FEATURE_COLS = [
    # Price position (ATR-normalised so scale is comparable across regimes)
    "dist_ema9_atr", "dist_ema21_atr", "dist_ema50_atr", "dist_ema200_atr",
    "dist_vwap_atr",
    # Momentum
    "rsi_14", "macd_hist", "bb_pct",
    # Volatility
    "atr_pct", "rvol_ratio",
    # Cross-instrument
    "corr_us100", "corr_gold", "rel_str_nd",
    # Time
    "hour_sin", "hour_cos", "day_of_week", "is_us_session",
    # Volume
    "rel_volume",
]


def build_features(
    base_df: pd.DataFrame,
    other_dfs: dict,
) -> pd.DataFrame:
    """
    Returns a DataFrame with FEATURE_COLS + 'time' column.
    Rows align 1-to-1 with base_df.  NaNs from indicator warm-up are kept
    (caller should dropna before training).
    """
    df = add_all(base_df)

    aligned = align_instruments(base_df, other_dfs)
    corr_us100 = calc_cross_corr(aligned, "close_base", "close_US100", CORR_PERIOD)
    corr_gold  = calc_cross_corr(aligned, "close_base", "close_GOLD",  CORR_PERIOD)
    rel_nd     = calc_relative_strength(aligned, "close_base", "close_US100", CORR_PERIOD)

    feats = pd.DataFrame(index=df.index)

    # ── Price position ────────────────────────────────────────────────────────
    feats["dist_ema9_atr"]   = (df["close"] - df["ema_9"])   / df["atr_14"]
    feats["dist_ema21_atr"]  = (df["close"] - df["ema_21"])  / df["atr_14"]
    feats["dist_ema50_atr"]  = (df["close"] - df["ema_50"])  / df["atr_14"]
    feats["dist_ema200_atr"] = (df["close"] - df["ema_200"]) / df["atr_14"]
    feats["dist_vwap_atr"]   = (df["close"] - df["vwap"])    / df["atr_14"]

    # ── Momentum ─────────────────────────────────────────────────────────────
    feats["rsi_14"]    = df["rsi_14"]
    feats["macd_hist"] = df["macd_hist"]
    feats["bb_pct"]    = df["bb_pct"]

    # ── Volatility ───────────────────────────────────────────────────────────
    feats["atr_pct"]    = df["atr_14"] / df["close"] * 100      # relative ATR
    feats["rvol_ratio"] = df["rvol_20"] / df["rvol_60"]          # <1 = vol contracting

    # ── Cross-instrument ─────────────────────────────────────────────────────
    feats["corr_us100"] = corr_us100.values
    feats["corr_gold"]  = corr_gold.values
    feats["rel_str_nd"] = rel_nd.values

    # ── Time (cyclical encoding) ──────────────────────────────────────────────
    mins = df["time"].dt.hour * 60 + df["time"].dt.minute
    feats["hour_sin"]      = np.sin(2 * np.pi * mins / 1440)
    feats["hour_cos"]      = np.cos(2 * np.pi * mins / 1440)
    feats["day_of_week"]   = df["time"].dt.dayofweek.astype(float)
    feats["is_us_session"] = (
        ((df["time"].dt.hour * 60 + df["time"].dt.minute) >= 870) &   # 14:30 UTC
        ((df["time"].dt.hour * 60 + df["time"].dt.minute) <  1230)    # 20:30 UTC
    ).astype(float)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_ma = df["volume"].rolling(20).mean()
    feats["rel_volume"] = df["volume"] / vol_ma

    feats["time"] = df["time"].values
    return feats
