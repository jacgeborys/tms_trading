"""
ATR-based binary label generation.

For each bar i:
  label_long[i]  = 1  →  entering LONG  hits TP before SL within `lookahead` bars
  label_short[i] = 1  →  entering SHORT hits TP before SL within `lookahead` bars
  value  = 0  →  SL hit first, timeout, or both at same bar (pessimistic tie-break)
  value  = -1 →  undefined (last `lookahead` bars — exclude from training)

All computation is vectorized via numpy sliding-window views.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _compute_labels(
    closes: np.ndarray,
    tp_side: np.ndarray,   # high (long) or low (short)
    sl_side: np.ndarray,   # low  (long) or high (short)
    tp_level: np.ndarray,  # entry + tp_mult*ATR  (long) or entry - tp_mult*ATR (short)
    sl_level: np.ndarray,  # entry - sl_mult*ATR  (long) or entry + sl_mult*ATR (short)
    tp_cmp,                # np.greater_equal (long) or np.less_equal (short)
    sl_cmp,                # np.less_equal    (long) or np.greater_equal (short)
    lookahead: int,
) -> np.ndarray:
    n = len(closes)
    m = n - lookahead
    if m <= 0:
        return np.full(n, -1, dtype=np.int8)

    horizon_tp = sliding_window_view(tp_side, lookahead)[1 : m + 1]  # (m, la)
    horizon_sl = sliding_window_view(sl_side, lookahead)[1 : m + 1]  # (m, la)

    tp_lvl = tp_level[:m, None]  # (m, 1)
    sl_lvl = sl_level[:m, None]  # (m, 1)

    tp_hit = tp_cmp(horizon_tp, tp_lvl)  # (m, la)
    sl_hit = sl_cmp(horizon_sl, sl_lvl)  # (m, la)

    any_hit  = tp_hit | sl_hit
    has_hit  = any_hit.any(axis=1)
    first_j  = np.argmax(any_hit, axis=1)

    rows = np.arange(m)
    # TP wins only if TP hit at first_j AND SL not also hit at that same bar
    tp_first = tp_hit[rows, first_j] & ~sl_hit[rows, first_j]

    labels = np.full(n, -1, dtype=np.int8)
    labels[:m] = np.where(has_hit & tp_first, 1, 0)
    return labels


def make_long_labels(
    df: pd.DataFrame,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    lookahead: int = 48,
    atr_col: str = "atr_14",
) -> np.ndarray:
    closes = df["close"].values
    atrs   = df[atr_col].values
    return _compute_labels(
        closes,
        tp_side  = df["high"].values,
        sl_side  = df["low"].values,
        tp_level = closes + tp_mult * atrs,
        sl_level = closes - sl_mult * atrs,
        tp_cmp   = np.greater_equal,
        sl_cmp   = np.less_equal,
        lookahead = lookahead,
    )


def make_short_labels(
    df: pd.DataFrame,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    lookahead: int = 48,
    atr_col: str = "atr_14",
) -> np.ndarray:
    closes = df["close"].values
    atrs   = df[atr_col].values
    return _compute_labels(
        closes,
        tp_side  = df["low"].values,
        sl_side  = df["high"].values,
        tp_level = closes - tp_mult * atrs,
        sl_level = closes + sl_mult * atrs,
        tp_cmp   = np.less_equal,
        sl_cmp   = np.greater_equal,
        lookahead = lookahead,
    )
