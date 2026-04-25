"""
Grid search over (TP_mult, SL_mult, lookahead) to find which label parameters
have genuine expected value on the training set.

Metric: Expected Value per trade in ATR units = win_rate * TP - (1-win_rate) * SL
Positive EV = edge exists at that parameter combination.

The heatmap shows EV for each lookahead, so we can pick a stable region
(consistent positive EV, not a single hot cell).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from labeler import make_long_labels, make_short_labels

TP_MULTS   = [0.5, 0.75, 1.0, 1.25, 1.5]
SL_MULTS   = [0.5, 0.75, 1.0, 1.25]
LOOKAHEADS = [6, 12, 24, 48]    # 30min / 1h / 2h / 4h on M5


def expected_value(win_rate: float, tp_mult: float, sl_mult: float) -> float:
    return win_rate * tp_mult - (1.0 - win_rate) * sl_mult


def _session_mask(df: pd.DataFrame) -> np.ndarray:
    mins = df["time"].dt.hour * 60 + df["time"].dt.minute
    return ((mins >= 870) & (mins < 1230)).values   # 14:30–20:30 UTC


def compute_grid(df: pd.DataFrame, direction: str = "long") -> dict:
    """
    Compute win_rate and EV for every (TP, SL, lookahead) combination.
    Only considers bars that fall within the US trading session.

    Returns dict:  lookahead → {"win_rate": 2D array, "ev": 2D array}
                   rows = SL_MULTS, cols = TP_MULTS
    """
    label_fn = make_long_labels if direction == "long" else make_short_labels
    sess = _session_mask(df)
    results = {}

    for la in LOOKAHEADS:
        grid_wr = np.full((len(SL_MULTS), len(TP_MULTS)), np.nan)
        grid_ev = np.full((len(SL_MULTS), len(TP_MULTS)), np.nan)

        for j, tp in enumerate(TP_MULTS):
            for i, sl in enumerate(SL_MULTS):
                labels = label_fn(df, tp_mult=tp, sl_mult=sl, lookahead=la)
                valid  = sess & (labels >= 0)
                if valid.sum() < 50:
                    continue
                wr = labels[valid].mean()
                grid_wr[i, j] = wr
                grid_ev[i, j] = expected_value(wr, tp, sl)

        results[la] = {"win_rate": grid_wr, "ev": grid_ev}

    return results


def best_params(results: dict) -> tuple:
    """
    Pick the (TP_mult, SL_mult, lookahead) with the highest EV that is also
    meaningfully above the theoretical random-walk win rate.

    Random-walk win rate for a given (TP, SL) = SL / (TP + SL).
    We require the observed win rate to exceed random by at least MIN_EDGE.
    Among qualifying combos, pick by highest EV.
    """
    MIN_EDGE   = 0.02   # observed WR must beat random by at least 2 pp
    best_ev    = -np.inf
    best_combo = (1.0, 0.75, 24)   # sensible fallback

    for la, grids in results.items():
        wr = grids["win_rate"]
        ev = grids["ev"]
        for i, sl in enumerate(SL_MULTS):
            for j, tp in enumerate(TP_MULTS):
                v  = ev[i, j]
                w  = wr[i, j]
                if np.isnan(v) or np.isnan(w):
                    continue
                random_wr = sl / (tp + sl)
                if (w - random_wr) < MIN_EDGE:
                    continue          # not enough edge above random
                if v > best_ev:
                    best_ev    = v
                    best_combo = (tp, sl, la)

    if best_ev == -np.inf:
        # Nothing cleared MIN_EDGE — report it and use fallback
        print("      WARNING: No combo cleared the min-edge threshold. "
              "Using fallback TP=1.0, SL=0.75, N=24.")
    return best_combo


def plot_grid(results: dict, direction: str = "long"):
    n_la  = len(LOOKAHEADS)
    tp_lb = [str(v) for v in TP_MULTS]
    sl_lb = [str(v) for v in SL_MULTS]

    fig, axes = plt.subplots(2, n_la, figsize=(5 * n_la, 9))
    fig.suptitle(
        f"Label grid search — {direction.upper()} trades — US session bars\n"
        f"Rows=SL mult, Cols=TP mult",
        fontsize=11,
    )

    for col, la in enumerate(LOOKAHEADS):
        wr = results[la]["win_rate"]
        ev = results[la]["ev"]

        ax_wr = axes[0, col]
        ax_ev = axes[1, col]

        sns.heatmap(wr, ax=ax_wr, xticklabels=tp_lb, yticklabels=sl_lb,
                    annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0.30, vmax=0.70, cbar=col == n_la - 1)
        ax_wr.set_title(f"Win rate  N={la} ({la*5}min)")
        ax_wr.set_xlabel("TP mult (×ATR)")
        ax_wr.set_ylabel("SL mult (×ATR)")

        sns.heatmap(ev, ax=ax_ev, xticklabels=tp_lb, yticklabels=sl_lb,
                    annot=True, fmt=".3f", cmap="RdYlGn",
                    center=0, cbar=col == n_la - 1)
        ax_ev.set_title(f"Expected value (ATR units)")
        ax_ev.set_xlabel("TP mult (×ATR)")
        ax_ev.set_ylabel("SL mult (×ATR)")

    plt.tight_layout()
    return fig
