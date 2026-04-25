"""
XGBoost classifier with walk-forward (expanding-window) validation.

Walk-forward schedule on N usable bars:
  Fold 1: train [0 : 40%], test [40% : 55%]
  Fold 2: train [0 : 55%], test [55% : 70%]
  Fold 3: train [0 : 70%], test [70% : 85%]
  Hold-out: [85% : 100%]  ← touched ONLY at the very end

Each fold uses predict_proba → we pick an entry threshold on the validation
folds, then apply that threshold to trades via the backtester.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import roc_auc_score
import xgboost as xgb

from features import FEATURE_COLS
from backtest import simulate, metrics, print_metrics, SPREAD_PTS

FOLDS = [0.40, 0.55, 0.70, 0.85]   # train-end breakpoints (as fraction of data)
HOLDOUT_START = 0.85                  # hold-out begins here

XGB_PARAMS = dict(
    n_estimators      = 500,
    max_depth         = 3,       # shallower = less overfit on noisy data
    learning_rate     = 0.02,
    subsample         = 0.7,
    colsample_bytree  = 0.7,
    min_child_weight  = 20,      # requires more samples per leaf
    gamma             = 2.0,     # higher = more conservative splits
    reg_lambda        = 5.0,     # stronger L2 regularisation
    scale_pos_weight  = 3.0,     # compensate for class imbalance (~25% positives)
    eval_metric       = "auc",
    random_state      = 42,
    n_jobs            = -1,
)


def _split_indices(n: int):
    """Return list of (train_end, val_start, val_end) index tuples."""
    splits = []
    breakpoints = [int(f * n) for f in FOLDS]
    for k, train_end in enumerate(breakpoints[:-1]):
        val_end = breakpoints[k + 1]
        splits.append((train_end, train_end, val_end))
    holdout = int(HOLDOUT_START * n)
    return splits, holdout


def walk_forward(
    feats: pd.DataFrame,
    labels: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    atr: np.ndarray,
    times: np.ndarray,
    entry_threshold: float = 0.55,
) -> dict:
    """
    Run walk-forward XGBoost.  Returns a dict with:
      oof_probs, fold_aucs, importances,
      val_trades (list of trade DataFrames per fold),
      holdout_trades, holdout_metrics
    """
    X = feats[FEATURE_COLS].values
    y = labels

    n = len(X)
    splits, holdout_idx = _split_indices(n)

    oof_probs   = np.full(n, np.nan)
    fold_aucs   = []
    importances = np.zeros(len(FEATURE_COLS))
    val_trades_list = []

    print("\n── Walk-forward training ─────────────────────────────")
    for fold_num, (train_end, val_start, val_end) in enumerate(splits):
        X_tr, y_tr = X[:train_end],        y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        # Need enough positive labels to train
        if y_tr.sum() < 100 or y_val.sum() < 20:
            print(f"  Fold {fold_num+1}: skipped (too few positives)")
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_start:val_end] = probs

        auc = roc_auc_score(y_val, probs)
        fold_aucs.append(auc)
        importances += model.feature_importances_

        # Simulate trades on this validation fold
        signal_mask = (probs >= entry_threshold)
        # pad to full-length arrays for the val slice
        trades = simulate(
            signal_mask,
            y_val,
            tp_mult, sl_mult,
            atr[val_start:val_end],
            times[val_start:val_end],
            SPREAD_PTS,
        )
        val_trades_list.append(trades)

        m = metrics(trades)
        print(
            f"  Fold {fold_num+1}: AUC={auc:.4f} | "
            f"trades={m['n_trades']} | WR={m['win_rate']:.1%} | "
            f"Sharpe={m['sharpe']:.2f} | PnL={m['total_pnl']:+.1f}pts"
        )

    if fold_aucs:
        importances /= len(fold_aucs)

    # ── Final model on everything up to holdout, evaluate on holdout ─────────
    print("\n── Hold-out evaluation ───────────────────────────────")
    X_train_full = X[:holdout_idx]
    y_train_full = y[:holdout_idx]
    X_hold       = X[holdout_idx:]
    y_hold       = y[holdout_idx:]

    holdout_trades = pd.DataFrame()
    holdout_m      = {}

    if y_train_full.sum() >= 100 and y_hold.sum() >= 10:
        final_model = xgb.XGBClassifier(**XGB_PARAMS)
        final_model.fit(X_train_full, y_train_full, verbose=False)
        hold_probs = final_model.predict_proba(X_hold)[:, 1]

        hold_auc = roc_auc_score(y_hold, hold_probs)
        print(f"  Hold-out AUC: {hold_auc:.4f}")

        h_signals = (hold_probs >= entry_threshold)
        holdout_trades = simulate(
            h_signals, y_hold, tp_mult, sl_mult,
            atr[holdout_idx:], times[holdout_idx:], SPREAD_PTS,
        )
        holdout_m = metrics(holdout_trades)
        print_metrics(holdout_m, "Hold-out trades")
    else:
        print("  Not enough data for hold-out evaluation.")

    return {
        "oof_probs":       oof_probs,
        "fold_aucs":       fold_aucs,
        "importances":     importances,
        "val_trades":      val_trades_list,
        "holdout_trades":  holdout_trades,
        "holdout_metrics": holdout_m,
        "feature_names":   FEATURE_COLS,
        "final_model":     final_model if y_train_full.sum() >= 100 else None,
    }


def plot_results(results: dict, baseline_trades: pd.DataFrame):
    """Two-panel figure: feature importance + equity comparison."""
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Feature importance ───────────────────────────────────────────────────
    ax_imp = fig.add_subplot(gs[:, 0])
    names  = results["feature_names"]
    imps   = results["importances"]
    order  = np.argsort(imps)

    colors = ["#26a69a" if imps[i] > np.median(imps) else "#607080" for i in order]
    ax_imp.barh([names[i] for i in order], imps[order], color=colors)
    ax_imp.set_xlabel("Mean feature importance (XGBoost gain)")
    ax_imp.set_title("Feature importance")
    ax_imp.grid(True, axis="x", alpha=0.2)

    # ── OOF cumulative equity ─────────────────────────────────────────────────
    ax_oof = fig.add_subplot(gs[0, 1])
    all_val = pd.concat(results["val_trades"]).sort_values("bar_idx").reset_index(drop=True)
    if len(all_val):
        cum_model = all_val["pnl"].cumsum()
        x = all_val["time"] if "time" in all_val.columns else all_val["bar_idx"]
        ax_oof.plot(x, cum_model, color="#40a0f0", lw=1.5, label="XGB (OOF)")

    if len(baseline_trades):
        cum_base = baseline_trades["pnl"].cumsum()
        xb = baseline_trades["time"] if "time" in baseline_trades.columns else baseline_trades["bar_idx"]
        ax_oof.plot(xb, cum_base, color="#f0c040", lw=1, alpha=0.7, label="Baseline")

    ax_oof.axhline(0, color="#888888", lw=0.7, ls="--")
    ax_oof.set_title("Validation folds — cumulative P&L (pts)")
    ax_oof.set_ylabel("Cumulative P&L (pts)")
    ax_oof.legend(fontsize=8)
    ax_oof.grid(True, alpha=0.2)
    plt.setp(ax_oof.get_xticklabels(), rotation=20, ha="right", fontsize=7)

    # ── Hold-out equity ───────────────────────────────────────────────────────
    ax_hold = fig.add_subplot(gs[1, 1])
    hold_trades = results["holdout_trades"]
    if len(hold_trades):
        cum_h = hold_trades["pnl"].cumsum()
        xh = hold_trades["time"] if "time" in hold_trades.columns else hold_trades["bar_idx"]
        ax_hold.plot(xh, cum_h, color="#40a0f0", lw=1.5, label="XGB hold-out")
        ax_hold.fill_between(xh, cum_h, 0,
                             where=(cum_h >= 0), color="#26a69a", alpha=0.15)
        ax_hold.fill_between(xh, cum_h, 0,
                             where=(cum_h <  0), color="#ef5350", alpha=0.15)
    ax_hold.axhline(0, color="#888888", lw=0.7, ls="--")
    ax_hold.set_title("Hold-out equity curve")
    ax_hold.set_ylabel("Cumulative P&L (pts)")
    ax_hold.legend(fontsize=8)
    ax_hold.grid(True, alpha=0.2)
    plt.setp(ax_hold.get_xticklabels(), rotation=20, ha="right", fontsize=7)

    fig.suptitle("XGBoost walk-forward results", fontsize=12)
    return fig
