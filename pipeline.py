"""
Full pipeline — runs all steps, caches data, saves all outputs.

Folder structure after a run:
  data/
    US500_pro_M5.pkl          raw OHLCV (re-fetched if >12h old)
    US100_pro_M5.pkl          ...same for cross instruments
    US30_pro_M5.pkl
    GOLD_pro_M5.pkl
    features_latest.pkl       feature matrix (re-built each run)
  models/
    xgb_final.pkl             trained XGBoost (final model on train+val set)
  results/
    baseline_metrics.json
    xgb_holdout_metrics.json
    baseline_trades.csv
    xgb_holdout_trades.csv
    runs.json                 cumulative log of all pipeline runs
    charts/
      heatmap_long.png
      baseline_equity.png
      model_results.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cache
from mt5_client import connect, disconnect
from data import get_candles
from indicators import add_all
from features import build_features, FEATURE_COLS
from labeler import make_long_labels, make_short_labels
from heatmap_search import compute_grid, best_params, plot_grid
from baseline import run as run_baseline, plot_equity as plot_baseline_equity
from backtest import print_metrics, metrics
from model import walk_forward, plot_results

SYMBOL = "US500.pro"
TF     = "M5"
N_BARS = 50_000

CROSS = {
    "US100": "US100.pro",
    "US30":  "US30.pro",
    "GOLD":  "GOLD.pro",
}

CACHE_MAX_AGE_H = 12.0   # re-fetch from MT5 if data is older than this


# ─────────────────────────────────────────────────────────────────────────────
def fetch_with_cache(symbol: str, name: str, timeframe: str, n: int) -> pd.DataFrame:
    """Return cached OHLCV if fresh, otherwise fetch from MT5 and save."""
    if cache.ohlcv_is_fresh(symbol, timeframe, CACHE_MAX_AGE_H):
        df = cache.load_ohlcv(symbol, timeframe)
        if df is not None:
            return df
    df = get_candles(n=n, symbol=symbol, timeframe=timeframe)
    cache.save_ohlcv(df, symbol, timeframe)
    return df


def main():
    plt.style.use("dark_background")

    # ── Step 1: Fetch / load from cache ──────────────────────────────────────
    all_fresh = all(
        cache.ohlcv_is_fresh(sym, TF, CACHE_MAX_AGE_H)
        for sym in [SYMBOL] + list(CROSS.values())
    )

    if not all_fresh:
        print("\n[1/6] Connecting to MT5 to fetch fresh data...")
        if not connect():
            return
        try:
            base_df   = fetch_with_cache(SYMBOL, "US500", TF, N_BARS)
            other_dfs = {
                name: fetch_with_cache(sym, name, TF, N_BARS)
                for name, sym in CROSS.items()
            }
        finally:
            disconnect()
    else:
        print("\n[1/6] All data fresh in cache — skipping MT5 connection.")
        base_df   = cache.load_ohlcv(SYMBOL, TF)
        other_dfs = {name: cache.load_ohlcv(sym, TF) for name, sym in CROSS.items()}

    print(f"      {len(base_df):,} bars  "
          f"({str(base_df['time'].iloc[0])[:10]} → {str(base_df['time'].iloc[-1])[:10]})")

    # ── Step 2: Features ──────────────────────────────────────────────────────
    print("\n[2/6] Computing indicators & features...")
    df_ind = add_all(base_df)
    feats  = build_features(base_df, other_dfs)

    valid_rows = feats[FEATURE_COLS].notna().all(axis=1)
    df_ind = df_ind[valid_rows].reset_index(drop=True)
    feats  = feats[valid_rows].reset_index(drop=True)
    print(f"      Usable bars after warm-up: {len(df_ind):,}")

    cache.save_features(feats, tag="latest")

    # ── Step 3: Heatmap grid search (training set = first 80%) ───────────────
    print("\n[3/6] Running label parameter grid search (training set only)...")
    train_end = int(len(df_ind) * 0.80)
    df_train  = df_ind.iloc[:train_end]

    grid_long  = compute_grid(df_train, direction="long")
    grid_short = compute_grid(df_train, direction="short")

    tp_opt, sl_opt, la_opt = best_params(grid_long)
    print(f"      Best params: TP={tp_opt}×ATR  SL={sl_opt}×ATR  N={la_opt} bars ({la_opt*5}min)")

    fig_heat = plot_grid(grid_long, direction="long")
    fig_heat.suptitle(
        f"Grid search — LONG trades — training set only\n"
        f"Best: TP={tp_opt}×ATR  SL={sl_opt}×ATR  N={la_opt} bars",
        fontsize=11,
    )
    cache.save_chart(fig_heat, "heatmap_long")
    plt.show()

    # ── Step 4: Generate labels ───────────────────────────────────────────────
    print(f"\n[4/6] Generating labels  (TP={tp_opt}×ATR  SL={sl_opt}×ATR  N={la_opt})...")
    long_lbl  = make_long_labels( df_ind, tp_mult=tp_opt, sl_mult=sl_opt, lookahead=la_opt)
    short_lbl = make_short_labels(df_ind, tp_mult=tp_opt, sl_mult=sl_opt, lookahead=la_opt)

    valid_long  = long_lbl  >= 0
    valid_short = short_lbl >= 0
    print(f"      LONG  labels: {valid_long.sum():,} valid  |  win rate = {long_lbl[valid_long].mean():.1%}")
    print(f"      SHORT labels: {valid_short.sum():,} valid  |  win rate = {short_lbl[valid_short].mean():.1%}")
    random_wr = sl_opt / (tp_opt + sl_opt)
    print(f"      Random-walk baseline win rate: {random_wr:.1%}")

    # ── Step 5: Baseline ──────────────────────────────────────────────────────
    print("\n[5/6] Running rule-based baseline strategy...")
    b_trades, b_long_m, b_short_m, b_all_m = run_baseline(
        df_ind, long_lbl, short_lbl, tp_opt, sl_opt
    )
    print_metrics(b_long_m,  "Baseline — LONG")
    print_metrics(b_short_m, "Baseline — SHORT")
    print_metrics(b_all_m,   "Baseline — ALL")

    cache.save_metrics(b_long_m,  "baseline_long_metrics")
    cache.save_metrics(b_short_m, "baseline_short_metrics")
    cache.save_metrics(b_all_m,   "baseline_all_metrics")
    cache.save_trades(b_trades,   "baseline_trades")

    fig_base, ax_base = plt.subplots(figsize=(14, 4))
    fig_base.patch.set_facecolor("#1a1a2e")
    ax_base.set_facecolor("#1a1a2e")
    plot_baseline_equity(b_trades, ax=ax_base, label="Baseline (long + short)")
    ax_base.set_title("Baseline equity curve — full dataset")
    plt.tight_layout()
    cache.save_chart(fig_base, "baseline_equity")
    plt.show()

    # ── Step 6: XGBoost walk-forward ─────────────────────────────────────────
    print("\n[6/6] Walk-forward XGBoost...")
    usable = valid_long & valid_short
    X_df   = feats[usable].reset_index(drop=True)
    y_long = long_lbl[usable]
    atr_   = df_ind["atr_14"].values[usable]
    times_ = df_ind["time"].values[usable]

    results = walk_forward(
        feats   = X_df,
        labels  = y_long,
        tp_mult = tp_opt,
        sl_mult = sl_opt,
        atr     = atr_,
        times   = times_,
        entry_threshold = 0.52,
    )

    cache.save_metrics(results.get("holdout_metrics", {}), "xgb_holdout_metrics")
    cache.save_trades(results.get("holdout_trades", pd.DataFrame()), "xgb_holdout_trades")

    if results.get("final_model") is not None:
        cache.save_model(results["final_model"], "xgb_final")

    # ── Step 7: Comparison plot ───────────────────────────────────────────────
    b_usable = b_trades[b_trades["bar_idx"] < int(usable.sum())] if len(b_trades) else b_trades
    fig_model = plot_results(results, b_usable)
    cache.save_chart(fig_model, "model_results")
    plt.show()

    # ── Run summary ───────────────────────────────────────────────────────────
    run_params = {
        "symbol": SYMBOL, "timeframe": TF, "n_bars": len(df_ind),
        "tp_mult": tp_opt, "sl_mult": sl_opt, "lookahead": la_opt,
    }
    all_metrics = {
        "baseline_sharpe":    b_all_m.get("sharpe"),
        "baseline_win_rate":  b_all_m.get("win_rate"),
        "xgb_oof_auc":        float(np.mean(results["fold_aucs"])) if results["fold_aucs"] else None,
        "xgb_holdout_sharpe": results.get("holdout_metrics", {}).get("sharpe"),
    }
    cache.save_run_summary(run_params, all_metrics)

    # ── Final print ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print("═" * 60)
    print(f"  Data          : {len(df_ind):,} {TF} bars")
    print(f"  Best params   : TP={tp_opt}×ATR  SL={sl_opt}×ATR  N={la_opt} bars")
    print(f"  Random WR     : {random_wr:.1%}")
    print(f"  Baseline WR   : {b_all_m.get('win_rate', 0):.1%}  "
          f"Sharpe={b_all_m.get('sharpe', 0):.2f}")
    if results["fold_aucs"]:
        print(f"  XGB OOF AUC   : {np.mean(results['fold_aucs']):.4f}  "
              f"folds={[f'{a:.3f}' for a in results['fold_aucs']]}")
    hm = results.get("holdout_metrics", {})
    print(f"  XGB hold-out  : Sharpe={hm.get('sharpe', 0):.2f}  "
          f"WR={hm.get('win_rate', 0):.1%}")
    print("═" * 60)
    print("\n  Outputs saved:")
    print("    data/           ← OHLCV + feature matrix")
    print("    models/         ← XGBoost model")
    print("    results/        ← metrics JSON + trade CSVs")
    print("    results/charts/ ← PNG plots")


if __name__ == "__main__":
    main()
