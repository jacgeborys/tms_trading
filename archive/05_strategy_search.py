"""
05_strategy_search.py — Find which conditions predict winning trades.

DEPENDS ON: 04_build_features.py  (must be run first)
  Loads results/04_features_full.csv — the 80-column feature matrix built by 04.

HOW IT WORKS
------------
1. Loads the pre-built feature matrix (04_features_full.csv).
2. Converts continuous features into ~50 binary yes/no conditions.
   Example: "macd_hist > 0" becomes a 1/0 flag called "macd_positive".
3. Trains a Random Forest to rank which conditions are most predictive.
4. Tests every single condition: win rate, trade count, expected P&L.
5. Tests the most promising two-condition combinations.
6. Saves a ranked strategy table to results/05_strategy_table.csv.

READING THE OUTPUT
------------------
Break-even win rate (TP=SL=1×ATR, spread=0.7 pts) = 58.2%
Any row marked ← PROFITABLE is a candidate strategy to backtest.

Usage:
  python 05_strategy_search.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import cache
from backtest import SPREAD_PTS

MIN_TRADES = 100
TP_MULT    = 1.0
SL_MULT    = 1.0


def breakeven_wr(tp: float, sl: float, spread: float = SPREAD_PTS) -> float:
    return (sl + spread) / (tp + sl + 2 * spread)


def ev_per_trade(wr: float, avg_atr: float, tp: float, sl: float,
                 spread: float = SPREAD_PTS) -> float:
    return wr * tp * avg_atr - (1 - wr) * sl * avg_atr - spread


# ── Binary condition definitions ──────────────────────────────────────────────

def compute_conditions(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the continuous feature columns from 04_build_features into
    binary yes/no flags. Each flag = one candidate entry condition.

    All conditions are for LONG entries only (buying).
    """
    c = pd.DataFrame(index=feat.index)

    # ── Time ──────────────────────────────────────────────────────────────────
    c["time_power_hour"]   = (feat["is_power_hour"]  == 1).astype(int)
    c["time_us_session"]   = (feat["is_us_session"]  == 1).astype(int)
    c["time_us_open"]      = (feat["is_us_open"]     == 1).astype(int)
    c["time_monday"]       = (feat["day_of_week"]    == 0).astype(int)
    c["time_friday"]       = (feat["day_of_week"]    == 4).astype(int)
    c["time_midweek"]      = feat["day_of_week"].between(1, 3).astype(int)

    # ── MACD ──────────────────────────────────────────────────────────────────
    # Is the momentum histogram positive (bulls in control)?
    c["macd_positive"]     = (feat["macd_hist"]       > 0).astype(int)
    # Is the histogram getting bigger (momentum building)?
    c["macd_rising"]       = (feat["macd_slope"]      > 0).astype(int)
    # Did MACD just cross from negative to positive (fresh signal)?
    # bars_since_macd_bull is normalized 0–1; < 0.1 means < 5 bars ago
    c["macd_fresh_bull"]   = (feat["bars_since_macd_bull"] < 0.10).astype(int)
    # Is MACD line above signal line?
    c["macd_abv_signal"]   = (feat["macd_signal_gap"] > 0).astype(int)
    # Is histogram accelerating (momentum of momentum)?
    c["macd_accel_up"]     = (feat["macd_accel"]      > 0).astype(int)
    # Bullish divergence: price went lower but MACD didn't (reversal signal)
    c["macd_div_bull"]     = (feat["macd_div_bull"]   > 0.5).astype(int)

    # ── RSI ───────────────────────────────────────────────────────────────────
    # rsi_norm is RSI/100, so 0.30 = RSI of 30 (oversold)
    c["rsi_oversold"]      = (feat["rsi_norm"]        < 0.30).astype(int)
    c["rsi_weak"]          = (feat["rsi_norm"]        < 0.40).astype(int)
    c["rsi_below_mid"]     = (feat["rsi_norm"]        < 0.50).astype(int)
    c["rsi_above_mid"]     = (feat["rsi_norm"]        > 0.50).astype(int)
    c["rsi_strong"]        = (feat["rsi_norm"]        > 0.60).astype(int)
    c["rsi_rising"]        = (feat["rsi_slope_3bar"]  > 0).astype(int)
    # Was RSI oversold recently (within last ~10 bars / 50 min)?
    c["recent_oversold"]   = (feat["bars_since_oversold"] < 0.20).astype(int)
    c["rsi_div_bull"]      = (feat["rsi_div_bull"]    > 0.5).astype(int)

    # ── EMA / Trend ───────────────────────────────────────────────────────────
    # dist_ema*_atr > 0 means price is above that moving average
    c["above_ema9"]        = (feat["dist_ema9_atr"]   > 0).astype(int)
    c["above_ema21"]       = (feat["dist_ema21_atr"]  > 0).astype(int)
    c["above_ema50"]       = (feat["dist_ema50_atr"]  > 0).astype(int)
    c["above_ema200"]      = (feat["dist_ema200_atr"] > 0).astype(int)
    # ema_alignment: +3 = all EMAs bullish-stacked, -3 = all bearish
    c["ema_bullish"]       = (feat["ema_alignment"]   > 0).astype(int)
    c["ema_all_bull"]      = (feat["ema_alignment"]  >= 2).astype(int)
    # Are the EMAs themselves rising?
    c["ema9_rising"]       = (feat["slope_ema9"]      > 0).astype(int)
    c["ema50_rising"]      = (feat["slope_ema50"]     > 0).astype(int)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    c["above_vwap"]        = (feat["dist_vwap_atr"]   > 0).astype(int)
    c["vwap_rising"]       = (feat["vwap_slope"]      > 0).astype(int)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    c["bb_lower_half"]     = (feat["bb_pct"]          < 0.50).astype(int)
    c["bb_not_overbought"] = (feat["bb_pct"]          < 0.80).astype(int)
    c["bb_squeeze"]        = (feat["bb_squeeze"]      < 0.80).astype(int)
    c["bb_walk_up"]        = (feat["bb_walk_up"]      > 0.5).astype(int)

    # ── Volatility ────────────────────────────────────────────────────────────
    c["vol_contracting"]   = (feat["rvol_ratio"]      < 1.0).astype(int)
    c["normal_vol"]        = (feat["atr_vs_median"]   < 1.5).astype(int)

    # ── Price action / bar structure ──────────────────────────────────────────
    c["green_bar"]         = (feat["bar_direction"]   > 0).astype(int)
    # Strong green bar: body > 0.3 ATR
    c["strong_green"]      = (feat["bar_body_atr"]    > 0.30).astype(int)
    # Consecutive same-direction bars
    c["consec_green_2"]    = (feat["consec_green"]    >= 2).astype(int)
    c["consec_red_2"]      = (feat["consec_red"]      >= 2).astype(int)
    # Long lower wick = buyers absorbed selling pressure (bullish)
    c["big_lower_wick"]    = (feat["lower_wick_pct"]  > 0.40).astype(int)
    # Trend consistency over last 1h: > 60% green bars = uptrend
    c["trend_bull_1h"]     = (feat["trend_consist_1h"] > 0.60).astype(int)

    # ── Position in recent range ──────────────────────────────────────────────
    # pos_in_range20: 0 = at 20-bar low, 1 = at 20-bar high
    c["near_range_low"]    = (feat["pos_in_range20"]  < 0.30).astype(int)
    c["near_range_high"]   = (feat["pos_in_range20"]  > 0.70).astype(int)
    c["mid_range"]         = feat["pos_in_range20"].between(0.30, 0.70).astype(int)
    # pos_in_sess_range: same but for today's session
    c["near_session_low"]  = (feat["pos_in_sess_range"] < 0.20).astype(int)
    c["near_session_high"] = (feat["pos_in_sess_range"] > 0.80).astype(int)

    # ── Previous day / gap ────────────────────────────────────────────────────
    c["above_prev_close"]  = (feat["above_prev_close"] > 0).astype(int)
    c["gap_up"]            = (feat["gap_atr"]           > 0.20).astype(int)
    c["prev_bull_day"]     = (feat["prev_day_return"]   > 0).astype(int)

    # ── Opening range breakout ────────────────────────────────────────────────
    # above_opening_range > 0 means price is above the first 30-min high
    c["above_open_range"]  = (feat["above_opening_range"] > 0).astype(int)

    # ── Multi-timeframe momentum ──────────────────────────────────────────────
    c["1h_momentum_up"]    = (feat["ret_1h"]  > 0).astype(int)
    c["4h_momentum_up"]    = (feat["ret_4h"]  > 0).astype(int)
    # Slight pullback after an upward move = mean-reversion entry
    c["30m_pullback"]      = (feat["ret_30m"] < -0.20).astype(int)

    # ── Volume ────────────────────────────────────────────────────────────────
    c["high_vol_bar"]      = (feat["rel_volume"]  > 1.5).astype(int)
    c["vol_trending_up"]   = (feat["vol_trend"]   > 1.2).astype(int)

    # Drop any columns with NaN from features (can happen on edge rows)
    for col in c.columns:
        c[col] = c[col].fillna(0).astype(int)

    return c


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_single(cond_df: pd.DataFrame, feat: pd.DataFrame,
                   labels: np.ndarray, tp: float, sl: float) -> pd.DataFrame:
    atr        = feat["atr_14"].values
    be_wr      = breakeven_wr(tp, sl)
    valid_mask = labels >= 0
    base_wr    = float(labels[valid_mask].mean())
    rows = []

    for col in cond_df.columns:
        mask = (cond_df[col].values == 1) & valid_mask
        n    = int(mask.sum())
        if n < MIN_TRADES:
            continue
        wr      = float(labels[mask].mean())
        avg_atr = float(atr[mask].mean())
        ev      = ev_per_trade(wr, avg_atr, tp, sl)
        rows.append({
            "conditions": col,
            "n_trades":   n,
            "win_rate":   round(wr, 4),
            "ev_pts":     round(ev, 3),
            "vs_base":    round(wr - base_wr, 4),
            "profitable": wr > be_wr,
        })

    return pd.DataFrame(rows).sort_values("win_rate", ascending=False)


def analyze_combinations(cond_df: pd.DataFrame, feat: pd.DataFrame,
                          labels: np.ndarray, top_cols: list,
                          tp: float, sl: float) -> pd.DataFrame:
    atr        = feat["atr_14"].values
    be_wr      = breakeven_wr(tp, sl)
    valid_mask = labels >= 0
    base_wr    = float(labels[valid_mask].mean())
    rows = []

    for i, col_a in enumerate(top_cols):
        for col_b in top_cols[i + 1:]:
            mask = ((cond_df[col_a].values == 1) &
                    (cond_df[col_b].values == 1) & valid_mask)
            n = int(mask.sum())
            if n < MIN_TRADES:
                continue
            wr      = float(labels[mask].mean())
            avg_atr = float(atr[mask].mean())
            ev      = ev_per_trade(wr, avg_atr, tp, sl)
            rows.append({
                "conditions": f"{col_a}  +  {col_b}",
                "n_trades":   n,
                "win_rate":   round(wr, 4),
                "ev_pts":     round(ev, 3),
                "vs_base":    round(wr - base_wr, 4),
                "profitable": wr > be_wr,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False)


def rf_importance(cond_df: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    valid = labels >= 0
    X     = cond_df[valid].values.astype(np.float32)
    y     = labels[valid]
    split = int(len(X) * 0.70)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=30,
        n_jobs=-1, random_state=42
    )
    rf.fit(X[:split], y[:split])

    try:
        proba = rf.predict_proba(X[split:])[:, 1]
        auc   = roc_auc_score(y[split:], proba)
    except Exception:
        auc = 0.5

    print(f"  Random Forest validation AUC: {auc:.3f}")
    if auc < 0.53:
        print("  (AUC near 0.50 — no single condition predicts direction cleanly.")
        print("   Feature importances still rank which conditions matter most.)")

    return pd.Series(
        rf.feature_importances_, index=cond_df.columns
    ).sort_values(ascending=False)


def plot_importance(importances: pd.Series, be_wr: float,
                    single_df: pd.DataFrame) -> plt.Figure:
    plt.style.use("dark_background")
    profitable_conds = set(single_df[single_df["profitable"]]["conditions"])
    top = importances.head(25)
    colors = ["#00e676" if c in profitable_conds else "#2196F3" for c in top.index]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Feature Importance (Random Forest)", fontsize=9)
    ax.set_title(
        f"Condition importance — US500 M5 long trades\n"
        f"green = WR > break-even ({be_wr:.1%})   blue = important but not profitable alone",
        fontsize=9
    )
    ax.grid(True, axis="x", alpha=0.2)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load pre-built feature matrix from 04_build_features.py
    feat_path = cache.RESULTS / "04_features_full.csv"
    if not feat_path.exists():
        print(f"Feature file not found: {feat_path}")
        print("Run 04_build_features.py first.")
        return

    print(f"Loading feature matrix from {feat_path.name}...")
    feat = pd.read_csv(feat_path)

    # If 04 was generated before the column-name fix, day_of_week appears twice
    # (once as "Mon"/"Tue" string from meta, once as 0-4 float from features).
    # Deduplicate: keep the LAST occurrence, which is the numeric feature column.
    if feat.columns.duplicated().any():
        dupes = feat.columns[feat.columns.duplicated(keep=False)].unique().tolist()
        print(f"  Deduplicating columns: {dupes} — keeping numeric versions")
        feat = feat.loc[:, ~feat.columns.duplicated(keep="last")]

    # Ensure columns used as numbers are actually numeric (guards against string bleed)
    for col in ["day_of_week", "is_power_hour", "is_us_session", "is_us_open",
                "label_long", "label_short"]:
        if col in feat.columns:
            feat[col] = pd.to_numeric(feat[col], errors="coerce")

    print(f"  {len(feat):,} bars × {len(feat.columns)} columns")

    # Labels are embedded in the feature file
    if "label_long" not in feat.columns:
        print("ERROR: label_long column missing. Regenerate with 04_build_features.py.")
        return

    labels  = feat["label_long"].values
    valid   = labels >= 0
    base_wr = float(labels[valid].mean())
    be_wr   = breakeven_wr(TP_MULT, SL_MULT)

    print(f"  Labelled bars  : {valid.sum():,}")
    print(f"  Base win rate  : {base_wr:.1%}  (all bars, no filter)")
    print(f"  Break-even WR  : {be_wr:.1%}  (minimum to profit after spread)")
    print(f"  Gap to target  : {base_wr - be_wr:+.1%}")

    # Build binary conditions from features
    print(f"\nComputing binary conditions from features...")
    cond_df = compute_conditions(feat)
    print(f"  {len(cond_df.columns)} conditions built")

    # Random Forest importance ranking
    print("\nTraining Random Forest on conditions...")
    importances = rf_importance(cond_df, labels)

    print("\nTop 20 conditions by RF importance:")
    for cond, imp in importances.head(20).items():
        print(f"  {cond:<30}  {imp:.4f}")

    # Single condition analysis
    print(f"\nTesting each condition individually...")
    single = analyze_single(cond_df, feat, labels, TP_MULT, SL_MULT)

    print(f"\n{'─'*82}")
    print(f"  SINGLE CONDITION RESULTS  (break-even = {be_wr:.1%})")
    print(f"{'─'*82}")
    print(f"  {'Condition':<30}  {'Trades':>7}  {'Win Rate':>9}  {'EV/trade':>9}  {'vs base':>8}")
    print(f"{'─'*82}")
    for _, row in single.iterrows():
        flag = "  ← PROFITABLE" if row["profitable"] else ""
        print(f"  {row['conditions']:<30}  {row['n_trades']:>7,}  "
              f"{row['win_rate']:>9.1%}  {row['ev_pts']:>+9.2f}  "
              f"{row['vs_base']:>+8.1%}{flag}")

    # Two-condition combinations using top 15 by RF importance
    top_conds = list(importances.head(15).index)
    n_combos  = len(top_conds) * (len(top_conds) - 1) // 2
    print(f"\nTesting {n_combos} two-condition combinations (top 15 × top 15)...")
    combos = analyze_combinations(cond_df, feat, labels, top_conds, TP_MULT, SL_MULT)

    if len(combos):
        print(f"\n{'─'*92}")
        print(f"  TOP TWO-CONDITION COMBINATIONS")
        print(f"{'─'*92}")
        print(f"  {'Conditions':<54}  {'Trades':>7}  {'Win Rate':>9}  {'EV/trade':>9}  {'vs base':>8}")
        print(f"{'─'*92}")
        for _, row in combos.head(30).iterrows():
            flag = "  ← PROFITABLE" if row["profitable"] else ""
            print(f"  {row['conditions']:<54}  {row['n_trades']:>7,}  "
                  f"{row['win_rate']:>9.1%}  {row['ev_pts']:>+9.2f}  "
                  f"{row['vs_base']:>+8.1%}{flag}")

    # Save
    all_results = pd.concat(
        [single, combos] if len(combos) else [single],
        ignore_index=True
    )
    out_path = cache.RESULTS / "05_strategy_table.csv"
    all_results.to_csv(out_path, index=False)

    profitable = all_results[all_results["profitable"]]
    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  Conditions tested   : {len(all_results):,}")
    print(f"  Profitable (>{be_wr:.1%}) : {len(profitable):,}")
    if len(profitable):
        print(f"\n  PROFITABLE CANDIDATES (open 05_strategy_table.csv for full list):")
        for _, row in profitable.sort_values("win_rate", ascending=False).head(10).iterrows():
            print(f"    {row['win_rate']:.1%}  {row['n_trades']:,} trades  {row['conditions']}")
    else:
        best = all_results.iloc[0]
        print(f"  No single/two-condition combo clears {be_wr:.1%} yet.")
        print(f"  Closest: {best['win_rate']:.1%}  {best['n_trades']:,} trades  {best['conditions']}")
    print(f"{'═'*60}")
    print(f"\nFull table → {out_path}")

    fig = plot_importance(importances, be_wr, single)
    cache.save_chart(fig, "05_strategy_importance")
    plt.show()


if __name__ == "__main__":
    main()
