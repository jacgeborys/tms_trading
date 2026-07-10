"""
09_build_features.py — Comprehensive feature engineering for US500 M5 bars.

Computes ~80 features per bar that simulate what a professional trader reads
from a chart across multiple timeframes. Features are grouped into categories:

  1.  Multi-timeframe returns    — how far did price move in last 5min / 1h / 4h?
  2.  EMA positions & slopes     — where is price vs each average, and is it rising?
  3.  EMA alignment score        — are all timeframes stacked bullish/bearish?
  4.  MACD dynamics              — slope, acceleration, bars since crossover, divergence
  5.  RSI dynamics               — level, slope, bars since extreme, divergence
  6.  Bollinger Bands            — %B, width, squeeze, band-walking
  7.  VWAP                       — distance, slope
  8.  Volatility regime          — ATR vs its own median, vol contracting/expanding
  9.  Bar structure              — body size, wick ratios, consecutive bars
  10. Swing distance             — how far from 20-bar high/low?
  11. Session context            — position in today's range, session progress
  12. Previous day levels        — gap, above/below yesterday's high/low/close
  13. Volume                     — relative volume, volume trend
  14. Round number distance      — psychological support/resistance at round levels
  15. Opening range              — above/below first 30 min of session (breakout signal)
  16. Trend consistency          — what % of last N bars moved in current direction?
  17. Time                       — cyclical encoding, session flags

OUTPUTS
-------
  results/04_features_full.csv      all bars (50k rows × 90 cols) — use in Excel/analysis
  results/04_features_preview.csv   first 500 rows — quick sanity check

Usage:
  python 09_build_features.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cache
from indicators import add_all
from labeler import make_long_labels, make_short_labels
from backtest import SPREAD_PTS

LOOKAHEAD = 24     # 2 hours on M5
TP_MULT   = 1.0
SL_MULT   = 1.0
WARMUP    = 300    # skip first N bars (features need lookback to warm up)


# ── Helper functions ──────────────────────────────────────────────────────────

def bars_since(condition: np.ndarray, max_val: int = 50) -> np.ndarray:
    """
    How many bars have passed since this condition was True?
    Example: bars_since_macd_cross_bull = 3  →  MACD turned positive 3 bars (15 min) ago.
    Capped at max_val so very old events don't dominate.
    """
    result = np.empty(len(condition), dtype=np.float32)
    last = float(max_val)
    for i in range(len(condition)):
        last = 0.0 if condition[i] else min(last + 1.0, float(max_val))
        result[i] = last
    return result


def count_consecutive(condition: np.ndarray) -> np.ndarray:
    """
    How many consecutive True values ending at each bar?
    [F, T, T, T, F, T] → [0, 1, 2, 3, 0, 1]
    Used to detect momentum runs and exhaustion.
    """
    result = np.zeros(len(condition), dtype=np.float32)
    cnt = 0
    for i in range(len(condition)):
        cnt = cnt + 1 if condition[i] else 0
        result[i] = cnt
    return result


def bullish_divergence(price: np.ndarray, indicator: np.ndarray,
                       lookback: int = 24) -> np.ndarray:
    """
    Classic bullish divergence: price made a lower low in the lookback window
    but the indicator (MACD or RSI) made a HIGHER low.
    This often precedes upward reversals — the selling is exhausting itself.
    Returns 1/0 per bar.
    """
    n = len(price)
    result = np.zeros(n, dtype=np.float32)
    half = lookback // 2
    for i in range(lookback, n):
        pw = price[i - lookback: i + 1]
        iw = indicator[i - lookback: i + 1]
        if pw[:half].min() > pw[half:].min() and iw[:half].min() < iw[half:].min():
            result[i] = 1.0
    return result


def bearish_divergence(price: np.ndarray, indicator: np.ndarray,
                       lookback: int = 24) -> np.ndarray:
    """
    Bearish divergence: price made a higher high but indicator made a lower high.
    Suggests upward momentum is fading — useful to AVOID entering longs.
    Returns 1/0 per bar.
    """
    n = len(price)
    result = np.zeros(n, dtype=np.float32)
    half = lookback // 2
    for i in range(lookback, n):
        pw = price[i - lookback: i + 1]
        iw = indicator[i - lookback: i + 1]
        if pw[:half].max() < pw[half:].max() and iw[:half].max() > iw[half:].max():
            result[i] = 1.0
    return result


def session_cummax(df: pd.DataFrame) -> np.ndarray:
    """Today's running high — resets at UTC midnight each day."""
    tmp = df.copy()
    tmp["_date"] = tmp["time"].dt.date
    return tmp.groupby("_date")["high"].cummax().values


def session_cummin(df: pd.DataFrame) -> np.ndarray:
    """Today's running low — resets at UTC midnight each day."""
    tmp = df.copy()
    tmp["_date"] = tmp["time"].dt.date
    return tmp.groupby("_date")["low"].cummin().values


def session_first_open(df: pd.DataFrame) -> np.ndarray:
    """Today's opening price (first bar's open) — broadcast to all bars of that day."""
    tmp = df.copy()
    tmp["_date"] = tmp["time"].dt.date
    return tmp.groupby("_date")["open"].transform("first").values


def opening_range(df: pd.DataFrame, minutes: int = 30) -> tuple:
    """
    Opening range high and low: the high and low of the first `minutes` of each session.
    Classic day-trading reference level — breakout above/below is a signal.
    """
    bars = minutes // 5
    tmp = df.copy()
    tmp["_date"] = tmp["time"].dt.date
    tmp["_bar_in_day"] = tmp.groupby("_date").cumcount()
    or_high = tmp[tmp["_bar_in_day"] < bars].groupby("_date")["high"].max()
    or_low  = tmp[tmp["_bar_in_day"] < bars].groupby("_date")["low"].min()

    dates = tmp["_date"]
    hi = or_high.reindex(dates).values
    lo = or_low.reindex(dates).values
    return hi.astype(float), lo.astype(float)


def prev_day_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Previous calendar day's open, high, low, close mapped back to every M5 bar.
    Used to compute gap, above/below yesterday's key levels, etc.
    """
    dates = df["time"].dt.date
    daily = df.groupby(dates).agg(
        d_o=("open",  "first"),
        d_h=("high",  "max"),
        d_l=("low",   "min"),
        d_c=("close", "last"),
    )
    daily.index = pd.to_datetime(daily.index)
    prev = daily.shift(1)

    bar_dates = pd.to_datetime(dates.values)
    result = pd.DataFrame({
        "prev_open":  prev["d_o"].reindex(bar_dates).values,
        "prev_high":  prev["d_h"].reindex(bar_dates).values,
        "prev_low":   prev["d_l"].reindex(bar_dates).values,
        "prev_close": prev["d_c"].reindex(bar_dates).values,
    }, index=df.index)
    return result


# ── Core feature builder ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes an indicator-enriched candle DataFrame (output of add_all)
    and returns a wide feature DataFrame, one row per bar.
    All features are normalized — raw prices are not included.
    """
    out   = pd.DataFrame(index=df.index)
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    vol   = df["volume"].values.astype(float)
    atr   = df["atr_14"].values
    rsi   = df["rsi_14"].values
    hist  = df["macd_hist"].values
    macd_ = df["macd"].values
    sig   = df["macd_signal"].values
    bb_p  = df["bb_pct"].values
    bb_u  = df["bb_upper"].values
    bb_l  = df["bb_lower"].values
    vwap  = df["vwap"].values
    e9    = df["ema_9"].values
    e21   = df["ema_21"].values
    e50   = df["ema_50"].values
    e200  = df["ema_200"].values
    rv20  = df["rvol_20"].values
    rv60  = df["rvol_60"].values
    h     = df["time"].dt.hour.values
    m     = df["time"].dt.minute.values
    dow   = df["time"].dt.dayofweek.values

    safe_atr = np.where(atr > 1e-9, atr, 1e-6)

    # ── 1. Multi-timeframe returns ────────────────────────────────────────────
    # "How far did price travel in the last X bars?" — normalized by ATR
    # Positive = up, Negative = down.  1 unit ≈ 1×ATR.
    for lag, name in [
        (1,  "ret_5m"),    # 5 min
        (3,  "ret_15m"),   # 15 min
        (6,  "ret_30m"),   # 30 min
        (12, "ret_1h"),    # 1 hour
        (24, "ret_2h"),    # 2 hours
        (48, "ret_4h"),    # 4 hours
        (72, "ret_6h"),    # 6 hours — covers most of a session
    ]:
        pad = np.full(lag, np.nan)
        shifted = np.concatenate([pad, close[:-lag]])
        out[name] = (close - shifted) / safe_atr

    # ── 2. EMA positions ──────────────────────────────────────────────────────
    # Where is price relative to each moving average? (normalized by ATR)
    # Positive = above EMA (bullish context), Negative = below (bearish)
    out["dist_ema9_atr"]   = (close - e9)   / safe_atr
    out["dist_ema21_atr"]  = (close - e21)  / safe_atr
    out["dist_ema50_atr"]  = (close - e50)  / safe_atr
    out["dist_ema200_atr"] = (close - e200) / safe_atr

    # ── 3. EMA slopes ─────────────────────────────────────────────────────────
    # Is each moving average itself rising or falling?
    # Positive = rising (trend up), Negative = falling (trend down)
    for arr, lag, name in [
        (e9,   6,  "slope_ema9"),    # 9-bar EMA slope over last 30 min
        (e21,  12, "slope_ema21"),   # 21-bar EMA slope over last 1 h
        (e50,  24, "slope_ema50"),   # 50-bar EMA slope over last 2 h
        (e200, 48, "slope_ema200"),  # 200-bar EMA slope over last 4 h
    ]:
        pad = np.full(lag, np.nan)
        shifted = np.concatenate([pad, arr[:-lag]])
        out[name] = (arr - shifted) / safe_atr

    # ── 4. EMA spread & alignment score ──────────────────────────────────────
    # EMA9 vs EMA21: positive = short trend up (fast above slow)
    out["ema9_vs_21"]  = (e9  - e21)  / safe_atr
    out["ema21_vs_50"] = (e21 - e50)  / safe_atr

    # Alignment: how many EMA pairs are bullish-stacked?
    # +3 = all 4 EMAs in bull order (9>21>50>200), -3 = all bear, 0 = mixed
    out["ema_alignment"] = (
        (e9  > e21).astype(float) + (e21 > e50).astype(float) + (e50 > e200).astype(float)
      - (e9  < e21).astype(float) - (e21 < e50).astype(float) - (e50 < e200).astype(float)
    )

    # ── 5. MACD dynamics ──────────────────────────────────────────────────────
    h1 = np.concatenate([[np.nan], hist[:-1]])
    h2 = np.concatenate([[np.nan, np.nan], hist[:-2]])
    h5 = np.concatenate([np.full(5, np.nan), hist[:-5]])

    # Current histogram and MACD-signal spread
    out["macd_hist"]       = hist  / safe_atr   # histogram size (pos=bull, neg=bear)
    out["macd_signal_gap"] = (macd_ - sig) / safe_atr   # how far is MACD from signal?

    # Histogram momentum: is it getting bigger or smaller?
    out["macd_slope"]   = (hist - h1) / safe_atr   # 1-bar change
    out["macd_accel"]   = ((hist - h1) - (h1 - h2)) / safe_atr   # rate of change of slope
    out["macd_5bar"]    = (hist - h5) / safe_atr   # 25-min histogram trend

    # Timing: how many bars since MACD last crossed direction?
    # Low value = fresh crossover (momentum just started)
    # High value = trend has been running for a while (potential exhaustion)
    cross_bull = (hist > 0) & (h1 <= 0)
    cross_bear = (hist < 0) & (h1 >= 0)
    out["bars_since_macd_bull"] = bars_since(cross_bull, max_val=50) / 50.0  # 0=fresh, 1=old
    out["bars_since_macd_bear"] = bars_since(cross_bear, max_val=50) / 50.0

    # Divergence: is price and MACD disagreeing about direction?
    # Bullish divergence = price down but MACD recovering → potential reversal up
    print("  Computing MACD divergences...")
    out["macd_div_bull"] = bullish_divergence(close, hist, lookback=24)
    out["macd_div_bear"] = bearish_divergence(close, hist, lookback=24)

    # ── 6. RSI dynamics ───────────────────────────────────────────────────────
    r3 = np.concatenate([np.full(3, np.nan), rsi[:-3]])
    r6 = np.concatenate([np.full(6, np.nan), rsi[:-6]])

    out["rsi_norm"]       = rsi / 100.0           # 0–1 scale
    out["rsi_dist_50"]    = (rsi - 50) / 50.0     # signed distance from neutral midpoint
    out["rsi_slope_3bar"] = (rsi - r3) / 30.0     # RSI rising or falling?
    out["rsi_slope_6bar"] = (rsi - r6) / 30.0

    # Timing: how long ago was RSI in an extreme zone?
    # Short = recently oversold/overbought (potential for bounce/fade)
    out["bars_since_oversold"]   = bars_since(rsi < 30, max_val=50) / 50.0
    out["bars_since_overbought"] = bars_since(rsi > 70, max_val=50) / 50.0

    # RSI divergence: same logic as MACD — disagreement signals reversals
    print("  Computing RSI divergences...")
    out["rsi_div_bull"] = bullish_divergence(close, rsi, lookback=24)
    out["rsi_div_bear"] = bearish_divergence(close, rsi, lookback=24)

    # ── 7. Bollinger Bands ────────────────────────────────────────────────────
    bb_width = bb_u - bb_l
    bb_w_ma  = pd.Series(bb_width).rolling(20).mean().values

    out["bb_pct"]          = bb_p                  # 0=at lower band, 1=at upper band
    out["bb_width_atr"]    = bb_width / safe_atr   # how wide are the bands? (vol expansion)
    out["bb_squeeze"]      = np.where(bb_w_ma > 0, bb_width / bb_w_ma, 1.0)   # <1 = squeeze

    # Is band width expanding or contracting? (potential breakout incoming if tightening)
    bww6 = np.concatenate([np.full(6, np.nan), bb_width[:-6]])
    out["bb_width_trend"]  = (bb_width - bww6) / safe_atr

    # "Walking the band": price hugging upper/lower band (strong trend signal)
    bb_s = pd.Series(bb_p)
    out["bb_walk_up"]      = (bb_s.rolling(3).min() > 0.80).astype(float).values
    out["bb_walk_down"]    = (bb_s.rolling(3).max() < 0.20).astype(float).values

    # ── 8. VWAP ───────────────────────────────────────────────────────────────
    vw6 = np.concatenate([np.full(6, np.nan), vwap[:-6]])
    out["dist_vwap_atr"]  = (close - vwap) / safe_atr   # above=bullish, below=bearish
    out["vwap_slope"]     = (vwap - vw6)   / safe_atr   # is day's average price rising?

    # ── 9. Volatility regime ──────────────────────────────────────────────────
    atr_pct    = safe_atr / np.where(close > 0, close, 1.0) * 100
    atr_med50  = pd.Series(atr).rolling(50).median().values
    out["atr_pct"]        = atr_pct
    out["atr_vs_median"]  = np.where(atr_med50 > 0, atr / atr_med50, 1.0)   # >1=high vol day
    out["rvol_ratio"]     = np.where(rv60 > 0, rv20 / rv60, 1.0)   # <1=vol contracting
    out["bar_range_atr"]  = (high - low) / safe_atr   # this bar's range vs ATR

    # ── 10. Bar structure (what the candle body and wicks tell you) ───────────
    bar_range = np.where((high - low) > 1e-9, high - low, 1e-9)
    top_wick  = high - np.maximum(open_, close)
    bot_wick  = np.minimum(open_, close) - low

    out["body_pct"]       = np.abs(close - open_) / bar_range   # 1=full body, 0=doji
    out["upper_wick_pct"] = top_wick / bar_range   # large = sellers rejecting higher prices
    out["lower_wick_pct"] = bot_wick / bar_range   # large = buyers defending lower prices
    out["bar_direction"]  = np.sign(close - open_).astype(float)   # +1=green, -1=red
    out["bar_body_atr"]   = (close - open_) / safe_atr   # signed bar move in ATR units

    # Consecutive same-direction bars: momentum run or exhaustion
    out["consec_green"]   = count_consecutive(close > open_)
    out["consec_red"]     = count_consecutive(close < open_)

    # Trend consistency: what % of last 12 bars (1h) closed in the bull direction?
    green_float = (close > open_).astype(float)
    out["trend_consist_1h"]  = pd.Series(green_float).rolling(12).mean().values
    out["trend_consist_30m"] = pd.Series(green_float).rolling(6).mean().values

    # ── 11. Swing distance ────────────────────────────────────────────────────
    hi20 = pd.Series(high).rolling(20).max().values
    lo20 = pd.Series(low).rolling(20).min().values
    range20 = np.where((hi20 - lo20) > 0, hi20 - lo20, safe_atr)

    out["dist_hi20_atr"]  = (hi20 - close) / safe_atr   # how far below 20-bar high?
    out["dist_lo20_atr"]  = (close - lo20) / safe_atr   # how far above 20-bar low?
    out["pos_in_range20"] = (close - lo20) / range20     # 0=at low, 1=at high of 20-bar range

    # ── 12. Session context ───────────────────────────────────────────────────
    sess_hi    = session_cummax(df)
    sess_lo    = session_cummin(df)
    sess_range = np.where((sess_hi - sess_lo) > 0, sess_hi - sess_lo, safe_atr)

    out["dist_sess_hi_atr"]   = (sess_hi - close)  / safe_atr   # how far from day high?
    out["dist_sess_lo_atr"]   = (close   - sess_lo) / safe_atr  # how far from day low?
    out["pos_in_sess_range"]  = (close   - sess_lo) / sess_range # 0=at day low, 1=at day high
    out["sess_range_atr"]     = sess_range / safe_atr            # how large is today's range?

    # Opening range breakout: above/below the first 30 min high/low
    or_hi, or_lo = opening_range(df, minutes=30)
    out["above_opening_range"]  = np.where(
        ~np.isnan(or_hi), (close - or_hi) / safe_atr, np.nan
    )   # positive = above (bullish breakout of opening range)
    out["below_opening_range"]  = np.where(
        ~np.isnan(or_lo), (or_lo - close) / safe_atr, np.nan
    )   # positive = below (bearish breakdown)

    # ── 13. Previous day levels ───────────────────────────────────────────────
    # Critical reference: did we gap up? Are we trading above yesterday's high?
    pday     = prev_day_ohlc(df)
    p_close  = pday["prev_close"].values
    p_high   = pday["prev_high"].values
    p_low    = pday["prev_low"].values
    p_open   = pday["prev_open"].values
    day_open = session_first_open(df)

    # Gap: how much did price jump (or fall) overnight vs yesterday's close?
    out["gap_atr"] = np.where(
        ~np.isnan(p_close), (day_open - p_close) / safe_atr, np.nan
    )
    # Above/below yesterday's close — continuation vs reversal
    out["above_prev_close"] = np.where(
        ~np.isnan(p_close), (close - p_close) / safe_atr, np.nan
    )
    # Distance from yesterday's high (resistance) and low (support)
    out["dist_prev_high"] = np.where(
        ~np.isnan(p_high), (close - p_high) / safe_atr, np.nan
    )
    out["dist_prev_low"]  = np.where(
        ~np.isnan(p_low),  (close - p_low)  / safe_atr, np.nan
    )
    # Was yesterday a bull or bear day? (directional context)
    out["prev_day_return"] = np.where(
        (~np.isnan(p_close)) & (~np.isnan(p_open)) & (p_open > 0),
        (p_close - p_open) / p_open * 100, np.nan
    )
    # Previous day's range in ATR units (was it a volatile day?)
    out["prev_day_range_atr"] = np.where(
        (~np.isnan(p_high)) & (~np.isnan(p_low)),
        (p_high - p_low) / safe_atr, np.nan
    )

    # ── 14. Volume ────────────────────────────────────────────────────────────
    vol_ma20 = pd.Series(vol).rolling(20).mean().values
    vol_ma5  = pd.Series(vol).rolling(5).mean().values
    out["rel_volume"]    = np.where(vol_ma20 > 0, vol / vol_ma20, 1.0)   # >1.5 = high vol bar
    out["vol_trend"]     = np.where(vol_ma20 > 0, vol_ma5 / vol_ma20, 1.0)  # rising vol?

    # ── 15. Round number distance ─────────────────────────────────────────────
    # S&P 500 traders watch round numbers (4000, 4050, 4100...) like magnets.
    # "How close is price to the nearest 50-point level?"
    # 0 = right at a round number, 0.5 = halfway between them (normalized)
    nearest_50   = np.round(close / 50) * 50
    out["dist_round50_atr"] = np.abs(close - nearest_50) / safe_atr
    nearest_100  = np.round(close / 100) * 100
    out["dist_round100_atr"] = np.abs(close - nearest_100) / safe_atr

    # ── 16. Time features ─────────────────────────────────────────────────────
    # Cyclical encoding: 23:55 and 00:05 should be "close" to each other
    total_mins = (h * 60 + m).astype(float)
    out["hour_sin"]     = np.sin(2 * np.pi * h / 24)
    out["hour_cos"]     = np.cos(2 * np.pi * h / 24)
    out["minute_sin"]   = np.sin(2 * np.pi * total_mins / 1440)
    out["minute_cos"]   = np.cos(2 * np.pi * total_mins / 1440)
    out["day_of_week"]  = dow.astype(float)

    # Session flags (discrete, useful as context)
    out["is_power_hour"] = ((h == 22) & (m < 45)).astype(float)
    out["is_us_open"]    = ((h >= 14) & (h < 16)).astype(float)
    out["is_us_session"] = ((h >= 14) & (h < 21)).astype(float)

    return out


# ── Analysis helpers ──────────────────────────────────────────────────────────

def feature_stats(feat_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    For each feature, compute:
    - Correlation with the label (absolute Pearson r)
    - % of non-NaN values
    - Mean and std
    """
    valid = labels >= 0
    y     = labels[valid].astype(float)
    rows  = []
    for col in feat_df.columns:
        x = feat_df[col].values[valid].astype(float)
        non_nan = np.sum(~np.isnan(x)) / len(x)
        if non_nan < 0.5:
            corr = np.nan
        else:
            mask2 = ~np.isnan(x)
            try:
                corr = float(np.corrcoef(x[mask2], y[mask2])[0, 1])
            except Exception:
                corr = np.nan
        rows.append({
            "feature":    col,
            "corr_label": round(corr, 5)  if not np.isnan(corr) else None,
            "abs_corr":   round(abs(corr), 5) if not np.isnan(corr) else None,
            "coverage":   round(non_nan, 3),
            "mean":       round(float(np.nanmean(feat_df[col])), 4),
            "std":        round(float(np.nanstd(feat_df[col])),  4),
        })
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading M5 data...")
    raw = cache.load_ohlcv("US500.pro", "M5")
    if raw is None:
        print("No data. Run 01_fetch_data.py first.")
        return

    print("Computing indicators...")
    df = add_all(raw)
    print(f"  {len(df):,} M5 bars loaded")

    print("\nEngineering features (divergence scans take ~30s)...")
    feats = build_features(df)
    print(f"  Done — {len(feats.columns)} features")

    # Identification columns (raw, not used as features)
    meta = df[["time", "open", "high", "low", "close", "volume", "atr_14"]].copy()
    meta["date"]        = df["time"].dt.date
    meta["hour"]        = df["time"].dt.hour
    meta["minute"]      = df["time"].dt.minute
    meta["day_name"]    = df["time"].dt.day_name().str[:3]   # human label; numeric in feats

    # Labels
    print("\nLabelling bars...")
    lbl_long  = make_long_labels( df, tp_mult=TP_MULT, sl_mult=SL_MULT, lookahead=LOOKAHEAD)
    lbl_short = make_short_labels(df, tp_mult=TP_MULT, sl_mult=SL_MULT, lookahead=LOOKAHEAD)
    meta["label_long"]  = lbl_long
    meta["label_short"] = lbl_short

    # Assemble
    full = pd.concat([
        meta.reset_index(drop=True),
        feats.reset_index(drop=True),
    ], axis=1).iloc[WARMUP:].reset_index(drop=True)

    # Save (round floats to 4 dp at write time to keep file size manageable)
    out_full    = cache.RESULTS / "04_features_full.csv"
    out_preview = cache.RESULTS / "04_features_preview.csv"
    full.round(4).to_csv(out_full,           index=False)
    full.head(500).round(4).to_csv(out_preview, index=False)

    # Stats
    valid = full[full["label_long"] >= 0]
    base_wr = (valid["label_long"] == 1).mean()
    print(f"\n{'═'*60}")
    print(f"  FEATURE SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total bars     : {len(full):,}")
    print(f"  Features       : {len(feats.columns)}")
    print(f"  Labelled bars  : {len(valid):,}")
    print(f"  Base WR (long) : {base_wr:.1%}  (no filter applied)")
    print(f"\nFiles saved:")
    print(f"  {out_full}  ({len(full):,} rows × {len(full.columns)} cols)")
    print(f"  {out_preview}  (500-row preview)")

    # Feature correlation ranking
    print(f"\n{'─'*60}")
    print(f"  TOP FEATURES BY CORRELATION WITH LABEL")
    print(f"{'─'*60}")
    stats = feature_stats(feats.iloc[WARMUP:].reset_index(drop=True),
                          full["label_long"].values)
    print(f"  {'Feature':<28}  {'|r|':>7}  {'r':>8}  {'Coverage':>9}")
    for _, row in stats.head(20).iterrows():
        direction = "↑" if (row["corr_label"] or 0) > 0 else "↓"
        print(f"  {row['feature']:<28}  "
              f"{row['abs_corr']:>7.4f}  "
              f"{row['corr_label']:>+8.4f} {direction}  "
              f"{row['coverage']:>8.0%}")

    print(f"\n  All {len(feats.columns)} features:")
    cats = {
        "Multi-timeframe returns": [c for c in feats.columns if c.startswith("ret_")],
        "EMA positions":           [c for c in feats.columns if "dist_ema" in c or "ema_align" in c or "ema9_vs" in c or "ema21_vs" in c],
        "EMA slopes":              [c for c in feats.columns if c.startswith("slope_")],
        "MACD":                    [c for c in feats.columns if "macd" in c],
        "RSI":                     [c for c in feats.columns if "rsi" in c or "oversold" in c or "overbought" in c],
        "Bollinger":               [c for c in feats.columns if "bb_" in c],
        "VWAP":                    [c for c in feats.columns if "vwap" in c],
        "Volatility":              [c for c in feats.columns if c in ("atr_pct","atr_vs_median","rvol_ratio","bar_range_atr")],
        "Bar structure":           [c for c in feats.columns if c in ("body_pct","upper_wick_pct","lower_wick_pct","bar_direction","bar_body_atr","consec_green","consec_red","trend_consist_1h","trend_consist_30m")],
        "Swing distance":          [c for c in feats.columns if "hi20" in c or "lo20" in c or "range20" in c],
        "Session context":         [c for c in feats.columns if "sess" in c or "opening_range" in c],
        "Previous day":            [c for c in feats.columns if "prev" in c or "gap_atr" in c],
        "Volume":                  [c for c in feats.columns if "vol" in c and "rvol" not in c],
        "Round numbers":           [c for c in feats.columns if "round" in c],
        "Time":                    [c for c in feats.columns if c in ("hour_sin","hour_cos","minute_sin","minute_cos","day_of_week","is_power_hour","is_us_open","is_us_session")],
    }
    for cat, cols in cats.items():
        if cols:
            print(f"\n  {cat} ({len(cols)}): {', '.join(cols)}")

    # Importance chart
    plt.style.use("dark_background")
    stats_top = stats.dropna(subset=["abs_corr"]).head(30)
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#00e676" if c > 0 else "#ef5350"
              for c in stats_top["corr_label"].fillna(0)]
    ax.barh(stats_top["feature"][::-1], stats_top["abs_corr"][::-1],
            color=colors[::-1], edgecolor="none")
    ax.set_xlabel("|Pearson correlation with win label|", fontsize=9)
    ax.set_title(
        "Feature correlation with winning long trade\n"
        "green = positive correlation (feature↑ → more wins)   "
        "red = negative (feature↑ → fewer wins)",
        fontsize=9
    )
    ax.grid(True, axis="x", alpha=0.2)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    cache.save_chart(fig, "04_feature_correlations")
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
