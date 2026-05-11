"""
monte_carlo/analysis.py — Portfolio metrics computed for every path × day.

All operations are vectorised — no Python loops over paths.
"""

from typing import Dict, List, Tuple
import numpy as np

import config


def compute_portfolio(
    prices: np.ndarray,                          # (N, T+1)
    fill_day: np.ndarray,                        # (N, P)  -1 = unfilled
    open_positions: List[Tuple[float, float]],   # [(lots, entry_price), ...]
    pending_orders: List[Tuple[float, float]],   # [(lots, order_price), ...]
    balance: float,
    coeff: float  = config.COEFF,
    leverage: int = config.LEVERAGE,
) -> Dict[str, np.ndarray]:
    """
    For every (path, day) compute equity, margin used, margin level, free margin.

    Memory note: temporary (N, P, T+1) float64 arrays are ~87 MB for default
    parameters — tolerable on any modern workstation.

    Returns dict of (N, T+1) arrays plus scalar `open_margin`.
    """
    n_paths, T1 = prices.shape
    T           = T1 - 1
    days        = np.arange(T1, dtype=np.int32)  # 0..T

    # ── Open positions: always active from day 0 ──────────────────────────────
    open_lots  = np.array([p[0] for p in open_positions])  # (M,)
    open_entry = np.array([p[1] for p in open_positions])  # (M,)

    # Scalar: total margin locked by current open positions (constant over time)
    open_margin = float((open_lots * open_entry * coeff / leverage).sum())

    # Unrealised P&L from open positions at each price bar
    # prices[:, :, None] − open_entry[None, None, :] → (N, T+1, M)
    upnl_open = (
        (prices[:, :, np.newaxis] - open_entry[np.newaxis, np.newaxis, :])
        * (open_lots * coeff)[np.newaxis, np.newaxis, :]
    ).sum(axis=2)  # (N, T+1)

    # ── Pending orders: active from fill_day onward ───────────────────────────
    pend_lots  = np.array([o[0] for o in pending_orders])   # (P,)
    pend_price = np.array([o[1] for o in pending_orders])   # (P,)

    # active[n, p, t] = True if order p filled in path n and day t >= fill_day
    fill_day_3d = fill_day[:, :, np.newaxis]        # (N, P, 1)
    days_3d     = days[np.newaxis, np.newaxis, :]   # (1, 1, T+1)
    active      = (fill_day_3d >= 0) & (days_3d >= fill_day_3d)  # (N, P, T+1)

    # Margin from filled pending orders
    pend_margin_each = (pend_lots * pend_price * coeff / leverage)  # (P,)
    pend_margin = (
        active * pend_margin_each[np.newaxis, :, np.newaxis]
    ).sum(axis=1)  # (N, T+1)

    # Unrealised P&L from filled pending orders
    pend_pnl_scale = (pend_lots * coeff)[np.newaxis, :, np.newaxis]   # (1, P, 1)
    pend_entry_3d  = pend_price[np.newaxis, :, np.newaxis]             # (1, P, 1)
    prices_3d      = prices[:, np.newaxis, :]                           # (N, 1, T+1)
    upnl_pend = (
        active * pend_pnl_scale * (prices_3d - pend_entry_3d)
    ).sum(axis=1)  # (N, T+1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total_upnl   = upnl_open + upnl_pend                               # (N, T+1)
    margin_used  = open_margin + pend_margin                           # (N, T+1)
    equity       = balance + total_upnl                                # (N, T+1)
    margin_level = np.where(
        margin_used > 0, equity / margin_used * 100.0, np.inf
    )                                                                   # (N, T+1)
    free_margin  = equity - margin_used                                # (N, T+1)

    # ── Stop-out mask ─────────────────────────────────────────────────────────
    # A path is "stopped out" when margin level first drops below 100%.
    # Once stopped, it stays stopped (broker closes all positions).
    stopped_bar  = margin_level < config.STOPOUT_LEVEL                 # (N, T+1)
    ever_stopped = np.maximum.accumulate(stopped_bar, axis=1)          # (N, T+1)

    return {
        "upnl":         total_upnl,
        "margin_used":  margin_used,
        "equity":       equity,
        "margin_level": margin_level,
        "free_margin":  free_margin,
        "stopped":      ever_stopped,
        "open_margin":  open_margin,
    }


def fill_probabilities(
    fill_day: np.ndarray,
    horizons: Tuple[int, ...] = (30, 60, 90),
) -> np.ndarray:
    """
    Fraction of paths where each pending order fills within each horizon.
    Returns (n_orders, len(horizons)).
    """
    n_paths, n_orders = fill_day.shape
    result = np.zeros((n_orders, len(horizons)))
    for j, h in enumerate(horizons):
        result[:, j] = ((fill_day >= 0) & (fill_day <= h)).mean(axis=0)
    return result


def liquidation_curve(stopped: np.ndarray) -> np.ndarray:
    """
    For each day t: fraction of paths stopped out at any point up to t.
    Returns (T+1,) — value at t=0 should be 0.
    """
    return stopped.mean(axis=0)


def center_of_weight(
    prices: np.ndarray,
    sigma_ann: float,
    price_range: np.ndarray,
    coeff: float  = config.COEFF,
    leverage: int = config.LEVERAGE,
) -> np.ndarray:
    """
    For each candidate entry price level, compute the expected risk-adjusted
    P&L per 0.001 lot placed there, as a function of fill probability
    and expected gain to day-90 close.

    Score(level) = fill_prob × E[price_T − level | filled] × coeff × 0.001
                  − margin_per_0001_lot

    Returns (len(price_range),) array of scores.
    """
    sigma_daily = sigma_ann / np.sqrt(252)
    low_factor  = np.exp(-0.4 * sigma_daily)
    daily_lows  = np.minimum(prices[:, :-1], prices[:, 1:]) * low_factor  # (N, T)

    final_prices = prices[:, -1]   # (N,)
    scores       = np.empty(len(price_range))

    for k, level in enumerate(price_range):
        filled = (daily_lows <= level).any(axis=1)   # (N,) bool
        fp     = filled.mean()

        if fp < 0.005:
            scores[k] = 0.0
            continue

        # Expected final price conditional on filling (optimistic: filled paths
        # tend to be those where price dipped, not necessarily where it ended up)
        e_final = final_prices[filled].mean()
        expected_gain_per_lot = (e_final - level) * coeff           # per 1.0 lot
        margin_per_lot        = level * coeff / leverage

        # Score per 0.001 lot, penalised by margin opportunity cost
        scores[k] = fp * expected_gain_per_lot * 0.001 - margin_per_lot * 0.001 * 0.1

    return scores


def optimal_lots(
    prices: np.ndarray,
    fill_day: np.ndarray,
    pending_orders: List[Tuple[float, float]],
    cow_scores: np.ndarray,
    cow_prices: np.ndarray,
    portfolio: Dict[str, np.ndarray],
    balance: float,
    coeff: float  = config.COEFF,
    leverage: int = config.LEVERAGE,
    target_ml: float = config.TARGET_ML_P5,
) -> np.ndarray:
    """
    Suggest lot sizes for pending orders that:
      1. Are proportional to the CoW score at each order price.
      2. Keep the 5th-percentile margin level at day 90 above target_ml.

    Returns (n_orders,) array of suggested lot sizes.
    """
    n_orders = len(pending_orders)
    pend_price = np.array([o[1] for o in pending_orders], dtype=float)
    fill_prob_90 = ((fill_day >= 0) & (fill_day <= 90)).mean(axis=0)  # (P,)

    # CoW score interpolated to each order price
    cow_at_order = np.interp(pend_price, cow_prices, cow_scores)
    cow_at_order = np.clip(cow_at_order, 0, None)

    # Ideal relative weight: fill_prob × CoW (higher score → more lots)
    weight = fill_prob_90 * cow_at_order
    total_weight = weight.sum()
    if total_weight <= 0:
        return np.array([o[0] for o in pending_orders])  # unchanged

    # Normalise weights to match the average current lot size
    avg_current_lot = np.mean([o[0] for o in pending_orders])
    ideal_lots = weight / total_weight * avg_current_lot * n_orders

    # ── Constraint: 5th pct margin level at day 90 >= target_ml ─────────────
    # Analytical approximation using p5 price from simulation
    p5_price = float(np.percentile(prices[:, 90], 5))

    open_lots  = np.array([p[0] for p in config.OPEN_POSITIONS])
    open_entry = np.array([p[1] for p in config.OPEN_POSITIONS])
    open_upnl_p5 = float(((p5_price - open_entry) * open_lots * coeff).sum())
    open_margin  = float((open_lots * open_entry * coeff / leverage).sum())

    # For each scale factor k applied to ideal_lots:
    # pending contribution at p5_price (only orders with order_price > p5 have losses)
    def _p5_metrics(k: float) -> Tuple[float, float]:
        lots  = ideal_lots * k
        f90   = fill_prob_90  # probability of being filled by day 90

        add_margin = float((f90 * lots * pend_price * coeff / leverage).sum())
        add_upnl   = float((f90 * lots * (p5_price - pend_price) * coeff).sum())

        eq_p5  = balance + open_upnl_p5 + add_upnl
        mar_p5 = open_margin + add_margin
        ml_p5  = (eq_p5 / mar_p5 * 100.0) if mar_p5 > 0 else np.inf
        return ml_p5, eq_p5

    # Binary search for maximum k that keeps p5 ML >= target_ml
    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid    = (lo + hi) / 2
        ml, _  = _p5_metrics(mid)
        if ml >= target_ml:
            lo = mid
        else:
            hi = mid

    best_k   = lo
    best_lots = np.round(ideal_lots * best_k, 3)
    best_lots = np.maximum(best_lots, 0.001)  # floor at minimum tradeable size

    return best_lots
