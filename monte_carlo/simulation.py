"""
monte_carlo/simulation.py — GBM price path engine and order fill logic.

All heavy computation is vectorised over the (n_paths, n_days) matrix.
No Python loops over paths.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

import config


def fetch_calibration(verbose: bool = True) -> Tuple[float, float]:
    """
    Download SPX daily closes via yfinance and return
    (annualised_drift, annualised_vol) calibrated from the last HIST_DAYS.
    """
    import yfinance as yf

    ticker = yf.Ticker(config.SPX_TICKER)
    hist   = ticker.history(period=f"{config.HIST_DAYS + 20}d")["Close"]
    log_r  = np.log(hist / hist.shift(1)).dropna().values
    log_r  = log_r[-config.HIST_DAYS:]

    mu_daily    = log_r.mean()
    sigma_daily = log_r.std(ddof=1)

    mu_ann    = mu_daily * 252
    sigma_ann = sigma_daily * np.sqrt(252)

    if verbose:
        print(f"  Calibration ({len(log_r)} days):  "
              f"μ = {mu_ann:+.2%}/yr   σ = {sigma_ann:.2%}/yr")
    return mu_ann, sigma_ann


def generate_paths(
    S0: float,
    mu: float,
    sigma: float,
    n_paths: int = config.N_PATHS,
    n_days: int  = config.N_DAYS,
    df: int      = config.T_DF,
    seed: int    = 42,
) -> np.ndarray:
    """
    GBM with Student-t(df) innovations scaled to unit variance.

    Returns price array of shape (n_paths, n_days + 1).
    Column 0 = S0, column t = price at end of trading day t.
    """
    rng  = np.random.default_rng(seed)
    dt   = 1.0 / 252.0

    # t(df) has variance df/(df-2); normalise to unit variance
    z = rng.standard_t(df=df, size=(n_paths, n_days)) * np.sqrt((df - 2) / df)

    log_ret    = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z  # (N, T)
    log_prices = np.cumsum(log_ret, axis=1)                               # (N, T)
    prices     = S0 * np.exp(
        np.concatenate([np.zeros((n_paths, 1)), log_prices], axis=1)
    )  # (N, T+1)

    return prices


def compute_daily_lows(prices: np.ndarray, sigma_ann: float) -> np.ndarray:
    """
    Approximate the intraday low for each day.

    Uses:  low[d] = min(close[d-1], close[d]) × exp(-0.4 × σ_daily)

    Taking min of prev/current close is more conservative than the
    prompt's close-only formula: on big down days the low extends below
    both closes; on up days the low is still anchored to the opening level.

    Returns shape (n_paths, n_days) — one low per trading day (day 1..T).
    """
    sigma_daily = sigma_ann / np.sqrt(252)
    factor      = np.exp(-0.4 * sigma_daily)

    prev_close = prices[:, :-1]   # (N, T)
    curr_close = prices[:, 1:]    # (N, T)
    return np.minimum(prev_close, curr_close) * factor  # (N, T)


def simulate_fills(
    prices: np.ndarray,
    daily_lows: np.ndarray,
    pending_orders: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Determine first fill day for every pending buy-limit order in every path.

    A buy limit at price P fills on the first day where daily_low <= P.

    Parameters
    ----------
    prices       : (n_paths, n_days+1)
    daily_lows   : (n_paths, n_days)   — from compute_daily_lows()
    pending_orders : list of (lots, limit_price)

    Returns
    -------
    fill_day : (n_paths, n_orders) int32
        1-based day index of first fill, or -1 if never filled.
    """
    n_paths, n_days = daily_lows.shape
    n_orders        = len(pending_orders)
    fill_day        = np.full((n_paths, n_orders), -1, dtype=np.int32)

    for i, (_lots, order_price) in enumerate(pending_orders):
        hit       = daily_lows <= order_price        # (N, T) bool
        first_idx = np.argmax(hit, axis=1)           # (N,)  — 0-indexed into T
        has_hit   = hit[np.arange(n_paths), first_idx]
        fill_day[:, i] = np.where(has_hit, first_idx + 1, -1)

    return fill_day
