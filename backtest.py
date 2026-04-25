"""
Vectorized backtester.

simulate() maps (signals, labels) → individual trade P&L in points.
metrics() computes Sharpe, win-rate, drawdown, profit factor.
All costs (spread) are deducted per trade.
"""

import numpy as np
import pandas as pd

SPREAD_PTS = 0.7   # typical US500.pro round-trip cost (bid-ask on entry)
BARS_PER_DAY = 270
TRADING_DAYS = 252
BARS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS


def simulate(
    signal_mask: np.ndarray,
    labels: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    atr: np.ndarray,
    times: np.ndarray = None,
    spread_pts: float = SPREAD_PTS,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    signal_mask : bool array, True where strategy wants to enter
    labels      : int8 array from labeler (1=win, 0=loss, -1=undefined)
    tp_mult, sl_mult : ATR multiples matching the label definition
    atr         : ATR array aligned with signal_mask / labels
    times       : optional datetime array for the trade log
    """
    valid = signal_mask & (labels >= 0)
    idx   = np.where(valid)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["bar_idx", "time", "label", "atr", "tp_pts", "sl_pts", "pnl"])

    lab  = labels[idx]
    atrs = atr[idx]

    tp_pts = tp_mult * atrs
    sl_pts = sl_mult * atrs
    pnl    = np.where(lab == 1, tp_pts, -sl_pts) - spread_pts

    result = {
        "bar_idx": idx,
        "label":   lab,
        "atr":     atrs,
        "tp_pts":  tp_pts,
        "sl_pts":  sl_pts,
        "pnl":     pnl,
    }
    if times is not None:
        result["time"] = times[idx]
    return pd.DataFrame(result)


def metrics(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"n_trades": 0, "win_rate": 0, "avg_pnl": 0,
                "total_pnl": 0, "sharpe": 0, "max_drawdown": 0, "profit_factor": 0}

    pnl  = trades["pnl"].values
    n    = len(pnl)
    wins = (trades["label"] == 1).sum()

    mean_pnl = pnl.mean()
    std_pnl  = pnl.std(ddof=1)

    # Annualised Sharpe at trade level
    span_bars = int(trades["bar_idx"].max() - trades["bar_idx"].min()) + 1
    tpy = n * (BARS_PER_YEAR / max(span_bars, 1))   # implied trades per year
    sharpe = (mean_pnl / std_pnl * np.sqrt(tpy)) if std_pnl > 0 else 0.0

    cum   = pd.Series(pnl).cumsum()
    dd    = (cum - cum.cummax()).min()
    gains = pnl[pnl > 0].sum()
    losss = abs(pnl[pnl < 0].sum())
    pf    = gains / losss if losss > 0 else np.inf

    return {
        "n_trades":     n,
        "win_rate":     wins / n,
        "avg_pnl":      mean_pnl,
        "total_pnl":    pnl.sum(),
        "sharpe":       sharpe,
        "max_drawdown": dd,
        "profit_factor": pf,
    }


def print_metrics(m: dict, name: str = ""):
    bar = "=" * 48
    print(f"\n{bar}")
    if name:
        print(f"  {name}")
        print(bar)
    print(f"  Trades        : {m['n_trades']:,}")
    print(f"  Win rate      : {m['win_rate']:.1%}")
    print(f"  Avg P&L       : {m['avg_pnl']:+.2f} pts")
    print(f"  Total P&L     : {m['total_pnl']:+.1f} pts")
    print(f"  Sharpe        : {m['sharpe']:.2f}")
    print(f"  Max drawdown  : {m['max_drawdown']:.1f} pts")
    print(f"  Profit factor : {m['profit_factor']:.2f}")
    print(bar)
