"""
Strategy interface. Add your signal logic here.

Each strategy receives a candle DataFrame and returns a signal:
  +1  = go long
  -1  = go short
   0  = no trade / flat
"""

import pandas as pd


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> int:
    """
    Simple moving average crossover baseline.
    Returns +1 if fast SMA just crossed above slow SMA, -1 if below, 0 otherwise.
    """
    if len(df) < slow + 1:
        return 0

    close = df["close"]
    fast_now  = close.iloc[-fast:].mean()
    fast_prev = close.iloc[-fast - 1:-1].mean()
    slow_now  = close.iloc[-slow:].mean()
    slow_prev = close.iloc[-slow - 1:-1].mean()

    crossed_up   = fast_prev <= slow_prev and fast_now > slow_now
    crossed_down = fast_prev >= slow_prev and fast_now < slow_now

    if crossed_up:
        return 1
    if crossed_down:
        return -1
    return 0


# Registry: name -> callable(df) -> int
STRATEGIES = {
    "sma_crossover": sma_crossover,
}
