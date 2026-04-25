"""Fetch OHLCV candles from MT5."""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from config import SYMBOL, TIMEFRAME_MAP, DEFAULT_TIMEFRAME


def get_timeframe(tf_str: str) -> int:
    """Convert string like 'M5' to MT5 timeframe constant."""
    tf = TIMEFRAME_MAP.get(tf_str.upper())
    if tf is None:
        raise ValueError(f"Unknown timeframe '{tf_str}'. Valid: {list(TIMEFRAME_MAP)}")
    # MT5 constants for M1-M30 are just the minute count; for H/D use the real constant
    tf_const_map = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }
    return tf_const_map[tf_str.upper()]


def get_candles(
    n: int = 500,
    symbol: str = SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """
    Fetch the last `n` closed candles for `symbol` at `timeframe`.
    Returns a DataFrame with columns: time, open, high, low, close, volume.
    """
    tf = get_timeframe(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No candle data for {symbol} ({timeframe}): {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )
    return df.reset_index(drop=True)


def get_candles_range(
    start: datetime,
    end: datetime,
    symbol: str = SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """Fetch candles between two datetime objects."""
    tf = get_timeframe(timeframe)
    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No candle data for {symbol} ({timeframe}): {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )
    return df.reset_index(drop=True)


def get_tick(symbol: str = SYMBOL) -> dict:
    """Latest bid/ask tick."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick for {symbol}: {mt5.last_error()}")
    return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}
