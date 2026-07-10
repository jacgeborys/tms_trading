"""
01_fetch_data.py — Download fresh OHLCV data from MT5.

Connects to MetaTrader 5 (must be open and logged in), downloads candles
for all tracked symbols, and saves to the data/ cache.

Run this:
  - On first setup
  - When data is stale (cache auto-refreshes if >12 h old anyway)
  - Before any analysis after a weekend gap

Usage:
  python 01_fetch_data.py
"""

import sys
from pathlib import Path
from mt5_client import connect, disconnect
from data import get_candles

_DATA = Path(__file__).parent / "data"

def _save_ohlcv(df, symbol: str, timeframe: str):
    _DATA.mkdir(parents=True, exist_ok=True)
    path = _DATA / f"{symbol.replace('.', '_')}_{timeframe}.pkl"
    df.to_pickle(path)
    print(f"  Saved {len(df):,} bars → {path.name}")

SYMBOLS = ["US500.pro", "US100.pro", "US30.pro", "GOLD.pro"]
TIMEFRAMES = {
    "M5": 50000,   # ~8.5 months of 5-min bars
    "H1":  5000,   # ~8.5 years of hourly bars (used for trend filters)
}


def main():
    print("Connecting to MT5...")
    if not connect():
        print("ERROR: Could not connect. Is MetaTrader 5 open and logged in?")
        sys.exit(1)

    for symbol in SYMBOLS:
        for tf, n_bars in TIMEFRAMES.items():
            print(f"  Fetching {symbol} {tf}  (up to {n_bars:,} bars)...")
            try:
                df = get_candles(n=n_bars, symbol=symbol, timeframe=tf)
                _save_ohlcv(df, symbol, tf)
            except Exception as e:
                print(f"    WARNING: {e}")

    disconnect()
    print("\nDone. All data saved to data/")


if __name__ == "__main__":
    main()
