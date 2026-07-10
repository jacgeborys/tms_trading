"""
TMS Trading Bot - main entry point.
Connects to MT5 (OANDA demo), prints account info, fetches recent US500 candles,
and runs a strategy signal check.
"""

from mt5_client import connect, disconnect
from account import print_snapshot, open_positions
from data import get_candles, get_tick
from strategy import sma_crossover
from config import SYMBOL, DEFAULT_TIMEFRAME


def main():
    if not connect():
        return

    try:
        print_snapshot()

        # Live tick
        tick = get_tick(SYMBOL)
        print(f"\nLive {SYMBOL}  bid={tick['bid']}  ask={tick['ask']}")

        # Recent candles
        df = get_candles(n=200, symbol=SYMBOL, timeframe=DEFAULT_TIMEFRAME)
        print(f"\nLoaded {len(df)} {DEFAULT_TIMEFRAME} candles for {SYMBOL}")
        print(df.tail(3).to_string(index=False))

        # Open positions
        positions = open_positions(SYMBOL)
        if positions.empty:
            print("\nNo open positions.")
        else:
            print(f"\nOpen positions:\n{positions.to_string(index=False)}")

        # Strategy signal
        signal = sma_crossover(df)
        label = {1: "LONG", -1: "SHORT", 0: "FLAT"}[signal]
        print(f"\nSMA crossover signal: {label}")

    finally:
        disconnect()


if __name__ == "__main__":
    main()
