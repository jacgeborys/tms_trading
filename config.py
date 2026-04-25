SYMBOL = "US500.pro"
LEVERAGE = 20
LOT_SIZE = 0.1          # default lot size for orders
MAGIC = 20250101        # magic number to tag bot's own trades
SLIPPAGE = 3            # max slippage in points

# Candle granularity options (MT5 timeframe constants loaded at runtime)
DEFAULT_TIMEFRAME = "M5"   # 5-minute bars

TIMEFRAME_MAP = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  16385,
    "H4":  16388,
    "D1":  16408,
}
