"""
MT5 connection wrapper. Call connect() once at startup, disconnect() at exit.
All other modules import `mt5` from here so there's one shared connection.
"""

import MetaTrader5 as mt5


def connect() -> bool:
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    print(f"Connected | Account {info.login} | {info.server}")
    print(f"Balance: {info.balance:.2f} {info.currency}  Equity: {info.equity:.2f}")
    return True


def disconnect():
    mt5.shutdown()
