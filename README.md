# TMS — US500 Trading Tools

Portfolio management and analysis scripts for **US500.pro** (S&P 500 CFD) via MetaTrader 5. Python 3.8.

Current strategy: **buy-and-hold accumulation** with ladder pending orders. Research phase (power-hour scalping) concluded — archived.

> **For Claude Code:** Read `CLAUDE.md` for full project state, verified findings, and architecture conventions.

---

## Setup

1. **MetaTrader 5** — install the terminal, log in with your broker account, leave it running
2. **Python 3.8** — required for the `MetaTrader5` package (Windows only)
3. **Install dependencies:** `pip install MetaTrader5 pandas matplotlib numpy`
4. **Edit `config.py`** — change `SYMBOL` to match your broker's naming (e.g. `"US500"`, `"SPX500"`)

---

## Scripts

```bash
python 01_fetch_data.py          # refresh M5/H1 price data (auto-skips if <12h old)
python 01a_account_history.py    # equity curve, rollover costs, margin chart
python 01b_positions_chart.py    # live position map + what-if equity at price drops
python 01c_ladder_calc.py        # ladder safety: margin level at each rung trigger
python 01d_rollover_tax_calc.py  # swap cost vs tax scenario analysis
python 01e_tower_chart.py        # position tower by entry price + top-heaviness score

python 02_support_levels.py      # support zone detection for ladder placement

python 03_manage_orders.py list  # list / place / cancel individual pending orders
python 03a_deploy_ladder.py      # deploy proposed ladder (dry-run by default, --execute to apply)
```

---

## Project structure

```
tms/
├── 01_fetch_data.py             fetch OHLCV for US500 + correlated instruments
├── 01a_account_history.py       account dashboard (equity / margin / P&L)
├── 01b_positions_chart.py       position map + stress-test scenarios
├── 01c_ladder_calc.py           ladder order safety calculator
├── 01d_rollover_tax_calc.py     rollover tax analysis
├── 01e_tower_chart.py           position tower visualization
├── 02_support_levels.py         support zone detection (swing lows + volume profile)
├── 03_manage_orders.py          place / cancel / list individual pending orders
├── 03a_deploy_ladder.py         batch deploy proposed ladder (diff-based, dry-run default)
├── import_02.py                 import shim for 02_support_levels (digit-prefixed module)
├── config.py                    symbol, leverage, magic number constants
├── mt5_client.py                MT5 connect/disconnect wrappers
├── data.py                      candle and tick data helpers
├── data/                        OHLCV pickles (gitignored, auto-generated)
└── archive/                     concluded research phase (power-hour strategy)
```
