# TMS — US500 Trading Tools

Portfolio management and analysis scripts for **US500.pro** (S&P 500 CFD) via MetaTrader 5. Python 3.8.

Current strategy: **buy-and-hold accumulation** with ladder pending orders. Research phase (power-hour scalping) concluded — archived.

> **For Claude Code:** Read `CLAUDE.md` for full project state, verified findings, and architecture conventions.

---

## Status (Oct 2026)

- Holding a long US500.pro position plus a buy-limit ladder down to 7000.
- `03a_deploy_ladder.py` now has a margin guard and models margin as fixed at entry price (OANDA behaviour). Current ladder fails the guard: orders below ~7,530 sit under 100% margin level.
- Sep 16, 2026 rollover was paid (positions held). **Next rollover: Dec 16, 2026** — close by 22:55 Polish time with `04_rollover_close.py --execute`.

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
python 01d_rollover_tax_calc.py  # swap cost vs tax scenario analysis
python 01e_tower_chart.py        # position tower by entry price + top-heaviness score

python 02_support_levels.py      # support zone detection for ladder placement

python 03_manage_orders.py list  # list / place / cancel individual pending orders
python 03a_deploy_ladder.py      # deploy proposed ladder (dry-run by default, --execute to apply)
                                 # margin guard refuses orders below --min-ml (100%); raise --floor or --force

python 04_rollover_close.py      # close all positions before rollover (dry-run default, retries failures)
python 04a_rollover_reopen.py --price 7899 --lots 0.260  # reopen via buy-limits

cd monte_carlo && python main.py # Monte Carlo scenario sim -> monte_carlo/output/
```

---

## Project structure

```
tms/
├── 01_fetch_data.py             fetch OHLCV for US500 + correlated instruments
├── 01a_account_history.py       account dashboard (equity / margin / P&L)
├── 01b_positions_chart.py       position map + stress-test scenarios
├── 01d_rollover_tax_calc.py     rollover tax analysis
├── 01e_tower_chart.py           position tower visualization
├── 02_support_levels.py         support zone detection (swing lows + volume profile)
├── 03_manage_orders.py          place / cancel / list individual pending orders
├── 03a_deploy_ladder.py         batch deploy proposed ladder (diff-based, dry-run default)
├── import_02.py                 import shim for 02_support_levels (digit-prefixed module)
├── 04_rollover_close.py         close all positions before rollover
├── 04a_rollover_reopen.py       buy-limit ladder to reopen after rollover
├── config.py                    SYMBOL / MAGIC / leverage — imported by all trading scripts
├── mt5_client.py                MT5 connect/disconnect wrappers
├── data.py                      candle and tick data helpers
├── data/                        OHLCV pickles, rollover ledger, ladder audit (gitignored)
├── monte_carlo/                 standalone Monte Carlo scenario simulation
└── archive/                     concluded research phase (power-hour strategy)
```
