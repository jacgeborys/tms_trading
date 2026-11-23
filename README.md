# TMS — US500 Trading Tools

Portfolio management and analysis scripts for **US500.pro** (S&P 500 CFD) via MetaTrader 5. Python 3.8.

Current strategy: **buy-and-hold accumulation** with ladder pending orders. Research phase (power-hour scalping) concluded — archived.

> **For Claude Code:** Read `CLAUDE.md` for full project state, verified findings, and architecture conventions.

---

## Status (Nov 2026)

- **Partial stop-out Nov 5–20, 2026** (−12.5% drop from the Oct peak): 64 positions / 0.327 lots force-closed. 24 positions (0.024 lots, avg entry ~7,588) survive; no pending orders.
- `03a_deploy_ladder.py` now checks **survival below the last rung**: `--fit` keeps the most rungs (nearest first) that survive a further fall to `--survive-pct` (default 20%) with ML ≥ `--survive-ml` (default 100%). `--deposit X` simulates added cash (dry-run only). `--execute` is refused if the survival check fails.
- **Nov 23 restart:** closed the 24 survivors (bought at ~7,588; they alone would have been stopped out ~5% lower). Deployed a fitted ladder: 9 × 0.001L buy-limits 7,190→7,110, sized to survive −15% (to ~6,117) above stop-out. No open positions at deploy time. A top-up was pending; re-run `python 03a_deploy_ladder.py --fit --survive-pct 15 --survive-ml 50` once it lands to extend the ladder (diff-based, keeps existing rungs).
- **Next rollover: Dec 16, 2026** — close by 22:55 Polish time with `04_rollover_close.py --execute`.

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
python 03a_deploy_ladder.py --fit --deposit 2000   # how many rungs survive a -20% fall with 2000 PLN added

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
