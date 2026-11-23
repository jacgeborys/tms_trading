# TMS — US500 Trading Tools

Portfolio management and analysis scripts for **US500.pro** (S&P 500 CFD) via MetaTrader 5. Python 3.8.

Current strategy: **buy-and-hold accumulation** with ladder pending orders. Research phase (power-hour scalping) concluded — archived.

> **For Claude Code:** Read `CLAUDE.md` for full project state, verified findings, and architecture conventions.

---

## Status (Nov 2026)

- **Partial stop-out Nov 5–20, 2026** (−12.5% drop from the Oct peak): 64 positions / 0.327 lots force-closed. 24 positions (0.024 lots, avg entry ~7,588) survive; no pending orders.
- `03a_deploy_ladder.py` now checks **survival below the last rung**: `--fit` keeps the most rungs (nearest first) that survive a further fall to `--survive-pct` (default 20%) with ML ≥ `--survive-ml` (default 100%). `--deposit X` simulates added cash (dry-run only). `--survive-from-peak` measures the survival fall from the 250-day high instead of the current price: looser in a dip, strict near all-time highs. `--execute` is refused if the survival check fails.
- **Nov 23 restart:** closed the 24 survivors (bought at ~7,588; they alone would have been stopped out ~5% lower). Deployed a fitted ladder 7,200→7,095; the 7,200 rung filled. Later the same day, at the user's request, moved the whole ladder up to near price and extended it after a further 1,000 PLN deposit. Later switched to **survival measured from the peak**: `--fit --survive-from-peak --survive-pct 20 --survive-ml 50` (peak 7,898 → stop-out line ~6,318, −13% from 7,299). State after a further 1,000 PLN deposit: balance 5,388 PLN, 8 open (0.008L, 7,200–7,290), 15 buy-limits (0.015L, 7,310→7,228). With everything filled, the stop-out line is ~6,318 (ML 61%, ~1,046 PLN equity left). Note: `--fit` fills nearest-first, so each re-run after a rally moves the ladder up. After further top-ups, re-run `python 03a_deploy_ladder.py --fit --survive-pct 15 --survive-ml 50` to extend the ladder (diff-based, keeps existing rungs).
- **Daily auto re-fit:** Windows task "TMS daily ladder refit" runs `03b_daily_refit.bat` Mon–Fri 16:00 (interactive only, so the PC must be on with MT5 logged in). It runs `03a --fit --survive-from-peak --survive-pct 20 --survive-ml 50 --execute` and skips Mon–Thu of rollover weeks. Log: `logs/03b_daily_refit.log`. To switch it to log-only, add `--dry-run` to the .bat. To remove it: `schtasks /Delete /TN "TMS daily ladder refit" /F`.
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
python 03b_daily_refit.py --dry-run  # what the daily scheduled re-fit would do (log: logs/03b_daily_refit.log)

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
├── 03b_daily_refit.py           unattended daily re-fit (Task Scheduler, skips rollover week)
├── 03b_daily_refit.bat          launcher for the scheduled task
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
