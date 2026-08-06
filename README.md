# TMS — US500 Algorithmic Trading Bot

Research and backtesting system for **US500.pro** (S&P 500 CFD) via MetaTrader 5 on an OANDA demo account. Python 3.8. Currently in research/backtesting phase — no live orders placed yet.

> **For Claude Code:** Read `CLAUDE.md` for the full project state, research findings, architecture conventions, and exact next steps. This README is the orientation layer.

---

## Accounts

| | Value |
|---|---|
| Live account (user trades here) | personal account on OANDATMS-MT5 — log in manually in the MT5 terminal |
| Demo/research account (scripts use) | demo account on OANDATMS-MT5 — number in CLAUDE.md (not public) |
| Instrument | US500.pro — S&P 500 CFD |
| Contract size | 50 (1 lot = 50 × index price, profit in USD) |
| Leverage | 20× |
| Spread | Fixed 0.7 pts |
| Account currency | PLN |

---

## Quick start

```bash
python 01_fetch_data.py          # refresh M5/H1 price cache (if >12h old)
python 01a_account_history.py    # account health chart → results/charts/01a_account_history.png
python 02_account.py             # live balance / equity / open positions
python 05_strategy_search.py     # find entry filters → results/05_strategy_table.csv
python 06_backtest.py            # backtest power-hour strategy
```

---

## Project structure

```
tms/
├── CLAUDE.md                    full project state — read this first
├── README.md                    this file
│
├── 01_fetch_data.py             fetch M5/H1 OHLCV for US500 + correlated symbols
├── 01a_account_history.py       account health chart (equity / margin / P&L history)
├── 01b_positions_chart.py       live position map on price chart + what-if scenarios
├── 01c_ladder_calc.py           ladder safety calculator
├── 01d_rollover_tax_calc.py     swap vs Belka tax scenario analysis
├── 01e_tower_chart.py           position tower — vertical stack by entry price
├── 02_account.py                live account snapshot
├── 03_time_analysis.py          win-rate by UTC hour and day-of-week
├── 04_build_features.py         build 83-feature matrix → results/04_features_full.csv
├── 05_strategy_search.py        RF importance + single/pair condition search
├── 06_backtest.py               power_hour / baseline strategy backtests
├── 07_trade_charts.py           per-trade candle charts
├── 08_export_csv.py             power-hour bars CSV + overview chart
│
├── config.py                    symbol, leverage, magic number constants
├── mt5_client.py                connect() / disconnect() wrappers
├── cache.py                     disk I/O — all paths relative to project root
├── indicators.py                all indicator functions + add_all()
├── backtest.py                  simulate(), metrics(), SPREAD_PTS=0.7
├── power_hour.py                power-hour strategy (market/stop/limit entry modes)
├── baseline.py                  rule-based baseline strategy
│
├── data/                        OHLCV pickles — gitignored, auto-refreshed if >12h old
├── models/                      trained models — gitignored
└── results/                     CSVs, JSONs, charts — gitignored
```

---

## Core research finding

**Hour 22 UTC (17:00 ET) has a statistically significant long bias — the only one that survives Bonferroni correction across all 24 hours.**

| Mode | Win rate | Break-even needed | Gap |
|---|---|---|---|
| Market entry | 56.4% | 58.2% | −1.8pp |
| Stop entry (+0.15×ATR) | 57.3% | 58.2% | −0.9pp |

The signal is real. The spread eats the edge. Next step: find a secondary filter that pushes WR above 58.5% with N > 150 trades. See `results/05_strategy_table.csv`.

---

## 01a_account_history.py

Reconstructs full account equity/margin/P&L history from MT5 deal history + cached M5 prices. Produces a 3-panel chart:

1. **Equity vs margin** — stacked band (orange = margin used, teal = free margin, red = margin call zone), equity line, balance step, deposit markers
2. **Capital composition** — deposited cash + realised P&L + unrealised P&L
3. **Margin level %** — only shown when positions are open, capped at 1000%

**How upnl is reconstructed (important for future debugging):**

The instrument is USD-denominated but the account is PLN. Rather than fetching a USDPLN exchange rate, we anchor to actual MT5 PLN values and use price ratios so USDPLN cancels out:

- **Current open positions:** `upnl(bar) = p.profit × (close(bar) − entry) / (last_close − entry)`
  `p.profit` is the exact PLN profit from MT5 at the last cached bar.
- **Historical closed positions:** `upnl(bar) = net_pln × (close(bar) − entry) / (close_at_close_bar − entry)`
  `net_pln = sum(d.profit + commission + swap + fee)` from deal history — already in PLN.
- **Margin bands:** `entry_price × vol × cs / leverage × margin_scale` where `margin_scale = info.margin / Σ(entry × vol × cs / leverage)` ≈ PLN/USD rate.
- `seen_pids` is pre-populated with currently open position IDs to prevent partial-close double-counting.

```bash
python 01a_account_history.py             # last 6 months (default)
python 01a_account_history.py --months 3
python 01a_account_history.py --all       # full history from first position
```

---

## Still to build

- `orders.py` — place/close market and pending orders via MT5 (`MAGIC = 20250101`)
- Walk-forward validation of the best filtered strategy
- Position sizing (Kelly fraction, start at 25%)
- Live execution loop with configurable check interval
- Short strategy — no signal strong enough yet, do not build
