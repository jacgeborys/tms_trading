# TMS Trading Bot — Project State

## What this project is

Algorithmic trading bot for **US500.pro** (S&P 500 CFD) via **MetaTrader 5 connected to OANDA live account**. Written in Python 3.8. **Current phase: buy-and-hold accumulation with ladder pending orders.** Power-hour research phase concluded (WR 56.4%, below 58.2% break-even) — archived.

---

## Account & instrument facts

| Property | Value |
|---|---|
| Account | *(redacted)* on OANDATMS-MT5 |
| Balance | *(redacted — read live from MT5)* |
| Instrument | US500.pro |
| Contract size | 50 (1 lot = 50 × index price) |
| Leverage | 20× |
| Spread | **Fixed 0.7 pts at all hours** (confirmed from 66,959 ticks) |
| Lot limits | 0.001 – 4.0, step 0.001 |
| Price feed | Tracks **cash S&P 500 index** continuously — no futures price gap at rollover |
| Overnight swap | None (no daily financing charge) |
| Quarterly rollover | Yes — charged as swap on the position at rollover date. OANDA rolls on **Wednesday before 3rd Friday** of Mar/Jun/Sep/Dec. 2026 dates: Mar 18, Jun 17, Sep 16, Dec 16. Source: `data/tabela_rolowan_1.01.2026_pl_mt5_0.pdf` |
| Rollover cost (observed) | March 2026: −9.52 PLN / 0.001 lot; June 2026: −12.53 PLN / 0.001 lot (back-calculated from open positions via `print_rollover_model()` in 01a) |
| MT5 data depth | M1: ~7 weeks, M5: ~8.5 months, H1: ~8.5 years, D1: ~16 years |

The user trades exclusively US500 with frequent small operations (scalping / intraday).

---

## Project structure

```
tms/
├── CLAUDE.md                   ← this file
│
├── ── ACTIVE SCRIPTS ──────────────────────────────────────────────────
├── 01_fetch_data.py            fetch M5/H1 OHLCV for US500 + correlated instruments
├── 01a_account_history.py      equity curve, rollover model, margin chart (main daily script)
├── 01b_positions_chart.py      live position map + what-if equity at price drop scenarios
├── 01d_rollover_tax_calc.py    swap cost vs Belka tax scenario analysis (Sep + Dec rollovers)
├── 01e_tower_chart.py          position tower: vertical stack by entry price, top-heaviness score
│
├── ── MARKET ANALYSIS ────────────────────────────────────────────────
├── 02_support_levels.py        support zone detection (swing lows + volume profile + round numbers)
│
├── ── ORDER MANAGEMENT ──────────────────────────────────────────────
├── 03_manage_orders.py         place / cancel / list individual pending orders
├── 03a_deploy_ladder.py        batch deploy proposed ladder (diff-based, dry-run default)
├── import_02.py                import shim for 02_support_levels (digit-prefixed module)
│
├── ── ROLLOVER ──────────────────────────────────────────────────────
├── 04_rollover_close.py        market-sell all positions before rollover
├── 04a_rollover_reopen.py      buy-limit ladder to reopen after rollover
│
├── ── LIBRARY MODULES ─────────────────────────────────────────────────
├── config.py                   symbol, leverage, magic number constants
├── mt5_client.py               connect() / disconnect() wrappers
├── data.py                     get_candles(), get_candles_range(), get_tick()
│
├── data/
│   ├── US500_pro_M5.pkl        raw OHLCV (auto-refreshes if >12h old)
│   ├── US100_pro_M5.pkl
│   ├── US30_pro_M5.pkl
│   ├── GOLD_pro_M5.pkl
│   ├── rollover_ledger.csv     persistent per-rollover cost history (written by 01a)
│   └── ladder_deployed.csv    audit trail of last deployed ladder (written by 03a)
│
└── archive/                    ── POWER-HOUR RESEARCH PHASE (concluded, WR < break-even) ──
    ├── 02_account.py           (superseded by 01a)
    ├── 03_time_analysis.py     (key finding: 22 UTC WR 56.4% — archived)
    ├── 04_build_features.py
    ├── 05_strategy_search.py
    ├── 06_backtest.py
    ├── 07_trade_charts.py
    ├── 08_export_csv.py
    ├── 09_explore_chart.py
    ├── account.py / backtest.py / baseline.py / cache.py
    ├── features.py / heatmap_search.py / indicators.py / labeler.py
    └── model.py / pipeline.py / power_hour.py / strategy.py
```

---

## Research phase summary (archived in archive/)

- **XGBoost / ML**: AUC ≈ 0.50 — dead end. No bar-level signal in 8 months of M5 data.
- **Power-hour (22 UTC)**: Long WR 56.4%, but break-even needs 58.2% (spread = 0.7 pts). Gap −1.8pp. Not tradeable profitably without a secondary filter that hasn't been found yet.
- All research code moved to `archive/`. Strategy pivoted to buy-and-hold accumulation.

---

## Current strategy — buy-and-hold accumulation with ladder

Holding a large long position in US500.pro built up via buy-limit ladder orders. Not actively scalping. Daily workflow:

1. `python 01_fetch_data.py` — refresh OHLCV if >12h old
2. `python 01a_account_history.py` — equity curve, rollover costs, margin level chart
3. `python 01b_positions_chart.py` — check current position map + what-if scenarios
4. `python 03a_deploy_ladder.py` — review ladder cascade (dry-run), deploy with `--execute`

**Next rollover: September 16, 2026** — close all positions before daily break, reopen when trading resumes. Saves ~42× vs paying the rollover (~12.5 PLN/0.001 lot vs ~0.26 PLN spread). See **Rollover execution plan** below for exact timing.

**Tax consideration (live account):** Closing positions crystallizes unrealized gains as taxable income under Polish Belka tax (19% flat, PIT-38, net annual, loss carryforward 5 years at max 50%/year). Key insight: swap costs are tax-deductible when positions eventually close, so effective swap cost = face × 0.81. Run `01d_rollover_tax_calc.py` for full scenario analysis before each rollover decision.

**Rollover cost log:** `data/rollover_ledger.csv` — updated automatically by `01a_account_history.py`.

### Rollover execution plan

Rollover is applied **after the daily trading session close at 22:59 Polish time** (confirmed by OANDA TMS broker advisor). In summer (CEST = UTC+2) this is **20:59 UTC**. Trading halts at 22:58 M1 bar and resumes at ~00:00 UTC next day (~62 min break).

**Execution sequence for Sep 16, 2026 (summer/CEST):**

| Step | Polish time | UTC | Action |
|------|------------|-----|--------|
| 1 | 22:50–22:55 | 20:50–20:55 | Close all positions (market sell) |
| 2 | 22:59 | 20:59 | Trading day ends — rollover applied to open positions |
| 3 | ~00:00+1d | ~00:00+1d | Trading resumes — reopen full position (market buy) |

**Why market orders, not limits:** The 0.7 pt spread cost (~29 PLN for 0.210 lots) is 1.1% of the ~2,625 PLN rollover saved. Optimizing execution with limits risks missing the fill and chasing price after the gap.

**Overnight gap risk (from 117 daily breaks, Jan–Jul 2026):**
- Median gap: +0.2 pts (negligible)
- 90% of gaps fall within ±6 pts (≈ ±252 PLN for 0.210 lots)
- 99% within −13 / +22 pts
- Worst observed: +52.5 pts (March 18, a crash-bounce day — unrelated to rollover)

**No evidence of mass close-reopen behavior on rollover dates.** Volume in the last trading hour on June 17 rollover was normal-to-low (609 ticks in last M5 bar vs 2,940 average). US500.pro is an OTC CFD — the broker is the counterparty, and most retail traders don't optimize for rollover costs.

**Pre-rollover checklist:**
1. Check economic calendar — avoid closing during FOMC or major news (June 17 had a −56 pt crash at 20:00 UTC, likely macro-driven)
2. Verify position sizes via `01b_positions_chart.py`
3. Confirm the rollover hasn't been applied early (check `position.swap` field)
4. Close at 22:50–22:55 Polish time, reopen at 00:00 UTC next day

---

## Architecture conventions

- **Labels:** `make_long_labels(df, tp_mult=1.0, sl_mult=1.0, lookahead=24)` — entry at bar close, pessimistic tie-break (same-bar TP+SL → loss). Same for short.
- **Spread:** Always deducted. `SPREAD_PTS = 0.7` in `backtest.py`.
- **Cache freshness:** OHLCV pickles auto-refresh >12h old. No MT5 needed if data is fresh.
- **Walk-forward:** Expanding windows at 40/55/70/85%; last 15% = hold-out.
- **Python 3.8** — MT5 wheel is cp38. Use `from typing import Dict` for type hints.
- **Chart style:** `plt.style.use("dark_background")` everywhere.
- **Magic number:** `MAGIC = 20250101` for live bot orders.
- **Output prefixes:** Each numbered script produces files prefixed with its own number (e.g., `04_*.csv`, `05_*.csv`).

---

## How to resume

```bash
# Refresh OHLCV data (required if >12h since last run)
python 01_fetch_data.py

# Full account dashboard: equity curve, rollover model, margin chart
python 01a_account_history.py

# Live position map + what-if equity drop scenarios
python 01b_positions_chart.py

# Market analysis + ladder proposal
python 02_support_levels.py

# Deploy ladder (dry-run first, then --execute)
# Includes cascade chart: equity/ML/margin as price drops through ladder
python 03a_deploy_ladder.py
python 03a_deploy_ladder.py --execute
```

---

## Still to build

- Rollover alert: notify on rollover day (Sep 16) that positions must be closed by 22:55 Polish time

---

## Verified empirical findings

Things that required real investigation to establish — do not re-derive without new evidence.

| Finding | Evidence | Implication |
|---|---|---|
| US500.pro tracks cash S&P 500, not futures | Zero price gap in M5 data around March 18 and June 17 2026 rollovers (max bar-to-bar move 0.3 pts, normal noise) | Close-reopen at rollover is not a wash — you reopen at the same price, saving the full swap cost |
| Rollover swap is not visible as a deal in MT5 history | Scanned all deal types (BALANCE, CORRECTION, DIVIDEND, INTEREST, etc.) — none correspond to rollover charges | Swap accumulates silently on `position.swap` field; only visible via `mt5.positions_get()` |
| Rollover costs: March −9.52, June −12.53 PLN / 0.001 lot | Back-calculated from survivorship grouping of open positions by rollover count | Projected September cost ~−12–13 PLN / 0.001 lot at current rates |
| Rollover cutoff is 22:59 Polish time (after daily session close) | Confirmed by OANDA TMS broker advisor ("po zamknięciu doby handlowej czyli po 22:59"). Consistent with June 17 data: positions opened at 16:47 and 20:05 UTC (both before 20:59 UTC = 22:59 CEST) got charged | Close positions before 22:55 Polish time on rollover day; reopen after 00:00 UTC next day |
| Daily session break: 22:58 UTC → ~00:00 UTC (~62 min) | M1 data shows last bar at 22:58, first bar at 00:00 on every trading day | You're unavoidably unhedged for ~62 min during the break |
| Overnight gap is small and symmetric | 117 daily breaks: median +0.2 pts, P5/P95 = −6/+5 pts, worst +52.5 pts (crash-bounce) | Gap risk for 0.210 lots is ~±252 PLN at P95 — small vs ~2,625 PLN rollover saved |
| No mass close-reopen on rollover dates | June 17 rollover: last M5 bar volume 609 ticks vs 2,940 daily average. No selling pressure spike before break | No need to worry about adverse price impact from other traders closing |
| S&P dividend yield (~1.3%) < Fed funds rate (~4–4.5%) | Standard macro data | Rollover swap will remain negative for longs until rates fall below dividend yield (~ZIRP conditions). Not expected near-term |
| Power-hour edge (22 UTC, WR 56.4%) is not profitable | Break-even requires 58.2% WR given 0.7 pt spread. Gap −1.8pp. Confirmed across market/stop/limit entry modes | Research phase closed. No secondary filter found that clears the gap |

---

## Known bugs fixed

| File | Bug | Fix |
|---|---|---|
| `04_build_features.py` | `full[float_cols] = full[float_cols].round(4)` raises pandas `__setitem__` error after `pd.concat` + `reset_index` | Moved rounding to write time: `full.round(4).to_csv(...)` |
| `04_build_features.py` | `meta["day_of_week"]` (string "Mon") collided with `feats["day_of_week"]` (float) after `pd.concat`, creating duplicate columns | Renamed meta column to `meta["day_name"]` |
| `05_strategy_search.py` | `feat["day_of_week"].between(1, 3)` crashed on object dtype (string from old CSV) | Added deduplication (`~feat.columns.duplicated(keep="last")`) + `pd.to_numeric(errors="coerce")` on load |
