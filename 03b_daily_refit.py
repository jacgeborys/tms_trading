"""
03b_daily_refit.py -- Unattended daily ladder re-fit (run by Windows Task Scheduler).

Runs 03a_deploy_ladder.py with the agreed survival rule and --execute, and
appends everything to logs/03b_daily_refit.log.

Risk glide path (RISK_TIERS): survive -20% from peak below 10k PLN equity,
-25% from 10k, -30% from 20k. Only ever tightens -- the tightest tier
reached is kept in data/03b_risk_tier.json even if equity later falls.

Skips:
  - weekends
  - rollover week: Monday-Thursday around each OANDA rollover
    (Wednesday before the 3rd Friday of Mar/Jun/Sep/Dec), so no new
    orders are added just before positions are closed for the rollover.

Usage:
  python 03b_daily_refit.py            # what the scheduled task runs
  python 03b_daily_refit.py --dry-run  # same, without --execute
"""

import sys
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent
_LOG = _ROOT / "logs" / "03b_daily_refit.log"

REFIT_ARGS = ["--fit", "--no-chart", "--survive-from-peak", "--survive-ml", "50"]

# Risk glide path: aggressive while the account is small, tighter as it grows.
# (equity threshold PLN, survival fall from peak %). Ratchets: never loosens.
RISK_TIERS = [(0, 20.0), (10_000, 25.0), (20_000, 30.0)]
_TIER_FILE = _ROOT / "data" / "03b_risk_tier.json"


def current_survive_pct(log) -> float:
    """Tier from live equity, never looser than the tightest tier reached so far."""
    import json
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    equity = mt5.account_info().equity
    mt5.shutdown()

    pct = max(p for thr, p in RISK_TIERS if equity >= thr)
    reached = 0.0
    if _TIER_FILE.exists():
        reached = float(json.loads(_TIER_FILE.read_text()).get("survive_pct", 0))
    if pct > reached:
        _TIER_FILE.parent.mkdir(parents=True, exist_ok=True)
        _TIER_FILE.write_text(json.dumps({"survive_pct": pct,
                                          "reached_at": datetime.now().isoformat(),
                                          "equity": round(equity, 2)}))
        if reached:
            log.write(f"Risk tier tightened: -{reached:.0f}% -> -{pct:.0f}% from peak\n")
    pct = max(pct, reached)
    log.write(f"Equity {equity:,.0f} PLN -> survive -{pct:.0f}% from peak\n")
    return pct


def rollover_date(year: int, month: int) -> date:
    """Wednesday before the 3rd Friday of the month."""
    d = date(year, month, 1)
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(days=14) - timedelta(days=2)


def in_rollover_week(today: date) -> bool:
    if today.month not in (3, 6, 9, 12):
        return False
    r = rollover_date(today.year, today.month)
    return r - timedelta(days=2) <= today <= r + timedelta(days=1)


def main():
    dry = "--dry-run" in sys.argv
    today = date.today()
    _LOG.parent.mkdir(parents=True, exist_ok=True)

    with open(_LOG, "a", encoding="utf-8") as log:
        log.write(f"\n{'#'*70}\n# {datetime.now():%Y-%m-%d %H:%M}  "
                  f"{'DRY RUN' if dry else 'EXECUTE'}\n{'#'*70}\n")
        if today.weekday() >= 5:
            log.write("Weekend -- skipped.\n")
            return
        if in_rollover_week(today):
            log.write(f"Rollover week ({rollover_date(today.year, today.month)}) "
                      f"-- skipped. Close positions with 04_rollover_close.py.\n")
            return

        try:
            pct = current_survive_pct(log)
        except Exception as e:
            log.write(f"Could not read equity ({e}) -- skipped.\n")
            return
        cmd = ([sys.executable, str(_ROOT / "03a_deploy_ladder.py")] + REFIT_ARGS
               + ["--survive-pct", f"{pct:g}"])
        if not dry:
            cmd.append("--execute")
        log.write("Running: " + " ".join(cmd[1:]) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**__import__("os").environ, "MPLBACKEND": "Agg",
                                   "PYTHONIOENCODING": "utf-8"})
        log.write(f"Exit code {proc.returncode}\n")


if __name__ == "__main__":
    main()
