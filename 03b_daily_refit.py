"""
03b_daily_refit.py -- Unattended daily ladder re-fit (run by Windows Task Scheduler).

Runs 03a_deploy_ladder.py with the agreed survival rule and --execute, and
appends everything to logs/03b_daily_refit.log.

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

REFIT_ARGS = ["--fit", "--no-chart", "--survive-from-peak",
              "--survive-pct", "20", "--survive-ml", "50"]


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

        cmd = [sys.executable, str(_ROOT / "03a_deploy_ladder.py")] + REFIT_ARGS
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
