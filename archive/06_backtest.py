"""
04_backtest.py — Run strategy backtests and compare results.

Runs the power-hour strategy (market / stop / limit entry modes)
plus the rule-based baseline. Prints comparison table and saves
equity charts + trade logs to results/.

Usage:
  python 04_backtest.py                # run everything
  python 04_backtest.py power_hour     # power hour only
  python 04_backtest.py baseline       # baseline only
  python 04_backtest.py pipeline       # full pipeline (includes ML, slower)
"""

import sys
import power_hour
import baseline as baseline_module
import cache
from indicators import add_all
from labeler import make_long_labels, make_short_labels
from backtest import metrics, print_metrics


def run_baseline():
    print("\n" + "=" * 55)
    print("  BASELINE STRATEGY  (rule-based mean reversion)")
    print("=" * 55)
    raw = cache.load_ohlcv("US500.pro", "M5")
    if raw is None:
        print("No data. Run 01_fetch_data.py first.")
        return
    df = add_all(raw)
    tp, sl = 1.0, 1.0
    long_lbl  = make_long_labels(df,  tp_mult=tp, sl_mult=sl, lookahead=24)
    short_lbl = make_short_labels(df, tp_mult=tp, sl_mult=sl, lookahead=24)
    all_trades, l_m, s_m, all_m = baseline_module.run(df, long_lbl, short_lbl, tp, sl)
    print_metrics(l_m,   "Baseline — LONG")
    print_metrics(s_m,   "Baseline — SHORT")
    print_metrics(all_m, "Baseline — ALL")
    cache.save_metrics(all_m, "06_baseline_metrics")
    cache.save_trades(all_trades, "06_baseline_trades")


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if mode in ("all", "power_hour"):
        power_hour.main()

    if mode in ("all", "baseline"):
        run_baseline()

    if mode == "pipeline":
        import pipeline
        pipeline.main()


if __name__ == "__main__":
    main()
