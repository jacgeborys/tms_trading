"""
monte_carlo/main.py — Orchestration entry point.

Usage:
    cd monte_carlo
    python main.py

Outputs:
    output/scenario_analysis.png
    output/summary.txt
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# Allow running as `python monte_carlo/main.py` from the project root too
_here       = os.path.dirname(os.path.abspath(__file__))
_parent     = os.path.dirname(_here)
sys.path.insert(0, _parent)   # gives access to mt5_client from parent project
sys.path.insert(0, _here)     # monte_carlo/ must be first so its config.py wins

import config
from simulation  import fetch_calibration, generate_paths, compute_daily_lows, simulate_fills
from analysis    import (compute_portfolio, fill_probabilities, liquidation_curve,
                          center_of_weight, optimal_lots)
from visualisation import plot_all

OUTPUT_DIR = os.path.join(_here, "output")


def fetch_live_state():
    """
    Connect to MT5 and return (balance, equity, margin_used, current_price,
    open_positions, pending_orders) as live values.
    Falls back to config defaults if MT5 is unavailable.
    """
    try:
        import MetaTrader5 as mt5

        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

        info         = mt5.account_info()
        balance      = info.balance
        equity       = info.equity
        margin_used  = info.margin

        symbol        = "US500.pro"
        tick          = mt5.symbol_info_tick(symbol)
        current_price = tick.bid if tick else config.CURRENT_PRICE

        # Open positions — buy only (type 0)
        raw_pos = mt5.positions_get(symbol=symbol) or []
        open_positions = [
            (p.volume, p.price_open)
            for p in raw_pos
            if p.type == 0   # buy
        ]

        # Pending buy-limit orders
        raw_orders = mt5.orders_get(symbol=symbol) or []
        pending_orders = [
            (o.volume_initial, o.price_open)
            for o in raw_orders
            if o.type == 2   # ORDER_TYPE_BUY_LIMIT
        ]

        mt5.shutdown()

        print(f"  Live: balance={balance:,.0f}  equity={equity:,.0f}  "
              f"margin={margin_used:,.0f}  price={current_price:.1f}")
        print(f"  {len(open_positions)} open positions, "
              f"{len(pending_orders)} pending buy limits")

        return balance, equity, margin_used, current_price, open_positions, pending_orders

    except Exception as e:
        print(f"  MT5 unavailable ({e}) — using config defaults")
        return (config.BALANCE, config.EQUITY, config.MARGIN_USED,
                config.CURRENT_PRICE, config.OPEN_POSITIONS, config.PENDING_ORDERS)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Monte Carlo Portfolio Simulator — US500.pro CFD")
    print("=" * 60)

    # 0. Fetch live account state from MT5 ────────────────────────────────────
    print("\n[0/6] Connecting to MT5 for live positions...")
    balance, equity, margin_used, current_price, open_positions, pending_orders = \
        fetch_live_state()

    if not open_positions:
        print("  WARNING: no open positions found — check MT5 connection or config")
    if not pending_orders:
        print("  WARNING: no pending orders found")

    # 1. Calibrate from live SPX history ─────────────────────────────────────
    print("\n[1/6] Fetching calibration data from Yahoo Finance...")
    mu_ann, sigma_ann = fetch_calibration()

    # 2. Generate price paths ─────────────────────────────────────────────────
    print(f"\n[2/6] Generating {config.N_PATHS:,} paths × {config.N_DAYS} days...")
    t0     = time.time()
    prices = generate_paths(
        S0      = current_price,
        mu      = mu_ann,
        sigma   = sigma_ann,
        n_paths = config.N_PATHS,
        n_days  = config.N_DAYS,
        df      = config.T_DF,
    )
    daily_lows = compute_daily_lows(prices, sigma_ann)
    print(f"  Done in {time.time() - t0:.1f}s  |  "
          f"price range at day 90: [{prices[:, -1].min():.0f}, {prices[:, -1].max():.0f}]")

    # 3. Simulate pending order fills ─────────────────────────────────────────
    print("\n[3/6] Simulating pending order fills...")
    fill_day   = simulate_fills(prices, daily_lows, pending_orders)
    fill_probs = fill_probabilities(fill_day, horizons=(30, 60, 90))
    avg_fills  = ((fill_day >= 0) & (fill_day <= 90)).sum(axis=1).mean()
    print(f"  Average fills by day 90: {avg_fills:.1f} / {len(pending_orders)}")

    # 4. Compute portfolio metrics ─────────────────────────────────────────────
    print("\n[4/6] Computing portfolio metrics...")
    portfolio = compute_portfolio(
        prices, fill_day,
        open_positions, pending_orders,
        balance,
    )
    liq = liquidation_curve(portfolio["stopped"])

    # 5. Center-of-weight and optimal lots ────────────────────────────────────
    print("\n[5/6] Computing center of weight and optimal lot distribution...")
    cow_prices = np.arange(config.COW_PRICE_MIN, config.COW_PRICE_MAX + 1,
                           config.COW_PRICE_STEP, dtype=float)
    cow_scores = center_of_weight(prices, sigma_ann, cow_prices)
    cow_peak   = float(cow_prices[np.argmax(cow_scores)])

    opt = optimal_lots(
        prices, fill_day,
        pending_orders,
        cow_scores, cow_prices,
        portfolio,
        balance,
        open_positions=open_positions,
    )

    # 6. Summary ───────────────────────────────────────────────────────────────
    ml = portfolio["margin_level"]
    eq = portfolio["equity"]

    liq_30 = liq[30]  * 100
    liq_60 = liq[60]  * 100
    liq_90 = liq[90]  * 100

    med_eq_30 = float(np.median(eq[:, 30]))
    med_eq_60 = float(np.median(eq[:, 60]))
    med_eq_90 = float(np.median(eq[:, 90]))
    p5_eq_90  = float(np.percentile(eq[:, 90], 5))

    med_ml_90 = float(np.nanmedian(
        np.where(np.isinf(ml[:, 90]), np.nan, ml[:, 90])
    ))

    lines = [
        "Monte Carlo Summary — US500.pro CFD Portfolio",
        "=" * 55,
        f"Paths       : {config.N_PATHS:,}",
        f"Horizon     : {config.N_DAYS} trading days",
        f"Calibration : μ={mu_ann:+.2%}/yr   σ={sigma_ann:.2%}/yr   t-df={config.T_DF}",
        f"Balance     : {balance:,.0f} PLN",
        f"Equity (T0) : {equity:,.0f} PLN",
        f"Margin used : {margin_used:,.0f} PLN",
        f"Price (T0)  : {current_price:.1f}",
        f"Open pos    : {len(open_positions)}",
        "",
        "── Liquidation probability (margin level < 100%) ──",
        f"  Day  30  :  {liq_30:.3f}%",
        f"  Day  60  :  {liq_60:.3f}%",
        f"  Day  90  :  {liq_90:.3f}%",
        "",
        "── Equity percentiles ──────────────────────────────",
        f"  Median  day 30  :  {med_eq_30:>10,.0f} PLN",
        f"  Median  day 60  :  {med_eq_60:>10,.0f} PLN",
        f"  Median  day 90  :  {med_eq_90:>10,.0f} PLN",
        f"  p5      day 90  :  {p5_eq_90:>10,.0f} PLN",
        f"  Median ML day 90:  {med_ml_90:>10,.1f}%",
        "",
        f"── Pending orders ──────────────────────────────────",
        f"  Avg fills by day 90 :  {avg_fills:.1f} / {len(pending_orders)}",
        f"  Center-of-weight px :  {cow_peak:.0f}",
        "",
        f"{'Price':>6}  {'Lots':>6}  {'p(30d)':>7}  {'p(60d)':>7}  {'p(90d)':>7}  "
        f"{'Opt lots':>9}",
        "-" * 55,
    ]
    for i, (lots, price) in enumerate(pending_orders):
        p30, p60, p90 = fill_probs[i] * 100
        lines.append(
            f"  {price:>5.0f}  {lots:>6.3f}  {p30:>6.1f}%  {p60:>6.1f}%  "
            f"{p90:>6.1f}%  {opt[i]:>9.3f}"
        )

    summary = "\n".join(lines)
    print("\n" + summary)

    out_txt = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSummary written to {out_txt}")

    # 7. Plot ──────────────────────────────────────────────────────────────────
    print("\n[6/6] Generating chart...")
    fig = plot_all(
        prices         = prices,
        portfolio      = portfolio,
        liq_curve      = liq,
        fill_probs     = fill_probs,
        cow_prices     = cow_prices,
        cow_scores     = cow_scores,
        cow_peak       = cow_peak,
        pending_orders = pending_orders,
        opt_lots       = opt,
        mu_ann         = mu_ann,
        sigma_ann      = sigma_ann,
    )

    out_png = os.path.join(OUTPUT_DIR, "scenario_analysis.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
