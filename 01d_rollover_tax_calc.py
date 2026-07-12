"""
01d_rollover_tax_calc.py — Swap cost vs Belka tax: scenario analysis for Sep + Dec 2026 rollovers.

Compares three strategies for the two remaining 2026 rollovers (Sep 16, Dec 16):

  Option A : Hold all year — pay Sep swap + pay Dec swap. No tax event in 2026.
  Option B : Close before Sep rollover, reopen. Pay Dec swap. Tax on Sep-close gain.
  Option D : Close before Sep rollover AND before Dec rollover. Tax on net annual gain.
             (Option C = pay Sep + close Dec is always dominated by D; excluded.)

Key accounting facts:
  - Polish Belka tax (PIT-38): 19% flat on NET realized capital gains per calendar year.
  - Gains and losses within the same year are netted before applying 19%.
  - Net annual loss carries forward 5 years, deductible at max 50% of original loss per year.
  - Swap accumulated on a position REDUCES the realized gain when the position is closed
    (swap is already negative in MT5 — it is part of the P&L). This means Option A's swap
    costs are partially tax-deductible when positions are eventually closed.
  - "After-tax effective swap cost" = face_value × (1 − 0.19) = face × 0.81.
    This is the true long-run cost of paying the swap, assuming the position is eventually closed.
  - USD/PLN rate assumed constant at current value for all future scenarios.
  - Lot count assumed fixed (no new positions added before rollovers).

Usage:
  python 01d_rollover_tax_calc.py
"""

import sys
from mt5_client import connect, disconnect
import MetaTrader5 as mt5

BELKA          = 0.19
SWAP_RATE      = -12.53   # PLN per 0.001 lot per rollover (observed June 2026; Sep/Dec estimated same)
SYMBOL         = "US500.pro"
SEP_LABEL      = "Sep 16, 2026"
DEC_LABEL      = "Dec 16, 2026"

# Price change scenarios to display (relative to current price)
SCENARIOS = [-0.20, -0.15, -0.10, -0.07, -0.05, -0.03, 0.00,
             +0.03, +0.05, +0.07, +0.10, +0.15, +0.20]


# ── Helpers ────────────────────────────────────────────────────────────────────

def profit_at(price, avg_entry, pln_per_point):
    """Estimated portfolio profit (PLN) if price moves to `price`."""
    return pln_per_point * (price - avg_entry)


def tax_on(gain):
    """Belka tax owed on a realized gain. Zero if gain <= 0."""
    return BELKA * gain if gain > 0 else 0.0


def loss_carryforward_note(loss):
    """Human-readable note about a tax loss carryforward."""
    annual = loss * BELKA * 0.50   # 50% of loss usable per year → max annual tax saving
    return f"Loss carryforward {loss:,.0f} PLN → up to {annual:,.0f} PLN/yr tax saving (5 yrs)"


def div(width=65):
    print("─" * width)


def header(title, width=65):
    div(width)
    print(f"  {title}")
    div(width)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to MT5...")
    if not connect():
        print("ERROR: Could not connect to MT5.")
        sys.exit(1)

    try:
        positions_raw = mt5.positions_get(symbol=SYMBOL)
        tick          = mt5.symbol_info_tick(SYMBOL)
    finally:
        disconnect()

    if not positions_raw:
        print(f"No open {SYMBOL} positions found.")
        sys.exit(0)
    if tick is None:
        print("ERROR: Could not fetch current tick.")
        sys.exit(1)

    # ── Derive portfolio constants ─────────────────────────────────────────────
    total_lots    = sum(p.volume for p in positions_raw)
    profit_curr   = sum(p.profit for p in positions_raw)   # MT5 profit, excludes swap
    swap_curr     = sum(p.swap   for p in positions_raw)   # negative, accumulated so far
    avg_entry     = sum(p.price_open * p.volume for p in positions_raw) / total_lots
    current_price = (tick.bid + tick.ask) / 2

    price_diff = current_price - avg_entry
    if abs(price_diff) < 1.0:
        print("ERROR: Current price too close to average entry — pln_per_point unreliable.")
        sys.exit(1)

    pln_per_point   = profit_curr / price_diff          # PLN change per 1-pt index move
    net_unrealized  = profit_curr + swap_curr            # what you'd realize if closing now
    swap_per_ro     = SWAP_RATE * (total_lots / 0.001)  # one rollover charge (negative)

    # ── Current state ──────────────────────────────────────────────────────────
    print()
    print("═" * 65)
    print("  SWAP vs BELKA TAX — 2026 rollover scenario analysis")
    print("═" * 65)
    print(f"  Current price         {current_price:>10,.1f} pts")
    print(f"  Weighted avg entry    {avg_entry:>10,.1f} pts")
    print(f"  Total lots            {total_lots:>10.3f}")
    print(f"  PLN per point         {pln_per_point:>10.2f} PLN/pt")
    print()
    print(f"  MT5 profit (ex-swap)  {profit_curr:>10,.2f} PLN")
    print(f"  Accumulated swap      {swap_curr:>10,.2f} PLN")
    print(f"  Net unrealized        {net_unrealized:>10,.2f} PLN")
    print()
    print(f"  Swap per rollover     {swap_per_ro:>10,.2f} PLN  ({SWAP_RATE} PLN/0.001 lot)")
    print(f"  Sep + Dec total       {2*swap_per_ro:>10,.2f} PLN  (face value)")
    print(f"  After-tax effective   {2*swap_per_ro*(1-BELKA):>10,.2f} PLN  (×0.81, deductible at future close)")
    print()
    print("  Assumptions: lots fixed, USD/PLN constant, swap rate same as June 2026.")
    print()

    # ── September 16 decision ─────────────────────────────────────────────────
    # Close Sep if swap saved > tax on Sep gain.
    # Gain at Sep close = profit_at_sep + swap_curr  (swap_curr is fixed until Sep rollover)
    # Breakeven: |swap_per_ro| = 0.19 × gain_sep  →  gain_sep = |swap_per_ro| / 0.19
    # With after-tax adjustment: |swap_per_ro| × 0.81 = 0.19 × gain_sep
    #   →  gain_sep = |swap_per_ro| × 0.81 / 0.19

    be_gain_sep_naive    = abs(swap_per_ro) / BELKA
    be_gain_sep_adjusted = abs(swap_per_ro) * (1 - BELKA) / BELKA

    be_price_sep_naive    = (be_gain_sep_naive    - swap_curr) / pln_per_point + avg_entry
    be_price_sep_adjusted = (be_gain_sep_adjusted - swap_curr) / pln_per_point + avg_entry

    header(f"SEPTEMBER 16 DECISION  ({SEP_LABEL})")
    print(f"  Close Sep if net unrealized at Sep is BELOW:")
    print(f"    Naive breakeven    {be_gain_sep_naive:>8,.0f} PLN  "
          f"→ price {be_price_sep_naive:,.0f} pts  ({(be_price_sep_naive/current_price-1)*100:+.1f}%)")
    print(f"    Adjusted breakeven {be_gain_sep_adjusted:>8,.0f} PLN  "
          f"→ price {be_price_sep_adjusted:,.0f} pts  ({(be_price_sep_adjusted/current_price-1)*100:+.1f}%)")
    print(f"  (Adjusted accounts for swaps being tax-deductible at eventual close.)")
    print()
    print(f"  Current net unrealized: {net_unrealized:,.0f} PLN")
    if net_unrealized < be_gain_sep_adjusted:
        verdict = "CLOSE before Sep rollover"
        reason  = "swap saving exceeds after-tax effective swap cost"
    elif net_unrealized < be_gain_sep_naive:
        verdict = "BORDERLINE — depends on whether you plan to hold long-term"
        reason  = "naive: close wins; adjusted (long-term hold): hold wins"
    else:
        verdict = "PAY the Sep swap"
        reason  = "tax cost exceeds both naive and after-tax breakeven"
    print(f"  → Verdict: {verdict}")
    print(f"    ({reason})")
    print()

    # ── December 16 decision ─────────────────────────────────────────────────
    # Key insight: Option D annual taxable gain = profit_at_dec + swap_curr
    # This is INDEPENDENT of Sep price (Sep and Dec gains net to the same total).
    # swap_curr is the swap accumulated up to today — it does NOT change between
    # now and Sep 16 (no rollover between now and Sep), and is realized at Sep close.
    # After reopening, the Dec position starts with zero swap.
    # At Dec close: gain from new position + Sep gain = profit_at_dec + swap_curr.
    #
    # Option A: 2026 cash = |swap_sep| + |swap_dec|, deferred tax still to come
    # Option B: 2026 cash = tax(profit_sep + swap_curr) + |swap_dec|
    # Option D: 2026 cash = tax(profit_dec + swap_curr), no swaps paid

    cost_A_face     = 2 * abs(swap_per_ro)
    cost_A_adjusted = cost_A_face * (1 - BELKA)   # after accounting for future tax deduction

    # Breakevens for Option D vs A (Dec price where D = A)
    # Naive:    0.19 × (profit_dec + swap_curr) = cost_A_face
    be_profit_dec_naive    = cost_A_face    / BELKA - swap_curr
    # Adjusted: 0.19 × (profit_dec + swap_curr) = cost_A_adjusted
    be_profit_dec_adjusted = cost_A_adjusted / BELKA - swap_curr

    be_price_dec_naive    = be_profit_dec_naive    / pln_per_point + avg_entry
    be_price_dec_adjusted = be_profit_dec_adjusted / pln_per_point + avg_entry

    header(f"DECEMBER 16 DECISION  ({DEC_LABEL})")
    print(f"  Option A 2026 cost (face):     {cost_A_face:>8,.0f} PLN  (Sep + Dec swaps paid)")
    print(f"  Option A effective cost:       {cost_A_adjusted:>8,.0f} PLN  (×0.81, tax-deductible at future close)")
    print()
    print(f"  Option D (close Sep+Dec) breakevens — D wins BELOW these December prices:")
    print(f"    Naive    (ignore future deduction): profit {be_profit_dec_naive:>8,.0f} PLN "
          f"→ {be_price_dec_naive:,.0f} pts  ({(be_price_dec_naive/current_price-1)*100:+.1f}%)")
    print(f"    Adjusted (long-term hold assumed):  profit {be_profit_dec_adjusted:>8,.0f} PLN "
          f"→ {be_price_dec_adjusted:,.0f} pts  ({(be_price_dec_adjusted/current_price-1)*100:+.1f}%)")
    print()

    # Scenario table
    col = "{:>10}  {:>7}  {:>12}  {:>10}  {:>10}  {:>10}  {}"
    print(col.format("Dec price", "Change", "Net gain (D)", "Cost A", "Cost D", "D saves", "Verdict"))
    div()
    for chg in SCENARIOS:
        dec_price  = current_price * (1 + chg)
        p_dec      = profit_at(dec_price, avg_entry, pln_per_point)
        gain_D     = p_dec + swap_curr                       # Option D annual taxable gain

        cost_D     = tax_on(gain_D)
        d_saves    = cost_A_face - cost_D                    # positive = D cheaper (naive)
        d_saves_adj = cost_A_adjusted - cost_D               # adjusted comparison

        if gain_D <= 0:
            verdict = f"D ✓ + {loss_carryforward_note(-gain_D)}"
        elif d_saves_adj >= 0:
            verdict = f"D ✓  (saves {d_saves_adj:,.0f} PLN adjusted)"
        elif d_saves >= 0:
            verdict = f"D ✓ naive  (A wins long-term by {-d_saves_adj:,.0f} PLN)"
        else:
            verdict = f"A   (A cheaper by {-d_saves:,.0f} PLN naive, {-d_saves_adj:,.0f} PLN adjusted)"

        print(col.format(
            f"{dec_price:,.0f}",
            f"{chg:+.0%}",
            f"{gain_D:,.0f}",
            f"{cost_A_face:,.0f}",
            f"{cost_D:,.0f}",
            f"{d_saves:+,.0f}",
            verdict,
        ))

    print()

    # ── Option B table (Close Sep, Pay Dec) ───────────────────────────────────
    header(f"OPTION B — Close Sep, Pay Dec swap  (for reference)")
    print(f"  Sep gain at current price: {net_unrealized:,.0f} PLN  →  Sep tax: {tax_on(net_unrealized):,.0f} PLN")
    print(f"  Dec swap still owed:       {abs(swap_per_ro):,.0f} PLN")
    print(f"  Option B total 2026 cost at current price: "
          f"{tax_on(net_unrealized) + abs(swap_per_ro):,.0f} PLN")
    print()
    print(f"  {'Sep price':>10}  {'Sep gain':>10}  {'Sep tax':>10}  {'+ Dec swap':>10}  {'B total':>10}  vs A  vs D(curr dec)")
    div()
    for chg in SCENARIOS:
        sep_price = current_price * (1 + chg)
        p_sep     = profit_at(sep_price, avg_entry, pln_per_point)
        gain_sep  = p_sep + swap_curr
        sep_tax   = tax_on(gain_sep)
        cost_B    = sep_tax + abs(swap_per_ro)
        vs_A      = cost_A_face - cost_B
        # Option D cost at same price used as Dec price (for comparison when Sep=Dec)
        gain_D_same = p_sep + swap_curr
        cost_D_same = tax_on(gain_D_same)
        vs_D      = cost_D_same - cost_B   # positive = B cheaper than D at same price

        note = ""
        if gain_sep <= 0:
            note = f"  sep loss carryforward {abs(gain_sep):,.0f} PLN"

        print(f"  {sep_price:>10,.0f}  {gain_sep:>10,.0f}  {sep_tax:>10,.0f}  "
              f"{abs(swap_per_ro):>10,.0f}  {cost_B:>10,.0f}  "
              f"{vs_A:>+5,.0f}  {vs_D:>+5,.0f}{note}")

    print()

    # ── Key notes ─────────────────────────────────────────────────────────────
    header("KEY NOTES")
    print(f"  1. Option D annual gain = profit_at_dec + swap_curr  regardless of Sep price.")
    print(f"     Closing Sep + Dec nets to the same taxable result as closing once in Dec.")
    print()
    print(f"  2. Option A swaps ({cost_A_face:,.0f} PLN) reduce your future taxable gain when")
    print(f"     positions are eventually closed — saving {cost_A_face*BELKA:,.0f} PLN in future tax.")
    print(f"     Effective after-tax swap cost = {cost_A_adjusted:,.0f} PLN.  Use the 'adjusted'")
    print(f"     breakeven if you plan to hold positions for years before closing.")
    print()
    print(f"  3. Loss carryforward (Polish PIT-38): losses offset same-year gains in full.")
    print(f"     Remaining loss carries forward 5 years, max 50% of original loss per year.")
    print()
    print(f"  4. After Option D close-reopen: 2027 starts with zero accumulated swap.")
    print(f"     The clock resets — positions entering March 2027 rollover are fresh.")
    print()
    print(f"  5. Tax due in April 2027 (for 2026 transactions). Option A = zero tax in Apr 2027.")
    print()
    print(f"  6. USD/PLN changes will affect actual PLN values proportionally.")
    print(f"     pln_per_point = {pln_per_point:.2f} PLN/pt  (back-calculated from live positions).")


if __name__ == "__main__":
    main()
