"""
monte_carlo/visualisation.py — Six-panel scenario analysis chart.

Dark background, consistent with the rest of the project dashboard.
"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import config


# ── Colour palette ─────────────────────────────────────────────────────────────
C_EQUITY   = "#f0f0f0"
C_GREEN    = "#26a69a"
C_RED      = "#ef5350"
C_ORANGE   = "#f08040"
C_YELLOW   = "#f0c040"
C_BLUE     = "#40a0f0"
C_PURPLE   = "#9c64d0"
C_BAND_MED = "#80c0ff"


def _pct_bands(arr: np.ndarray, axis: int = 0):
    """Return (p5, p25, p50, p75, p95) along given axis."""
    return (
        np.percentile(arr, 5,  axis=axis),
        np.percentile(arr, 25, axis=axis),
        np.percentile(arr, 50, axis=axis),
        np.percentile(arr, 75, axis=axis),
        np.percentile(arr, 95, axis=axis),
    )


def plot_all(
    prices: np.ndarray,                         # (N, T+1)
    portfolio: dict,                             # from analysis.compute_portfolio()
    liq_curve: np.ndarray,                       # (T+1,)
    fill_probs: np.ndarray,                      # (n_orders, 3)  — days 30/60/90
    cow_prices: np.ndarray,
    cow_scores: np.ndarray,
    cow_peak: float,
    pending_orders: List[Tuple[float, float]],
    opt_lots: np.ndarray,                        # (n_orders,) suggested lots
    mu_ann: float,
    sigma_ann: float,
) -> plt.Figure:
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.48, wspace=0.30,
        left=0.07, right=0.97, top=0.95, bottom=0.05,
    )

    days = np.arange(prices.shape[1])
    T    = prices.shape[1] - 1
    ml   = portfolio["margin_level"]
    eq   = portfolio["equity"]

    fig.suptitle(
        f"Monte Carlo — US500.pro CFD   |   {config.N_PATHS:,} paths × {T} days   |   "
        f"μ={mu_ann:+.1%}/yr   σ={sigma_ann:.1%}/yr   t-df={config.T_DF}",
        fontsize=12, y=0.975,
    )

    # ── Panel 1: Price path fan chart ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    p5, p25, p50, p75, p95 = _pct_bands(prices)

    ax1.fill_between(days, p5,  p95, color=C_BLUE,    alpha=0.15, label="5–95th pct")
    ax1.fill_between(days, p25, p75, color=C_BLUE,    alpha=0.30, label="25–75th pct")
    ax1.plot(days, p50, color=C_BAND_MED, lw=2.0, label="Median path")
    ax1.axhline(config.CURRENT_PRICE, color=C_EQUITY, lw=0.8, ls="--", alpha=0.5,
                label=f"Current  {config.CURRENT_PRICE:.0f}")

    for lots, price in pending_orders:
        ax1.axhline(price, color=C_ORANGE, lw=0.6, ls=":", alpha=0.55)
        ax1.text(T * 1.005, price, f"{price:.0f}", va="center",
                 fontsize=6, color=C_ORANGE, clip_on=False)

    # Mark open position entry prices (small ticks on y-axis)
    for lots, entry in config.OPEN_POSITIONS:
        ax1.axhline(entry, color=C_GREEN, lw=0.4, ls="-", alpha=0.25)

    ax1.set_title("Price path fan  (orange = pending limits, green = open entries)",
                  fontsize=9)
    ax1.set_ylabel("US500.pro")
    ax1.set_xlabel("Trading day")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.grid(True, alpha=0.15)

    # ── Panel 2: Margin level distribution over time ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    # Clip to [0, 1000] for display; infinite values (no margin) → skip
    finite_ml = np.where(np.isinf(ml), np.nan, np.clip(ml, 0, 1000))
    ml5, ml25, ml50, ml75, ml95 = _pct_bands(finite_ml)

    ax2.fill_between(days, ml5,  ml95, color=C_YELLOW, alpha=0.12, label="5–95th pct")
    ax2.fill_between(days, ml25, ml75, color=C_YELLOW, alpha=0.28, label="25–75th pct")
    ax2.plot(days, ml50, color=C_YELLOW, lw=2.0, label="Median ML")

    ax2.axhline(config.STOPOUT_LEVEL, color=C_RED,    lw=1.5, ls="--",
                label=f"Stop-out  ({config.STOPOUT_LEVEL:.0f}%)")
    ax2.axhline(config.WARNING_LEVEL, color=C_ORANGE, lw=1.0, ls=":",
                label=f"Warning   ({config.WARNING_LEVEL:.0f}%)")
    ax2.axhspan(0, config.STOPOUT_LEVEL, color=C_RED, alpha=0.08)
    ax2.axhline(config.TARGET_ML_P5, color=C_PURPLE, lw=0.8, ls="--", alpha=0.6,
                label=f"Target p5  ({config.TARGET_ML_P5:.0f}%)")

    ax2.set_title("Margin level distribution over time (%)", fontsize=9)
    ax2.set_ylabel("Margin level (%)")
    ax2.set_xlabel("Trading day")
    ax2.set_ylim(bottom=0, top=min(1000, float(np.nanpercentile(finite_ml, 99)) * 1.1))
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.15)

    # ── Panel 3: Liquidation probability curve ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    liq_pct = liq_curve * 100.0
    ax3.plot(days, liq_pct, color=C_RED, lw=2.0)
    ax3.fill_between(days, 0, liq_pct, color=C_RED, alpha=0.20)

    for d, label in [(30, "30d"), (60, "60d"), (90, "90d")]:
        v = liq_pct[d]
        ax3.axvline(d, color=C_ORANGE, lw=0.7, ls=":", alpha=0.6)
        ax3.text(d + 0.8, v + 0.05, f"{v:.2f}%", fontsize=7.5,
                 color=C_ORANGE, va="bottom")

    ax3.set_title("Liquidation probability — fraction of paths ever below 100% ML",
                  fontsize=9)
    ax3.set_ylabel("Probability (%)")
    ax3.set_xlabel("Trading day")
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.15)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # ── Panel 4: Fill probability bar chart ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    n_orders   = len(pending_orders)
    order_labels = [f"{p:.0f}" for _, p in pending_orders]
    x          = np.arange(n_orders)
    width      = 0.25

    bar_colors = [C_GREEN, C_YELLOW, C_ORANGE]
    horizons   = [30, 60, 90]
    for j, (h, col) in enumerate(zip(horizons, bar_colors)):
        probs = fill_probs[:, j] * 100
        bars  = ax4.bar(x + (j - 1) * width, probs, width,
                        label=f"Day {h}", color=col, alpha=0.80)
        for bar, prob in zip(bars, probs):
            if prob > 3:
                ax4.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f"{prob:.0f}", ha="center", va="bottom",
                         fontsize=5.5, color=col)

    ax4.set_title("Fill probability by horizon — pending buy limits", fontsize=9)
    ax4.set_ylabel("Probability (%)")
    ax4.set_xlabel("Limit price")
    ax4.set_xticks(x)
    ax4.set_xticklabels(order_labels, rotation=45, ha="right", fontsize=7)
    ax4.set_ylim(0, 105)
    ax4.legend(fontsize=8)
    ax4.grid(True, axis="y", alpha=0.15)

    # ── Panel 5: Center-of-weight curve ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])

    pos_mask = cow_scores > 0
    ax5.plot(cow_prices, cow_scores, color=C_BLUE, lw=1.5)
    ax5.fill_between(cow_prices, 0, cow_scores, where=pos_mask,
                     color=C_BLUE, alpha=0.25, label="Positive expected value")
    ax5.fill_between(cow_prices, cow_scores, 0, where=~pos_mask,
                     color=C_RED, alpha=0.25, label="Negative expected value")

    ax5.axvline(cow_peak, color=C_YELLOW, lw=1.5, ls="--",
                label=f"Peak @ {cow_peak:.0f}")
    ax5.axvline(config.CURRENT_PRICE, color=C_EQUITY, lw=0.8, ls=":",
                alpha=0.5, label=f"Current price  {config.CURRENT_PRICE:.0f}")

    # Mark pending order prices
    for _, price in pending_orders:
        ax5.axvline(price, color=C_ORANGE, lw=0.5, ls="--", alpha=0.4)

    ax5.set_title(
        "Center of weight — expected risk-adj. P&L per 0.001 lot  "
        "(fill probability × expected gain − margin cost)",
        fontsize=9,
    )
    ax5.set_ylabel("Score (PLN per 0.001 lot)")
    ax5.set_xlabel("Entry price level")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.15)
    ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ── Panel 6: Optimal vs current lot distribution ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])

    current_lots = np.array([o[0] for o in pending_orders])
    x6           = np.arange(n_orders)
    w6           = 0.35

    ax6.bar(x6 - w6 / 2, current_lots, w6,
            color=C_ORANGE, alpha=0.75, label="Current lots")
    ax6.bar(x6 + w6 / 2, opt_lots,     w6,
            color=C_GREEN,  alpha=0.75, label=f"Suggested lots  (p5 ML ≥ {config.TARGET_ML_P5:.0f}%)")

    # Annotate suggested lots
    for i, (cl, ol) in enumerate(zip(current_lots, opt_lots)):
        if ol != cl:
            color = C_GREEN if ol > cl else C_RED
            ax6.text(i + w6 / 2, ol + 0.0001, f"{ol:.3f}",
                     ha="center", va="bottom", fontsize=6, color=color)

    ax6.set_title(
        f"Optimal lot distribution — CoW-weighted, constrained to p5 ML ≥ {config.TARGET_ML_P5:.0f}%",
        fontsize=9,
    )
    ax6.set_ylabel("Lot size")
    ax6.set_xlabel("Limit price")
    ax6.set_xticks(x6)
    ax6.set_xticklabels(order_labels, rotation=45, ha="right", fontsize=7)
    ax6.legend(fontsize=8)
    ax6.grid(True, axis="y", alpha=0.15)
    ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))

    return fig
