"""
monte_carlo/config.py — Account snapshot and position configuration.

Update BALANCE, EQUITY, MARGIN_USED, CURRENT_PRICE, OPEN_POSITIONS,
and PENDING_ORDERS before each run to reflect the live account state.
"""

# ── Account snapshot (update before each run) ─────────────────────────────────
BALANCE       = 15_376.0   # PLN
EQUITY        = 26_707.0   # PLN  (balance + floating P&L at snapshot time)
MARGIN_USED   =  7_439.0   # PLN  (locked by broker)
CURRENT_PRICE =  7_370.0   # US500.pro index points

# ── Instrument constants ───────────────────────────────────────────────────────
# 1 lot = 50 index points × USDPLN.  At USDPLN ≈ 3.586: COEFF = 50 × 3.586 = 179.3
COEFF    = 179.3   # PLN per index point per 1.0 lot
LEVERAGE = 20

# ── Open long positions: (lots, entry_price) ──────────────────────────────────
OPEN_POSITIONS = [
    (0.010, 6899.9),
    (0.010, 6849.9),
    (0.010, 6799.1),
    (0.010, 6949.9),
    (0.010, 6850.0),
    (0.005, 7000.1),
    (0.005, 6948.9),
    (0.005, 6899.4),
    (0.005, 6775.0),
    (0.005, 6748.1),
    (0.005, 6724.9),
    (0.005, 6660.4),
    (0.005, 6549.3),
    (0.005, 6553.0),
    (0.005, 6800.3),
    (0.002, 6975.1),
    (0.001, 6998.6),
    (0.001, 6949.9),
    (0.001, 7050.0),
    (0.001, 7100.0),
    (0.001, 7100.0),
    (0.001, 7150.3),
    (0.001, 7175.0),
]

# ── Pending buy limit orders: (lots, limit_price) ─────────────────────────────
# Fill when any simulated daily low touches or crosses below limit_price.
PENDING_ORDERS = [
    (0.005, 7250),
    (0.005, 7200),
    (0.005, 7150),
    (0.005, 7100),
    (0.005, 7050),
    (0.005, 7000),
    (0.005, 6975),
    (0.005, 6950),
    (0.005, 6925),
    (0.005, 6900),
    (0.005, 6800),
    (0.005, 6700),
]

# ── Simulation parameters ─────────────────────────────────────────────────────
N_PATHS   = 10_000   # Monte Carlo paths
N_DAYS    = 90       # trading days to simulate
T_DF      = 4        # Student-t degrees of freedom (fat tails)
HIST_DAYS = 252      # days of SPX history to calibrate drift / vol
SPX_TICKER = "^GSPC"

# ── Risk thresholds ───────────────────────────────────────────────────────────
STOPOUT_LEVEL = 100.0   # % margin level → broker liquidates all positions
WARNING_LEVEL = 200.0   # % margin level → warning zone
TARGET_ML_P5  = 150.0   # target: 5th percentile margin level stays above this

# ── Center-of-weight price scan range ────────────────────────────────────────
COW_PRICE_MIN  = 6_400
COW_PRICE_MAX  = 7_500
COW_PRICE_STEP = 25
