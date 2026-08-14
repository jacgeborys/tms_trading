"""
import_02.py -- Re-export functions from 02_support_levels.py.

Python can't import modules starting with a digit directly.
This shim uses importlib so 03a_deploy_ladder.py can do:
    from import_02 import get_pending_orders, detect_supports, ...
"""

import importlib

_mod = importlib.import_module("02_support_levels")

fetch_d1 = _mod.fetch_d1
get_pending_orders = _mod.get_pending_orders
detect_supports = _mod.detect_supports
propose_ladder = _mod.propose_ladder
