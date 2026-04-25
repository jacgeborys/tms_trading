"""
Disk cache for OHLCV data and pipeline outputs.

Layout:
  data/           raw OHLCV pickles, one file per symbol+timeframe
  models/         saved XGBoost models (joblib)
  results/        metrics JSON + trade log CSVs, one subfolder per run
  results/charts/ PNG plots

All paths are relative to the project root (directory of this file).
"""

import json
import pickle
import joblib
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent
DATA    = ROOT / "data"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"
CHARTS  = RESULTS / "charts"

for d in (DATA, MODELS, RESULTS, CHARTS):
    d.mkdir(parents=True, exist_ok=True)


# ── OHLCV data ────────────────────────────────────────────────────────────────

def _ohlcv_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace(".", "_")
    return DATA / f"{safe}_{timeframe}.pkl"


def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str):
    path = _ohlcv_path(symbol, timeframe)
    df.to_pickle(path)
    print(f"  Saved {len(df):,} bars → {path.name}")


def load_ohlcv(symbol: str, timeframe: str):
    path = _ohlcv_path(symbol, timeframe)
    if path.exists():
        df = pd.read_pickle(path)
        age_h = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        print(f"  Loaded {len(df):,} bars from cache ({age_h:.1f}h old) ← {path.name}")
        return df
    return None


def ohlcv_is_fresh(symbol: str, timeframe: str, max_age_hours: float = 12.0) -> bool:
    path = _ohlcv_path(symbol, timeframe)
    if not path.exists():
        return False
    age_h = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_h < max_age_hours


# ── Feature matrix ────────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame, tag: str = "latest"):
    path = DATA / f"features_{tag}.pkl"
    df.to_pickle(path)
    print(f"  Saved feature matrix ({df.shape}) → {path.name}")


def load_features(tag: str = "latest"):
    path = DATA / f"features_{tag}.pkl"
    if path.exists():
        df = pd.read_pickle(path)
        print(f"  Loaded feature matrix ({df.shape}) ← {path.name}")
        return df
    return None


# ── Models ────────────────────────────────────────────────────────────────────

def save_model(model, name: str):
    path = MODELS / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"  Saved model → {path.name}")


def load_model(name: str):
    path = MODELS / f"{name}.pkl"
    if path.exists():
        return joblib.load(path)
    return None


# ── Results / metrics ─────────────────────────────────────────────────────────

def save_metrics(metrics_dict: dict, name: str):
    """Save a metrics dict as JSON."""
    def _serial(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    path = RESULTS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=_serial)
    print(f"  Saved metrics → {path.name}")


def save_trades(trades: pd.DataFrame, name: str):
    if trades is None or len(trades) == 0:
        return
    path = RESULTS / f"{name}.csv"
    trades.to_csv(path, index=False)
    print(f"  Saved {len(trades):,} trades → {path.name}")


def save_chart(fig, name: str):
    path = CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved chart → results/charts/{name}.png")


# ── Run manifest ──────────────────────────────────────────────────────────────

def save_run_summary(params: dict, all_metrics: dict):
    """Append a one-line summary to results/runs.json."""
    path = RESULTS / "runs.json"
    runs = []
    if path.exists():
        with open(path) as f:
            runs = json.load(f)

    def _serial(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return str(obj)

    runs.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params":    params,
        "metrics":   all_metrics,
    })
    with open(path, "w") as f:
        json.dump(runs, f, indent=2, default=_serial)
    print(f"  Run logged → results/runs.json  ({len(runs)} total runs)")
