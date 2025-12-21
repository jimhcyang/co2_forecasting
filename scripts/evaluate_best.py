#!/usr/bin/env python3
from __future__ import annotations

"""
evaluate_best.py  (PyTorch, rolling-window evaluation with fixed best configs)

- Loads best hyperparameters from:
      results/co2_best_configs.json
- Evaluates on a NEW date slice (default 2023-01-01 to 2025-11-30)
  to avoid mixing tuning and reporting.

Walk-forward rolling windows:
  - y_t = Price_{t+1}
  - X_t includes today's Price_t and all other features_t
  - For each window of length W:
      train on first (W - test_size) rows
      predict on last test_size rows
      slide window by 1 day

Logs:
  test_log/co2/<MODEL>/<MARKET>/<MODEL>_<MARKET>.csv

Summary:
  results/co2_test_metrics.csv

Baseline:
  Always includes NAIVE_LAST_PRICE (config_index=-1, hyperparams={})
  even if naive isn't stored in JSON.

Metric priority / display order:
  MAPE (min), RMSE_over_mean_price (min), MAE (min), R2 (max)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch

try:
    from .models import compute_metrics, sliding_window_forecast
except ImportError:  # pragma: no cover
    from scripts.models import compute_metrics, sliding_window_forecast


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CO2_PROC_DIR = DATA_DIR / "co2_processed"

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_CONFIGS_PATH = RESULTS_DIR / "co2_best_configs.json"

TEST_LOG_BASE = ROOT_DIR / "test_log" / "co2"
TEST_LOG_BASE.mkdir(parents=True, exist_ok=True)


CO2_PROCESSED_FILES: Dict[str, str] = {
    "Australia": "Australia_features.csv",
    "California": "California_features.csv",
    "EU_EEX": "EU_EEX_features.csv",
    "NewZealand": "NewZealand_features.csv",
    "RGGI": "RGGI_features.csv",
    "Shanghai": "Shanghai_features.csv",
}


def load_market_panel(market: str, start_date: str, end_date: str) -> pd.DataFrame:
    filename = CO2_PROCESSED_FILES[market]
    path = CO2_PROC_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found for market {market}")

    df = pd.read_csv(path)

    rename_map = {"Unnamed: 0": "date", "Date": "date", "date": "date"}
    df = df.rename(columns=lambda c: rename_map.get(str(c).strip(), str(c).strip()))

    if "date" not in df.columns:
        raise ValueError(f"{path} must contain a date column (e.g. 'Unnamed: 0').")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[start_date:end_date]
    df = df[df.index.dayofweek < 5]  # business days only

    if "Price" not in df.columns:
        raise ValueError(f"{path} is missing required 'Price' column.")
    df = df[df["Price"] > 0]

    return df


def build_labeled_df(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["y"] = df["Price"].shift(-1)  # tomorrow price
    df = df.dropna(subset=["y"])
    return df


def load_best_configs(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Best-config JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be a JSON object (dict).")
    return obj


def _log_path_for(market: str, model_name: str) -> Path:
    mtag = model_name.upper()
    log_dir = TEST_LOG_BASE / mtag / market
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{mtag}_{market}.csv"


def _get_cfg_rec(best_cfg: Dict[str, Dict[str, Any]], market: str, model_name: str) -> Dict[str, Any]:
    if model_name == "naive_last_price":
        rec = (best_cfg.get(market, {}) or {}).get(model_name)
        if isinstance(rec, dict):
            return rec
        return {"config_index": -1, "hyperparams": {}}

    rec = (best_cfg.get(market, {}) or {}).get(model_name)
    if not isinstance(rec, dict):
        raise KeyError(f"No best-config entry for market='{market}', model='{model_name}' in JSON.")
    return rec


def run_one(
    *,
    market: str,
    model_name: str,
    cfg_rec: Dict[str, Any],
    start_date: str,
    end_date: str,
    window_size: int,
    test_ratio: float,
    device: torch.device,
    max_windows: Optional[int],
    skip_existing: bool,
    write_logs: bool,
) -> Dict[str, Any]:
    panel = load_market_panel(market, start_date, end_date)
    df = build_labeled_df(panel).dropna()

    if len(df) < window_size:
        raise ValueError(f"{market}: labeled rows N={len(df)} < window_size={window_size}")

    cfg_idx = int(cfg_rec.get("config_index", -1))
    hp = cfg_rec.get("hyperparams", {}) or {}

    log_path = _log_path_for(market, model_name) if write_logs else None
    if skip_existing and log_path is not None and log_path.exists():
        return {
            "market": market,
            "model": model_name,
            "config_index": cfg_idx,
            "skipped": True,
            "reason": f"log_exists: {log_path}",
        }

    sw = sliding_window_forecast(
        df=df,
        model_name=model_name,
        hp=hp,
        target_col="y",
        window_size=window_size,
        test_ratio=test_ratio,
        device=device,
        max_windows=max_windows,
        log_path=log_path,
        log_extra={"market": market, "config_index": cfg_idx},
    )

    metrics = compute_metrics(sw.y_true, sw.y_pred)

    mean_price = float(df["Price"].mean())
    rmse = float(metrics.get("RMSE", np.nan))
    rmse_over_mean_price = (rmse / mean_price) if (np.isfinite(rmse) and mean_price > 0) else np.nan

    out_row: Dict[str, Any] = {
        "market": market,
        "model": model_name,
        "config_index": cfg_idx,
        "window_size": int(window_size),
        "test_ratio": float(test_ratio),
        "n_windows": int(sw.n_windows),
        "train_loss_last_mean": float(sw.train_loss_last_mean) if np.isfinite(sw.train_loss_last_mean) else None,
        "mean_price": mean_price,
        "RMSE_over_mean_price": float(rmse_over_mean_price) if np.isfinite(rmse_over_mean_price) else None,
        "n_samples": int(len(df)),
        "hyperparams_json": json.dumps(hp, sort_keys=True),
        **hp,
        **metrics,
        "skipped": False,
        "reason": "",
    }
    return out_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CO2 rolling-window evaluation using fixed best configs (no tuning).")
    p.add_argument("--start", type=str, default="2023-01-01")
    p.add_argument("--end", type=str, default="2025-11-30")
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--markets", nargs="*", default=list(CO2_PROCESSED_FILES.keys()))
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--best-configs", type=str, default=str(BEST_CONFIGS_PATH))
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-logs", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        try:
            print(f"[INFO] CUDA GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("[INFO] CUDA GPU: <unknown>")
    else:
        print("[INFO] CPU only.")

    best_cfg = load_best_configs(Path(args.best_configs))

    rows: List[Dict[str, Any]] = []

    for market in args.markets:
        if market not in CO2_PROCESSED_FILES:
            print(f"[WARN] Unknown market '{market}' — skipping.")
            continue

        print("=" * 80)
        print(f"[MARKET] {market}")

        models_in_json = list((best_cfg.get(market, {}) or {}).keys())

        if args.models is None:
            run_models = sorted(set(models_in_json + ["naive_last_price"]))
        else:
            run_models = [m.strip() for m in args.models]

        for model_name in tqdm(run_models, desc=f"{market}", unit="model", leave=False):
            try:
                rec = _get_cfg_rec(best_cfg, market, model_name)
                row = run_one(
                    market=market,
                    model_name=model_name,
                    cfg_rec=rec,
                    start_date=args.start,
                    end_date=args.end,
                    window_size=args.window_size,
                    test_ratio=args.test_ratio,
                    device=device,
                    max_windows=args.max_windows,
                    skip_existing=bool(args.skip_existing),
                    write_logs=not args.no_logs,
                )
                rows.append(row)

                if row.get("skipped"):
                    print(f"[SKIP] {market}-{model_name}: {row.get('reason')}")
                else:
                    print(f"[DONE] {market}-{model_name}: n_windows={row.get('n_windows')}, MAPE={row.get('MAPE')}")
            except KeyError as e:
                print(f"[WARN] {market}-{model_name}: {e}")
            except Exception as e:
                print(f"[ERROR] {market}-{model_name}: {e}")

    if not rows:
        print("[WARN] No results produced.")
        return

    out = pd.DataFrame(rows)

    score_cols = ["MAPE", "RMSE_over_mean_price", "MAE", "R2", "RMSE", "mean_price", "n_samples"]
    id_cols = ["market", "model", "config_index", "window_size", "test_ratio", "n_windows", "train_loss_last_mean"]
    front = [c for c in id_cols if c in out.columns] + [c for c in score_cols if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    out_path = RESULTS_DIR / "co2_test_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote: {out_path}")


if __name__ == "__main__":
    main()
