#!/usr/bin/env python3
from __future__ import annotations

"""
tuning.py  (PyTorch, rolling-window version)

Updated methodology (matches your daily workflow)
-------------------------------------------------
For each CO2 market we load its processed daily feature panel:

  data/co2_processed/<Market>_features.csv

We then build the supervised label:

  y_t = Price_{t+1}

Crucially, **today's Price stays in X** as one of the feature columns.
So, for each labeled day t, we learn:

  X_t (includes Price_t + other features_t)  ->  y_t (= Price_{t+1})

Evaluation is walk-forward using fixed rolling windows (default window_size=10,
test_ratio=0.1 -> train first 9 rows, predict last 1 row, then slide by 1 day).

Outputs
-------
1) Per-window prediction logs (append-only):
   dl_logs/co2/<MODEL>/<MARKET>/<MODEL>_<MARKET>.csv

2) Summary metrics per config:
   results/co2_hparam_metrics.csv
   results/co2_best_configs.json (best = min MAPE, ties by RMSE/mean, MAE, then max R2)
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch

from .models import compute_metrics, sliding_window_forecast


# -----------------------------
# Paths (script is under /scripts)
# -----------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]   # repo root
DATA_DIR = ROOT_DIR / "data"
CO2_PROC_DIR = DATA_DIR / "co2_processed"

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DL_LOG_BASE = ROOT_DIR / "dl_logs" / "co2"
DL_LOG_BASE.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Market files
# -----------------------------

CO2_PROCESSED_FILES: Dict[str, str] = {
    "China": "China_features.csv",
    "Korea": "Korea_features.csv",
    "Australia": "Australia_features.csv",
    "California": "California_features.csv",
    "EuropeanUnion": "EuropeanUnion_features.csv",
    "NewZealand": "NewZealand_features.csv",
    "RGGI": "RGGI_features.csv",
}


# -----------------------------
# Hyperparameter grids
# -----------------------------

HP_HIDDEN_UNITS = [32, 64, 128]
HP_NUM_LAYERS = [2, 3, 4]
HP_ACTIVATIONS = ["relu", "tanh", "swish"]
HP_LR = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
HP_BATCH_SIZE = [32]
HP_EPOCHS = [16, 32 ,64]

HP_TCN_CHANNELS = [32, 64, 128]
HP_TCN_BLOCKS = [2, 3, 4]
HP_TCN_KERNEL_SIZE = 5  # fixed


def _core_grid() -> List[Dict[str, Any]]:
    grid = []
    for hidden_units, num_layers, activation, lr, batch, epochs in itertools.product(
        HP_HIDDEN_UNITS, HP_NUM_LAYERS, HP_ACTIVATIONS, HP_LR, HP_BATCH_SIZE, HP_EPOCHS
    ):
        grid.append(
            dict(
                hidden_units=hidden_units,
                num_layers=num_layers,
                activation=activation,
                learning_rate=lr,
                batch_size=batch,
                epochs=epochs,
            )
        )
    return grid


def _tcn_grid() -> List[Dict[str, Any]]:
    grid = []
    for channels, blocks, activation, lr, batch, epochs in itertools.product(
        HP_TCN_CHANNELS, HP_TCN_BLOCKS, HP_ACTIVATIONS, HP_LR, HP_BATCH_SIZE, HP_EPOCHS
    ):
        grid.append(
            dict(
                channels=channels,
                blocks=blocks,
                kernel_size=HP_TCN_KERNEL_SIZE,
                activation=activation,
                learning_rate=lr,
                batch_size=batch,
                epochs=epochs,
            )
        )
    return grid


CORE_GRID = _core_grid()
TCN_GRID = _tcn_grid()
assert len(CORE_GRID) == (len(HP_HIDDEN_UNITS)*len(HP_NUM_LAYERS)*len(HP_ACTIVATIONS)*len(HP_LR)*len(HP_BATCH_SIZE)*len(HP_EPOCHS))
assert len(TCN_GRID) == (len(HP_TCN_CHANNELS)*len(HP_TCN_BLOCKS)*len(HP_ACTIVATIONS)*len(HP_LR)*len(HP_BATCH_SIZE)*len(HP_EPOCHS))


# -----------------------------
# Data loading
# -----------------------------

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
    df = df[df.index.dayofweek < 5]

    if "Price" not in df.columns:
        raise ValueError(f"{path} is missing required 'Price' column.")
    df = df[df["Price"] > 0]

    return df


def build_labeled_df(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["y"] = df["Price"].shift(-1)   # tomorrow price
    df = df.dropna(subset=["y"])
    return df


# -----------------------------
# Driver per market/model
# -----------------------------

def _log_path_for(market: str, model_name: str) -> Path:
    mtag = model_name.upper()
    log_dir = DL_LOG_BASE / mtag / market
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{mtag}_{market}.csv"


def run_market(
    *,
    market: str,
    start_date: str,
    end_date: str,
    window_size: int,
    test_ratio: float,
    device: torch.device,
    max_configs: Optional[int],
    max_windows: Optional[int],
    skip_existing: bool,
    write_logs: bool,
) -> pd.DataFrame:
    panel = load_market_panel(market, start_date, end_date)
    df = build_labeled_df(panel).dropna()

    if len(df) < window_size:
        raise ValueError(f"{market}: labeled rows N={len(df)} < window_size={window_size}")

    rows: List[Dict[str, Any]] = []

    model_specs = [
        # Baseline uses config_index = -1 (matches existing log style)
        ("naive_last_price", [dict(__config_index=-1)]),
        ("mlp", CORE_GRID),
        ("rnn", CORE_GRID),
        ("lstm", CORE_GRID),
        ("gru", CORE_GRID),
        ("tcn", TCN_GRID),
    ]

    for model_name, grid in model_specs:
        if max_configs is not None:
            grid = grid[:max_configs]

        log_path = _log_path_for(market, model_name) if write_logs else None

        existing_cfg_idx = set()
        if skip_existing and log_path is not None and log_path.exists():
            try:
                prev = pd.read_csv(log_path, usecols=["config_index"])
                existing_cfg_idx = set(int(x) for x in prev["config_index"].dropna().unique())
            except Exception:
                existing_cfg_idx = set()

        desc = f"{market}-{model_name}"
        for enum_idx, hp in enumerate(tqdm(grid, desc=desc, unit="cfg", leave=False)):
            hp_local = dict(hp)
            cfg_idx = int(hp_local.pop("__config_index", enum_idx))

            if skip_existing and cfg_idx in existing_cfg_idx:
                continue

            sw = sliding_window_forecast(
                df=df,
                model_name=model_name,
                hp=hp_local,
                target_col="y",
                window_size=window_size,
                test_ratio=test_ratio,
                device=device,
                max_windows=max_windows,
                log_path=log_path,
                log_extra={"market": market, "config_index": cfg_idx},
            )

            metrics = compute_metrics(sw.y_true, sw.y_pred)
            row = {
                "market": market,
                "model": model_name,
                "config_index": cfg_idx,
                "window_size": int(window_size),
                "test_ratio": float(test_ratio),
                "n_windows": int(sw.n_windows),
                "train_loss_last_mean": float(sw.train_loss_last_mean) if np.isfinite(sw.train_loss_last_mean) else None,
                "hyperparams_json": json.dumps(hp_local, sort_keys=True),
                **hp_local,
                **metrics,
            }
            rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CO2 rolling-window hyperparameter tuning.")
    p.add_argument("--start", type=str, default="2022-09-01")
    p.add_argument("--end", type=str, default="2025-08-31")
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--markets", nargs="*", default=list(CO2_PROCESSED_FILES.keys()))
    p.add_argument("--max-configs", type=int, default=None)
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
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] CUDA GPU: {gpu_name}")
        except Exception:
            print("[INFO] CUDA GPU: <unknown>")
    else:
        print("[INFO] CPU only.")

    all_rows: List[pd.DataFrame] = []
    best_cfg: Dict[str, Dict[str, Any]] = {}

    for market in args.markets:
        if market not in CO2_PROCESSED_FILES:
            print(f"[WARN] Unknown market '{market}' — skipping.")
            continue

        print("=" * 80)
        print(f"[MARKET] {market}")

        try:
            df_metrics = run_market(
                market=market,
                start_date=args.start,
                end_date=args.end,
                window_size=args.window_size,
                test_ratio=args.test_ratio,
                device=device,
                max_configs=args.max_configs,
                max_windows=args.max_windows,
                skip_existing=bool(args.skip_existing),
                write_logs=not args.no_logs,
            )
            all_rows.append(df_metrics)

            best_cfg[market] = {}
            for model in df_metrics["model"].unique():
                sub = df_metrics[df_metrics["model"] == model].sort_values(
                    ["MAPE", "RMSE_over_mean_price", "MAE", "R2"],
                    ascending=[True, True, True, False],
                    kind="stable",
                )
                best_row = sub.iloc[0].to_dict()

                hp_dict: Dict[str, Any] = {}
                hp_json = best_row.get("hyperparams_json")
                if isinstance(hp_json, str) and hp_json.strip():
                    try:
                        hp_dict = json.loads(hp_json)
                    except Exception:
                        hp_dict = {}

                best_cfg[market][model] = {
                    "market": market,
                    "model": model,
                    "config_index": int(best_row.get("config_index", -1)),
                    "window_size": int(best_row.get("window_size", args.window_size)),
                    "test_ratio": float(best_row.get("test_ratio", args.test_ratio)),
                    "n_windows": int(best_row.get("n_windows", 0)),
                    "n_samples": int(best_row.get("n_samples", 0)) if best_row.get("n_samples") is not None else None,
                    "mean_price": float(best_row.get("mean_price")) if best_row.get("mean_price") is not None else None,
                    # metric priority: MAPE, RMSE/mean, MAE, R2
                    "MAPE": float(best_row.get("MAPE")) if best_row.get("MAPE") is not None else None,
                    "RMSE_over_mean_price": float(best_row.get("RMSE_over_mean_price")) if best_row.get("RMSE_over_mean_price") is not None else None,
                    "MAE": float(best_row.get("MAE")) if best_row.get("MAE") is not None else None,
                    "R2": float(best_row.get("R2")) if best_row.get("R2") is not None else None,
                    # extras
                    "RMSE": float(best_row.get("RMSE")) if best_row.get("RMSE") is not None else None,
                    "train_loss_last_mean": float(best_row.get("train_loss_last_mean")) if best_row.get("train_loss_last_mean") is not None else None,
                    "hyperparams": hp_dict,
                    "hyperparams_json": best_row.get("hyperparams_json", json.dumps(hp_dict, sort_keys=True)),
                }

            print(f"[DONE] {market}: produced {len(df_metrics)} summary rows.")
        except Exception as e:
            print(f"[ERROR] {market}: {e}")

    if not all_rows:
        print("[WARN] No results produced.")
        return

    out = pd.concat(all_rows, ignore_index=True)
    out_path = RESULTS_DIR / "co2_hparam_metrics.csv"

    # Consistent metric display order
    metric_cols = ["MAPE", "RMSE_over_mean_price", "MAE", "R2"]
    front_cols = ["market", "model", "config_index", "window_size", "test_ratio", "n_windows"] + metric_cols
    ordered_cols = [c for c in front_cols if c in out.columns] + [c for c in out.columns if c not in front_cols]
    out = out[ordered_cols]
    out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote: {out_path}")

    best_path = RESULTS_DIR / "co2_best_configs.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2, default=str)
    print(f"[INFO] Wrote: {best_path}")


if __name__ == "__main__":
    main()
