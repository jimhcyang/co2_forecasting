#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

REQUIRED_COLS = [
    "market",
    "model_name",
    "config_index",
    "window_id",
    "model_hyperparameters_dict",
    "test_data_values_list",
    "test_data_model_predictions_list",
]

LIST_COL_TRUE = "test_data_values_list"
LIST_COL_PRED = "test_data_model_predictions_list"


def _safe_parse_list(x: Any) -> List[float]:
    """Logs store lists as strings like "[1.0]" (JSON) or python-literal style."""
    if isinstance(x, list):
        return [float(v) for v in x]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [float(v) for v in obj]
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [float(v) for v in obj]
    except Exception:
        pass
    return []


def _safe_parse_dict(x: Any) -> Dict[str, Any]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    s = str(x).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    mean_price = float(np.mean(y_true)) if len(y_true) > 0 else np.nan
    rmse_over_mean = float(rmse / mean_price) if mean_price not in (0.0, np.nan) else np.nan

    # order requested: MAPE, RMSE/mean(price), MAE, R2
    return {
        "MAPE": mape,
        "RMSE_over_mean_price": rmse_over_mean,
        "MAE": mae,
        "R2": r2,
        # extra helpful fields
        "RMSE": rmse,
        "mean_price": mean_price,
        "n_samples": int(len(y_true)),
    }


def aggregate_one_log_file(path: Path, *, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print(f"[INFO] Reading: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["model_name"] = df["model_name"].astype(str).str.lower()

    out_rows: List[Dict[str, Any]] = []

    # group by config within this single file (single market/model)
    grouped = df.groupby("config_index", sort=False)
    if verbose:
        print(f"[INFO]   Found {len(grouped)} configs")

    for cfg, g in grouped:
        g = g.sort_values("window_id", kind="stable")

        y_true_all: List[float] = []
        y_pred_all: List[float] = []

        for _, row in g.iterrows():
            y_true_all.extend(_safe_parse_list(row[LIST_COL_TRUE]))
            y_pred_all.extend(_safe_parse_list(row[LIST_COL_PRED]))

        if len(y_true_all) == 0 or len(y_pred_all) == 0:
            continue

        n = min(len(y_true_all), len(y_pred_all))
        y_true = np.asarray(y_true_all[:n], dtype=float)
        y_pred = np.asarray(y_pred_all[:n], dtype=float)

        metrics = compute_metrics(y_true, y_pred)
        hp = _safe_parse_dict(g.iloc[0].get("model_hyperparameters_dict", ""))

        out_rows.append(
            {
                "market": str(g.iloc[0]["market"]),
                "model": str(g.iloc[0]["model_name"]),
                "config_index": int(cfg),
                "n_windows": int(g["window_id"].nunique()),
                **metrics,
                "hyperparams_json": json.dumps(hp, sort_keys=True),
            }
        )

    return pd.DataFrame(out_rows)


def load_all_config_scores(logs_dir: Path, *, verbose: bool = True) -> pd.DataFrame:
    """
    Walk dl_logs/co2/**/**.csv and aggregate each file.
    Returns one row per (market, model, config_index).
    """
    csv_files = sorted(logs_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV logs found under: {logs_dir}")

    if verbose:
        print(f"[INFO] Scanning logs under: {logs_dir}")
        print(f"[INFO] Found {len(csv_files)} CSV files")

    parts: List[pd.DataFrame] = []
    for i, p in enumerate(csv_files, 1):
        try:
            if verbose:
                print(f"[INFO] ({i}/{len(csv_files)}) Aggregating {p.name}")
            part = aggregate_one_log_file(p, verbose=False)
            if len(part) > 0:
                parts.append(part)
        except Exception as e:
            print(f"[WARN] Skipping {p} due to error: {e}")

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)

    # Order columns: requested score order first
    col_order = [
        "market",
        "model",
        "config_index",
        "n_windows",
        "MAPE",
        "RMSE_over_mean_price",
        "MAE",
        "R2",
        "RMSE",
        "mean_price",
        "n_samples",
        "hyperparams_json",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # helpful ordering (best first by your new priorities)
    df = df.sort_values(
        ["market", "model", "MAPE", "RMSE_over_mean_price", "MAE", "R2"],
        ascending=[True, True, True, True, True, False],
    )
    return df


def pick_best_by_market_model(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Best config per (market, model) by:
      1) min MAPE
      2) min RMSE_over_mean_price
      3) min MAE
      4) max R2
    """
    if scores.empty:
        return scores

    best = (
        scores.sort_values(
            ["market", "model", "MAPE", "RMSE_over_mean_price", "MAE", "R2"],
            ascending=[True, True, True, True, True, False],
        )
        .groupby(["market", "model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best


def pick_best_by_model_across_markets(scores: pd.DataFrame) -> pd.DataFrame:
    """
    One overall hyperparam choice per model across all markets.

    We group by (model, hyperparams_json) because config_index is stable
    only if the grid ordering never changes.

    Sort priority:
      1) max markets_covered
      2) min mean_MAPE
      3) min mean_RMSE_over_mean_price
      4) min mean_MAE
      5) max mean_R2
    """
    if scores.empty:
        return scores

    agg = (
        scores.groupby(["model", "hyperparams_json"], as_index=False)
        .agg(
            markets_covered=("market", "nunique"),
            mean_MAPE=("MAPE", "mean"),
            mean_RMSE_over_mean_price=("RMSE_over_mean_price", "mean"),
            mean_MAE=("MAE", "mean"),
            mean_R2=("R2", "mean"),
        )
    )

    best = (
        agg.sort_values(
            ["model", "markets_covered", "mean_MAPE", "mean_RMSE_over_mean_price", "mean_MAE", "mean_R2"],
            ascending=[True, False, True, True, True, False],
        )
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best


def write_best_json(best_by_market_model: pd.DataFrame, out_path: Path) -> None:
    """
    JSON structure:
      { "<Market>": { "<model>": { ...row... } } }
    """
    obj: Dict[str, Dict[str, Any]] = {}

    for _, r in best_by_market_model.iterrows():
        mkt = str(r["market"])
        mdl = str(r["model"])
        obj.setdefault(mkt, {})

        row = r.to_dict()
        row["hyperparams"] = json.loads(row.get("hyperparams_json", "{}") or "{}")

        obj[mkt][mdl] = row

    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    print(f"[INFO] Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate rolling-window dl_logs and score each config.")
    p.add_argument("--logs-dir", type=str, default="dl_logs/co2", help="Root logs dir (default dl_logs/co2)")
    p.add_argument("--out-dir", type=str, default="results", help="Output dir (default results)")
    p.add_argument("--quiet", action="store_true", help="Reduce printouts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("[RUN] Aggregating config scores from rolling-window logs")
        print(f"[RUN] logs_dir = {logs_dir}")
        print(f"[RUN] out_dir  = {out_dir}")
        print("=" * 80)

    scores = load_all_config_scores(logs_dir, verbose=verbose)
    if scores.empty:
        print("[WARN] No scores computed.")
        return

    scores_path = out_dir / "co2_config_scores.csv"
    scores.to_csv(scores_path, index=False)
    print(f"[INFO] Saved: {scores_path} ({len(scores)} rows)")

    best_mm = pick_best_by_market_model(scores)
    best_mm_path = out_dir / "co2_best_by_market_model.csv"
    best_mm.to_csv(best_mm_path, index=False)
    print(f"[INFO] Saved: {best_mm_path} ({len(best_mm)} rows)")

    best_model = pick_best_by_model_across_markets(scores)
    best_model_path = out_dir / "co2_best_by_model_across_markets.csv"
    best_model.to_csv(best_model_path, index=False)
    print(f"[INFO] Saved: {best_model_path} ({len(best_model)} rows)")

    best_json_path = out_dir / "co2_best_configs.json"
    write_best_json(best_mm, best_json_path)

    if verbose:
        print("[DONE] Aggregation complete.")


if __name__ == "__main__":
    main()
