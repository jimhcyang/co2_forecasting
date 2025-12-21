#!/usr/bin/env python3
from __future__ import annotations

"""
plot.py

Creates 3x2 subplot grids (6 markets) for "CO2 Price vs ONE feature" at a time.

For each feature:
  - Version A: all dates
  - Version B: sliced to 2022-09-01 .. 2023-01-02

Plots use two y-axes:
  - Left y: CO2 Price (solid, thicker)
  - Right y: feature (dotted, thinner)

Updates requested:
  1) Remove gridlines for the second y-axis (ax2)
  2) Remove duplicate legends (single combined legend per subplot)

Run (from repo root):
  python scripts/plot.py \
    --data-dir data/co2_processed \
    --out-dir visualizations/pairwise_grids
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


MARKETS = ["Australia", "California", "EU_EEX", "NewZealand", "RGGI", "Shanghai"]

DEFAULT_SLICE_START = "2022-09-01"
DEFAULT_SLICE_END = "2023-01-02"

TECH_PREFIXES = ("SMA_", "PPO_", "RSI_", "ROC_", "ATR_", "CO_chaikin_")


def load_market_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        for cand in ["Unnamed: 0", "Date", "DATE"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    if "date" not in df.columns:
        raise ValueError(f"{csv_path.name}: no date column found (expected date/Date/Unnamed: 0).")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    if "Price" not in df.columns:
        raise ValueError(f"{csv_path.name}: missing required column 'Price'.")

    return df


def infer_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    tech_cols = [c for c in df.columns if c.startswith(TECH_PREFIXES)]
    financial_cols = [c for c in ["equity_index", "fx_rate"] if c in df.columns]
    commodity_cols = [c for c in df.columns if c.endswith("_F")]

    all_features: list[str] = []
    for lst in [tech_cols, financial_cols, commodity_cols]:
        for c in lst:
            if c not in all_features and c not in ["Price", "y"]:
                all_features.append(c)

    return {
        "technical": tech_cols,
        "financial": financial_cols,
        "commodities": commodity_cols,
        "all_features": all_features,
    }


def _safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)


def plot_feature_grid(
    market_dfs: dict[str, pd.DataFrame],
    feature: str,
    *,
    outpath: Path,
    title_suffix: str,
    date_start: str | None = None,
    date_end: str | None = None,
) -> None:
    sns.set_theme(style="white")  # keep clean; seaborn can add grids depending on style

    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=False)
    axes = axes.ravel()

    for i, market in enumerate(MARKETS):
        ax = axes[i]
        df = market_dfs.get(market)

        ax.set_title(market)

        if df is None:
            ax.text(0.5, 0.5, "Missing file", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        d = df.copy()
        if date_start or date_end:
            d = d.loc[(date_start or d.index.min()) : (date_end or d.index.max())]

        if feature not in d.columns:
            ax.text(0.5, 0.5, f"Missing column:\n{feature}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Date")
            ax.set_ylabel("CO2 Price")
            continue

        d = d[["Price", feature]].dropna()
        if d.empty:
            ax.text(0.5, 0.5, "No data after dropna", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Date")
            ax.set_ylabel("CO2 Price")
            continue

        # Left axis: Price (solid, thicker) — no seaborn legend (we'll add a combined legend ourselves)
        sns.lineplot(
            x=d.index,
            y=d["Price"],
            ax=ax,
            linewidth=2.8,
            linestyle="-",
            color="tab:blue",
            label="Price",
            legend=False,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("CO2 Price")

        # Right axis: feature (dotted, thinner)
        ax2 = ax.twinx()
        sns.lineplot(
            x=d.index,
            y=d[feature],
            ax=ax2,
            linewidth=1.2,
            linestyle=":",
            color="tab:orange",
            label=feature,
            legend=False,
        )
        ax2.set_ylabel(feature)

        # (1) remove gridlines for second axis (and keep first axis as-is)
        ax2.grid(False)
        ax2.yaxis.grid(False)
        ax2.xaxis.grid(False)

        # (2) single combined legend per subplot (remove any existing legends first)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()

        handles = ax.get_lines() + ax2.get_lines()
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc="upper left", fontsize=9, frameon=True)

    for j in range(len(MARKETS), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(f"CO2 Price vs {feature} — {title_suffix}", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/co2_processed", help="Directory containing *_features.csv")
    p.add_argument("--out-dir", type=str, default="visualizations/pairwise_grids", help="Output dir for pngs")
    p.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional list of features to plot. If omitted, inferred from the first market file.",
    )
    p.add_argument("--slice-start", type=str, default=DEFAULT_SLICE_START)
    p.add_argument("--slice-end", type=str, default=DEFAULT_SLICE_END)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    market_dfs: dict[str, pd.DataFrame] = {}
    for m in MARKETS:
        fpath = data_dir / f"{m}_features.csv"
        if fpath.exists():
            market_dfs[m] = load_market_df(fpath)

    if not market_dfs:
        raise FileNotFoundError(f"No market files found under {data_dir} for: {MARKETS}")

    if args.features:
        features = list(args.features)
    else:
        first_mkt = next(iter(market_dfs.keys()))
        features = infer_feature_columns(market_dfs[first_mkt])["all_features"]

    if not features:
        raise ValueError("No features found to plot (besides Price).")

    for feat in features:
        slug = _safe_slug(feat)

        plot_feature_grid(
            market_dfs,
            feat,
            outpath=out_dir / f"{slug}__ALL_DATES.png",
            title_suffix="All Dates",
            date_start=None,
            date_end=None,
        )

        plot_feature_grid(
            market_dfs,
            feat,
            outpath=out_dir / f"{slug}__{args.slice_start}_to_{args.slice_end}.png",
            title_suffix=f"{args.slice_start} to {args.slice_end}",
            date_start=args.slice_start,
            date_end=args.slice_end,
        )

    print(f"[SUCCESS] Wrote pairwise grids to: {out_dir}")


if __name__ == "__main__":
    main()
