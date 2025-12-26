#!/usr/bin/env python3
from __future__ import annotations

"""
input.py  (place this file under: scripts/input.py)

For each local CO2 market file in data/co2:

1. Load the CO2 price series.
   - For China + Korea CSVs, use the column "Secondary Market".
   - Drop any row where Secondary Market is empty/NaN or equals 0.
   - Drop any row whose date is not a business day (Mon–Fri).
2. Reindex to all business days from 2022-06-01 to 2025-12-01 and interpolate
   + forward/backward fill so every business day has a price.
3. Download (Yahoo Finance) and align to the same business-day index:
   - One market-specific equity index.
   - One market-specific FX rate.
   - Common commodity tickers: CL=F, NG=F, RB=F, HRC=F, ALI=F.
4. Compute 6 technical indicators **on the CO2 price only**:
   SMA_5, PPO_12_26, RSI_14, ROC_10, ATR_14, CO_chaikin_3_10.
5. Truncate the final DataFrame to dates **on/after 2022-09-01**.
6. Save one CSV per market with 14 columns:
   [Price, equity_index, fx_rate, CL_F, NG_F, RB_F, HRC_F, ALI_F,
    SMA_5, PPO_12_26, RSI_14, ROC_10, ATR_14, CO_chaikin_3_10].
7. Print per-market stats to terminal:
   - earliest date in file (after dropping invalid/0 + non-business days, within selected timeframe)
   - latest date in file (same definition)
   - total # business days in selected timeframe (2022-09-01..2025-12-01)
   - # business days with observed (not dropped) rows in selected timeframe
   - # business days filled via interpolation/ffill/bfill
   - % filled
8. Save a single line plot (log y-scale) of Price for all markets over the selected timeframe.

Expected local input files under: data/co2/
  - china.csv
  - korea.csv
  - Aus spot.xlsx
  - Cali ETF.xlsx
  - EEX Spot Bl.xlsx
  - New Zealand spot.xlsx
  - RGGI spot.xlsx
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="yfinance")

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------

START_DATE = "2022-06-01"
END_DATE = "2025-09-03"  # inclusive for our index; yfinance end is exclusive
FREQ = "B"  # business days
TRUNCATE_START = "2022-09-01"  # selected timeframe start (for modeling + stats)

# This script lives in /scripts, so repo root is one level up.
REPO_ROOT = Path(__file__).resolve().parents[1]

CO2_DIR = REPO_ROOT / "data" / "co2"
OUT_DIR = REPO_ROOT / "data" / "co2_processed"
PLOT_DIR = REPO_ROOT / "visualizations"

DATE_INDEX = pd.date_range(START_DATE, END_DATE, freq=FREQ)
TRUNC_INDEX = DATE_INDEX[DATE_INDEX >= pd.Timestamp(TRUNCATE_START)]

# yfinance: end is exclusive, so add 1 day to include END_DATE
YF_END = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# Common commodity tickers (same for all markets)
COMMODITY_TICKERS: Dict[str, str] = {
    "CL_F": "CL=F",    # Crude Oil
    "NG_F": "NG=F",    # Natural Gas
    "RB_F": "RB=F",    # RBOB Gasoline
    "HRC_F": "HRC=F",  # U.S. Midwest Domestic Hot-Rolled Coil Steel
    "ALI_F": "ALI=F",  # Aluminum
}

# filename -> {market_name, equity_ticker, fx_ticker}
# NOTE: Shanghai final.csv dropped; China + Korea added.
MARKET_CONFIGS: Dict[str, Dict[str, str]] = {
    "china.csv": {
        "market_name": "China",
        "equity_ticker": "000001.SS",  # retain original Shanghai equity proxy
        "fx_ticker": "CNY=X",          # retain original Shanghai FX proxy
    },
    "korea.csv": {
        "market_name": "Korea",
        "equity_ticker": "^KS11",      # KOSPI
        "fx_ticker": "KRW=X",          # USD/KRW
    },
    "Aus spot.xlsx": {
        "market_name": "Australia",
        "equity_ticker": "^AXJO",      # S&P/ASX 200
        "fx_ticker": "AUDUSD=X",       # AUD/USD
    },
    "Cali ETF.xlsx": {
        "market_name": "California",
        "equity_ticker": "^GSPC",      # S&P 500
        "fx_ticker": "DX-Y.NYB",       # USD index
    },
    "EEX Spot Bl.xlsx": {
        "market_name": "EuropeanUnion",
        "equity_ticker": "^STOXX50E",  # EURO STOXX 50
        "fx_ticker": "EURUSD=X",       # EUR/USD
    },
    "New Zealand spot.xlsx": {
        "market_name": "NewZealand",
        "equity_ticker": "^NZ50",      # S&P/NZX 50 (as used previously)
        "fx_ticker": "NZDUSD=X",       # NZD/USD
    },
    "RGGI spot.xlsx": {
        "market_name": "RGGI",
        "equity_ticker": "^GSPC",      # S&P 500
        "fx_ticker": "DX-Y.NYB",       # USD index
    },
}

# Requested print order
MARKET_ORDER = ["China", "Korea", "Australia", "California", "EuropeanUnion", "NewZealand", "RGGI"]

# Simple cache so we don't re-download the same ticker N times
YF_CACHE: Dict[str, pd.Series] = {}

# ---------------------------------------------------------------------
# Helpers: cleaning + stats
# ---------------------------------------------------------------------

def _is_business_day_index(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.dayofweek < 5  # Mon=0 ... Fri=4

def clean_price_series(raw: pd.Series) -> pd.Series:
    """
    Clean raw price series:
      - coerce numeric
      - drop NaN and 0
      - drop non-business days
      - sort and deduplicate
    """
    s = raw.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.dropna()
    s = s[s != 0]

    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]

    # Drop rows that are not business days
    bd_mask = _is_business_day_index(s.index)
    s = s.loc[bd_mask]

    # Clip to our global timeframe
    s = s.loc[(s.index >= pd.Timestamp(START_DATE)) & (s.index <= pd.Timestamp(END_DATE))]

    s.name = "Price"
    return s

# ---------------------------------------------------------------------
# Technical indicators on PRICE ONLY
# ---------------------------------------------------------------------

def compute_price_technicals(price: pd.Series) -> pd.DataFrame:
    """
    Compute 6 indicators from a single price series:
      - SMA_5
      - PPO_12_26
      - RSI_14
      - ROC_10
      - ATR_14 (volatility proxy via mean |ΔP|)
      - CO_chaikin_3_10 (price-direction-only proxy, no volume)
    """
    price = pd.to_numeric(price, errors="coerce").sort_index()

    # 1) SMA_5 – 5-day SMA
    sma_5 = price.rolling(window=5, min_periods=1).mean()

    # 2) PPO_12_26 = (EMA_12 - EMA_26) / EMA_26 * 100
    ema_fast = price.ewm(span=12, adjust=False).mean()
    ema_slow = price.ewm(span=26, adjust=False).mean()
    ppo_12_26 = (ema_fast - ema_slow) / ema_slow * 100

    # 3) RSI_14 (Wilder-style)
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))

    # 4) ROC_10 – % change over last 10 days
    roc_10 = price.pct_change(periods=10) * 100

    # 5) ATR_14 – mean |ΔP| over last 14 days (volatility proxy)
    atr_14 = delta.abs().rolling(window=14, min_periods=1).mean()

    # 6) Chaikin-like oscillator from price direction only
    direction = delta.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    adl = direction.cumsum()  # pseudo "accumulation/distribution line"
    ad_short = adl.ewm(span=3, adjust=False).mean()
    ad_long = adl.ewm(span=10, adjust=False).mean()
    chaikin = ad_short - ad_long

    out = pd.DataFrame(
        {
            "SMA_5": sma_5,
            "PPO_12_26": ppo_12_26,
            "RSI_14": rsi_14,
            "ROC_10": roc_10,
            "ATR_14": atr_14,
            "CO_chaikin_3_10": chaikin,
        },
        index=price.index,
    )
    return out

# ---------------------------------------------------------------------
# Yahoo Finance helpers (Adj Close only, aligned to DATE_INDEX)
# ---------------------------------------------------------------------

def get_yf_series(ticker: str) -> pd.Series:
    """
    Download a single ticker from Yahoo, pull its Adj Close (or Close),
    align to DATE_INDEX (BUSINESS DAYS), interpolate missing days, and cache.
    """
    if ticker in YF_CACHE:
        return YF_CACHE[ticker].copy()

    print(f"[INFO] Downloading {ticker} from Yahoo ...")
    df = yf.download(
        ticker,
        start=START_DATE,
        end=YF_END,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"[WARN] {ticker}: no data returned from Yahoo. Filling with NaNs.")
        s = pd.Series(index=DATE_INDEX, dtype=float, name=ticker)
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            s = df["Close"]

        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]

        s = s.reindex(DATE_INDEX)
        s = s.interpolate(method="time").ffill().bfill()
        s.name = ticker

    YF_CACHE[ticker] = s
    return s.copy()

# ---------------------------------------------------------------------
# Local CO2 loaders (raw -> clean -> aligned)
# ---------------------------------------------------------------------

def load_bbg_single_series(path: Path) -> pd.Series:
    """
    Load Bloomberg-style single-series Excel (Aus, EEX, NZ, RGGI):
      - first column: dates
      - second column: PX_LAST (price)
    """
    raw = pd.read_excel(path)
    date_col = raw.columns[0]
    value_col = raw.columns[1]

    dates = pd.to_datetime(raw[date_col], errors="coerce")
    values = pd.to_numeric(raw[value_col], errors="coerce")

    mask = dates.notna() & values.notna()
    s = pd.Series(values[mask].values, index=dates[mask])
    s.name = "Price"
    return clean_price_series(s)

def load_cali_price(path: Path) -> pd.Series:
    """
    Load California ETF Excel and pull a single price series (Adj Close or Close).
    """
    raw = pd.read_excel(path)
    raw.columns = [str(c).replace("\xa0", "").strip() for c in raw.columns]

    if "Date" not in raw.columns:
        # fallback: assume first column is date
        raw["Date"] = raw.iloc[:, 0]

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    if "Adj Close" in raw.columns:
        price = pd.to_numeric(raw["Adj Close"], errors="coerce")
    else:
        price = pd.to_numeric(raw["Close"], errors="coerce")

    mask = raw["Date"].notna() & price.notna()
    s = pd.Series(price[mask].values, index=raw["Date"][mask])
    s.name = "Price"
    return clean_price_series(s)

def _rename_from_first_row_if_headerlike(df: pd.DataFrame) -> pd.DataFrame:
    """
    For China/Korea CSVs: first row contains label strings like "Date", "Secondary Market".
    If so, use row0 values as column names where present, then drop row0.
    """
    if df.empty:
        return df
    row0 = df.iloc[0]
    new_cols = []
    for c in df.columns:
        v = row0.get(c, None)
        v_str = "" if pd.isna(v) else str(v).strip()
        if v_str and v_str.lower() not in {"nan"}:
            new_cols.append(v_str)
        else:
            new_cols.append(str(c).strip())
    out = df.iloc[1:].copy()
    out.columns = new_cols
    return out

def load_secondary_market_csv(path: Path) -> pd.Series:
    """
    Load a CSV and extract:
      - Date column (must exist after normalization; otherwise fallback to first column)
      - Secondary Market column (must exist after normalization)
    """
    df = pd.read_csv(path)

    # If the first row looks like embedded headers, normalize using it
    # (works for your provided china.csv and korea.csv)
    df = _rename_from_first_row_if_headerlike(df)

    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]

    # Date column
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
    else:
        dt = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    # Secondary Market column
    sec_candidates = [c for c in df.columns if c.strip().lower() == "secondary market"]
    if not sec_candidates:
        raise ValueError(f"{path.name}: could not find a 'Secondary Market' column after parsing.")
    sec_col = sec_candidates[0]

    price = pd.to_numeric(df[sec_col], errors="coerce")

    mask = dt.notna() & price.notna()
    s = pd.Series(price[mask].values, index=dt[mask])
    s.name = "Price"

    # Drop empty / 0 rows, drop non-business days, etc.
    return clean_price_series(s)

def load_local_price_clean(filename: str) -> pd.Series:
    """
    Load and CLEAN a local CO2 price series (drops NaN/0 + non-business days).
    Returns a sparse business-day series (only observed days).
    """
    path = CO2_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    fname_lower = filename.lower()
    if fname_lower.endswith(".csv"):
        # China/Korea use Secondary Market
        if fname_lower in {"china.csv", "korea.csv"}:
            return load_secondary_market_csv(path)

        # Generic csv fallback: first col date, second col price
        df = pd.read_csv(path)
        dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        price = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        mask = dates.notna() & price.notna()
        s = pd.Series(price[mask].values, index=dates[mask])
        s.name = "Price"
        return clean_price_series(s)

    # Excel files
    if "cali" in fname_lower:
        return load_cali_price(path)

    # Default Bloomberg-like
    return load_bbg_single_series(path)

def align_and_fill_price(
    s_clean: pd.Series,
) -> Tuple[pd.Series, Dict[str, object]]:
    """
    Align a cleaned (sparse) price series to DATE_INDEX and fill.
    Returns (filled_series, stats_dict).
    """
    # Observed days (in selected timeframe)
    observed_days = s_clean.index.intersection(TRUNC_INDEX)
    observed_bd_selected = int(observed_days.size)

    total_bd_selected = int(len(TRUNC_INDEX))
    filled_bd_selected = int(total_bd_selected - observed_bd_selected)
    pct_filled = (filled_bd_selected / total_bd_selected * 100.0) if total_bd_selected else np.nan

    # Earliest/latest observed (post-drop, post-business-filter) within selected timeframe
    s_sel = s_clean.loc[s_clean.index.intersection(TRUNC_INDEX)]
    earliest = s_sel.index.min() if not s_sel.empty else pd.NaT
    latest = s_sel.index.max() if not s_sel.empty else pd.NaT

    # Align + fill
    s_aligned = s_clean.reindex(DATE_INDEX)
    # interpolate across internal gaps on business days, then ensure full coverage via ffill/bfill
    s_filled = s_aligned.interpolate(method="time").ffill().bfill()
    s_filled.name = "Price"

    stats = {
        "earliest_date_in_file": earliest,
        "latest_date_in_file": latest,
        "total_bd_selected": total_bd_selected,
        "observed_bd_selected": observed_bd_selected,
        "filled_bd_selected": filled_bd_selected,
        "pct_filled": pct_filled,
    }
    return s_filled, stats

# ---------------------------------------------------------------------
# Build per-market DataFrame
# ---------------------------------------------------------------------

def build_market_frame(filename: str, cfg: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    For one CO2 market:
      - load & clean local CO2 price (drop NaN/0 + non-business days)
      - align to DATE_INDEX and fill (interpolate/ffill/bfill)
      - fetch equity index, FX, commodities
      - compute technicals on price
      - return (combined DataFrame, stats dict).
    """
    equity_ticker = cfg["equity_ticker"]
    fx_ticker = cfg["fx_ticker"]

    s_clean = load_local_price_clean(filename)
    price, stats = align_and_fill_price(s_clean)

    df = pd.DataFrame(index=DATE_INDEX)
    df["Price"] = price

    df["equity_index"] = get_yf_series(equity_ticker)
    df["fx_rate"] = get_yf_series(fx_ticker)

    for col_name, ticker in COMMODITY_TICKERS.items():
        df[col_name] = get_yf_series(ticker)

    tech = compute_price_technicals(df["Price"])
    df = df.join(tech)

    # enforce column order exactly
    ordered_cols = [
        "Price",
        "equity_index",
        "fx_rate",
        "CL_F",
        "NG_F",
        "RB_F",
        "HRC_F",
        "ALI_F",
        "SMA_5",
        "PPO_12_26",
        "RSI_14",
        "ROC_10",
        "ATR_14",
        "CO_chaikin_3_10",
    ]
    df = df[ordered_cols]

    return df, stats

# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------

def save_all_markets_plot(price_by_market: Dict[str, pd.Series], out_path: Path) -> None:
    """
    Save a single line plot with all markets' prices on log y-scale.
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot in requested order
    for m in MARKET_ORDER:
        if m not in price_by_market:
            continue
        s = price_by_market[m].copy()
        s = s.loc[TRUNC_INDEX]
        # log-safe
        s = s.replace([np.inf, -np.inf], np.nan)
        s = s.clip(lower=1e-9)
        ax.plot(s.index, s.values, label=m, linewidth=1.5)

    ax.set_yscale("log")
    ax.set_title("CO₂ Market Prices (log-scale) — Selected Timeframe")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (log scale)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using date range {START_DATE} to {END_DATE} (inclusive), freq={FREQ}")
    print(f"[INFO] Selected timeframe for stats/output truncation starts at {TRUNCATE_START}")
    print(f"[INFO] CO2 input directory: {CO2_DIR}")
    print(f"[INFO] Output directory: {OUT_DIR}")
    print(f"[INFO] Plot directory: {PLOT_DIR}")
    print(f"[INFO] Total business days in selected timeframe: {len(TRUNC_INDEX)}")

    # Build in config order but store by market name
    results: Dict[str, Dict[str, object]] = {}
    price_for_plot: Dict[str, pd.Series] = {}

    for filename, cfg in MARKET_CONFIGS.items():
        market = cfg["market_name"]
        path = CO2_DIR / filename
        if not path.exists():
            print(f"[WARN] Skipping {filename} (file not found at {path}).")
            continue

        print(f"\n[INFO] Processing market {market} from {filename} ...")
        df_market, stats = build_market_frame(filename, cfg)

        # Truncate to selected timeframe for saving
        df_market = df_market.loc[df_market.index >= pd.Timestamp(TRUNCATE_START)]

        out_path = OUT_DIR / f"{market}_features.csv"
        df_market.to_csv(out_path)

        print(
            f"[INFO] Saved {out_path} with shape {df_market.shape} "
            f"(columns: {list(df_market.columns)})"
        )

        results[market] = stats
        price_for_plot[market] = df_market["Price"].copy()

    # Print stats in requested order
    print("\n" + "=" * 72)
    print("[STATS] Per-market data coverage (selected timeframe only)")
    print("=" * 72)

    for market in MARKET_ORDER:
        if market not in results:
            print(f"\n[STATS] {market}: (missing / skipped)")
            continue

        st = results[market]
        earliest = st["earliest_date_in_file"]
        latest = st["latest_date_in_file"]

        # format dates
        earliest_s = "NaT" if pd.isna(earliest) else pd.Timestamp(earliest).strftime("%Y-%m-%d")
        latest_s = "NaT" if pd.isna(latest) else pd.Timestamp(latest).strftime("%Y-%m-%d")

        total_bd = st["total_bd_selected"]
        obs_bd = st["observed_bd_selected"]
        filled_bd = st["filled_bd_selected"]
        pct = st["pct_filled"]

        print(f"\n[STATS] {market}")
        print(f"  Earliest date in file (post-drop, business-days, in timeframe): {earliest_s}")
        print(f"  Latest date in file   (post-drop, business-days, in timeframe): {latest_s}")
        print(f"  Total business days in selected timeframe: {total_bd}")
        print(f"  Business days observed (not dropped):      {obs_bd}")
        print(f"  Business days filled (interp/ffill/bfill): {filled_bd}")
        print(f"  % days filled:                            {pct:0.2f}%")

    # Save combined plot
    plot_path = PLOT_DIR / "co2_prices_all_markets_logscale.png"
    save_all_markets_plot(price_for_plot, plot_path)
    print("\n" + "=" * 72)
    print(f"[INFO] Saved log-scale multi-market price plot to: {plot_path}")
    print("=" * 72)

if __name__ == "__main__":
    main()
