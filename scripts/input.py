#!/usr/bin/env python3
from __future__ import annotations

"""
input.py  (place this file under: scripts/input.py)

For each local CO2 market file in data/co2:

1. Load the CO2 price series (spot / ETF).
2. Reindex to all business days from 2022-06-01 to 2025-12-01 and interpolate
   + forward/backward fill so every business day has a price.
3. Download:
   - One market-specific equity index.
   - One market-specific FX rate.
   - Common commodity tickers: CL=F, NG=F, RB=F, HRC=F, ALI=F.
   Each is aligned to the same business-day index and interpolated/fill.
4. Compute 6 technical indicators **on the CO2 price only**:
   SMA_5, PPO_12_26, RSI_14, ROC_10, ATR_14, CO_chaikin_3_10.
5. Truncate the final DataFrame to dates **on/after 2022-09-01**.
6. Save one CSV per market with 14 columns:
   [Price, equity_index, fx_rate, CL_F, NG_F, RB_F, HRC_F, ALI_F,
    SMA_5, PPO_12_26, RSI_14, ROC_10, ATR_14, CO_chaikin_3_10].
All indices are BUSINESS DAYS only.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='yfinance')


# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------

START_DATE = "2022-06-01"
END_DATE = "2025-12-01"
FREQ = "B"  # business days
TRUNCATE_START = "2022-09-01"  # keep only data from this date onward

# This script lives in /scripts, so repo root is one level up.
REPO_ROOT = Path(__file__).resolve().parents[1]

CO2_DIR = REPO_ROOT / "data" / "co2"
OUT_DIR = REPO_ROOT / "data" / "co2_processed"

DATE_INDEX = pd.date_range(START_DATE, END_DATE, freq=FREQ)

# Common commodity tickers (same for all markets)
COMMODITY_TICKERS: Dict[str, str] = {
    "CL_F": "CL=F",    # Crude Oil
    "NG_F": "NG=F",    # Natural Gas
    "RB_F": "RB=F",    # RBOB Gasoline
    "HRC_F": "HRC=F",  # U.S. Midwest Domestic Hot-Rolled Coil Steel
    "ALI_F": "ALI=F",  # Aluminum
}

# One config entry per local CO2 file
# filename -> {market_name, equity_ticker, fx_ticker}
MARKET_CONFIGS: Dict[str, Dict[str, str]] = {
    # Australia spot market
    "Aus spot.xlsx": {
        "market_name": "Australia",
        "equity_ticker": "^AXJO",       # S&P/ASX 200
        "fx_ticker": "AUDUSD=X",        # AUD / USD
    },
    # California ETF (US carbon market proxy)
    "Cali ETF.xlsx": {
        "market_name": "California",
        "equity_ticker": "^GSPC",       # S&P 500
        "fx_ticker": "DX-Y.NYB",        # USD index
    },
    # EU EEX spot
    "EEX Spot Bl.xlsx": {
        "market_name": "EU_EEX",
        "equity_ticker": "^STOXX50E",   # EURO STOXX 50
        "fx_ticker": "EURUSD=X",        # EUR / USD
    },
    # New Zealand spot
    "New Zealand spot.xlsx": {
        "market_name": "NewZealand",
        "equity_ticker": "^NZ50",       # S&P/NZX 50
        "fx_ticker": "NZDUSD=X",        # NZD / USD
    },
    # RGGI spot (US)
    "RGGI spot.xlsx": {
        "market_name": "RGGI",
        "equity_ticker": "^GSPC",       # S&P 500
        "fx_ticker": "DX-Y.NYB",        # USD index
    },
    # Shanghai spot
    "Shanghai final.csv": {
        "market_name": "Shanghai",
        "equity_ticker": "000001.SS",   # SSE Composite
        "fx_ticker": "CNY=X",           # USD / CNY
    },
}

# Simple cache so we don't re-download the same ticker N times
YF_CACHE: Dict[str, pd.Series] = {}


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
    price = pd.to_numeric(price, errors="coerce")
    price = price.sort_index()

    # 1) SMA_5 – 20-day SMA
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
        end=END_DATE,
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
# Local CO2 loaders (price only)
# ---------------------------------------------------------------------

def load_bbg_single_series(path: Path) -> pd.Series:
    """
    Load Bloomberg-style single-series Excel (Aus, EEX, NZ, RGGI):
        - first column: dates
        - second column: PX_LAST (price)
    Return: Series named 'Price' with DatetimeIndex.
    """
    raw = pd.read_excel(path)
    date_col = raw.columns[0]
    value_col = raw.columns[1]

    dates = pd.to_datetime(raw[date_col], errors="coerce")
    values = pd.to_numeric(raw[value_col], errors="coerce")

    mask = dates.notna() & values.notna()
    s = pd.Series(values[mask].values, index=dates[mask])
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "Price"
    return s


def load_shanghai_csv(path: Path) -> pd.Series:
    """
    Load Shanghai spot CSV.
    Expected (but we fall back if slightly different):
        'Exchange Date', 'Trade Price'
    Return: Series named 'Price' with DatetimeIndex.
    """
    df = pd.read_csv(path)

    if "Exchange Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Exchange Date"], errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    if "Trade Price" in df.columns:
        price = pd.to_numeric(df["Trade Price"], errors="coerce")
    else:
        price = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    mask = df["Date"].notna() & price.notna()
    s = pd.Series(price[mask].values, index=df["Date"][mask])
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "Price"
    return s


def load_cali_price(path: Path) -> pd.Series:
    """
    Load California ETF Excel and pull a single price series (Adj Close or Close).
    Return: Series named 'Price' with DatetimeIndex.
    """
    raw = pd.read_excel(path)
    raw.columns = [c.replace("\xa0", "").strip() for c in raw.columns]

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    if "Adj Close" in raw.columns:
        price = pd.to_numeric(raw["Adj Close"], errors="coerce")
    else:
        price = pd.to_numeric(raw["Close"], errors="coerce")

    mask = raw["Date"].notna() & price.notna()
    s = pd.Series(price[mask].values, index=raw["Date"][mask])
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "Price"
    return s


def load_local_price(filename: str) -> pd.Series:
    """
    Load a local CO2 price from data/co2/<filename> and align to DATE_INDEX
    (BUSINESS DAYS) with interpolation + ffill + bfill.
    """
    path = CO2_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    fname_lower = filename.lower()
    if filename.lower().endswith(".csv"):
        if "shanghai" in fname_lower:
            s = load_shanghai_csv(path)
        else:
            df = pd.read_csv(path)
            dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            price = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            mask = dates.notna() & price.notna()
            s = pd.Series(price[mask].values, index=dates[mask])
            s = s.sort_index()
            s = s[~s.index.duplicated(keep="last")]
            s.name = "Price"
    else:
        if "cali" in fname_lower:
            s = load_cali_price(path)
        elif "shanghai" in fname_lower:
            s = load_shanghai_csv(path)
        else:
            s = load_bbg_single_series(path)

    s = s.reindex(DATE_INDEX)
    s = s.interpolate(method="time").ffill().bfill()
    s.name = "Price"
    return s


# ---------------------------------------------------------------------
# Build per-market DataFrame
# ---------------------------------------------------------------------

def build_market_frame(filename: str, cfg: Dict[str, str]) -> pd.DataFrame:
    """
    For one CO2 market:
      - load & align CO2 price (business-day DATE_INDEX)
      - fetch equity index, FX, commodities (CL/NG/RB/HRC/ALI)
      - compute technicals on price
      - return combined DataFrame (14 columns total).
    """
    equity_ticker = cfg["equity_ticker"]
    fx_ticker = cfg["fx_ticker"]

    price = load_local_price(filename)

    df = pd.DataFrame(index=DATE_INDEX)
    df["Price"] = price

    df["equity_index"] = get_yf_series(equity_ticker)
    df["fx_rate"] = get_yf_series(fx_ticker)

    for col_name, ticker in COMMODITY_TICKERS.items():
        df[col_name] = get_yf_series(ticker)

    tech = compute_price_technicals(df["Price"])
    df = df.join(tech)

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using date range {START_DATE} to {END_DATE}, freq={FREQ}")
    print(f"[INFO] CO2 input directory: {CO2_DIR}")
    print(f"[INFO] Output directory: {OUT_DIR}")

    for filename, cfg in MARKET_CONFIGS.items():
        path = CO2_DIR / filename
        if not path.exists():
            print(f"[WARN] Skipping {filename} (file not found at {path}).")
            continue

        print(f"\n[INFO] Processing market {cfg['market_name']} from {filename} ...")
        df_market = build_market_frame(filename, cfg)

        df_market = df_market.loc[df_market.index >= TRUNCATE_START]

        out_path = OUT_DIR / f"{cfg['market_name']}_features.csv"
        df_market.to_csv(out_path)
        print(
            f"[INFO] Saved {out_path} "
            f"with shape {df_market.shape} "
            f"(columns: {list(df_market.columns)})"
        )


if __name__ == "__main__":
    main()