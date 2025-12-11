#!/usr/bin/env python3
from __future__ import annotations

# co2_hyperparam_pipeline.py
#
# Grid-search training for CO2 markets using *processed* feature panels:
#
#   - Input: data/co2_processed/<Market>_features.csv
#       Columns (per download/processing script):
#         date (Unnamed: 0), Price, equity_index, fx_rate,
#         CL_F, NG_F, RB_F, HRC_F,
#         SMA_5, PPO_5_10, RSI_14, ROC_10, ATR_14, CO_chaikin_3_10
#
#   - Time slice (defaults, overridable via CLI):
#         start_date = 2022-09-01
#         end_date   = 2022-10-31
#       (Two full months: September + October 2022.)
#
#   - Target:
#         y_t = Price_{t+1}   (next-day price)
#
#   - Features:
#         All columns except 'Price' and 'y'
#         (Price is used only for target & naive baseline).
#
#   - Sequence dataset:
#         LOOKBACK = 9
#         For i in [LOOKBACK, N-1]:
#           X_i = features[i-LOOKBACK : i]  (shape: 9 × F)
#           y_i = y[i]            (Price_{t+1})
#           naive_i = Price[i]    (predict next-day price with today's price)
#
#   - Train / Test split:
#         chronological split with TRAIN_RATIO (default 0.8) on sequences.
#         (i.e. last 20% of sequences are used as test set.)
#
#   - Models and grids (per market):
#
#       Shared hyperparams for MLP / RNN / LSTM / GRU:
#         hidden_units   ∈ {32, 64, 128}
#         num_layers     ∈ {2, 3, 4}
#         activation     ∈ {relu, tanh, swish}
#         learning_rate  ∈ {1e-2, 3e-3, 1e-3}
#         batch_size     ∈ {16, 32, 64}
#         epochs         ∈ {32}
#
#         => 3 * 3 * 3 * 3 * 3 * 1 = 243 configs per model.
#
#       TCN-specific grid:
#         channels       ∈ {32, 64, 128}
#         blocks         ∈ {2, 3, 4}
#         kernel_size    = 5 (fixed)
#         activation     ∈ {relu, tanh, swish}
#         learning_rate  ∈ {1e-2, 3e-3, 1e-3}
#         batch_size     ∈ {16, 32, 64}
#         epochs         ∈ {32}
#
#         => 3 * 3 * 3 * 3 * 3 * 1 = 243 configs.
#
#   - Output:
#       results/co2_hparam_metrics.csv  (all markets, all models, all configs)
#       results/co2_models/<market>_<model>_best.keras  (best-R² weights per model)
#       dl_logs/co2/<MODEL>/<MARKET>/<MODEL>_<MARKET>.csv
#           (per-config logs with dates, y_true list, y_pred list, and
#            model_hyperparameters_dict)

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import itertools
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K


# -------------------------------------------------------------------
# PATHS & GLOBAL CONFIG
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]  # root
DATA_DIR = ROOT_DIR / "data"
CO2_PROC_DIR = DATA_DIR / "co2_processed"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = RESULTS_DIR / "co2_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DL_LOG_BASE = ROOT_DIR / "dl_logs" / "co2"
DL_LOG_BASE.mkdir(parents=True, exist_ok=True)

# Map of market name -> processed features CSV
CO2_PROCESSED_FILES: Dict[str, str] = {
    "Australia": "Australia_features.csv",
    "California": "California_features.csv",
    "EU_EEX": "EU_EEX_features.csv",
    "NewZealand": "NewZealand_features.csv",
    "RGGI": "RGGI_features.csv",
    "Shanghai": "Shanghai_features.csv",
}

# Defaults (overridable via CLI)
START_DATE_DEFAULT = "2022-09-01"
END_DATE_DEFAULT = "2022-10-31"
LOOKBACK_DEFAULT = 9  # 9 days history + 1 day ahead = 10-day rolling window
TRAIN_RATIO_DEFAULT = 0.8

# -------------------------------------------------------------------
# HYPERPARAMETER GRIDS
# -------------------------------------------------------------------

HP_HIDDEN_UNITS = [32, 64, 128]
HP_NUM_LAYERS = [2, 3, 4]
HP_ACTIVATIONS = ["relu", "tanh", "swish"]
HP_LR = [1e-2, 3e-3, 1e-3]
HP_BATCH_SIZE = [16, 32, 64]
HP_EPOCHS = [32]

# Core grid size for MLP/RNN/LSTM/GRU
GRID_SIZE_CORE = (
    len(HP_HIDDEN_UNITS)
    * len(HP_NUM_LAYERS)
    * len(HP_ACTIVATIONS)
    * len(HP_LR)
    * len(HP_BATCH_SIZE)
    * len(HP_EPOCHS)
)

# TCN-specific
HP_TCN_CHANNELS = [32, 64, 128]
HP_TCN_BLOCKS = [2, 3, 4]
HP_TCN_KERNEL_SIZE = 5  # fixed

GRID_SIZE_TCN = (
    len(HP_TCN_CHANNELS)
    * len(HP_TCN_BLOCKS)
    * len(HP_ACTIVATIONS)
    * len(HP_LR)
    * len(HP_BATCH_SIZE)
    * len(HP_EPOCHS)
)

assert GRID_SIZE_CORE == 243, f"CORE grid size is {GRID_SIZE_CORE}, expected 243"
assert GRID_SIZE_TCN == 243, f"TCN grid size is {GRID_SIZE_TCN}, expected 243"


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

def set_global_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "n_samples": int(len(y_true)),
    }


def log_predictions_for_config(
    *,
    market: str,
    model_name: str,
    config_index: int,
    hyperparams: Dict[str, Any],
    dates_te: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Append a single row to dl_logs/co2/<MODEL>/<MARKET>/<MODEL>_<MARKET>.csv
    matching the ml_full_tuning logging style.

    We also add `market` and `config_index` fields for convenience.
    """
    model_tag = model_name.upper()
    log_dir = DL_LOG_BASE / model_tag / market
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_tag}_{market}.csv"

    if len(y_true) == 0:
        return

    row = {
        "market": market,
        "model_name": model_tag,
        "config_index": config_index,
        "model_hyperparameters_dict": json.dumps(hyperparams),
        "start_date": str(pd.to_datetime(dates_te[0])),
        "end_date": str(pd.to_datetime(dates_te[-1])),
        "test_data_values_list": json.dumps([float(v) for v in y_true]),
        "test_data_model_predictions_list": json.dumps([float(v) for v in y_pred]),
    }

    df_row = pd.DataFrame([row])
    if log_path.exists():
        df_prev = pd.read_csv(log_path)
        df_out = pd.concat([df_prev, df_row], ignore_index=True)
    else:
        df_out = df_row
    df_out.to_csv(log_path, index=False)


# -------------------------------------------------------------------
# DATA LOADING & SEQUENCE BUILDING
# -------------------------------------------------------------------

def load_market_panel(
    market: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load processed feature panel for a given market, slice to [start_date, end_date],
    keep only BUSINESS DAYS (Mon–Fri), and return a DataFrame indexed by date.
    """
    filename = CO2_PROCESSED_FILES[market]
    path = CO2_PROC_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found for market {market}")

    print(f"[INFO] Loading processed panel for {market} from {path}")
    df = pd.read_csv(path)

    # Normalize / parse date
    rename_map = {
        "Unnamed: 0": "date",
        "Date": "date",
        "date": "date",
    }
    df = df.rename(columns=lambda c: rename_map.get(str(c).strip(), str(c).strip()))

    if "date" not in df.columns:
        raise ValueError(f"{path} must contain a date column (e.g. 'Unnamed: 0').")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()

    # Keep numeric columns only for features
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace infinities with NaN so they get dropped later
    df = df.replace([np.inf, -np.inf], np.nan)

    # Slice to project window
    df = df.loc[start_date:end_date]

    # Keep only business days (Mon–Fri)
    df = df[df.index.dayofweek < 5]

    # Filter out non-positive prices just in case
    if "Price" not in df.columns:
        raise ValueError(f"{path} is missing 'Price' column.")
    df = df[df["Price"] > 0]

    print(f"[INFO] {market} panel shape after slice (business days only): {df.shape}")
    return df


def make_sequence_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    price_col: str,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a sequence dataset from a panel with given feature & target columns.

      df: must contain feature_cols + [target_col, price_col]
      feature_cols: columns to use as features
      target_col: y (e.g. next-day Price)
      price_col: for naive baseline (e.g. 'Price')
      lookback: sequence length (number of *historical* days = 9 by default)

    Returns:
      X:      (N, lookback, F)
      y:      (N,)
      dates:  (N,)
      naive:  (N,)  (df[price_col] at time of target)
    """
    df = df.copy()
    df = df[feature_cols + [target_col, price_col]].dropna()

    feature_values = df[feature_cols].values
    target_values = df[target_col].values
    price_values = df[price_col].values
    dates_index = df.index

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    naive_list: List[float] = []
    date_list: List[pd.Timestamp] = []

    for i in range(lookback, len(df)):
        # 9 historical days + 1-day-ahead target => 10-day rolling window
        X_seq = feature_values[i - lookback : i, :]
        y_val = target_values[i]
        naive_val = price_values[i]  # predict next-day price with today's price

        X_list.append(X_seq)
        y_list.append(y_val)
        naive_list.append(naive_val)
        date_list.append(dates_index[i])

    X = np.array(X_list)
    y = np.array(y_list)
    dates = np.array(date_list)
    naive = np.array(naive_list)

    return X, y, dates, naive


# -------------------------------------------------------------------
# MODEL BUILDERS
# -------------------------------------------------------------------

def build_mlp(
    input_dim: int,
    hidden_units: int,
    num_layers: int,
    activation: str,
    learning_rate: float,
) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(layers.Dense(hidden_units, activation=activation))
    model.add(layers.Dense(1))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse")
    return model


def build_rnn(
    input_shape: Tuple[int, int],
    hidden_units: int,
    num_layers: int,
    activation: str,
    learning_rate: float,
) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        model.add(
            layers.SimpleRNN(
                hidden_units,
                activation="tanh",
                return_sequences=return_sequences,
            )
        )
    # activation here is for the final dense head (relu/tanh/swish)
    model.add(layers.Dense(1, activation=activation))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse")
    return model


def build_lstm(
    input_shape: Tuple[int, int],
    hidden_units: int,
    num_layers: int,
    activation: str,
    learning_rate: float,
) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        model.add(
            layers.LSTM(
                hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=return_sequences,
            )
        )
    model.add(layers.Dense(1, activation=activation))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse")
    return model


def build_gru(
    input_shape: Tuple[int, int],
    hidden_units: int,
    num_layers: int,
    activation: str,
    learning_rate: float,
) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        model.add(
            layers.GRU(
                hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=return_sequences,
            )
        )
    model.add(layers.Dense(1, activation=activation))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse")
    return model


def _tcn_residual_block(
    x: keras.Tensor,
    channels: int,
    kernel_size: int,
    activation: str,
    dilation_rate: int,
) -> Tuple[keras.Tensor, int]:
    conv1 = layers.Conv1D(
        filters=channels,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)
    act1 = layers.Activation(activation)(conv1)

    conv2 = layers.Conv1D(
        filters=channels,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(act1)
    act2 = layers.Activation(activation)(conv2)

    # Residual path to match shape if needed
    if x.shape[-1] != channels:
        res = layers.Conv1D(filters=channels, kernel_size=1, padding="same")(x)
    else:
        res = x

    out = layers.Add()([act2, res])
    out = layers.Activation(activation)(out)

    return out, dilation_rate * 2


def build_tcn(
    input_shape: Tuple[int, int],
    blocks: int,
    channels: int,
    kernel_size: int,
    activation: str,
    learning_rate: float,
) -> keras.Model:
    """
    Temporal Convolutional Network (TCN) with:
      - 'blocks' residual blocks
      - 2 Conv1D layers per block (same dilation within a block)
      - exponentially increasing dilations: 1, 2, 4, ...
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    dilation_rate = 1
    for _ in range(blocks):
        x, dilation_rate = _tcn_residual_block(
            x,
            channels=channels,
            kernel_size=kernel_size,
            activation=activation,
            dilation_rate=dilation_rate,
        )
    # Take last time step
    x = layers.Lambda(lambda t: t[:, -1, :])(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse")
    return model


# -------------------------------------------------------------------
# GRID SEARCH HELPERS
# -------------------------------------------------------------------

def run_core_grid_for_model(
    model_name: str,
    build_fn,
    X_tr_flat_s: np.ndarray,
    X_te_flat_s: np.ndarray,
    X_tr_seq_s: np.ndarray,
    X_te_seq_s: np.ndarray,
    y_tr_s: np.ndarray,
    y_te: np.ndarray,
    dates_te: np.ndarray,
    sy: StandardScaler,
    input_dim: int,
    input_shape_seq: Tuple[int, int],
    market: str,
    max_configs: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Run the 243-config core grid for one of: MLP / RNN / LSTM / GRU.
    """
    results: List[Dict[str, Any]] = []
    best_r2 = -np.inf
    best_model_path = MODELS_DIR / f"{market}_{model_name}_best.keras"

    # Build full grid and optionally truncate for debugging
    grid = list(
        itertools.product(
            HP_HIDDEN_UNITS,
            HP_NUM_LAYERS,
            HP_ACTIVATIONS,
            HP_LR,
            HP_BATCH_SIZE,
            HP_EPOCHS,
        )
    )
    if max_configs is not None:
        grid = grid[:max_configs]

    total_configs = len(grid)
    print(f"[GRID] {market} - {model_name}: {total_configs} configs")

    for config_idx, (hidden_units, num_layers, activation, lr, batch_size, epochs) in enumerate(
        tqdm(grid, desc=f"{market}-{model_name}", unit="cfg", leave=False)
    ):
        set_global_seeds(42)
        K.clear_session()

        if model_name == "mlp":
            model = build_fn(
                input_dim=input_dim,
                hidden_units=hidden_units,
                num_layers=num_layers,
                activation=activation,
                learning_rate=lr,
            )
            history = model.fit(
                X_tr_flat_s,
                y_tr_s,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )
            y_pred_s = model.predict(X_te_flat_s, verbose=0).ravel()
        else:
            # sequence models: RNN, LSTM, GRU
            model = build_fn(
                input_shape=input_shape_seq,
                hidden_units=hidden_units,
                num_layers=num_layers,
                activation=activation,
                learning_rate=lr,
            )
            history = model.fit(
                X_tr_seq_s,
                y_tr_s,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )
            y_pred_s = model.predict(X_te_seq_s, verbose=0).ravel()

        y_pred = sy.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        metrics = compute_metrics(y_te, y_pred)

        hyperparams = {
            "hidden_units": hidden_units,
            "num_layers": num_layers,
            "activation": activation,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
        }

        # log per-config prediction traces (single global test window)
        log_predictions_for_config(
            market=market,
            model_name=model_name,
            config_index=config_idx,
            hyperparams=hyperparams,
            dates_te=dates_te,
            y_true=y_te,
            y_pred=y_pred,
        )

        row = {
            "market": market,
            "model": model_name,
            "config_index": config_idx,
            **hyperparams,
            "train_loss_last": float(history.history["loss"][-1]),
            **metrics,
        }
        results.append(row)

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            model.save(best_model_path)

    return results


def run_tcn_grid(
    X_tr_seq_s: np.ndarray,
    X_te_seq_s: np.ndarray,
    y_tr_s: np.ndarray,
    y_te: np.ndarray,
    dates_te: np.ndarray,
    sy: StandardScaler,
    input_shape_seq: Tuple[int, int],
    market: str,
    max_configs: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Run the 243-config grid for TCN.
    """
    results: List[Dict[str, Any]] = []
    best_r2 = -np.inf
    best_model_path = MODELS_DIR / f"{market}_tcn_best.keras"

    grid = list(
        itertools.product(
            HP_TCN_CHANNELS,
            HP_TCN_BLOCKS,
            HP_ACTIVATIONS,
            HP_LR,
            HP_BATCH_SIZE,
            HP_EPOCHS,
        )
    )
    if max_configs is not None:
        grid = grid[:max_configs]

    total_configs = len(grid)
    print(f"[GRID] {market} - tcn: {total_configs} configs")

    for config_idx, (channels, blocks, activation, lr, batch_size, epochs) in enumerate(
        tqdm(grid, desc=f"{market}-tcn", unit="cfg", leave=False)
    ):
        set_global_seeds(42)
        K.clear_session()

        model = build_tcn(
            input_shape=input_shape_seq,
            blocks=blocks,
            channels=channels,
            kernel_size=HP_TCN_KERNEL_SIZE,
            activation=activation,
            learning_rate=lr,
        )
        history = model.fit(
            X_tr_seq_s,
            y_tr_s,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        y_pred_s = model.predict(X_te_seq_s, verbose=0).ravel()
        y_pred = sy.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        metrics = compute_metrics(y_te, y_pred)

        hyperparams = {
            "channels": channels,
            "blocks": blocks,
            "kernel_size": HP_TCN_KERNEL_SIZE,
            "activation": activation,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
        }

        log_predictions_for_config(
            market=market,
            model_name="tcn",
            config_index=config_idx,
            hyperparams=hyperparams,
            dates_te=dates_te,
            y_true=y_te,
            y_pred=y_pred,
        )

        row = {
            "market": market,
            "model": "tcn",
            "config_index": config_idx,
            **hyperparams,
            "train_loss_last": float(history.history["loss"][-1]),
            **metrics,
        }
        results.append(row)

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            model.save(best_model_path)

    return results


# -------------------------------------------------------------------
# MAIN PER-MARKET TRAINING
# -------------------------------------------------------------------

def train_models_for_market(
    market: str,
    start_date: str,
    end_date: str,
    lookback: int,
    train_ratio: float,
    max_configs: int | None = None,
) -> pd.DataFrame:
    """
    Full hyperparameter training pipeline for a single market.
    """
    panel = load_market_panel(market, start_date, end_date)
    if len(panel) < lookback + 20:
        raise ValueError(
            f"{market}: not enough data in [{start_date}, {end_date}] "
            f"for lookback={lookback}"
        )

    # 1) Define target: next-day price
    panel = panel.copy()
    panel["y"] = panel["Price"].shift(-1)

    # 2) Feature columns: everything except Price and y
    feature_cols = [c for c in panel.columns if c not in ["Price", "y"]]
    if not feature_cols:
        raise ValueError(f"{market}: no feature columns found.")

    # Drop rows with NaN y / features
    df_model = panel[feature_cols + ["Price", "y"]].dropna()
    if len(df_model) < lookback + 20:
        raise ValueError(
            f"{market}: not enough data after dropna for lookback={lookback}"
        )

    # 3) Sequence dataset
    X, y, dates, naive = make_sequence_dataset(
        df_model,
        feature_cols=feature_cols,
        target_col="y",
        price_col="Price",
        lookback=lookback,
    )
    N, L, F = X.shape
    print(f"[INFO] {market} sequence dataset: X={X.shape}, y={y.shape}")

    # 4) Train/test split (chronological; last 20% is test)
    split = int(N * train_ratio)
    if split <= 0 or split >= N:
        raise ValueError(f"{market}: invalid train/test split with N={N}")

    X_tr_seq, X_te_seq = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    naive_tr, naive_te = naive[:split], naive[split:]
    dates_tr, dates_te = dates[:split], dates[split:]

    # Flatten for MLP
    X_tr_flat = X_tr_seq.reshape(len(X_tr_seq), -1)
    X_te_flat = X_te_seq.reshape(len(X_te_seq), -1)

    # Scale features (fit on TRAIN ONLY) → no leakage
    sx = StandardScaler().fit(X_tr_flat)
    X_tr_flat_s = sx.transform(X_tr_flat)
    X_te_flat_s = sx.transform(X_te_flat)

    # For sequence models: reshape scaled features back
    X_tr_seq_s = X_tr_flat_s.reshape(X_tr_seq.shape)
    X_te_seq_s = X_te_flat_s.reshape(X_te_seq.shape)

    # Scale targets (fit on TRAIN ONLY) → no leakage
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))
    y_tr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

    results: List[Dict[str, Any]] = []

    # ----------------- Naive baseline -----------------
    naive_pred = naive_te  # already on price scale
    naive_metrics = compute_metrics(y_te, naive_pred)
    results.append(
        {
            "market": market,
            "model": "naive_last_price",
            "config_index": -1,
            "hidden_units": None,
            "num_layers": None,
            "activation": None,
            "learning_rate": None,
            "batch_size": None,
            "epochs": None,
            "train_loss_last": None,
            **naive_metrics,
        }
    )
    print(
        f"[BASELINE] {market} naive_last_price: "
        f"RMSE={naive_metrics['RMSE']:.4f}, "
        f"MAPE={naive_metrics['MAPE']:.4f}, "
        f"R2={naive_metrics['R2']:.4f}"
    )

    # Log baseline predictions as well (no hyperparams)
    log_predictions_for_config(
        market=market,
        model_name="naive_last_price",
        config_index=-1,
        hyperparams={},
        dates_te=dates_te,
        y_true=y_te,
        y_pred=naive_pred,
    )

    input_dim = X_tr_flat_s.shape[1]
    input_shape_seq = (L, F)

    # ----------------- MLP GRID -----------------
    mlp_results = run_core_grid_for_model(
        model_name="mlp",
        build_fn=build_mlp,
        X_tr_flat_s=X_tr_flat_s,
        X_te_flat_s=X_te_flat_s,
        X_tr_seq_s=X_tr_seq_s,
        X_te_seq_s=X_te_seq_s,
        y_tr_s=y_tr_s,
        y_te=y_te,
        dates_te=dates_te,
        sy=sy,
        input_dim=input_dim,
        input_shape_seq=input_shape_seq,
        market=market,
        max_configs=max_configs,
    )
    results.extend(mlp_results)

    # ----------------- RNN GRID -----------------
    rnn_results = run_core_grid_for_model(
        model_name="rnn",
        build_fn=build_rnn,
        X_tr_flat_s=X_tr_flat_s,
        X_te_flat_s=X_te_flat_s,
        X_tr_seq_s=X_tr_seq_s,
        X_te_seq_s=X_te_seq_s,
        y_tr_s=y_tr_s,
        y_te=y_te,
        dates_te=dates_te,
        sy=sy,
        input_dim=input_dim,
        input_shape_seq=input_shape_seq,
        market=market,
        max_configs=max_configs,
    )
    results.extend(rnn_results)

    # ----------------- LSTM GRID -----------------
    lstm_results = run_core_grid_for_model(
        model_name="lstm",
        build_fn=build_lstm,
        X_tr_flat_s=X_tr_flat_s,
        X_te_flat_s=X_te_flat_s,
        X_tr_seq_s=X_tr_seq_s,
        X_te_seq_s=X_te_seq_s,
        y_tr_s=y_tr_s,
        y_te=y_te,
        dates_te=dates_te,
        sy=sy,
        input_dim=input_dim,
        input_shape_seq=input_shape_seq,
        market=market,
        max_configs=max_configs,
    )
    results.extend(lstm_results)

    # ----------------- GRU GRID -----------------
    gru_results = run_core_grid_for_model(
        model_name="gru",
        build_fn=build_gru,
        X_tr_flat_s=X_tr_flat_s,
        X_te_flat_s=X_te_flat_s,
        X_tr_seq_s=X_tr_seq_s,
        X_te_seq_s=X_te_seq_s,
        y_tr_s=y_tr_s,
        y_te=y_te,
        dates_te=dates_te,
        sy=sy,
        input_dim=input_dim,
        input_shape_seq=input_shape_seq,
        market=market,
        max_configs=max_configs,
    )
    results.extend(gru_results)

    # ----------------- TCN GRID -----------------
    tcn_results = run_tcn_grid(
        X_tr_seq_s=X_tr_seq_s,
        X_te_seq_s=X_te_seq_s,
        y_tr_s=y_tr_s,
        y_te=y_te,
        dates_te=dates_te,
        sy=sy,
        input_shape_seq=input_shape_seq,
        market=market,
        max_configs=max_configs,
    )
    results.extend(tcn_results)

    metrics_df = pd.DataFrame(results)
    metrics_df["n_features"] = len(feature_cols)
    metrics_df["features"] = [",".join(feature_cols)] * len(metrics_df)

    return metrics_df


# -------------------------------------------------------------------
# CLI / MAIN
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter grid search for CO2 markets using processed "
            "feature panels in data/co2_processed."
        )
    )
    parser.add_argument(
        "--start",
        type=str,
        default=START_DATE_DEFAULT,
        help=f"Start date (YYYY-MM-DD), default {START_DATE_DEFAULT}",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=END_DATE_DEFAULT,
        help=f"End date (YYYY-MM-DD), default {END_DATE_DEFAULT}",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK_DEFAULT,
        help=f"Sequence length (default {LOOKBACK_DEFAULT})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO_DEFAULT,
        help=f"Train ratio on sequences (default {TRAIN_RATIO_DEFAULT})",
    )
    parser.add_argument(
        "--markets",
        nargs="*",
        default=list(CO2_PROCESSED_FILES.keys()),
        help="Subset of markets to run; default is all.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help=(
            "Optional cap on number of configs per model (for debugging). "
            "If None, run full 243 configs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = args.start
    end_date = args.end
    lookback = args.lookback
    train_ratio = args.train_ratio
    markets = args.markets
    max_configs = args.max_configs

    print(
        f"[RUN] CO2 hyperparam pipeline\n"
        f"      Date range: {start_date} -> {end_date}\n"
        f"      Lookback:   {lookback}\n"
        f"      Train ratio:{train_ratio}\n"
        f"      Markets:    {markets}\n"
        f"      Max configs:{max_configs if max_configs is not None else 'FULL (243)'}"
    )

    all_metrics: List[pd.DataFrame] = []

    for market in markets:
        if market not in CO2_PROCESSED_FILES:
            print(f"[WARN] Market {market} not in CO2_PROCESSED_FILES; skipping.")
            continue

        print("=" * 80)
        print(f"[MARKET] {market}")
        try:
            mkt_df = train_models_for_market(
                market=market,
                start_date=start_date,
                end_date=end_date,
                lookback=lookback,
                train_ratio=train_ratio,
                max_configs=max_configs,
            )
            all_metrics.append(mkt_df)
            print(
                f"[DONE] {market}: {len(mkt_df)} rows of metrics "
                f"(models × configs)"
            )
        except Exception as e:
            print(f"[ERROR] {market}: {e}")

    if all_metrics:
        metrics_all_df = pd.concat(all_metrics, ignore_index=True)
        out_path = RESULTS_DIR / "co2_hparam_metrics.csv"
        metrics_all_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved combined metrics to {out_path}")
    else:
        print("[WARN] No metrics produced for any market.")


if __name__ == "__main__":
    main()
