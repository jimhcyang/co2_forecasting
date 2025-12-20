#!/usr/bin/env python3
from __future__ import annotations

"""
models.py

Reusable helpers for CO2 forecasting models + walk-forward evaluation that matches
the project's existing *daily* workflow (see ml_full_tuning.py / dl_models_lib.py):

Core idea
---------
We treat each *day* as one supervised example:

  X_t = [today's Price + today's features...]
  y_t = Price_{t+1}   (tomorrow's price)

So, after building y via `df["y"] = df["Price"].shift(-1)`, the last row has no
label and is dropped.

Walk-forward / rolling-window evaluation
----------------------------------------
Given a labeled panel with N rows (after dropping the final NaN-y row),
and a `window_size` (e.g. 10), we iterate windows:

  window 0 uses rows [0 .. window_size-1]
  window 1 uses rows [1 .. window_size]
  ...
  total_windows = N - window_size + 1

Within each window we do a chronological split:

  split = int(window_size * (1 - test_ratio))  (e.g. 10 * 0.9 = 9)
  train rows  = window[:split]
  test rows   = window[split:]

We train a fresh model on the train rows and predict the test rows.
With window_size=10 and test_ratio=0.1, each window produces exactly 1 prediction
(the last day in the window) — i.e., a rolling 9-day training window producing a
1-step-ahead forecast.

Note on "sequence models"
-------------------------
To match the existing daily pipeline, we treat each day's feature vector as one
"step" (sequence length = 1). This keeps the methodology consistent across
models (MLP/RNN/LSTM/GRU/TCN) and isolates the temporal structure to the rolling
window retraining (not within-sample sequences).
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import json
import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Metrics / utils
# -----------------------------

def set_global_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAPE": mape, "R2": r2, "n_samples": int(len(y_true))}


def _to_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


# -----------------------------
# Activations
# -----------------------------

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "swish":
        return Swish()
    raise ValueError(f"Unknown activation: {name}")


# -----------------------------
# Model definitions
# -----------------------------

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, num_layers: int, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_units))
            layers.append(get_activation(activation))
            d = hidden_units
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(x))


class RNNRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, num_layers: int):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_units, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)


class Chomp1d(nn.Module):
    """Trim padding on the right to keep output length equal to input length (causal conv)."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, activation: str):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.act1 = get_activation(activation)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.act2 = get_activation(activation)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.out_act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.chomp1(self.conv1(x)))
        out = self.act2(self.chomp2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(out + res)


class TCNRegressor(nn.Module):
    """
    Minimal TCN regressor. With seq_len=1 (the default in this workflow),
    the temporal conv is effectively degenerate — kept only for methodology parity.
    """
    def __init__(self, input_size: int, channels: int, blocks: int, kernel_size: int, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = input_size
        dilation = 1
        for _ in range(blocks):
            layers.append(TemporalBlock(in_ch, channels, kernel_size=kernel_size, dilation=dilation, activation=activation))
            in_ch = channels
            dilation *= 2
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, feat) -> (batch, feat, seq)
        x = x.transpose(1, 2)
        x = self.net(x)          # (batch, channels, seq)
        x = x[:, :, -1]          # last step
        return self.fc(x)


def build_model(model_name: str, input_dim: int, hp: Dict[str, Any]) -> nn.Module:
    """
    model_name: one of {"mlp","rnn","lstm","gru","tcn"}
    hp: hyperparameters dict (keys depend on model)
    """
    m = model_name.lower()
    if m == "mlp":
        return MLPRegressor(
            input_dim=input_dim,
            hidden_units=int(hp["hidden_units"]),
            num_layers=int(hp["num_layers"]),
            activation=str(hp["activation"]),
        )
    if m == "rnn":
        return RNNRegressor(
            input_size=input_dim,
            hidden_units=int(hp["hidden_units"]),
            num_layers=int(hp["num_layers"]),
        )
    if m == "lstm":
        return LSTMRegressor(
            input_size=input_dim,
            hidden_units=int(hp["hidden_units"]),
            num_layers=int(hp["num_layers"]),
        )
    if m == "gru":
        return GRURegressor(
            input_size=input_dim,
            hidden_units=int(hp["hidden_units"]),
            num_layers=int(hp["num_layers"]),
        )
    if m == "tcn":
        return TCNRegressor(
            input_size=input_dim,
            channels=int(hp["channels"]),
            blocks=int(hp["blocks"]),
            kernel_size=int(hp.get("kernel_size", 5)),
            activation=str(hp["activation"]),
        )
    raise ValueError(f"Unknown model_name: {model_name}")


# -----------------------------
# Training
# -----------------------------

@dataclass
class TrainResult:
    y_pred: np.ndarray
    train_loss_last: float


def _fit_one_model(
    *,
    model_name: str,
    hp: Dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    device: torch.device,
    seed: int = 42,
) -> TrainResult:
    """
    Fits a fresh model and predicts on X_te.

    Inputs are expected to already be scaled, and y_tr is expected to be scaled.
    """
    set_global_seeds(seed)

    X_tr = np.asarray(X_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.float32).reshape(-1, 1)
    X_te = np.asarray(X_te, dtype=np.float32)

    input_dim = X_tr.shape[1]
    model = build_model(model_name, input_dim=input_dim, hp=hp).to(device)

    # Shape for "sequence models" (seq_len=1)
    if model_name.lower() in {"rnn", "lstm", "gru", "tcn"}:
        X_tr_t = torch.from_numpy(X_tr[:, None, :]).to(device)
        X_te_t = torch.from_numpy(X_te[:, None, :]).to(device)
    else:
        X_tr_t = torch.from_numpy(X_tr).to(device)
        X_te_t = torch.from_numpy(X_te).to(device)

    y_tr_t = torch.from_numpy(y_tr).to(device)

    batch_size = int(hp.get("batch_size", 32))
    epochs = int(hp.get("epochs", 32))
    lr = float(hp.get("learning_rate", 1e-3))

    ds = TensorDataset(X_tr_t, y_tr_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_last = math.nan
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n = 0
        for xb, yb in dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            epoch_loss += loss.item() * bs
            n += bs
        train_loss_last = float(epoch_loss / max(n, 1))

    model.eval()
    with torch.no_grad():
        yhat = model(X_te_t).detach().cpu().numpy().reshape(-1)

    # free
    del model, optimizer, dl, ds, X_tr_t, X_te_t, y_tr_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return TrainResult(y_pred=yhat, train_loss_last=train_loss_last)


# -----------------------------
# Rolling-window evaluation
# -----------------------------

@dataclass
class SlidingWindowResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    dates: np.ndarray
    train_loss_last_mean: float
    n_windows: int


def sliding_window_forecast(
    *,
    df: pd.DataFrame,
    model_name: str,
    hp: Dict[str, Any],
    target_col: str = "y",
    window_size: int = 10,
    test_ratio: float = 0.1,
    drop_cols: Optional[Iterable[str]] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    max_windows: Optional[int] = None,
    log_path: Optional["pathlib.Path"] = None,
    log_extra: Optional[Dict[str, Any]] = None,
) -> SlidingWindowResult:
    """
    Walk-forward evaluation with fixed rolling windows and per-window retraining.

    Logging:
      If log_path is provided, appends one row per window in the "ml_full_tuning"
      style (with lists for y_true/y_pred).
    """
    from pathlib import Path  # local import to keep header light

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if drop_cols is None:
        drop_cols = []
    drop_cols = list(drop_cols)

    if target_col not in df.columns:
        raise ValueError(f"df must contain target_col='{target_col}'")

    df = df.copy().sort_index()
    feature_cols = [c for c in df.columns if c != target_col and c not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns left after drop_cols/target_col removal.")

    df = df[feature_cols + [target_col]].dropna()
    if len(df) < window_size:
        raise ValueError(f"Not enough labeled rows for window_size={window_size}: N={len(df)}")

    test_ratio = float(test_ratio)
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0,1)")

    split = int(round(window_size * (1.0 - test_ratio)))
    split = max(1, min(split, window_size - 1))

    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    dates_all: List[pd.Timestamp] = []
    train_losses: List[float] = []

    total_windows = len(df) - window_size + 1
    if max_windows is not None:
        total_windows = min(total_windows, int(max_windows))

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    for w in range(total_windows):
        win = df.iloc[w : w + window_size]
        X_win = win[feature_cols].to_numpy(dtype=float)
        y_win = win[target_col].to_numpy(dtype=float)

        X_tr, X_te = X_win[:split], X_win[split:]
        y_tr, y_te = y_win[:split], y_win[split:]
        dates_te = win.index[split:]

        # Naive baseline: yhat = today's Price
        if model_name.lower() == "naive_last_price":
            if "Price" not in win.columns:
                raise ValueError("Naive baseline requires a 'Price' column present in df/features.")
            y_hat = win["Price"].to_numpy(dtype=float)[split:]
            y_true_all.extend([float(v) for v in y_te])
            y_pred_all.extend([float(v) for v in y_hat])
            dates_all.extend(list(dates_te))
            train_losses.append(float("nan"))

            if log_path is not None:
                row = {
                    "timeframe": "co2",
                    "ticker": log_extra.get("market") if log_extra else None,
                    "market": log_extra.get("market") if log_extra else None,
                    "model_name": "NAIVE_LAST_PRICE",
                    "config_index": log_extra.get("config_index") if log_extra else -1,
                    "window_id": w,
                    "model_hyperparameters_dict": json.dumps({}),
                    "start_date": str(pd.to_datetime(dates_te[0])) if len(dates_te) else None,
                    "end_date": str(pd.to_datetime(dates_te[-1])) if len(dates_te) else None,
                    "test_data_values_list": json.dumps([float(v) for v in y_te]),
                    "test_data_model_predictions_list": json.dumps([float(v) for v in y_hat]),
                }
                pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            continue

        # Scale within-window using TRAIN portion only
        sx = RobustScaler().fit(X_tr)
        X_tr_s = sx.transform(X_tr)
        X_te_s = sx.transform(X_te)

        sy = StandardScaler().fit(_to_2d(y_tr))
        y_tr_s = sy.transform(_to_2d(y_tr)).reshape(-1)

        tr = _fit_one_model(
            model_name=model_name,
            hp=hp,
            X_tr=X_tr_s,
            y_tr=y_tr_s,
            X_te=X_te_s,
            device=device,
            seed=seed,
        )
        y_hat_s = tr.y_pred
        y_hat = sy.inverse_transform(_to_2d(y_hat_s)).reshape(-1)

        y_true_all.extend([float(v) for v in y_te])
        y_pred_all.extend([float(v) for v in y_hat])
        dates_all.extend(list(dates_te))
        train_losses.append(float(tr.train_loss_last))

        if log_path is not None:
            row = {
                "timeframe": "co2",
                "ticker": log_extra.get("market") if log_extra else None,
                "market": log_extra.get("market") if log_extra else None,
                "model_name": model_name.upper(),
                "config_index": log_extra.get("config_index") if log_extra else None,
                "window_id": w,
                "model_hyperparameters_dict": json.dumps(hp),
                "start_date": str(pd.to_datetime(dates_te[0])) if len(dates_te) else None,
                "end_date": str(pd.to_datetime(dates_te[-1])) if len(dates_te) else None,
                "test_data_values_list": json.dumps([float(v) for v in y_te]),
                "test_data_model_predictions_list": json.dumps([float(v) for v in y_hat]),
            }
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    y_true = np.asarray(y_true_all, dtype=float)
    y_pred = np.asarray(y_pred_all, dtype=float)
    dates = np.asarray(dates_all)

    loss_mean = float(np.nanmean(np.asarray(train_losses, dtype=float))) if train_losses else float("nan")
    return SlidingWindowResult(
        y_true=y_true,
        y_pred=y_pred,
        dates=dates,
        train_loss_last_mean=loss_mean,
        n_windows=int(total_windows),
    )
