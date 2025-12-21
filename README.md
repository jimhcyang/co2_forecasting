# CO₂ Price Forecasting (Rolling Window Tuning, PyTorch)

This repository contains a **reproducible pipeline** for building daily CO₂ market feature panels and running a **rolling-window hyperparameter search** for multiple neural forecasting models.

The core goal is:

> Use **today’s data** (including today’s CO₂ price and macro/technical features) to predict **tomorrow’s CO₂ price**.

**Evaluation + ranking convention (repo-wide):**

- We always *report* metrics in this order:
  1) **MAPE**  2) **RMSE/mean(price)**  3) **MAE**  4) **R²**
- We always pick “best” configs by:
  **min MAPE → min RMSE/mean(price) → min MAE → max R²**.

This repo supports both:
- **Tuning** (grid search) that selects best configs using the above rule.
- **Final evaluation** on a later date range (no tuning) using `evaluate_best.py`, logging to `test_log/`.

---

## Directory structure

Recommended layout:

```
repo_root/
  scripts/
    input.py          # builds processed per-market feature CSVs
    models.py         # model definitions + rolling-window walk-forward evaluation
    tuning.py         # hyperparameter grid search + logging + summary outputs (best = MAPE-first)
    evaluate_best.py  # run the chosen best configs on a (later) evaluation range; logs to test_log/
    score_logs.py     # optional: aggregate dl_logs into per-config scores + best-config JSON (MAPE-first)
  data/
    co2/              # raw market files (Excel/CSV, per MARKET_CONFIGS in input.py)
    co2_processed/
      Australia_features.csv
      California_features.csv
      ...
  dl_logs/
    co2/
      <MODEL>/
        <MARKET>/
          <MODEL>_<MARKET>.csv
  test_log/
    co2/
      <MODEL>/
        <MARKET>/
          <MODEL>_<MARKET>.csv
  results/
    co2_hparam_metrics.csv
    co2_best_configs.json
    co2_test_metrics.csv
    # optional (if you run score_logs.py):
    # co2_config_scores.csv
    # co2_best_by_market_model.csv
    # co2_best_by_model_across_markets.csv
  requirements.txt
```


> **If you renamed files:**  
> - `tuning.py` imports `from models import ...`  
> - so `models.py` must remain importable as `scripts/models.py` (or you must update the import).
> - Likewise `input.py` is expected at `scripts/input.py` when following the examples below.

---

## Data: what’s in `data/co2_processed/<Market>_features.csv`

`input.py` constructs **business-day** panels and writes one processed CSV per market.

### Business-day index + interpolation
- The pipeline reindexes everything to a fixed **business-day index** (`freq="B"`) between the configured `START_DATE` and `END_DATE`.
- Missing values are filled with **time interpolation + forward/backward fill** so each business day has a complete feature vector.

This means **weekends are removed** and holidays may be represented via interpolated values (depending on the original series).

### Columns (example)
Each processed file contains:

- `Price`: the CO₂ market price series (spot or ETF proxy depending on market)
- `equity_index`: market-specific equity index (Yahoo Finance)
- `fx_rate`: market-specific FX rate (Yahoo Finance)
- Common commodities from Yahoo Finance:
  - `CL_F` (crude oil), `NG_F` (natural gas), `RB_F` (gasoline), `HRC_F` (steel), `ALI_F` (aluminum)
- Technical indicators computed **on the CO₂ Price only**:
  - `SMA_20`, `PPO_12_26`, `RSI_14`, `ROC_10`, `ATR_14`, `CO_chaikin_3_10`

See `scripts/input.py` for exact tickers and processing steps.

---

## Supervised learning formulation

In `tuning.py`, for each market panel we build the label:

- **Target:** `y_t = Price_{t+1}` (tomorrow’s price)
- **Features:** all columns **including today’s `Price_t`**

Concretely:

```python
df["y"] = df["Price"].shift(-1)
df = df.dropna(subset=["y"])
```

Because of the `shift(-1)`, the final row in any date slice has no label and is dropped.

### Choosing `--start` and `--end` correctly

If you want to train/evaluate on **Sept 1, 2022 → Dec 31, 2022**, you should set:

- `--start 2022-09-01`
- `--end 2023-01-02`

Reason: the last supervised example inside the slice uses tomorrow’s price.
To keep the label for the final business day of Dec (typically 2022-12-30), you need the **next business day** (2023-01-02) included in the slice.

---

## Rolling-window (walk-forward) evaluation

All models are evaluated with **fixed-size rolling windows** using `sliding_window_forecast()` in `scripts/models.py`.

### Window mechanics

Parameters:
- `window_size = W` (e.g., 20)
- `test_ratio = r` (e.g., 0.05)

For each window of length `W`, we split chronologically:

- `split = round(W * (1 - r))`
- Train set: first `split` rows
- Test set: remaining rows

Example:

- `W = 20`, `r = 0.05`
- `split = round(20 * 0.95) = 19`
- Train on **19 rows**
- Predict on **1 row**

So each window produces a **1-step-ahead forecast** at its end.

Windows slide forward by 1 business day:

- window 0 uses rows `[0..W-1]`
- window 1 uses rows `[1..W]`
- …
- total windows = `N - W + 1` (where `N` is labeled rows in your slice)

### Why this rolling setup is used
This design mimics a realistic scenario:

- At each time step, you only use information that would have been available **up to that day**.
- You retrain frequently on the most recent `W-1` labeled examples and predict the next day.

It is especially useful when market dynamics shift over time and you want **local adaptation** rather than one global fit.

---

## Scaling and leakage prevention

Inside each rolling window:

- Features are scaled using `RobustScaler` **fit on the train rows only**
- Targets are scaled using `StandardScaler` **fit on the train rows only**
- Predictions are inverse-transformed back to the original price scale

This prevents leakage from the test portion of the window.

---

## Models

Implemented in `scripts/models.py`:

- `MLP`
- `RNN`
- `LSTM`
- `GRU`
- `TCN`
- Baseline: `NAIVE_LAST_PRICE` (predict tomorrow = today’s price)

> Note: For parity with the existing “daily” workflow, sequence models are used with `seq_len = 1`
> (each day is treated as a single step). The temporal structure primarily comes from the **rolling retrain**.

---

## Hyperparameter tuning

`tuning.py` runs grid search per market and per model:

### Core grid (MLP/RNN/LSTM/GRU)
- `hidden_units ∈ {32, 64, 128}`
- `num_layers ∈ {2, 3, 4}`
- `activation ∈ {relu, tanh, swish}`
- `learning_rate ∈ {1e-2, 3e-3, 1e-3, 3e-4, 1e-4}`
- `batch_size ∈ {32}`
- `epochs ∈ {16, 32, 64}`

Total (default in `tuning.py`): **405 configs per model**

> You can expand/shrink these lists in `scripts/tuning.py` to trade off compute vs coverage.

### TCN grid
- `channels ∈ {32, 64, 128}`
- `blocks ∈ {2, 3, 4}`
- `kernel_size = 5` (fixed)
- plus the same `{activation, learning_rate, batch_size, epochs}`

Total (default in `tuning.py`): **405 configs**

### What “best” means
After evaluating predictions across **all rolling windows** for a config, we compute (repo-wide metric order):

1. **MAPE** (lower is better)
2. **RMSE/mean(price)** a.k.a. `RMSE_over_mean_price` (lower is better)
3. **MAE** (lower is better)
4. **R²** (higher is better)

We may also record `RMSE`, `mean_price`, and `n_samples` as helpful auxiliary fields.

`tuning.py` selects the **best config per (market, model)** by:

**min MAPE → min RMSE/mean(price) → min MAE → max R²**

This same rule is also used by `score_logs.py` when aggregating from log files.

---


## Outputs

### 1) Per-window logs (append-only)

For each config and each rolling window, we append one row to:

```
dl_logs/co2/<MODEL>/<MARKET>/<MODEL>_<MARKET>.csv
```

Each row includes:
- `config_index`
- `window_id`
- `model_hyperparameters_dict`
- `test_data_values_list`
- `test_data_model_predictions_list`
- plus metadata like market/date range

This mirrors the “list-in-a-row” logging style used elsewhere in your project.

> **Note:** these logs store per-window *predictions*, not per-window metrics. All metrics (MAPE/RMSE/MAE/R²) are computed **once per config** on the full concatenated `y_true/y_pred` across all windows. If you want to reduce disk I/O, you can run `tuning.py` or `evaluate_best.py` with `--no-logs`.

### 2) Summary metrics per config

Written to:

```
results/co2_hparam_metrics.csv
```

One row per (market, model, config_index), containing aggregated metrics across all windows.

### 3) Best configs

Written to:

```
results/co2_best_configs.json
```

Structure:

```json
{
  "Australia": {
    "mlp": { ...best row dict... },
    "gru": { ... },
    ...
  },
  ...
}
```

### 4) Final evaluation logs (no tuning)

After you have selected best configs, you can run **out-of-sample evaluation** (e.g., later dates) using `evaluate_best.py`.
Per-window logs are written to:

```
test_log/co2/<MODEL>/<MARKET>/<MODEL>_<MARKET>.csv
```

### 5) Final evaluation summary metrics

Aggregated metrics for the evaluation run are written to:

```
results/co2_test_metrics.csv
```


---

## How to run

### 0) Install dependencies

From repo root:

```bash
pip install -r requirements.txt
```

### 1) Build processed feature panels (once)

```bash
python -m scripts.input
```

This writes `data/co2_processed/<Market>_features.csv`.

### 2) Run a **full** hyperparameter search (recommended settings)

For Sept–Dec 2022 with a 20-day window and 19-train/1-predict:

```bash
python -m scripts.tuning \
  --start 2022-09-01 \
  --end 2023-01-02 \
  --window-size 20 \
  --test-ratio 0.05
```

### Useful options

**Run only one market:**
```bash
python -m scripts.tuning --markets Australia --start 2022-09-01 --end 2023-01-02 --window-size 20 --test-ratio 0.05
```

**Resume (skip configs already logged):**
```bash
python -m scripts.tuning --skip-existing --start 2022-09-01 --end 2023-01-02 --window-size 20 --test-ratio 0.05
```

**Debug quickly:**
```bash
python -m scripts.tuning --max-configs 5 --max-windows 30 --markets Australia
```


### 3) (Optional) Aggregate from `dl_logs/` if you resumed runs

If you ran tuning in multiple sessions (or want to recompute best configs from logs), you can aggregate logs into scores and best-config files:

```bash
python -m scripts.score_logs
```

This writes `results/co2_config_scores.csv` and re-creates `results/co2_best_configs.json` using the **MAPE-first** rule.

### 4) Run final evaluation with the chosen best configs (no tuning)

Example (default date range 2023-01-01 → 2025-11-30):

```bash
python -m scripts.evaluate_best
```

Evaluate a subset of markets:

```bash
python -m scripts.evaluate_best --markets Australia EU_EEX
```

Skip any market/model pairs that already have a `test_log/` file:

```bash
python -m scripts.evaluate_best --skip-existing
```

---

## Selecting HPs for later evaluation

After tuning completes:

1. Open `results/co2_best_configs.json`
2. For each market/model, read the chosen hyperparameters (**chosen by min MAPE**, ties by RMSE/mean, MAE, then max R²)
3. Run `evaluate_best.py` on your final evaluation date range (and log to `test_log/`)

If you want to re-rank or audit the selection, use `results/co2_hparam_metrics.csv` (per-config summaries) or run `scripts/score_logs.py` to aggregate from `dl_logs/`.

---

## Notes / cautions

- **Compute cost**: full grid search is large:
  - 6 markets × (4 core models × 405 + TCN 405 + baseline)  
  This can take a long time on CPU.
- **Window size tradeoff**:
  - Small `window_size` adapts quickly but uses fewer training points per window.
  - Larger `window_size` gives more stable training but adapts more slowly.
- **Interpolation**: since inputs are aligned to business days and filled, some “days” may represent interpolations rather than actual trades.

---

## Script entrypoints

- `scripts/input.py`: build processed features
- `scripts/models.py`: models + rolling evaluation (library)
- `scripts/tuning.py`: grid search tuning + logs + best configs (MAPE-first)
- `scripts/evaluate_best.py`: final evaluation (no tuning) using `results/co2_best_configs.json` → logs to `test_log/` + writes `results/co2_test_metrics.csv`
- `scripts/score_logs.py`: optional aggregation from `dl_logs/` to rebuild `results/co2_best_configs.json` (MAPE-first)

