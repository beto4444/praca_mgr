from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ASSET_CODES = {
    "SPY": "SPY",
    "EURUSD": "EURUSD",
    "XAUUSD": "XAUUSD",
}

TASK_CODES = {
    "regression_return": "REG",
    "classification_direction": "CLS",
}

MODEL_CODES = {
    "naive": "NAV",
    "xgboost": "XGB",
    "lstm": "LSTM",
    "arima": "ARI",
    "tcn": "TCN",
    "historical_mean": "HME",
}

VALIDATION_CODES = {
    "walk_forward_expanding": "WFE",
    "full_sample": "FUL",
}

FEATURE_CODES = {
    "none": "NONE",
    "lags_only": "LAG10",
    "lags_rolling": "LAG10ROLL",
}

OBJECTIVE_CODES = {
    None: "NA",
    "mse": "MSE",
    "mae": "MAE",
    "directional": "DIR",
    "sharpe_proxy": "SHRP",
    "return_mean": "RET",
}


def _safe_code(mapping: dict, value, default: str = "UNK") -> str:
    return mapping.get(value, default)


def generate_spec_id(config) -> str:
    asset_code = _safe_code(ASSET_CODES, config.asset)
    task_code = _safe_code(TASK_CODES, config.task)
    model_code = _safe_code(MODEL_CODES, config.model_name)
    validation_code = _safe_code(VALIDATION_CODES, config.validation_name)
    feature_code = _safe_code(FEATURE_CODES, config.feature_set_name)
    objective_code = _safe_code(OBJECTIVE_CODES, config.objective_name)

    return (
        f"{asset_code}_"
        f"{task_code}_"
        f"{model_code}_"
        f"{validation_code}_"
        f"H{config.horizon}_"
        f"{feature_code}_"
        f"{objective_code}_"
        f"TR{config.min_train_size}_"
        f"TE{config.test_size}_"
        f"ST{config.step_size}"
    )


def current_run_date_str() -> str:
    return datetime.now().strftime("%d%m%y")


def _next_daily_sequence(sequence_file: str = "configs/sequences.dat") -> int:
    path = Path(sequence_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    today = current_run_date_str()

    if not path.exists():
        df = pd.DataFrame([{"date": today, "num_of_sequences": 1}])
        df.to_csv(path, index=False)
        return 1

    df = pd.read_csv(path, dtype={"date": str, "num_of_sequences": int})

    if df.empty:
        df = pd.DataFrame([{"date": today, "num_of_sequences": 1}])
        df.to_csv(path, index=False)
        return 1

    mask = df["date"] == today

    if mask.any():
        last_idx = df.index[mask][-1]
        next_seq = int(df.loc[last_idx, "num_of_sequences"]) + 1
        df.loc[last_idx, "num_of_sequences"] = next_seq
    else:
        next_seq = 1
        new_row = pd.DataFrame([{"date": today, "num_of_sequences": next_seq}])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(path, index=False)
    return next_seq


def generate_run_id(sequence_file: str = "configs/sequences.dat") -> str:
    date_str = current_run_date_str()
    seq = _next_daily_sequence(sequence_file=sequence_file)
    return f"{date_str}_{seq}"