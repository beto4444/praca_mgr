from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score


def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(mean_squared_error(y_true, y_pred))


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    true_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)
    return float(accuracy_score(true_dir, pred_dir))


def synthetic_returns(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    return np.sign(y_pred) * y_true


def mean_return(r: pd.Series) -> float:
    return float(np.mean(r))


def pseudo_sharpe(r: pd.Series, eps: float = 1e-8) -> float:
    std = float(np.std(r, ddof=0))
    if std < eps:
        return 0.0
    return float(np.mean(r) / std)


def evaluate_regression(df: pd.DataFrame) -> dict:
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    r_syn = synthetic_returns(y_true, y_pred)

    return {
        "n_obs": int(len(df)),
        "mse": mse(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "mean_synthetic_return": mean_return(r_syn),
        "pseudo_sharpe": pseudo_sharpe(r_syn),
    }