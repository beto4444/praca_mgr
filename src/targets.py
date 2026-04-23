from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return"] = np.log(out["close"] / out["close"].shift(1))
    return out


def build_regression_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Buduje target regresyjny jako skumulowany log-return
    w horyzoncie od t+1 do t+h.

    Dla horizon=1:
        y_true = return_{t+1}

    Dla horizon>1:
        y_true = return_{t+1} + return_{t+2} + ... + return_{t+h}
               = log(P_{t+h} / P_t)
    """
    if horizon < 1:
        raise ValueError("horizon musi być >= 1")

    out = df.copy()

    future_returns = [out["return"].shift(-i) for i in range(1, horizon + 1)]
    out["y_true"] = pd.concat(future_returns, axis=1).sum(axis=1, min_count=horizon)

    return out


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna().reset_index(drop=True)
    return out