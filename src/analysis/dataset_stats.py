from __future__ import annotations

import pandas as pd

from src.data_loader import load_price_data
from src.targets import add_log_returns, build_regression_target, finalize_dataset


def build_target_dataset(data_path: str, horizon: int) -> pd.DataFrame:
    df = load_price_data(data_path)
    df = add_log_returns(df)
    df = build_regression_target(df, horizon=horizon)
    df = finalize_dataset(df)
    return df


def summarize_target_distribution(df: pd.DataFrame) -> dict:
    y = df["y_true"]

    return {
        "n_obs": int(len(df)),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "mean_y_true": float(y.mean()),
        "std_y_true": float(y.std(ddof=0)),
        "pct_positive": float((y > 0).mean()),
        "pct_negative": float((y < 0).mean()),
        "pct_zero": float((y == 0).mean()),
        "q05": float(y.quantile(0.05)),
        "q25": float(y.quantile(0.25)),
        "median": float(y.quantile(0.50)),
        "q75": float(y.quantile(0.75)),
        "q95": float(y.quantile(0.95)),
    }


def summarize_asset_horizon(data_path: str, asset: str, horizon: int) -> dict:
    df = build_target_dataset(data_path=data_path, horizon=horizon)
    stats = summarize_target_distribution(df)
    return {
        "asset": asset,
        "horizon": horizon,
        **stats,
    }