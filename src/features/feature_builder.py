from __future__ import annotations

from typing import Iterable

import pandas as pd


def add_lag_features(df: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    """
    Dodaje cechy opóźnionych zwrotów:
    lag_1, lag_2, ..., lag_n
    """
    out = df.copy()

    if "return" not in out.columns:
        raise ValueError("Brak kolumny 'return' potrzebnej do budowy lag features.")

    for lag in lags:
        out[f"lag_{lag}"] = out["return"].shift(lag)

    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: Iterable[int],
) -> pd.DataFrame:
    """
    Dodaje podstawowe statystyki kroczące:
    rolling_mean_k
    rolling_std_k
    """
    out = df.copy()

    if "return" not in out.columns:
        raise ValueError("Brak kolumny 'return' potrzebnej do budowy rolling features.")

    for window in windows:
        out[f"rolling_mean_{window}"] = out["return"].rolling(window=window).mean()
        out[f"rolling_std_{window}"] = out["return"].rolling(window=window).std()

    return out


def build_features(df: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    """
    Główna funkcja budująca zestawy cech.

    Dostępne warianty:
    - 'none'          -> bez dodatkowych cech
    - 'lags_only'     -> tylko lag features
    - 'lags_rolling'  -> lagi + rolling mean/std
    """
    out = df.copy()

    if feature_set_name == "none":
        return out

    if feature_set_name == "lags_only":
        out = add_lag_features(out, lags=range(1, 11))
        return out

    if feature_set_name == "lags_rolling":
        out = add_lag_features(out, lags=range(1, 11))
        out = add_rolling_features(out, windows=[5, 10, 20])
        return out

    raise ValueError(f"Nieznany zestaw cech: {feature_set_name}")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Zwraca listę kolumn cech, pomijając kolumny metadanych i target.
    """
    excluded = {"date", "close", "return", "y_true", "y_pred"}
    return [col for col in df.columns if col not in excluded]


def finalize_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usuwa wiersze z NaN powstałe w wyniku budowy cech.
    """
    return df.dropna().reset_index(drop=True)