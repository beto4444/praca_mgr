from __future__ import annotations

import pandas as pd


class NaiveModel:
    name = "naive"

    def __init__(self, horizon: int = 1) -> None:
        if horizon < 1:
            raise ValueError("horizon musi być >= 1")
        self.horizon = horizon
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> None:
        self.is_fitted = True

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Model nie został wytrenowany (fit).")

        if "return" not in test_df.columns:
            raise ValueError("Brak kolumny 'return' w danych wejściowych.")

        return (self.horizon * test_df["return"]).copy()