from __future__ import annotations

import pandas as pd


class HistoricalMeanModel:
    name = "historical_mean"

    def __init__(self) -> None:
        self.is_fitted = False
        self.mean_target: float | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        if "y_true" not in train_df.columns:
            raise ValueError("Brak kolumny 'y_true' w danych treningowych.")

        self.mean_target = float(train_df["y_true"].mean())
        self.is_fitted = True

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.mean_target is None:
            raise RuntimeError("Model nie został wytrenowany (fit).")

        return pd.Series(
            self.mean_target,
            index=test_df.index,
            name="y_pred",
        )