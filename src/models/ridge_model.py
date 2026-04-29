from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RidgeModel:
    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.feature_columns: list[str] = []
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.is_fitted = False

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        excluded = {
            "date",
            "close",
            "return",
            "y_true",
            "y_pred",
        }

        return [
            col for col in df.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
        ]

    def fit(self, train_df: pd.DataFrame) -> None:
        self.feature_columns = self._get_feature_columns(train_df)

        if not self.feature_columns:
            raise ValueError("Brak kolumn cech do treningu Ridge.")

        X_train = train_df[self.feature_columns]
        y_train = train_df["y_true"]

        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Model Ridge nie został wytrenowany (fit).")

        missing_cols = [col for col in self.feature_columns if col not in test_df.columns]
        if missing_cols:
            raise ValueError(f"Brak kolumn cech w zbiorze testowym: {missing_cols}")

        X_test = test_df[self.feature_columns]
        y_pred = self.model.predict(X_test)

        return pd.Series(y_pred, index=test_df.index)