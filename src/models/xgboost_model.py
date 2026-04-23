from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from xgboost import XGBRegressor

from src.features.feature_builder import get_feature_columns


@dataclass
class XGBoostModelConfig:
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42


class XGBoostModel:
    name = "xgboost"

    def __init__(self, config: Optional[XGBoostModelConfig] = None) -> None:
        self.config = config or XGBoostModelConfig()
        self.feature_columns: list[str] | None = None
        self.model: XGBRegressor | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.feature_columns = get_feature_columns(train_df)

        if not self.feature_columns:
            raise ValueError("Brak kolumn cech do treningu XGBoost.")

        X_train = train_df[self.feature_columns]
        y_train = train_df["y_true"]

        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
        )

        self.model.fit(X_train, y_train)

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model XGBoost nie został wytrenowany.")
        if self.feature_columns is None:
            raise RuntimeError("Brak zapisanych kolumn cech.")

        X_test = test_df[self.feature_columns]
        preds = self.model.predict(X_test)

        return pd.Series(preds, index=test_df.index, name="y_pred")