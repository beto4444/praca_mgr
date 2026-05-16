from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    name = "arima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        horizon: int = 1,
    ) -> None:
        self.order = order
        self.horizon = horizon
        self.model_fit = None
        self.is_fitted = False
        self.train_mean_: float | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        if "y_true" not in train_df.columns:
            raise ValueError("Brak kolumny y_true do treningu ARIMA.")

        y_train = train_df["y_true"].astype(float).dropna()

        if y_train.empty:
            raise ValueError("Pusty szereg y_true do treningu ARIMA.")

        self.train_mean_ = float(y_train.mean())

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(
                    y_train,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.model_fit = model.fit()

            self.is_fitted = True

        except Exception:
            self.model_fit = None
            self.is_fitted = True

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Model ARIMA nie został wytrenowany (fit).")

        n_test = len(test_df)

        if n_test == 0:
            return pd.Series(dtype=float, index=test_df.index)

        # Po purge train kończy się horizon kroków przed pierwszym targetem testowym.
        # Dla test_size=1:
        # H1 -> bierzemy forecast step 1
        # H3 -> bierzemy forecast step 3
        # H5 -> bierzemy forecast step 5
        forecast_steps = self.horizon + n_test - 1

        if self.model_fit is None:
            if self.train_mean_ is None:
                raise RuntimeError("Brak modelu ARIMA i brak fallback mean.")
            y_pred = np.repeat(self.train_mean_, n_test)
            return pd.Series(y_pred, index=test_df.index)

        try:
            forecast = self.model_fit.forecast(steps=forecast_steps)
            forecast_values = np.asarray(forecast, dtype=float)

            y_pred = forecast_values[self.horizon - 1 : self.horizon - 1 + n_test]

            if len(y_pred) != n_test:
                y_pred = np.resize(y_pred, n_test)

            return pd.Series(y_pred, index=test_df.index)

        except Exception:
            if self.train_mean_ is None:
                raise RuntimeError("Forecast ARIMA nieudany i brak fallback mean.")

            y_pred = np.repeat(self.train_mean_, n_test)
            return pd.Series(y_pred, index=test_df.index)