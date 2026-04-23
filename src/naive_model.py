from __future__ import annotations

import pandas as pd


def naive_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Naive forecast:
    przewidujemy, że następny return będzie równy ostatniemu zaobserwowanemu returnowi.
    Dla targetu y_true = return_{t+1},
    predykcja w chwili t to y_pred = return_t.
    """
    out = df.copy()
    out["y_pred"] = out["return"]
    return out