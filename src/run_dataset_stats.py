from __future__ import annotations

import pandas as pd

from src.analysis.dataset_stats import summarize_asset_horizon


if __name__ == "__main__":
    data_paths = {
        "SPY": "data/spy.csv",
        "EURUSD": "data/eurusd.csv",
        "XAUUSD": "data/xauusd.csv",
    }

    rows = []
    for asset, path in data_paths.items():
        for horizon in [1, 3, 5]:
            rows.append(summarize_asset_horizon(path, asset, horizon))

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("results/dataset_stats.csv", index=False)