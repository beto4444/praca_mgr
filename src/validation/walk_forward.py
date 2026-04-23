from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import pandas as pd


@dataclass
class WalkForwardConfig:
    min_train_size: int
    test_size: int = 1
    step_size: int = 1


class WalkForwardExpandingSplitter:
    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def split(self, df: pd.DataFrame) -> Iterator[Tuple[pd.Index, pd.Index]]:
        n = len(df)
        start_test = self.config.min_train_size

        while start_test + self.config.test_size <= n:
            train_idx = df.index[:start_test]
            test_idx = df.index[start_test:start_test + self.config.test_size]

            yield train_idx, test_idx

            start_test += self.config.step_size