from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.experiments.experiment_id import generate_spec_id


@dataclass
class ExperimentConfig:
    asset: str
    data_path: str

    task: str  # "regression_return" / "classification_direction"
    horizon: int

    model_name: str  # "naive", "xgboost", "lstm"
    validation_name: str  # "walk_forward_expanding"

    min_train_size: int
    test_size: int
    step_size: int

    feature_set_name: str = "none"
    objective_name: Optional[str] = None
    user_id: Optional[str] = None
    spec_id: str = field(init=False)

    def __post_init__(self):
        if self.user_id is not None:
            self.user_id = self.user_id.strip()
            if self.user_id == "" or self.user_id.lower() == "auto":
                self.user_id = None

        self.spec_id = generate_spec_id(self)