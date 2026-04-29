from __future__ import annotations

from itertools import product
from typing import Optional

from src.experiments.config import ExperimentConfig


def build_experiment_plan(
    assets: list[str],
    tasks: list[str],
    horizons: list[int],
    model_names: list[str],
    validation_name: str,
    min_train_size: int,
    test_size: int,
    step_size: int,
    data_paths: dict[str, str],
    feature_set_names: list[str],
    objective_names: list[Optional[str]],
) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []

    for asset, task, horizon, model_name, feature_set_name, objective_name in product(
        assets,
        tasks,
        horizons,
        model_names,
        feature_set_names,
        objective_names,
    ):
        # Normalizacja kombinacji model-specyficznych
        if model_name in {"naive", "historical_mean"}:
            feature_set_name = "none"
            objective_name = None

        elif model_name == "xgboost":
            if feature_set_name == "none":
                continue  # XGBoost bez cech nie ma sensu
            if objective_name is None:
                objective_name = "mse"

        elif model_name == "ridge":
            if feature_set_name == "none":
                continue  # Ridge bez cech nie ma sensu
            if objective_name is None:
                objective_name = "mse"

        config = ExperimentConfig(
            asset=asset,
            data_path=data_paths[asset],
            task=task,
            horizon=horizon,
            model_name=model_name,
            validation_name=validation_name,
            min_train_size=min_train_size,
            test_size=test_size,
            step_size=step_size,
            feature_set_name=feature_set_name,
            objective_name=objective_name,
        )
        configs.append(config)

    # Usunięcie duplikatów po spec_id
    unique_configs: dict[str, ExperimentConfig] = {}
    for cfg in configs:
        unique_configs[cfg.spec_id] = cfg

    return list(unique_configs.values())