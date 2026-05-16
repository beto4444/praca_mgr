from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd

from src.data_loader import load_price_data
from src.targets import add_log_returns, build_regression_target
from src.metrics import evaluate_regression
from src.validation.walk_forward import WalkForwardConfig, WalkForwardExpandingSplitter
from src.experiments.config import ExperimentConfig
from src.experiments.experiment_id import generate_run_id
from src.models.naive_model import NaiveModel
from src.features.feature_builder import build_features, finalize_feature_dataset
from src.models.xgboost_model import XGBoostModel
from src.models.historical_mean_model import HistoricalMeanModel
from src.models.ridge_model import RidgeModel
from src.models.arima_model import ARIMAModel

METRIC_KEY_COLUMNS = [
    "run_id",
]


def save_predictions(df_pred: pd.DataFrame, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(output, index=False)


def save_metrics(result: dict, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([result])

    if output.exists():
        old_df = pd.read_csv(output)
        missing_cols = [c for c in METRIC_KEY_COLUMNS if c not in old_df.columns]

        if missing_cols:
            final_df = new_df
        else:
            mask = pd.Series(True, index=old_df.index)
            for col in METRIC_KEY_COLUMNS:
                old_val = old_df[col].astype(str)
                new_val = str(new_df.iloc[0][col])
                mask &= old_val == new_val

            old_df = old_df.loc[~mask].copy()
            final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_csv(output, index=False)


def prepare_base_dataset(data_path: str, horizon: int, feature_set_name: str) -> pd.DataFrame:
    if feature_set_name is None:
        feature_set_name = "none"

    df = load_price_data(data_path)
    df = add_log_returns(df)
    df = build_regression_target(df, horizon=horizon)
    df = build_features(df, feature_set_name=feature_set_name)
    df = finalize_feature_dataset(df)
    return df


def build_splitter(config: ExperimentConfig):
    if config.validation_name == "walk_forward_expanding":
        return WalkForwardExpandingSplitter(
            WalkForwardConfig(
                min_train_size=config.min_train_size,
                test_size=config.test_size,
                step_size=config.step_size,
            )
        )

    raise ValueError(f"Nieznana metoda walidacji: {config.validation_name}")


def build_model(config: ExperimentConfig):
    if config.model_name == "naive":
        return NaiveModel(horizon=config.horizon)

    if config.model_name == "xgboost":
        return XGBoostModel()

    if config.model_name == "historical_mean":
        return HistoricalMeanModel()

    if config.model_name == "ridge":
        return RidgeModel()

    if config.model_name == "arima":
        return ARIMAModel(order=(1, 0, 1), horizon=config.horizon)

    raise ValueError(f"Nieznany model: {config.model_name}")

def run_experiment(config: ExperimentConfig):
    df = prepare_base_dataset(config.data_path, config.horizon, config.feature_set_name)
    splitter = build_splitter(config)
    model = build_model(config)

    prediction_frames = []

    for train_idx, test_idx in splitter.split(df):
        train_idx = list(train_idx)
        test_idx = list(test_idx)

        if config.horizon > 1:
            first_test_idx = min(test_idx)
            max_allowed_train_idx = first_test_idx - config.horizon
            train_idx = [idx for idx in train_idx if idx <= max_allowed_train_idx]

        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()

        model.fit(train_df)
        test_df["y_pred"] = model.predict(test_df)
        pred_part = test_df[["date", "close", "return", "y_true", "y_pred"]].copy()
        prediction_frames.append(pred_part)

    predictions = pd.concat(prediction_frames, ignore_index=True)

    metrics = evaluate_regression(predictions)

    result = {
        "user_id": config.user_id,
        "spec_id": config.spec_id,
        "asset": config.asset,
        "task": config.task,
        "model": config.model_name,
        "validation": config.validation_name,
        "horizon": config.horizon,
        "feature_set_name": config.feature_set_name,
        "objective_name": config.objective_name,
        "min_train_size": config.min_train_size,
        "test_size": config.test_size,
        "step_size": config.step_size,
        **metrics,
    }

    return predictions, result


def execute_experiment(config: ExperimentConfig):
    run_id = generate_run_id()
    started_at = datetime.now()

    try:
        predictions, result = run_experiment(config)
        finished_at = datetime.now()
        elapsed_seconds = (finished_at - started_at).total_seconds()

        predictions = predictions.copy()
        predictions["run_id"] = run_id
        predictions["spec_id"] = config.spec_id
        predictions["user_id"] = config.user_id

        result = {
            "run_id": run_id,
            "status": "SUCCESS",
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed_seconds,
            "error_type": None,
            "error_message": None,
            **result,
        }

        return predictions, result

    except KeyboardInterrupt:
        finished_at = datetime.now()
        elapsed_seconds = (finished_at - started_at).total_seconds()

        interrupted_result = {
            "run_id": run_id,
            "status": "INTERRUPTED",
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed_seconds,
            "error_type": "KeyboardInterrupt",
            "error_message": "Experiment interrupted by user.",
            "user_id": config.user_id,
            "spec_id": config.spec_id,
            "asset": config.asset,
            "task": config.task,
            "model": config.model_name,
            "validation": config.validation_name,
            "horizon": config.horizon,
            "feature_set_name": config.feature_set_name,
            "objective_name": config.objective_name,
            "min_train_size": config.min_train_size,
            "test_size": config.test_size,
            "step_size": config.step_size,
        }

        return None, interrupted_result

    except Exception as e:
        finished_at = datetime.now()
        elapsed_seconds = (finished_at - started_at).total_seconds()

        fail_result = {
            "run_id": run_id,
            "status": "FAIL",
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed_seconds,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "user_id": config.user_id,
            "spec_id": config.spec_id,
            "asset": config.asset,
            "task": config.task,
            "model": config.model_name,
            "validation": config.validation_name,
            "horizon": config.horizon,
            "feature_set_name": config.feature_set_name,
            "objective_name": config.objective_name,
            "min_train_size": config.min_train_size,
            "test_size": config.test_size,
            "step_size": config.step_size,
        }

        return None, fail_result