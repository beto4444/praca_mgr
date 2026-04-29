from __future__ import annotations

from src.experiments.planner import build_experiment_plan
from src.experiments.runner import execute_experiment, save_predictions, save_metrics


if __name__ == "__main__":
    data_paths = {
        "SPY": "data/spy.csv",
        "EURUSD": "data/eurusd.csv",
        "XAUUSD": "data/xauusd.csv",
    }

    plan = build_experiment_plan(
        assets=["SPY", "EURUSD", "XAUUSD"],
        tasks=["regression_return"],
        horizons=[1, 3, 5],
        #model_names=["naive", "historical_mean", "xgboost"],
        model_names=["ridge"], #testowo tylko Ridge
        validation_name="walk_forward_expanding",
        min_train_size=1260,
        test_size=1,
        step_size=1,
        data_paths=data_paths,
        feature_set_names=["none", "lags_only", "lags_rolling"],
        objective_names=[None, "mse"],
    )

    print(f"Liczba eksperymentów w planie: {len(plan)}")

    for i, cfg in enumerate(plan, start=1):
        print(f"\n=== EKSPERYMENT {i}/{len(plan)} ===")
        print(f"user_id: {cfg.user_id}")
        print(f"spec_id: {cfg.spec_id}")

        predictions, result = execute_experiment(cfg)

        if result["status"] == "SUCCESS":
            pred_path = f"results/predictions/{result['run_id']}__{result['spec_id']}.csv"
            save_predictions(predictions, pred_path)

            print("\n======================\nMetryki:")
            for k, v in result.items():
                print(f"{k}: {v}")
        else:
            print("Eksperyment nie zakończył się sukcesem:")
            print(f"error_type: {result['error_type']}")
            print(f"error_message: {result['error_message']}")

        save_metrics(result, "results/metrics.csv")