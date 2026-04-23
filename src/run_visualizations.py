from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import pandas as pd

from src.visualization.metrics_plots import (
    plot_metric_heatmap,
    plot_metric_scatter,
    export_best_models_table,
)
from src.visualization.target_plots import (
    plot_target_stat_scatter_for_historical_mean,
)


def build_visualization_run_dir(base_dir: str = "results/plots") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_metadata(
    metrics_path: str,
    dataset_stats_path: str,
    output_dir: Path,
) -> dict:
    metadata: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "metrics_path": metrics_path,
        "dataset_stats_path": dataset_stats_path,
    }

    # Metadata z metrics.csv
    metrics_df = pd.read_csv(metrics_path)
    metadata["metrics_total_rows"] = int(len(metrics_df))

    if "status" in metrics_df.columns:
        success_df = metrics_df[metrics_df["status"] == "SUCCESS"].copy()
        metadata["metrics_success_rows"] = int(len(success_df))
    else:
        success_df = metrics_df.copy()
        metadata["metrics_success_rows"] = None

    if "spec_id" in metrics_df.columns:
        metadata["metrics_unique_spec_ids"] = int(metrics_df["spec_id"].nunique())
    else:
        metadata["metrics_unique_spec_ids"] = None

    if "run_id" in metrics_df.columns:
        metadata["metrics_unique_run_ids"] = int(metrics_df["run_id"].nunique())
        metadata["latest_run_id"] = str(metrics_df["run_id"].iloc[-1]) if len(metrics_df) > 0 else None
    else:
        metadata["metrics_unique_run_ids"] = None
        metadata["latest_run_id"] = None

    if "asset" in success_df.columns:
        metadata["assets_in_success_metrics"] = sorted(success_df["asset"].dropna().astype(str).unique().tolist())
    else:
        metadata["assets_in_success_metrics"] = []

    if "horizon" in success_df.columns:
        metadata["horizons_in_success_metrics"] = sorted(success_df["horizon"].dropna().astype(int).unique().tolist())
    else:
        metadata["horizons_in_success_metrics"] = []

    if "model" in success_df.columns:
        metadata["models_in_success_metrics"] = sorted(success_df["model"].dropna().astype(str).unique().tolist())
    else:
        metadata["models_in_success_metrics"] = []

    # Metadata z dataset_stats.csv
    stats_df = pd.read_csv(dataset_stats_path)
    metadata["dataset_stats_rows"] = int(len(stats_df))

    if "asset" in stats_df.columns:
        metadata["assets_in_dataset_stats"] = sorted(stats_df["asset"].dropna().astype(str).unique().tolist())
    else:
        metadata["assets_in_dataset_stats"] = []

    if "horizon" in stats_df.columns:
        metadata["horizons_in_dataset_stats"] = sorted(stats_df["horizon"].dropna().astype(int).unique().tolist())
    else:
        metadata["horizons_in_dataset_stats"] = []

    return metadata


def save_metadata(metadata: dict, output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    metrics_path = "results/metrics.csv"
    dataset_stats_path = "results/dataset_stats.csv"

    output_dir = build_visualization_run_dir()

    metadata = build_metadata(
        metrics_path=metrics_path,
        dataset_stats_path=dataset_stats_path,
        output_dir=output_dir,
    )
    save_metadata(metadata, output_dir)

    # Heatmapy głównych metryk
    plot_metric_heatmap(
        metrics_path=metrics_path,
        metric_col="pseudo_sharpe",
        output_filename="heatmap_pseudo_sharpe.png",
        title="Pseudo-Sharpe by asset/horizon/model",
        output_dir=str(output_dir),
    )

    plot_metric_heatmap(
        metrics_path=metrics_path,
        metric_col="mean_synthetic_return",
        output_filename="heatmap_mean_synthetic_return.png",
        title="Mean Synthetic Return by asset/horizon/model",
        output_dir=str(output_dir),
    )

    plot_metric_heatmap(
        metrics_path=metrics_path,
        metric_col="directional_accuracy",
        output_filename="heatmap_directional_accuracy.png",
        title="Directional Accuracy by asset/horizon/model",
        output_dir=str(output_dir),
    )

    plot_metric_heatmap(
        metrics_path=metrics_path,
        metric_col="mse",
        output_filename="heatmap_mse.png",
        title="MSE by asset/horizon/model",
        output_dir=str(output_dir),
    )

    # Scattery relacji metryk
    plot_metric_scatter(
        metrics_path=metrics_path,
        x_col="directional_accuracy",
        y_col="mean_synthetic_return",
        output_filename="scatter_da_vs_mean_synthetic_return.png",
        title="Mean Synthetic Return vs Directional Accuracy",
        output_dir=str(output_dir),
        annotate_mode="top_bottom",
        top_n=6,
        bottom_n=4,
    )

    plot_metric_scatter(
        metrics_path=metrics_path,
        x_col="mse",
        y_col="mean_synthetic_return",
        output_filename="scatter_mse_vs_mean_synthetic_return.png",
        title="Mean Synthetic Return vs MSE",
        output_dir=str(output_dir),
        annotate_mode="top_bottom",
        top_n=6,
        bottom_n=4,
    )

    plot_metric_scatter(
        metrics_path=metrics_path,
        x_col="pseudo_sharpe",
        y_col="mean_synthetic_return",
        output_filename="scatter_ps_vs_mean_synthetic_return.png",
        title="Mean Synthetic Return vs Pseudo-Sharpe",
        output_dir=str(output_dir),
        annotate_mode="top_y",
        top_n=8,
        bottom_n=0,
    )

    # Powiązanie stats targetu z wynikiem historical_mean
    plot_target_stat_scatter_for_historical_mean(
        stats_path=dataset_stats_path,
        metrics_path=metrics_path,
        x_col="mean_y_true",
        y_col="pseudo_sharpe",
        output_filename="scatter_mean_y_true_vs_hme_pseudo_sharpe.png",
        title="HistoricalMean: pseudo_sharpe vs mean_y_true",
        output_dir=str(output_dir),
    )

    plot_target_stat_scatter_for_historical_mean(
        stats_path=dataset_stats_path,
        metrics_path=metrics_path,
        x_col="pct_positive",
        y_col="directional_accuracy",
        output_filename="scatter_pct_positive_vs_hme_da.png",
        title="HistoricalMean: directional_accuracy vs pct_positive",
        output_dir=str(output_dir),
    )

    # Tabelki najlepszych modeli
    export_best_models_table(
        metrics_path=metrics_path,
        metric_col="pseudo_sharpe",
        output_csv=str(output_dir / "best_by_pseudo_sharpe.csv"),
        maximize=True,
    )

    export_best_models_table(
        metrics_path=metrics_path,
        metric_col="mean_synthetic_return",
        output_csv=str(output_dir / "best_by_mean_synthetic_return.csv"),
        maximize=True,
    )

    export_best_models_table(
        metrics_path=metrics_path,
        metric_col="directional_accuracy",
        output_csv=str(output_dir / "best_by_directional_accuracy.csv"),
        maximize=True,
    )

    export_best_models_table(
        metrics_path=metrics_path,
        metric_col="mse",
        output_csv=str(output_dir / "best_by_mse.csv"),
        maximize=False,
    )

    print(f"Wizualizacje i tabelki zapisane do: {output_dir}")
    print(f"Metadata zapisane do: {output_dir / 'metadata.json'}")