from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


PLOT_DIR = Path("results/plots")


def _ensure_output_dir(output_dir: Optional[str] = None) -> Path:
    out_dir = Path(output_dir) if output_dir is not None else PLOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_dataset_stats(stats_path: str) -> pd.DataFrame:
    df = pd.read_csv(stats_path)
    if df.empty:
        raise ValueError(f"Pusty plik dataset stats: {stats_path}")
    return df


def _load_metrics(metrics_path: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError(f"Pusty plik metrics: {metrics_path}")
    return df


def plot_target_stat_scatter_for_historical_mean(
    stats_path: str,
    metrics_path: str,
    x_col: str,
    y_col: str,
    output_filename: str,
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    stats_df = _load_dataset_stats(stats_path)
    metrics_df = _load_metrics(metrics_path)

    metrics_df = metrics_df[
        (metrics_df["status"] == "SUCCESS") &
        (metrics_df["model"] == "historical_mean")
    ].copy()

    merged = pd.merge(
        stats_df,
        metrics_df,
        on=["asset", "horizon"],
        how="inner",
        suffixes=("_stats", "_metrics"),
    )

    if x_col not in merged.columns:
        raise ValueError(f"Brak kolumny x: {x_col}")
    if y_col not in merged.columns:
        raise ValueError(f"Brak kolumny y: {y_col}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged[x_col], merged[y_col])

    for _, row in merged.iterrows():
        label = f"{row['asset']}_H{row['horizon']}"
        ax.annotate(
            label,
            (row[x_col], row[y_col]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} vs {x_col} for historical_mean")
    fig.tight_layout()

    out_dir = _ensure_output_dir(output_dir)
    fig.savefig(out_dir / output_filename, dpi=150)
    plt.close(fig)


def plot_target_histograms(
    stats_source_csv: str,
    raw_target_csv_dir: Optional[str] = None,
) -> None:
    # Placeholder pod przyszłe rozszerzenie, jeśli będziesz chciał
    # renderować histogramy z pełnych targetów per asset/horizon.
    # Na razie zostawiamy pusty stub, bo masz już dobre stats CSV.
    _ = stats_source_csv
    _ = raw_target_csv_dir