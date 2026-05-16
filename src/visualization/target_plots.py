from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


PLOT_DIR = Path("results/plots")


ASSET_COLOR_MAP = {
    "SPY": "tab:blue",
    "EURUSD": "tab:orange",
    "XAUUSD": "tab:green",
}


MODEL_BASE_LABELS = {
    "historical_mean": "HME",
    "naive": "NAV",
    "ridge": "RID",
    "xgboost": "XGB",
    "arima": "ARI",
    "lstm": "LSTM",
}


FEATURE_LABELS = {
    "none": "",
    "lags_only": "LAG",
    "lags_rolling": "ROLL",
}


MODEL_MARKER_MAP = {
    "HME": "o",
    "NAV": "s",
    "RID_LAG": "P",
    "RID_ROLL": "X",
    "XGB_LAG": "^",
    "XGB_ROLL": "D",
    "ARI": "v",
    "LSTM": "*",
}


PREFERRED_MODEL_ORDER = [
    "HME",
    "NAV",
    "RID_LAG",
    "RID_ROLL",
    "XGB_LAG",
    "XGB_ROLL",
    "ARI",
    "LSTM",
]


def _ensure_output_dir(output_dir: Optional[str] = None) -> Path:
    out_dir = Path(output_dir) if output_dir is not None else PLOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_dataset_stats(stats_path: str) -> pd.DataFrame:
    df = pd.read_csv(stats_path)

    if df.empty:
        raise ValueError(f"Pusty plik dataset stats: {stats_path}")

    required_cols = {
        "asset",
        "horizon",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Brak wymaganych kolumn w dataset stats: {missing_cols}")

    return df


def _load_metrics(metrics_path: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)

    if df.empty:
        raise ValueError(f"Pusty plik metrics: {metrics_path}")

    required_cols = {
        "asset",
        "horizon",
        "model",
        "status",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Brak wymaganych kolumn w metrics: {missing_cols}")

    return df


def _extract_run_seq(run_id: str) -> int:
    try:
        return int(str(run_id).split("_")[-1])
    except Exception:
        return -1


def _keep_latest_run_per_spec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zostawia najnowszy run dla każdego spec_id.

    Priorytet:
    1. finished_at, jeśli istnieje i da się sparsować
    2. run_id jako fallback
    3. bez zmian, jeśli brakuje spec_id
    """
    out = df.copy()

    if "spec_id" not in out.columns:
        return out

    if "finished_at" in out.columns:
        out["__finished_at"] = pd.to_datetime(out["finished_at"], errors="coerce")

        if out["__finished_at"].notna().any():
            out = (
                out.sort_values(["spec_id", "__finished_at"])
                .drop_duplicates(subset=["spec_id"], keep="last")
                .drop(columns="__finished_at")
            )
            return out

        out = out.drop(columns="__finished_at")

    if "run_id" in out.columns:
        out["__run_seq"] = out["run_id"].astype(str).apply(_extract_run_seq)
        out = (
            out.sort_values(["spec_id", "__run_seq"])
            .drop_duplicates(subset=["spec_id"], keep="last")
            .drop(columns="__run_seq")
        )
        return out

    return out


def _short_asset_name(asset: str) -> str:
    if asset == "XAUUSD":
        return "XAU"
    return str(asset)


def _normalize_feature_set(feature_set) -> str:
    if pd.isna(feature_set) or feature_set is None or feature_set == "":
        return "none"
    return str(feature_set)


def _short_model_label(row: pd.Series) -> str:
    model = str(row["model"])
    feature_set = _normalize_feature_set(row.get("feature_set_name", "none"))

    base_label = MODEL_BASE_LABELS.get(model, model.upper())
    feature_label = FEATURE_LABELS.get(feature_set, feature_set.upper())

    if model in {"historical_mean", "naive"}:
        return base_label

    if feature_label == "":
        return base_label

    return f"{base_label}_{feature_label}"


def _build_asset_horizon_label(row: pd.Series) -> str:
    return f"{_short_asset_name(str(row['asset']))}_H{int(row['horizon'])}"


def _build_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "feature_set_name" not in out.columns:
        out["feature_set_name"] = "none"

    out["feature_set_name"] = out["feature_set_name"].apply(_normalize_feature_set)
    out["model_label"] = out.apply(_short_model_label, axis=1)

    out["asset_horizon"] = (
        out["asset"].astype(str)
        + "_H"
        + out["horizon"].astype(str)
    )

    out["point_label"] = (
        out["asset"].astype(str).map(_short_asset_name)
        + "-H"
        + out["horizon"].astype(str)
        + " "
        + out["model_label"]
    )

    out["asset_color"] = out["asset"].map(ASSET_COLOR_MAP).fillna("tab:gray")
    out["marker"] = out["model_label"].map(MODEL_MARKER_MAP).fillna("o")

    return out


def _order_model_labels(labels) -> list[str]:
    labels = list(labels)
    ordered = [label for label in PREFERRED_MODEL_ORDER if label in labels]
    remaining = sorted([label for label in labels if label not in ordered])
    return ordered + remaining


def _select_points_to_annotate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    annotate_mode: str = "top_bottom",
    top_n: int = 6,
    bottom_n: int = 4,
) -> set[int]:
    if annotate_mode == "none":
        return set()

    if annotate_mode == "all":
        return set(df.index)

    if annotate_mode == "top_bottom":
        top_idx = df.nlargest(top_n, y_col).index
        bottom_idx = df.nsmallest(bottom_n, y_col).index
        return set(top_idx).union(set(bottom_idx))

    if annotate_mode == "top_y":
        return set(df.nlargest(top_n, y_col).index)

    if annotate_mode == "bottom_y":
        return set(df.nsmallest(bottom_n, y_col).index)

    if annotate_mode == "extremes_xy":
        idx = set()
        idx.update(df.nlargest(top_n, y_col).index)
        idx.update(df.nsmallest(bottom_n, y_col).index)
        idx.update(df.nlargest(top_n, x_col).index)
        idx.update(df.nsmallest(bottom_n, x_col).index)
        return idx

    raise ValueError(f"Nieznany annotate_mode: {annotate_mode}")


def _build_asset_legend_handles() -> list:
    handles = []

    for asset, color in ASSET_COLOR_MAP.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label=asset,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=8,
            )
        )

    return handles


def _build_model_legend_handles(df: pd.DataFrame) -> list:
    handles = []

    labels = _order_model_labels(df["model_label"].dropna().unique())

    for label in labels:
        marker = MODEL_MARKER_MAP.get(label, "o")
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="",
                color="black",
                label=label,
                markersize=8,
            )
        )

    return handles


def plot_target_stat_scatter_for_historical_mean(
    stats_path: str,
    metrics_path: str,
    x_col: str,
    y_col: str,
    output_filename: str,
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    keep_latest_only: bool = True,
) -> None:
    stats_df = _load_dataset_stats(stats_path)
    metrics_df = _load_metrics(metrics_path)

    metrics_df = metrics_df[
        (metrics_df["status"] == "SUCCESS")
        & (metrics_df["model"] == "historical_mean")
    ].copy()

    if keep_latest_only:
        metrics_df = _keep_latest_run_per_spec(metrics_df)

    merged = pd.merge(
        stats_df,
        metrics_df,
        on=["asset", "horizon"],
        how="inner",
        suffixes=("_stats", "_metrics"),
    )

    if merged.empty:
        raise ValueError(
            "Brak wspólnych rekordów po połączeniu dataset_stats i metrics "
            "dla modelu historical_mean."
        )

    if x_col not in merged.columns:
        raise ValueError(f"Brak kolumny x: {x_col}")

    if y_col not in merged.columns:
        raise ValueError(f"Brak kolumny y: {y_col}")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        merged[x_col],
        merged[y_col],
        s=80,
        alpha=0.9,
    )

    for _, row in merged.iterrows():
        label = _build_asset_horizon_label(row)
        ax.annotate(
            label,
            (row[x_col], row[y_col]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"Historical Mean: {y_col} vs {x_col}")

    fig.tight_layout()

    out_dir = _ensure_output_dir(output_dir)
    fig.savefig(out_dir / output_filename, dpi=150)
    plt.close(fig)


def plot_target_stat_scatter_all_models(
    stats_path: str,
    metrics_path: str,
    x_col: str,
    y_col: str,
    output_filename: str,
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    annotate_mode: str = "top_bottom",
    top_n: int = 6,
    bottom_n: int = 4,
    keep_latest_only: bool = True,
) -> None:
    stats_df = _load_dataset_stats(stats_path)
    metrics_df = _load_metrics(metrics_path)

    metrics_df = metrics_df[metrics_df["status"] == "SUCCESS"].copy()

    if keep_latest_only:
        metrics_df = _keep_latest_run_per_spec(metrics_df)

    merged = pd.merge(
        stats_df,
        metrics_df,
        on=["asset", "horizon"],
        how="inner",
        suffixes=("_stats", "_metrics"),
    )

    if merged.empty:
        raise ValueError(
            "Brak wspólnych rekordów po połączeniu dataset_stats i metrics."
        )

    merged = _build_plot_columns(merged)

    if x_col not in merged.columns:
        raise ValueError(f"Brak kolumny x: {x_col}")

    if y_col not in merged.columns:
        raise ValueError(f"Brak kolumny y: {y_col}")

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in merged.iterrows():
        ax.scatter(
            row[x_col],
            row[y_col],
            color=row["asset_color"],
            marker=row["marker"],
            s=80,
            alpha=0.9,
        )

    idx_to_annotate = _select_points_to_annotate(
        df=merged,
        x_col=x_col,
        y_col=y_col,
        annotate_mode=annotate_mode,
        top_n=top_n,
        bottom_n=bottom_n,
    )

    for idx, row in merged.iterrows():
        if idx in idx_to_annotate:
            ax.annotate(
                row["point_label"],
                (row[x_col], row[y_col]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"All models: {y_col} vs {x_col}")

    legend1 = ax.legend(
        handles=_build_asset_legend_handles(),
        title="Asset",
        loc="upper left",
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=_build_model_legend_handles(merged),
        title="Model",
        loc="lower right",
    )

    fig.tight_layout()

    out_dir = _ensure_output_dir(output_dir)
    fig.savefig(out_dir / output_filename, dpi=150)
    plt.close(fig)


def plot_target_histograms(
    stats_source_csv: str,
    raw_target_csv_dir: Optional[str] = None,
) -> None:
    """
    Placeholder pod przyszłe rozszerzenie.

    Docelowo ten wykres powinien raczej czytać pełne targety per asset/horizon,
    a nie dataset_stats.csv, bo histogram wymaga rozkładu obserwacji, nie samych
    agregatów typu mean/std/quantile.
    """
    _ = stats_source_csv
    _ = raw_target_csv_dir