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


def _load_metrics(metrics_path: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError(f"Pusty plik metrics: {metrics_path}")
    return df


def _extract_run_seq(run_id: str) -> int:
    try:
        return int(str(run_id).split("_")[-1])
    except Exception:
        return -1


def _keep_latest_run_per_spec(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "run_id" not in out.columns or "spec_id" not in out.columns:
        return out

    out["__run_seq"] = out["run_id"].astype(str).apply(_extract_run_seq)
    out = out.sort_values(["spec_id", "__run_seq"]).drop_duplicates(
        subset=["spec_id"],
        keep="last",
    )
    out = out.drop(columns="__run_seq")
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

    # Benchmarki i modele bez cech nie potrzebują suffixu.
    if model in {"historical_mean", "naive"}:
        return base_label

    if feature_label == "":
        return base_label

    return f"{base_label}_{feature_label}"


def _build_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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


def _order_columns(columns: pd.Index) -> list[str]:
    ordered = [c for c in PREFERRED_MODEL_ORDER if c in columns]
    remaining = [c for c in columns if c not in ordered]
    return ordered + sorted(remaining)


def _plot_heatmap_from_pivot(
    pivot_df: pd.DataFrame,
    title: str,
    output_path: Path,
    fmt: str = ".4f",
) -> None:
    fig_width = max(11, 1.4 * len(pivot_df.columns))
    fig_height = max(6, 0.6 * len(pivot_df.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    values = pivot_df.values
    im = ax.imshow(values, aspect="auto")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=30, ha="right")

    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_heatmap(
    metrics_path: str,
    metric_col: str,
    output_filename: str,
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    keep_latest_only: bool = True,
) -> None:
    df = _load_metrics(metrics_path)
    df = df[df["status"] == "SUCCESS"].copy()

    if keep_latest_only:
        df = _keep_latest_run_per_spec(df)

    df = _build_plot_columns(df)

    if metric_col not in df.columns:
        raise ValueError(f"Brak kolumny metryki: {metric_col}")

    pivot_df = df.pivot_table(
        index="asset_horizon",
        columns="model_label",
        values=metric_col,
        aggfunc="first",
    )

    pivot_df = pivot_df[_order_columns(pivot_df.columns)]

    out_dir = _ensure_output_dir(output_dir)
    plot_title = title or f"Heatmap: {metric_col}"
    output_path = out_dir / output_filename

    _plot_heatmap_from_pivot(
        pivot_df=pivot_df,
        title=plot_title,
        output_path=output_path,
    )


def _select_points_to_annotate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    annotate_mode: str = "top_bottom",
    top_n: int = 5,
    bottom_n: int = 3,
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

    labels = _order_columns(pd.Index(df["model_label"].dropna().unique()))

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


def plot_metric_scatter(
    metrics_path: str,
    x_col: str,
    y_col: str,
    output_filename: str,
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    annotate_mode: str = "top_bottom",
    top_n: int = 5,
    bottom_n: int = 3,
    keep_latest_only: bool = True,
) -> None:
    df = _load_metrics(metrics_path)
    df = df[df["status"] == "SUCCESS"].copy()

    if keep_latest_only:
        df = _keep_latest_run_per_spec(df)

    df = _build_plot_columns(df)

    missing = [c for c in [x_col, y_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn: {missing}")

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        ax.scatter(
            row[x_col],
            row[y_col],
            color=row["asset_color"],
            marker=row["marker"],
            s=80,
            alpha=0.9,
        )

    idx_to_annotate = _select_points_to_annotate(
        df=df,
        x_col=x_col,
        y_col=y_col,
        annotate_mode=annotate_mode,
        top_n=top_n,
        bottom_n=bottom_n,
    )

    for idx, row in df.iterrows():
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
    ax.set_title(title or f"{y_col} vs {x_col}")

    legend1 = ax.legend(
        handles=_build_asset_legend_handles(),
        title="Asset",
        loc="upper left",
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=_build_model_legend_handles(df),
        title="Model",
        loc="lower right",
    )

    fig.tight_layout()

    out_dir = _ensure_output_dir(output_dir)
    fig.savefig(out_dir / output_filename, dpi=150)
    plt.close(fig)


def export_best_models_table(
    metrics_path: str,
    metric_col: str,
    output_csv: str,
    maximize: bool = True,
    keep_latest_only: bool = True,
) -> None:
    df = _load_metrics(metrics_path)
    df = df[df["status"] == "SUCCESS"].copy()

    if keep_latest_only:
        df = _keep_latest_run_per_spec(df)

    if metric_col not in df.columns:
        raise ValueError(f"Brak kolumny metryki: {metric_col}")

    if maximize:
        idx = df.groupby(["asset", "horizon"])[metric_col].idxmax()
    else:
        idx = df.groupby(["asset", "horizon"])[metric_col].idxmin()

    best_df = df.loc[idx].sort_values(["asset", "horizon"]).reset_index(drop=True)
    best_df = _build_plot_columns(best_df)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(output_csv, index=False)