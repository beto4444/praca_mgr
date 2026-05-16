from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd


METRIC_COLUMNS = [
    "mse",
    "directional_accuracy",
    "mean_synthetic_return",
    "pseudo_sharpe",
]

MAXIMIZE_METRIC = {
    "mse": False,
    "directional_accuracy": True,
    "mean_synthetic_return": True,
    "pseudo_sharpe": True,
}

MODEL_LABEL_MAP = {
    "naive": "NAV",
    "historical_mean": "HME",
    "ridge": "RID",
    "xgboost": "XGB",
    "arima": "ARI",
    "lstm": "LSTM",
}


def _ensure_output_dir(output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_metrics(metrics_path: str | Path) -> pd.DataFrame:
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
    """
    Zostawia najnowsze wykonanie dla każdego spec_id.

    Priorytet:
    1. finished_at, jeśli istnieje i daje się sparsować,
    2. fallback do końcówki run_id.
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


def _prepare_success_metrics(
    metrics_path: str | Path,
    keep_latest_only: bool = True,
) -> pd.DataFrame:
    df = _load_metrics(metrics_path)

    if "status" in df.columns:
        df = df[df["status"] == "SUCCESS"].copy()

    if keep_latest_only:
        df = _keep_latest_run_per_spec(df)

    if df.empty:
        raise ValueError("Brak poprawnych eksperymentów SUCCESS po filtrowaniu.")

    df = _add_model_labels(df)
    return df


def _safe_str_value(value: object, empty_value: str = "NA") -> str:
    if pd.isna(value) or value == "":
        return empty_value
    return str(value)


def _join_unique(values: pd.Series) -> str:
    clean_values = sorted({_safe_str_value(v) for v in values.dropna().tolist()})
    return ";".join(clean_values)


def _add_model_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def label_row(row: pd.Series) -> str:
        model = str(row.get("model", "unknown"))
        feature_set = _safe_str_value(row.get("feature_set_name", "NA"))

        base = MODEL_LABEL_MAP.get(model, model.upper())

        if model in {"ridge", "xgboost"}:
            if feature_set == "lags_only":
                return f"{base}_LAG"
            if feature_set == "lags_rolling":
                return f"{base}_ROLL"
            return base

        return base

    out["model_label"] = out.apply(label_row, axis=1)

    return out


def export_experiment_scope_table(
    metrics_path: str | Path,
    output_dir: str | Path,
    output_filename: str = "summary_experiment_scope.csv",
    keep_latest_only: bool = True,
) -> Path:
    """
    Eksportuje zakres wykonanych eksperymentów.

    Przydatne do rozdziału:
    6.1. Zakres przeprowadzonych eksperymentów.
    """
    df = _load_metrics(metrics_path)

    if keep_latest_only:
        df = _keep_latest_run_per_spec(df)

    required_cols = [
        "model",
        "feature_set_name",
        "objective_name",
        "asset",
        "horizon",
        "status",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn w metrics.csv: {missing}")

    grouped = (
        df.groupby(["model", "feature_set_name", "objective_name"], dropna=False)
        .agg(
            n_experiments=("run_id", "count"),
            n_success=("status", lambda x: int((x == "SUCCESS").sum())),
            n_failed=("status", lambda x: int((x != "SUCCESS").sum())),
            assets=("asset", _join_unique),
            horizons=("horizon", lambda x: _join_unique(x.astype(str))),
            spec_ids=("spec_id", "nunique"),
        )
        .reset_index()
    )

    grouped["objective_name"] = grouped["objective_name"].fillna("NA")
    grouped["feature_set_name"] = grouped["feature_set_name"].fillna("NA")

    grouped = grouped.sort_values(
        ["model", "feature_set_name", "objective_name"]
    ).reset_index(drop=True)

    out_dir = _ensure_output_dir(output_dir)
    output_path = out_dir / output_filename
    grouped.to_csv(output_path, index=False)

    return output_path


def export_best_by_all_metrics_table(
    metrics_path: str | Path,
    output_dir: str | Path,
    output_filename: str = "summary_best_by_all_metrics.csv",
    keep_latest_only: bool = True,
) -> Path:
    """
    Eksportuje jedną tabelę:
    asset + horizon -> najlepszy model według MSE, DA, MSR, PS.

    Przydatne do:
    6.8. Najlepsze modele dla poszczególnych aktywów i horyzontów.
    """
    df = _prepare_success_metrics(
        metrics_path=metrics_path,
        keep_latest_only=keep_latest_only,
    )

    required_cols = ["asset", "horizon"] + METRIC_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn: {missing}")

    rows: list[dict] = []

    for (asset, horizon), group in df.groupby(["asset", "horizon"]):
        row: dict = {
            "asset": asset,
            "horizon": horizon,
        }

        for metric in METRIC_COLUMNS:
            group_metric = group.dropna(subset=[metric]).copy()

            if group_metric.empty:
                row[f"best_{metric}_model_label"] = None
                row[f"best_{metric}_value"] = None
                continue

            if MAXIMIZE_METRIC[metric]:
                best_idx = group_metric[metric].idxmax()
            else:
                best_idx = group_metric[metric].idxmin()

            best = group_metric.loc[best_idx]

            row[f"best_{metric}_model_label"] = best["model_label"]
            row[f"best_{metric}_model"] = best["model"]
            row[f"best_{metric}_feature_set"] = _safe_str_value(
                best.get("feature_set_name")
            )
            row[f"best_{metric}_objective"] = _safe_str_value(
                best.get("objective_name")
            )
            row[f"best_{metric}_value"] = best[metric]
            row[f"best_{metric}_spec_id"] = best.get("spec_id")
            row[f"best_{metric}_run_id"] = best.get("run_id")

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["asset", "horizon"]).reset_index(drop=True)

    out_dir = _ensure_output_dir(output_dir)
    output_path = out_dir / output_filename
    out.to_csv(output_path, index=False)

    return output_path


def export_best_models_long_table(
    metrics_path: str | Path,
    output_dir: str | Path,
    output_filename: str = "summary_best_models_long.csv",
    keep_latest_only: bool = True,
) -> Path:
    """
    Eksportuje tabelę w formacie long:
    asset, horizon, metric -> najlepszy model.

    Ta forma jest wygodniejsza do dalszej analizy albo LaTeX pivotów.
    """
    df = _prepare_success_metrics(
        metrics_path=metrics_path,
        keep_latest_only=keep_latest_only,
    )

    rows: list[dict] = []

    for (asset, horizon), group in df.groupby(["asset", "horizon"]):
        for metric in METRIC_COLUMNS:
            group_metric = group.dropna(subset=[metric]).copy()

            if group_metric.empty:
                continue

            if MAXIMIZE_METRIC[metric]:
                best_idx = group_metric[metric].idxmax()
            else:
                best_idx = group_metric[metric].idxmin()

            best = group_metric.loc[best_idx]

            rows.append(
                {
                    "asset": asset,
                    "horizon": horizon,
                    "metric": metric,
                    "maximize": MAXIMIZE_METRIC[metric],
                    "best_model_label": best["model_label"],
                    "best_model": best["model"],
                    "best_feature_set": _safe_str_value(best.get("feature_set_name")),
                    "best_objective": _safe_str_value(best.get("objective_name")),
                    "best_value": best[metric],
                    "mse": best.get("mse"),
                    "directional_accuracy": best.get("directional_accuracy"),
                    "mean_synthetic_return": best.get("mean_synthetic_return"),
                    "pseudo_sharpe": best.get("pseudo_sharpe"),
                    "spec_id": best.get("spec_id"),
                    "run_id": best.get("run_id"),
                }
            )

    out = pd.DataFrame(rows).sort_values(["asset", "horizon", "metric"])

    out_dir = _ensure_output_dir(output_dir)
    output_path = out_dir / output_filename
    out.to_csv(output_path, index=False)

    return output_path


def export_metric_correlations(
    metrics_path: str | Path,
    output_dir: str | Path,
    output_filename: str = "summary_metric_correlations.csv",
    keep_latest_only: bool = True,
) -> Path:
    """
    Eksportuje korelacje Pearson/Spearman między metrykami.
    """
    df = _prepare_success_metrics(
        metrics_path=metrics_path,
        keep_latest_only=keep_latest_only,
    )

    missing = [c for c in METRIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Brak metryk do korelacji: {missing}")

    rows: list[dict] = []

    for metric_x, metric_y in combinations(METRIC_COLUMNS, 2):
        sub = df[[metric_x, metric_y]].dropna().copy()

        if len(sub) < 2:
            pearson = None
            spearman = None
        else:
            pearson = sub[metric_x].corr(sub[metric_y], method="pearson")
            spearman = sub[metric_x].corr(sub[metric_y], method="spearman")

        rows.append(
            {
                "metric_x": metric_x,
                "metric_y": metric_y,
                "pearson": pearson,
                "spearman": spearman,
                "n": int(len(sub)),
            }
        )

    out = pd.DataFrame(rows)

    out_dir = _ensure_output_dir(output_dir)
    output_path = out_dir / output_filename
    out.to_csv(output_path, index=False)

    return output_path


def export_model_metric_summary(
    metrics_path: str | Path,
    output_dir: str | Path,
    output_filename: str = "summary_model_metric_summary.csv",
    keep_latest_only: bool = True,
) -> Path:
    """
    Eksportuje średnie i mediany metryk po modelach / feature setach.
    """
    df = _prepare_success_metrics(
        metrics_path=metrics_path,
        keep_latest_only=keep_latest_only,
    )

    required_cols = ["model_label", "model", "feature_set_name", "objective_name"]
    missing = [c for c in required_cols + METRIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn: {missing}")

    grouped = (
        df.groupby(
            ["model_label", "model", "feature_set_name", "objective_name"],
            dropna=False,
        )
        .agg(
            n=("run_id", "count"),
            mse_mean=("mse", "mean"),
            mse_median=("mse", "median"),
            directional_accuracy_mean=("directional_accuracy", "mean"),
            directional_accuracy_median=("directional_accuracy", "median"),
            mean_synthetic_return_mean=("mean_synthetic_return", "mean"),
            mean_synthetic_return_median=("mean_synthetic_return", "median"),
            pseudo_sharpe_mean=("pseudo_sharpe", "mean"),
            pseudo_sharpe_median=("pseudo_sharpe", "median"),
        )
        .reset_index()
    )

    grouped["objective_name"] = grouped["objective_name"].fillna("NA")
    grouped["feature_set_name"] = grouped["feature_set_name"].fillna("NA")

    grouped = grouped.sort_values(["model_label"]).reset_index(drop=True)

    out_dir = _ensure_output_dir(output_dir)
    output_path = out_dir / output_filename
    grouped.to_csv(output_path, index=False)

    return output_path


def export_best_counts_by_metric(
    metrics_path: str | Path,
    output_dir: str | Path,
    output_filename: str = "summary_best_counts_by_metric.csv",
    keep_latest_only: bool = True,
) -> Path:
    """
    Liczy, ile razy dany model_label wygrał według każdej metryki
    w ramach grup asset+horizon.
    """
    best_long_path = export_best_models_long_table(
        metrics_path=metrics_path,
        output_dir=output_dir,
        output_filename="summary_best_models_long.csv",
        keep_latest_only=keep_latest_only,
    )

    best_long = pd.read_csv(best_long_path)

    grouped = (
        best_long.groupby(["metric", "best_model_label"])
        .size()
        .reset_index(name="n_wins")
        .sort_values(["metric", "n_wins"], ascending=[True, False])
        .reset_index(drop=True)
    )

    out_dir = _ensure_output_dir(output_dir)
    output_path = out_dir / output_filename
    grouped.to_csv(output_path, index=False)

    return output_path


def export_dataset_stats_for_chapter(
    dataset_stats_path: str | Path,
    output_dir: str | Path,
    output_filename: str = "summary_dataset_stats_for_chapter.csv",
) -> Path:
    """
    Eksportuje skróconą wersję dataset_stats do tabeli w rozdziale 6.2.
    """
    df = pd.read_csv(dataset_stats_path)

    if df.empty:
        raise ValueError(f"Pusty plik dataset_stats: {dataset_stats_path}")

    keep_cols = [
        "asset",
        "horizon",
        "n_obs",
        "date_min",
        "date_max",
        "mean_y_true",
        "std_y_true",
        "pct_positive",
        "pct_negative",
        "median",
    ]

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn w dataset_stats: {missing}")

    out = df[keep_cols].copy()
    out = out.sort_values(["asset", "horizon"]).reset_index(drop=True)

    out_dir = _ensure_output_dir(output_dir)
    output_path = out_dir / output_filename
    out.to_csv(output_path, index=False)

    return output_path


def export_all_summary_tables(
    metrics_path: str | Path,
    output_dir: str | Path,
    dataset_stats_path: Optional[str | Path] = None,
    keep_latest_only: bool = True,
) -> list[Path]:
    """
    Wygodny wrapper generujący wszystkie tabele podsumowujące.
    """
    generated_paths: list[Path] = []

    generated_paths.append(
        export_experiment_scope_table(
            metrics_path=metrics_path,
            output_dir=output_dir,
            keep_latest_only=keep_latest_only,
        )
    )

    generated_paths.append(
        export_best_by_all_metrics_table(
            metrics_path=metrics_path,
            output_dir=output_dir,
            keep_latest_only=keep_latest_only,
        )
    )

    generated_paths.append(
        export_best_models_long_table(
            metrics_path=metrics_path,
            output_dir=output_dir,
            keep_latest_only=keep_latest_only,
        )
    )

    generated_paths.append(
        export_metric_correlations(
            metrics_path=metrics_path,
            output_dir=output_dir,
            keep_latest_only=keep_latest_only,
        )
    )

    generated_paths.append(
        export_model_metric_summary(
            metrics_path=metrics_path,
            output_dir=output_dir,
            keep_latest_only=keep_latest_only,
        )
    )

    generated_paths.append(
        export_best_counts_by_metric(
            metrics_path=metrics_path,
            output_dir=output_dir,
            keep_latest_only=keep_latest_only,
        )
    )

    if dataset_stats_path is not None:
        generated_paths.append(
            export_dataset_stats_for_chapter(
                dataset_stats_path=dataset_stats_path,
                output_dir=output_dir,
            )
        )

    return generated_paths