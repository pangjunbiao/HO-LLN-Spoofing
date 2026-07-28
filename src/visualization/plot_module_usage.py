"""
Visualization utilities for Step-14 full-model module-usage diagnostics.

Step 14 purpose:
- Plot diagnostic summaries from:
  - module_usage.py
  - feature_importance.py
  - conductance_analysis.py
  - third_order_analysis.py
  - liquid_state_analysis.py
  - occlusion_tests.py

Input CSVs are expected under:
    results/figures/module_usage/

Output figures are saved under:
    results/figures/module_usage/plots/

Important:
- These figures are diagnostic only.
- Occlusion plots are not official ablation results.
- Official ablations must still be retrained from scratch later.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diagnostics.module_usage import (
    Step14DiagnosticsPaths,
    build_step14_diagnostics_config,
    build_step14_paths,
    save_json_safe,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir


@dataclass
class PlotArtifact:
    """One generated Step-14 plot artifact."""

    name: str
    source_csv: str
    output_path: str
    status: str
    rows_used: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleUsagePlotConfig:
    """Plotting configuration for Step-14 module-usage figures."""

    output_dir: str = "results/figures/module_usage"
    plots_subdir: str = "plots"

    top_k: int = 20
    dpi: int = 180
    figure_width: float = 12.0
    figure_height: float = 7.0

    save_png: bool = True
    save_pdf: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _project_path(config: Mapping[str, Any], value: str) -> Path:
    """Resolve a project-relative path."""
    return resolve_project_path(config, value)


def build_module_usage_plot_config(config: Mapping[str, Any]) -> ModuleUsagePlotConfig:
    """Build plotting config from project config."""
    return ModuleUsagePlotConfig(
        output_dir=str(
            get_by_path(
                config,
                "paths.module_usage_figures_dir",
                get_by_path(config, "experiments.step14.output_dir", "results/figures/module_usage"),
            )
        ),
        plots_subdir=str(
            get_by_path(
                config,
                "experiments.step14.plots.subdir",
                "plots",
            )
        ),
        top_k=int(
            get_by_path(
                config,
                "experiments.step14.plots.top_k",
                20,
            )
        ),
        dpi=int(
            get_by_path(
                config,
                "experiments.step14.plots.dpi",
                180,
            )
        ),
        figure_width=float(
            get_by_path(
                config,
                "experiments.step14.plots.figure_width",
                12.0,
            )
        ),
        figure_height=float(
            get_by_path(
                config,
                "experiments.step14.plots.figure_height",
                7.0,
            )
        ),
        save_png=bool(
            get_by_path(
                config,
                "experiments.step14.plots.save_png",
                True,
            )
        ),
        save_pdf=bool(
            get_by_path(
                config,
                "experiments.step14.plots.save_pdf",
                True,
            )
        ),
    )


def get_plot_output_dir(
    config: Mapping[str, Any],
    plot_config: Optional[ModuleUsagePlotConfig] = None,
) -> Path:
    """Return Step-14 plot output directory."""
    if plot_config is None:
        plot_config = build_module_usage_plot_config(config)

    output_dir = _project_path(config, plot_config.output_dir)
    plot_dir = output_dir / plot_config.plots_subdir
    ensure_dir(plot_dir)

    return plot_dir


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Read CSV if it exists and is non-empty."""
    if not path.exists():
        return None

    try:
        if path.stat().st_size == 0:
            return None

        df = pd.read_csv(path)

    except Exception:
        return None

    if df.empty:
        return None

    return df


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric safely."""
    return pd.to_numeric(series, errors="coerce")


def _shorten_label(label: Any, max_len: int = 52) -> str:
    """Shorten long plot labels."""
    text = str(label)

    if len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."


def _clean_metric_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return copy with numeric metric column and finite rows."""
    if column not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out[column] = _safe_numeric(out[column])
    out = out[np.isfinite(out[column].to_numpy(dtype=float, na_value=np.nan))]
    return out


def _top_abs_rows(
    df: pd.DataFrame,
    value_column: str,
    top_k: int,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return top rows by absolute metric magnitude."""
    if df.empty or value_column not in df.columns:
        return pd.DataFrame()

    out = _clean_metric_column(df, value_column)

    if out.empty:
        return out

    out["_abs_metric_for_sort"] = out[value_column].abs()
    out = out.sort_values("_abs_metric_for_sort", ascending=ascending)
    out = out.head(int(top_k)).copy()
    out = out.drop(columns=["_abs_metric_for_sort"])

    return out


def _top_value_rows(
    df: pd.DataFrame,
    value_column: str,
    top_k: int,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return top rows by signed metric value."""
    if df.empty or value_column not in df.columns:
        return pd.DataFrame()

    out = _clean_metric_column(df, value_column)

    if out.empty:
        return out

    out = out.sort_values(value_column, ascending=ascending)
    return out.head(int(top_k)).copy()


def _save_current_figure(
    output_base: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[str]:
    """Save current matplotlib figure as PNG/PDF."""
    output_paths: List[str] = []
    ensure_dir(output_base.parent)

    if plot_config.save_png:
        png_path = output_base.with_suffix(".png")
        plt.savefig(png_path, dpi=int(plot_config.dpi), bbox_inches="tight")
        output_paths.append(str(png_path))

    if plot_config.save_pdf:
        pdf_path = output_base.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        output_paths.append(str(pdf_path))

    plt.close()

    return output_paths


def _barh_plot(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    xlabel: str,
    output_base: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[str]:
    """Create one horizontal bar chart."""
    if df.empty:
        return []

    labels = [_shorten_label(item) for item in df[label_column].tolist()]
    values = _safe_numeric(df[value_column]).to_numpy(dtype=float)

    y_pos = np.arange(len(labels))

    plt.figure(figsize=(plot_config.figure_width, plot_config.figure_height))
    plt.barh(y_pos, values)
    plt.yticks(y_pos, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.axvline(0.0, linewidth=1.0)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return _save_current_figure(output_base, plot_config)


def _scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_base: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[str]:
    """Create one scatter plot."""
    if df.empty:
        return []

    if x_column not in df.columns or y_column not in df.columns:
        return []

    out = df.copy()
    out[x_column] = _safe_numeric(out[x_column])
    out[y_column] = _safe_numeric(out[y_column])
    out = out.dropna(subset=[x_column, y_column])

    if out.empty:
        return []

    plt.figure(figsize=(plot_config.figure_width, plot_config.figure_height))
    plt.scatter(out[x_column].to_numpy(dtype=float), out[y_column].to_numpy(dtype=float))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(0.0, linewidth=1.0)
    plt.axvline(0.0, linewidth=1.0)
    plt.tight_layout()

    return _save_current_figure(output_base, plot_config)


def _make_artifact(
    name: str,
    source_csv: Path,
    output_paths: Sequence[str],
    rows_used: int,
    message: str = "",
) -> PlotArtifact:
    """Create PlotArtifact."""
    if output_paths:
        status = "PASSED"
        output_path = "; ".join(output_paths)
    else:
        status = "SKIPPED"
        output_path = ""

    return PlotArtifact(
        name=name,
        source_csv=str(source_csv),
        output_path=output_path,
        status=status,
        rows_used=int(rows_used),
        message=message,
    )


def plot_module_activation_summary(
    paths: Step14DiagnosticsPaths,
    plot_dir: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[PlotArtifact]:
    """Plot module activation summaries."""
    artifacts: List[PlotArtifact] = []

    source = paths.module_usage_csv
    df = _read_csv_if_exists(source)

    if df is None:
        artifacts.append(
            PlotArtifact(
                name="module_activation_mean_abs",
                source_csv=str(source),
                output_path="",
                status="SKIPPED",
                rows_used=0,
                message="module activation CSV not found or empty",
            )
        )
        return artifacts

    if "split" not in df.columns:
        df["split"] = "unknown"

    for split_name, split_df in df.groupby("split", sort=False):
        if "module_name" not in split_df.columns or "mean_abs" not in split_df.columns:
            continue

        plot_df = _top_value_rows(
            split_df,
            value_column="mean_abs",
            top_k=plot_config.top_k,
            ascending=False,
        )

        if plot_df.empty:
            continue

        output_base = plot_dir / f"module_activation_mean_abs_{split_name}"

        output_paths = _barh_plot(
            df=plot_df,
            label_column="module_name",
            value_column="mean_abs",
            title=f"Step 14 Module Activation Magnitude | split={split_name}",
            xlabel="Mean absolute activation",
            output_base=output_base,
            plot_config=plot_config,
        )

        artifacts.append(
            _make_artifact(
                name=f"module_activation_mean_abs_{split_name}",
                source_csv=source,
                output_paths=output_paths,
                rows_used=len(plot_df),
            )
        )

    return artifacts


def plot_feature_importance_summary(
    paths: Step14DiagnosticsPaths,
    plot_dir: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[PlotArtifact]:
    """Plot feature-importance summary."""
    artifacts: List[PlotArtifact] = []

    source = paths.feature_importance_csv
    df = _read_csv_if_exists(source)

    if df is None:
        artifacts.append(
            PlotArtifact(
                name="feature_importance",
                source_csv=str(source),
                output_path="",
                status="SKIPPED",
                rows_used=0,
                message="feature importance CSV not found or empty",
            )
        )
        return artifacts

    if "split" not in df.columns:
        df["split"] = "unknown"

    for split_name, split_df in df.groupby("split", sort=False):
        if "group_name" not in split_df.columns:
            continue

        if "delta_f1" in split_df.columns:
            plot_df = _top_abs_rows(
                split_df,
                value_column="delta_f1",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=plot_df,
                label_column="group_name",
                value_column="delta_f1",
                title=f"Step 14 Feature Diagnostic Occlusion: ΔF1 | split={split_name}",
                xlabel="Occluded F1 - baseline F1",
                output_base=plot_dir / f"feature_importance_delta_f1_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"feature_importance_delta_f1_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(plot_df),
                )
            )

        if "mean_abs_probability_change" in split_df.columns:
            plot_df = _top_value_rows(
                split_df,
                value_column="mean_abs_probability_change",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=plot_df,
                label_column="group_name",
                value_column="mean_abs_probability_change",
                title=f"Step 14 Feature Diagnostic Occlusion: mean |Δp| | split={split_name}",
                xlabel="Mean absolute probability change",
                output_base=plot_dir / f"feature_importance_mean_abs_probability_change_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"feature_importance_mean_abs_probability_change_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(plot_df),
                )
            )

    return artifacts


def plot_conductance_summary(
    paths: Step14DiagnosticsPaths,
    plot_dir: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[PlotArtifact]:
    """Plot conductance/exchange summaries."""
    artifacts: List[PlotArtifact] = []

    source = paths.conductance_csv
    df = _read_csv_if_exists(source)

    if df is None:
        artifacts.append(
            PlotArtifact(
                name="conductance_summary",
                source_csv=str(source),
                output_path="",
                status="SKIPPED",
                rows_used=0,
                message="conductance CSV not found or empty",
            )
        )
        return artifacts

    if "split" not in df.columns:
        df["split"] = "unknown"

    label_column = "tensor_role" if "tensor_role" in df.columns else "module_name"

    for split_name, split_df in df.groupby("split", sort=False):
        if "mean_abs" in split_df.columns:
            plot_df = split_df.copy()
            plot_df["plot_label"] = (
                plot_df.get("module_name", "").astype(str)
                + " | "
                + plot_df.get(label_column, "").astype(str)
            )

            plot_df = _top_value_rows(
                plot_df,
                value_column="mean_abs",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=plot_df,
                label_column="plot_label",
                value_column="mean_abs",
                title=f"Step 14 Kirchhoff/Conductance Magnitude | split={split_name}",
                xlabel="Mean absolute tensor value",
                output_base=plot_dir / f"conductance_mean_abs_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"conductance_mean_abs_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(plot_df),
                )
            )

        if "attack_minus_normal_mean_abs" in split_df.columns:
            plot_df = split_df.copy()
            plot_df["plot_label"] = (
                plot_df.get("module_name", "").astype(str)
                + " | "
                + plot_df.get(label_column, "").astype(str)
            )

            plot_df = _top_abs_rows(
                plot_df,
                value_column="attack_minus_normal_mean_abs",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=plot_df,
                label_column="plot_label",
                value_column="attack_minus_normal_mean_abs",
                title=f"Step 14 Kirchhoff/Conductance Attack-Normal Difference | split={split_name}",
                xlabel="Attack mean |tensor| - normal mean |tensor|",
                output_base=plot_dir / f"conductance_attack_minus_normal_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"conductance_attack_minus_normal_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(plot_df),
                )
            )

    return artifacts


def plot_third_order_summary(
    paths: Step14DiagnosticsPaths,
    plot_dir: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[PlotArtifact]:
    """Plot third-order/high-order summaries."""
    artifacts: List[PlotArtifact] = []

    source = paths.third_order_csv
    df = _read_csv_if_exists(source)

    if df is None:
        artifacts.append(
            PlotArtifact(
                name="third_order_summary",
                source_csv=str(source),
                output_path="",
                status="SKIPPED",
                rows_used=0,
                message="third-order CSV not found or empty",
            )
        )
        return artifacts

    if "split" not in df.columns:
        df["split"] = "unknown"

    for split_name, split_df in df.groupby("split", sort=False):
        plot_df = split_df.copy()

        if "tensor_role" in plot_df.columns:
            plot_df["plot_label"] = (
                plot_df.get("module_name", "").astype(str)
                + " | "
                + plot_df.get("tensor_role", "").astype(str)
            )
        else:
            plot_df["plot_label"] = plot_df.get("module_name", "").astype(str)

        if "mean_abs" in plot_df.columns:
            top_df = _top_value_rows(
                plot_df,
                value_column="mean_abs",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=top_df,
                label_column="plot_label",
                value_column="mean_abs",
                title=f"Step 14 Third-Order / High-Order Magnitude | split={split_name}",
                xlabel="Mean absolute tensor value",
                output_base=plot_dir / f"third_order_mean_abs_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"third_order_mean_abs_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(top_df),
                )
            )

        if "active_fraction_abs_gt_001" in plot_df.columns:
            top_df = _top_value_rows(
                plot_df,
                value_column="active_fraction_abs_gt_001",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=top_df,
                label_column="plot_label",
                value_column="active_fraction_abs_gt_001",
                title=f"Step 14 Third-Order Active Fraction | split={split_name}",
                xlabel="Fraction of tensor entries with |value| > 0.01",
                output_base=plot_dir / f"third_order_active_fraction_001_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"third_order_active_fraction_001_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(top_df),
                )
            )

    return artifacts


def plot_liquid_state_summary(
    paths: Step14DiagnosticsPaths,
    plot_dir: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[PlotArtifact]:
    """Plot liquid-state summaries."""
    artifacts: List[PlotArtifact] = []

    source = paths.liquid_state_csv
    df = _read_csv_if_exists(source)

    if df is None:
        artifacts.append(
            PlotArtifact(
                name="liquid_state_summary",
                source_csv=str(source),
                output_path="",
                status="SKIPPED",
                rows_used=0,
                message="liquid-state CSV not found or empty",
            )
        )
        return artifacts

    if "split" not in df.columns:
        df["split"] = "unknown"

    for split_name, split_df in df.groupby("split", sort=False):
        plot_df = split_df.copy()

        if "tensor_role" in plot_df.columns:
            plot_df["plot_label"] = (
                plot_df.get("module_name", "").astype(str)
                + " | "
                + plot_df.get("tensor_role", "").astype(str)
            )
        else:
            plot_df["plot_label"] = plot_df.get("module_name", "").astype(str)

        if "mean_abs" in plot_df.columns:
            top_df = _top_value_rows(
                plot_df,
                value_column="mean_abs",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=top_df,
                label_column="plot_label",
                value_column="mean_abs",
                title=f"Step 14 Liquid/Temporal State Magnitude | split={split_name}",
                xlabel="Mean absolute tensor value",
                output_base=plot_dir / f"liquid_state_mean_abs_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"liquid_state_mean_abs_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(top_df),
                )
            )

        if "mean_abs_temporal_delta" in plot_df.columns:
            top_df = _top_value_rows(
                plot_df,
                value_column="mean_abs_temporal_delta",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=top_df,
                label_column="plot_label",
                value_column="mean_abs_temporal_delta",
                title=f"Step 14 Liquid/Temporal State Variation | split={split_name}",
                xlabel="Mean absolute adjacent-time change",
                output_base=plot_dir / f"liquid_state_temporal_delta_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"liquid_state_temporal_delta_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(top_df),
                )
            )

        if "attack_minus_normal_mean_abs" in plot_df.columns:
            top_df = _top_abs_rows(
                plot_df,
                value_column="attack_minus_normal_mean_abs",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=top_df,
                label_column="plot_label",
                value_column="attack_minus_normal_mean_abs",
                title=f"Step 14 Liquid/Temporal Attack-Normal Difference | split={split_name}",
                xlabel="Attack mean |tensor| - normal mean |tensor|",
                output_base=plot_dir / f"liquid_state_attack_minus_normal_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"liquid_state_attack_minus_normal_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(top_df),
                )
            )

    return artifacts


def plot_occlusion_summary(
    paths: Step14DiagnosticsPaths,
    plot_dir: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[PlotArtifact]:
    """Plot diagnostic occlusion-test summaries."""
    artifacts: List[PlotArtifact] = []

    source = paths.occlusion_csv
    df = _read_csv_if_exists(source)

    if df is None:
        artifacts.append(
            PlotArtifact(
                name="occlusion_summary",
                source_csv=str(source),
                output_path="",
                status="SKIPPED",
                rows_used=0,
                message="occlusion CSV not found or empty",
            )
        )
        return artifacts

    if "split" not in df.columns:
        df["split"] = "unknown"

    label_column = "scenario_name" if "scenario_name" in df.columns else "group_name"

    for split_name, split_df in df.groupby("split", sort=False):
        if "delta_f1" in split_df.columns:
            plot_df = _top_abs_rows(
                split_df,
                value_column="delta_f1",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=plot_df,
                label_column=label_column,
                value_column="delta_f1",
                title=f"Step 14 Diagnostic Occlusion: ΔF1 | split={split_name}",
                xlabel="Occluded F1 - baseline F1",
                output_base=plot_dir / f"occlusion_delta_f1_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"occlusion_delta_f1_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(plot_df),
                )
            )

        if "mean_abs_probability_change" in split_df.columns:
            plot_df = _top_value_rows(
                split_df,
                value_column="mean_abs_probability_change",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=plot_df,
                label_column=label_column,
                value_column="mean_abs_probability_change",
                title=f"Step 14 Diagnostic Occlusion: mean |Δp| | split={split_name}",
                xlabel="Mean absolute probability change",
                output_base=plot_dir / f"occlusion_mean_abs_probability_change_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"occlusion_mean_abs_probability_change_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(plot_df),
                )
            )

        if "delta_detection_delay" in split_df.columns:
            plot_df = _top_abs_rows(
                split_df,
                value_column="delta_detection_delay",
                top_k=plot_config.top_k,
                ascending=False,
            )

            output_paths = _barh_plot(
                df=plot_df,
                label_column=label_column,
                value_column="delta_detection_delay",
                title=f"Step 14 Diagnostic Occlusion: ΔDetection Delay | split={split_name}",
                xlabel="Occluded mean delay - baseline mean delay",
                output_base=plot_dir / f"occlusion_delta_detection_delay_{split_name}",
                plot_config=plot_config,
            )

            artifacts.append(
                _make_artifact(
                    name=f"occlusion_delta_detection_delay_{split_name}",
                    source_csv=source,
                    output_paths=output_paths,
                    rows_used=len(plot_df),
                )
            )

    return artifacts


def plot_metric_tradeoff_summary(
    paths: Step14DiagnosticsPaths,
    plot_dir: Path,
    plot_config: ModuleUsagePlotConfig,
) -> List[PlotArtifact]:
    """
    Plot simple occlusion tradeoff scatter:
    mean |Δp| vs ΔF1.
    """
    artifacts: List[PlotArtifact] = []

    source = paths.occlusion_csv
    df = _read_csv_if_exists(source)

    if df is None:
        return artifacts

    if "split" not in df.columns:
        df["split"] = "unknown"

    for split_name, split_df in df.groupby("split", sort=False):
        output_paths = _scatter_plot(
            df=split_df,
            x_column="mean_abs_probability_change",
            y_column="delta_f1",
            title=f"Step 14 Occlusion Sensitivity Tradeoff | split={split_name}",
            xlabel="Mean absolute probability change",
            ylabel="Occluded F1 - baseline F1",
            output_base=plot_dir / f"occlusion_tradeoff_mean_abs_probability_change_vs_delta_f1_{split_name}",
            plot_config=plot_config,
        )

        artifacts.append(
            _make_artifact(
                name=f"occlusion_tradeoff_{split_name}",
                source_csv=source,
                output_paths=output_paths,
                rows_used=len(split_df),
                message="" if output_paths else "required columns missing or empty",
            )
        )

    return artifacts


def run_module_usage_plots(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate Step-14 module-usage diagnostic plots.

    This function should be called after Step-14 diagnostic CSVs have been saved.
    """
    diagnostics_config = build_step14_diagnostics_config(config)
    paths = build_step14_paths(config, diagnostics_config)
    plot_config = build_module_usage_plot_config(config)
    plot_dir = get_plot_output_dir(config, plot_config)

    print("=" * 100)
    print("STEP 14 MODULE-USAGE PLOTS")
    print("=" * 100)
    print(f"Input directory  : {paths.output_dir}")
    print(f"Plot directory   : {plot_dir}")
    print(f"Top K            : {plot_config.top_k}")
    print(f"Save PNG/PDF     : {plot_config.save_png} / {plot_config.save_pdf}")
    print("=" * 100)

    artifacts: List[PlotArtifact] = []

    artifacts.extend(
        plot_module_activation_summary(
            paths=paths,
            plot_dir=plot_dir,
            plot_config=plot_config,
        )
    )
    artifacts.extend(
        plot_feature_importance_summary(
            paths=paths,
            plot_dir=plot_dir,
            plot_config=plot_config,
        )
    )
    artifacts.extend(
        plot_conductance_summary(
            paths=paths,
            plot_dir=plot_dir,
            plot_config=plot_config,
        )
    )
    artifacts.extend(
        plot_third_order_summary(
            paths=paths,
            plot_dir=plot_dir,
            plot_config=plot_config,
        )
    )
    artifacts.extend(
        plot_liquid_state_summary(
            paths=paths,
            plot_dir=plot_dir,
            plot_config=plot_config,
        )
    )
    artifacts.extend(
        plot_occlusion_summary(
            paths=paths,
            plot_dir=plot_dir,
            plot_config=plot_config,
        )
    )
    artifacts.extend(
        plot_metric_tradeoff_summary(
            paths=paths,
            plot_dir=plot_dir,
            plot_config=plot_config,
        )
    )

    rows = [artifact.to_dict() for artifact in artifacts]

    manifest_path = plot_dir / "module_usage_plot_manifest.json"
    save_json_safe(
        {
            "status": "PASSED",
            "active_seed": int(active_seed),
            "plot_config": plot_config.to_dict(),
            "diagnostics_paths": paths.to_dict(),
            "artifact_count": len(rows),
            "artifacts": rows,
            "interpretation_note": (
                "These plots visualize diagnostic module usage and occlusion tests. "
                "They are not official ablation results."
            ),
        },
        manifest_path,
    )

    passed = [item for item in artifacts if item.status == "PASSED"]
    skipped = [item for item in artifacts if item.status == "SKIPPED"]

    print("Generated plot artifacts:")
    for item in passed:
        print(f"  PASSED  | {item.name}: {item.output_path}")

    if skipped:
        print("Skipped plot artifacts:")
        for item in skipped:
            print(f"  SKIPPED | {item.name}: {item.message}")

    print(f"Manifest: {manifest_path}")
    print("=" * 100)

    return {
        "status": "PASSED",
        "active_seed": int(active_seed),
        "plot_dir": str(plot_dir),
        "manifest_path": str(manifest_path),
        "artifact_count": len(rows),
        "passed_count": len(passed),
        "skipped_count": len(skipped),
        "artifacts": rows,
    }


__all__ = [
    "PlotArtifact",
    "ModuleUsagePlotConfig",
    "build_module_usage_plot_config",
    "get_plot_output_dir",
    "plot_module_activation_summary",
    "plot_feature_importance_summary",
    "plot_conductance_summary",
    "plot_third_order_summary",
    "plot_liquid_state_summary",
    "plot_occlusion_summary",
    "plot_metric_tradeoff_summary",
    "run_module_usage_plots",
]