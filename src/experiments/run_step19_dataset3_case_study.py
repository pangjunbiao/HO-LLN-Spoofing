"""
Step 19: Dataset-3 online EKF comparison and case-study visualization.

This step does not train, retune, or rerun the Proposed model.

It uses:
- existing Proposed Dataset-3 prediction CSV from Step 13,
- Dataset-3 EKF Detector column,
- same event-level online alarm rule for Proposed and EKF,
- one comparison table,
- one case-study figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation.evaluate_dataset3 import (
    compute_dataset3_metrics_from_confirmed_alarm_df,
    get_dataset3_paths,
    load_saved_dataset3_predictions_csv,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_json
from src.visualization.plot_dataset3_case import plot_dataset3_case_study


def _project_path(config: Mapping[str, Any], path_value: str | Path) -> Path:
    """Resolve a project-relative path."""
    return resolve_project_path(config, str(path_value))


def _get_optional_config_path(
    config: Mapping[str, Any],
    key: str,
) -> Optional[Path]:
    """Resolve an optional config path."""
    value = get_by_path(config, key, None)
    if value is None or str(value).strip() == "":
        return None
    return _project_path(config, str(value))


def _unique_existing_paths(paths: Sequence[Optional[Path]]) -> List[Path]:
    """Keep unique existing paths in order."""
    seen = set()
    out: List[Path] = []

    for path in paths:
        if path is None:
            continue

        resolved = Path(path)

        key = str(resolved.resolve()) if resolved.exists() else str(resolved)
        if key in seen:
            continue

        seen.add(key)

        if resolved.exists():
            out.append(resolved)

    return out


def _step19_output_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    """Resolve Step-19 output paths."""
    dataset3_paths = get_dataset3_paths(config)

    figure_dir = _project_path(
        config,
        str(
            get_by_path(
                config,
                "step19_dataset3_case_study.outputs.figure_dir",
                get_by_path(
                    config,
                    "paths.dataset3_case_study_figure_dir",
                    "results/figures/dataset3_case_study",
                ),
            )
        ),
    )

    plot_source_csv = _project_path(
        config,
        str(
            get_by_path(
                config,
                "step19_dataset3_case_study.outputs.plot_source_csv",
                get_by_path(
                    config,
                    "paths.dataset3_online_case_plot_source_csv",
                    "results/tables/dataset3_online_case_study_plot_source.csv",
                ),
            )
        ),
    )

    return {
        "comparison_csv": dataset3_paths["dataset3_ekf_comparison_csv"],
        "summary_json": dataset3_paths["dataset3_ekf_comparison_summary_json"],
        "figure_dir": figure_dir,
        "plot_source_csv": plot_source_csv,
    }


def _candidate_source_paths(config: Mapping[str, Any]) -> List[Path]:
    """
    Candidate CSV files that may contain:
    - EKF Detector,
    - xi_q / xi_q_scaled,
    - xi_accum_log / xi_accum_log_scaled.
    """
    candidates: List[Optional[Path]] = [
        _get_optional_config_path(
            config,
            "step19_dataset3_case_study.inputs.online_xi_csv",
        ),
        _get_optional_config_path(config, "paths.online_xi_csv"),
        _get_optional_config_path(config, "paths.processed_online_xi_csv"),
        _project_path(config, "data/processed/online_xi.csv"),
        _get_optional_config_path(
            config,
            "step19_dataset3_case_study.inputs.raw_dataset3_csv",
        ),
        _get_optional_config_path(config, "paths.dataset3_raw_csv"),
        _get_optional_config_path(config, "paths.raw_dataset3_csv"),
        _get_optional_config_path(config, "dataset.dataset3_path"),
        _get_optional_config_path(config, "dataset.raw_dataset3_path"),
    ]

    return _unique_existing_paths(candidates)


def _to_binary_alarm(values: Sequence[Any], column_name: str) -> np.ndarray:
    """Convert EKF Detector values to binary 0/1 alarm."""
    series = pd.Series(values)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(series):
        return (numeric.to_numpy(dtype=float) >= 0.5).astype(int)

    positive_tokens = {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "alarm",
        "attack",
        "spoof",
        "spoofing",
        "detected",
        "positive",
        "warning",
    }
    negative_tokens = {
        "0",
        "false",
        "f",
        "no",
        "n",
        "normal",
        "none",
        "clear",
        "negative",
        "safe",
    }

    out = np.zeros(len(series), dtype=int)

    for i, value in enumerate(series.astype(str).str.strip().str.lower()):
        if value in positive_tokens:
            out[i] = 1
        elif value in negative_tokens or value == "" or value == "nan":
            out[i] = 0
        else:
            maybe_number = pd.to_numeric(value, errors="coerce")
            if pd.notna(maybe_number):
                out[i] = int(float(maybe_number) >= 0.5)
            else:
                raise ValueError(
                    f"Column '{column_name}' contains non-binary value '{value}'. "
                    "Please convert EKF Detector to 0/1 or a recognized binary token."
                )

    return out


def _attach_columns_by_alignment(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    columns: Sequence[str],
    source_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Attach selected columns from source dataframe to base dataframe.

    Alignment rule:
    1. If both have row_index, merge by row_index.
    2. Otherwise, if lengths match, attach by row order.
    3. Otherwise, raise a clear error.
    """
    available = [column for column in columns if column in source_df.columns]
    if not available:
        return base_df, []

    out = base_df.copy()

    if "row_index" in out.columns and "row_index" in source_df.columns:
        temp = source_df[["row_index", *available]].copy()
        merged = out.merge(
            temp,
            on="row_index",
            how="left",
            suffixes=("", "_source"),
        )

        attached: List[str] = []
        for column in available:
            if column in out.columns:
                source_column = f"{column}_source"
                if source_column in merged.columns:
                    missing_before = merged[column].isna()
                    merged[column] = merged[column].where(
                        ~missing_before,
                        merged[source_column],
                    )
                    merged = merged.drop(columns=[source_column])
                    attached.append(column)
            else:
                attached.append(column)

        return merged, attached

    if len(out) == len(source_df):
        attached = []
        for column in available:
            if column not in out.columns:
                out[column] = source_df[column].to_numpy()
                attached.append(column)
        return out, attached

    raise ValueError(
        f"Cannot align source file '{source_name}' with Dataset-3 predictions. "
        f"Prediction rows={len(out)}, source rows={len(source_df)}. "
        "Add a row_index column to the source file or provide a matching online_xi.csv."
    )


def _load_and_attach_step19_sources(
    config: Mapping[str, Any],
    prediction_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Attach EKF Detector and diagnostic feature columns to saved predictions.
    """
    ekf_column = str(
        get_by_path(
            config,
            "step19_dataset3_case_study.ekf.column",
            get_by_path(config, "dataset.ekf_column", "EKF Detector"),
        )
    )

    residual_candidates = list(
        get_by_path(
            config,
            "step19_dataset3_case_study.plotting.residual_energy_candidates",
            ["xi_q", "xi_q_scaled"],
        )
    )

    accum_candidates = list(
        get_by_path(
            config,
            "step19_dataset3_case_study.plotting.accumulated_evidence_candidates",
            ["xi_accum_log", "xi_accum_log_scaled"],
        )
    )

    needed_columns = [ekf_column, *residual_candidates, *accum_candidates]

    out = prediction_df.copy()
    attached_columns: Dict[str, List[str]] = {}
    source_paths = _candidate_source_paths(config)

    for path in source_paths:
        try:
            source_df = pd.read_csv(path)
        except Exception as exc:
            attached_columns[str(path)] = [f"SKIPPED_READ_ERROR: {exc}"]
            continue

        out, attached = _attach_columns_by_alignment(
            base_df=out,
            source_df=source_df,
            columns=needed_columns,
            source_name=str(path),
        )

        if attached:
            attached_columns[str(path)] = attached

    if ekf_column not in out.columns:
        checked = [str(path) for path in source_paths]
        raise KeyError(
            f"EKF Detector column '{ekf_column}' was not found. "
            f"Checked source files: {checked}. "
            "Add step19_dataset3_case_study.inputs.online_xi_csv or raw_dataset3_csv "
            "in configs/experiments.yaml."
        )

    out["ekf_alarm"] = _to_binary_alarm(
        values=out[ekf_column].to_numpy(),
        column_name=ekf_column,
    )

    return out, {
        "ekf_column": ekf_column,
        "source_paths_checked": [str(path) for path in source_paths],
        "attached_columns": attached_columns,
        "residual_energy_candidates": residual_candidates,
        "accumulated_evidence_candidates": accum_candidates,
    }


def _metric_value(metrics: Mapping[str, Any], key: str) -> Any:
    """Return metric value while preserving None."""
    value = metrics.get(key)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _comparison_row(method: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Build one Step-19 comparison table row."""
    return {
        "Method": method,
        "False Alarm Rows": _metric_value(metrics, "row_level_false_alarms"),
        "False Alarm Events": _metric_value(metrics, "false_alarm_events"),
        "Attack-1 Delay": _metric_value(metrics, "attack_1_delay"),
        "Attack-2 Delay": _metric_value(metrics, "attack_2_delay"),
        "Mean Delay": _metric_value(metrics, "mean_detection_delay"),
        "Attack Detection Rate": _metric_value(metrics, "attack_detection_rate"),
    }


def _compact_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove large dataframe before JSON saving."""
    out = dict(payload)
    out.pop("evaluation_frame", None)
    return out


def _json_safe(obj: Any) -> Any:
    """Convert common numpy/pandas objects to JSON-safe values."""
    if isinstance(obj, Mapping):
        return {str(key): _json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_json_safe(value) for value in obj]

    if isinstance(obj, tuple):
        return [_json_safe(value) for value in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value

    if isinstance(obj, np.ndarray):
        return [_json_safe(value) for value in obj.tolist()]

    if pd.isna(obj) if not isinstance(obj, (str, bytes, list, tuple, dict)) else False:
        return None

    return obj


def _load_selected_threshold_for_plot(
    config: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    """Load selected theta/persistence for plot annotation when available."""
    paths = get_dataset3_paths(config)
    threshold_path = paths["threshold_selection_json"]

    if not threshold_path.exists():
        return {"theta": None, "persistence": None, "source": str(threshold_path)}

    try:
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    except Exception:
        return {"theta": None, "persistence": None, "source": str(threshold_path)}

    theta = (
        payload.get("theta")
        or payload.get("selected_theta")
        or payload.get("best_theta")
    )
    persistence = (
        payload.get("persistence")
        or payload.get("selected_persistence")
        or payload.get("best_persistence")
    )

    if theta is None and isinstance(payload.get("selected_candidate"), Mapping):
        theta = payload["selected_candidate"].get("theta")
        persistence = payload["selected_candidate"].get("persistence", persistence)

    if theta is None and isinstance(payload.get("selected"), Mapping):
        theta = payload["selected"].get("theta")
        persistence = payload["selected"].get("persistence", persistence)

    try:
        theta = None if theta is None else float(theta)
    except Exception:
        theta = None

    try:
        persistence = None if persistence is None else int(persistence)
    except Exception:
        persistence = None

    return {
        "theta": theta,
        "persistence": persistence,
        "source": str(threshold_path),
    }


def _run_sanity_check(
    config: Mapping[str, Any],
    proposed_metrics: Mapping[str, Any],
) -> List[str]:
    """
    Compare Proposed Step-19 metrics against expected Step-13 values when configured.

    By default this warns only. Set:
    step19_dataset3_case_study.sanity_check.strict: true
    to raise an error.
    """
    enabled = bool(
        get_by_path(
            config,
            "step19_dataset3_case_study.sanity_check.enabled",
            False,
        )
    )

    if not enabled:
        return []

    expected = get_by_path(
        config,
        "step19_dataset3_case_study.sanity_check.expected_proposed",
        {},
    )

    if not isinstance(expected, Mapping):
        return []

    key_map = {
        "false_alarm_events": "false_alarm_events",
        "attack_1_delay": "attack_1_delay",
        "attack_2_delay": "attack_2_delay",
        "mean_delay": "mean_detection_delay",
        "mean_detection_delay": "mean_detection_delay",
        "attack_detection_rate": "attack_detection_rate",
    }

    warnings: List[str] = []

    for expected_key, metric_key in key_map.items():
        if expected_key not in expected:
            continue

        expected_value = expected.get(expected_key)
        observed_value = proposed_metrics.get(metric_key)

        try:
            expected_float = float(expected_value)
            observed_float = float(observed_value)
            matches = abs(expected_float - observed_float) <= 1e-6
        except Exception:
            matches = expected_value == observed_value

        if not matches:
            warnings.append(
                f"Sanity check mismatch for {expected_key}: "
                f"expected={expected_value}, observed={observed_value}"
            )

    strict = bool(
        get_by_path(
            config,
            "step19_dataset3_case_study.sanity_check.strict",
            False,
        )
    )

    if warnings and strict:
        raise AssertionError("Step-19 sanity check failed: " + "; ".join(warnings))

    return warnings


def run_step19_dataset3_case_study(
    config: Mapping[str, Any],
    active_seed: int = 42,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run Step 19: Proposed vs EKF Detector on Dataset-3 and case-study plot.

    This function assumes Step 13 has already created:
    results/tables/dataset3_online_predictions.csv
    """
    if logger is not None:
        logger.info("Step 19 Dataset-3 EKF comparison and case-study plotting is running.")

    print("=" * 120)
    print("STEP 19 DATASET-3 ONLINE EKF COMPARISON AND CASE-STUDY FIGURE")
    print("=" * 120)
    print("Rule: no training, no threshold retuning, no Dataset-3 tuning.")
    print("Input: saved Proposed Dataset-3 predictions from Step 13.")
    print("=" * 120)

    output_paths = _step19_output_paths(config)

    prediction_df = load_saved_dataset3_predictions_csv(config=config)
    case_df, source_info = _load_and_attach_step19_sources(
        config=config,
        prediction_df=prediction_df,
    )

    proposed_payload = compute_dataset3_metrics_from_confirmed_alarm_df(
        df=case_df,
        alarm_column="confirmed_alarm",
        method_name="Proposed",
    )
    ekf_payload = compute_dataset3_metrics_from_confirmed_alarm_df(
        df=case_df,
        alarm_column="ekf_alarm",
        method_name="EKF Detector",
    )

    proposed_metrics = proposed_payload["metrics"]
    ekf_metrics = ekf_payload["metrics"]

    sanity_warnings = _run_sanity_check(
        config=config,
        proposed_metrics=proposed_metrics,
    )

    comparison_rows = [
        _comparison_row("Proposed", proposed_metrics),
        _comparison_row("EKF Detector", ekf_metrics),
    ]

    comparison_df = pd.DataFrame(comparison_rows)

    ensure_dir(output_paths["comparison_csv"].parent)
    comparison_df.to_csv(output_paths["comparison_csv"], index=False)

    ensure_dir(output_paths["plot_source_csv"].parent)
    case_df.to_csv(output_paths["plot_source_csv"], index=False)

    threshold_info = _load_selected_threshold_for_plot(config)

    figure_result = plot_dataset3_case_study(
        df=case_df,
        output_dir=output_paths["figure_dir"],
        # title="Dataset-3 online case study: Proposed vs EKF Detector",
        title="Dataset-3 causal sequential case study: Proposed vs EKF Detector",
        probability_column="probability",
        proposed_alarm_column="confirmed_alarm",
        ekf_alarm_column="ekf_alarm",
        label_column="label",
        residual_energy_candidates=source_info["residual_energy_candidates"],
        accumulated_evidence_candidates=source_info["accumulated_evidence_candidates"],
        theta=threshold_info.get("theta"),
        save_png=bool(
            get_by_path(
                config,
                "step19_dataset3_case_study.plotting.save_png",
                True,
            )
        ),
        save_pdf=bool(
            get_by_path(
                config,
                "step19_dataset3_case_study.plotting.save_pdf",
                True,
            )
        ),
    )

    summary = {
        "final_status": "PASSED",
        "active_seed": int(active_seed),
        "step": "step19_dataset3_case_study",
        "rule": {
            "training_used": False,
            "threshold_retuning_used": False,
            "dataset3_used_for_threshold_selection": False,
            "proposed_predictions_reused_from_step13": True,
            "ekf_detector_used_as_input_to_model": False,
        },
        "input_summary": {
            "prediction_rows": int(len(prediction_df)),
            "case_rows": int(len(case_df)),
            **source_info,
        },
        "selected_threshold_for_plot": threshold_info,
        "comparison_table": comparison_rows,
        "proposed_payload": _compact_payload(proposed_payload),
        "ekf_payload": _compact_payload(ekf_payload),
        "figure_result": figure_result,
        "sanity_warnings": sanity_warnings,
        "output_paths": {
            "comparison_csv": str(output_paths["comparison_csv"]),
            "summary_json": str(output_paths["summary_json"]),
            "plot_source_csv": str(output_paths["plot_source_csv"]),
            "figure_dir": str(output_paths["figure_dir"]),
            "figure_png": figure_result.get("png_path"),
            "figure_pdf": figure_result.get("pdf_path"),
        },
    }

    ensure_dir(output_paths["summary_json"].parent)
    save_json(_json_safe(summary), output_paths["summary_json"], indent=2)

    print("STEP 19 PRIMARY COMPARISON")
    print("-" * 120)
    print(comparison_df.to_string(index=False))
    print("-" * 120)

    if sanity_warnings:
        print("Sanity warnings:")
        for warning in sanity_warnings:
            print(f"  - {warning}")

    print("Saved Step-19 artifacts:")
    print(f"  comparison_csv : {output_paths['comparison_csv']}")
    print(f"  summary_json   : {output_paths['summary_json']}")
    print(f"  plot_source_csv: {output_paths['plot_source_csv']}")
    print(f"  figure_dir     : {output_paths['figure_dir']}")
    print("=" * 120)

    if logger is not None:
        logger.info("Step 19 final status: PASSED")
        logger.info("Step 19 comparison CSV: %s", output_paths["comparison_csv"])
        logger.info("Step 19 figure directory: %s", output_paths["figure_dir"])

    return summary


__all__ = [
    "run_step19_dataset3_case_study",
]