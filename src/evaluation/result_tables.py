"""
Result-table utilities for Step 13 and later experiments.

Purpose:
- save reviewer-ready comparison tables,
- keep locked metric columns consistent,
- update rows without duplicating the same model,
- support Dataset-1, Dataset-2, and Dataset-3 tables.

Primary metrics locked for the project:
- AUPRC
- F1-score
- FPR
- Attack Detection Rate
- Detection Delay

Secondary metrics may also be saved:
- AUROC
- Precision
- Recall
- Runtime
- Normal-Segment FAR
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import math

import pandas as pd
from pandas.errors import EmptyDataError


DATASET1_MAIN_COLUMNS = [
    "Model",
    "Split",
    "AUROC",
    "AUPRC",
    "F1",
    "Precision",
    "Recall",
    "FPR",
    "Attack Detection Rate",
    "Detection Delay",
    "Runtime",
    "Threshold",
    "Persistence",
    "Checkpoint",
    "Notes",
]

DATASET2_EXTERNAL_COLUMNS = [
    "Model",
    "AUROC",
    "AUPRC",
    "F1",
    "Recall",
    "FPR",
    "Precision",
    "Attack Detection Rate",
    "Detection Delay",
    "Runtime",
    "Threshold",
    "Persistence",
    "Checkpoint",
    "Notes",
]

DATASET3_ONLINE_CASE_COLUMNS = [
    "Method",
    "False Alarms",
    "Attack-1 Delay",
    "Attack-2 Delay",
    "Mean Delay",
    "Attack Detection Rate",
    "F1",
    "Precision",
    "Recall",
    "FPR",
    "Threshold",
    "Persistence",
    "Checkpoint",
    "Notes",
]

SENSITIVITY_COLUMNS = [
    "Model",
    "Split",
    "Sensitivity Parameter",
    "Sensitivity Value",
    "theta",
    "persistence",
    "AUROC",
    "AUPRC",
    "F1",
    "Precision",
    "Recall",
    "FPR",
    "Attack Detection Rate",
    "Detection Delay",
    "False Alarms",
    "is_official_selected",
    "status",
    "notes",
]

PRIMARY_METRIC_COLUMNS = [
    "AUPRC",
    "F1",
    "FPR",
    "Attack Detection Rate",
    "Detection Delay",
]


def ensure_parent_dir(path: Path | str) -> Path:
    """Ensure parent directory exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_scalar(value: Any) -> Any:
    """Convert metric values to CSV-safe scalars."""
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)

    if isinstance(value, int):
        return int(value)

    try:
        converted = float(value)
        if math.isfinite(converted):
            return converted
    except Exception:
        pass

    return value


def normalize_row(row: Mapping[str, Any], columns: Sequence[str]) -> Dict[str, Any]:
    """Return row with exactly the requested columns."""
    return {column: _safe_scalar(row.get(column, None)) for column in columns}


def read_existing_table(path: Path | str, columns: Sequence[str]) -> pd.DataFrame:
    """
    Read existing table or return empty table.

    Robust to:
    - missing file,
    - zero-byte empty file,
    - whitespace-only file,
    - corrupted/empty CSV from a previously interrupted run.
    """
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(columns=list(columns))

    try:
        if path.stat().st_size == 0:
            return pd.DataFrame(columns=list(columns))

        df = pd.read_csv(path)

    except EmptyDataError:
        return pd.DataFrame(columns=list(columns))

    except pd.errors.ParserError:
        backup_path = path.with_suffix(path.suffix + ".corrupt_backup")
        path.replace(backup_path)
        return pd.DataFrame(columns=list(columns))

    for column in columns:
        if column not in df.columns:
            df[column] = None

    return df[list(columns)]

def upsert_result_row(
    output_path: Path | str,
    row: Mapping[str, Any],
    columns: Sequence[str],
    key_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Insert/update one row in a result table.

    If a row with the exact same key already exists, it is replaced.
    Rows with the same model but different split are preserved.
    """
    output_path = ensure_parent_dir(output_path)

    df = read_existing_table(output_path, columns)
    normalized = normalize_row(row, columns)

    if len(df) > 0 and len(key_columns) > 0:
        match_mask = pd.Series([True] * len(df), index=df.index)

        for key in key_columns:
            if key not in df.columns:
                match_mask = pd.Series([False] * len(df), index=df.index)
                break

            match_mask = match_mask & (
                df[key].astype(str) == str(normalized.get(key))
            )

        df = df[~match_mask].copy()

    new_df = pd.concat([df, pd.DataFrame([normalized])], ignore_index=True)
    new_df = new_df[list(columns)]
    new_df.to_csv(output_path, index=False)

    return new_df


def append_result_rows(
    output_path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> pd.DataFrame:
    """Append rows to a result table without upsert."""
    output_path = ensure_parent_dir(output_path)

    df = read_existing_table(output_path, columns)
    normalized_rows = [normalize_row(row, columns) for row in rows]

    if normalized_rows:
        df = pd.concat([df, pd.DataFrame(normalized_rows)], ignore_index=True)

    df = df[list(columns)]
    df.to_csv(output_path, index=False)

    return df


def save_dataset1_main_comparison_row(
    output_path: Path | str,
    row: Mapping[str, Any],
    model_key: str = "Model",
) -> pd.DataFrame:
    """Save/update Dataset-1 main-comparison row."""
    return upsert_result_row(
        output_path=output_path,
        row=row,
        columns=DATASET1_MAIN_COLUMNS,
        key_columns=[model_key, "Split"],
    )


def save_dataset2_external_comparison_row(
    output_path: Path | str,
    row: Mapping[str, Any],
    model_key: str = "Model",
) -> pd.DataFrame:
    """Save/update Dataset-2 external-comparison row."""
    return upsert_result_row(
        output_path=output_path,
        row=row,
        columns=DATASET2_EXTERNAL_COLUMNS,
        key_columns=[model_key],
    )


def save_dataset3_online_case_row(
    output_path: Path | str,
    row: Mapping[str, Any],
    method_key: str = "Method",
) -> pd.DataFrame:
    """Save/update Dataset-3 online case-study row."""
    return upsert_result_row(
        output_path=output_path,
        row=row,
        columns=DATASET3_ONLINE_CASE_COLUMNS,
        key_columns=[method_key],
    )


def save_sensitivity_results_table(
    output_path: Path | str,
    rows: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """
    Save Step-20 sensitivity results table.

    Unlike Dataset-1/2/3 comparison tables, this overwrites the full table
    on each run so stale sensitivity rows are not mixed with the current config.
    """
    output_path = ensure_parent_dir(output_path)

    normalized_rows = [
        normalize_row(row, SENSITIVITY_COLUMNS)
        for row in rows
    ]

    df = pd.DataFrame(normalized_rows, columns=SENSITIVITY_COLUMNS)
    df.to_csv(output_path, index=False)

    return df




def metrics_to_dataset1_row(
    model_name: str,
    split: str,
    metrics: Mapping[str, Any],
    threshold: Optional[float],
    persistence: Optional[int],
    checkpoint_path: Optional[str],
    notes: str = "",
) -> Dict[str, Any]:
    """Convert metric payload to Dataset-1 table row."""
    return {
        "Model": model_name,
        "Split": split,
        "AUROC": metrics.get("auroc"),
        "AUPRC": metrics.get("auprc"),
        "F1": metrics.get("f1"),
        "Precision": metrics.get("precision"),
        "Recall": metrics.get("recall"),
        "FPR": metrics.get("fpr"),
        "Attack Detection Rate": metrics.get("attack_detection_rate"),
        "Detection Delay": metrics.get("mean_detection_delay"),
        "Runtime": metrics.get("runtime_seconds"),
        "Threshold": threshold,
        "Persistence": persistence,
        "Checkpoint": checkpoint_path,
        "Notes": notes,
    }


def metrics_to_dataset2_row(
    model_name: str,
    metrics: Mapping[str, Any],
    threshold: Optional[float],
    persistence: Optional[int],
    checkpoint_path: Optional[str],
    notes: str = "",
) -> Dict[str, Any]:
    """Convert metric payload to Dataset-2 table row."""
    return {
        "Model": model_name,
        "AUROC": metrics.get("auroc"),
        "AUPRC": metrics.get("auprc"),
        "F1": metrics.get("f1"),
        "Recall": metrics.get("recall"),
        "FPR": metrics.get("fpr"),
        "Precision": metrics.get("precision"),
        "Attack Detection Rate": metrics.get("attack_detection_rate"),
        "Detection Delay": metrics.get("mean_detection_delay"),
        "Runtime": metrics.get("runtime_seconds"),
        "Threshold": threshold,
        "Persistence": persistence,
        "Checkpoint": checkpoint_path,
        "Notes": notes,
    }


def metrics_to_dataset3_row(
    method_name: str,
    metrics: Mapping[str, Any],
    threshold: Optional[float],
    persistence: Optional[int],
    checkpoint_path: Optional[str],
    notes: str = "",
) -> Dict[str, Any]:
    """Convert metric payload to Dataset-3 online case-study row."""
    return {
        "Method": method_name,
        "False Alarms": metrics.get("false_alarms"),
        "Attack-1 Delay": metrics.get("attack_1_delay"),
        "Attack-2 Delay": metrics.get("attack_2_delay"),
        "Mean Delay": metrics.get("mean_detection_delay"),
        "Attack Detection Rate": metrics.get("attack_detection_rate"),
        "F1": metrics.get("f1"),
        "Precision": metrics.get("precision"),
        "Recall": metrics.get("recall"),
        "FPR": metrics.get("fpr"),
        "Threshold": threshold,
        "Persistence": persistence,
        "Checkpoint": checkpoint_path,
        "Notes": notes,
    }


def metrics_to_sensitivity_row(
    sensitivity_parameter: str,
    sensitivity_value: Any,
    metrics: Mapping[str, Any],
    theta: Optional[float],
    persistence: Optional[int],
    model_name: str = "Proposed",
    split: str = "Dataset-1 Test",
    is_official_selected: bool = False,
    status: str = "PASSED",
    notes: str = "",
) -> Dict[str, Any]:
    """Convert metric payload to Step-20 sensitivity-table row."""
    return {
        "Model": model_name,
        "Split": split,
        "Sensitivity Parameter": sensitivity_parameter,
        "Sensitivity Value": sensitivity_value,
        "theta": theta,
        "persistence": persistence,
        "AUROC": metrics.get("auroc"),
        "AUPRC": metrics.get("auprc"),
        "F1": metrics.get("f1"),
        "Precision": metrics.get("precision"),
        "Recall": metrics.get("recall"),
        "FPR": metrics.get("fpr"),
        "Attack Detection Rate": metrics.get("attack_detection_rate"),
        "Detection Delay": metrics.get("mean_detection_delay"),
        "False Alarms": metrics.get("false_alarms"),
        "is_official_selected": is_official_selected,
        "status": status,
        "notes": notes,
    }


def extract_primary_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract locked primary metrics from a metric payload."""
    return {
        "AUPRC": metrics.get("auprc"),
        "F1": metrics.get("f1"),
        "FPR": metrics.get("fpr"),
        "Attack Detection Rate": metrics.get("attack_detection_rate"),
        "Detection Delay": metrics.get("mean_detection_delay"),
    }


def print_primary_metric_table(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    model_key: str = "Model",
) -> None:
    """Print compact primary metric table to console."""
    print("=" * 100)
    print(title)
    print("=" * 100)

    headers = [model_key] + PRIMARY_METRIC_COLUMNS
    widths = {
        model_key: 28,
        "AUPRC": 12,
        "F1": 12,
        "FPR": 12,
        "Attack Detection Rate": 24,
        "Detection Delay": 18,
    }

    header_line = " | ".join(f"{header:<{widths[header]}}" for header in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        values = []

        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))

        line = " | ".join(
            f"{value:<{widths[header]}}"
            for value, header in zip(values, headers)
        )
        print(line)

    print("=" * 100)

def print_sensitivity_table(
    title: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Print compact Step-20 sensitivity table to console."""
    print("=" * 120)
    print(title)
    print("=" * 120)

    headers = [
        "Sensitivity Parameter",
        "Sensitivity Value",
        "theta",
        "persistence",
        "AUPRC",
        "F1",
        "FPR",
        "Attack Detection Rate",
        "Detection Delay",
        "status",
    ]

    widths = {
        "Sensitivity Parameter": 24,
        "Sensitivity Value": 20,
        "theta": 10,
        "persistence": 12,
        "AUPRC": 12,
        "F1": 12,
        "FPR": 12,
        "Attack Detection Rate": 24,
        "Detection Delay": 18,
        "status": 28,
    }

    header_line = " | ".join(f"{header:<{widths[header]}}" for header in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        values = []

        for header in headers:
            value = row.get(header)

            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))

        line = " | ".join(
            f"{value:<{widths[header]}}"
            for value, header in zip(values, headers)
        )
        print(line)

    print("=" * 120)

__all__ = [
    "DATASET1_MAIN_COLUMNS",
    "DATASET2_EXTERNAL_COLUMNS",
    "DATASET3_ONLINE_CASE_COLUMNS",
    "PRIMARY_METRIC_COLUMNS",
    "ensure_parent_dir",
    "normalize_row",
    "read_existing_table",
    "upsert_result_row",
    "append_result_rows",
    "save_dataset1_main_comparison_row",
    "save_dataset2_external_comparison_row",
    "save_dataset3_online_case_row",
    "metrics_to_dataset1_row",
    "metrics_to_dataset2_row",
    "metrics_to_dataset3_row",
    "extract_primary_metrics",
    "print_primary_metric_table",
    "SENSITIVITY_COLUMNS",
    "save_sensitivity_results_table",
    "metrics_to_sensitivity_row",
    "print_sensitivity_table",
]