"""
Raw dataset inspection utilities for the AV-GPS causal spoofing detection project.

Step 2 purpose:
- inspect all four raw AV-GPS files,
- confirm schema and label structure,
- count missing values and duplicates,
- verify EKF Detector appears only in Dataset-3,
- verify shortcut-prone columns exist but are marked as excluded from model input,
- summarize file roles.

This step does not preprocess, segment, split, train, or build xi_t.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.data.load_raw import RawDatasetBundle
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import save_json


DEFAULT_SHORTCUT_COLUMNS = [
    "Clock Date",
    "Clock Time",
    "GPS MGRS",
    "GPS HDOP",
    "GPS VDOP",
    "Satellite Count",
    "Satellite Locks",
    "Throttle (%)",
    "Longitudinal Vibration",
    "Lateral Vibration",
    "Vertical Vibration",
    "Distance To Home (m)",
    "Travelled Distance (m)",
    "X-Track Error",
    "Mission Index",
    "Distance To GCS (m)",
    "EKF Detector",
]


DEFAULT_CORE_GNSS_COLUMNS = [
    "GPS Latitude",
    "GPS Longitude",
]


DEFAULT_CORE_MOTION_COLUMNS = [
    "Velocity (m/s)",
    "Heading (deg)",
    "Yaw (deg)",
    "Yaw Rate (deg/s)",
    "Steering Angle (deg)",
]


@dataclass
class DatasetInspectionResult:
    """
    Summary of one raw dataset file.
    """

    key: str
    role: str
    path: str
    rows: int
    columns: int
    duplicate_rows: int
    total_missing_cells: int
    rows_with_missing: int
    label_column: str
    label_counts: Dict[str, int]
    label_percentages: Dict[str, float]
    has_ekf_detector: bool
    ekf_unique_values: List[str]
    shortcut_columns_present: List[str]
    shortcut_columns_missing: List[str]
    core_gnss_columns_present: List[str]
    core_motion_columns_present: List[str]
    missing_by_column_top: Dict[str, int]
    dtypes: Dict[str, str]


@dataclass
class FullInspectionReport:
    """
    Full inspection report across all raw dataset files.
    """

    dataset_results: Dict[str, DatasetInspectionResult]
    schema_consistency: Dict[str, Any]
    ekf_detector_check: Dict[str, Any]
    shortcut_column_check: Dict[str, Any]
    core_column_check: Dict[str, Any]
    final_step2_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_results": {
                key: asdict(value)
                for key, value in self.dataset_results.items()
            },
            "schema_consistency": self.schema_consistency,
            "ekf_detector_check": self.ekf_detector_check,
            "shortcut_column_check": self.shortcut_column_check,
            "core_column_check": self.core_column_check,
            "final_step2_status": self.final_step2_status,
        }


def _safe_value_counts(series: pd.Series) -> Dict[str, int]:
    """
    Return value counts with string keys for JSON safety.
    """
    counts = series.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _safe_percentages(counts: Dict[str, int], total: int) -> Dict[str, float]:
    """
    Convert counts into percentages.
    """
    if total <= 0:
        return {key: 0.0 for key in counts.keys()}

    return {
        key: round((value / total) * 100.0, 4)
        for key, value in counts.items()
    }


def _top_missing_columns(df: pd.DataFrame, top_k: int = 15) -> Dict[str, int]:
    """
    Return top missing columns as {column: missing_count}.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(top_k)
    return {str(column): int(count) for column, count in missing.items()}


def _present_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [column for column in columns if column in df.columns]


def _missing_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [column for column in columns if column not in df.columns]


def inspect_single_dataset(
    key: str,
    df: pd.DataFrame,
    role: str,
    path: Path,
    label_column: str = "Data Type",
    ekf_column: str = "EKF Detector",
    shortcut_columns: Optional[Sequence[str]] = None,
    core_gnss_columns: Optional[Sequence[str]] = None,
    core_motion_columns: Optional[Sequence[str]] = None,
) -> DatasetInspectionResult:
    """
    Inspect one raw AV-GPS dataset DataFrame.
    """
    shortcut_columns = list(shortcut_columns or DEFAULT_SHORTCUT_COLUMNS)
    core_gnss_columns = list(core_gnss_columns or DEFAULT_CORE_GNSS_COLUMNS)
    core_motion_columns = list(core_motion_columns or DEFAULT_CORE_MOTION_COLUMNS)

    rows = int(df.shape[0])
    columns = int(df.shape[1])
    duplicate_rows = int(df.duplicated().sum())
    total_missing_cells = int(df.isna().sum().sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())

    if label_column in df.columns:
        label_counts = _safe_value_counts(df[label_column])
        label_percentages = _safe_percentages(label_counts, rows)
    else:
        label_counts = {}
        label_percentages = {}

    has_ekf_detector = ekf_column in df.columns
    if has_ekf_detector:
        ekf_unique_values = [str(value) for value in sorted(df[ekf_column].dropna().unique())]
    else:
        ekf_unique_values = []

    shortcut_present = _present_columns(df, shortcut_columns)
    shortcut_missing = _missing_columns(df, shortcut_columns)

    core_gnss_present = _present_columns(df, core_gnss_columns)
    core_motion_present = _present_columns(df, core_motion_columns)

    dtypes = {str(column): str(dtype) for column, dtype in df.dtypes.items()}

    return DatasetInspectionResult(
        key=key,
        role=role,
        path=str(path),
        rows=rows,
        columns=columns,
        duplicate_rows=duplicate_rows,
        total_missing_cells=total_missing_cells,
        rows_with_missing=rows_with_missing,
        label_column=label_column,
        label_counts=label_counts,
        label_percentages=label_percentages,
        has_ekf_detector=has_ekf_detector,
        ekf_unique_values=ekf_unique_values,
        shortcut_columns_present=shortcut_present,
        shortcut_columns_missing=shortcut_missing,
        core_gnss_columns_present=core_gnss_present,
        core_motion_columns_present=core_motion_present,
        missing_by_column_top=_top_missing_columns(df, top_k=15),
        dtypes=dtypes,
    )


def check_schema_consistency(
    bundle: RawDatasetBundle,
    ekf_column: str = "EKF Detector",
    ignore_metadata_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Check column consistency across raw files.

    Dataset-3 may contain EKF Detector as an extra column.
    The added metadata columns source_key/source_file are also ignored.
    """
    ignore_metadata_columns = set(ignore_metadata_columns or ["source_key", "source_file"])

    normalized_columns: Dict[str, List[str]] = {}

    for key, df in bundle.dataframes.items():
        cols = [
            column
            for column in df.columns
            if column not in ignore_metadata_columns and column != ekf_column
        ]
        normalized_columns[key] = cols

    first_key = next(iter(normalized_columns))
    reference = normalized_columns[first_key]

    mismatches: Dict[str, Dict[str, List[str]]] = {}

    reference_set = set(reference)

    for key, cols in normalized_columns.items():
        current_set = set(cols)

        missing_vs_reference = sorted(list(reference_set - current_set))
        extra_vs_reference = sorted(list(current_set - reference_set))

        if missing_vs_reference or extra_vs_reference:
            mismatches[key] = {
                "missing_vs_reference": missing_vs_reference,
                "extra_vs_reference": extra_vs_reference,
            }

    return {
        "reference_key": first_key,
        "reference_column_count_excluding_metadata_and_ekf": len(reference),
        "all_common_schema_excluding_ekf": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def check_ekf_detector_rule(
    bundle: RawDatasetBundle,
    ekf_column: str = "EKF Detector",
) -> Dict[str, Any]:
    """
    Verify EKF Detector exists only in Dataset-3 according to specs.
    """
    per_dataset: Dict[str, Any] = {}
    all_ok = True

    for key, df in bundle.dataframes.items():
        expected = bundle.specs[key].expected_has_ekf_detector
        actual = ekf_column in df.columns
        ok = expected == actual

        if not ok:
            all_ok = False

        per_dataset[key] = {
            "expected_has_ekf_detector": expected,
            "actual_has_ekf_detector": actual,
            "ok": ok,
        }

    return {
        "all_ok": all_ok,
        "ekf_column": ekf_column,
        "per_dataset": per_dataset,
    }


def check_shortcut_columns(
    bundle: RawDatasetBundle,
    shortcut_columns: Sequence[str],
) -> Dict[str, Any]:
    """
    Check shortcut-prone columns are present and clearly marked.

    This does not remove them yet. It only confirms their presence for Step 2.
    Removal/exclusion happens later in preprocessing/clean_columns.py.
    """
    per_dataset: Dict[str, Any] = {}

    for key, df in bundle.dataframes.items():
        present = _present_columns(df, shortcut_columns)
        missing = _missing_columns(df, shortcut_columns)

        per_dataset[key] = {
            "present": present,
            "missing": missing,
            "present_count": len(present),
            "missing_count": len(missing),
        }

    return {
        "note": (
            "Shortcut-prone columns are inspected here only. "
            "They must not be used as direct model inputs later."
        ),
        "shortcut_columns": list(shortcut_columns),
        "per_dataset": per_dataset,
    }


def check_core_columns(
    bundle: RawDatasetBundle,
    core_gnss_columns: Sequence[str],
    core_motion_columns: Sequence[str],
) -> Dict[str, Any]:
    """
    Check whether core GNSS and motion columns needed for our methodology exist.
    """
    per_dataset: Dict[str, Any] = {}
    all_ok = True

    required = list(core_gnss_columns) + list(core_motion_columns)

    for key, df in bundle.dataframes.items():
        present = _present_columns(df, required)
        missing = _missing_columns(df, required)
        ok = len(missing) == 0

        if not ok:
            all_ok = False

        per_dataset[key] = {
            "required": required,
            "present": present,
            "missing": missing,
            "ok": ok,
        }

    return {
        "all_ok": all_ok,
        "core_gnss_columns": list(core_gnss_columns),
        "core_motion_columns": list(core_motion_columns),
        "per_dataset": per_dataset,
    }


def inspect_raw_datasets(
    bundle: RawDatasetBundle,
    config: Mapping[str, Any],
) -> FullInspectionReport:
    """
    Inspect all loaded raw AV-GPS datasets.
    """
    label_column = str(get_by_path(config, "dataset.label_column", "Data Type"))
    ekf_column = str(get_by_path(config, "dataset.ekf_column", "EKF Detector"))

    shortcut_columns = get_by_path(
        config,
        "dataset.shortcut_columns",
        DEFAULT_SHORTCUT_COLUMNS,
    )
    core_gnss_columns = get_by_path(
        config,
        "dataset.core_gnss_columns",
        DEFAULT_CORE_GNSS_COLUMNS,
    )
    core_motion_columns = get_by_path(
        config,
        "dataset.core_motion_columns",
        DEFAULT_CORE_MOTION_COLUMNS,
    )

    dataset_results: Dict[str, DatasetInspectionResult] = {}

    for key, df in bundle.dataframes.items():
        dataset_results[key] = inspect_single_dataset(
            key=key,
            df=df,
            role=bundle.specs[key].role,
            path=bundle.paths[key],
            label_column=label_column,
            ekf_column=ekf_column,
            shortcut_columns=shortcut_columns,
            core_gnss_columns=core_gnss_columns,
            core_motion_columns=core_motion_columns,
        )

    schema_consistency = check_schema_consistency(
        bundle=bundle,
        ekf_column=ekf_column,
    )

    ekf_detector_check = check_ekf_detector_rule(
        bundle=bundle,
        ekf_column=ekf_column,
    )

    shortcut_column_check = check_shortcut_columns(
        bundle=bundle,
        shortcut_columns=shortcut_columns,
    )

    core_column_check = check_core_columns(
        bundle=bundle,
        core_gnss_columns=core_gnss_columns,
        core_motion_columns=core_motion_columns,
    )

    all_ok = (
        schema_consistency["all_common_schema_excluding_ekf"]
        and ekf_detector_check["all_ok"]
        and core_column_check["all_ok"]
    )

    status = "PASSED" if all_ok else "WARNING_CHECK_REPORT"

    return FullInspectionReport(
        dataset_results=dataset_results,
        schema_consistency=schema_consistency,
        ekf_detector_check=ekf_detector_check,
        shortcut_column_check=shortcut_column_check,
        core_column_check=core_column_check,
        final_step2_status=status,
    )


def print_single_dataset_inspection(result: DatasetInspectionResult) -> None:
    """
    Print one dataset inspection summary.
    """
    print("-" * 100)
    print(f"Dataset key          : {result.key}")
    print(f"Role                 : {result.role}")
    print(f"Path                 : {result.path}")
    print(f"Rows                 : {result.rows}")
    print(f"Columns              : {result.columns}")
    print(f"Duplicate rows       : {result.duplicate_rows}")
    print(f"Total missing cells  : {result.total_missing_cells}")
    print(f"Rows with missing    : {result.rows_with_missing}")
    print(f"Label column         : {result.label_column}")
    print(f"Label counts         : {result.label_counts}")
    print(f"Label percentages    : {result.label_percentages}")
    print(f"Has EKF Detector     : {result.has_ekf_detector}")
    print(f"EKF unique values    : {result.ekf_unique_values}")
    print(f"Shortcut present     : {len(result.shortcut_columns_present)} columns")
    print(f"Core GNSS present    : {result.core_gnss_columns_present}")
    print(f"Core motion present  : {result.core_motion_columns_present}")
    print(f"Top missing columns  : {result.missing_by_column_top}")


def print_full_inspection_report(report: FullInspectionReport) -> None:
    """
    Print full Step 2 inspection report to console.
    """
    print("=" * 100)
    print("STEP 2 RAW DATASET INSPECTION REPORT")
    print("=" * 100)

    for result in report.dataset_results.values():
        print_single_dataset_inspection(result)

    print("-" * 100)
    print("SCHEMA CONSISTENCY")
    print(report.schema_consistency)

    print("-" * 100)
    print("EKF DETECTOR CHECK")
    print(report.ekf_detector_check)

    print("-" * 100)
    print("CORE COLUMN CHECK")
    print(report.core_column_check)

    print("-" * 100)
    print("SHORTCUT COLUMN CHECK")
    print("Shortcut columns are present for inspection, but must not be direct model inputs.")
    for key, item in report.shortcut_column_check["per_dataset"].items():
        print(
            f"{key}: present={item['present_count']}, "
            f"missing={item['missing_count']}"
        )

    print("-" * 100)
    print(f"Final Step 2 status: {report.final_step2_status}")
    print("=" * 100)


def save_inspection_report(
    report: FullInspectionReport,
    config: Mapping[str, Any],
    file_name: str = "step2_raw_dataset_inspection.json",
) -> Path:
    """
    Save inspection report as JSON.

    This is an inspection output only, not processed training data.
    """
    output_dir_value = get_by_path(
        config,
        "paths.tables_dir",
        default="results/tables",
    )
    output_dir = resolve_project_path(config, output_dir_value)
    output_path = output_dir / file_name

    save_json(report.to_dict(), output_path, indent=2)
    return output_path


def run_raw_dataset_inspection(
    bundle: RawDatasetBundle,
    config: Mapping[str, Any],
    save_report: bool = True,
) -> FullInspectionReport:
    """
    Main Step 2 inspection function.

    Loads already happen in load_raw.py. This function inspects the loaded bundle.
    """
    report = inspect_raw_datasets(bundle=bundle, config=config)

    print_full_inspection_report(report)

    if save_report:
        output_path = save_inspection_report(report, config=config)
        print(f"Saved Step 2 inspection JSON: {output_path}")

    return report