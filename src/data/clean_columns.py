"""
Shortcut-column exclusion and clean causal column selection.

Step 5 purpose:
- make sure this project does not become raw-feature tabular classification,
- remove shortcut-prone raw columns from cleaned intermediate files,
- keep only metadata, labels, segment information, preliminary validity columns,
  and causal source columns required for GNSS-motion evidence construction,
- save cleaned intermediate CSV files for later residual/evidence building.

Important:
The cleaned files are NOT final model inputs.
They are only the safe causal source files used to construct xi_t.

Final proposed model and all baselines must later use reconstructed evidence xi_t,
not raw shortcut columns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_csv, save_json


DEFAULT_DATASET_KEYS = [
    "dataset1",
    "dataset1_normal",
    "dataset2",
    "dataset3",
]


DEFAULT_SEGMENTED_FILES = {
    "dataset1": "dataset1_segmented.csv",
    "dataset1_normal": "dataset1_normal_segmented.csv",
    "dataset2": "dataset2_segmented.csv",
    "dataset3": "dataset3_segmented.csv",
}


DEFAULT_CLEANED_FILES = {
    "dataset1": "dataset1_cleaned.csv",
    "dataset1_normal": "dataset1_normal_cleaned.csv",
    "dataset2": "dataset2_cleaned.csv",
    "dataset3": "dataset3_cleaned.csv",
}


DEFAULT_EVIDENCE_SOURCE_COLUMNS = [
    "GPS Latitude",
    "GPS Longitude",
    "Velocity (m/s)",
    "Heading (deg)",
    "Yaw (deg)",
    "Yaw Rate (deg/s)",
    "Steering Angle (deg)",
]


DEFAULT_OPTIONAL_EVIDENCE_SOURCE_COLUMNS = [
    "Longitudinal Velocity (m/s)",
    "Lateral Velocity (m/s)",
]


DEFAULT_FORBIDDEN_SHORTCUT_COLUMNS = [
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


DEFAULT_METADATA_COLUMNS = [
    "raw_row_index",
    "row_order_in_source",
    "row_index_original",
    "source_key",
    "source_file",
    "dataset_role",
]


DEFAULT_SEGMENT_COLUMNS = [
    "segment_id",
    "segment_index",
    "within_segment_index",
]


DEFAULT_TRANSITION_VALIDITY_COLUMNS = [
    "delta_t_seconds",
    "valid_transition_prelim",
    "nu_prelim",
    "invalid_transition_reason",
    "is_segment_start",
    "crosses_segment_boundary",
    "segment_boundary",
    "segment_boundary_reason",
    "normal_to_attack_transition",
    "attack_to_normal_transition",
]


@dataclass
class CleanColumnPolicy:
    """
    Column policy used by Step 5.
    """

    label_column: str
    required_evidence_source_columns: List[str]
    optional_evidence_source_columns: List[str]
    metadata_columns: List[str]
    segment_columns: List[str]
    transition_validity_columns: List[str]
    forbidden_shortcut_columns: List[str]
    keep_only_policy_columns: bool
    allow_missing_optional_columns: bool


@dataclass
class CleanedDatasetSummary:
    """
    Summary for one cleaned dataset.
    """

    dataset_key: str
    input_path: str
    output_path: str
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    kept_columns: List[str]
    removed_forbidden_columns: List[str]
    removed_other_columns: List[str]
    required_evidence_columns_present: List[str]
    required_evidence_columns_missing: List[str]
    optional_evidence_columns_present: List[str]
    optional_evidence_columns_missing: List[str]
    forbidden_columns_remaining: List[str]
    final_status: str


@dataclass
class FullCleanColumnReport:
    """
    Full Step-5 report.
    """

    dataset_summaries: Dict[str, CleanedDatasetSummary]
    clean_column_policy: Dict[str, Any]
    final_step5_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_summaries": {
                key: asdict(value)
                for key, value in self.dataset_summaries.items()
            },
            "clean_column_policy": self.clean_column_policy,
            "final_step5_status": self.final_step5_status,
        }


def get_clean_column_policy(config: Mapping[str, Any]) -> CleanColumnPolicy:
    """
    Read Step-5 column policy from config, with safe defaults.
    """
    label_column = str(get_by_path(config, "dataset.label_column", "Data Type"))

    required_evidence = list(
        get_by_path(
            config,
            "dataset.evidence_source_columns",
            DEFAULT_EVIDENCE_SOURCE_COLUMNS,
        )
    )

    optional_evidence = list(
        get_by_path(
            config,
            "dataset.optional_evidence_source_columns",
            get_by_path(
                config,
                "dataset.optional_motion_columns",
                DEFAULT_OPTIONAL_EVIDENCE_SOURCE_COLUMNS,
            ),
        )
    )

    forbidden = list(
        get_by_path(
            config,
            "dataset.shortcut_columns",
            DEFAULT_FORBIDDEN_SHORTCUT_COLUMNS,
        )
    )

    metadata_columns = list(
        get_by_path(
            config,
            "preprocessing.clean_columns.metadata_columns",
            DEFAULT_METADATA_COLUMNS,
        )
    )

    segment_columns = list(
        get_by_path(
            config,
            "preprocessing.clean_columns.segment_columns",
            DEFAULT_SEGMENT_COLUMNS,
        )
    )

    transition_columns = list(
        get_by_path(
            config,
            "preprocessing.clean_columns.transition_validity_columns",
            DEFAULT_TRANSITION_VALIDITY_COLUMNS,
        )
    )

    keep_only_policy_columns = bool(
        get_by_path(config, "preprocessing.clean_columns.keep_only_policy_columns", True)
    )

    allow_missing_optional = bool(
        get_by_path(config, "preprocessing.clean_columns.allow_missing_optional_columns", True)
    )

    return CleanColumnPolicy(
        label_column=label_column,
        required_evidence_source_columns=required_evidence,
        optional_evidence_source_columns=optional_evidence,
        metadata_columns=metadata_columns,
        segment_columns=segment_columns,
        transition_validity_columns=transition_columns,
        forbidden_shortcut_columns=forbidden,
        keep_only_policy_columns=keep_only_policy_columns,
        allow_missing_optional_columns=allow_missing_optional,
    )


def get_interim_dir(config: Mapping[str, Any]) -> Path:
    """
    Resolve data/interim directory.
    """
    value = get_by_path(config, "paths.interim_data_dir", "data/interim")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_step5_summary_path(config: Mapping[str, Any]) -> Path:
    """
    Resolve Step-5 summary JSON path.
    """
    value = get_by_path(
        config,
        "paths.step5_clean_columns_json",
        "results/tables/step5_clean_columns_summary.json",
    )
    return resolve_project_path(config, value)


def get_segmented_file_path(config: Mapping[str, Any], dataset_key: str) -> Path:
    """
    Resolve segmented input file path.
    """
    interim_dir = get_interim_dir(config)

    file_name = get_by_path(
        config,
        f"dataset.segmented_files.{dataset_key}",
        DEFAULT_SEGMENTED_FILES.get(dataset_key, f"{dataset_key}_segmented.csv"),
    )

    return (interim_dir / str(file_name)).resolve()


def get_cleaned_file_path(config: Mapping[str, Any], dataset_key: str) -> Path:
    """
    Resolve cleaned output file path.
    """
    interim_dir = get_interim_dir(config)

    file_name = get_by_path(
        config,
        f"dataset.cleaned_files.{dataset_key}",
        DEFAULT_CLEANED_FILES.get(dataset_key, f"{dataset_key}_cleaned.csv"),
    )

    return (interim_dir / str(file_name)).resolve()


def load_segmented_for_cleaning(config: Mapping[str, Any], dataset_key: str) -> pd.DataFrame:
    """
    Load one segmented dataset from data/interim/.
    """
    path = get_segmented_file_path(config, dataset_key)

    if not path.exists():
        raise FileNotFoundError(
            f"Segmented file not found for {dataset_key}: {path}\n"
            "Run Step 3 first."
        )

    return pd.read_csv(path, low_memory=False)


def _present_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """
    Return columns present in DataFrame.
    """
    return [column for column in columns if column in df.columns]


def _missing_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """
    Return columns missing from DataFrame.
    """
    return [column for column in columns if column not in df.columns]


def build_keep_columns(
    df: pd.DataFrame,
    policy: CleanColumnPolicy,
) -> List[str]:
    """
    Build final kept-column list.

    Kept columns include:
    - metadata columns,
    - segment columns,
    - label column,
    - required evidence source columns,
    - optional evidence source columns if present,
    - transition/validity columns.

    Forbidden shortcut columns are excluded even if present.
    """
    candidate_columns: List[str] = []

    candidate_columns.extend(policy.metadata_columns)
    candidate_columns.extend(policy.segment_columns)
    candidate_columns.append(policy.label_column)
    candidate_columns.extend(policy.required_evidence_source_columns)
    candidate_columns.extend(policy.optional_evidence_source_columns)
    candidate_columns.extend(policy.transition_validity_columns)

    forbidden_set = set(policy.forbidden_shortcut_columns)

    kept: List[str] = []
    seen = set()

    for column in candidate_columns:
        if column in seen:
            continue

        seen.add(column)

        if column not in df.columns:
            continue

        if column in forbidden_set:
            continue

        kept.append(column)

    return kept


def validate_required_evidence_columns(
    df: pd.DataFrame,
    policy: CleanColumnPolicy,
    dataset_key: str,
) -> None:
    """
    Ensure required evidence-source columns exist.
    """
    missing = _missing_columns(df, policy.required_evidence_source_columns)

    if missing:
        raise KeyError(
            f"{dataset_key} is missing required evidence source columns: {missing}. "
            "These are needed for GNSS-motion residual evidence construction."
        )


def clean_single_dataset_columns(
    df: pd.DataFrame,
    dataset_key: str,
    input_path: Path,
    output_path: Path,
    policy: CleanColumnPolicy,
) -> tuple[pd.DataFrame, CleanedDatasetSummary]:
    """
    Clean one segmented dataset.

    This removes shortcut columns and keeps only safe causal source columns
    plus metadata/labels/segment/validity columns.
    """
    input_rows = int(df.shape[0])
    input_columns = int(df.shape[1])

    validate_required_evidence_columns(df, policy, dataset_key)

    required_present = _present_columns(df, policy.required_evidence_source_columns)
    required_missing = _missing_columns(df, policy.required_evidence_source_columns)

    optional_present = _present_columns(df, policy.optional_evidence_source_columns)
    optional_missing = _missing_columns(df, policy.optional_evidence_source_columns)

    if optional_missing and not policy.allow_missing_optional_columns:
        raise KeyError(
            f"{dataset_key} is missing optional evidence columns, but config "
            f"does not allow missing optional columns: {optional_missing}"
        )

    kept_columns = build_keep_columns(df, policy)

    if policy.keep_only_policy_columns:
        cleaned = df.loc[:, kept_columns].copy()
    else:
        forbidden_set = set(policy.forbidden_shortcut_columns)
        cleaned = df.drop(
            columns=[col for col in df.columns if col in forbidden_set],
            errors="ignore",
        ).copy()
        kept_columns = list(cleaned.columns)

    removed_forbidden = [
        col for col in policy.forbidden_shortcut_columns
        if col in df.columns and col not in cleaned.columns
    ]

    removed_other = [
        col for col in df.columns
        if col not in cleaned.columns and col not in removed_forbidden
    ]

    forbidden_remaining = [
        col for col in policy.forbidden_shortcut_columns
        if col in cleaned.columns
    ]

    final_status = "PASSED" if len(forbidden_remaining) == 0 else "FAILED_SHORTCUT_REMAINING"

    summary = CleanedDatasetSummary(
        dataset_key=dataset_key,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=input_rows,
        output_rows=int(cleaned.shape[0]),
        input_columns=input_columns,
        output_columns=int(cleaned.shape[1]),
        kept_columns=kept_columns,
        removed_forbidden_columns=removed_forbidden,
        removed_other_columns=removed_other,
        required_evidence_columns_present=required_present,
        required_evidence_columns_missing=required_missing,
        optional_evidence_columns_present=optional_present,
        optional_evidence_columns_missing=optional_missing,
        forbidden_columns_remaining=forbidden_remaining,
        final_status=final_status,
    )

    return cleaned, summary


def print_cleaned_dataset_summary(summary: CleanedDatasetSummary) -> None:
    """
    Print one dataset cleaning summary.
    """
    print("=" * 100)
    print(f"STEP 5 CLEAN COLUMN SUMMARY | {summary.dataset_key}")
    print("=" * 100)
    print(f"Input path                       : {summary.input_path}")
    print(f"Output path                      : {summary.output_path}")
    print(f"Rows                             : {summary.input_rows} -> {summary.output_rows}")
    print(f"Columns                          : {summary.input_columns} -> {summary.output_columns}")
    print(f"Required evidence present         : {summary.required_evidence_columns_present}")
    print(f"Required evidence missing         : {summary.required_evidence_columns_missing}")
    print(f"Optional evidence present         : {summary.optional_evidence_columns_present}")
    print(f"Optional evidence missing         : {summary.optional_evidence_columns_missing}")
    print(f"Removed forbidden shortcut cols   : {summary.removed_forbidden_columns}")
    print(f"Forbidden columns remaining       : {summary.forbidden_columns_remaining}")
    print(f"Final status                      : {summary.final_status}")
    print("=" * 100)


def print_full_clean_column_report(report: FullCleanColumnReport) -> None:
    """
    Print full Step-5 report.
    """
    print("=" * 100)
    print("STEP 5 SHORTCUT-COLUMN EXCLUSION REPORT")
    print("=" * 100)

    for summary in report.dataset_summaries.values():
        print_cleaned_dataset_summary(summary)

    print("-" * 100)
    print("Clean column policy:")
    print(f"Label column                       : {report.clean_column_policy['label_column']}")
    print(f"Required evidence source columns    : {report.clean_column_policy['required_evidence_source_columns']}")
    print(f"Optional evidence source columns    : {report.clean_column_policy['optional_evidence_source_columns']}")
    print(f"Forbidden shortcut columns          : {report.clean_column_policy['forbidden_shortcut_columns']}")
    print(f"Final Step 5 status                 : {report.final_step5_status}")
    print("=" * 100)


def run_shortcut_column_exclusion(
    config: Mapping[str, Any],
    dataset_keys: Optional[Sequence[str]] = None,
    save_outputs: bool = True,
) -> FullCleanColumnReport:
    """
    Main Step-5 entry point.

    Loads segmented datasets from data/interim/, removes shortcut columns,
    saves cleaned intermediate files, and saves a JSON summary report.
    """
    keys = list(dataset_keys or DEFAULT_DATASET_KEYS)
    policy = get_clean_column_policy(config)

    summaries: Dict[str, CleanedDatasetSummary] = {}

    for dataset_key in keys:
        input_path = get_segmented_file_path(config, dataset_key)
        output_path = get_cleaned_file_path(config, dataset_key)

        df = load_segmented_for_cleaning(config, dataset_key)

        cleaned, summary = clean_single_dataset_columns(
            df=df,
            dataset_key=dataset_key,
            input_path=input_path,
            output_path=output_path,
            policy=policy,
        )

        if save_outputs:
            save_csv(cleaned, output_path, index=False)

        summaries[dataset_key] = summary

    all_passed = all(summary.final_status == "PASSED" for summary in summaries.values())
    final_status = "PASSED" if all_passed else "FAILED_SHORTCUT_CHECK"

    report = FullCleanColumnReport(
        dataset_summaries=summaries,
        clean_column_policy=asdict(policy),
        final_step5_status=final_status,
    )

    if save_outputs:
        summary_path = get_step5_summary_path(config)
        save_json(report.to_dict(), summary_path, indent=2)
        print(f"Saved Step 5 clean-column JSON: {summary_path}")

    print_full_clean_column_report(report)

    if report.final_step5_status != "PASSED":
        raise RuntimeError(
            f"Step 5 failed with status: {report.final_step5_status}. "
            "Shortcut columns remain in cleaned outputs."
        )

    return report