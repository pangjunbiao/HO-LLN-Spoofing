"""
Residual construction for AV-GPS causal spoofing detection.

Step 7 purpose:
- load Step-6 physical files,
- compute residual r_t = Delta p_t^g - Delta p_t^u,
- build final residual validity nu_t by combining:
    gnss_displacement_valid
    AND motion_displacement_valid
    AND nu_prelim
    AND finite residual values
- save residual intermediate files for all datasets.

Important:
This file does not compute training covariance/statistics by itself.
Training-only normal statistics are computed in normal_statistics.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.preprocessing.validity_mask import add_final_residual_validity_mask
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_csv, save_json


DEFAULT_DATASET_KEYS = [
    "dataset1",
    "dataset1_normal",
    "dataset2",
    "dataset3",
]


DEFAULT_PHYSICAL_FILES = {
    "dataset1": "dataset1_physical.csv",
    "dataset1_normal": "dataset1_normal_physical.csv",
    "dataset2": "dataset2_physical.csv",
    "dataset3": "dataset3_physical.csv",
}


DEFAULT_RESIDUAL_FILES = {
    "dataset1": "dataset1_residual.csv",
    "dataset1_normal": "dataset1_normal_residual.csv",
    "dataset2": "dataset2_residual.csv",
    "dataset3": "dataset3_residual.csv",
}


@dataclass
class ResidualBuilderConfig:
    """Configuration for residual construction."""

    segment_column: str
    order_column: str
    label_column: str
    delta_t_column: str

    gnss_delta_east_column: str
    gnss_delta_north_column: str
    motion_delta_east_column: str
    motion_delta_north_column: str

    gnss_valid_column: str
    motion_valid_column: str
    preliminary_nu_column: str

    residual_east_column: str
    residual_north_column: str
    residual_norm_column: str
    residual_finite_column: str
    residual_valid_column: str
    residual_invalid_reason_column: str

    max_delta_seconds: float


@dataclass
class ResidualDatasetSummary:
    """Summary for one residual dataset."""

    dataset_key: str
    input_path: str
    output_path: str

    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int

    segments: int
    total_rows: int
    residual_finite_rows: int
    valid_residual_rows: int
    invalid_residual_rows: int

    valid_normal_residual_rows: int
    valid_attack_residual_rows: int

    gnss_valid_rows: int
    motion_valid_rows: int
    preliminary_nu_rows: int

    invalid_reason_counts: Dict[str, int]
    required_columns_present: List[str]
    required_columns_missing: List[str]
    output_columns_added: List[str]

    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FullResidualBuilderReport:
    """Full residual-builder report."""

    dataset_summaries: Dict[str, ResidualDatasetSummary]
    residual_builder_config: Dict[str, Any]
    final_residual_builder_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_summaries": {
                key: value.to_dict()
                for key, value in self.dataset_summaries.items()
            },
            "residual_builder_config": self.residual_builder_config,
            "final_residual_builder_status": self.final_residual_builder_status,
        }


def get_residual_builder_config(config: Mapping[str, Any]) -> ResidualBuilderConfig:
    """
    Read residual-builder configuration.
    """
    return ResidualBuilderConfig(
        segment_column=str(
            get_by_path(config, "preprocessing.residual.segment_column", "segment_id")
        ),
        order_column=str(
            get_by_path(config, "preprocessing.residual.order_column", "within_segment_index")
        ),
        label_column=str(
            get_by_path(config, "dataset.label_column", "Data Type")
        ),
        delta_t_column=str(
            get_by_path(config, "preprocessing.residual.delta_t_column", "delta_t_seconds")
        ),

        gnss_delta_east_column=str(
            get_by_path(config, "preprocessing.residual.gnss_delta_east_column", "delta_pos_g_east_m")
        ),
        gnss_delta_north_column=str(
            get_by_path(config, "preprocessing.residual.gnss_delta_north_column", "delta_pos_g_north_m")
        ),
        motion_delta_east_column=str(
            get_by_path(config, "preprocessing.residual.motion_delta_east_column", "delta_pos_u_east_m")
        ),
        motion_delta_north_column=str(
            get_by_path(config, "preprocessing.residual.motion_delta_north_column", "delta_pos_u_north_m")
        ),

        gnss_valid_column=str(
            get_by_path(config, "preprocessing.residual.gnss_valid_column", "gnss_displacement_valid")
        ),
        motion_valid_column=str(
            get_by_path(config, "preprocessing.residual.motion_valid_column", "motion_displacement_valid")
        ),
        preliminary_nu_column=str(
            get_by_path(config, "preprocessing.residual.preliminary_nu_column", "nu_prelim")
        ),

        residual_east_column=str(
            get_by_path(config, "preprocessing.residual.residual_east_column", "residual_east_m")
        ),
        residual_north_column=str(
            get_by_path(config, "preprocessing.residual.residual_north_column", "residual_north_m")
        ),
        residual_norm_column=str(
            get_by_path(config, "preprocessing.residual.residual_norm_column", "residual_norm_m")
        ),
        residual_finite_column=str(
            get_by_path(config, "preprocessing.residual.residual_finite_column", "residual_finite")
        ),
        residual_valid_column=str(
            get_by_path(config, "preprocessing.residual.residual_valid_column", "nu")
        ),
        residual_invalid_reason_column=str(
            get_by_path(config, "preprocessing.residual.residual_invalid_reason_column", "residual_invalid_reason")
        ),

        max_delta_seconds=float(
            get_by_path(config, "preprocessing.residual.max_delta_seconds", 5.0)
        ),
    )


def get_interim_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/interim directory."""
    value = get_by_path(config, "paths.interim_data_dir", "data/interim")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_step7_residual_summary_path(config: Mapping[str, Any]) -> Path:
    """Resolve residual-builder summary path."""
    value = get_by_path(
        config,
        "paths.step7_residual_summary_json",
        "results/tables/step7_residual_summary.json",
    )
    return resolve_project_path(config, value)


def get_physical_file_path(config: Mapping[str, Any], dataset_key: str) -> Path:
    """Resolve Step-6 physical input file."""
    file_name = get_by_path(
        config,
        f"dataset.physical_files.{dataset_key}",
        DEFAULT_PHYSICAL_FILES.get(dataset_key, f"{dataset_key}_physical.csv"),
    )
    return (get_interim_dir(config) / str(file_name)).resolve()


def get_residual_file_path(config: Mapping[str, Any], dataset_key: str) -> Path:
    """Resolve Step-7 residual output file."""
    file_name = get_by_path(
        config,
        f"dataset.residual_files.{dataset_key}",
        DEFAULT_RESIDUAL_FILES.get(dataset_key, f"{dataset_key}_residual.csv"),
    )
    return (get_interim_dir(config) / str(file_name)).resolve()


def load_physical_dataset_for_residuals(
    config: Mapping[str, Any],
    dataset_key: str,
) -> pd.DataFrame:
    """Load one Step-6 physical dataset."""
    path = get_physical_file_path(config, dataset_key)

    if not path.exists():
        raise FileNotFoundError(
            f"Physical file not found for {dataset_key}: {path}\n"
            "Run Step 6 first."
        )

    return pd.read_csv(path, low_memory=False)


def _present_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def _missing_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col not in df.columns]


def validate_residual_input_columns(
    df: pd.DataFrame,
    cfg: ResidualBuilderConfig,
) -> None:
    """
    Validate columns required for residual construction.
    """
    required = [
        cfg.segment_column,
        cfg.order_column,
        cfg.label_column,
        cfg.delta_t_column,
        cfg.gnss_delta_east_column,
        cfg.gnss_delta_north_column,
        cfg.motion_delta_east_column,
        cfg.motion_delta_north_column,
        cfg.gnss_valid_column,
        cfg.motion_valid_column,
        cfg.preliminary_nu_column,
    ]

    missing = _missing_columns(df, required)

    if missing:
        raise KeyError(f"Missing required residual-builder columns: {missing}")


def _count_reason_strings(series: pd.Series) -> Dict[str, int]:
    """
    Count pipe-separated invalid reason strings.
    """
    counts: Dict[str, int] = {}

    for value in series.dropna().astype(str):
        if value.strip() == "":
            continue

        for part in value.split("|"):
            part = part.strip()
            if not part:
                continue
            counts[part] = counts.get(part, 0) + 1

    return dict(sorted(counts.items()))


def add_raw_residual_columns(
    df: pd.DataFrame,
    cfg: ResidualBuilderConfig,
) -> pd.DataFrame:
    """
    Compute residual vector columns.

        r_east  = Delta p_g_east  - Delta p_u_east
        r_north = Delta p_g_north - Delta p_u_north
    """
    validate_residual_input_columns(df, cfg)

    out = df.copy()

    g_east = pd.to_numeric(out[cfg.gnss_delta_east_column], errors="coerce")
    g_north = pd.to_numeric(out[cfg.gnss_delta_north_column], errors="coerce")
    u_east = pd.to_numeric(out[cfg.motion_delta_east_column], errors="coerce")
    u_north = pd.to_numeric(out[cfg.motion_delta_north_column], errors="coerce")

    residual_east = g_east - u_east
    residual_north = g_north - u_north

    residual_finite = (
        residual_east.notna()
        & residual_north.notna()
        & np.isfinite(residual_east)
        & np.isfinite(residual_north)
    )

    residual_norm = np.sqrt(residual_east ** 2 + residual_north ** 2)

    out[cfg.residual_east_column] = residual_east
    out[cfg.residual_north_column] = residual_north
    out[cfg.residual_norm_column] = residual_norm
    out[cfg.residual_finite_column] = residual_finite.astype(int)

    return out


def add_residual_validity(
    df: pd.DataFrame,
    cfg: ResidualBuilderConfig,
) -> pd.DataFrame:
    """
    Add final residual validity nu_t.

    nu_t is stricter than Step-6 physical validity because it also includes
    Step-3 preliminary transition validity.
    """
    out = add_final_residual_validity_mask(
        df=df,
        gnss_valid_column=cfg.gnss_valid_column,
        motion_valid_column=cfg.motion_valid_column,
        preliminary_nu_column=cfg.preliminary_nu_column,
        residual_finite_column=cfg.residual_finite_column,
        delta_t_column=cfg.delta_t_column,
        output_valid_column=cfg.residual_valid_column,
        output_reason_column=cfg.residual_invalid_reason_column,
        max_delta_seconds=cfg.max_delta_seconds,
        copy=True,
    )

    return out


def build_residual_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, ResidualDatasetSummary]:
    """
    Build residual dataset and summary for one dataset.
    """
    cfg = get_residual_builder_config(config)

    required = [
        cfg.segment_column,
        cfg.order_column,
        cfg.label_column,
        cfg.delta_t_column,
        cfg.gnss_delta_east_column,
        cfg.gnss_delta_north_column,
        cfg.motion_delta_east_column,
        cfg.motion_delta_north_column,
        cfg.gnss_valid_column,
        cfg.motion_valid_column,
        cfg.preliminary_nu_column,
    ]

    required_present = _present_columns(df, required)
    required_missing = _missing_columns(df, required)

    if required_missing:
        raise KeyError(
            f"{dataset_key} is missing residual-builder columns: {required_missing}"
        )

    input_rows = int(df.shape[0])
    input_columns = int(df.shape[1])

    out = add_raw_residual_columns(df, cfg)
    out = add_residual_validity(out, cfg)

    label_col = cfg.label_column
    normal_label = int(get_by_path(config, "dataset.normal_label", 0))
    attack_label = int(get_by_path(config, "dataset.attack_label", 1))

    valid_mask = out[cfg.residual_valid_column].astype(int) == 1

    if label_col in out.columns:
        labels = out[label_col]
        valid_normal = int((valid_mask & (labels == normal_label)).sum())
        valid_attack = int((valid_mask & (labels == attack_label)).sum())
    else:
        valid_normal = 0
        valid_attack = 0

    output_columns_added = [
        cfg.residual_east_column,
        cfg.residual_north_column,
        cfg.residual_norm_column,
        cfg.residual_finite_column,
        cfg.residual_valid_column,
        cfg.residual_invalid_reason_column,
    ]

    residual_finite_count = int((out[cfg.residual_finite_column].astype(int) == 1).sum())
    valid_residual_count = int(valid_mask.sum())
    invalid_residual_count = int(len(out) - valid_residual_count)

    gnss_valid_count = int((out[cfg.gnss_valid_column].astype(int) == 1).sum())
    motion_valid_count = int((out[cfg.motion_valid_column].astype(int) == 1).sum())
    prelim_nu_count = int((out[cfg.preliminary_nu_column].astype(int) == 1).sum())

    reason_counts = _count_reason_strings(out[cfg.residual_invalid_reason_column])

    segments = int(out[cfg.segment_column].nunique())

    final_status = "PASSED"
    if valid_residual_count <= 0:
        final_status = "FAILED_NO_VALID_RESIDUALS"

    input_path = get_physical_file_path(config, dataset_key)
    output_path = get_residual_file_path(config, dataset_key)

    summary = ResidualDatasetSummary(
        dataset_key=dataset_key,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=input_rows,
        output_rows=int(out.shape[0]),
        input_columns=input_columns,
        output_columns=int(out.shape[1]),
        segments=segments,
        total_rows=int(len(out)),
        residual_finite_rows=residual_finite_count,
        valid_residual_rows=valid_residual_count,
        invalid_residual_rows=invalid_residual_count,
        valid_normal_residual_rows=valid_normal,
        valid_attack_residual_rows=valid_attack,
        gnss_valid_rows=gnss_valid_count,
        motion_valid_rows=motion_valid_count,
        preliminary_nu_rows=prelim_nu_count,
        invalid_reason_counts=reason_counts,
        required_columns_present=required_present,
        required_columns_missing=required_missing,
        output_columns_added=output_columns_added,
        final_status=final_status,
    )

    return out, summary


def process_single_dataset_residual_step7(
    config: Mapping[str, Any],
    dataset_key: str,
    save_outputs: bool = True,
) -> ResidualDatasetSummary:
    """
    Process one physical dataset into a residual dataset.
    """
    input_path = get_physical_file_path(config, dataset_key)
    output_path = get_residual_file_path(config, dataset_key)

    df = load_physical_dataset_for_residuals(config, dataset_key)

    out, summary = build_residual_dataset(
        df=df,
        dataset_key=dataset_key,
        config=config,
    )

    if save_outputs:
        save_csv(out, output_path, index=False)

    return summary


def print_residual_dataset_summary(summary: ResidualDatasetSummary) -> None:
    """Print one residual dataset summary."""
    print("=" * 100)
    print(f"STEP 7 RESIDUAL SUMMARY | {summary.dataset_key}")
    print("=" * 100)
    print(f"Input path                         : {summary.input_path}")
    print(f"Output path                        : {summary.output_path}")
    print(f"Rows                               : {summary.input_rows} -> {summary.output_rows}")
    print(f"Columns                            : {summary.input_columns} -> {summary.output_columns}")
    print(f"Segments                            : {summary.segments}")
    print(f"Required residual columns missing   : {summary.required_columns_missing}")
    print(f"GNSS valid rows                     : {summary.gnss_valid_rows}")
    print(f"Motion valid rows                   : {summary.motion_valid_rows}")
    print(f"Preliminary nu rows                 : {summary.preliminary_nu_rows}")
    print(f"Residual finite rows                : {summary.residual_finite_rows}")
    print(f"Final valid residual rows nu=1      : {summary.valid_residual_rows}")
    print(f"Invalid residual rows nu=0          : {summary.invalid_residual_rows}")
    print(f"Valid normal residual rows          : {summary.valid_normal_residual_rows}")
    print(f"Valid attack residual rows          : {summary.valid_attack_residual_rows}")
    print(f"Invalid reason counts               : {summary.invalid_reason_counts}")
    print(f"Final status                        : {summary.final_status}")
    print("=" * 100)


def print_full_residual_builder_report(report: FullResidualBuilderReport) -> None:
    """Print full residual-builder report."""
    print("=" * 100)
    print("STEP 7 RESIDUAL BUILDER REPORT")
    print("=" * 100)

    for summary in report.dataset_summaries.values():
        print_residual_dataset_summary(summary)

    print("-" * 100)
    print(f"Residual builder config             : {report.residual_builder_config}")
    print(f"Final residual builder status        : {report.final_residual_builder_status}")
    print("=" * 100)


def run_residual_builder_step(
    config: Mapping[str, Any],
    dataset_keys: Optional[Sequence[str]] = None,
    save_outputs: bool = True,
) -> FullResidualBuilderReport:
    """
    Main residual-builder entry point.

    Saves:
    - data/interim/dataset1_residual.csv
    - data/interim/dataset1_normal_residual.csv
    - data/interim/dataset2_residual.csv
    - data/interim/dataset3_residual.csv
    - results/tables/step7_residual_summary.json
    """
    keys = list(dataset_keys or DEFAULT_DATASET_KEYS)

    summaries: Dict[str, ResidualDatasetSummary] = {}

    for dataset_key in keys:
        summary = process_single_dataset_residual_step7(
            config=config,
            dataset_key=dataset_key,
            save_outputs=save_outputs,
        )
        summaries[dataset_key] = summary

    all_passed = all(summary.final_status == "PASSED" for summary in summaries.values())
    final_status = "PASSED" if all_passed else "FAILED_RESIDUAL_BUILDER_CHECK"

    report = FullResidualBuilderReport(
        dataset_summaries=summaries,
        residual_builder_config=asdict(get_residual_builder_config(config)),
        final_residual_builder_status=final_status,
    )

    if save_outputs:
        summary_path = get_step7_residual_summary_path(config)
        save_json(report.to_dict(), summary_path, indent=2)
        print(f"Saved Step 7 residual summary JSON: {summary_path}")

    print_full_residual_builder_report(report)

    if report.final_residual_builder_status != "PASSED":
        raise RuntimeError(
            f"Step 7 residual builder failed with status: "
            f"{report.final_residual_builder_status}"
        )

    return report