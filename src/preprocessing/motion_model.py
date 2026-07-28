"""
Causal onboard-motion displacement model for AV-GPS spoofing detection.

Step 6 purpose:
- compute onboard-motion displacement Delta p_t^u,
- combine with coordinate_transform output,
- save physical intermediate files ready for residual construction.

Method:
    Delta p_t^u = K(u_{t-1}, u_t, delta_t)

Default K:
- causal trapezoidal integration of velocity in the local east/north frame,
- uses only current and previous rows inside the same segment,
- heading convention: degrees clockwise from north,
      east_velocity  = speed * sin(heading)
      north_velocity = speed * cos(heading)

Optional:
- if longitudinal/lateral velocity components are used, body-frame velocities
  are rotated into east/north using the same heading convention.

This file does not create final model inputs.
It prepares physical quantities needed later for residual evidence xi_t.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.coordinate_transform import (
    CoordinateTransformSummary,
    add_coordinate_transform_features,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_csv, save_json


DEFAULT_DATASET_KEYS = [
    "dataset1",
    "dataset1_normal",
    "dataset2",
    "dataset3",
]


DEFAULT_CLEANED_FILES = {
    "dataset1": "dataset1_cleaned.csv",
    "dataset1_normal": "dataset1_normal_cleaned.csv",
    "dataset2": "dataset2_cleaned.csv",
    "dataset3": "dataset3_cleaned.csv",
}


DEFAULT_PHYSICAL_FILES = {
    "dataset1": "dataset1_physical.csv",
    "dataset1_normal": "dataset1_normal_physical.csv",
    "dataset2": "dataset2_physical.csv",
    "dataset3": "dataset3_physical.csv",
}


@dataclass
class MotionModelConfig:
    """Configuration for onboard motion integration."""

    segment_column: str
    order_column: str
    delta_t_column: str

    speed_column: str
    heading_column: str
    yaw_column: str
    yaw_rate_column: str
    steering_angle_column: str

    longitudinal_velocity_column: str
    lateral_velocity_column: str

    heading_source: str
    prefer_velocity_components: bool
    max_abs_speed_mps: float
    max_delta_seconds: float


@dataclass
class MotionModelSummary:
    """Summary for one dataset after motion model."""

    dataset_key: str
    rows: int
    segments: int
    velocity_rows_valid: int
    velocity_rows_invalid: int
    heading_rows_valid: int
    heading_rows_invalid: int
    valid_motion_displacement_rows: int
    invalid_motion_displacement_rows: int
    used_velocity_component_rows: int
    used_speed_heading_rows: int
    output_columns_added: List[str]
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicalDatasetSummary:
    """Full physical Step-6 summary for one dataset."""

    dataset_key: str
    input_path: str
    output_path: str
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    coordinate_summary: Dict[str, Any]
    motion_summary: Dict[str, Any]
    required_columns_present: List[str]
    required_columns_missing: List[str]
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FullPhysicalStep6Report:
    """Full Step-6 report."""

    dataset_summaries: Dict[str, PhysicalDatasetSummary]
    motion_config: Dict[str, Any]
    final_step6_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_summaries": {
                key: value.to_dict()
                for key, value in self.dataset_summaries.items()
            },
            "motion_config": self.motion_config,
            "final_step6_status": self.final_step6_status,
        }


def get_motion_model_config(config: Mapping[str, Any]) -> MotionModelConfig:
    """
    Read motion-model configuration from config.
    """
    return MotionModelConfig(
        segment_column=str(get_by_path(config, "preprocessing.motion_model.segment_column", "segment_id")),
        order_column=str(get_by_path(config, "preprocessing.motion_model.order_column", "within_segment_index")),
        delta_t_column=str(get_by_path(config, "preprocessing.motion_model.delta_t_column", "delta_t_seconds")),

        speed_column=str(get_by_path(config, "preprocessing.motion_model.speed_column", "Velocity (m/s)")),
        heading_column=str(get_by_path(config, "preprocessing.motion_model.heading_column", "Heading (deg)")),
        yaw_column=str(get_by_path(config, "preprocessing.motion_model.yaw_column", "Yaw (deg)")),
        yaw_rate_column=str(get_by_path(config, "preprocessing.motion_model.yaw_rate_column", "Yaw Rate (deg/s)")),
        steering_angle_column=str(get_by_path(config, "preprocessing.motion_model.steering_angle_column", "Steering Angle (deg)")),

        longitudinal_velocity_column=str(
            get_by_path(config, "preprocessing.motion_model.longitudinal_velocity_column", "Longitudinal Velocity (m/s)")
        ),
        lateral_velocity_column=str(
            get_by_path(config, "preprocessing.motion_model.lateral_velocity_column", "Lateral Velocity (m/s)")
        ),

        heading_source=str(get_by_path(config, "preprocessing.motion_model.heading_source", "heading_then_yaw")),
        prefer_velocity_components=bool(
            get_by_path(config, "preprocessing.motion_model.prefer_velocity_components", False)
        ),
        max_abs_speed_mps=float(get_by_path(config, "preprocessing.motion_model.max_abs_speed_mps", 80.0)),
        max_delta_seconds=float(get_by_path(config, "preprocessing.motion_model.max_delta_seconds", 5.0)),
    )


def get_interim_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/interim directory."""
    value = get_by_path(config, "paths.interim_data_dir", "data/interim")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_step6_summary_path(config: Mapping[str, Any]) -> Path:
    """Resolve Step-6 summary JSON path."""
    value = get_by_path(
        config,
        "paths.step6_physical_model_json",
        "results/tables/step6_coordinate_motion_summary.json",
    )
    return resolve_project_path(config, value)


def get_cleaned_file_path(config: Mapping[str, Any], dataset_key: str) -> Path:
    """Resolve cleaned input file."""
    file_name = get_by_path(
        config,
        f"dataset.cleaned_files.{dataset_key}",
        DEFAULT_CLEANED_FILES.get(dataset_key, f"{dataset_key}_cleaned.csv"),
    )
    return (get_interim_dir(config) / str(file_name)).resolve()


def get_physical_file_path(config: Mapping[str, Any], dataset_key: str) -> Path:
    """Resolve physical output file."""
    file_name = get_by_path(
        config,
        f"dataset.physical_files.{dataset_key}",
        DEFAULT_PHYSICAL_FILES.get(dataset_key, f"{dataset_key}_physical.csv"),
    )
    return (get_interim_dir(config) / str(file_name)).resolve()


def load_cleaned_dataset_for_physical_model(
    config: Mapping[str, Any],
    dataset_key: str,
) -> pd.DataFrame:
    """Load cleaned intermediate dataset."""
    path = get_cleaned_file_path(config, dataset_key)

    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned file not found for {dataset_key}: {path}\n"
            "Run Step 5 first."
        )

    return pd.read_csv(path, low_memory=False)


def _present_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def _missing_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col not in df.columns]


def validate_motion_model_columns(
    df: pd.DataFrame,
    cfg: MotionModelConfig,
) -> None:
    """
    Validate minimum required columns for motion displacement.
    """
    required = [
        cfg.segment_column,
        cfg.delta_t_column,
        cfg.speed_column,
    ]

    missing = _missing_columns(df, required)

    if missing:
        raise KeyError(f"Missing required motion-model columns: {missing}")

    has_heading = cfg.heading_column in df.columns
    has_yaw = cfg.yaw_column in df.columns

    if not has_heading and not has_yaw:
        raise KeyError(
            f"Motion model needs at least one direction column: "
            f"'{cfg.heading_column}' or '{cfg.yaw_column}'."
        )


def choose_direction_degrees(
    df: pd.DataFrame,
    cfg: MotionModelConfig,
) -> pd.Series:
    """
    Choose causal direction angle in degrees.

    Supported heading_source:
    - heading_then_yaw
    - yaw_then_heading
    - heading
    - yaw
    """
    heading = (
        pd.to_numeric(df[cfg.heading_column], errors="coerce")
        if cfg.heading_column in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    yaw = (
        pd.to_numeric(df[cfg.yaw_column], errors="coerce")
        if cfg.yaw_column in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    source = cfg.heading_source.lower().strip()

    if source == "heading":
        return heading

    if source == "yaw":
        return yaw

    if source == "yaw_then_heading":
        return yaw.where(yaw.notna(), heading)

    return heading.where(heading.notna(), yaw)


def add_motion_velocity_components(
    df: pd.DataFrame,
    cfg: MotionModelConfig,
) -> pd.DataFrame:
    """
    Add east/north velocity components from onboard motion observations.

    New columns:
    - motion_speed_mps
    - motion_direction_deg
    - motion_direction_rad
    - motion_velocity_valid
    - motion_used_velocity_components
    - motion_east_velocity_mps
    - motion_north_velocity_mps
    """
    validate_motion_model_columns(df, cfg)

    out = df.copy()

    speed = pd.to_numeric(out[cfg.speed_column], errors="coerce")
    direction_deg = choose_direction_degrees(out, cfg)
    direction_rad = np.deg2rad(direction_deg)

    speed_valid = (
        speed.notna()
        & np.isfinite(speed)
        & (speed.abs() <= cfg.max_abs_speed_mps)
    )

    direction_valid = (
        direction_deg.notna()
        & np.isfinite(direction_deg)
    )

    use_components = pd.Series(False, index=out.index)

    if (
        cfg.prefer_velocity_components
        and cfg.longitudinal_velocity_column in out.columns
        and cfg.lateral_velocity_column in out.columns
    ):
        longitudinal = pd.to_numeric(out[cfg.longitudinal_velocity_column], errors="coerce")
        lateral = pd.to_numeric(out[cfg.lateral_velocity_column], errors="coerce")

        component_valid = (
            longitudinal.notna()
            & lateral.notna()
            & np.isfinite(longitudinal)
            & np.isfinite(lateral)
            & (longitudinal.abs() <= cfg.max_abs_speed_mps)
            & (lateral.abs() <= cfg.max_abs_speed_mps)
            & direction_valid
        )

        use_components = component_valid

        east_from_components = longitudinal * np.sin(direction_rad) + lateral * np.cos(direction_rad)
        north_from_components = longitudinal * np.cos(direction_rad) - lateral * np.sin(direction_rad)
    else:
        east_from_components = pd.Series(np.nan, index=out.index)
        north_from_components = pd.Series(np.nan, index=out.index)

    east_from_speed = speed * np.sin(direction_rad)
    north_from_speed = speed * np.cos(direction_rad)

    motion_east = east_from_speed.where(~use_components, east_from_components)
    motion_north = north_from_speed.where(~use_components, north_from_components)

    velocity_valid = ((speed_valid & direction_valid) | use_components) & motion_east.notna() & motion_north.notna()

    out["motion_speed_mps"] = speed
    out["motion_direction_deg"] = direction_deg
    out["motion_direction_rad"] = direction_rad
    out["motion_velocity_valid"] = velocity_valid.astype(int)
    out["motion_used_velocity_components"] = use_components.astype(int)
    out["motion_east_velocity_mps"] = motion_east
    out["motion_north_velocity_mps"] = motion_north

    return out


def _append_reason(existing: Any, reason: str) -> str:
    """Append invalid reason text."""
    if existing is None or pd.isna(existing) or str(existing).strip() == "":
        return reason

    existing_str = str(existing)

    if reason in existing_str.split("|"):
        return existing_str

    return f"{existing_str}|{reason}"


def add_motion_displacement(
    df: pd.DataFrame,
    cfg: MotionModelConfig,
) -> pd.DataFrame:
    """
    Compute causal onboard-motion displacement per segment.

    New columns:
    - delta_pos_u_east_m
    - delta_pos_u_north_m
    - motion_displacement_valid
    - motion_displacement_invalid_reason

    Uses trapezoidal integration:
        Delta p_u = 0.5 * (v_t + v_{t-1}) * delta_t
    """
    required = [
        cfg.segment_column,
        cfg.delta_t_column,
        "motion_velocity_valid",
        "motion_east_velocity_mps",
        "motion_north_velocity_mps",
    ]

    missing = _missing_columns(df, required)

    if missing:
        raise KeyError(f"Missing columns for motion displacement: {missing}")

    out = df.copy()

    out["delta_pos_u_east_m"] = np.nan
    out["delta_pos_u_north_m"] = np.nan
    out["motion_displacement_valid"] = 0
    out["motion_displacement_invalid_reason"] = ""

    for _, group in out.groupby(cfg.segment_column, sort=False):
        if cfg.order_column in group.columns:
            group = group.sort_values(cfg.order_column)

        idx = group.index

        dt = pd.to_numeric(group[cfg.delta_t_column], errors="coerce")
        dt_valid = dt.notna() & np.isfinite(dt) & (dt > 0) & (dt <= cfg.max_delta_seconds)

        curr_valid = group["motion_velocity_valid"].astype(int) == 1
        prev_valid = group["motion_velocity_valid"].shift(1).fillna(0).astype(int) == 1

        curr_east_v = group["motion_east_velocity_mps"]
        curr_north_v = group["motion_north_velocity_mps"]
        prev_east_v = curr_east_v.shift(1)
        prev_north_v = curr_north_v.shift(1)

        delta_east = 0.5 * (curr_east_v + prev_east_v) * dt
        delta_north = 0.5 * (curr_north_v + prev_north_v) * dt

        valid = curr_valid & prev_valid & dt_valid

        out.loc[idx, "delta_pos_u_east_m"] = delta_east.to_numpy()
        out.loc[idx, "delta_pos_u_north_m"] = delta_north.to_numpy()
        out.loc[idx[valid.to_numpy()], "motion_displacement_valid"] = 1

        first_idx = idx[0]
        out.loc[first_idx, "motion_displacement_invalid_reason"] = _append_reason(
            out.loc[first_idx, "motion_displacement_invalid_reason"],
            "segment_start",
        )

        for bad_idx in idx[(~curr_valid).to_numpy()]:
            out.loc[bad_idx, "motion_displacement_invalid_reason"] = _append_reason(
                out.loc[bad_idx, "motion_displacement_invalid_reason"],
                "missing_current_motion",
            )

        for bad_idx in idx[(~prev_valid).to_numpy()]:
            out.loc[bad_idx, "motion_displacement_invalid_reason"] = _append_reason(
                out.loc[bad_idx, "motion_displacement_invalid_reason"],
                "missing_previous_motion",
            )

        for bad_idx in idx[(~dt_valid).to_numpy()]:
            out.loc[bad_idx, "motion_displacement_invalid_reason"] = _append_reason(
                out.loc[bad_idx, "motion_displacement_invalid_reason"],
                "invalid_delta_t",
            )

    invalid_mask = out["motion_displacement_valid"] != 1
    out.loc[
        invalid_mask & (out["motion_displacement_invalid_reason"] == ""),
        "motion_displacement_invalid_reason",
    ] = "invalid_transition"

    return out


def add_motion_model_features(
    df: pd.DataFrame,
    dataset_key: str,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, MotionModelSummary]:
    """
    Full motion-model step for one dataset.
    """
    cfg = get_motion_model_config(config)

    out = add_motion_velocity_components(df, cfg)
    out = add_motion_displacement(out, cfg)

    added_columns = [
        "motion_speed_mps",
        "motion_direction_deg",
        "motion_direction_rad",
        "motion_velocity_valid",
        "motion_used_velocity_components",
        "motion_east_velocity_mps",
        "motion_north_velocity_mps",
        "delta_pos_u_east_m",
        "delta_pos_u_north_m",
        "motion_displacement_valid",
        "motion_displacement_invalid_reason",
    ]

    rows = int(len(out))
    segments = int(out[cfg.segment_column].nunique()) if cfg.segment_column in out.columns else 0

    velocity_valid = int((out["motion_velocity_valid"] == 1).sum())
    heading_valid = int(out["motion_direction_deg"].notna().sum())
    motion_valid = int((out["motion_displacement_valid"] == 1).sum())

    status = "PASSED"
    if velocity_valid == 0 or motion_valid == 0:
        status = "FAILED_NO_VALID_MOTION_FEATURES"

    summary = MotionModelSummary(
        dataset_key=dataset_key,
        rows=rows,
        segments=segments,
        velocity_rows_valid=velocity_valid,
        velocity_rows_invalid=rows - velocity_valid,
        heading_rows_valid=heading_valid,
        heading_rows_invalid=rows - heading_valid,
        valid_motion_displacement_rows=motion_valid,
        invalid_motion_displacement_rows=rows - motion_valid,
        used_velocity_component_rows=int((out["motion_used_velocity_components"] == 1).sum()),
        used_speed_heading_rows=int(((out["motion_velocity_valid"] == 1) & (out["motion_used_velocity_components"] == 0)).sum()),
        output_columns_added=added_columns,
        final_status=status,
    )

    return out, summary


def add_physical_coordinate_motion_features(
    df: pd.DataFrame,
    dataset_key: str,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, CoordinateTransformSummary, MotionModelSummary]:
    """
    Add both GNSS coordinate features and onboard motion model features.
    """
    out, coordinate_summary = add_coordinate_transform_features(
        df=df,
        dataset_key=dataset_key,
        config=config,
    )

    out, motion_summary = add_motion_model_features(
        df=out,
        dataset_key=dataset_key,
        config=config,
    )

    return out, coordinate_summary, motion_summary


def process_single_dataset_physical_step6(
    config: Mapping[str, Any],
    dataset_key: str,
    save_outputs: bool = True,
) -> PhysicalDatasetSummary:
    """
    Process one cleaned dataset into a physical intermediate file.
    """
    input_path = get_cleaned_file_path(config, dataset_key)
    output_path = get_physical_file_path(config, dataset_key)

    df = load_cleaned_dataset_for_physical_model(config, dataset_key)

    required = [
        "GPS Latitude",
        "GPS Longitude",
        "Velocity (m/s)",
        "Heading (deg)",
        "Yaw (deg)",
        "Yaw Rate (deg/s)",
        "Steering Angle (deg)",
        "delta_t_seconds",
        "segment_id",
    ]

    required_present = _present_columns(df, required)
    required_missing = _missing_columns(df, required)

    if required_missing:
        raise KeyError(
            f"{dataset_key} is missing required physical-model columns: {required_missing}"
        )

    input_rows = int(df.shape[0])
    input_columns = int(df.shape[1])

    out, coordinate_summary, motion_summary = add_physical_coordinate_motion_features(
        df=df,
        dataset_key=dataset_key,
        config=config,
    )

    if save_outputs:
        save_csv(out, output_path, index=False)

    final_status = "PASSED"
    if coordinate_summary.final_status != "PASSED" or motion_summary.final_status != "PASSED":
        final_status = "FAILED_PHYSICAL_FEATURE_CHECK"

    return PhysicalDatasetSummary(
        dataset_key=dataset_key,
        input_path=str(input_path),
        output_path=str(output_path),
        input_rows=input_rows,
        output_rows=int(out.shape[0]),
        input_columns=input_columns,
        output_columns=int(out.shape[1]),
        coordinate_summary=coordinate_summary.to_dict(),
        motion_summary=motion_summary.to_dict(),
        required_columns_present=required_present,
        required_columns_missing=required_missing,
        final_status=final_status,
    )


def print_physical_dataset_summary(summary: PhysicalDatasetSummary) -> None:
    """
    Print one Step-6 dataset summary.
    """
    coordinate = summary.coordinate_summary
    motion = summary.motion_summary

    print("=" * 100)
    print(f"STEP 6 PHYSICAL MODEL SUMMARY | {summary.dataset_key}")
    print("=" * 100)
    print(f"Input path                         : {summary.input_path}")
    print(f"Output path                        : {summary.output_path}")
    print(f"Rows                               : {summary.input_rows} -> {summary.output_rows}")
    print(f"Columns                            : {summary.input_columns} -> {summary.output_columns}")
    print(f"Required physical columns missing   : {summary.required_columns_missing}")
    print(f"Segments                            : {coordinate['segments']}")
    print(f"Valid GNSS local positions          : {coordinate['rows_with_local_position']}")
    print(f"Valid GNSS displacements            : {coordinate['valid_gnss_displacement_rows']}")
    print(f"Valid motion velocities             : {motion['velocity_rows_valid']}")
    print(f"Valid motion displacements          : {motion['valid_motion_displacement_rows']}")
    print(f"Velocity component rows used         : {motion['used_velocity_component_rows']}")
    print(f"Speed-heading rows used              : {motion['used_speed_heading_rows']}")
    print(f"Final status                        : {summary.final_status}")
    print("=" * 100)


def print_full_step6_report(report: FullPhysicalStep6Report) -> None:
    """
    Print full Step-6 report.
    """
    print("=" * 100)
    print("STEP 6 COORDINATE TRANSFORM AND MOTION MODEL REPORT")
    print("=" * 100)

    for summary in report.dataset_summaries.values():
        print_physical_dataset_summary(summary)

    print("-" * 100)
    print(f"Motion config                       : {report.motion_config}")
    print(f"Final Step 6 status                 : {report.final_step6_status}")
    print("=" * 100)


def run_coordinate_motion_model_step(
    config: Mapping[str, Any],
    dataset_keys: Optional[Sequence[str]] = None,
    save_outputs: bool = True,
) -> FullPhysicalStep6Report:
    """
    Main Step-6 entry point.

    Loads cleaned datasets, computes:
    - local GNSS position p_t^g,
    - causal GNSS displacement Delta p_t^g,
    - causal motion-model displacement Delta p_t^u,

    then saves physical intermediate CSV files.
    """
    keys = list(dataset_keys or DEFAULT_DATASET_KEYS)

    summaries: Dict[str, PhysicalDatasetSummary] = {}

    for dataset_key in keys:
        summary = process_single_dataset_physical_step6(
            config=config,
            dataset_key=dataset_key,
            save_outputs=save_outputs,
        )
        summaries[dataset_key] = summary

    all_passed = all(summary.final_status == "PASSED" for summary in summaries.values())
    final_status = "PASSED" if all_passed else "FAILED_PHYSICAL_STEP6_CHECK"

    report = FullPhysicalStep6Report(
        dataset_summaries=summaries,
        motion_config=asdict(get_motion_model_config(config)),
        final_step6_status=final_status,
    )

    if save_outputs:
        summary_path = get_step6_summary_path(config)
        save_json(report.to_dict(), summary_path, indent=2)
        print(f"Saved Step 6 coordinate-motion JSON: {summary_path}")

    print_full_step6_report(report)

    if report.final_step6_status != "PASSED":
        raise RuntimeError(
            f"Step 6 failed with status: {report.final_step6_status}."
        )

    return report