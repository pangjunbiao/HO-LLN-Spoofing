"""
Validity-mask utilities for the AV-GPS causal spoofing detection project.

Step 3 purpose:
- mark invalid first rows,
- mark segment-boundary transitions invalid,
- mark invalid time-delta transitions,
- optionally mark attack-to-normal recovery boundaries invalid,
- create a preliminary causal validity flag.

This prepares the future methodology validity flag:

    nu_t = 1 if displacement pair is valid and temporally aligned,
           0 otherwise.

Later steps will refine nu_t when GPS/motion availability is checked.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


def _get_nested(config: Mapping[str, Any], key_path: str, default: Any = None) -> Any:
    """Lightweight nested config getter."""
    current: Any = config

    for key in key_path.split("."):
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return default

    return current


def _append_reason(existing: Any, reason: str) -> str:
    """
    Append validity reason to an existing reason string.
    """
    if existing is None or pd.isna(existing) or str(existing).strip() == "":
        return reason
    return f"{existing};{reason}"


def add_segment_boundary_flags(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add flags for segment starts and segment boundary transitions.

    Added columns:
    - prev_segment_id
    - is_first_row_global
    - is_segment_start
    - crosses_segment_boundary

    A row is a segment start if:
    - it is the first row globally, or
    - its segment_id differs from previous row's segment_id.
    """
    out = df.copy() if copy else df

    if segment_col not in out.columns:
        raise KeyError(
            f"Missing segment column '{segment_col}'. "
            "Segmentation must run before validity masking."
        )

    out["prev_segment_id"] = out[segment_col].shift(1)
    out["is_first_row_global"] = False

    if len(out) > 0:
        out.loc[out.index[0], "is_first_row_global"] = True

    out["crosses_segment_boundary"] = out[segment_col] != out["prev_segment_id"]
    out["is_segment_start"] = (
        out["is_first_row_global"] | out["crosses_segment_boundary"]
    )

    return out


def add_label_transition_flags(
    df: pd.DataFrame,
    label_col: str = "Data Type",
    normal_label: int = 0,
    attack_label: int = 1,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add label-transition flags.

    Added columns:
    - prev_label
    - normal_to_attack_transition
    - attack_to_normal_transition

    The attack-to-normal transition can be invalidated to avoid recovery jumps
    being counted as ordinary normal evidence.
    """
    out = df.copy() if copy else df

    if label_col not in out.columns:
        out["prev_label"] = np.nan
        out["normal_to_attack_transition"] = False
        out["attack_to_normal_transition"] = False
        return out

    out["prev_label"] = out[label_col].shift(1)

    out["normal_to_attack_transition"] = (
        (out["prev_label"] == normal_label) & (out[label_col] == attack_label)
    )

    out["attack_to_normal_transition"] = (
        (out["prev_label"] == attack_label) & (out[label_col] == normal_label)
    )

    return out


def add_time_validity_flags(
    df: pd.DataFrame,
    min_delta_seconds: float = 0.0,
    max_delta_seconds: float = 5.0,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add time-validity flags.

    Requires delta_t_seconds.

    Added columns:
    - missing_delta_t
    - nonpositive_delta_t
    - too_large_delta_t
    - valid_time_delta
    """
    out = df.copy() if copy else df

    if "delta_t_seconds" not in out.columns:
        raise KeyError(
            "Missing 'delta_t_seconds'. Call add_delta_t_column() before "
            "add_time_validity_flags()."
        )

    out["missing_delta_t"] = out["delta_t_seconds"].isna()
    out["nonpositive_delta_t"] = out["delta_t_seconds"] <= float(min_delta_seconds)
    out["too_large_delta_t"] = out["delta_t_seconds"] > float(max_delta_seconds)

    out["valid_time_delta"] = ~(
        out["missing_delta_t"]
        | out["nonpositive_delta_t"]
        | out["too_large_delta_t"]
    )

    return out


def add_sensor_availability_flags(
    df: pd.DataFrame,
    gps_lat_col: str = "GPS Latitude",
    gps_lon_col: str = "GPS Longitude",
    motion_columns: Optional[list[str]] = None,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add preliminary GPS/motion availability flags.

    Added columns:
    - has_gps_position
    - has_motion_observation
    - has_required_observation

    This is still preliminary. Later residual construction may add stronger checks.
    """
    out = df.copy() if copy else df

    motion_columns = motion_columns or [
        "Velocity (m/s)",
        "Heading (deg)",
        "Yaw (deg)",
        "Yaw Rate (deg/s)",
        "Steering Angle (deg)",
    ]

    if gps_lat_col in out.columns and gps_lon_col in out.columns:
        out["has_gps_position"] = out[gps_lat_col].notna() & out[gps_lon_col].notna()
    else:
        out["has_gps_position"] = False

    existing_motion_cols = [col for col in motion_columns if col in out.columns]

    if existing_motion_cols:
        out["has_motion_observation"] = out[existing_motion_cols].notna().any(axis=1)
    else:
        out["has_motion_observation"] = False

    out["has_required_observation"] = (
        out["has_gps_position"] & out["has_motion_observation"]
    )

    return out


def build_preliminary_validity_mask(
    df: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
    segment_col: str = "segment_id",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Build preliminary transition-validity mask for Step 3.

    Added columns:
    - invalid_transition_reason
    - valid_transition_prelim
    - nu_prelim

    Invalid when:
    - first row of a segment,
    - segment boundary,
    - invalid delta_t,
    - missing GPS/motion observation,
    - optional attack-to-normal recovery boundary.

    Config keys used:
        dataset.label_column
        dataset.normal_label
        dataset.attack_label
        dataset.core_gnss_columns
        dataset.core_motion_columns
        preprocessing.validity.min_delta_seconds
        preprocessing.validity.max_delta_seconds
        preprocessing.validity.invalidate_attack_to_normal_boundary
    """
    config = config or {}

    out = df.copy() if copy else df

    label_col = str(_get_nested(config, "dataset.label_column", "Data Type"))
    normal_label = int(_get_nested(config, "dataset.normal_label", 0))
    attack_label = int(_get_nested(config, "dataset.attack_label", 1))

    core_gnss = _get_nested(
        config,
        "dataset.core_gnss_columns",
        ["GPS Latitude", "GPS Longitude"],
    )
    core_motion = _get_nested(
        config,
        "dataset.core_motion_columns",
        [
            "Velocity (m/s)",
            "Heading (deg)",
            "Yaw (deg)",
            "Yaw Rate (deg/s)",
            "Steering Angle (deg)",
        ],
    )

    gps_lat_col = core_gnss[0] if len(core_gnss) > 0 else "GPS Latitude"
    gps_lon_col = core_gnss[1] if len(core_gnss) > 1 else "GPS Longitude"

    min_delta = float(
        _get_nested(config, "preprocessing.validity.min_delta_seconds", 0.0)
    )
    max_delta = float(
        _get_nested(config, "preprocessing.validity.max_delta_seconds", 5.0)
    )
    invalidate_attack_to_normal = bool(
        _get_nested(
            config,
            "preprocessing.validity.invalidate_attack_to_normal_boundary",
            True,
        )
    )

    out = add_segment_boundary_flags(
        out,
        segment_col=segment_col,
        copy=False,
    )

    out = add_label_transition_flags(
        out,
        label_col=label_col,
        normal_label=normal_label,
        attack_label=attack_label,
        copy=False,
    )

    out = add_time_validity_flags(
        out,
        min_delta_seconds=min_delta,
        max_delta_seconds=max_delta,
        copy=False,
    )

    out = add_sensor_availability_flags(
        out,
        gps_lat_col=gps_lat_col,
        gps_lon_col=gps_lon_col,
        motion_columns=list(core_motion),
        copy=False,
    )

    out["invalid_transition_reason"] = ""

    invalid_reason_masks = {
        "segment_start": out["is_segment_start"],
        "invalid_delta_t": ~out["valid_time_delta"],
        "missing_required_observation": ~out["has_required_observation"],
    }

    if invalidate_attack_to_normal:
        invalid_reason_masks["attack_to_normal_recovery_boundary"] = out[
            "attack_to_normal_transition"
        ]

    for reason, mask in invalid_reason_masks.items():
        mask = mask.fillna(False)
        out.loc[mask, "invalid_transition_reason"] = out.loc[
            mask, "invalid_transition_reason"
        ].apply(lambda old: _append_reason(old, reason))

    out["valid_transition_prelim"] = out["invalid_transition_reason"].str.strip() == ""
    out["nu_prelim"] = out["valid_transition_prelim"].astype(int)

    return out


def summarize_validity_mask(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize preliminary validity mask.
    """
    summary: Dict[str, Any] = {}

    if "valid_transition_prelim" in df.columns:
        summary["valid_transition_count"] = int(df["valid_transition_prelim"].sum())
        summary["invalid_transition_count"] = int((~df["valid_transition_prelim"]).sum())

    if "nu_prelim" in df.columns:
        summary["nu_prelim_ones"] = int((df["nu_prelim"] == 1).sum())
        summary["nu_prelim_zeros"] = int((df["nu_prelim"] == 0).sum())

    reason_counts: Dict[str, int] = {}

    if "invalid_transition_reason" in df.columns:
        for value in df["invalid_transition_reason"].fillna(""):
            text = str(value).strip()
            if text == "":
                continue

            for reason in text.split(";"):
                reason = reason.strip()
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

    summary["invalid_reason_counts"] = reason_counts

    if "is_segment_start" in df.columns:
        summary["segment_start_count"] = int(df["is_segment_start"].sum())

    if "attack_to_normal_transition" in df.columns:
        summary["attack_to_normal_transition_count"] = int(
            df["attack_to_normal_transition"].sum()
        )

    if "normal_to_attack_transition" in df.columns:
        summary["normal_to_attack_transition_count"] = int(
            df["normal_to_attack_transition"].sum()
        )

    return summary


def print_validity_summary(
    df: pd.DataFrame,
    dataset_key: str,
) -> None:
    """
    Print preliminary validity summary for one dataset.
    """
    summary = summarize_validity_mask(df)

    print("=" * 100)
    print(f"PRELIMINARY VALIDITY SUMMARY | {dataset_key}")
    print("=" * 100)

    for key, value in summary.items():
        print(f"{key:40s}: {value}")

    print("=" * 100)

def _append_final_reason(existing: Any, reason: str) -> str:
    """
    Append final residual-validity reason.

    Uses pipe separator because Step-7 residual summaries count pipe-separated
    reason strings.
    """
    if existing is None or pd.isna(existing) or str(existing).strip() == "":
        return reason

    existing_str = str(existing)

    if reason in existing_str.split("|"):
        return existing_str

    return f"{existing_str}|{reason}"


def add_final_residual_validity_mask(
    df: pd.DataFrame,
    gnss_valid_column: str = "gnss_displacement_valid",
    motion_valid_column: str = "motion_displacement_valid",
    preliminary_nu_column: str = "nu_prelim",
    residual_finite_column: str = "residual_finite",
    delta_t_column: str = "delta_t_seconds",
    output_valid_column: str = "nu",
    output_reason_column: str = "residual_invalid_reason",
    max_delta_seconds: float = 5.0,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Build final residual validity nu_t for Step 7.

    Final rule:
        nu_t = 1 only if:
            - GNSS displacement is valid,
            - onboard motion displacement is valid,
            - preliminary Step-3 transition validity is valid,
            - residual values are finite,
            - delta_t is positive and not too large.

    This is stricter than Step-6 physical validity.

    Important:
    - This protects segment starts.
    - This protects invalid time transitions.
    - This protects Dataset-3 attack-to-normal recovery boundary through nu_prelim.
    - This does not use future rows.
    """
    out = df.copy() if copy else df

    required_columns = [
        gnss_valid_column,
        motion_valid_column,
        preliminary_nu_column,
        residual_finite_column,
        delta_t_column,
    ]

    missing = [col for col in required_columns if col not in out.columns]
    if missing:
        raise KeyError(
            "Missing required columns for final residual validity mask: "
            f"{missing}"
        )

    gnss_valid = out[gnss_valid_column].fillna(0).astype(int) == 1
    motion_valid = out[motion_valid_column].fillna(0).astype(int) == 1
    prelim_valid = out[preliminary_nu_column].fillna(0).astype(int) == 1
    residual_finite = out[residual_finite_column].fillna(0).astype(int) == 1

    delta_t = pd.to_numeric(out[delta_t_column], errors="coerce")
    delta_t_valid = (
        delta_t.notna()
        & np.isfinite(delta_t)
        & (delta_t > 0)
        & (delta_t <= float(max_delta_seconds))
    )

    final_valid = (
        gnss_valid
        & motion_valid
        & prelim_valid
        & residual_finite
        & delta_t_valid
    )

    out[output_reason_column] = ""

    invalid_reason_masks = {
        "invalid_gnss_displacement": ~gnss_valid,
        "invalid_motion_displacement": ~motion_valid,
        "invalid_preliminary_transition": ~prelim_valid,
        "nonfinite_residual": ~residual_finite,
        "invalid_delta_t": ~delta_t_valid,
    }

    for reason, mask in invalid_reason_masks.items():
        mask = mask.fillna(False)
        out.loc[mask, output_reason_column] = out.loc[
            mask, output_reason_column
        ].apply(lambda old: _append_final_reason(old, reason))

    out[output_valid_column] = final_valid.astype(int)

    return out


def summarize_final_residual_validity_mask(
    df: pd.DataFrame,
    valid_column: str = "nu",
    reason_column: str = "residual_invalid_reason",
) -> Dict[str, Any]:
    """
    Summarize final residual validity mask.

    This is used for Step-7 inspection.
    """
    summary: Dict[str, Any] = {}

    if valid_column in df.columns:
        valid = df[valid_column].fillna(0).astype(int) == 1
        summary["final_nu_ones"] = int(valid.sum())
        summary["final_nu_zeros"] = int((~valid).sum())

    reason_counts: Dict[str, int] = {}

    if reason_column in df.columns:
        for value in df[reason_column].fillna("").astype(str):
            text = value.strip()
            if text == "":
                continue

            for reason in text.split("|"):
                reason = reason.strip()
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

    summary["final_invalid_reason_counts"] = reason_counts

    return summary


def print_final_residual_validity_summary(
    df: pd.DataFrame,
    dataset_key: str,
    valid_column: str = "nu",
    reason_column: str = "residual_invalid_reason",
) -> None:
    """
    Print final residual validity summary for one dataset.
    """
    summary = summarize_final_residual_validity_mask(
        df=df,
        valid_column=valid_column,
        reason_column=reason_column,
    )

    print("=" * 100)
    print(f"FINAL RESIDUAL VALIDITY SUMMARY | {dataset_key}")
    print("=" * 100)

    for key, value in summary.items():
        print(f"{key:40s}: {value}")

    print("=" * 100)