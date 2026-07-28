"""
Coordinate transform utilities for AV-GPS causal spoofing detection.

Step 6 purpose:
- convert GNSS latitude/longitude to local east-north coordinates,
- compute causal GNSS displacement within each segment,
- never use future rows for transition-level displacement.

Method:
    p_t^g = T_loc(lat_t, lon_t; lat_ref, lon_ref)
    Delta p_t^g = p_t^g - p_{t-1}^g

Reference handling:
- A separate local reference is used for every independent segment.
- The reference is the first valid GNSS coordinate in that segment.
- This is a fixed coordinate origin, not a model feature.
- Transition displacement remains causal because Delta p_t^g uses only t and t-1.

This file does not create model inputs directly.
It prepares physical quantities needed later for residual evidence xi_t.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.config import get_by_path


EARTH_RADIUS_M = 6378137.0


@dataclass
class CoordinateTransformConfig:
    """Configuration for local GNSS coordinate transform."""

    segment_column: str
    order_column: str
    latitude_column: str
    longitude_column: str
    delta_t_column: str
    max_abs_latitude: float
    max_abs_longitude: float


@dataclass
class CoordinateTransformSummary:
    """Summary for one dataset after coordinate transform."""

    dataset_key: str
    rows: int
    segments: int
    valid_lat_lon_rows: int
    invalid_lat_lon_rows: int
    rows_with_local_position: int
    valid_gnss_displacement_rows: int
    invalid_gnss_displacement_rows: int
    local_reference_count: int
    output_columns_added: List[str]
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_coordinate_transform_config(config: Mapping[str, Any]) -> CoordinateTransformConfig:
    """
    Read coordinate-transform settings from config.
    """
    return CoordinateTransformConfig(
        segment_column=str(
            get_by_path(config, "preprocessing.coordinate_transform.segment_column", "segment_id")
        ),
        order_column=str(
            get_by_path(config, "preprocessing.coordinate_transform.order_column", "within_segment_index")
        ),
        latitude_column=str(
            get_by_path(config, "preprocessing.coordinate_transform.latitude_column", "GPS Latitude")
        ),
        longitude_column=str(
            get_by_path(config, "preprocessing.coordinate_transform.longitude_column", "GPS Longitude")
        ),
        delta_t_column=str(
            get_by_path(config, "preprocessing.coordinate_transform.delta_t_column", "delta_t_seconds")
        ),
        max_abs_latitude=float(
            get_by_path(config, "preprocessing.coordinate_transform.max_abs_latitude", 90.0)
        ),
        max_abs_longitude=float(
            get_by_path(config, "preprocessing.coordinate_transform.max_abs_longitude", 180.0)
        ),
    )


def validate_coordinate_columns(
    df: pd.DataFrame,
    cfg: CoordinateTransformConfig,
) -> None:
    """
    Validate required coordinate columns.
    """
    required = [
        cfg.segment_column,
        cfg.latitude_column,
        cfg.longitude_column,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing coordinate-transform columns: {missing}")


def add_lat_lon_validity(
    df: pd.DataFrame,
    cfg: CoordinateTransformConfig,
) -> pd.DataFrame:
    """
    Add boolean validity mask for latitude/longitude.
    """
    out = df.copy()

    lat = pd.to_numeric(out[cfg.latitude_column], errors="coerce")
    lon = pd.to_numeric(out[cfg.longitude_column], errors="coerce")

    valid = (
        lat.notna()
        & lon.notna()
        & np.isfinite(lat)
        & np.isfinite(lon)
        & (lat.abs() <= cfg.max_abs_latitude)
        & (lon.abs() <= cfg.max_abs_longitude)
    )

    out["gnss_latitude_numeric"] = lat
    out["gnss_longitude_numeric"] = lon
    out["gnss_lat_lon_valid"] = valid.astype(int)

    return out


def local_east_north_from_reference(
    lat_deg: pd.Series,
    lon_deg: pd.Series,
    ref_lat_deg: float,
    ref_lon_deg: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Convert latitude/longitude to local east/north meters using an equirectangular
    approximation around the segment reference.

    This is appropriate for local vehicle trajectories and avoids requiring pyproj.
    """
    lat_rad = np.deg2rad(pd.to_numeric(lat_deg, errors="coerce"))
    lon_rad = np.deg2rad(pd.to_numeric(lon_deg, errors="coerce"))

    ref_lat_rad = np.deg2rad(float(ref_lat_deg))
    ref_lon_rad = np.deg2rad(float(ref_lon_deg))

    east_m = EARTH_RADIUS_M * np.cos(ref_lat_rad) * (lon_rad - ref_lon_rad)
    north_m = EARTH_RADIUS_M * (lat_rad - ref_lat_rad)

    return pd.Series(east_m, index=lat_deg.index), pd.Series(north_m, index=lat_deg.index)


def add_local_gnss_coordinates(
    df: pd.DataFrame,
    cfg: CoordinateTransformConfig,
) -> pd.DataFrame:
    """
    Add local GNSS east/north coordinates for each segment.

    New columns:
    - gnss_ref_latitude
    - gnss_ref_longitude
    - gnss_local_east_m
    - gnss_local_north_m
    - gnss_local_position_valid
    """
    validate_coordinate_columns(df, cfg)

    out = add_lat_lon_validity(df, cfg)

    for col in [
        "gnss_ref_latitude",
        "gnss_ref_longitude",
        "gnss_local_east_m",
        "gnss_local_north_m",
    ]:
        out[col] = np.nan

    out["gnss_local_position_valid"] = 0

    groupby_obj = out.groupby(cfg.segment_column, sort=False)

    for _, group in groupby_obj:
        if cfg.order_column in group.columns:
            group = group.sort_values(cfg.order_column)

        valid_group = group[group["gnss_lat_lon_valid"] == 1]

        if valid_group.empty:
            continue

        ref_lat = float(valid_group["gnss_latitude_numeric"].iloc[0])
        ref_lon = float(valid_group["gnss_longitude_numeric"].iloc[0])

        east_m, north_m = local_east_north_from_reference(
            lat_deg=group["gnss_latitude_numeric"],
            lon_deg=group["gnss_longitude_numeric"],
            ref_lat_deg=ref_lat,
            ref_lon_deg=ref_lon,
        )

        valid_idx = group.index[group["gnss_lat_lon_valid"] == 1]

        out.loc[group.index, "gnss_ref_latitude"] = ref_lat
        out.loc[group.index, "gnss_ref_longitude"] = ref_lon
        out.loc[valid_idx, "gnss_local_east_m"] = east_m.loc[valid_idx]
        out.loc[valid_idx, "gnss_local_north_m"] = north_m.loc[valid_idx]
        out.loc[valid_idx, "gnss_local_position_valid"] = 1

    return out


def _append_reason(existing: Any, reason: str) -> str:
    """
    Append an invalid reason string.
    """
    if existing is None or pd.isna(existing) or str(existing).strip() == "":
        return reason

    existing_str = str(existing)

    if reason in existing_str.split("|"):
        return existing_str

    return f"{existing_str}|{reason}"


def add_gnss_displacement(
    df: pd.DataFrame,
    cfg: CoordinateTransformConfig,
) -> pd.DataFrame:
    """
    Compute causal GNSS displacement per segment.

    New columns:
    - delta_pos_g_east_m
    - delta_pos_g_north_m
    - gnss_displacement_valid
    - gnss_displacement_invalid_reason

    Displacement at row t uses only p_t and p_{t-1} in the same segment.
    """
    required = [
        cfg.segment_column,
        "gnss_local_east_m",
        "gnss_local_north_m",
        "gnss_local_position_valid",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing columns for GNSS displacement: {missing}")

    out = df.copy()

    out["delta_pos_g_east_m"] = np.nan
    out["delta_pos_g_north_m"] = np.nan
    out["gnss_displacement_valid"] = 0
    out["gnss_displacement_invalid_reason"] = ""

    for _, group in out.groupby(cfg.segment_column, sort=False):
        if cfg.order_column in group.columns:
            group = group.sort_values(cfg.order_column)

        idx = group.index

        prev_east = group["gnss_local_east_m"].shift(1)
        prev_north = group["gnss_local_north_m"].shift(1)
        prev_valid = group["gnss_local_position_valid"].shift(1).fillna(0).astype(int)

        curr_east = group["gnss_local_east_m"]
        curr_north = group["gnss_local_north_m"]
        curr_valid = group["gnss_local_position_valid"].astype(int)

        delta_east = curr_east - prev_east
        delta_north = curr_north - prev_north

        transition_valid = (curr_valid == 1) & (prev_valid == 1)

        if cfg.delta_t_column in group.columns:
            delta_t = pd.to_numeric(group[cfg.delta_t_column], errors="coerce")
            transition_valid = transition_valid & delta_t.notna() & np.isfinite(delta_t) & (delta_t > 0)

        first_idx = idx[0]

        out.loc[idx, "delta_pos_g_east_m"] = delta_east.to_numpy()
        out.loc[idx, "delta_pos_g_north_m"] = delta_north.to_numpy()
        out.loc[idx[transition_valid.to_numpy()], "gnss_displacement_valid"] = 1

        out.loc[first_idx, "gnss_displacement_invalid_reason"] = _append_reason(
            out.loc[first_idx, "gnss_displacement_invalid_reason"],
            "segment_start",
        )

        missing_current = curr_valid != 1
        missing_previous = prev_valid != 1

        for bad_idx in idx[missing_current.to_numpy()]:
            out.loc[bad_idx, "gnss_displacement_invalid_reason"] = _append_reason(
                out.loc[bad_idx, "gnss_displacement_invalid_reason"],
                "missing_current_gnss",
            )

        for bad_idx in idx[missing_previous.to_numpy()]:
            out.loc[bad_idx, "gnss_displacement_invalid_reason"] = _append_reason(
                out.loc[bad_idx, "gnss_displacement_invalid_reason"],
                "missing_previous_gnss",
            )

        if cfg.delta_t_column in group.columns:
            invalid_dt = ~(pd.to_numeric(group[cfg.delta_t_column], errors="coerce").notna())
            invalid_dt = invalid_dt | ~(np.isfinite(pd.to_numeric(group[cfg.delta_t_column], errors="coerce")))
            invalid_dt = invalid_dt | (pd.to_numeric(group[cfg.delta_t_column], errors="coerce") <= 0)

            for bad_idx in idx[invalid_dt.to_numpy()]:
                out.loc[bad_idx, "gnss_displacement_invalid_reason"] = _append_reason(
                    out.loc[bad_idx, "gnss_displacement_invalid_reason"],
                    "invalid_delta_t",
                )

    invalid_mask = out["gnss_displacement_valid"] != 1
    out.loc[invalid_mask & (out["gnss_displacement_invalid_reason"] == ""), "gnss_displacement_invalid_reason"] = "invalid_transition"

    return out


def add_coordinate_transform_features(
    df: pd.DataFrame,
    dataset_key: str,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, CoordinateTransformSummary]:
    """
    Full coordinate-transform step for one dataset.
    """
    cfg = get_coordinate_transform_config(config)

    out = add_local_gnss_coordinates(df, cfg)
    out = add_gnss_displacement(out, cfg)

    added_columns = [
        "gnss_latitude_numeric",
        "gnss_longitude_numeric",
        "gnss_lat_lon_valid",
        "gnss_ref_latitude",
        "gnss_ref_longitude",
        "gnss_local_east_m",
        "gnss_local_north_m",
        "gnss_local_position_valid",
        "delta_pos_g_east_m",
        "delta_pos_g_north_m",
        "gnss_displacement_valid",
        "gnss_displacement_invalid_reason",
    ]

    rows = int(len(out))
    segments = int(out[cfg.segment_column].nunique()) if cfg.segment_column in out.columns else 0

    valid_lat_lon_rows = int((out["gnss_lat_lon_valid"] == 1).sum())
    valid_position_rows = int((out["gnss_local_position_valid"] == 1).sum())
    valid_displacement_rows = int((out["gnss_displacement_valid"] == 1).sum())

    local_reference_count = int(
        out[[cfg.segment_column, "gnss_ref_latitude", "gnss_ref_longitude"]]
        .dropna()
        .drop_duplicates()
        .shape[0]
    )

    status = "PASSED"
    if valid_position_rows == 0 or valid_displacement_rows == 0:
        status = "FAILED_NO_VALID_GNSS_PHYSICAL_FEATURES"

    summary = CoordinateTransformSummary(
        dataset_key=dataset_key,
        rows=rows,
        segments=segments,
        valid_lat_lon_rows=valid_lat_lon_rows,
        invalid_lat_lon_rows=rows - valid_lat_lon_rows,
        rows_with_local_position=valid_position_rows,
        valid_gnss_displacement_rows=valid_displacement_rows,
        invalid_gnss_displacement_rows=rows - valid_displacement_rows,
        local_reference_count=local_reference_count,
        output_columns_added=added_columns,
        final_status=status,
    )

    return out, summary