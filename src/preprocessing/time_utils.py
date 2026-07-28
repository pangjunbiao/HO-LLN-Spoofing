"""
Time utilities for the AV-GPS causal spoofing detection project.

Step 3 purpose:
- parse AV-GPS time columns,
- convert Run Time / Hobbs into seconds,
- combine Clock Date + Clock Time into datetime,
- compute adjacent time differences,
- detect time resets, date/session discontinuities, and large gaps.

These utilities are used by:
- src/data/segment_trajectories.py
- src/preprocessing/validity_mask.py

Important:
This file does not create final segment IDs by itself. It prepares reliable
time columns and transition diagnostics used by the segmenter.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

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


def parse_duration_to_seconds(value: Any) -> float:
    """
    Parse a duration-like value into seconds.

    Supports common AV-GPS formats:
    - "0:00:01"
    - "00:00:01"
    - "1:02:03"
    - "02:03"
    - pandas Timedelta
    - numeric seconds

    Returns np.nan if parsing fails.
    """
    if value is None:
        return np.nan

    if pd.isna(value):
        return np.nan

    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())

    if isinstance(value, np.timedelta64):
        try:
            return float(pd.to_timedelta(value).total_seconds())
        except Exception:
            return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if text == "":
        return np.nan

    # Try pandas first because it handles many forms safely.
    try:
        td = pd.to_timedelta(text)
        if not pd.isna(td):
            return float(td.total_seconds())
    except Exception:
        pass

    # Manual fallback for H:M:S or M:S.
    if re.fullmatch(r"\d+:\d{1,2}:\d{1,2}(\.\d+)?", text):
        parts = text.split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds

    if re.fullmatch(r"\d+:\d{1,2}(\.\d+)?", text):
        parts = text.split(":")
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds

    # Last attempt: numeric string.
    try:
        return float(text)
    except Exception:
        return np.nan


def parse_clock_datetime(
    date_value: Any,
    time_value: Any,
    dayfirst: bool = False,
) -> pd.Timestamp:
    """
    Parse Clock Date + Clock Time into pandas Timestamp.

    Returns pd.NaT if parsing fails.

    AV-GPS examples look like:
    - date: "5/24/2022"
    - time: "10:13:45 AM" or similar
    """
    if date_value is None or time_value is None:
        return pd.NaT

    if pd.isna(date_value) or pd.isna(time_value):
        return pd.NaT

    date_text = str(date_value).strip()
    time_text = str(time_value).strip()

    if date_text == "" or time_text == "":
        return pd.NaT

    combined = f"{date_text} {time_text}"

    try:
        return pd.to_datetime(combined, errors="coerce", dayfirst=dayfirst)
    except Exception:
        return pd.NaT


def add_time_columns(
    df: pd.DataFrame,
    runtime_col: str = "Run Time",
    hobbs_col: str = "Hobbs",
    clock_date_col: str = "Clock Date",
    clock_time_col: str = "Clock Time",
    dayfirst: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add parsed time columns to a DataFrame.

    Added columns:
    - row_index_original
    - runtime_seconds
    - hobbs_seconds
    - clock_datetime
    - clock_timestamp_seconds

    Missing source columns are tolerated and produce NaN/NaT columns.
    """
    out = df.copy() if copy else df

    if "row_index_original" not in out.columns:
        out.insert(0, "row_index_original", np.arange(len(out), dtype=int))

    if runtime_col in out.columns:
        out["runtime_seconds"] = out[runtime_col].apply(parse_duration_to_seconds)
    else:
        out["runtime_seconds"] = np.nan

    if hobbs_col in out.columns:
        out["hobbs_seconds"] = out[hobbs_col].apply(parse_duration_to_seconds)
    else:
        out["hobbs_seconds"] = np.nan

    if clock_date_col in out.columns and clock_time_col in out.columns:
        out["clock_datetime"] = [
            parse_clock_datetime(date_value, time_value, dayfirst=dayfirst)
            for date_value, time_value in zip(out[clock_date_col], out[clock_time_col])
        ]
        out["clock_datetime"] = pd.to_datetime(out["clock_datetime"], errors="coerce")
    else:
        out["clock_datetime"] = pd.NaT

    out["clock_timestamp_seconds"] = out["clock_datetime"].astype("int64") / 1e9
    out.loc[out["clock_datetime"].isna(), "clock_timestamp_seconds"] = np.nan

    return out


def compute_adjacent_time_deltas(
    df: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Compute adjacent row time differences in current row order.

    Added columns:
    - prev_runtime_seconds
    - prev_hobbs_seconds
    - prev_clock_datetime
    - delta_runtime_seconds
    - delta_hobbs_seconds
    - delta_clock_seconds

    Notes:
    This is not grouped by segment yet. It is used to detect boundaries.
    """
    out = df.copy() if copy else df

    for required in ["runtime_seconds", "hobbs_seconds", "clock_datetime"]:
        if required not in out.columns:
            raise KeyError(
                f"Missing '{required}'. Call add_time_columns() before "
                "compute_adjacent_time_deltas()."
            )

    out["prev_runtime_seconds"] = out["runtime_seconds"].shift(1)
    out["prev_hobbs_seconds"] = out["hobbs_seconds"].shift(1)
    out["prev_clock_datetime"] = out["clock_datetime"].shift(1)

    out["delta_runtime_seconds"] = out["runtime_seconds"] - out["prev_runtime_seconds"]
    out["delta_hobbs_seconds"] = out["hobbs_seconds"] - out["prev_hobbs_seconds"]
    out["delta_clock_seconds"] = (
        out["clock_datetime"] - out["prev_clock_datetime"]
    ).dt.total_seconds()

    return out


def add_time_diagnostics(
    df: pd.DataFrame,
    max_gap_seconds: float = 5.0,
    reset_tolerance_seconds: float = -0.5,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add boolean diagnostics for time resets and discontinuities.

    Added columns:
    - missing_runtime_pair
    - missing_clock_pair
    - runtime_reset
    - hobbs_reset
    - clock_backward
    - large_runtime_gap
    - large_clock_gap
    - nonpositive_runtime_delta
    - nonpositive_clock_delta

    These are used by trajectory segmentation.
    """
    out = df.copy() if copy else df

    required = [
        "runtime_seconds",
        "hobbs_seconds",
        "clock_datetime",
        "delta_runtime_seconds",
        "delta_hobbs_seconds",
        "delta_clock_seconds",
    ]
    missing_required = [col for col in required if col not in out.columns]
    if missing_required:
        raise KeyError(
            "Missing required time columns before add_time_diagnostics(): "
            f"{missing_required}"
        )

    out["missing_runtime_pair"] = (
        out["runtime_seconds"].isna() | out["prev_runtime_seconds"].isna()
    )
    out["missing_hobbs_pair"] = (
        out["hobbs_seconds"].isna() | out["prev_hobbs_seconds"].isna()
    )
    out["missing_clock_pair"] = (
        out["clock_datetime"].isna() | out["prev_clock_datetime"].isna()
    )

    out["runtime_reset"] = out["delta_runtime_seconds"] < reset_tolerance_seconds
    out["hobbs_reset"] = out["delta_hobbs_seconds"] < reset_tolerance_seconds
    out["clock_backward"] = out["delta_clock_seconds"] < reset_tolerance_seconds

    out["large_runtime_gap"] = out["delta_runtime_seconds"] > float(max_gap_seconds)
    out["large_clock_gap"] = out["delta_clock_seconds"] > float(max_gap_seconds)

    out["nonpositive_runtime_delta"] = out["delta_runtime_seconds"] <= 0
    out["nonpositive_clock_delta"] = out["delta_clock_seconds"] <= 0

    # First row cannot be a valid transition.
    if len(out) > 0:
        first_idx = out.index[0]
        boundary_cols = [
            "missing_runtime_pair",
            "missing_hobbs_pair",
            "missing_clock_pair",
            "runtime_reset",
            "hobbs_reset",
            "clock_backward",
            "large_runtime_gap",
            "large_clock_gap",
            "nonpositive_runtime_delta",
            "nonpositive_clock_delta",
        ]
        for col in boundary_cols:
            out.loc[first_idx, col] = True

    return out


def prepare_time_features(
    df: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Full Step-3 time-feature preparation.

    This combines:
    - add_time_columns()
    - compute_adjacent_time_deltas()
    - add_time_diagnostics()

    Config keys used:
        preprocessing.time.runtime_col
        preprocessing.time.hobbs_col
        preprocessing.time.clock_date_col
        preprocessing.time.clock_time_col
        preprocessing.time.max_gap_seconds
        preprocessing.time.reset_tolerance_seconds
        preprocessing.time.dayfirst
    """
    config = config or {}

    runtime_col = str(_get_nested(config, "preprocessing.time.runtime_col", "Run Time"))
    hobbs_col = str(_get_nested(config, "preprocessing.time.hobbs_col", "Hobbs"))
    clock_date_col = str(
        _get_nested(config, "preprocessing.time.clock_date_col", "Clock Date")
    )
    clock_time_col = str(
        _get_nested(config, "preprocessing.time.clock_time_col", "Clock Time")
    )
    dayfirst = bool(_get_nested(config, "preprocessing.time.dayfirst", False))

    max_gap_seconds = float(
        _get_nested(config, "preprocessing.time.max_gap_seconds", 5.0)
    )
    reset_tolerance_seconds = float(
        _get_nested(config, "preprocessing.time.reset_tolerance_seconds", -0.5)
    )

    out = add_time_columns(
        df=df,
        runtime_col=runtime_col,
        hobbs_col=hobbs_col,
        clock_date_col=clock_date_col,
        clock_time_col=clock_time_col,
        dayfirst=dayfirst,
        copy=copy,
    )
    out = compute_adjacent_time_deltas(out, copy=False)
    out = add_time_diagnostics(
        out,
        max_gap_seconds=max_gap_seconds,
        reset_tolerance_seconds=reset_tolerance_seconds,
        copy=False,
    )

    return out


def choose_delta_t_seconds(
    row: pd.Series,
    prefer_clock: bool = True,
) -> float:
    """
    Choose a transition delta_t for later causal computation.

    Priority:
    - if prefer_clock and delta_clock_seconds is valid positive, use clock delta;
    - otherwise use runtime delta if valid positive;
    - otherwise return NaN.

    This does not check segment boundaries. validity_mask.py handles that.
    """
    delta_clock = row.get("delta_clock_seconds", np.nan)
    delta_runtime = row.get("delta_runtime_seconds", np.nan)

    if prefer_clock and pd.notna(delta_clock) and float(delta_clock) > 0:
        return float(delta_clock)

    if pd.notna(delta_runtime) and float(delta_runtime) > 0:
        return float(delta_runtime)

    if pd.notna(delta_clock) and float(delta_clock) > 0:
        return float(delta_clock)

    return np.nan


def add_delta_t_column(
    df: pd.DataFrame,
    prefer_clock: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add delta_t_seconds column for later residual/evidence construction.

    This is preliminary. Segment starts and invalid transitions will later have
    valid_transition=False and nu_t=0.
    """
    out = df.copy() if copy else df
    out["delta_t_seconds"] = out.apply(
        lambda row: choose_delta_t_seconds(row, prefer_clock=prefer_clock),
        axis=1,
    )
    return out


def summarize_time_diagnostics(df: pd.DataFrame) -> Dict[str, int]:
    """
    Summarize time diagnostics as integer counts.
    """
    diagnostic_columns = [
        "missing_runtime_pair",
        "missing_hobbs_pair",
        "missing_clock_pair",
        "runtime_reset",
        "hobbs_reset",
        "clock_backward",
        "large_runtime_gap",
        "large_clock_gap",
        "nonpositive_runtime_delta",
        "nonpositive_clock_delta",
    ]

    summary: Dict[str, int] = {}

    for col in diagnostic_columns:
        if col in df.columns:
            summary[col] = int(df[col].fillna(False).sum())

    return summary


def print_time_diagnostics_summary(
    df: pd.DataFrame,
    dataset_key: str,
) -> None:
    """
    Print time diagnostics summary for one dataset.
    """
    summary = summarize_time_diagnostics(df)

    print("=" * 100)
    print(f"TIME DIAGNOSTICS SUMMARY | {dataset_key}")
    print("=" * 100)

    for key, value in summary.items():
        print(f"{key:30s}: {value}")

    print("=" * 100)