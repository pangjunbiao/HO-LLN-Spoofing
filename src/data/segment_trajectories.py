"""
Trajectory segmentation utilities for the AV-GPS causal spoofing detection project.

Step 3 purpose:
- convert each raw AV-GPS source file into causal trajectory segments,
- detect runtime resets, Hobbs resets, clock backward jumps, date/session changes,
  and large time gaps,
- preserve Dataset-3 as an online sequence when configured,
- create segment_id and within-segment index,
- mark preliminary invalid transitions using validity_mask.py,
- save segmented CSV files into data/interim/.

Important project rule:
Each independent segment resets later sequential states:

    a_0 = 0, h_0 = 0, v_0 = 0

This file does not build xi_t yet. It only prepares correct causal segments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.data.load_raw import RawDatasetBundle, load_and_validate_raw_datasets
from src.preprocessing.time_utils import (
    add_delta_t_column,
    prepare_time_features,
    summarize_time_diagnostics,
)
from src.preprocessing.validity_mask import (
    build_preliminary_validity_mask,
    summarize_validity_mask,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_csv, save_json


@dataclass
class SegmentationSummary:
    """Summary of segmentation result for one dataset."""

    dataset_key: str
    role: str
    output_path: str
    rows: int
    segment_count: int
    min_segment_length: int
    median_segment_length: float
    max_segment_length: int
    label_counts: Dict[str, int]
    boundary_reason_counts: Dict[str, int]
    time_diagnostics: Dict[str, int]
    validity_summary: Dict[str, Any]


@dataclass
class FullSegmentationReport:
    """Full Step-3 segmentation report."""

    dataset_summaries: Dict[str, SegmentationSummary]
    final_step3_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_summaries": {
                key: asdict(value) for key, value in self.dataset_summaries.items()
            },
            "final_step3_status": self.final_step3_status,
        }


def _get_nested(config: Mapping[str, Any], key_path: str, default: Any = None) -> Any:
    """Lightweight nested config getter."""
    return get_by_path(config, key_path, default)


def _safe_label_counts(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    """Return label counts with JSON-safe string keys."""
    if label_col not in df.columns:
        return {}

    counts = df[label_col].value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _append_reason(existing: Any, reason: str) -> str:
    """Append boundary reason to string."""
    if existing is None or pd.isna(existing) or str(existing).strip() == "":
        return reason
    return f"{existing};{reason}"


def _count_reason_strings(series: pd.Series) -> Dict[str, int]:
    """Count semicolon-separated reason strings."""
    counts: Dict[str, int] = {}

    for value in series.fillna(""):
        text = str(value).strip()
        if not text:
            continue

        for reason in text.split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1

    return counts


def _date_change_flag(df: pd.DataFrame, clock_date_col: str) -> pd.Series:
    """
    Detect date/session changes using Clock Date.

    Missing dates do not automatically create a date-change boundary.
    """
    if clock_date_col not in df.columns:
        return pd.Series(False, index=df.index)

    current_date = df[clock_date_col].astype("string")
    previous_date = current_date.shift(1)

    both_present = current_date.notna() & previous_date.notna()
    return (current_date != previous_date) & both_present


def _source_change_flag(df: pd.DataFrame) -> pd.Series:
    """Detect source file/key changes if multiple sources are ever concatenated."""
    flags = pd.Series(False, index=df.index)

    if "source_key" in df.columns:
        flags = flags | (df["source_key"] != df["source_key"].shift(1))

    if "source_file" in df.columns:
        flags = flags | (df["source_file"] != df["source_file"].shift(1))

    if len(flags) > 0:
        flags.iloc[0] = False

    return flags.fillna(False)


def build_segmentation_boundary_flags(
    df: pd.DataFrame,
    dataset_key: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Build segmentation boundary flags.

    Boundary triggers:
    - first row,
    - source change,
    - date/session change,
    - runtime reset,
    - Hobbs reset,
    - clock backward,
    - large runtime gap,
    - large clock gap,
    - optional label change,
    - optional missing clock pair,
    - optional nonpositive runtime/clock delta.

    Dataset-3 can be preserved as one online sequence by config.
    """
    out = df.copy()

    preserve_as_single_sequence = bool(
        _get_nested(
            config,
            f"preprocessing.segmentation.per_dataset.{dataset_key}.preserve_as_single_sequence",
            False,
        )
    )

    clock_date_col = str(
        _get_nested(config, "preprocessing.time.clock_date_col", "Clock Date")
    )
    label_col = str(_get_nested(config, "dataset.label_column", "Data Type"))

    split_on_label_change = bool(
        _get_nested(config, "preprocessing.segmentation.split_on_label_change", False)
    )
    split_on_missing_clock_pair = bool(
        _get_nested(config, "preprocessing.segmentation.split_on_missing_clock_pair", False)
    )
    split_on_nonpositive_time_delta = bool(
        _get_nested(
            config,
            "preprocessing.segmentation.split_on_nonpositive_time_delta",
            False,
        )
    )

    out["boundary_first_row"] = False
    if len(out) > 0:
        out.loc[out.index[0], "boundary_first_row"] = True

    if preserve_as_single_sequence:
        # Dataset-3 online case study should remain one causal sequence.
        out["boundary_source_change"] = False
        out["boundary_date_change"] = False
        out["boundary_runtime_reset"] = False
        out["boundary_hobbs_reset"] = False
        out["boundary_clock_backward"] = False
        out["boundary_large_runtime_gap"] = False
        out["boundary_large_clock_gap"] = False
        out["boundary_label_change"] = False
        out["boundary_missing_clock_pair"] = False
        out["boundary_nonpositive_time_delta"] = False

    else:
        out["boundary_source_change"] = _source_change_flag(out)
        out["boundary_date_change"] = _date_change_flag(out, clock_date_col)

        out["boundary_runtime_reset"] = out.get(
            "runtime_reset", pd.Series(False, index=out.index)
        ).fillna(False)
        out["boundary_hobbs_reset"] = out.get(
            "hobbs_reset", pd.Series(False, index=out.index)
        ).fillna(False)
        out["boundary_clock_backward"] = out.get(
            "clock_backward", pd.Series(False, index=out.index)
        ).fillna(False)
        out["boundary_large_runtime_gap"] = out.get(
            "large_runtime_gap", pd.Series(False, index=out.index)
        ).fillna(False)
        out["boundary_large_clock_gap"] = out.get(
            "large_clock_gap", pd.Series(False, index=out.index)
        ).fillna(False)

        if split_on_label_change and label_col in out.columns:
            out["boundary_label_change"] = out[label_col] != out[label_col].shift(1)
            if len(out) > 0:
                out.loc[out.index[0], "boundary_label_change"] = False
        else:
            out["boundary_label_change"] = False

        if split_on_missing_clock_pair:
            out["boundary_missing_clock_pair"] = out.get(
                "missing_clock_pair", pd.Series(False, index=out.index)
            ).fillna(False)
        else:
            out["boundary_missing_clock_pair"] = False

        if split_on_nonpositive_time_delta:
            nonpositive_runtime = out.get(
                "nonpositive_runtime_delta", pd.Series(False, index=out.index)
            ).fillna(False)
            nonpositive_clock = out.get(
                "nonpositive_clock_delta", pd.Series(False, index=out.index)
            ).fillna(False)
            out["boundary_nonpositive_time_delta"] = nonpositive_runtime | nonpositive_clock
        else:
            out["boundary_nonpositive_time_delta"] = False

    boundary_columns = [
        "boundary_first_row",
        "boundary_source_change",
        "boundary_date_change",
        "boundary_runtime_reset",
        "boundary_hobbs_reset",
        "boundary_clock_backward",
        "boundary_large_runtime_gap",
        "boundary_large_clock_gap",
        "boundary_label_change",
        "boundary_missing_clock_pair",
        "boundary_nonpositive_time_delta",
    ]

    out["segment_boundary"] = False
    for col in boundary_columns:
        out["segment_boundary"] = out["segment_boundary"] | out[col].fillna(False)

    out["segment_boundary_reason"] = ""

    reason_map = {
        "boundary_first_row": "first_row",
        "boundary_source_change": "source_change",
        "boundary_date_change": "date_change",
        "boundary_runtime_reset": "runtime_reset",
        "boundary_hobbs_reset": "hobbs_reset",
        "boundary_clock_backward": "clock_backward",
        "boundary_large_runtime_gap": "large_runtime_gap",
        "boundary_large_clock_gap": "large_clock_gap",
        "boundary_label_change": "label_change",
        "boundary_missing_clock_pair": "missing_clock_pair",
        "boundary_nonpositive_time_delta": "nonpositive_time_delta",
    }

    for col, reason in reason_map.items():
        mask = out[col].fillna(False)
        out.loc[mask, "segment_boundary_reason"] = out.loc[
            mask, "segment_boundary_reason"
        ].apply(lambda old: _append_reason(old, reason))

    return out


def assign_segment_ids(
    df: pd.DataFrame,
    dataset_key: str,
    segment_id_prefix: Optional[str] = None,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Assign segment_index and segment_id using segment_boundary column.
    """
    out = df.copy() if copy else df

    if "segment_boundary" not in out.columns:
        raise KeyError("Missing 'segment_boundary'. Build boundary flags first.")

    prefix = segment_id_prefix or dataset_key

    out["segment_index"] = out["segment_boundary"].astype(int).cumsum() - 1
    out["segment_index"] = out["segment_index"].astype(int)

    out["segment_id"] = out["segment_index"].apply(
        lambda idx: f"{prefix}_seg_{int(idx):04d}"
    )

    out["within_segment_index"] = out.groupby("segment_id").cumcount()

    return out


def segment_single_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    role: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Segment one raw dataset DataFrame.

    Pipeline:
    1. parse time columns,
    2. compute time deltas and diagnostics,
    3. add preliminary delta_t_seconds,
    4. create segmentation boundary flags,
    5. assign segment_id,
    6. build preliminary validity mask.
    """
    out = df.copy()

    # Preserve original order. Do not sort yet.
    if "row_order_in_source" not in out.columns:
        out.insert(0, "row_order_in_source", np.arange(len(out), dtype=int))

    out = prepare_time_features(
        df=out,
        config=config,
        copy=False,
    )

    prefer_clock_delta = bool(
        _get_nested(config, "preprocessing.time.prefer_clock_delta", True)
    )
    out = add_delta_t_column(
        df=out,
        prefer_clock=prefer_clock_delta,
        copy=False,
    )

    out = build_segmentation_boundary_flags(
        df=out,
        dataset_key=dataset_key,
        config=config,
    )

    out = assign_segment_ids(
        df=out,
        dataset_key=dataset_key,
        segment_id_prefix=dataset_key,
        copy=False,
    )

    out = build_preliminary_validity_mask(
        df=out,
        config=config,
        segment_col="segment_id",
        copy=False,
    )

    out["dataset_role"] = role

    return out


def summarize_segmented_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    role: str,
    output_path: Path,
    config: Mapping[str, Any],
) -> SegmentationSummary:
    """
    Create segmentation summary for one dataset.
    """
    label_col = str(_get_nested(config, "dataset.label_column", "Data Type"))

    segment_lengths = df.groupby("segment_id").size()

    if len(segment_lengths) == 0:
        min_len = 0
        median_len = 0.0
        max_len = 0
    else:
        min_len = int(segment_lengths.min())
        median_len = float(segment_lengths.median())
        max_len = int(segment_lengths.max())

    boundary_reason_counts = _count_reason_strings(df["segment_boundary_reason"])

    time_diagnostics = summarize_time_diagnostics(df)
    validity_summary = summarize_validity_mask(df)

    return SegmentationSummary(
        dataset_key=dataset_key,
        role=role,
        output_path=str(output_path),
        rows=int(df.shape[0]),
        segment_count=int(df["segment_id"].nunique()),
        min_segment_length=min_len,
        median_segment_length=median_len,
        max_segment_length=max_len,
        label_counts=_safe_label_counts(df, label_col),
        boundary_reason_counts=boundary_reason_counts,
        time_diagnostics=time_diagnostics,
        validity_summary=validity_summary,
    )


def print_segmentation_summary(summary: SegmentationSummary) -> None:
    """
    Print segmentation summary for console inspection.
    """
    print("=" * 100)
    print(f"SEGMENTATION SUMMARY | {summary.dataset_key}")
    print("=" * 100)
    print(f"Role                    : {summary.role}")
    print(f"Rows                    : {summary.rows}")
    print(f"Segments                : {summary.segment_count}")
    print(f"Min segment length      : {summary.min_segment_length}")
    print(f"Median segment length   : {summary.median_segment_length}")
    print(f"Max segment length      : {summary.max_segment_length}")
    print(f"Label counts            : {summary.label_counts}")
    print(f"Boundary reasons        : {summary.boundary_reason_counts}")
    print(f"Time diagnostics        : {summary.time_diagnostics}")
    print(f"Validity summary        : {summary.validity_summary}")
    print(f"Saved to                : {summary.output_path}")
    print("=" * 100)


def get_segmented_output_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    """
    Resolve output paths for segmented CSV files.
    """
    interim_dir_value = _get_nested(config, "paths.interim_data_dir", "data/interim")
    interim_dir = resolve_project_path(config, interim_dir_value)
    ensure_dir(interim_dir)

    configured = _get_nested(config, "preprocessing.segmentation.output_files", {})

    defaults = {
        "dataset1": "dataset1_segmented.csv",
        "dataset1_normal": "dataset1_normal_segmented.csv",
        "dataset2": "dataset2_segmented.csv",
        "dataset3": "dataset3_segmented.csv",
    }

    output_paths: Dict[str, Path] = {}

    for key, default_file_name in defaults.items():
        file_name = default_file_name
        if isinstance(configured, Mapping) and key in configured:
            file_name = str(configured[key])

        output_paths[key] = (interim_dir / file_name).resolve()

    return output_paths


def save_segmented_dataset(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Save segmented dataset CSV.
    """
    return save_csv(df, output_path, index=False)


def run_trajectory_segmentation(
    config: Mapping[str, Any],
    bundle: Optional[RawDatasetBundle] = None,
    save_outputs: bool = True,
) -> FullSegmentationReport:
    """
    Main Step-3 function.

    Loads raw data if bundle is not provided, segments every dataset,
    saves outputs, prints summaries, and saves a JSON segmentation report.
    """
    if bundle is None:
        bundle = load_and_validate_raw_datasets(config=config)

    output_paths = get_segmented_output_paths(config)

    summaries: Dict[str, SegmentationSummary] = {}

    for dataset_key in bundle.keys():
        df_raw = bundle.get(dataset_key)
        role = bundle.specs[dataset_key].role

        segmented = segment_single_dataset(
            df=df_raw,
            dataset_key=dataset_key,
            role=role,
            config=config,
        )

        output_path = output_paths[dataset_key]

        if save_outputs:
            save_segmented_dataset(segmented, output_path)

        summary = summarize_segmented_dataset(
            df=segmented,
            dataset_key=dataset_key,
            role=role,
            output_path=output_path,
            config=config,
        )

        summaries[dataset_key] = summary
        print_segmentation_summary(summary)

    all_have_segments = all(summary.segment_count > 0 for summary in summaries.values())
    status = "PASSED" if all_have_segments else "FAILED_NO_SEGMENTS"

    report = FullSegmentationReport(
        dataset_summaries=summaries,
        final_step3_status=status,
    )

    report_path_value = _get_nested(
        config,
        "paths.step3_segmentation_json",
        "results/tables/step3_segmentation_summary.json",
    )
    report_path = resolve_project_path(config, report_path_value)

    if save_outputs:
        save_json(report.to_dict(), report_path, indent=2)
        print(f"Saved Step 3 segmentation JSON: {report_path}")

    print("=" * 100)
    print(f"Final Step 3 status: {status}")
    print("=" * 100)

    return report