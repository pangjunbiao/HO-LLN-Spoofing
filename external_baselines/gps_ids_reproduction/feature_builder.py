"""
Build locked GPS-IDS behavior-feature CSV files from segmented raw AV-GPS data.

Input contract
--------------
- Dataset 1/2/3 segmented raw CSVs, before clean-column pruning.
- Dataset-1 train/validation/test segment JSON files.
- No train_xi.csv, validation_xi.csv, test_xi.csv, or other xi artifacts.

Output contract
---------------
Each feature CSV contains, in exact order:
    segment_id, row_index, within_segment_index, delta_t, split,
    label, valid_mask, feature_complete_mask, target_yaw_missing,
    <15 locked GPS-IDS model features>

Rows are never dropped. `valid_mask` represents only row/segment/time integrity.
Missing model features do not invalidate event-evaluation rows. Feature
availability is recorded separately by `feature_complete_mask`, and target-yaw
availability is recorded by `target_yaw_missing`. None of these masks consumes
labels, attack transitions, nu_prelim, xi_nu, or future rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from external_baselines.gps_ids_reproduction.feature_contract import (
    FORBIDDEN_XI_INPUT_BASENAMES,
    GPS_IDS_MODEL_FEATURES,
    GPS_IDS_OUTPUT_METADATA_COLUMNS,
    GPSIDSFeatureContract,
    validate_contract,
)


DATASET_KEYS: Tuple[str, ...] = ("dataset1", "dataset2", "dataset3")
OUTPUT_SPLITS: Tuple[str, ...] = (
    "train",
    "validation",
    "test",
    "dataset2",
    "dataset3",
)

ROW_INDEX_ALIASES: Tuple[str, ...] = (
    "raw_row_index",
    "row_index_original",
    "row_order_in_source",
)
WITHIN_SEGMENT_INDEX_ALIASES: Tuple[str, ...] = (
    "within_segment_index",
)
DELTA_T_ALIASES: Tuple[str, ...] = (
    "delta_t_seconds",
    "delta_t",
)
SEGMENT_ID_ALIASES: Tuple[str, ...] = (
    "segment_id",
)
LABEL_ALIASES: Tuple[str, ...] = (
    "Data Type",
    "label",
)

PROOF_OF_PRE_PRUNING_COLUMNS: Tuple[str, ...] = (
    "GPS HDOP",
    "GPS VDOP",
    "Satellite Count",
    "Satellite Locks",
    "Throttle (%)",
)


@dataclass(frozen=True)
class FeatureFileSummary:
    split_name: str
    source_dataset: str
    source_path: str
    output_path: str
    rows: int
    columns: int
    segments: int
    normal_rows: int
    attack_rows: int
    valid_rows: int
    invalid_rows: int
    feature_complete_rows: int
    feature_incomplete_rows: int
    target_yaw_missing_rows: int
    missing_feature_counts: Dict[str, int]
    nonfinite_feature_counts: Dict[str, int]
    feature_hash: str
    output_sha256: str
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPSIDSFeatureBuildReport:
    contract_version: str
    feature_hash: str
    feature_names: List[str]
    output_column_order: List[str]
    source_paths: Dict[str, str]
    split_paths: Dict[str, str]
    file_summaries: Dict[str, FeatureFileSummary]
    dataset1_split_leakage_check: Dict[str, Any]
    leakage_assertions: Dict[str, Any]
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "feature_hash": self.feature_hash,
            "feature_names": list(self.feature_names),
            "output_column_order": list(self.output_column_order),
            "source_paths": dict(self.source_paths),
            "split_paths": dict(self.split_paths),
            "file_summaries": {
                key: summary.to_dict()
                for key, summary in self.file_summaries.items()
            },
            "dataset1_split_leakage_check": dict(
                self.dataset1_split_leakage_check
            ),
            "leakage_assertions": dict(self.leakage_assertions),
            "final_status": self.final_status,
        }


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_exact_alias(
    columns: Sequence[str],
    aliases: Sequence[str],
    logical_name: str,
) -> str:
    """Resolve a feature alias strictly; multiple feature aliases are unsafe."""
    present = [alias for alias in aliases if alias in columns]
    if len(present) == 0:
        raise KeyError(
            f"Missing required source column for {logical_name}; "
            f"accepted aliases={list(aliases)}"
        )
    if len(present) > 1:
        raise ValueError(
            f"Ambiguous source columns for {logical_name}: {present}. "
            "Keep exactly one accepted alias."
        )
    return present[0]


def _resolve_priority_identity_alias(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    logical_name: str,
) -> str:
    """
    Resolve identity metadata by locked priority and verify duplicate aliases.

    The reviewed segmentation pipeline intentionally preserves raw_row_index,
    row_index_original, and row_order_in_source together. Their coexistence is
    not ambiguity if they encode the same row identity.
    """
    present = [alias for alias in aliases if alias in frame.columns]
    if not present:
        raise KeyError(
            f"Missing required source column for {logical_name}; "
            f"accepted aliases={list(aliases)}"
        )

    selected = present[0]
    selected_numeric = pd.to_numeric(frame[selected], errors="coerce").to_numpy(dtype=float)

    for alternative in present[1:]:
        alternative_numeric = pd.to_numeric(
            frame[alternative],
            errors="coerce",
        ).to_numpy(dtype=float)

        same_nan = np.isnan(selected_numeric) & np.isnan(alternative_numeric)
        same_finite = (
            np.isfinite(selected_numeric)
            & np.isfinite(alternative_numeric)
            & np.isclose(
                selected_numeric,
                alternative_numeric,
                atol=0.0,
                rtol=0.0,
            )
        )
        if not np.all(same_nan | same_finite):
            mismatch_count = int((~(same_nan | same_finite)).sum())
            raise ValueError(
                f"Identity aliases for {logical_name} disagree: "
                f"{selected!r} vs {alternative!r} on {mismatch_count} rows."
            )

    return selected


def _validate_source_path(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    basename = path.name.lower()
    if basename in {name.lower() for name in FORBIDDEN_XI_INPUT_BASENAMES}:
        raise ValueError(
            f"Forbidden xi input supplied to GPS-IDS feature builder: {path}"
        )
    if "_xi" in basename or basename.endswith("xi.csv"):
        raise ValueError(
            f"GPS-IDS inputs must be segmented raw CSVs, not xi files: {path}"
        )
    if "segmented" not in basename:
        raise ValueError(
            f"GPS-IDS source must be a segmented raw CSV: {path}"
        )


def load_segmented_raw_csv(path: Path, dataset_key: str) -> pd.DataFrame:
    _validate_source_path(path)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"{dataset_key} segmented source is empty: {path}")

    missing_proof = [
        column
        for column in PROOF_OF_PRE_PRUNING_COLUMNS
        if column not in frame.columns
    ]
    if missing_proof:
        raise KeyError(
            f"{dataset_key} does not look like pre-pruning segmented raw data. "
            f"Missing proof columns: {missing_proof}"
        )
    return frame


def load_segment_set(path: Path, expected_split: str) -> Set[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError(f"Split file must contain a JSON object: {path}")

    split_name = str(payload.get("split_name", "")).strip().lower()
    accepted = {
        "train": {"train"},
        "validation": {"validation", "val"},
        "test": {"test"},
    }[expected_split]
    if split_name not in accepted:
        raise ValueError(
            f"Split file {path} declares split_name={split_name!r}; "
            f"expected one of {sorted(accepted)}."
        )

    values = payload.get("segments")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Split file has no nonempty segments list: {path}")

    segments = [str(value).strip() for value in values]
    if any(not value for value in segments):
        raise ValueError(f"Split file contains an empty segment ID: {path}")
    if len(segments) != len(set(segments)):
        raise ValueError(f"Split file contains duplicate segment IDs: {path}")
    return set(segments)


def validate_dataset1_split_sets(
    all_segment_ids: Sequence[str],
    train_segments: Set[str],
    validation_segments: Set[str],
    test_segments: Set[str],
) -> Dict[str, Any]:
    all_segments = set(str(value) for value in all_segment_ids)
    intersections = {
        "train_validation": sorted(train_segments & validation_segments),
        "train_test": sorted(train_segments & test_segments),
        "validation_test": sorted(validation_segments & test_segments),
    }
    overlap_free = all(len(values) == 0 for values in intersections.values())
    assigned = train_segments | validation_segments | test_segments
    missing = sorted(all_segments - assigned)
    extra = sorted(assigned - all_segments)
    passed = overlap_free and not missing and not extra

    result = {
        "all_segment_count": len(all_segments),
        "train_segment_count": len(train_segments),
        "validation_segment_count": len(validation_segments),
        "test_segment_count": len(test_segments),
        "intersections": intersections,
        "missing_segments": missing,
        "extra_segments": extra,
        "passed": bool(passed),
    }
    if not passed:
        raise ValueError(f"Dataset-1 split leakage/coverage check failed: {result}")
    return result


def _validate_contiguous_segments(segment_ids: np.ndarray) -> None:
    seen: Set[str] = set()
    previous: Optional[str] = None
    for position, raw_segment in enumerate(segment_ids):
        segment = str(raw_segment)
        if segment != previous:
            if segment in seen:
                raise ValueError(
                    f"Segment {segment!r} reappears at row position {position}; "
                    "source rows must remain in contiguous chronological blocks."
                )
            seen.add(segment)
            previous = segment


def _coerce_integer_identity(
    series: pd.Series,
    name: str,
    *,
    nonnegative: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    integer_like = finite & np.isclose(numeric, np.rint(numeric), atol=0.0, rtol=0.0)
    if nonnegative:
        integer_like &= numeric >= 0

    output = np.zeros(numeric.size, dtype=np.int64)
    output[integer_like] = np.rint(numeric[integer_like]).astype(np.int64)
    return output, integer_like


def _canonical_delta_t(
    frame: pd.DataFrame,
    segment_ids: np.ndarray,
    source_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(frame[source_column], errors="coerce").to_numpy(dtype=float)
    output = numeric.copy()
    valid = np.isfinite(numeric) & (numeric >= 0.0)

    segment_start = np.ones(segment_ids.size, dtype=bool)
    if segment_ids.size > 1:
        segment_start[1:] = segment_ids[1:] != segment_ids[:-1]

    # The first row of each segment has no preceding interval by definition.
    output[segment_start] = 0.0
    valid[segment_start] = True

    # Preserve row alignment while making delta_t consumable by the unified
    # evaluator. Invalid non-start intervals become zero and invalidate the row.
    output[~valid] = 0.0
    return output.astype(float), valid


def _validate_monotonic_identity(
    segment_ids: np.ndarray,
    row_indices: np.ndarray,
    within_indices: np.ndarray,
    row_valid: np.ndarray,
    within_valid: np.ndarray,
) -> None:
    if not np.all(row_valid):
        count = int((~row_valid).sum())
        raise ValueError(f"row_index has {count} invalid/noninteger rows.")
    if not np.all(within_valid):
        count = int((~within_valid).sum())
        raise ValueError(
            f"within_segment_index has {count} invalid/noninteger rows."
        )

    row_keys = [
        (str(segment), int(row))
        for segment, row in zip(segment_ids, row_indices)
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("Duplicate (segment_id, row_index) identities found.")

    within_keys = [
        (str(segment), int(index))
        for segment, index in zip(segment_ids, within_indices)
    ]
    if len(within_keys) != len(set(within_keys)):
        raise ValueError(
            "Duplicate (segment_id, within_segment_index) identities found."
        )

    start = 0
    while start < segment_ids.size:
        segment = segment_ids[start]
        end = start + 1
        while end < segment_ids.size and segment_ids[end] == segment:
            end += 1
        if end - start > 1:
            if np.any(np.diff(row_indices[start:end]) <= 0):
                raise ValueError(
                    f"row_index is not increasing inside segment {segment!r}."
                )
            if np.any(np.diff(within_indices[start:end]) <= 0):
                raise ValueError(
                    "within_segment_index is not increasing inside segment "
                    f"{segment!r}."
                )
        start = end


def build_canonical_feature_frame(
    raw_frame: pd.DataFrame,
    *,
    dataset_key: str,
    split_name: str,
    contract: GPSIDSFeatureContract,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    validate_contract(contract)
    columns = list(raw_frame.columns)

    segment_column = _resolve_exact_alias(
        columns, SEGMENT_ID_ALIASES, "segment_id"
    )
    row_column = _resolve_priority_identity_alias(
        raw_frame, ROW_INDEX_ALIASES, "row_index"
    )
    within_column = _resolve_exact_alias(
        columns,
        WITHIN_SEGMENT_INDEX_ALIASES,
        "within_segment_index",
    )
    delta_column = _resolve_exact_alias(
        columns, DELTA_T_ALIASES, "delta_t"
    )
    label_column = _resolve_exact_alias(
        columns, LABEL_ALIASES, "label"
    )

    resolved_feature_columns: Dict[str, str] = {}
    for feature_name in contract.final_model_feature_names:
        resolved_feature_columns[feature_name] = _resolve_exact_alias(
            columns,
            contract.raw_column_aliases[feature_name],
            feature_name,
        )

    segment_series = raw_frame[segment_column]
    if segment_series.isna().any():
        raise ValueError(f"{dataset_key} has missing segment IDs.")
    segment_ids = segment_series.astype(str).str.strip().to_numpy(dtype=object)
    if any(not str(value) or str(value).lower() == "nan" for value in segment_ids):
        raise ValueError(f"{dataset_key} has empty/invalid segment IDs.")
    _validate_contiguous_segments(segment_ids)

    row_indices, row_valid = _coerce_integer_identity(
        raw_frame[row_column],
        "row_index",
    )
    within_indices, within_valid = _coerce_integer_identity(
        raw_frame[within_column],
        "within_segment_index",
    )
    _validate_monotonic_identity(
        segment_ids,
        row_indices,
        within_indices,
        row_valid,
        within_valid,
    )

    delta_t, delta_valid = _canonical_delta_t(
        raw_frame,
        segment_ids,
        delta_column,
    )

    labels_numeric = pd.to_numeric(
        raw_frame[label_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    label_valid = (
        np.isfinite(labels_numeric)
        & np.isin(labels_numeric, [0.0, 1.0])
    )
    if not np.all(label_valid):
        count = int((~label_valid).sum())
        raise ValueError(
            f"{dataset_key} has {count} invalid/nonbinary labels."
        )
    labels = labels_numeric.astype(np.int8)

    output = pd.DataFrame(
        {
            "segment_id": segment_ids.astype(str),
            "row_index": row_indices,
            "within_segment_index": within_indices,
            "delta_t": delta_t,
            "split": str(split_name),
            "label": labels,
        }
    )

    missing_counts: Dict[str, int] = {}
    nonfinite_counts: Dict[str, int] = {}
    feature_complete = np.ones(len(raw_frame), dtype=bool)
    target_yaw_missing = np.zeros(len(raw_frame), dtype=bool)

    for feature_name in contract.final_model_feature_names:
        source_column = resolved_feature_columns[feature_name]
        numeric = pd.to_numeric(
            raw_frame[source_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite = np.isfinite(numeric)
        missing_counts[feature_name] = int(raw_frame[source_column].isna().sum())
        nonfinite_counts[feature_name] = int((~finite).sum())
        feature_complete &= finite
        if feature_name == "target_yaw_deg":
            target_yaw_missing = ~finite
        output[feature_name] = numeric

    # Evaluation validity is deliberately independent of feature missingness,
    # labels, and attack-transition metadata. Identity violations are rejected
    # earlier; non-start invalid delta_t rows are retained but masked.
    output["valid_mask"] = (
        row_valid
        & within_valid
        & delta_valid
    ).astype(np.int8)

    # Missingness metadata is exported for transparent train-only
    # preprocessing in the classifier step. These columns are never in X.
    output["feature_complete_mask"] = feature_complete.astype(np.int8)
    output["target_yaw_missing"] = target_yaw_missing.astype(np.int8)

    ordered_columns = [
        *GPS_IDS_OUTPUT_METADATA_COLUMNS,
        *contract.final_model_feature_names,
    ]
    output = output.loc[:, ordered_columns]

    # The classifier feature matrix is selected only from the locked feature
    # contract; canonical metadata and raw leakage columns are excluded.
    overlap = sorted(
        set(contract.final_model_feature_names)
        & set(contract.excluded_model_columns)
    )
    if overlap:
        raise AssertionError(
            f"Locked model features overlap excluded columns: {overlap}"
        )

    summary = {
        "dataset_key": dataset_key,
        "split_name": split_name,
        "rows": int(len(output)),
        "segments": int(output["segment_id"].nunique()),
        "normal_rows": int((output["label"] == 0).sum()),
        "attack_rows": int((output["label"] == 1).sum()),
        "valid_rows": int((output["valid_mask"] == 1).sum()),
        "invalid_rows": int((output["valid_mask"] == 0).sum()),
        "feature_complete_rows": int(
            (output["feature_complete_mask"] == 1).sum()
        ),
        "feature_incomplete_rows": int(
            (output["feature_complete_mask"] == 0).sum()
        ),
        "target_yaw_missing_rows": int(
            (output["target_yaw_missing"] == 1).sum()
        ),
        "missing_feature_counts": missing_counts,
        "nonfinite_feature_counts": nonfinite_counts,
        "resolved_feature_columns": resolved_feature_columns,
        "identity_columns": {
            "segment_id": segment_column,
            "row_index": row_column,
            "within_segment_index": within_column,
            "delta_t": delta_column,
            "label": label_column,
        },
        "label_used_to_construct_valid_mask": False,
        "feature_missingness_used_to_construct_valid_mask": False,
        "feature_complete_mask_is_metadata_only": True,
        "target_yaw_missing_is_metadata_only": True,
        "attack_transition_columns_used": False,
        "future_rows_used": False,
        "rows_dropped": 0,
    }
    return output, summary


def _subset_by_segments(
    dataset1_frame: pd.DataFrame,
    segment_column: str,
    segment_set: Set[str],
) -> pd.DataFrame:
    mask = dataset1_frame[segment_column].astype(str).isin(segment_set)
    subset = dataset1_frame.loc[mask].copy()
    if subset.empty:
        raise ValueError("Dataset-1 split produced an empty frame.")
    return subset


def build_gps_ids_feature_files(
    *,
    contract: GPSIDSFeatureContract,
    source_paths: Mapping[str, Path],
    split_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> GPSIDSFeatureBuildReport:
    validate_contract(contract)

    missing_sources = sorted(set(DATASET_KEYS) - set(source_paths))
    if missing_sources:
        raise KeyError(f"Missing source paths: {missing_sources}")
    missing_outputs = sorted(set(OUTPUT_SPLITS) - set(output_paths))
    if missing_outputs:
        raise KeyError(f"Missing output paths: {missing_outputs}")

    dataset1 = load_segmented_raw_csv(
        Path(source_paths["dataset1"]),
        "dataset1",
    )
    dataset2 = load_segmented_raw_csv(
        Path(source_paths["dataset2"]),
        "dataset2",
    )
    dataset3 = load_segmented_raw_csv(
        Path(source_paths["dataset3"]),
        "dataset3",
    )

    segment_column = _resolve_exact_alias(
        dataset1.columns,
        SEGMENT_ID_ALIASES,
        "segment_id",
    )

    train_segments = load_segment_set(
        Path(split_paths["train"]),
        "train",
    )
    validation_segments = load_segment_set(
        Path(split_paths["validation"]),
        "validation",
    )
    test_segments = load_segment_set(
        Path(split_paths["test"]),
        "test",
    )

    split_check = validate_dataset1_split_sets(
        all_segment_ids=dataset1[segment_column].astype(str).unique(),
        train_segments=train_segments,
        validation_segments=validation_segments,
        test_segments=test_segments,
    )

    raw_frames = {
        "train": _subset_by_segments(
            dataset1, segment_column, train_segments
        ),
        "validation": _subset_by_segments(
            dataset1, segment_column, validation_segments
        ),
        "test": _subset_by_segments(
            dataset1, segment_column, test_segments
        ),
        "dataset2": dataset2,
        "dataset3": dataset3,
    }
    source_dataset_names = {
        "train": "dataset1",
        "validation": "dataset1",
        "test": "dataset1",
        "dataset2": "dataset2",
        "dataset3": "dataset3",
    }

    summaries: Dict[str, FeatureFileSummary] = {}

    for split_name in OUTPUT_SPLITS:
        canonical, diagnostics = build_canonical_feature_frame(
            raw_frames[split_name],
            dataset_key=source_dataset_names[split_name],
            split_name=split_name,
            contract=contract,
        )

        output_path = Path(output_paths[split_name])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_csv(output_path, index=False)

        reloaded = pd.read_csv(output_path, low_memory=False)
        expected_columns = [
            *GPS_IDS_OUTPUT_METADATA_COLUMNS,
            *contract.final_model_feature_names,
        ]
        if reloaded.columns.tolist() != expected_columns:
            raise AssertionError(
                f"Output column order changed for {split_name}."
            )
        if len(reloaded) != len(canonical):
            raise AssertionError(
                f"Output row count changed for {split_name}."
            )

        summaries[split_name] = FeatureFileSummary(
            split_name=split_name,
            source_dataset=source_dataset_names[split_name],
            source_path=str(
                Path(source_paths[source_dataset_names[split_name]]).resolve()
            ),
            output_path=str(output_path.resolve()),
            rows=int(len(canonical)),
            columns=int(canonical.shape[1]),
            segments=int(canonical["segment_id"].nunique()),
            normal_rows=int((canonical["label"] == 0).sum()),
            attack_rows=int((canonical["label"] == 1).sum()),
            valid_rows=int((canonical["valid_mask"] == 1).sum()),
            invalid_rows=int((canonical["valid_mask"] == 0).sum()),
            feature_complete_rows=int(
                (canonical["feature_complete_mask"] == 1).sum()
            ),
            feature_incomplete_rows=int(
                (canonical["feature_complete_mask"] == 0).sum()
            ),
            target_yaw_missing_rows=int(
                (canonical["target_yaw_missing"] == 1).sum()
            ),
            missing_feature_counts=dict(
                diagnostics["missing_feature_counts"]
            ),
            nonfinite_feature_counts=dict(
                diagnostics["nonfinite_feature_counts"]
            ),
            feature_hash=contract.feature_hash,
            output_sha256=sha256_file(output_path),
            final_status="PASSED",
        )

    leakage_assertions = {
        "source_is_segmented_raw_pre_pruning": True,
        "xi_files_used_as_input": False,
        "label_in_model_features": False,
        "ekf_detector_in_model_features": False,
        "source_identity_in_model_features": False,
        "row_identity_in_model_features": False,
        "split_identity_in_model_features": False,
        "attack_derived_validity_used": False,
        "future_rows_used": False,
        "post_event_summary_features_used": False,
        "rows_dropped": False,
        "feature_scaling_fit_in_step4": False,
        "feature_imputation_fit_in_step4": False,
        "feature_missingness_changes_valid_mask": False,
        "feature_complete_mask_in_model_features": False,
        "target_yaw_missing_in_model_features": False,
    }

    return GPSIDSFeatureBuildReport(
        contract_version=contract.contract_version,
        feature_hash=contract.feature_hash,
        feature_names=list(contract.final_model_feature_names),
        output_column_order=[
            *GPS_IDS_OUTPUT_METADATA_COLUMNS,
            *contract.final_model_feature_names,
        ],
        source_paths={
            key: str(Path(value).resolve())
            for key, value in source_paths.items()
        },
        split_paths={
            key: str(Path(value).resolve())
            for key, value in split_paths.items()
        },
        file_summaries=summaries,
        dataset1_split_leakage_check=split_check,
        leakage_assertions=leakage_assertions,
        final_status="PASSED",
    )


__all__ = [
    "DATASET_KEYS",
    "OUTPUT_SPLITS",
    "FeatureFileSummary",
    "GPSIDSFeatureBuildReport",
    "build_canonical_feature_frame",
    "build_gps_ids_feature_files",
    "load_segment_set",
    "load_segmented_raw_csv",
    "sha256_file",
    "validate_dataset1_split_sets",
]
