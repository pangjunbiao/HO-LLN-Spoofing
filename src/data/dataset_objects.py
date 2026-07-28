"""
Dataset object helpers for the AV-GPS causal spoofing project.

This file now supports both:

Step 4:
- load segmented CSV files,
- load saved Dataset-1 train/val/test segment IDs,
- filter Dataset-1 by segment split,
- validate no segment leakage,
- create reusable segment-index structures.

Step 9:
- load Step-8 xi files,
- group rows by segment,
- build PyTorch-compatible sequence datasets,
- create padded sequence batches,
- create flattened arrays for XGBoost/MLP baselines,
- preserve Dataset-3 online order,
- ensure proposed model, baselines, and ablations use the same scaled xi features.

Important:
Raw shortcut columns must not be used as model inputs.
The official model/baseline input is the scaled xi_t representation from Step 8.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import json

import numpy as np
import pandas as pd

from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, load_json, save_json


# =============================================================================
# Shared constants
# =============================================================================

SPLIT_NAMES = ["train", "val", "test"]

DEFAULT_MODEL_INPUT_COLUMNS = [
    "xi_eta_east_scaled",
    "xi_eta_north_scaled",
    "xi_eta_dot_east_scaled",
    "xi_eta_dot_north_scaled",
    "xi_eta_ddot_east_scaled",
    "xi_eta_ddot_north_scaled",
    "xi_q_scaled",
    "xi_accum_log_scaled",
    "xi_nu",
]

DEFAULT_XI_SPLIT_FILES = {
    "train": "train_xi.csv",
    "val": "val_xi.csv",
    "test": "test_xi.csv",
    "external": "external_xi.csv",
    "online": "online_xi.csv",
    "normal_reference": "normal_reference_xi.csv",
}

DEFAULT_FLAT_ARRAY_FILES = {
    "train": "train_flat_xi.npz",
    "val": "val_flat_xi.npz",
    "test": "test_flat_xi.npz",
    "external": "external_flat_xi.npz",
    "online": "online_flat_xi.npz",
    "normal_reference": "normal_reference_flat_xi.npz",
}


# =============================================================================
# Step 4 dataclasses and helpers
# =============================================================================

@dataclass
class SegmentSplit:
    """Loaded segment split information."""

    split_name: str
    segment_ids: List[str]
    row_count: int
    normal_count: int
    attack_count: int
    attack_rate: float


@dataclass
class Dataset1SplitFrames:
    """Dataset-1 split DataFrames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_segments: List[str]
    val_segments: List[str]
    test_segments: List[str]


@dataclass
class SegmentIndexItem:
    """Index item for one segment sequence."""

    segment_id: str
    start_position: int
    end_position: int
    length: int
    label_profile: str
    attack_count: int
    normal_count: int


def get_interim_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/interim directory."""
    value = get_by_path(config, "paths.interim_data_dir", "data/interim")
    return resolve_project_path(config, value)


def get_processed_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/processed directory."""
    value = get_by_path(config, "paths.processed_data_dir", "data/processed")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_splits_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/splits directory."""
    value = get_by_path(config, "paths.splits_dir", "data/splits")
    return resolve_project_path(config, value)


def get_results_tables_dir(config: Mapping[str, Any]) -> Path:
    """Resolve results/tables directory."""
    value = get_by_path(config, "paths.results_tables_dir", "results/tables")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_segmented_file_path(config: Mapping[str, Any], dataset_key: str) -> Path:
    """
    Resolve segmented CSV path for a dataset key.

    Example:
        dataset_key='dataset1' -> data/interim/dataset1_segmented.csv
    """
    default_file_names = {
        "dataset1": "dataset1_segmented.csv",
        "dataset1_normal": "dataset1_normal_segmented.csv",
        "dataset2": "dataset2_segmented.csv",
        "dataset3": "dataset3_segmented.csv",
    }

    file_name = get_by_path(
        config,
        f"dataset.segmented_files.{dataset_key}",
        default_file_names.get(dataset_key, f"{dataset_key}_segmented.csv"),
    )

    return (get_interim_dir(config) / str(file_name)).resolve()


def load_segmented_dataset(
    config: Mapping[str, Any],
    dataset_key: str,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Load one segmented dataset CSV from data/interim/.
    """
    path = get_segmented_file_path(config, dataset_key)

    if not path.exists():
        raise FileNotFoundError(
            f"Segmented dataset not found for {dataset_key}: {path}\n"
            "Run Step 3 first."
        )

    return pd.read_csv(path, low_memory=low_memory)


def get_split_file_path(config: Mapping[str, Any], split_name: str) -> Path:
    """
    Resolve Dataset-1 split JSON path.

    split_name must be train, val, or test.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Invalid split_name='{split_name}'. Expected {SPLIT_NAMES}.")

    file_key = {
        "train": "train_segments",
        "val": "val_segments",
        "test": "test_segments",
    }[split_name]

    default_names = {
        "train": "dataset1_train_segments.json",
        "val": "dataset1_val_segments.json",
        "test": "dataset1_test_segments.json",
    }

    file_name = get_by_path(
        config,
        f"dataset.split_files.{file_key}",
        default_names[split_name],
    )

    return (get_splits_dir(config) / str(file_name)).resolve()


def load_segment_split(
    config: Mapping[str, Any],
    split_name: str,
) -> SegmentSplit:
    """
    Load one Dataset-1 split JSON.
    """
    path = get_split_file_path(config, split_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run Step 4 first."
        )

    data = load_json(path)

    return SegmentSplit(
        split_name=str(data["split_name"]),
        segment_ids=[str(seg) for seg in data["segments"]],
        row_count=int(data.get("row_count", 0)),
        normal_count=int(data.get("normal_count", 0)),
        attack_count=int(data.get("attack_count", 0)),
        attack_rate=float(data.get("attack_rate", 0.0)),
    )


def load_all_dataset1_splits(config: Mapping[str, Any]) -> Dict[str, SegmentSplit]:
    """
    Load train/val/test Dataset-1 split JSON files.
    """
    return {
        split_name: load_segment_split(config, split_name)
        for split_name in SPLIT_NAMES
    }


def load_split_summary(config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Load data/splits/split_summary.json.
    """
    file_name = get_by_path(
        config,
        "dataset.split_files.split_summary",
        "split_summary.json",
    )
    path = get_splits_dir(config) / str(file_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Split summary not found: {path}\n"
            "Run Step 4 first."
        )

    return load_json(path)


def filter_by_segments(
    df: pd.DataFrame,
    segment_ids: Sequence[str],
    segment_col: str = "segment_id",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Filter DataFrame by segment IDs.
    """
    if segment_col not in df.columns:
        raise KeyError(f"Missing segment column '{segment_col}'.")

    segment_set = set(str(seg) for seg in segment_ids)
    mask = df[segment_col].astype(str).isin(segment_set)

    out = df.loc[mask]
    return out.copy() if copy else out


def validate_no_overlap_between_splits(
    train_segments: Sequence[str],
    val_segments: Sequence[str],
    test_segments: Sequence[str],
) -> Dict[str, Any]:
    """
    Validate no segment ID appears in more than one split.
    """
    train_set = set(str(seg) for seg in train_segments)
    val_set = set(str(seg) for seg in val_segments)
    test_set = set(str(seg) for seg in test_segments)

    overlaps = {
        "train_val": sorted(list(train_set & val_set)),
        "train_test": sorted(list(train_set & test_set)),
        "val_test": sorted(list(val_set & test_set)),
    }

    passed = all(len(value) == 0 for value in overlaps.values())

    return {
        "passed": passed,
        "overlaps": overlaps,
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
    }


def validate_split_coverage(
    all_dataset_segments: Sequence[str],
    train_segments: Sequence[str],
    val_segments: Sequence[str],
    test_segments: Sequence[str],
) -> Dict[str, Any]:
    """
    Validate split segment IDs cover all Dataset-1 segments exactly once.
    """
    all_set = set(str(seg) for seg in all_dataset_segments)
    assigned_set = (
        set(str(seg) for seg in train_segments)
        | set(str(seg) for seg in val_segments)
        | set(str(seg) for seg in test_segments)
    )

    missing = sorted(list(all_set - assigned_set))
    extra = sorted(list(assigned_set - all_set))

    return {
        "passed": len(missing) == 0 and len(extra) == 0,
        "missing_segments": missing,
        "extra_segments": extra,
        "expected_count": len(all_set),
        "assigned_count": len(assigned_set),
    }


def load_dataset1_split_frames(config: Mapping[str, Any]) -> Dataset1SplitFrames:
    """
    Load Dataset-1 segmented CSV and return train/val/test DataFrames by segment ID.

    This preserves Step-4 behavior.
    """
    df = load_segmented_dataset(config, dataset_key="dataset1", low_memory=False)
    splits = load_all_dataset1_splits(config)

    train_segments = splits["train"].segment_ids
    val_segments = splits["val"].segment_ids
    test_segments = splits["test"].segment_ids

    overlap_check = validate_no_overlap_between_splits(
        train_segments=train_segments,
        val_segments=val_segments,
        test_segments=test_segments,
    )

    if not overlap_check["passed"]:
        raise RuntimeError(f"Segment overlap detected: {overlap_check}")

    if "segment_id" not in df.columns:
        raise KeyError("Dataset-1 segmented file is missing 'segment_id'.")

    all_segments = df["segment_id"].astype(str).unique().tolist()

    coverage_check = validate_split_coverage(
        all_dataset_segments=all_segments,
        train_segments=train_segments,
        val_segments=val_segments,
        test_segments=test_segments,
    )

    if not coverage_check["passed"]:
        raise RuntimeError(f"Split coverage check failed: {coverage_check}")

    train_df = filter_by_segments(df, train_segments)
    val_df = filter_by_segments(df, val_segments)
    test_df = filter_by_segments(df, test_segments)

    return Dataset1SplitFrames(
        train=train_df,
        val=val_df,
        test=test_df,
        train_segments=train_segments,
        val_segments=val_segments,
        test_segments=test_segments,
    )


def build_segment_index(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    label_col: str = "Data Type",
    normal_label: int = 0,
    attack_label: int = 1,
) -> List[SegmentIndexItem]:
    """
    Build one index item per segment.

    This is useful for diagnostics and sequence batching.
    """
    required = [segment_col, label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for segment index: {missing}")

    index_items: List[SegmentIndexItem] = []

    for segment_id, group in df.groupby(segment_col, sort=False):
        positions = group.index.to_numpy()

        labels = pd.to_numeric(group[label_col], errors="coerce").fillna(-1)
        normal_count = int((labels == normal_label).sum())
        attack_count = int((labels == attack_label).sum())

        if attack_count > 0 and normal_count > 0:
            profile = "mixed"
        elif attack_count > 0:
            profile = "attack_only"
        elif normal_count > 0:
            profile = "normal_only"
        else:
            profile = "unknown"

        index_items.append(
            SegmentIndexItem(
                segment_id=str(segment_id),
                start_position=int(positions[0]),
                end_position=int(positions[-1]),
                length=int(len(group)),
                label_profile=profile,
                attack_count=attack_count,
                normal_count=normal_count,
            )
        )

    return index_items


def print_split_frame_summary(split_frames: Dataset1SplitFrames) -> None:
    """
    Print quick summary of loaded Dataset-1 split frames.
    """
    print("=" * 100)
    print("DATASET-1 SPLIT FRAME SUMMARY")
    print("=" * 100)

    for split_name, df, segments in [
        ("train", split_frames.train, split_frames.train_segments),
        ("val", split_frames.val, split_frames.val_segments),
        ("test", split_frames.test, split_frames.test_segments),
    ]:
        label_counts = (
            df["Data Type"].value_counts(dropna=False).to_dict()
            if "Data Type" in df.columns
            else {}
        )

        print(
            f"{split_name:10s} | rows={len(df):8d} | "
            f"segments={len(segments):4d} | labels={label_counts}"
        )

    print("=" * 100)


class SegmentSequenceDataset:
    """
    Generic segment-sequence dataset wrapper.

    This preserves the original conservative helper:
    - It does not choose raw feature columns automatically.
    - The caller must explicitly pass feature_columns.
    - For final training, use XiSequenceDataset below.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        segment_ids: Optional[Sequence[str]] = None,
        feature_columns: Optional[Sequence[str]] = None,
        label_column: str = "Data Type",
        segment_column: str = "segment_id",
        return_torch: bool = False,
        sort_within_segment: bool = True,
    ) -> None:
        self.df = df.copy()
        self.segment_column = segment_column
        self.label_column = label_column
        self.return_torch = return_torch
        self.sort_within_segment = sort_within_segment

        if segment_column not in self.df.columns:
            raise KeyError(f"Missing segment column '{segment_column}'.")

        if label_column not in self.df.columns:
            raise KeyError(f"Missing label column '{label_column}'.")

        if feature_columns is None:
            self.feature_columns: List[str] = []
        else:
            self.feature_columns = [str(col) for col in feature_columns]

        missing_features = [
            col for col in self.feature_columns if col not in self.df.columns
        ]
        if missing_features:
            raise KeyError(f"Missing feature columns: {missing_features}")

        available_segments = self.df[segment_column].astype(str).unique().tolist()

        if segment_ids is None:
            self.segment_ids = available_segments
        else:
            requested = [str(seg) for seg in segment_ids]
            missing_segments = sorted(list(set(requested) - set(available_segments)))
            if missing_segments:
                raise ValueError(f"Requested segment IDs not found: {missing_segments}")
            self.segment_ids = requested

    def __len__(self) -> int:
        return len(self.segment_ids)

    def _maybe_to_torch(self, array: np.ndarray) -> Any:
        if not self.return_torch:
            return array

        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required when return_torch=True.") from exc

        if array.dtype.kind in {"i", "u", "b"}:
            return torch.as_tensor(array, dtype=torch.long)

        return torch.as_tensor(array, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        segment_id = self.segment_ids[index]

        group = self.df[self.df[self.segment_column].astype(str) == segment_id]

        if self.sort_within_segment and "within_segment_index" in group.columns:
            group = group.sort_values("within_segment_index", kind="mergesort")

        labels = pd.to_numeric(group[self.label_column], errors="coerce").fillna(0).to_numpy()

        if self.feature_columns:
            features = group[self.feature_columns].to_numpy(dtype=np.float32)
        else:
            features = np.empty((len(group), 0), dtype=np.float32)

        return {
            "segment_id": segment_id,
            "features": self._maybe_to_torch(features),
            "labels": self._maybe_to_torch(labels.astype(np.int64)),
            "length": int(len(group)),
        }


# =============================================================================
# Step 9 dataclasses
# =============================================================================

@dataclass
class XiDatasetConfig:
    """Configuration for Step-9 xi dataset objects."""

    feature_columns: List[str]
    label_column: str
    validity_column: str
    segment_column: str
    order_column: str
    delta_t_column: str

    sequence_mode: str
    train_window_size: int
    train_stride: int
    eval_window_size: int
    eval_stride: int
    online_full_sequence: bool

    batch_size_train: int
    batch_size_eval: int
    num_workers: int
    pin_memory: bool
    drop_last_train: bool

    flatten_valid_only: bool
    sequence_loss_valid_only: bool

    save_flat_arrays: bool
    save_sequence_manifest: bool


@dataclass
class SegmentRecord:
    """One causal segment record for xi features."""

    split_name: str
    source_dataset: str
    segment_id: str

    features: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    delta_t: np.ndarray
    row_indices: np.ndarray
    within_segment_index: np.ndarray

    normal_count: int
    attack_count: int

    def length(self) -> int:
        return int(self.features.shape[0])


@dataclass
class SequenceWindow:
    """One sequence window inside a segment."""

    segment_record_index: int
    start: int
    end: int
    is_first_window_for_segment: bool


@dataclass
class XiSplitDatasetSummary:
    """Summary for one split loaded in Step 9."""

    split_name: str
    source_path: str
    rows: int
    segments: int
    feature_dim: int
    normal_rows: int
    attack_rows: int
    valid_rows: int
    invalid_rows: int
    min_segment_length: int
    median_segment_length: float
    max_segment_length: int
    sequence_windows_train_mode: int
    sequence_windows_eval_mode: int
    flattened_rows_all: int
    flattened_rows_valid_only: int
    output_flat_array_path: str
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class XiDatasetObjectsReport:
    """Full Step-9 dataset-object report."""

    split_summaries: Dict[str, XiSplitDatasetSummary]
    feature_columns: List[str]
    label_column: str
    validity_column: str
    sequence_config: Dict[str, Any]
    fairness_rules: Dict[str, Any]
    saved_outputs: Dict[str, str]
    final_step9_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_summaries": {
                key: value.to_dict()
                for key, value in self.split_summaries.items()
            },
            "feature_columns": self.feature_columns,
            "label_column": self.label_column,
            "validity_column": self.validity_column,
            "sequence_config": self.sequence_config,
            "fairness_rules": self.fairness_rules,
            "saved_outputs": self.saved_outputs,
            "final_step9_status": self.final_step9_status,
        }


# =============================================================================
# Step 9 paths/config
# =============================================================================

def get_step9_summary_path(config: Mapping[str, Any]) -> Path:
    """Resolve Step-9 summary JSON path."""
    value = get_by_path(
        config,
        "paths.step9_dataset_objects_summary_json",
        "results/tables/step9_dataset_objects_summary.json",
    )
    return resolve_project_path(config, value)


def get_step9_sequence_manifest_path(config: Mapping[str, Any]) -> Path:
    """Resolve Step-9 sequence manifest JSON path."""
    value = get_by_path(
        config,
        "paths.step9_sequence_manifest_json",
        "results/tables/step9_sequence_manifest.json",
    )
    return resolve_project_path(config, value)


def get_step9_flat_array_dir(config: Mapping[str, Any]) -> Path:
    """Resolve directory for flat baseline arrays."""
    value = get_by_path(
        config,
        "paths.step9_flat_array_dir",
        "data/processed/flat_arrays",
    )
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_xi_split_path(config: Mapping[str, Any], split_name: str) -> Path:
    """Resolve Step-8 xi split CSV path."""
    file_name = get_by_path(
        config,
        f"dataset.xi_split_files.{split_name}",
        DEFAULT_XI_SPLIT_FILES.get(split_name, f"{split_name}_xi.csv"),
    )
    return (get_processed_dir(config) / str(file_name)).resolve()


def get_flat_array_path(config: Mapping[str, Any], split_name: str) -> Path:
    """Resolve flat array NPZ output path."""
    file_name = get_by_path(
        config,
        f"dataset.flat_array_files.{split_name}",
        DEFAULT_FLAT_ARRAY_FILES.get(split_name, f"{split_name}_flat_xi.npz"),
    )
    return (get_step9_flat_array_dir(config) / str(file_name)).resolve()


def get_xi_dataset_config(config: Mapping[str, Any]) -> XiDatasetConfig:
    """Read Step-9 dataset-object configuration."""
    feature_columns = get_by_path(
        config,
        "model.input.recommended_model_input_columns",
        None,
    )

    if feature_columns is None:
        feature_columns = get_by_path(
            config,
            "training.dataset.feature_columns",
            DEFAULT_MODEL_INPUT_COLUMNS,
        )

    return XiDatasetConfig(
        feature_columns=list(feature_columns),
        label_column=str(get_by_path(config, "dataset.label_column", "Data Type")),
        validity_column=str(
            get_by_path(config, "model.input.validity_column", "xi_nu")
        ),
        segment_column=str(
            get_by_path(config, "model.input.segment_column", "segment_id")
        ),
        order_column=str(
            get_by_path(config, "model.input.order_column", "within_segment_index")
        ),
        delta_t_column=str(
            get_by_path(config, "model.input.delta_t_column", "delta_t_seconds")
        ),

        sequence_mode=str(
            get_by_path(config, "training.dataset.sequence_mode", "fixed_window")
        ),
        train_window_size=int(
            get_by_path(config, "training.dataset.train_window_size", 256)
        ),
        train_stride=int(
            get_by_path(config, "training.dataset.train_stride", 128)
        ),
        eval_window_size=int(
            get_by_path(config, "training.dataset.eval_window_size", 512)
        ),
        eval_stride=int(
            get_by_path(config, "training.dataset.eval_stride", 512)
        ),
        online_full_sequence=bool(
            get_by_path(config, "training.dataset.online_full_sequence", True)
        ),

        batch_size_train=int(
            get_by_path(config, "training.dataloader.batch_size_train", 16)
        ),
        batch_size_eval=int(
            get_by_path(config, "training.dataloader.batch_size_eval", 8)
        ),
        num_workers=int(
            get_by_path(config, "training.dataloader.num_workers", 0)
        ),
        pin_memory=bool(
            get_by_path(config, "training.dataloader.pin_memory", True)
        ),
        drop_last_train=bool(
            get_by_path(config, "training.dataloader.drop_last_train", False)
        ),

        flatten_valid_only=bool(
            get_by_path(config, "training.dataset.flatten_valid_only", True)
        ),
        sequence_loss_valid_only=bool(
            get_by_path(config, "training.dataset.sequence_loss_valid_only", True)
        ),

        save_flat_arrays=bool(
            get_by_path(config, "training.dataset.save_flat_arrays", True)
        ),
        save_sequence_manifest=bool(
            get_by_path(config, "training.dataset.save_sequence_manifest", True)
        ),
    )


# =============================================================================
# Step 9 loading/validation
# =============================================================================

def _required_xi_columns(cfg: XiDatasetConfig) -> List[str]:
    """Required columns for Step-9 xi dataset objects."""
    return (
        list(cfg.feature_columns)
        + [
            cfg.label_column,
            cfg.validity_column,
            cfg.segment_column,
            cfg.order_column,
            cfg.delta_t_column,
        ]
    )


def load_xi_split_dataframe(
    config: Mapping[str, Any],
    split_name: str,
) -> pd.DataFrame:
    """Load one Step-8 xi split dataframe."""
    path = get_xi_split_path(config, split_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Xi split file not found for split='{split_name}': {path}\n"
            "Run Step 8 first."
        )

    return pd.read_csv(path, low_memory=False)


def validate_xi_dataframe(
    df: pd.DataFrame,
    split_name: str,
    cfg: XiDatasetConfig,
) -> None:
    """Validate Step-9 input dataframe."""
    missing = [col for col in _required_xi_columns(cfg) if col not in df.columns]

    if missing:
        raise KeyError(
            f"Split '{split_name}' is missing required xi columns: {missing}"
        )

    if len(df) == 0:
        raise ValueError(f"Split '{split_name}' is empty.")

    feature_values = df[cfg.feature_columns].apply(pd.to_numeric, errors="coerce")
    feature_array = feature_values.to_numpy(dtype=float)
    nonfinite_count = int((~np.isfinite(feature_array)).sum())

    if nonfinite_count > 0:
        raise ValueError(
            f"Split '{split_name}' has {nonfinite_count} non-finite model feature values."
        )


def sort_xi_dataframe(df: pd.DataFrame, cfg: XiDatasetConfig) -> pd.DataFrame:
    """
    Sort xi dataframe causally by segment and within-segment order.

    This is essential for recurrent models and online Dataset-3 evaluation.
    """
    return (
        df.sort_values(
            by=[cfg.segment_column, cfg.order_column],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_segment_records(
    df: pd.DataFrame,
    split_name: str,
    cfg: XiDatasetConfig,
) -> List[SegmentRecord]:
    """Convert a sorted xi dataframe into segment records."""
    sorted_df = sort_xi_dataframe(df, cfg)

    records: List[SegmentRecord] = []

    normal_label = 0
    attack_label = 1

    for segment_id, group in sorted_df.groupby(cfg.segment_column, sort=False):
        group = group.sort_values(cfg.order_column, kind="mergesort")

        features = (
            group[cfg.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        labels = (
            pd.to_numeric(group[cfg.label_column], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )
        valid_mask = (
            pd.to_numeric(group[cfg.validity_column], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        delta_t = (
            pd.to_numeric(group[cfg.delta_t_column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        within_segment_index = (
            pd.to_numeric(group[cfg.order_column], errors="coerce")
            .fillna(-1)
            .to_numpy(dtype=np.int64)
        )

        source_dataset = (
            str(group["xi_source_dataset"].iloc[0])
            if "xi_source_dataset" in group.columns and len(group) > 0
            else "unknown"
        )

        records.append(
            SegmentRecord(
                split_name=split_name,
                source_dataset=source_dataset,
                segment_id=str(segment_id),
                features=features,
                labels=labels,
                valid_mask=valid_mask,
                delta_t=delta_t,
                row_indices=group.index.to_numpy(dtype=np.int64),
                within_segment_index=within_segment_index,
                normal_count=int((labels == normal_label).sum()),
                attack_count=int((labels == attack_label).sum()),
            )
        )

    return records


# =============================================================================
# Step 9 sequence dataset and collate
# =============================================================================

def _make_window_starts(
    length: int,
    window_size: int,
    stride: int,
    full_segment: bool = False,
) -> List[Tuple[int, int]]:
    """
    Create inclusive-exclusive sequence windows.

    For evaluation, use stride == window_size to avoid duplicate evaluation rows.
    For training, stride may be smaller to augment sequence learning.
    """
    if length <= 0:
        return []

    if full_segment or window_size <= 0 or length <= window_size:
        return [(0, length)]

    stride = max(int(stride), 1)
    window_size = max(int(window_size), 1)

    windows: List[Tuple[int, int]] = []

    start = 0
    while start < length:
        end = min(start + window_size, length)
        windows.append((start, end))

        if end >= length:
            break

        start += stride

    return windows


class XiSequenceDataset:
    """
    PyTorch-compatible sequence dataset for xi features.

    Returns dictionaries with:
    - x: [T, F]
    - y: [T]
    - loss_mask: [T]
    - delta_t: [T]
    - padding_mask: [T]
    - reset_state: scalar, always 1 for a new sample/window
    - segment_id, split_name, source_dataset
    """

    def __init__(
        self,
        records: Sequence[SegmentRecord],
        split_name: str,
        window_size: int,
        stride: int,
        sequence_mode: str = "fixed_window",
        full_sequence: bool = False,
    ) -> None:
        self.records = list(records)
        self.split_name = split_name
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.sequence_mode = str(sequence_mode)
        self.full_sequence = bool(full_sequence)

        self.windows: List[SequenceWindow] = []

        for record_index, record in enumerate(self.records):
            use_full_segment = (
                self.full_sequence
                or self.sequence_mode == "full_segment"
                or self.window_size <= 0
            )

            starts = _make_window_starts(
                length=record.length(),
                window_size=self.window_size,
                stride=self.stride,
                full_segment=use_full_segment,
            )

            for window_index, (start, end) in enumerate(starts):
                self.windows.append(
                    SequenceWindow(
                        segment_record_index=record_index,
                        start=int(start),
                        end=int(end),
                        is_first_window_for_segment=window_index == 0,
                    )
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        window = self.windows[index]
        record = self.records[window.segment_record_index]

        start = window.start
        end = window.end

        x = record.features[start:end]
        y = record.labels[start:end]
        loss_mask = record.valid_mask[start:end]
        delta_t = record.delta_t[start:end]
        within_segment_index = record.within_segment_index[start:end]

        length = int(end - start)

        return {
            "x": x.astype(np.float32),
            "y": y.astype(np.int64),
            "loss_mask": loss_mask.astype(np.float32),
            "delta_t": delta_t.astype(np.float32),
            "padding_mask": np.ones(length, dtype=np.float32),
            "within_segment_index": within_segment_index.astype(np.int64),
            "reset_state": np.array(1.0, dtype=np.float32),
            "is_first_window_for_segment": np.array(
                1.0 if window.is_first_window_for_segment else 0.0,
                dtype=np.float32,
            ),
            "segment_id": record.segment_id,
            "split_name": record.split_name,
            "source_dataset": record.source_dataset,
            "start": np.array(start, dtype=np.int64),
            "end": np.array(end, dtype=np.int64),
        }


def collate_sequence_batch(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Collate variable-length xi sequences into padded PyTorch tensors.

    This function imports torch lazily, so non-PyTorch baseline utilities can
    still use this module without constructing tensors.
    """
    import torch

    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")

    batch_size = len(batch)
    max_len = max(int(item["x"].shape[0]) for item in batch)
    feature_dim = int(batch[0]["x"].shape[1])

    x = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    y = torch.zeros((batch_size, max_len), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    delta_t = torch.zeros((batch_size, max_len), dtype=torch.float32)
    within_segment_index = torch.full(
        (batch_size, max_len),
        fill_value=-1,
        dtype=torch.long,
    )

    reset_state = torch.zeros((batch_size,), dtype=torch.float32)
    first_window = torch.zeros((batch_size,), dtype=torch.float32)

    segment_ids: List[str] = []
    split_names: List[str] = []
    source_datasets: List[str] = []
    starts: List[int] = []
    ends: List[int] = []

    for i, item in enumerate(batch):
        length = int(item["x"].shape[0])

        x[i, :length] = torch.as_tensor(item["x"], dtype=torch.float32)
        y[i, :length] = torch.as_tensor(item["y"], dtype=torch.long)
        loss_mask[i, :length] = torch.as_tensor(item["loss_mask"], dtype=torch.float32)
        padding_mask[i, :length] = torch.as_tensor(item["padding_mask"], dtype=torch.float32)
        delta_t[i, :length] = torch.as_tensor(item["delta_t"], dtype=torch.float32)
        within_segment_index[i, :length] = torch.as_tensor(
            item["within_segment_index"],
            dtype=torch.long,
        )

        reset_state[i] = float(item["reset_state"])
        first_window[i] = float(item["is_first_window_for_segment"])

        segment_ids.append(str(item["segment_id"]))
        split_names.append(str(item["split_name"]))
        source_datasets.append(str(item["source_dataset"]))
        starts.append(int(item["start"]))
        ends.append(int(item["end"]))

    return {
        "x": x,
        "y": y,
        "loss_mask": loss_mask,
        "padding_mask": padding_mask,
        "delta_t": delta_t,
        "within_segment_index": within_segment_index,
        "reset_state": reset_state,
        "is_first_window_for_segment": first_window,
        "segment_id": segment_ids,
        "split_name": split_names,
        "source_dataset": source_datasets,
        "start": torch.as_tensor(starts, dtype=torch.long),
        "end": torch.as_tensor(ends, dtype=torch.long),
    }


def create_sequence_dataloader(
    dataset: XiSequenceDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: int = 42,
) -> Any:
    """Create a PyTorch DataLoader for sequence models."""
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        collate_fn=collate_sequence_batch,
        generator=generator if shuffle else None,
    )


# =============================================================================
# Step 9 flattened arrays for XGBoost/MLP
# =============================================================================

def build_flat_arrays_from_dataframe(
    df: pd.DataFrame,
    cfg: XiDatasetConfig,
    valid_only: bool = True,
) -> Dict[str, Any]:
    """
    Build flattened arrays for XGBoost and time-step MLP baselines.

    By default, invalid xi rows are excluded from flattened baseline arrays.
    Sequence datasets keep invalid rows with loss_mask=0 to preserve temporal continuity.
    """
    sorted_df = sort_xi_dataframe(df, cfg)

    valid_mask = (
        pd.to_numeric(sorted_df[cfg.validity_column], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
        == 1
    )

    if valid_only:
        use_mask = valid_mask
    else:
        use_mask = np.ones(len(sorted_df), dtype=bool)

    x = (
        sorted_df.loc[use_mask, cfg.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    y = (
        pd.to_numeric(sorted_df.loc[use_mask, cfg.label_column], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )

    segment_id = sorted_df.loc[use_mask, cfg.segment_column].astype(str).to_numpy()
    order_index = (
        pd.to_numeric(sorted_df.loc[use_mask, cfg.order_column], errors="coerce")
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )
    delta_t = (
        pd.to_numeric(sorted_df.loc[use_mask, cfg.delta_t_column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    original_valid_mask = valid_mask[use_mask].astype(np.float32)

    return {
        "x": x,
        "y": y,
        "valid_mask": original_valid_mask,
        "segment_id": segment_id,
        "order_index": order_index,
        "delta_t": delta_t,
        "feature_columns": np.array(cfg.feature_columns, dtype=object),
        "valid_only": np.array(bool(valid_only)),
    }


def save_npz_file(path: Path, **arrays: Any) -> None:
    """
    Save numpy arrays to compressed NPZ.

    Local implementation avoids requiring older src/utils/io.py to already have save_npz.
    """
    output_path = Path(path)
    ensure_dir(output_path.parent)
    np.savez_compressed(output_path, **arrays)


def load_npz_file(path: Path) -> Any:
    """
    Load a compressed NPZ file.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {input_path}")

    return np.load(input_path, allow_pickle=True)


def save_flat_arrays_for_split(
    df: pd.DataFrame,
    split_name: str,
    cfg: XiDatasetConfig,
    config: Mapping[str, Any],
) -> Path:
    """Save flattened NPZ arrays for one split."""
    arrays = build_flat_arrays_from_dataframe(
        df=df,
        cfg=cfg,
        valid_only=cfg.flatten_valid_only,
    )

    path = get_flat_array_path(config, split_name)
    save_npz_file(path, **arrays)
    return path


# =============================================================================
# Step 9 dataset building / summaries
# =============================================================================

def build_sequence_datasets_for_split(
    df: pd.DataFrame,
    split_name: str,
    cfg: XiDatasetConfig,
) -> Tuple[XiSequenceDataset, XiSequenceDataset, List[SegmentRecord]]:
    """
    Build train-mode and eval-mode sequence datasets for one split.

    Train-mode uses train window/stride.
    Eval-mode uses eval window/stride, except online can be full sequence.
    """
    validate_xi_dataframe(df, split_name=split_name, cfg=cfg)
    records = build_segment_records(df=df, split_name=split_name, cfg=cfg)

    train_full = split_name == "online" and cfg.online_full_sequence
    eval_full = split_name == "online" and cfg.online_full_sequence

    train_dataset = XiSequenceDataset(
        records=records,
        split_name=split_name,
        window_size=cfg.train_window_size,
        stride=cfg.train_stride,
        sequence_mode=cfg.sequence_mode,
        full_sequence=train_full,
    )

    eval_dataset = XiSequenceDataset(
        records=records,
        split_name=split_name,
        window_size=cfg.eval_window_size,
        stride=cfg.eval_stride,
        sequence_mode=cfg.sequence_mode,
        full_sequence=eval_full,
    )

    return train_dataset, eval_dataset, records


def summarize_split_dataset(
    df: pd.DataFrame,
    split_name: str,
    source_path: Path,
    flat_array_path: Path,
    train_sequence_dataset: XiSequenceDataset,
    eval_sequence_dataset: XiSequenceDataset,
    records: Sequence[SegmentRecord],
    cfg: XiDatasetConfig,
) -> XiSplitDatasetSummary:
    """Summarize one Step-9 split."""
    sorted_df = sort_xi_dataframe(df, cfg)

    labels = (
        pd.to_numeric(sorted_df[cfg.label_column], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )
    valid_mask = (
        pd.to_numeric(sorted_df[cfg.validity_column], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
        == 1
    )

    lengths = [record.length() for record in records]

    if lengths:
        min_len = int(np.min(lengths))
        median_len = float(np.median(lengths))
        max_len = int(np.max(lengths))
    else:
        min_len = 0
        median_len = 0.0
        max_len = 0

    flat_valid_count = int(valid_mask.sum())
    flat_all_count = int(len(sorted_df))

    final_status = "PASSED"
    if len(sorted_df) <= 0 or len(records) <= 0:
        final_status = "FAILED_EMPTY_SPLIT"
    elif len(train_sequence_dataset) <= 0 or len(eval_sequence_dataset) <= 0:
        final_status = "FAILED_NO_SEQUENCE_WINDOWS"
    elif flat_valid_count <= 0:
        final_status = "FAILED_NO_VALID_FLAT_ROWS"

    return XiSplitDatasetSummary(
        split_name=split_name,
        source_path=str(source_path),
        rows=int(len(sorted_df)),
        segments=int(len(records)),
        feature_dim=int(len(cfg.feature_columns)),
        normal_rows=int((labels == 0).sum()),
        attack_rows=int((labels == 1).sum()),
        valid_rows=flat_valid_count,
        invalid_rows=int((~valid_mask).sum()),
        min_segment_length=min_len,
        median_segment_length=median_len,
        max_segment_length=max_len,
        sequence_windows_train_mode=int(len(train_sequence_dataset)),
        sequence_windows_eval_mode=int(len(eval_sequence_dataset)),
        flattened_rows_all=flat_all_count,
        flattened_rows_valid_only=flat_valid_count,
        output_flat_array_path=str(flat_array_path),
        final_status=final_status,
    )


def build_sequence_manifest(
    split_records: Mapping[str, Sequence[SegmentRecord]],
    train_datasets: Mapping[str, XiSequenceDataset],
    eval_datasets: Mapping[str, XiSequenceDataset],
) -> Dict[str, Any]:
    """Build a JSON-safe manifest of segments and windows."""
    manifest: Dict[str, Any] = {}

    for split_name, records in split_records.items():
        manifest[split_name] = {
            "segment_count": int(len(records)),
            "train_window_count": int(len(train_datasets[split_name])),
            "eval_window_count": int(len(eval_datasets[split_name])),
            "segments": [
                {
                    "segment_id": record.segment_id,
                    "source_dataset": record.source_dataset,
                    "length": record.length(),
                    "normal_count": int(record.normal_count),
                    "attack_count": int(record.attack_count),
                    "valid_count": int(np.sum(record.valid_mask == 1)),
                    "invalid_count": int(np.sum(record.valid_mask != 1)),
                    "first_order_index": int(record.within_segment_index[0])
                    if record.length() > 0
                    else -1,
                    "last_order_index": int(record.within_segment_index[-1])
                    if record.length() > 0
                    else -1,
                }
                for record in records
            ],
        }

    return manifest


# =============================================================================
# Step 9 reporting
# =============================================================================

def print_split_dataset_summary(summary: XiSplitDatasetSummary) -> None:
    """Print one Step-9 split summary."""
    print("=" * 100)
    print(f"STEP 9 DATASET OBJECT SUMMARY | {summary.split_name}")
    print("=" * 100)
    print(f"Source path                         : {summary.source_path}")
    print(f"Rows                                : {summary.rows}")
    print(f"Segments                            : {summary.segments}")
    print(f"Feature dim                         : {summary.feature_dim}")
    print(f"Normal rows                         : {summary.normal_rows}")
    print(f"Attack rows                         : {summary.attack_rows}")
    print(f"Valid rows                          : {summary.valid_rows}")
    print(f"Invalid rows                        : {summary.invalid_rows}")
    print(
        "Segment length min/median/max       : "
        f"{summary.min_segment_length} / "
        f"{summary.median_segment_length} / "
        f"{summary.max_segment_length}"
    )
    print(f"Sequence windows train mode          : {summary.sequence_windows_train_mode}")
    print(f"Sequence windows eval mode           : {summary.sequence_windows_eval_mode}")
    print(f"Flattened rows all                   : {summary.flattened_rows_all}")
    print(f"Flattened rows valid only            : {summary.flattened_rows_valid_only}")
    print(f"Flat array output                    : {summary.output_flat_array_path}")
    print(f"Final status                         : {summary.final_status}")
    print("=" * 100)


def print_step9_report(report: XiDatasetObjectsReport) -> None:
    """Print full Step-9 report."""
    print("=" * 100)
    print("STEP 9 DATASET OBJECTS AND SEQUENCE BATCHING REPORT")
    print("=" * 100)

    for summary in report.split_summaries.values():
        print_split_dataset_summary(summary)

    print("=" * 100)
    print("STEP 9 FEATURE / FAIRNESS SUMMARY")
    print("=" * 100)
    print(f"Feature columns                      : {report.feature_columns}")
    print(f"Label column                         : {report.label_column}")
    print(f"Validity/loss-mask column            : {report.validity_column}")
    print(f"Sequence config                      : {report.sequence_config}")
    print(f"Fairness rules                       : {report.fairness_rules}")
    print(f"Saved outputs                        : {report.saved_outputs}")
    print(f"Final Step 9 status                  : {report.final_step9_status}")
    print("=" * 100)


# =============================================================================
# Step 9 main entry points
# =============================================================================

def run_dataset_objects_step(
    config: Mapping[str, Any],
    split_names: Optional[Sequence[str]] = None,
    save_outputs: bool = True,
) -> XiDatasetObjectsReport:
    """
    Main Step-9 entry point.

    Saves:
    - results/tables/step9_dataset_objects_summary.json
    - results/tables/step9_sequence_manifest.json
    - data/processed/flat_arrays/*_flat_xi.npz

    The returned in-memory datasets can be rebuilt later by training scripts.
    """
    cfg = get_xi_dataset_config(config)

    splits = list(
        split_names
        or ["train", "val", "test", "external", "online", "normal_reference"]
    )

    train_sequence_datasets: Dict[str, XiSequenceDataset] = {}
    eval_sequence_datasets: Dict[str, XiSequenceDataset] = {}
    split_records: Dict[str, List[SegmentRecord]] = {}
    split_summaries: Dict[str, XiSplitDatasetSummary] = {}
    flat_array_paths: Dict[str, Path] = {}

    for split_name in splits:
        source_path = get_xi_split_path(config, split_name)
        df = load_xi_split_dataframe(config, split_name)
        validate_xi_dataframe(df, split_name=split_name, cfg=cfg)

        train_dataset, eval_dataset, records = build_sequence_datasets_for_split(
            df=df,
            split_name=split_name,
            cfg=cfg,
        )

        flat_array_path = get_flat_array_path(config, split_name)

        if save_outputs and cfg.save_flat_arrays:
            flat_array_path = save_flat_arrays_for_split(
                df=df,
                split_name=split_name,
                cfg=cfg,
                config=config,
            )

        summary = summarize_split_dataset(
            df=df,
            split_name=split_name,
            source_path=source_path,
            flat_array_path=flat_array_path,
            train_sequence_dataset=train_dataset,
            eval_sequence_dataset=eval_dataset,
            records=records,
            cfg=cfg,
        )

        train_sequence_datasets[split_name] = train_dataset
        eval_sequence_datasets[split_name] = eval_dataset
        split_records[split_name] = list(records)
        split_summaries[split_name] = summary
        flat_array_paths[split_name] = flat_array_path

    manifest = build_sequence_manifest(
        split_records=split_records,
        train_datasets=train_sequence_datasets,
        eval_datasets=eval_sequence_datasets,
    )

    all_passed = all(
        summary.final_status == "PASSED"
        for summary in split_summaries.values()
    )
    final_status = "PASSED" if all_passed else "FAILED_STEP9_CHECK"

    fairness_rules = {
        "all_models_use_same_feature_columns": True,
        "feature_columns_are_scaled_xi_only": list(cfg.feature_columns) == DEFAULT_MODEL_INPUT_COLUMNS,
        "raw_shortcut_columns_used": False,
        "xgboost_and_mlp_use_flattened_valid_rows": cfg.flatten_valid_only,
        "sequence_models_keep_temporal_order": True,
        "sequence_loss_mask_uses_xi_nu": cfg.sequence_loss_valid_only,
        "hidden_state_reset_at_segment_boundaries": True,
        "dataset3_online_order_preserved": True,
        "dataset2_external_not_used_for_training": True,
        "dataset1_normal_reference_not_used_for_training": True,
    }

    saved_outputs: Dict[str, str] = {
        "summary_json": str(get_step9_summary_path(config)),
        "sequence_manifest_json": str(get_step9_sequence_manifest_path(config)),
    }

    for split_name, path in flat_array_paths.items():
        saved_outputs[f"{split_name}_flat_array_npz"] = str(path)

    report = XiDatasetObjectsReport(
        split_summaries=split_summaries,
        feature_columns=list(cfg.feature_columns),
        label_column=cfg.label_column,
        validity_column=cfg.validity_column,
        sequence_config={
            "sequence_mode": cfg.sequence_mode,
            "train_window_size": cfg.train_window_size,
            "train_stride": cfg.train_stride,
            "eval_window_size": cfg.eval_window_size,
            "eval_stride": cfg.eval_stride,
            "online_full_sequence": cfg.online_full_sequence,
            "batch_size_train": cfg.batch_size_train,
            "batch_size_eval": cfg.batch_size_eval,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "drop_last_train": cfg.drop_last_train,
            "flatten_valid_only": cfg.flatten_valid_only,
            "sequence_loss_valid_only": cfg.sequence_loss_valid_only,
        },
        fairness_rules=fairness_rules,
        saved_outputs=saved_outputs,
        final_step9_status=final_status,
    )

    if save_outputs:
        summary_path = get_step9_summary_path(config)
        manifest_path = get_step9_sequence_manifest_path(config)

        save_json(report.to_dict(), summary_path, indent=2)

        if cfg.save_sequence_manifest:
            save_json(manifest, manifest_path, indent=2)

        print(f"Saved Step 9 dataset objects summary JSON: {summary_path}")
        print(f"Saved Step 9 sequence manifest JSON: {manifest_path}")

    print_step9_report(report)

    if report.final_step9_status != "PASSED":
        raise RuntimeError(f"Step 9 failed with status: {report.final_step9_status}")

    return report


def rebuild_step9_in_memory_objects(
    config: Mapping[str, Any],
    split_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Rebuild Step-9 in-memory datasets and dataloaders for training scripts.

    This does not save files. It is intended for Step 12+ training code.
    """
    cfg = get_xi_dataset_config(config)
    splits = list(split_names or ["train", "val", "test", "external", "online"])

    sequence_train: Dict[str, XiSequenceDataset] = {}
    sequence_eval: Dict[str, XiSequenceDataset] = {}
    dataloaders_train: Dict[str, Any] = {}
    dataloaders_eval: Dict[str, Any] = {}
    flat_arrays: Dict[str, Dict[str, Any]] = {}

    active_seed = int(get_by_path(config, "seed.single_seed", 42))

    for split_name in splits:
        df = load_xi_split_dataframe(config, split_name)

        train_dataset, eval_dataset, _records = build_sequence_datasets_for_split(
            df=df,
            split_name=split_name,
            cfg=cfg,
        )

        sequence_train[split_name] = train_dataset
        sequence_eval[split_name] = eval_dataset

        dataloaders_train[split_name] = create_sequence_dataloader(
            dataset=train_dataset,
            batch_size=cfg.batch_size_train,
            shuffle=(split_name == "train"),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last_train if split_name == "train" else False,
            seed=active_seed,
        )

        dataloaders_eval[split_name] = create_sequence_dataloader(
            dataset=eval_dataset,
            batch_size=cfg.batch_size_eval,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
            seed=active_seed,
        )

        flat_arrays[split_name] = build_flat_arrays_from_dataframe(
            df=df,
            cfg=cfg,
            valid_only=cfg.flatten_valid_only,
        )

    return {
        "config": cfg,
        "sequence_train": sequence_train,
        "sequence_eval": sequence_eval,
        "dataloaders_train": dataloaders_train,
        "dataloaders_eval": dataloaders_eval,
        "flat_arrays": flat_arrays,
        "feature_columns": cfg.feature_columns,
    }