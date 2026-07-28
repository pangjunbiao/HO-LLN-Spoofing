"""
Source-aware Dataset-1 segment splitting for the AV-GPS causal spoofing project.

Step 4 purpose:
- use only Dataset-1 for main development,
- split by segment_id, not rows,
- prevent same-segment leakage between train/validation/internal-test,
- preserve Dataset-2 and Dataset-3 untouched for external/case-study evaluation,
- save split segment IDs and split summary into data/splits/.

Important:
This step does not build final model inputs and does not use raw shortcut
columns as model features. It only creates leakage-safe segment IDs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_csv, save_json


SPLIT_NAMES = ["train", "val", "test"]


@dataclass
class SplitConfig:
    """Configuration for Dataset-1 segment-level split."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    n_trials: int
    require_attack_each_split: bool
    require_normal_each_split: bool
    min_segments_train: int
    min_segments_val: int
    min_segments_test: int


@dataclass
class SplitFile:
    """One split file summary."""

    split_name: str
    output_path: str
    segment_count: int
    row_count: int
    normal_count: int
    attack_count: int
    attack_rate: float
    segments: List[str]


@dataclass
class Dataset1SplitReport:
    """Full Step-4 split report."""

    dataset_key: str
    source_file: str
    segmented_input_path: str
    total_segments: int
    total_rows: int
    total_normal: int
    total_attack: int
    total_attack_rate: float
    split_config: Dict[str, Any]
    split_files: Dict[str, SplitFile]
    leakage_check: Dict[str, Any]
    external_datasets_untouched: Dict[str, bool]
    segment_metadata_path: str
    split_summary_path: str
    final_step4_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "source_file": self.source_file,
            "segmented_input_path": self.segmented_input_path,
            "total_segments": self.total_segments,
            "total_rows": self.total_rows,
            "total_normal": self.total_normal,
            "total_attack": self.total_attack,
            "total_attack_rate": self.total_attack_rate,
            "split_config": self.split_config,
            "split_files": {
                split_name: asdict(split_file)
                for split_name, split_file in self.split_files.items()
            },
            "leakage_check": self.leakage_check,
            "external_datasets_untouched": self.external_datasets_untouched,
            "segment_metadata_path": self.segment_metadata_path,
            "split_summary_path": self.split_summary_path,
            "final_step4_status": self.final_step4_status,
        }


def _safe_int(value: Any) -> int:
    """Convert numeric-like value to int safely."""
    if pd.isna(value):
        return 0
    return int(value)


def _safe_float(value: Any, digits: int = 6) -> float:
    """Convert numeric-like value to rounded float safely."""
    if pd.isna(value):
        return 0.0
    return round(float(value), digits)


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, float]:
    """Normalize split ratios so they sum to one."""
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }

    if any(value <= 0 for value in ratios.values()):
        raise ValueError(f"All split ratios must be positive. Got: {ratios}")

    total = sum(ratios.values())

    if total <= 0:
        raise ValueError("Split ratio total must be positive.")

    return {key: value / total for key, value in ratios.items()}


def get_split_config(config: Mapping[str, Any]) -> SplitConfig:
    """
    Read Step-4 split config.

    Expected config keys are added in Part 2:

        training:
          split:
            train_ratio: 0.70
            val_ratio: 0.15
            test_ratio: 0.15
            random_seed: 42
            n_trials: 3000
            require_attack_each_split: true
            require_normal_each_split: true
    """
    default_seed = int(get_by_path(config, "seed.single_seed", 42))

    return SplitConfig(
        train_ratio=float(get_by_path(config, "training.split.train_ratio", 0.70)),
        val_ratio=float(get_by_path(config, "training.split.val_ratio", 0.15)),
        test_ratio=float(get_by_path(config, "training.split.test_ratio", 0.15)),
        random_seed=int(get_by_path(config, "training.split.random_seed", default_seed)),
        n_trials=int(get_by_path(config, "training.split.n_trials", 3000)),
        require_attack_each_split=bool(
            get_by_path(config, "training.split.require_attack_each_split", True)
        ),
        require_normal_each_split=bool(
            get_by_path(config, "training.split.require_normal_each_split", True)
        ),
        min_segments_train=int(get_by_path(config, "training.split.min_segments_train", 1)),
        min_segments_val=int(get_by_path(config, "training.split.min_segments_val", 1)),
        min_segments_test=int(get_by_path(config, "training.split.min_segments_test", 1)),
    )


def get_dataset1_segmented_path(config: Mapping[str, Any]) -> Path:
    """
    Resolve data/interim/dataset1_segmented.csv.
    """
    interim_dir_value = get_by_path(config, "paths.interim_data_dir", "data/interim")
    interim_dir = resolve_project_path(config, interim_dir_value)

    file_name = get_by_path(
        config,
        "dataset.segmented_files.dataset1",
        "dataset1_segmented.csv",
    )

    return (interim_dir / str(file_name)).resolve()


def get_split_output_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    """
    Resolve Step-4 split output paths.
    """
    splits_dir_value = get_by_path(config, "paths.splits_dir", "data/splits")
    splits_dir = resolve_project_path(config, splits_dir_value)
    ensure_dir(splits_dir)

    return {
        "train": (splits_dir / "dataset1_train_segments.json").resolve(),
        "val": (splits_dir / "dataset1_val_segments.json").resolve(),
        "test": (splits_dir / "dataset1_test_segments.json").resolve(),
        "summary": (splits_dir / "split_summary.json").resolve(),
        "segment_metadata": (splits_dir / "dataset1_segment_metadata.csv").resolve(),
    }


def load_dataset1_segmented(config: Mapping[str, Any]) -> pd.DataFrame:
    """
    Load segmented Dataset-1 from data/interim/.
    """
    path = get_dataset1_segmented_path(config)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset-1 segmented file not found: {path}\n"
            "Run Step 3 first: python main.py --mode step3"
        )

    df = pd.read_csv(path, low_memory=False)
    return df


def validate_dataset1_segmented(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """
    Validate that the input is segmented Dataset-1 and has required columns.
    """
    required_columns = [
        "segment_id",
        "within_segment_index",
        "source_key",
        "source_file",
        get_by_path(config, "dataset.label_column", "Data Type"),
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(
            "Segmented Dataset-1 is missing required columns: "
            f"{missing}. Run Step 3 again or inspect the segmented CSV."
        )

    source_keys = set(df["source_key"].dropna().astype(str).unique().tolist())
    if source_keys != {"dataset1"}:
        raise ValueError(
            "Step 4 must use Dataset-1 only. "
            f"Found source_key values: {sorted(source_keys)}"
        )

    if df["segment_id"].isna().any():
        raise ValueError("segment_id contains missing values.")

    if df["segment_id"].nunique() < 3:
        raise ValueError(
            "Need at least 3 segments to create train/val/test split."
        )


def build_dataset1_segment_metadata(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Build segment-level metadata table for Dataset-1.

    One row per segment_id.
    """
    label_col = str(get_by_path(config, "dataset.label_column", "Data Type"))
    normal_label = int(get_by_path(config, "dataset.normal_label", 0))
    attack_label = int(get_by_path(config, "dataset.attack_label", 1))

    records: List[Dict[str, Any]] = []

    grouped = df.groupby("segment_id", sort=False)

    for segment_id, group in grouped:
        labels = group[label_col]

        normal_count = int((labels == normal_label).sum())
        attack_count = int((labels == attack_label).sum())
        row_count = int(len(group))
        attack_rate = float(attack_count / row_count) if row_count > 0 else 0.0

        has_attack = attack_count > 0
        has_normal = normal_count > 0

        if has_attack and has_normal:
            label_profile = "mixed"
        elif has_attack:
            label_profile = "attack_only"
        elif has_normal:
            label_profile = "normal_only"
        else:
            label_profile = "unknown"

        if "raw_row_index" in group.columns:
            start_row = int(group["raw_row_index"].iloc[0])
            end_row = int(group["raw_row_index"].iloc[-1])
        elif "row_index_original" in group.columns:
            start_row = int(group["row_index_original"].iloc[0])
            end_row = int(group["row_index_original"].iloc[-1])
        else:
            start_row = int(group.index[0])
            end_row = int(group.index[-1])

        record = {
            "segment_id": str(segment_id),
            "source_key": str(group["source_key"].iloc[0]) if "source_key" in group.columns else "",
            "source_file": str(group["source_file"].iloc[0]) if "source_file" in group.columns else "",
            "row_count": row_count,
            "normal_count": normal_count,
            "attack_count": attack_count,
            "attack_rate": round(attack_rate, 6),
            "has_normal": bool(has_normal),
            "has_attack": bool(has_attack),
            "label_profile": label_profile,
            "start_row_index": start_row,
            "end_row_index": end_row,
            "first_within_segment_index": int(group["within_segment_index"].iloc[0]),
            "last_within_segment_index": int(group["within_segment_index"].iloc[-1]),
        }

        if "nu_prelim" in group.columns:
            record["valid_transition_count"] = int((group["nu_prelim"] == 1).sum())
            record["invalid_transition_count"] = int((group["nu_prelim"] == 0).sum())

        if "Clock Date" in group.columns:
            record["first_clock_date"] = str(group["Clock Date"].iloc[0])
            record["last_clock_date"] = str(group["Clock Date"].iloc[-1])

        if "Clock Time" in group.columns:
            record["first_clock_time"] = str(group["Clock Time"].iloc[0])
            record["last_clock_time"] = str(group["Clock Time"].iloc[-1])

        if "runtime_seconds" in group.columns:
            record["first_runtime_seconds"] = _safe_float(group["runtime_seconds"].iloc[0])
            record["last_runtime_seconds"] = _safe_float(group["runtime_seconds"].iloc[-1])

        records.append(record)

    metadata = pd.DataFrame(records)

    metadata = metadata.sort_values(
        by=["start_row_index", "segment_id"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return metadata


def _split_stats(segment_metadata: pd.DataFrame, segment_ids: Sequence[str]) -> Dict[str, Any]:
    """Compute summary stats for a candidate split."""
    sub = segment_metadata[segment_metadata["segment_id"].isin(segment_ids)]

    row_count = int(sub["row_count"].sum())
    normal_count = int(sub["normal_count"].sum())
    attack_count = int(sub["attack_count"].sum())
    segment_count = int(len(sub))
    attack_rate = float(attack_count / row_count) if row_count > 0 else 0.0

    return {
        "segment_count": segment_count,
        "row_count": row_count,
        "normal_count": normal_count,
        "attack_count": attack_count,
        "attack_rate": attack_rate,
    }


def _full_assignment_score(
    assignment: Dict[str, List[str]],
    segment_metadata: pd.DataFrame,
    ratios: Dict[str, float],
    split_config: SplitConfig,
) -> float:
    """
    Score a complete split assignment.

    Lower is better.
    """
    total_rows = float(segment_metadata["row_count"].sum())
    total_attack = float(segment_metadata["attack_count"].sum())
    total_normal = float(segment_metadata["normal_count"].sum())
    total_segments = float(len(segment_metadata))

    total_attack_rate = total_attack / total_rows if total_rows > 0 else 0.0

    min_segments = {
        "train": split_config.min_segments_train,
        "val": split_config.min_segments_val,
        "test": split_config.min_segments_test,
    }

    score = 0.0

    for split_name in SPLIT_NAMES:
        stats = _split_stats(segment_metadata, assignment[split_name])

        target_row_ratio = ratios[split_name]
        target_segment_ratio = ratios[split_name]

        row_ratio = stats["row_count"] / total_rows if total_rows > 0 else 0.0
        segment_ratio = stats["segment_count"] / total_segments if total_segments > 0 else 0.0
        attack_count = float(stats["attack_count"])
        normal_count = float(stats["normal_count"])

        attack_ratio_share = attack_count / total_attack if total_attack > 0 else 0.0
        normal_ratio_share = normal_count / total_normal if total_normal > 0 else 0.0

        attack_rate = float(stats["attack_rate"])

        # Main balance terms.
        score += 4.0 * abs(row_ratio - target_row_ratio)
        score += 2.0 * abs(segment_ratio - target_segment_ratio)
        score += 3.0 * abs(attack_ratio_share - target_row_ratio)
        score += 1.5 * abs(normal_ratio_share - target_row_ratio)
        score += 1.0 * abs(attack_rate - total_attack_rate)

        # Hard-ish penalties.
        if stats["segment_count"] < min_segments[split_name]:
            score += 100.0

        if stats["row_count"] <= 0:
            score += 100.0

        if split_config.require_attack_each_split and total_attack > 0 and attack_count <= 0:
            score += 100.0

        if split_config.require_normal_each_split and total_normal > 0 and normal_count <= 0:
            score += 100.0

    return float(score)


def _greedy_candidate_assignment(
    segment_metadata: pd.DataFrame,
    ratios: Dict[str, float],
    split_config: SplitConfig,
    rng: np.random.Generator,
    trial_index: int,
) -> Dict[str, List[str]]:
    """
    Build one candidate segment assignment using randomized greedy search.
    """
    assignment: Dict[str, List[str]] = {split: [] for split in SPLIT_NAMES}

    if trial_index == 0:
        ordered = segment_metadata.sort_values(
            by=["row_count", "attack_count"],
            ascending=[False, False],
        )
    elif trial_index == 1:
        ordered = segment_metadata.sort_values(
            by=["attack_count", "row_count"],
            ascending=[False, False],
        )
    elif trial_index == 2:
        ordered = segment_metadata.sort_values(
            by=["attack_rate", "row_count"],
            ascending=[False, False],
        )
    else:
        ordered = segment_metadata.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )

    segment_ids = ordered["segment_id"].astype(str).tolist()

    for i, segment_id in enumerate(segment_ids):
        remaining_after_this = len(segment_ids) - i - 1
        empty_splits = [split for split in SPLIT_NAMES if len(assignment[split]) == 0]

        # Force filling empty splits if needed.
        if len(empty_splits) > remaining_after_this:
            candidate_splits = empty_splits
        else:
            candidate_splits = SPLIT_NAMES

        best_split = None
        best_score = float("inf")

        for split_name in candidate_splits:
            trial_assignment = {
                name: list(values) for name, values in assignment.items()
            }
            trial_assignment[split_name].append(segment_id)

            score = _full_assignment_score(
                assignment=trial_assignment,
                segment_metadata=segment_metadata,
                ratios=ratios,
                split_config=split_config,
            )

            # Tiny random tie-breaker.
            score += float(rng.random()) * 1e-6

            if score < best_score:
                best_score = score
                best_split = split_name

        if best_split is None:
            raise RuntimeError("Failed to choose split during greedy assignment.")

        assignment[best_split].append(segment_id)

    return assignment


def split_dataset1_segments(
    segment_metadata: pd.DataFrame,
    split_config: SplitConfig,
) -> Dict[str, List[str]]:
    """
    Find a balanced leakage-safe train/val/test split at segment level.

    The split is optimized to balance:
    - row counts,
    - segment counts,
    - attack counts,
    - normal counts,
    - attack-rate distribution.
    """
    ratios = _normalize_ratios(
        train_ratio=split_config.train_ratio,
        val_ratio=split_config.val_ratio,
        test_ratio=split_config.test_ratio,
    )

    rng = np.random.default_rng(split_config.random_seed)

    best_assignment: Optional[Dict[str, List[str]]] = None
    best_score = float("inf")

    n_trials = max(10, int(split_config.n_trials))

    for trial_index in range(n_trials):
        assignment = _greedy_candidate_assignment(
            segment_metadata=segment_metadata,
            ratios=ratios,
            split_config=split_config,
            rng=rng,
            trial_index=trial_index,
        )

        score = _full_assignment_score(
            assignment=assignment,
            segment_metadata=segment_metadata,
            ratios=ratios,
            split_config=split_config,
        )

        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Failed to create Dataset-1 segment split.")

    return {
        split_name: sorted(segment_ids)
        for split_name, segment_ids in best_assignment.items()
    }


def check_segment_leakage(
    assignment: Dict[str, List[str]],
    all_segment_ids: Sequence[str],
) -> Dict[str, Any]:
    """
    Check that no segment appears in more than one split.
    """
    split_sets = {
        split: set(segment_ids)
        for split, segment_ids in assignment.items()
    }

    overlaps = {
        "train_val": sorted(list(split_sets["train"] & split_sets["val"])),
        "train_test": sorted(list(split_sets["train"] & split_sets["test"])),
        "val_test": sorted(list(split_sets["val"] & split_sets["test"])),
    }

    assigned_all = set().union(*split_sets.values())
    expected_all = set(all_segment_ids)

    missing_segments = sorted(list(expected_all - assigned_all))
    extra_segments = sorted(list(assigned_all - expected_all))

    no_overlap = all(len(values) == 0 for values in overlaps.values())
    complete_coverage = len(missing_segments) == 0 and len(extra_segments) == 0

    return {
        "no_overlap": no_overlap,
        "complete_coverage": complete_coverage,
        "overlaps": overlaps,
        "missing_segments": missing_segments,
        "extra_segments": extra_segments,
        "train_segment_count": len(split_sets["train"]),
        "val_segment_count": len(split_sets["val"]),
        "test_segment_count": len(split_sets["test"]),
        "passed": bool(no_overlap and complete_coverage),
    }


def _make_split_file(
    split_name: str,
    segment_ids: List[str],
    segment_metadata: pd.DataFrame,
    output_path: Path,
) -> SplitFile:
    """Create SplitFile dataclass."""
    stats = _split_stats(segment_metadata, segment_ids)

    return SplitFile(
        split_name=split_name,
        output_path=str(output_path),
        segment_count=int(stats["segment_count"]),
        row_count=int(stats["row_count"]),
        normal_count=int(stats["normal_count"]),
        attack_count=int(stats["attack_count"]),
        attack_rate=round(float(stats["attack_rate"]), 6),
        segments=list(segment_ids),
    )


def save_split_files(
    assignment: Dict[str, List[str]],
    segment_metadata: pd.DataFrame,
    split_config: SplitConfig,
    config: Mapping[str, Any],
    input_path: Path,
) -> Dataset1SplitReport:
    """
    Save train/val/test segment JSON files and split_summary.json.
    """
    output_paths = get_split_output_paths(config)

    save_csv(segment_metadata, output_paths["segment_metadata"], index=False)

    split_files: Dict[str, SplitFile] = {}

    for split_name in SPLIT_NAMES:
        split_file = _make_split_file(
            split_name=split_name,
            segment_ids=assignment[split_name],
            segment_metadata=segment_metadata,
            output_path=output_paths[split_name],
        )

        split_files[split_name] = split_file

        save_json(
            {
                "dataset_key": "dataset1",
                "split_name": split_name,
                "segment_count": split_file.segment_count,
                "row_count": split_file.row_count,
                "normal_count": split_file.normal_count,
                "attack_count": split_file.attack_count,
                "attack_rate": split_file.attack_rate,
                "segments": split_file.segments,
                "note": "Segment-level split. Do not split Dataset-1 by rows.",
            },
            output_paths[split_name],
            indent=2,
        )

    all_segment_ids = segment_metadata["segment_id"].astype(str).tolist()
    leakage_check = check_segment_leakage(
        assignment=assignment,
        all_segment_ids=all_segment_ids,
    )

    total_rows = int(segment_metadata["row_count"].sum())
    total_normal = int(segment_metadata["normal_count"].sum())
    total_attack = int(segment_metadata["attack_count"].sum())
    total_attack_rate = float(total_attack / total_rows) if total_rows > 0 else 0.0

    source_files = sorted(segment_metadata["source_file"].astype(str).unique().tolist())
    source_file = source_files[0] if len(source_files) == 1 else ";".join(source_files)

    split_config_dict = asdict(split_config)
    split_config_dict["normalized_ratios"] = _normalize_ratios(
        split_config.train_ratio,
        split_config.val_ratio,
        split_config.test_ratio,
    )

    external_untouched = {
        "dataset2_untouched_for_external_source_shift_test": True,
        "dataset3_untouched_for_online_case_study": True,
        "dataset1_normal_not_used_as_independent_supervised_data": True,
    }

    status = "PASSED" if leakage_check["passed"] else "FAILED_LEAKAGE_CHECK"

    report = Dataset1SplitReport(
        dataset_key="dataset1",
        source_file=source_file,
        segmented_input_path=str(input_path),
        total_segments=int(len(segment_metadata)),
        total_rows=total_rows,
        total_normal=total_normal,
        total_attack=total_attack,
        total_attack_rate=round(total_attack_rate, 6),
        split_config=split_config_dict,
        split_files=split_files,
        leakage_check=leakage_check,
        external_datasets_untouched=external_untouched,
        segment_metadata_path=str(output_paths["segment_metadata"]),
        split_summary_path=str(output_paths["summary"]),
        final_step4_status=status,
    )

    save_json(report.to_dict(), output_paths["summary"], indent=2)

    return report


def print_split_report(report: Dataset1SplitReport) -> None:
    """Print Step-4 split report to console."""
    print("=" * 100)
    print("STEP 4 SOURCE-AWARE DATASET-1 SPLIT REPORT")
    print("=" * 100)
    print(f"Dataset key                 : {report.dataset_key}")
    print(f"Input segmented file         : {report.segmented_input_path}")
    print(f"Total segments               : {report.total_segments}")
    print(f"Total rows                   : {report.total_rows}")
    print(f"Total normal rows            : {report.total_normal}")
    print(f"Total attack rows            : {report.total_attack}")
    print(f"Total attack rate            : {report.total_attack_rate}")
    print("-" * 100)

    print(
        f"{'Split':10s} | {'Segments':>8s} | {'Rows':>10s} | "
        f"{'Normal':>10s} | {'Attack':>10s} | {'Attack rate':>12s}"
    )
    print("-" * 100)

    for split_name in SPLIT_NAMES:
        split_file = report.split_files[split_name]
        print(
            f"{split_name:10s} | "
            f"{split_file.segment_count:8d} | "
            f"{split_file.row_count:10d} | "
            f"{split_file.normal_count:10d} | "
            f"{split_file.attack_count:10d} | "
            f"{split_file.attack_rate:12.6f}"
        )

    print("-" * 100)
    print(f"No segment overlap           : {report.leakage_check['no_overlap']}")
    print(f"Complete segment coverage    : {report.leakage_check['complete_coverage']}")
    print(f"Leakage check passed         : {report.leakage_check['passed']}")
    print(f"Dataset-2 untouched          : {report.external_datasets_untouched['dataset2_untouched_for_external_source_shift_test']}")
    print(f"Dataset-3 untouched          : {report.external_datasets_untouched['dataset3_untouched_for_online_case_study']}")
    print(f"Segment metadata saved to    : {report.segment_metadata_path}")
    print(f"Split summary saved to       : {report.split_summary_path}")
    print(f"Final Step 4 status          : {report.final_step4_status}")
    print("=" * 100)


def run_dataset1_source_aware_split(
    config: Mapping[str, Any],
    save_outputs: bool = True,
) -> Dataset1SplitReport:
    """
    Main Step-4 entry point.

    Loads data/interim/dataset1_segmented.csv, creates leakage-safe segment-level
    train/val/test splits, saves JSON outputs, and prints a summary.
    """
    input_path = get_dataset1_segmented_path(config)
    df = load_dataset1_segmented(config)
    validate_dataset1_segmented(df, config)

    segment_metadata = build_dataset1_segment_metadata(df, config)

    split_config = get_split_config(config)

    assignment = split_dataset1_segments(
        segment_metadata=segment_metadata,
        split_config=split_config,
    )

    if save_outputs:
        report = save_split_files(
            assignment=assignment,
            segment_metadata=segment_metadata,
            split_config=split_config,
            config=config,
            input_path=input_path,
        )
    else:
        # Save_outputs=False is mostly for tests.
        output_paths = get_split_output_paths(config)
        split_files = {
            split_name: _make_split_file(
                split_name=split_name,
                segment_ids=assignment[split_name],
                segment_metadata=segment_metadata,
                output_path=output_paths[split_name],
            )
            for split_name in SPLIT_NAMES
        }

        leakage_check = check_segment_leakage(
            assignment=assignment,
            all_segment_ids=segment_metadata["segment_id"].astype(str).tolist(),
        )

        total_rows = int(segment_metadata["row_count"].sum())
        total_normal = int(segment_metadata["normal_count"].sum())
        total_attack = int(segment_metadata["attack_count"].sum())
        total_attack_rate = float(total_attack / total_rows) if total_rows > 0 else 0.0

        report = Dataset1SplitReport(
            dataset_key="dataset1",
            source_file="",
            segmented_input_path=str(input_path),
            total_segments=int(len(segment_metadata)),
            total_rows=total_rows,
            total_normal=total_normal,
            total_attack=total_attack,
            total_attack_rate=round(total_attack_rate, 6),
            split_config=asdict(split_config),
            split_files=split_files,
            leakage_check=leakage_check,
            external_datasets_untouched={
                "dataset2_untouched_for_external_source_shift_test": True,
                "dataset3_untouched_for_online_case_study": True,
                "dataset1_normal_not_used_as_independent_supervised_data": True,
            },
            segment_metadata_path=str(output_paths["segment_metadata"]),
            split_summary_path=str(output_paths["summary"]),
            final_step4_status="PASSED" if leakage_check["passed"] else "FAILED_LEAKAGE_CHECK",
        )

    print_split_report(report)

    if report.final_step4_status != "PASSED":
        raise RuntimeError(
            f"Step 4 failed with status: {report.final_step4_status}"
        )

    return report