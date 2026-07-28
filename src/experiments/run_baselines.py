"""
Step 15: train and evaluate official baselines.

Official baselines:
1. XGBoost-xi
2. MLP-xi
3. LSTM-xi
4. GRU-xi
5. Causal-TCN-xi

Purpose:
- Compare the full proposed model against standard baselines using the exact same
  reconstructed xi_t feature vector.
- Use Dataset-1 train only for baseline training.
- Use Dataset-1 validation only for threshold/persistence selection.
- Evaluate Dataset-1 test, Dataset-2 external, and Dataset-3 online.
- Save baseline checkpoints, prediction bundles, threshold-selection artifacts,
  comparison tables, and a Step-15 summary.

Important fairness rules:
- No raw shortcut columns.
- No EKF Detector as model input.
- No Dataset-2 or Dataset-3 tuning.
- No synthetic Step-10 theta.
- Each baseline gets its own real validation-selected theta and persistence.
- This step does not run official proposed-model ablations; that is Step 16.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.baselines.baseline_factory import (
    BASELINE_DISPLAY_NAMES,
    BaselinePredictionRecord,
    BaselineRuntimeRecord,
    baseline_artifact_summary,
    collect_predictions_for_enabled_baselines,
    get_enabled_baseline_keys,
    get_step15_evaluation_splits,
    get_step15_retrain_policy,
    train_or_load_enabled_baselines,
)
from src.evaluation.evaluate_dataset1 import (
    EvaluationPredictionBundle,
    evaluate_bundle_with_threshold,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir
from src.utils.seed import set_global_seed


DEFAULT_STEP15_THETA_GRID = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

DEFAULT_STEP15_PERSISTENCE_VALUES = [1, 2, 3, 4, 5]


@dataclass
class Step15ThresholdSelection:
    """Validation-selected threshold/persistence for one baseline."""

    baseline_key: str
    model_name: str
    theta: float
    persistence: int
    objective: str
    selected_metric_value: float
    validation_metrics: Dict[str, Any]
    candidate_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step15EvaluationResult:
    """Evaluation result for one baseline and split."""

    baseline_key: str
    model_name: str
    split_name: str
    theta: float
    persistence: int
    metrics: Dict[str, Any]
    checkpoint_path: str
    prediction_path: str
    runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step15Paths:
    """Resolved Step-15 output paths."""

    tables_dir: Path
    models_dir: Path
    predictions_dir: Path

    summary_json: Path

    threshold_selection_json: Path
    threshold_selection_csv: Path
    threshold_candidates_csv: Path

    baseline_training_records_csv: Path
    baseline_prediction_records_csv: Path

    all_baselines_comparison_csv: Path

    dataset1_main_comparison_csv: Path
    dataset2_external_comparison_csv: Path
    dataset3_online_case_study_csv: Path

    def to_dict(self) -> Dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass
class Step15Config:
    """Step-15 experiment config."""

    experiment_name: str = "baseline_comparison_step15"
    retrain_policy: str = "reuse_if_exists"

    enabled_baselines: List[str] = field(default_factory=list)
    evaluation_splits: List[str] = field(
        default_factory=lambda: ["val", "test", "external", "online"]
    )
    comparison_splits: List[str] = field(
        default_factory=lambda: ["test", "external", "online"]
    )

    theta_grid: List[float] = field(default_factory=lambda: list(DEFAULT_STEP15_THETA_GRID))
    persistence_values: List[int] = field(default_factory=lambda: list(DEFAULT_STEP15_PERSISTENCE_VALUES))
    threshold_objective: str = "maximize_f1"

    save_predictions: bool = True
    save_json: bool = True
    save_csv: bool = True
    print_console_summary: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step15BaselinesSummary:
    """Final Step-15 summary."""

    final_status: str
    active_seed: int
    experiment_name: str
    retrain_policy: str

    enabled_baselines: List[str]
    evaluation_splits: List[str]
    comparison_splits: List[str]

    threshold_grid: List[float]
    persistence_values: List[int]
    threshold_objective: str

    training_records: List[Dict[str, Any]]
    prediction_records: List[Dict[str, Any]]
    threshold_selections: List[Dict[str, Any]]
    evaluation_results: List[Dict[str, Any]]

    artifact_summaries: Dict[str, Any]
    output_paths: Dict[str, str]

    leakage_rules: Dict[str, Any]
    runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve project-relative path."""
    return resolve_project_path(config, str(value))


def _json_safe(value: Any) -> Any:
    """Convert payload to JSON-safe values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    return value


def save_json_safe(payload: Mapping[str, Any], output_path: Path | str, indent: int = 2) -> Path:
    """Save JSON safely."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(dict(payload)), file, indent=indent)

    return output_path


def _safe_float(value: Any) -> Optional[float]:
    """Return finite float or None."""
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def _metric(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read finite metric as float."""
    value = _safe_float(metrics.get(key))

    if value is None:
        return float(default)

    return float(value)


def _int_metric(metrics: Mapping[str, Any], key: str, default: int = 0) -> int:
    """Read metric as int."""
    value = metrics.get(key, default)

    if value is None:
        return int(default)

    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float_list(values: Sequence[Any]) -> List[float]:
    """Convert sequence to finite float list."""
    out: List[float] = []

    for value in values:
        f = _safe_float(value)
        if f is not None:
            out.append(float(f))

    return out


def _as_int_list(values: Sequence[Any]) -> List[int]:
    """Convert sequence to int list."""
    out: List[int] = []

    for value in values:
        try:
            out.append(int(value))
        except Exception:
            pass

    return out


def build_step15_config(config: Mapping[str, Any]) -> Step15Config:
    """Build Step-15 experiment config."""
    theta_grid = get_by_path(
        config,
        "experiments.step15.threshold_selection.theta_grid",
        get_by_path(
            config,
            "training.threshold_selection.theta_grid",
            DEFAULT_STEP15_THETA_GRID,
        ),
    )

    persistence_values = get_by_path(
        config,
        "experiments.step15.threshold_selection.persistence_values",
        get_by_path(
            config,
            "training.threshold_selection.persistence_values",
            DEFAULT_STEP15_PERSISTENCE_VALUES,
        ),
    )

    evaluation_splits = list(
        get_by_path(
            config,
            "experiments.step15.evaluation_splits",
            get_step15_evaluation_splits(config),
        )
    )

    if "val" not in evaluation_splits and "validation" not in evaluation_splits:
        evaluation_splits = ["val"] + evaluation_splits

    evaluation_splits = ["val" if str(split) == "validation" else str(split) for split in evaluation_splits]

    comparison_splits = list(
        get_by_path(
            config,
            "experiments.step15.comparison_splits",
            ["test", "external", "online"],
        )
    )
    comparison_splits = [
        "val" if str(split) == "validation" else str(split)
        for split in comparison_splits
    ]

    return Step15Config(
        experiment_name=str(
            get_by_path(config, "experiments.step15.experiment_name", "baseline_comparison_step15")
        ),
        retrain_policy=get_step15_retrain_policy(config),
        enabled_baselines=get_enabled_baseline_keys(config),
        evaluation_splits=evaluation_splits,
        comparison_splits=comparison_splits,
        theta_grid=_as_float_list(theta_grid),
        persistence_values=_as_int_list(persistence_values),
        threshold_objective=str(
            get_by_path(
                config,
                "experiments.step15.threshold_selection.objective",
                get_by_path(config, "training.threshold_selection.objective", "maximize_f1"),
            )
        ),
        save_predictions=bool(get_by_path(config, "experiments.step15.save_predictions", True)),
        save_json=bool(get_by_path(config, "experiments.step15.save_json", True)),
        save_csv=bool(get_by_path(config, "experiments.step15.save_csv", True)),
        print_console_summary=bool(
            get_by_path(config, "experiments.step15.print_console_summary", True)
        ),
    )


def build_step15_paths(config: Mapping[str, Any]) -> Step15Paths:
    """Resolve Step-15 paths."""
    tables_dir = _project_path(config, get_by_path(config, "paths.tables_dir", "results/tables"))
    models_dir = _project_path(config, get_by_path(config, "paths.models_dir", "results/models"))

    predictions_dir = _project_path(
        config,
        get_by_path(
            config,
            "paths.step15_baseline_predictions_dir",
            "results/tables/baseline_predictions",
        ),
    )

    ensure_dir(tables_dir)
    ensure_dir(models_dir)
    ensure_dir(predictions_dir)

    return Step15Paths(
        tables_dir=tables_dir,
        models_dir=models_dir,
        predictions_dir=predictions_dir,
        summary_json=_project_path(
            config,
            get_by_path(
                config,
                "paths.step15_baselines_summary_json",
                "results/tables/step15_baselines_summary.json",
            ),
        ),
        threshold_selection_json=_project_path(
            config,
            get_by_path(
                config,
                "paths.baseline_threshold_selection_json",
                "results/tables/baseline_threshold_selection.json",
            ),
        ),
        threshold_selection_csv=_project_path(
            config,
            get_by_path(
                config,
                "paths.baseline_threshold_selection_csv",
                "results/tables/baseline_threshold_selection.csv",
            ),
        ),
        threshold_candidates_csv=_project_path(
            config,
            get_by_path(
                config,
                "paths.baseline_threshold_candidates_csv",
                "results/tables/baseline_threshold_candidates.csv",
            ),
        ),
        baseline_training_records_csv=_project_path(
            config,
            get_by_path(
                config,
                "paths.baseline_training_records_csv",
                "results/tables/baseline_training_records.csv",
            ),
        ),
        baseline_prediction_records_csv=_project_path(
            config,
            get_by_path(
                config,
                "paths.baseline_prediction_records_csv",
                "results/tables/baseline_prediction_records.csv",
            ),
        ),
        all_baselines_comparison_csv=_project_path(
            config,
            get_by_path(
                config,
                "paths.baselines_comparison_csv",
                "results/tables/baselines_comparison.csv",
            ),
        ),
        dataset1_main_comparison_csv=_project_path(
            config,
            get_by_path(
                config,
                "paths.dataset1_main_comparison_csv",
                "results/tables/dataset1_main_comparison.csv",
            ),
        ),
        dataset2_external_comparison_csv=_project_path(
            config,
            get_by_path(
                config,
                "paths.dataset2_external_comparison_csv",
                "results/tables/dataset2_external_comparison.csv",
            ),
        ),
        dataset3_online_case_study_csv=_project_path(
            config,
            get_by_path(
                config,
                "paths.dataset3_online_case_study_csv",
                "results/tables/dataset3_online_case_study.csv",
            ),
        ),
    )


def read_table_or_empty(path: Path | str) -> pd.DataFrame:
    """Read CSV table or return empty DataFrame."""
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()

        return pd.read_csv(path, low_memory=False)

    except Exception:
        return pd.DataFrame()


def upsert_union_csv(
    path: Path | str,
    rows: Sequence[Mapping[str, Any]],
    key_columns: Sequence[str],
) -> Path:
    """
    Upsert rows into CSV with union columns.

    This is intentionally robust to older tables whose columns differ from the
    Step-15 baseline row schema.
    """
    path = Path(path)
    ensure_dir(path.parent)

    new_df = pd.DataFrame([_json_safe(dict(row)) for row in rows])

    if path.exists():
        old_df = read_table_or_empty(path)
    else:
        old_df = pd.DataFrame()

    if old_df.empty:
        out_df = new_df
    else:
        all_columns = list(old_df.columns)
        for column in new_df.columns:
            if column not in all_columns:
                all_columns.append(column)

        for column in all_columns:
            if column not in old_df.columns:
                old_df[column] = None
            if column not in new_df.columns:
                new_df[column] = None

        old_df = old_df[all_columns].copy()
        new_df = new_df[all_columns].copy()

        if key_columns:
            keep_mask = pd.Series([True] * len(old_df), index=old_df.index)

            for _, new_row in new_df.iterrows():
                match = pd.Series([True] * len(old_df), index=old_df.index)

                for key in key_columns:
                    if key not in old_df.columns or key not in new_df.columns:
                        match = pd.Series([False] * len(old_df), index=old_df.index)
                        break

                    match = match & (old_df[key].astype(str) == str(new_row.get(key)))

                keep_mask = keep_mask & (~match)

            old_df = old_df[keep_mask].copy()

        out_df = pd.concat([old_df, new_df], ignore_index=True)

    out_df.to_csv(path, index=False)
    return path


def save_prediction_bundle_npz(
    bundle: EvaluationPredictionBundle,
    output_path: Path | str,
) -> Path:
    """Save prediction bundle as NPZ."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    np.savez_compressed(
        output_path,
        split_name=str(bundle.split_name),
        probabilities=np.asarray(bundle.probabilities, dtype=np.float32),
        logits=np.asarray(bundle.logits, dtype=np.float32),
        labels=np.asarray(bundle.labels, dtype=np.int64),
        valid_mask=np.asarray(bundle.valid_mask, dtype=np.float32),
        segment_ids=np.asarray(bundle.segment_ids, dtype=object),
        row_indices=np.asarray(bundle.row_indices, dtype=np.int64),
        delta_t=np.asarray(bundle.delta_t, dtype=np.float32),
        checkpoint_path=str(bundle.checkpoint_path),
        model_name=str(bundle.model_name),
    )

    return output_path


def prediction_output_path(
    paths: Step15Paths,
    baseline_key: str,
    split_name: str,
) -> Path:
    """Return prediction NPZ output path."""
    safe_key = str(baseline_key).replace("/", "_").replace("\\", "_")
    safe_split = str(split_name).replace("/", "_").replace("\\", "_")
    return paths.predictions_dir / f"{safe_key}_{safe_split}_predictions.npz"


def apply_persistence_alarm(
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
    theta: float,
    persistence: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply raw threshold and persistence rule.

    The persistence counter resets at:
    - segment boundary,
    - invalid row,
    - raw negative row.
    """
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=float).reshape(-1)
    segment_ids = np.asarray(segment_ids).astype(str).reshape(-1)

    raw_positive = (probabilities >= float(theta)) & (valid_mask > 0.5)
    confirmed = np.zeros(len(raw_positive), dtype=np.int64)

    count = 0
    previous_segment: Optional[str] = None

    for i in range(len(raw_positive)):
        segment = str(segment_ids[i])

        if previous_segment is None or segment != previous_segment:
            count = 0

        previous_segment = segment

        if valid_mask[i] <= 0.5:
            count = 0
            continue

        if raw_positive[i]:
            count += 1
        else:
            count = 0

        if count >= int(persistence):
            confirmed[i] = 1

    return raw_positive.astype(np.int64), confirmed.astype(np.int64)


def find_contiguous_events(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
    positive_value: int = 1,
) -> List[Dict[str, Any]]:
    """Find contiguous valid events by label value and segment."""
    labels = np.asarray(labels).astype(int).reshape(-1)
    valid_mask = np.asarray(valid_mask).astype(float).reshape(-1)
    segment_ids = np.asarray(segment_ids).astype(str).reshape(-1)

    events: List[Dict[str, Any]] = []

    in_event = False
    start_index = 0
    current_segment = ""

    for i in range(len(labels)):
        is_positive = (
            valid_mask[i] > 0.5
            and int(labels[i]) == int(positive_value)
        )
        segment = str(segment_ids[i])

        if in_event:
            if not is_positive or segment != current_segment:
                events.append(
                    {
                        "segment_id": current_segment,
                        "start_index": int(start_index),
                        "end_index": int(i - 1),
                        "duration_rows": int(i - start_index),
                    }
                )
                in_event = False

        if is_positive and not in_event:
            in_event = True
            start_index = int(i)
            current_segment = segment

    if in_event:
        events.append(
            {
                "segment_id": current_segment,
                "start_index": int(start_index),
                "end_index": int(len(labels) - 1),
                "duration_rows": int(len(labels) - start_index),
            }
        )

    return events


def detection_delay_seconds(
    event_start: int,
    detection_index: int,
    delta_t: np.ndarray,
) -> float:
    """
    Compute delay using delta_t if available.

    For this dataset delta_t is usually 1 second, so this matches row delay.
    """
    if detection_index <= event_start:
        return 0.0

    delta_t = np.asarray(delta_t, dtype=float).reshape(-1)

    if len(delta_t) == 0:
        return float(detection_index - event_start)

    start = max(int(event_start) + 1, 0)
    end = min(int(detection_index) + 1, len(delta_t))

    if end <= start:
        return float(detection_index - event_start)

    return float(np.nansum(delta_t[start:end]))


def compute_event_and_online_metrics(
    bundle: EvaluationPredictionBundle,
    theta: float,
    persistence: int,
) -> Dict[str, Any]:
    """
    Compute event-level and online-case metrics from a prediction bundle.

    This supplements evaluate_bundle_with_threshold with explicit false-alarm
    event counts and attack-1 / attack-2 delays for Dataset-3-style online case.
    """
    labels = np.asarray(bundle.labels).astype(int).reshape(-1)
    valid_mask = np.asarray(bundle.valid_mask).astype(float).reshape(-1)
    segment_ids = np.asarray(bundle.segment_ids).astype(str).reshape(-1)
    delta_t = np.asarray(bundle.delta_t, dtype=float).reshape(-1)

    raw_positive, confirmed = apply_persistence_alarm(
        probabilities=np.asarray(bundle.probabilities, dtype=float),
        valid_mask=valid_mask,
        segment_ids=segment_ids,
        theta=theta,
        persistence=persistence,
    )

    attack_events = find_contiguous_events(
        labels=labels,
        valid_mask=valid_mask,
        segment_ids=segment_ids,
        positive_value=1,
    )

    normal_alarm_labels = ((labels == 0) & (confirmed == 1)).astype(int)
    false_alarm_events = find_contiguous_events(
        labels=normal_alarm_labels,
        valid_mask=valid_mask,
        segment_ids=segment_ids,
        positive_value=1,
    )

    detected_attack_events = 0
    delays: List[float] = []
    event_details: List[Dict[str, Any]] = []

    for event_index, event in enumerate(attack_events):
        start = int(event["start_index"])
        end = int(event["end_index"])

        event_alarm_indices = np.where(confirmed[start : end + 1] == 1)[0]

        if len(event_alarm_indices) > 0:
            first_detection = int(start + event_alarm_indices[0])
            detected = True
            delay = detection_delay_seconds(
                event_start=start,
                detection_index=first_detection,
                delta_t=delta_t,
            )
            detected_attack_events += 1
            delays.append(float(delay))
        else:
            first_detection = None
            detected = False
            delay = None

        event_details.append(
            {
                "event_index": int(event_index),
                "segment_id": str(event["segment_id"]),
                "start_index": start,
                "end_index": end,
                "duration_rows": int(event["duration_rows"]),
                "detected": bool(detected),
                "first_detection_index": first_detection,
                "detection_delay": delay,
            }
        )

    attack_event_count = int(len(attack_events))
    attack_detection_rate = (
        float(detected_attack_events / attack_event_count)
        if attack_event_count > 0
        else None
    )

    mean_detection_delay = float(np.mean(delays)) if delays else None

    attack_1_delay = delays[0] if len(delays) >= 1 else None
    attack_2_delay = delays[1] if len(delays) >= 2 else None

    false_alarm_event_details = []
    for event_index, event in enumerate(false_alarm_events):
        false_alarm_event_details.append(
            {
                "event_index": int(event_index),
                "segment_id": str(event["segment_id"]),
                "start_index": int(event["start_index"]),
                "end_index": int(event["end_index"]),
                "duration_rows": int(event["duration_rows"]),
            }
        )

    valid = valid_mask > 0.5

    return {
        "confirmed_positive_count": int((confirmed[valid] == 1).sum()),
        "raw_positive_count": int((raw_positive[valid] == 1).sum()),
        "row_level_false_alarms": int(((confirmed == 1) & (labels == 0) & valid).sum()),
        "false_alarms": int(((confirmed == 1) & (labels == 0) & valid).sum()),
        "normal_alarm_event_count": int(len(false_alarm_events)),
        "false_alarm_event_count": int(len(false_alarm_events)),
        "attack_event_count": attack_event_count,
        "detected_attack_event_count": int(detected_attack_events),
        "attack_detection_rate": attack_detection_rate,
        "mean_detection_delay": mean_detection_delay,
        "detection_delays": delays,
        "attack_1_delay": attack_1_delay,
        "attack_2_delay": attack_2_delay,
        "attack_event_details": event_details,
        "false_alarm_event_details": false_alarm_event_details,
    }


def evaluate_baseline_bundle(
    bundle: EvaluationPredictionBundle,
    theta: float,
    persistence: int,
) -> Dict[str, Any]:
    """Evaluate baseline bundle with threshold and persistence."""
    metrics = dict(
        evaluate_bundle_with_threshold(
            bundle=bundle,
            theta=float(theta),
            persistence=int(persistence),
        )
    )

    event_metrics = compute_event_and_online_metrics(
        bundle=bundle,
        theta=float(theta),
        persistence=int(persistence),
    )

    metrics.update(event_metrics)

    metrics["theta"] = float(theta)
    metrics["persistence"] = int(persistence)
    metrics["split_name"] = str(bundle.split_name)
    metrics["model_name"] = str(bundle.model_name)

    return metrics


def threshold_candidate_sort_key(
    metrics: Mapping[str, Any],
    objective: str,
) -> Tuple[float, float, float, float, float]:
    """
    Sort key for threshold selection.

    Higher is better for all tuple values.

    Primary objective currently:
    - maximize_f1

    Tie-breaks:
    - higher AUPRC,
    - lower FPR,
    - higher ADR,
    - lower delay.
    """
    objective = str(objective).lower().strip()

    if objective in {"maximize_f1", "f1"}:
        primary = _metric(metrics, "f1", 0.0)
    elif objective in {"maximize_auprc", "auprc"}:
        primary = _metric(metrics, "auprc", 0.0)
    elif objective in {"maximize_f1_low_fpr", "f1_low_fpr"}:
        primary = _metric(metrics, "f1", 0.0) - _metric(metrics, "fpr", 0.0)
    else:
        primary = _metric(metrics, "f1", 0.0)

    auprc = _metric(metrics, "auprc", 0.0)
    negative_fpr = -_metric(metrics, "fpr", 1.0)
    adr = _metric(metrics, "attack_detection_rate", 0.0)
    negative_delay = -_metric(metrics, "mean_detection_delay", 1.0e9)

    return (
        float(primary),
        float(auprc),
        float(negative_fpr),
        float(adr),
        float(negative_delay),
    )


def select_threshold_for_baseline(
    baseline_key: str,
    model_name: str,
    val_bundle: EvaluationPredictionBundle,
    theta_grid: Sequence[float],
    persistence_values: Sequence[int],
    objective: str,
) -> Tuple[Step15ThresholdSelection, List[Dict[str, Any]]]:
    """Select theta and persistence on Dataset-1 validation only."""
    candidates: List[Dict[str, Any]] = []

    best_metrics: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, float, float, float]] = None

    for theta in theta_grid:
        for persistence in persistence_values:
            metrics = evaluate_baseline_bundle(
                bundle=val_bundle,
                theta=float(theta),
                persistence=int(persistence),
            )

            row = {
                "baseline_key": baseline_key,
                "model_name": model_name,
                "split_name": "val",
                "theta": float(theta),
                "persistence": int(persistence),
                "objective": objective,
                **flatten_metrics_for_table(metrics),
            }
            candidates.append(row)

            current_key = threshold_candidate_sort_key(metrics, objective)

            if best_key is None or current_key > best_key:
                best_key = current_key
                best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError(f"No threshold candidates were evaluated for baseline {baseline_key}.")

    selected_value = threshold_candidate_sort_key(best_metrics, objective)[0]

    selection = Step15ThresholdSelection(
        baseline_key=baseline_key,
        model_name=model_name,
        theta=float(best_metrics["theta"]),
        persistence=int(best_metrics["persistence"]),
        objective=str(objective),
        selected_metric_value=float(selected_value),
        validation_metrics=best_metrics,
        candidate_count=int(len(candidates)),
    )

    return selection, candidates


def flatten_metrics_for_table(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten metric dictionary for CSV rows."""
    row: Dict[str, Any] = {}

    scalar_keys = [
        "auprc",
        "auroc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "tpr",
        "specificity",
        "theta",
        "persistence",
        "tp",
        "fp",
        "tn",
        "fn",
        "confirmed_positive_count",
        "raw_positive_count",
        "row_level_false_alarms",
        "false_alarms",
        "normal_alarm_event_count",
        "false_alarm_event_count",
        "attack_event_count",
        "detected_attack_event_count",
        "attack_detection_rate",
        "mean_detection_delay",
        "attack_1_delay",
        "attack_2_delay",
    ]

    for key in scalar_keys:
        value = metrics.get(key)
        if isinstance(value, (list, dict, tuple)):
            continue
        row[key] = _safe_float(value) if value is not None else None

    for int_key in [
        "persistence",
        "tp",
        "fp",
        "tn",
        "fn",
        "confirmed_positive_count",
        "raw_positive_count",
        "row_level_false_alarms",
        "false_alarms",
        "normal_alarm_event_count",
        "false_alarm_event_count",
        "attack_event_count",
        "detected_attack_event_count",
    ]:
        if int_key in row and row[int_key] is not None:
            row[int_key] = int(row[int_key])

    delays = metrics.get("detection_delays")
    if isinstance(delays, (list, tuple)):
        row["detection_delays"] = json.dumps(_json_safe(list(delays)))

    return row


def comparison_row_from_result(
    result: Step15EvaluationResult,
    active_seed: int,
) -> Dict[str, Any]:
    """Build one comparison-table row."""
    metrics = result.metrics
    row = {
        "model_name": result.model_name,
        "baseline_key": result.baseline_key,
        "split_name": result.split_name,
        "seed": int(active_seed),
        "theta": float(result.theta),
        "persistence": int(result.persistence),
        "checkpoint_path": str(result.checkpoint_path),
        "prediction_path": str(result.prediction_path),
        "runtime_seconds": float(result.runtime_seconds),
        "diagnostic_only": False,
        "uses_same_xi_features": True,
        "raw_shortcut_columns_used": False,
        "threshold_selected_on_dataset1_validation_only": True,
        "dataset2_used_for_tuning": False,
        "dataset3_used_for_tuning": False,
        "synthetic_step10_theta_used": False,
    }
    row.update(flatten_metrics_for_table(metrics))
    return row


def dataset3_case_row_from_result(
    result: Step15EvaluationResult,
    active_seed: int,
) -> Dict[str, Any]:
    """Build Dataset-3 online case-study row."""
    metrics = result.metrics

    return {
        "method": result.model_name,
        "model_name": result.model_name,
        "baseline_key": result.baseline_key,
        "split_name": result.split_name,
        "seed": int(active_seed),
        "theta": float(result.theta),
        "persistence": int(result.persistence),
        "false_alarms": _int_metric(metrics, "row_level_false_alarms"),
        "false_alarm_events": _int_metric(metrics, "false_alarm_event_count"),
        "normal_alarm_event_count": _int_metric(metrics, "normal_alarm_event_count"),
        "attack_1_delay": _safe_float(metrics.get("attack_1_delay")),
        "attack_2_delay": _safe_float(metrics.get("attack_2_delay")),
        "mean_delay": _safe_float(metrics.get("mean_detection_delay")),
        "mean_detection_delay": _safe_float(metrics.get("mean_detection_delay")),
        "attack_detection_rate": _safe_float(metrics.get("attack_detection_rate")),
        "attack_event_count": _int_metric(metrics, "attack_event_count"),
        "detected_attack_event_count": _int_metric(metrics, "detected_attack_event_count"),
        "auprc": _safe_float(metrics.get("auprc")),
        "auroc": _safe_float(metrics.get("auroc")),
        "f1": _safe_float(metrics.get("f1")),
        "precision": _safe_float(metrics.get("precision")),
        "recall": _safe_float(metrics.get("recall")),
        "fpr": _safe_float(metrics.get("fpr")),
        "checkpoint_path": str(result.checkpoint_path),
        "prediction_path": str(result.prediction_path),
        "runtime_seconds": float(result.runtime_seconds),
        "uses_same_xi_features": True,
        "raw_shortcut_columns_used": False,
        "threshold_selected_on_dataset1_validation_only": True,
    }


def save_step15_tables(
    paths: Step15Paths,
    config15: Step15Config,
    active_seed: int,
    training_records: Sequence[BaselineRuntimeRecord],
    prediction_records: Sequence[BaselinePredictionRecord],
    threshold_selections: Sequence[Step15ThresholdSelection],
    threshold_candidate_rows: Sequence[Mapping[str, Any]],
    evaluation_results: Sequence[Step15EvaluationResult],
) -> Dict[str, str]:
    """Save all Step-15 CSV/JSON tables."""
    output_paths: Dict[str, str] = {}

    if config15.save_csv:
        training_rows = [record.to_dict() for record in training_records]
        prediction_rows = [record.to_dict() for record in prediction_records]
        selection_rows = [selection.to_dict() for selection in threshold_selections]

        if training_rows:
            upsert_union_csv(
                paths.baseline_training_records_csv,
                training_rows,
                key_columns=["key", "display_name"],
            )
            output_paths["baseline_training_records_csv"] = str(paths.baseline_training_records_csv)

        if prediction_rows:
            upsert_union_csv(
                paths.baseline_prediction_records_csv,
                prediction_rows,
                key_columns=["key", "split_name"],
            )
            output_paths["baseline_prediction_records_csv"] = str(paths.baseline_prediction_records_csv)

        if selection_rows:
            flat_selection_rows = []
            for selection in threshold_selections:
                row = {
                    "baseline_key": selection.baseline_key,
                    "model_name": selection.model_name,
                    "theta": selection.theta,
                    "persistence": selection.persistence,
                    "objective": selection.objective,
                    "selected_metric_value": selection.selected_metric_value,
                    "candidate_count": selection.candidate_count,
                }
                row.update(flatten_metrics_for_table(selection.validation_metrics))
                flat_selection_rows.append(row)

            upsert_union_csv(
                paths.threshold_selection_csv,
                flat_selection_rows,
                key_columns=["baseline_key", "model_name"],
            )
            output_paths["threshold_selection_csv"] = str(paths.threshold_selection_csv)

        if threshold_candidate_rows:
            pd.DataFrame([_json_safe(dict(row)) for row in threshold_candidate_rows]).to_csv(
                paths.threshold_candidates_csv,
                index=False,
            )
            output_paths["threshold_candidates_csv"] = str(paths.threshold_candidates_csv)

        comparison_rows = [
            comparison_row_from_result(result, active_seed=active_seed)
            for result in evaluation_results
        ]

        if comparison_rows:
            upsert_union_csv(
                paths.all_baselines_comparison_csv,
                comparison_rows,
                key_columns=["model_name", "baseline_key", "split_name", "seed"],
            )
            output_paths["all_baselines_comparison_csv"] = str(paths.all_baselines_comparison_csv)

            dataset1_rows = [row for row in comparison_rows if row["split_name"] == "test"]
            dataset2_rows = [row for row in comparison_rows if row["split_name"] == "external"]

            if dataset1_rows:
                upsert_union_csv(
                    paths.dataset1_main_comparison_csv,
                    dataset1_rows,
                    key_columns=["model_name", "split_name", "seed"],
                )
                output_paths["dataset1_main_comparison_csv"] = str(paths.dataset1_main_comparison_csv)

            if dataset2_rows:
                upsert_union_csv(
                    paths.dataset2_external_comparison_csv,
                    dataset2_rows,
                    key_columns=["model_name", "split_name", "seed"],
                )
                output_paths["dataset2_external_comparison_csv"] = str(paths.dataset2_external_comparison_csv)

            dataset3_results = [result for result in evaluation_results if result.split_name == "online"]
            dataset3_rows = [
                dataset3_case_row_from_result(result, active_seed=active_seed)
                for result in dataset3_results
            ]

            if dataset3_rows:
                upsert_union_csv(
                    paths.dataset3_online_case_study_csv,
                    dataset3_rows,
                    key_columns=["model_name", "split_name", "seed"],
                )
                output_paths["dataset3_online_case_study_csv"] = str(paths.dataset3_online_case_study_csv)

    if config15.save_json:
        save_json_safe(
            {
                "threshold_selections": [selection.to_dict() for selection in threshold_selections],
                "threshold_candidate_count": int(len(threshold_candidate_rows)),
                "fairness_rules": {
                    "threshold_selected_on_dataset1_validation_only": True,
                    "dataset2_not_used_for_threshold_selection": True,
                    "dataset3_not_used_for_threshold_selection": True,
                    "synthetic_step10_theta_not_used": True,
                    "same_xi_features_as_proposed": True,
                },
            },
            paths.threshold_selection_json,
        )
        output_paths["threshold_selection_json"] = str(paths.threshold_selection_json)

    return output_paths


def print_step15_primary_table(
    evaluation_results: Sequence[Step15EvaluationResult],
) -> None:
    """Print compact primary metric table."""
    print("=" * 120)
    print("STEP 15 BASELINE PRIMARY METRICS")
    print("=" * 120)
    print(
        f"{'Model':<18} | {'Split':<8} | {'AUPRC':>9} | {'F1':>9} | "
        f"{'FPR':>9} | {'ADR':>9} | {'Delay':>9} | {'theta':>6} | {'Np':>3}"
    )
    print("-" * 120)

    for result in evaluation_results:
        metrics = result.metrics

        print(
            f"{result.model_name:<18} | "
            f"{result.split_name:<8} | "
            f"{_metric(metrics, 'auprc', float('nan')):>9.4f} | "
            f"{_metric(metrics, 'f1', float('nan')):>9.4f} | "
            f"{_metric(metrics, 'fpr', float('nan')):>9.4f} | "
            f"{_metric(metrics, 'attack_detection_rate', float('nan')):>9.4f} | "
            f"{_metric(metrics, 'mean_detection_delay', float('nan')):>9.4f} | "
            f"{result.theta:>6.2f} | "
            f"{result.persistence:>3d}"
        )

    print("=" * 120)


def print_step15_threshold_table(
    threshold_selections: Sequence[Step15ThresholdSelection],
) -> None:
    """Print threshold selections."""
    print("=" * 100)
    print("STEP 15 VALIDATION THRESHOLD SELECTION")
    print("=" * 100)
    print(f"{'Model':<18} | {'theta':>6} | {'Np':>3} | {'Val F1':>9} | {'Val AUPRC':>9} | {'Val FPR':>9}")
    print("-" * 100)

    for selection in threshold_selections:
        metrics = selection.validation_metrics
        print(
            f"{selection.model_name:<18} | "
            f"{selection.theta:>6.2f} | "
            f"{selection.persistence:>3d} | "
            f"{_metric(metrics, 'f1', float('nan')):>9.4f} | "
            f"{_metric(metrics, 'auprc', float('nan')):>9.4f} | "
            f"{_metric(metrics, 'fpr', float('nan')):>9.4f}"
        )

    print("=" * 100)


def run_step15_baselines_experiment(
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
) -> Step15BaselinesSummary:
    """
    Run full Step-15 baseline experiment.

    Procedure:
    1. Train/load enabled baselines.
    2. Collect validation/test/external/online predictions.
    3. Select theta and persistence per baseline on Dataset-1 validation only.
    4. Evaluate each baseline on Dataset-1 test, Dataset-2 external, Dataset-3 online.
    5. Save tables and summary.
    """
    start_time = time.perf_counter()

    set_global_seed(int(active_seed))

    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    config15 = build_step15_config(config)
    paths = build_step15_paths(config)

    if not config15.theta_grid:
        raise ValueError("Step 15 theta grid is empty.")

    if not config15.persistence_values:
        raise ValueError("Step 15 persistence grid is empty.")

    print("=" * 120)
    print("STEP 15 BASELINE EXPERIMENT START")
    print("=" * 120)
    print(f"Experiment name     : {config15.experiment_name}")
    print(f"Active seed         : {active_seed}")
    print(f"Device              : {device}")
    print(f"Retrain policy      : {config15.retrain_policy}")
    print(f"Enabled baselines   : {config15.enabled_baselines}")
    print(f"Evaluation splits   : {config15.evaluation_splits}")
    print(f"Comparison splits   : {config15.comparison_splits}")
    print(f"Theta grid          : {config15.theta_grid}")
    print(f"Persistence values  : {config15.persistence_values}")
    print("Fairness: train Dataset-1 train only; threshold Dataset-1 validation only.")
    print("Fairness: Dataset-2 and Dataset-3 are never used for tuning.")
    print("Fairness: all baselines use same reconstructed xi_t features only.")
    print("=" * 120)

    artifacts, training_records = train_or_load_enabled_baselines(
        config=config,
        active_seed=active_seed,
        device=device,
        retrain_policy=config15.retrain_policy,
    )

    bundles_by_baseline, prediction_records = collect_predictions_for_enabled_baselines(
        artifacts=artifacts,
        config=config,
        split_names=config15.evaluation_splits,
        device=device,
    )

    prediction_paths: Dict[str, Dict[str, str]] = {}

    if config15.save_predictions:
        for baseline_key, split_bundles in bundles_by_baseline.items():
            prediction_paths[baseline_key] = {}

            for split_name, bundle in split_bundles.items():
                path = prediction_output_path(
                    paths=paths,
                    baseline_key=baseline_key,
                    split_name=split_name,
                )
                save_prediction_bundle_npz(bundle=bundle, output_path=path)
                prediction_paths[baseline_key][split_name] = str(path)

    threshold_selections: List[Step15ThresholdSelection] = []
    threshold_candidate_rows: List[Dict[str, Any]] = []
    evaluation_results: List[Step15EvaluationResult] = []

    for baseline_key in config15.enabled_baselines:
        if baseline_key not in bundles_by_baseline:
            raise KeyError(f"Missing prediction bundles for baseline: {baseline_key}")

        split_bundles = bundles_by_baseline[baseline_key]

        if "val" not in split_bundles:
            raise KeyError(
                f"Missing validation prediction bundle for baseline {baseline_key}. "
                "Step 15 requires Dataset-1 validation for threshold selection."
            )

        model_name = str(split_bundles["val"].model_name)

        selection, candidates = select_threshold_for_baseline(
            baseline_key=baseline_key,
            model_name=model_name,
            val_bundle=split_bundles["val"],
            theta_grid=config15.theta_grid,
            persistence_values=config15.persistence_values,
            objective=config15.threshold_objective,
        )

        threshold_selections.append(selection)
        threshold_candidate_rows.extend(candidates)

        print(
            f"Selected for {model_name}: "
            f"theta={selection.theta}, "
            f"Np={selection.persistence}, "
            f"val_f1={_metric(selection.validation_metrics, 'f1', float('nan')):.6f}, "
            f"val_auprc={_metric(selection.validation_metrics, 'auprc', float('nan')):.6f}"
        )

        for split_name in config15.comparison_splits:
            if split_name not in split_bundles:
                raise KeyError(f"Missing split '{split_name}' for baseline {baseline_key}.")

            eval_start = time.perf_counter()

            bundle = split_bundles[split_name]
            metrics = evaluate_baseline_bundle(
                bundle=bundle,
                theta=selection.theta,
                persistence=selection.persistence,
            )

            runtime = float(time.perf_counter() - eval_start)

            result = Step15EvaluationResult(
                baseline_key=baseline_key,
                model_name=model_name,
                split_name=split_name,
                theta=float(selection.theta),
                persistence=int(selection.persistence),
                metrics=metrics,
                checkpoint_path=str(bundle.checkpoint_path),
                prediction_path=prediction_paths.get(baseline_key, {}).get(split_name, ""),
                runtime_seconds=runtime,
            )

            evaluation_results.append(result)

    print_step15_threshold_table(threshold_selections)
    print_step15_primary_table(evaluation_results)

    table_paths = save_step15_tables(
        paths=paths,
        config15=config15,
        active_seed=active_seed,
        training_records=training_records,
        prediction_records=prediction_records,
        threshold_selections=threshold_selections,
        threshold_candidate_rows=threshold_candidate_rows,
        evaluation_results=evaluation_results,
    )

    artifact_summaries = {
        key: baseline_artifact_summary(artifact)
        for key, artifact in artifacts.items()
    }

    leakage_rules = {
        "baselines_train_on_dataset1_train_only": True,
        "baselines_threshold_selected_on_dataset1_validation_only": True,
        "dataset1_test_used_only_for_internal_test": True,
        "dataset2_used_only_for_external_test": True,
        "dataset3_used_only_for_online_case_study": True,
        "synthetic_step10_theta_not_used": True,
        "raw_shortcut_columns_used": False,
        "same_xi_features_as_proposed": True,
        "official_ablation_not_done_in_step15": True,
    }

    runtime_seconds = float(time.perf_counter() - start_time)

    summary = Step15BaselinesSummary(
        final_status="PASSED",
        active_seed=int(active_seed),
        experiment_name=config15.experiment_name,
        retrain_policy=config15.retrain_policy,
        enabled_baselines=list(config15.enabled_baselines),
        evaluation_splits=list(config15.evaluation_splits),
        comparison_splits=list(config15.comparison_splits),
        threshold_grid=list(config15.theta_grid),
        persistence_values=list(config15.persistence_values),
        threshold_objective=str(config15.threshold_objective),
        training_records=[record.to_dict() for record in training_records],
        prediction_records=[record.to_dict() for record in prediction_records],
        threshold_selections=[selection.to_dict() for selection in threshold_selections],
        evaluation_results=[result.to_dict() for result in evaluation_results],
        artifact_summaries=artifact_summaries,
        output_paths={
            **paths.to_dict(),
            **table_paths,
            "prediction_paths": _json_safe(prediction_paths),
        },
        leakage_rules=leakage_rules,
        runtime_seconds=runtime_seconds,
    )

    if config15.save_json:
        save_json_safe(summary.to_dict(), paths.summary_json)
        print(f"Step 15 summary JSON: {paths.summary_json}")

    print("=" * 120)
    print("STEP 15 BASELINE EXPERIMENT SUMMARY")
    print("=" * 120)
    print("Final status       : PASSED")
    print(f"Active seed        : {active_seed}")
    print(f"Enabled baselines  : {config15.enabled_baselines}")
    print(f"Runtime seconds    : {runtime_seconds:.3f}")
    print(f"Summary JSON       : {paths.summary_json}")
    print(f"Comparison CSV     : {paths.all_baselines_comparison_csv}")
    print(f"Dataset1 table     : {paths.dataset1_main_comparison_csv}")
    print(f"Dataset2 table     : {paths.dataset2_external_comparison_csv}")
    print(f"Dataset3 table     : {paths.dataset3_online_case_study_csv}")
    print("=" * 120)

    return summary


run_step15_baselines = run_step15_baselines_experiment
run_baselines_experiment = run_step15_baselines_experiment


__all__ = [
    "DEFAULT_STEP15_THETA_GRID",
    "DEFAULT_STEP15_PERSISTENCE_VALUES",
    "Step15ThresholdSelection",
    "Step15EvaluationResult",
    "Step15Paths",
    "Step15Config",
    "Step15BaselinesSummary",
    "save_json_safe",
    "build_step15_config",
    "build_step15_paths",
    "read_table_or_empty",
    "upsert_union_csv",
    "save_prediction_bundle_npz",
    "prediction_output_path",
    "apply_persistence_alarm",
    "find_contiguous_events",
    "compute_event_and_online_metrics",
    "evaluate_baseline_bundle",
    "select_threshold_for_baseline",
    "flatten_metrics_for_table",
    "comparison_row_from_result",
    "dataset3_case_row_from_result",
    "save_step15_tables",
    "print_step15_primary_table",
    "print_step15_threshold_table",
    "run_step15_baselines_experiment",
    "run_step15_baselines",
    "run_baselines_experiment",
]