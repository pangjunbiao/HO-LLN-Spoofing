"""
Dataset-3 online sequential case-study evaluation for the full proposed model.

Step 13 responsibilities:
- load trained full proposed checkpoint,
- load real validation-selected theta and N_p from Dataset-1 validation,
- evaluate Dataset-3 as one ordered online sequence,
- compute false alarms, attack delays, mean delay, and attack detection rate,
- save Dataset-3 online case-study table,
- save prediction artifacts and summary JSON.

Important:
- Dataset-3 must never be used for threshold selection.
- Dataset-3 is used as an online sequential case study.
- The EKF Detector column is not a model input.
- The attack-to-normal recovery boundary invalidity is already encoded through xi_nu.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.evaluation.evaluate_dataset1 import (
    DatasetEvaluationResult,
    EvaluationPredictionBundle,
    apply_persistence_alarm,
    build_evaluation_dataloader,
    collect_model_predictions,
    compute_probability_metrics,
    evaluate_bundle_with_threshold,
    find_attack_events,
    load_trained_model_for_evaluation,
)
from src.evaluation.evaluate_dataset2 import SelectedThreshold, load_selected_threshold
from src.evaluation.result_tables import (
    extract_primary_metrics,
    metrics_to_dataset3_row,
    print_primary_metric_table,
    save_dataset3_online_case_row,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir, save_json
from src.evaluation.alarm_rules import evaluate_precomputed_alarm_sequence


@dataclass
class OnlineAlarmEvent:
    """Contiguous alarm event during normal rows."""

    event_id: int
    segment_id: str
    start_position: int
    end_position: int
    duration_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _project_path(config: Mapping[str, Any], path_value: str) -> Path:
    """Resolve project-relative path."""
    return resolve_project_path(config, path_value)


def get_dataset3_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    """Resolve Dataset-3 Step-13 output paths."""
    return {
        "dataset3_table": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.dataset3_online_case_study_csv",
                    "results/tables/dataset3_online_case_study.csv",
                )
            ),
        ),
        "dataset3_summary_json": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.dataset3_proposed_summary_json",
                    "results/tables/dataset3_proposed_summary.json",
                )
            ),
        ),
        "dataset3_predictions_npz": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.dataset3_online_predictions_npz",
                    "results/tables/dataset3_online_predictions.npz",
                )
            ),
        ),

        "dataset3_predictions_csv": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.dataset3_online_predictions_csv",
                    "results/tables/dataset3_online_predictions.csv",
                )
            ),
        ),

        "threshold_selection_json": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.proposed_threshold_selection_json",
                    "results/tables/proposed_threshold_selection.json",
                )
            ),
        ),

        "dataset3_ekf_comparison_csv": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.dataset3_online_ekf_comparison_csv",
                    "results/tables/dataset3_online_ekf_comparison.csv",
                )
            ),
        ),
        "dataset3_ekf_comparison_summary_json": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.dataset3_online_ekf_comparison_summary_json",
                    "results/tables/dataset3_online_ekf_comparison_summary.json",
                )
            ),
        ),
    }


def save_dataset3_prediction_bundle(
    bundle: EvaluationPredictionBundle,
    config: Mapping[str, Any],
) -> Path:
    """Save Dataset-3 prediction bundle."""
    paths = get_dataset3_paths(config)
    return bundle.save_npz(paths["dataset3_predictions_npz"])


def _count_normal_alarm_events(
    labels: np.ndarray,
    confirmed_alarm: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
) -> List[OnlineAlarmEvent]:
    """
    Count contiguous false-alarm events during valid normal periods.

    This is stricter than row-level false positives and useful for online case study.
    """
    events: List[OnlineAlarmEvent] = []

    in_event = False
    start_position = -1
    current_segment: Optional[str] = None
    event_id = 0

    for i in range(len(labels)):
        segment = str(segment_ids[i])
        is_normal_alarm = bool(
            labels[i] == 0
            and confirmed_alarm[i] == 1
            and valid_mask[i] > 0.5
        )

        if current_segment is None:
            current_segment = segment

        segment_changed = segment != current_segment

        if segment_changed and in_event:
            events.append(
                OnlineAlarmEvent(
                    event_id=event_id,
                    segment_id=str(current_segment),
                    start_position=start_position,
                    end_position=i - 1,
                    duration_steps=int(i - start_position),
                )
            )
            event_id += 1
            in_event = False

        if segment_changed:
            current_segment = segment

        if is_normal_alarm and not in_event:
            in_event = True
            start_position = i

        if in_event and not is_normal_alarm:
            events.append(
                OnlineAlarmEvent(
                    event_id=event_id,
                    segment_id=str(current_segment),
                    start_position=start_position,
                    end_position=i - 1,
                    duration_steps=int(i - start_position),
                )
            )
            event_id += 1
            in_event = False

    if in_event:
        events.append(
            OnlineAlarmEvent(
                event_id=event_id,
                segment_id=str(current_segment),
                start_position=start_position,
                end_position=len(labels) - 1,
                duration_steps=int(len(labels) - start_position),
            )
        )

    return events


def _within_segment_cumulative_time(delta_t: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """Return cumulative time within each segment."""
    cumulative = np.zeros(len(delta_t), dtype=np.float64)

    previous_segment = None
    running = 0.0

    for i in range(len(delta_t)):
        segment = str(segment_ids[i])

        if previous_segment is None or segment != previous_segment:
            running = 0.0
            previous_segment = segment
        else:
            dt = float(delta_t[i])
            if np.isfinite(dt) and dt > 0.0:
                running += dt

        cumulative[i] = running

    return cumulative


def compute_dataset3_online_metrics(
    bundle: EvaluationPredictionBundle,
    theta: float,
    persistence: int,
) -> Dict[str, Any]:
    """
    Compute Dataset-3 online metrics.

    Adds Dataset-3-specific fields:
    - false_alarm_events
    - attack_1_delay
    - attack_2_delay
    - mean_detection_delay
    """
    base_metrics = evaluate_bundle_with_threshold(
        bundle=bundle,
        theta=theta,
        persistence=persistence,
    )

    confirmed_alarm = apply_persistence_alarm(
        probabilities=bundle.probabilities,
        valid_mask=bundle.valid_mask,
        segment_ids=bundle.segment_ids,
        theta=theta,
        persistence=persistence,
    )

    attack_events = find_attack_events(
        labels=bundle.labels,
        valid_mask=bundle.valid_mask,
        segment_ids=bundle.segment_ids,
    )

    cumulative_time = _within_segment_cumulative_time(
        delta_t=bundle.delta_t,
        segment_ids=bundle.segment_ids,
    )

    attack_delays: List[Optional[float]] = []
    attack_event_payloads: List[Dict[str, Any]] = []

    for event in attack_events:
        alarm_offsets = np.where(
            confirmed_alarm[event.start_position : event.end_position + 1] == 1
        )[0]

        detected = len(alarm_offsets) > 0
        delay = None

        if detected:
            first_alarm_position = event.start_position + int(alarm_offsets[0])
            delay = float(
                cumulative_time[first_alarm_position]
                - cumulative_time[event.start_position]
            )

        attack_delays.append(delay)

        payload = event.to_dict()
        payload["detected"] = bool(detected)
        payload["delay_seconds"] = delay
        attack_event_payloads.append(payload)

    detected_delays = [delay for delay in attack_delays if delay is not None]

    normal_alarm_events = _count_normal_alarm_events(
        labels=bundle.labels,
        confirmed_alarm=confirmed_alarm,
        valid_mask=bundle.valid_mask,
        segment_ids=bundle.segment_ids,
    )

    probability_metrics = compute_probability_metrics(
        labels=bundle.labels,
        probabilities=bundle.probabilities,
        valid_mask=bundle.valid_mask,
    )

    metrics = {
        **base_metrics,
        **probability_metrics,
        "attack_1_delay": attack_delays[0] if len(attack_delays) >= 1 else None,
        "attack_2_delay": attack_delays[1] if len(attack_delays) >= 2 else None,
        "mean_detection_delay": float(np.mean(detected_delays)) if detected_delays else None,
        "attack_event_count": int(len(attack_events)),
        "detected_attack_event_count": int(
            sum(delay is not None for delay in attack_delays)
        ),
        "attack_detection_rate": (
            float(sum(delay is not None for delay in attack_delays) / len(attack_events))
            if len(attack_events) > 0
            else None
        ),
        "false_alarm_events": int(len(normal_alarm_events)),
        "false_alarm_event_details": [event.to_dict() for event in normal_alarm_events],
        "dataset3_attack_event_details": attack_event_payloads,
    }

    # Keep row-level false positives as "false_alarms" for compatibility with table.
    # Also expose event-level false alarms separately.
    metrics["row_level_false_alarms"] = metrics.get("false_alarms")
    metrics["normal_alarm_event_count"] = int(len(normal_alarm_events))

    return metrics


def save_dataset3_online_predictions_csv(
    bundle: EvaluationPredictionBundle,
    metrics: Mapping[str, Any],
    theta: float,
    persistence: int,
    config: Mapping[str, Any],
) -> Path:
    """
    Save row-level Dataset-3 online predictions as CSV for inspection.

    This is useful for checking alarm timing visually.
    """
    paths = get_dataset3_paths(config)
    csv_path = paths["dataset3_predictions_npz"].with_suffix(".csv")
    ensure_dir(csv_path.parent)

    confirmed_alarm = apply_persistence_alarm(
        probabilities=bundle.probabilities,
        valid_mask=bundle.valid_mask,
        segment_ids=bundle.segment_ids,
        theta=theta,
        persistence=persistence,
    )

    df = pd.DataFrame(
        {
            "row_index": bundle.row_indices,
            "segment_id": bundle.segment_ids.astype(str),
            "label": bundle.labels.astype(int),
            "xi_valid": bundle.valid_mask.astype(float),
            "probability": bundle.probabilities.astype(float),
            "logit": bundle.logits.astype(float),
            "raw_positive": (bundle.probabilities >= float(theta)).astype(int),
            "confirmed_alarm": confirmed_alarm.astype(int),
            "delta_t": bundle.delta_t.astype(float),
        }
    )

    df.to_csv(csv_path, index=False)

    return csv_path


def load_saved_dataset3_predictions_csv(
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Load saved Dataset-3 Proposed online predictions from Step 13.

    This avoids rerunning the Proposed model during Step 19.
    """
    paths = get_dataset3_paths(config)

    csv_path = paths.get("dataset3_predictions_csv")
    if csv_path is None:
        csv_path = paths["dataset3_predictions_npz"].with_suffix(".csv")

    if not csv_path.exists():
        fallback = paths["dataset3_predictions_npz"].with_suffix(".csv")
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError(
                "Dataset-3 prediction CSV not found. "
                f"Checked: {csv_path} and {fallback}. "
                "Run Step 13 first to create dataset3_online_predictions.csv."
            )

    df = pd.read_csv(csv_path)

    required = [
        "row_index",
        "segment_id",
        "label",
        "xi_valid",
        "probability",
        "confirmed_alarm",
        "delta_t",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Saved Dataset-3 prediction CSV is missing required columns: {missing}"
        )

    return df


def compute_dataset3_metrics_from_confirmed_alarm_df(
    df: pd.DataFrame,
    alarm_column: str,
    method_name: str,
) -> Dict[str, Any]:
    """
    Compute Dataset-3 online case-study metrics from an already-built alarm column.

    Used by Step 19 for:
    - Proposed confirmed_alarm from saved predictions,
    - EKF Detector alarm from Dataset-3.
    """
    required = ["label", "segment_id", "delta_t", alarm_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Cannot compute Dataset-3 case-study metrics for {method_name}. "
            f"Missing columns: {missing}"
        )

    order_index = (
        df["row_index"].to_numpy()
        if "row_index" in df.columns
        else np.arange(len(df), dtype=np.int64)
    )

    valid_mask = (
        df["xi_valid"].to_numpy()
        if "xi_valid" in df.columns
        else np.ones(len(df), dtype=np.int64)
    )

    payload = evaluate_precomputed_alarm_sequence(
        y_true=df["label"].to_numpy(),
        confirmed_alarm=df[alarm_column].to_numpy(),
        segment_id=df["segment_id"].to_numpy(),
        order_index=order_index,
        delta_t=df["delta_t"].to_numpy(),
        valid_mask=valid_mask,
        method_name=method_name,
    )

    return payload


def run_dataset3_online_evaluation(
    config: Mapping[str, Any],
    active_seed: int = 42,
    checkpoint_path: Optional[str] = None,
    theta: Optional[float] = None,
    persistence: Optional[int] = None,
    method_name: str = "Proposed",
    device: Optional[Any] = None,
) -> DatasetEvaluationResult:
    """
    Evaluate full proposed model on Dataset-3 online sequential case study.

    Returns:
        DatasetEvaluationResult for Dataset-3 online case study.
    """
    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    start_time = time.perf_counter()

    selected_threshold: SelectedThreshold = load_selected_threshold(
        config=config,
        theta=theta,
        persistence=persistence,
    )

    model, checkpoint, checkpoint_metadata = load_trained_model_for_evaluation(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        variant_name="full",
    )

    online_loader, online_dataset = build_evaluation_dataloader(
        config=config,
        split_name="online",
        active_seed=active_seed,
        full_sequence=True,
    )

    print("=" * 100)
    print("STEP 13 DATASET-3 ONLINE CASE-STUDY EVALUATION")
    print("=" * 100)
    print(f"Method                 : {method_name}")
    print(f"Checkpoint             : {checkpoint}")
    print(f"Online rows/windows    : {online_dataset.summary()['rows']} / {online_dataset.summary()['windows']}")
    print(f"Using validation theta : {selected_threshold.theta}")
    print(f"Using validation N_p   : {selected_threshold.persistence}")
    print(f"Threshold source       : {selected_threshold.source}")
    print("Dataset-3 is online case study only: no tuning, no threshold selection.")
    print("=" * 100)

    online_bundle = collect_model_predictions(
        model=model,
        dataloader=online_loader,
        device=device,
        split_name="online",
        checkpoint_path=str(checkpoint),
        model_name=method_name,
    )

    metrics = compute_dataset3_online_metrics(
        bundle=online_bundle,
        theta=selected_threshold.theta,
        persistence=selected_threshold.persistence,
    )
    metrics["runtime_seconds"] = float(time.perf_counter() - start_time)

    paths = get_dataset3_paths(config)

    predictions_npz = save_dataset3_prediction_bundle(
        bundle=online_bundle,
        config=config,
    )

    predictions_csv = save_dataset3_online_predictions_csv(
        bundle=online_bundle,
        metrics=metrics,
        theta=selected_threshold.theta,
        persistence=selected_threshold.persistence,
        config=config,
    )

    table_row = metrics_to_dataset3_row(
        method_name=method_name,
        metrics=metrics,
        threshold=selected_threshold.theta,
        persistence=selected_threshold.persistence,
        checkpoint_path=str(checkpoint),
        notes=(
            "Full proposed model; theta and N_p selected on Dataset-1 validation only; "
            "Dataset-3 evaluated as one online ordered sequence."
        ),
    )

    save_dataset3_online_case_row(
        output_path=paths["dataset3_table"],
        row=table_row,
    )

    result = DatasetEvaluationResult(
        model_name=method_name,
        split_name="Dataset-3 Online",
        metrics=metrics,
        threshold=selected_threshold.theta,
        persistence=selected_threshold.persistence,
        checkpoint_path=str(checkpoint),
        prediction_summary=online_bundle.to_dict_summary(),
        artifact_paths={
            "dataset3_table": str(paths["dataset3_table"]),
            "dataset3_summary_json": str(paths["dataset3_summary_json"]),
            "dataset3_predictions_npz": str(predictions_npz),
            "dataset3_predictions_csv": str(predictions_csv),
            "threshold_selection_json": str(paths["threshold_selection_json"]),
        },
    )

    summary_payload = {
        "result": result.to_dict(),
        "online_prediction_summary": online_bundle.to_dict_summary(),
        "selected_threshold": selected_threshold.to_dict(),
        "checkpoint_metadata": checkpoint_metadata,
        "online_dataset_summary": online_dataset.summary(),
        "dataset3_online_metrics": metrics,
        "leakage_rules": {
            "dataset3_used_for_training": False,
            "dataset3_used_for_validation": False,
            "dataset3_used_for_threshold_selection": False,
            "threshold_selected_on_dataset1_validation_only": True,
            "synthetic_step10_theta_not_used": True,
            "online_order_preserved": True,
        },
    }

    save_json(summary_payload, paths["dataset3_summary_json"], indent=2)

    primary_row = {
        "Method": method_name,
        **extract_primary_metrics(metrics),
    }
    print_primary_metric_table(
        title="STEP 13 DATASET-3 ONLINE PRIMARY METRICS",
        rows=[primary_row],
        model_key="Method",
    )

    print("Dataset-3 online case-study details:")
    print(f"  false alarms row-level     : {metrics.get('row_level_false_alarms')}")
    print(f"  false alarm events         : {metrics.get('normal_alarm_event_count')}")
    print(f"  attack-1 delay             : {metrics.get('attack_1_delay')}")
    print(f"  attack-2 delay             : {metrics.get('attack_2_delay')}")
    print(f"  mean detection delay       : {metrics.get('mean_detection_delay')}")
    print(f"  attack detection rate      : {metrics.get('attack_detection_rate')}")
    print("Saved Dataset-3 artifacts:")
    for key, value in result.artifact_paths.items():
        print(f"  {key}: {value}")
    print("=" * 100)

    return result


__all__ = [
    "OnlineAlarmEvent",
    "get_dataset3_paths",
    "save_dataset3_prediction_bundle",
    "compute_dataset3_online_metrics",
    "save_dataset3_online_predictions_csv",
    "run_dataset3_online_evaluation",
    "load_saved_dataset3_predictions_csv",
    "compute_dataset3_metrics_from_confirmed_alarm_df",
]