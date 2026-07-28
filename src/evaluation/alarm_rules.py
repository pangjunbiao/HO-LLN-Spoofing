"""
Causal alarm rules and event-level metrics.

Step 10 purpose:
- implement c_t = I(p_hat_t >= theta),
- implement confirmed alarm after N_p consecutive positives,
- compute Attack Detection Rate,
- compute Detection Delay,
- compute Normal-Segment FAR.

Critical rule:
theta and N_p are not selected here.
They must be selected by threshold_selection.py on validation only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_time_step_metric_bundle,
    threshold_scores,
)


@dataclass
class AttackEvent:
    """One contiguous attack interval inside one segment."""

    event_id: str
    segment_id: str
    start_position: int
    end_position: int
    start_order_index: int
    end_order_index: int
    duration_rows: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttackEventDetection:
    """Detection result for one attack event."""

    event_id: str
    segment_id: str
    detected: bool
    attack_start_position: int
    attack_end_position: int
    first_alarm_position: Optional[int]
    delay_rows: Optional[int]
    delay_seconds: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventLevelMetrics:
    """Event-level metrics from confirmed alarms."""

    attack_events_total: int
    attack_events_detected: int
    attack_events_missed: int
    attack_detection_rate: float

    mean_detection_delay_seconds: Optional[float]
    median_detection_delay_seconds: Optional[float]
    max_detection_delay_seconds: Optional[float]

    mean_detection_delay_rows: Optional[float]
    median_detection_delay_rows: Optional[float]
    max_detection_delay_rows: Optional[int]

    false_alarm_rows: int
    normal_rows: int
    normal_row_false_alarm_rate: float

    normal_segments_total: int
    normal_segments_with_alarm: int
    normal_segment_far: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_1d_array(values: Sequence[Any], name: str) -> np.ndarray:
    """Convert input to 1D numpy array."""
    arr = np.asarray(values)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    return arr


def _binary_array(values: Sequence[Any], name: str) -> np.ndarray:
    """Convert values to binary int array."""
    arr = _as_1d_array(values, name)
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return (arr >= 0.5).astype(np.int64)


def _float_array(values: Sequence[Any], name: str) -> np.ndarray:
    """Convert values to finite float array."""
    arr = _as_1d_array(values, name)
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def _string_array(values: Sequence[Any], name: str) -> np.ndarray:
    """Convert values to string array."""
    arr = _as_1d_array(values, name)
    return np.asarray(arr).astype(str)


def apply_persistence_rule_1d(
    binary_decision: Sequence[Any],
    persistence: int,
    valid_mask: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """
    Apply confirmed-alarm rule for one already-ordered sequence.

    c_t = I(p_hat_t >= theta)
    confirmed_alarm_t = 1 only after N_p consecutive positive c_t values.

    Invalid rows break the consecutive-positive run.
    """
    decisions = _binary_array(binary_decision, "binary_decision")
    n = len(decisions)

    if valid_mask is None:
        valid = np.ones(n, dtype=np.int64)
    else:
        valid = _binary_array(valid_mask, "valid_mask")

        if len(valid) != n:
            raise ValueError(
                f"valid_mask length mismatch. Expected {n}, got {len(valid)}."
            )

    persistence = max(int(persistence), 1)

    alarms = np.zeros(n, dtype=np.int64)
    run_length = 0

    for i in range(n):
        if valid[i] == 0:
            run_length = 0
            alarms[i] = 0
            continue

        if decisions[i] == 1:
            run_length += 1
        else:
            run_length = 0

        if run_length >= persistence:
            alarms[i] = 1

    return alarms


def apply_alarm_rule_by_segment(
    y_score: Sequence[Any],
    threshold: float,
    persistence: int,
    segment_id: Sequence[Any],
    order_index: Optional[Sequence[Any]] = None,
    valid_mask: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Apply threshold and persistence alarm rule independently per segment.

    This prevents alarms from carrying across segment boundaries.
    """
    scores = _float_array(y_score, "y_score")
    segments = _string_array(segment_id, "segment_id")

    if len(scores) != len(segments):
        raise ValueError(
            f"Length mismatch. y_score has {len(scores)}, segment_id has {len(segments)}."
        )

    n = len(scores)

    if order_index is None:
        order = np.arange(n, dtype=np.int64)
    else:
        order = _as_1d_array(order_index, "order_index").astype(np.int64)
        if len(order) != n:
            raise ValueError(
                f"order_index length mismatch. Expected {n}, got {len(order)}."
            )

    if valid_mask is None:
        valid = np.ones(n, dtype=np.int64)
    else:
        valid = _binary_array(valid_mask, "valid_mask")
        if len(valid) != n:
            raise ValueError(
                f"valid_mask length mismatch. Expected {n}, got {len(valid)}."
            )

    base_decisions = threshold_scores(
        y_score=scores,
        threshold=float(threshold),
        valid_mask=valid,
    )

    result_decision = np.zeros(n, dtype=np.int64)
    result_alarm = np.zeros(n, dtype=np.int64)

    # Stable sort by segment and order for causal application.
    sort_index = np.lexsort((order, segments))

    unique_segments = []
    segment_alarm_counts: Dict[str, int] = {}

    start = 0
    while start < n:
        current_segment = segments[sort_index[start]]
        end = start + 1

        while end < n and segments[sort_index[end]] == current_segment:
            end += 1

        idx = sort_index[start:end]

        local_order_sort = np.argsort(order[idx], kind="mergesort")
        idx = idx[local_order_sort]

        local_decision = base_decisions[idx]
        local_valid = valid[idx]

        local_alarm = apply_persistence_rule_1d(
            binary_decision=local_decision,
            persistence=persistence,
            valid_mask=local_valid,
        )

        result_decision[idx] = local_decision
        result_alarm[idx] = local_alarm

        segment_key = str(current_segment)
        unique_segments.append(segment_key)
        segment_alarm_counts[segment_key] = int(local_alarm.sum())

        start = end

    return {
        "decision": result_decision,
        "confirmed_alarm": result_alarm,
        "threshold": float(threshold),
        "persistence": int(persistence),
        "segment_alarm_counts": segment_alarm_counts,
        "segments_processed": unique_segments,
    }


def build_ordered_evaluation_frame(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    segment_id: Sequence[Any],
    order_index: Optional[Sequence[Any]] = None,
    delta_t: Optional[Sequence[Any]] = None,
    valid_mask: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """
    Build a canonical ordered evaluation dataframe.

    This helper is used by event-level detection metrics.
    """
    y = _binary_array(y_true, "y_true")
    score = _float_array(y_score, "y_score")
    segments = _string_array(segment_id, "segment_id")

    n = len(y)

    if len(score) != n or len(segments) != n:
        raise ValueError("y_true, y_score, and segment_id must have equal length.")

    if order_index is None:
        order = np.arange(n, dtype=np.int64)
    else:
        order = _as_1d_array(order_index, "order_index").astype(np.int64)
        if len(order) != n:
            raise ValueError("order_index length mismatch.")

    if delta_t is None:
        dt = np.ones(n, dtype=float)
    else:
        dt = _float_array(delta_t, "delta_t")
        if len(dt) != n:
            raise ValueError("delta_t length mismatch.")

    if valid_mask is None:
        valid = np.ones(n, dtype=np.int64)
    else:
        valid = _binary_array(valid_mask, "valid_mask")
        if len(valid) != n:
            raise ValueError("valid_mask length mismatch.")

    df = pd.DataFrame(
        {
            "y_true": y,
            "y_score": score,
            "segment_id": segments,
            "order_index": order,
            "delta_t": dt,
            "valid_mask": valid,
        }
    )

    return (
        df.sort_values(["segment_id", "order_index"], kind="mergesort")
        .reset_index(drop=True)
    )


def add_alarm_columns_to_frame(
    df: pd.DataFrame,
    threshold: float,
    persistence: int,
) -> pd.DataFrame:
    """
    Add decision and confirmed_alarm columns to an ordered evaluation dataframe.
    """
    required = ["y_score", "segment_id", "order_index", "valid_mask"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Evaluation frame missing required columns: {missing}")

    alarm_result = apply_alarm_rule_by_segment(
        y_score=df["y_score"].to_numpy(),
        threshold=threshold,
        persistence=persistence,
        segment_id=df["segment_id"].to_numpy(),
        order_index=df["order_index"].to_numpy(),
        valid_mask=df["valid_mask"].to_numpy(),
    )

    out = df.copy()
    out["decision"] = alarm_result["decision"].astype(np.int64)
    out["confirmed_alarm"] = alarm_result["confirmed_alarm"].astype(np.int64)

    return out


def _segment_local_times(group: pd.DataFrame) -> np.ndarray:
    """
    Build cumulative time inside a segment from delta_t.

    The first row starts at local time 0.
    """
    dt = (
        pd.to_numeric(group["delta_t"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    dt = np.where(np.isfinite(dt), dt, 0.0)
    dt = np.maximum(dt, 0.0)

    if len(dt) == 0:
        return np.array([], dtype=float)

    local_time = np.zeros(len(dt), dtype=float)

    for i in range(1, len(dt)):
        local_time[i] = local_time[i - 1] + float(dt[i])

    return local_time


def extract_attack_events(
    df: pd.DataFrame,
    label_column: str = "y_true",
) -> List[AttackEvent]:
    """
    Extract contiguous attack intervals per segment.

    The dataframe must already be sorted by segment_id and order_index.
    """
    required = ["segment_id", "order_index", "delta_t", label_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Cannot extract attack events. Missing columns: {missing}")

    events: List[AttackEvent] = []

    for segment_id, group in df.groupby("segment_id", sort=False):
        group = group.sort_values("order_index", kind="mergesort").reset_index(drop=True)
        y = _binary_array(group[label_column].to_numpy(), label_column)
        order = group["order_index"].to_numpy(dtype=np.int64)
        local_time = _segment_local_times(group)

        i = 0
        attack_event_index = 0

        while i < len(group):
            if y[i] != 1:
                i += 1
                continue

            start = i
            while i + 1 < len(group) and y[i + 1] == 1:
                i += 1
            end = i

            duration_seconds = float(local_time[end] - local_time[start])
            duration_rows = int(end - start + 1)

            events.append(
                AttackEvent(
                    event_id=f"{segment_id}_attack_{attack_event_index:04d}",
                    segment_id=str(segment_id),
                    start_position=int(start),
                    end_position=int(end),
                    start_order_index=int(order[start]),
                    end_order_index=int(order[end]),
                    duration_rows=duration_rows,
                    duration_seconds=duration_seconds,
                )
            )

            attack_event_index += 1
            i += 1

    return events


def compute_attack_event_detections(
    df: pd.DataFrame,
    events: Sequence[AttackEvent],
    alarm_column: str = "confirmed_alarm",
) -> List[AttackEventDetection]:
    """
    Compute detection result and delay for each attack event.

    Detection is counted if the first confirmed alarm occurs inside the attack interval.
    """
    required = ["segment_id", "order_index", "delta_t", alarm_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Cannot compute event detections. Missing columns: {missing}")

    detections: List[AttackEventDetection] = []

    groups = {
        str(segment_id): group.sort_values("order_index", kind="mergesort").reset_index(drop=True)
        for segment_id, group in df.groupby("segment_id", sort=False)
    }

    for event in events:
        group = groups[event.segment_id]
        local_time = _segment_local_times(group)
        alarm = _binary_array(group[alarm_column].to_numpy(), alarm_column)

        start = int(event.start_position)
        end = int(event.end_position)

        event_alarm_positions = np.where(alarm[start : end + 1] == 1)[0]

        if len(event_alarm_positions) == 0:
            detections.append(
                AttackEventDetection(
                    event_id=event.event_id,
                    segment_id=event.segment_id,
                    detected=False,
                    attack_start_position=start,
                    attack_end_position=end,
                    first_alarm_position=None,
                    delay_rows=None,
                    delay_seconds=None,
                )
            )
            continue

        first_alarm_position = int(start + int(event_alarm_positions[0]))
        delay_rows = int(first_alarm_position - start)
        delay_seconds = float(local_time[first_alarm_position] - local_time[start])

        detections.append(
            AttackEventDetection(
                event_id=event.event_id,
                segment_id=event.segment_id,
                detected=True,
                attack_start_position=start,
                attack_end_position=end,
                first_alarm_position=first_alarm_position,
                delay_rows=delay_rows,
                delay_seconds=delay_seconds,
            )
        )

    return detections


def compute_normal_segment_far(
    df: pd.DataFrame,
    label_column: str = "y_true",
    alarm_column: str = "confirmed_alarm",
) -> Dict[str, Any]:
    """
    Compute Normal-Segment FAR.

    Normal-Segment FAR = fraction of all-normal segments with at least one confirmed alarm.
    """
    required = ["segment_id", label_column, alarm_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Cannot compute normal-segment FAR. Missing columns: {missing}")

    normal_segments_total = 0
    normal_segments_with_alarm = 0
    normal_segment_ids: List[str] = []
    normal_segments_alarm_ids: List[str] = []

    for segment_id, group in df.groupby("segment_id", sort=False):
        y = _binary_array(group[label_column].to_numpy(), label_column)
        alarm = _binary_array(group[alarm_column].to_numpy(), alarm_column)

        if int((y == 1).sum()) == 0:
            normal_segments_total += 1
            normal_segment_ids.append(str(segment_id))

            if int(alarm.sum()) > 0:
                normal_segments_with_alarm += 1
                normal_segments_alarm_ids.append(str(segment_id))

    far = (
        0.0
        if normal_segments_total == 0
        else float(normal_segments_with_alarm / normal_segments_total)
    )

    return {
        "normal_segments_total": int(normal_segments_total),
        "normal_segments_with_alarm": int(normal_segments_with_alarm),
        "normal_segment_far": float(far),
        "normal_segment_ids": normal_segment_ids,
        "normal_segments_with_alarm_ids": normal_segments_alarm_ids,
    }


def compute_event_level_metrics(
    df: pd.DataFrame,
    label_column: str = "y_true",
    alarm_column: str = "confirmed_alarm",
) -> Dict[str, Any]:
    """
    Compute event-level metrics:
    - Attack Detection Rate
    - Detection Delay
    - Normal-Segment FAR
    """
    events = extract_attack_events(df=df, label_column=label_column)
    detections = compute_attack_event_detections(
        df=df,
        events=events,
        alarm_column=alarm_column,
    )

    detected = [item for item in detections if item.detected]
    missed = [item for item in detections if not item.detected]

    delay_seconds = [
        float(item.delay_seconds)
        for item in detected
        if item.delay_seconds is not None
    ]
    delay_rows = [
        int(item.delay_rows)
        for item in detected
        if item.delay_rows is not None
    ]

    total_events = int(len(events))
    detected_events = int(len(detected))
    missed_events = int(len(missed))

    attack_detection_rate = (
        0.0
        if total_events == 0
        else float(detected_events / total_events)
    )

    y = _binary_array(df[label_column].to_numpy(), label_column)
    alarm = _binary_array(df[alarm_column].to_numpy(), alarm_column)

    normal_mask = y == 0
    normal_rows = int(normal_mask.sum())
    false_alarm_rows = int(((alarm == 1) & normal_mask).sum())
    normal_row_far = (
        0.0
        if normal_rows == 0
        else float(false_alarm_rows / normal_rows)
    )

    normal_segment_far = compute_normal_segment_far(
        df=df,
        label_column=label_column,
        alarm_column=alarm_column,
    )

    metrics = EventLevelMetrics(
        attack_events_total=total_events,
        attack_events_detected=detected_events,
        attack_events_missed=missed_events,
        attack_detection_rate=attack_detection_rate,
        mean_detection_delay_seconds=None
        if len(delay_seconds) == 0
        else float(np.mean(delay_seconds)),
        median_detection_delay_seconds=None
        if len(delay_seconds) == 0
        else float(np.median(delay_seconds)),
        max_detection_delay_seconds=None
        if len(delay_seconds) == 0
        else float(np.max(delay_seconds)),
        mean_detection_delay_rows=None
        if len(delay_rows) == 0
        else float(np.mean(delay_rows)),
        median_detection_delay_rows=None
        if len(delay_rows) == 0
        else float(np.median(delay_rows)),
        max_detection_delay_rows=None
        if len(delay_rows) == 0
        else int(np.max(delay_rows)),
        false_alarm_rows=false_alarm_rows,
        normal_rows=normal_rows,
        normal_row_false_alarm_rate=normal_row_far,
        normal_segments_total=int(normal_segment_far["normal_segments_total"]),
        normal_segments_with_alarm=int(normal_segment_far["normal_segments_with_alarm"]),
        normal_segment_far=float(normal_segment_far["normal_segment_far"]),
    )

    return {
        "event_metrics": metrics.to_dict(),
        "attack_events": [event.to_dict() for event in events],
        "attack_event_detections": [
            detection.to_dict() for detection in detections
        ],
        "normal_segment_far_details": normal_segment_far,
    }


def count_contiguous_normal_alarm_events(
    df: pd.DataFrame,
    label_column: str = "y_true",
    alarm_column: str = "confirmed_alarm",
    valid_column: str = "valid_mask",
) -> List[Dict[str, Any]]:
    """
    Count contiguous false-alarm events during valid normal periods.

    This is used for online case-study reporting where row-level false positives
    and event-level false alarms should be reported separately.
    """
    required = ["segment_id", "order_index", label_column, alarm_column, valid_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Cannot count contiguous normal alarm events. Missing columns: {missing}")

    ordered = (
        df.sort_values(["segment_id", "order_index"], kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    ordered["_eval_position"] = np.arange(len(ordered), dtype=np.int64)

    events: List[Dict[str, Any]] = []
    event_id = 0

    for segment_id, group in ordered.groupby("segment_id", sort=False):
        group = group.sort_values("order_index", kind="mergesort").reset_index(drop=True)

        y = _binary_array(group[label_column].to_numpy(), label_column)
        alarm = _binary_array(group[alarm_column].to_numpy(), alarm_column)
        valid = _binary_array(group[valid_column].to_numpy(), valid_column)
        order = group["order_index"].to_numpy(dtype=np.int64)
        eval_position = group["_eval_position"].to_numpy(dtype=np.int64)

        in_event = False
        start_local = -1

        for i in range(len(group)):
            is_false_alarm = bool(y[i] == 0 and alarm[i] == 1 and valid[i] == 1)

            if is_false_alarm and not in_event:
                in_event = True
                start_local = i

            if in_event and not is_false_alarm:
                end_local = i - 1
                events.append(
                    {
                        "event_id": int(event_id),
                        "segment_id": str(segment_id),
                        "start_position": int(eval_position[start_local]),
                        "end_position": int(eval_position[end_local]),
                        "start_order_index": int(order[start_local]),
                        "end_order_index": int(order[end_local]),
                        "duration_steps": int(end_local - start_local + 1),
                    }
                )
                event_id += 1
                in_event = False

        if in_event:
            end_local = len(group) - 1
            events.append(
                {
                    "event_id": int(event_id),
                    "segment_id": str(segment_id),
                    "start_position": int(eval_position[start_local]),
                    "end_position": int(eval_position[end_local]),
                    "start_order_index": int(order[start_local]),
                    "end_order_index": int(order[end_local]),
                    "duration_steps": int(end_local - start_local + 1),
                }
            )
            event_id += 1

    return events


def evaluate_precomputed_alarm_sequence(
    y_true: Sequence[Any],
    confirmed_alarm: Sequence[Any],
    segment_id: Sequence[Any],
    order_index: Optional[Sequence[Any]] = None,
    delta_t: Optional[Sequence[Any]] = None,
    valid_mask: Optional[Sequence[Any]] = None,
    method_name: str = "method",
) -> Dict[str, Any]:
    """
    Evaluate a precomputed binary alarm sequence using the same event-level logic.

    This is useful for EKF Detector, because EKF already provides an alarm column
    rather than a probability requiring threshold selection.
    """
    frame = build_ordered_evaluation_frame(
        y_true=y_true,
        y_score=confirmed_alarm,
        segment_id=segment_id,
        order_index=order_index,
        delta_t=delta_t,
        valid_mask=valid_mask,
    )

    frame = frame.copy()
    frame["decision"] = _binary_array(frame["y_score"].to_numpy(), "confirmed_alarm")
    frame["confirmed_alarm"] = frame["decision"].astype(np.int64)

    event_payload = compute_event_level_metrics(
        df=frame,
        label_column="y_true",
        alarm_column="confirmed_alarm",
    )

    false_alarm_events = count_contiguous_normal_alarm_events(
        df=frame,
        label_column="y_true",
        alarm_column="confirmed_alarm",
        valid_column="valid_mask",
    )

    event_metrics = dict(event_payload["event_metrics"])
    detections = list(event_payload["attack_event_detections"])

    attack_delays = [
        item.get("delay_seconds") if bool(item.get("detected")) else None
        for item in detections
    ]

    metrics = {
        "method_name": method_name,
        "attack_event_count": int(event_metrics["attack_events_total"]),
        "detected_attack_event_count": int(event_metrics["attack_events_detected"]),
        "missed_attack_event_count": int(event_metrics["attack_events_missed"]),
        "attack_detection_rate": float(event_metrics["attack_detection_rate"]),
        "attack_1_delay": attack_delays[0] if len(attack_delays) >= 1 else None,
        "attack_2_delay": attack_delays[1] if len(attack_delays) >= 2 else None,
        "mean_detection_delay": event_metrics["mean_detection_delay_seconds"],
        "median_detection_delay": event_metrics["median_detection_delay_seconds"],
        "max_detection_delay": event_metrics["max_detection_delay_seconds"],
        "row_level_false_alarms": int(event_metrics["false_alarm_rows"]),
        "false_alarms": int(event_metrics["false_alarm_rows"]),
        "false_alarm_events": int(len(false_alarm_events)),
        "normal_alarm_event_count": int(len(false_alarm_events)),
        "normal_rows": int(event_metrics["normal_rows"]),
        "normal_row_false_alarm_rate": float(event_metrics["normal_row_false_alarm_rate"]),
        "normal_segment_far": float(event_metrics["normal_segment_far"]),
    }

    return {
        "method_name": method_name,
        "metrics": metrics,
        "attack_events": event_payload["attack_events"],
        "attack_event_detections": detections,
        "false_alarm_event_details": false_alarm_events,
        "evaluation_frame": frame,
    }


def evaluate_alarm_rule(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    segment_id: Sequence[Any],
    threshold: float,
    persistence: int,
    order_index: Optional[Sequence[Any]] = None,
    delta_t: Optional[Sequence[Any]] = None,
    valid_mask: Optional[Sequence[Any]] = None,
    runtime_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Full evaluation for one threshold/persistence pair.

    Returns:
    - time-step ranking metrics,
    - time-step threshold metrics using confirmed alarms,
    - event-level attack detection/delay metrics,
    - dataframe-like compact records for detailed inspection.
    """
    base_frame = build_ordered_evaluation_frame(
        y_true=y_true,
        y_score=y_score,
        segment_id=segment_id,
        order_index=order_index,
        delta_t=delta_t,
        valid_mask=valid_mask,
    )

    eval_frame = add_alarm_columns_to_frame(
        df=base_frame,
        threshold=threshold,
        persistence=persistence,
    )

    time_step_metrics = compute_time_step_metric_bundle(
        y_true=eval_frame["y_true"].to_numpy(),
        y_score=eval_frame["y_score"].to_numpy(),
        y_pred=eval_frame["confirmed_alarm"].to_numpy(),
        threshold=threshold,
        persistence=persistence,
        valid_mask=eval_frame["valid_mask"].to_numpy(),
        runtime_seconds=runtime_seconds,
    )

    event_payload = compute_event_level_metrics(
        df=eval_frame,
        label_column="y_true",
        alarm_column="confirmed_alarm",
    )

    primary_metrics = {
        "AUPRC": time_step_metrics["ranking"]["auprc"],
        "F1": time_step_metrics["threshold_metrics"]["f1"],
        "FPR": time_step_metrics["threshold_metrics"]["fpr"],
        "Attack Detection Rate": event_payload["event_metrics"]["attack_detection_rate"],
        "Detection Delay": event_payload["event_metrics"]["mean_detection_delay_seconds"],
    }

    secondary_metrics = {
        "AUROC": time_step_metrics["ranking"]["auroc"],
        "Precision": time_step_metrics["threshold_metrics"]["precision"],
        "Recall": time_step_metrics["threshold_metrics"]["recall"],
        "Runtime": runtime_seconds,
        "Normal-Segment FAR": event_payload["event_metrics"]["normal_segment_far"],
    }

    return {
        "threshold": float(threshold),
        "persistence": int(persistence),
        "time_step_metrics": time_step_metrics,
        "event_metrics": event_payload["event_metrics"],
        "attack_events": event_payload["attack_events"],
        "attack_event_detections": event_payload["attack_event_detections"],
        "normal_segment_far_details": event_payload["normal_segment_far_details"],
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
        "evaluation_frame": eval_frame,
    }


def evaluation_frame_to_records(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert evaluation_frame from evaluate_alarm_rule into JSON-safe records.

    This is useful for debugging selected validation/test cases.
    """
    frame = payload.get("evaluation_frame")

    if frame is None:
        return []

    if not isinstance(frame, pd.DataFrame):
        return []

    return frame.to_dict(orient="records")


def strip_large_frames(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Remove pandas DataFrame from evaluation payload before JSON saving.
    """
    out = dict(payload)
    out.pop("evaluation_frame", None)
    return out