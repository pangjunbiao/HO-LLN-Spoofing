"""
Authoritative event extraction and event-level metrics.

Ground-truth attack events are defined only by label continuity within a
segment. The validity mask never creates, removes, or splits an attack event.

Detection requires a *new confirmed-alarm onset* inside an attack interval.
A confirmed alarm that began in a preceding normal interval and merely remains
active at attack onset receives no zero-delay credit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.unified_alarm_rules import (
    as_binary_mask,
    as_segment_ids,
    validate_equal_lengths,
)


@dataclass(frozen=True)
class AttackEvent:
    event_index: int
    event_id: str
    segment_id: str
    start_position: int
    end_position: int
    duration_rows: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttackEventDetection:
    event_index: int
    event_id: str
    segment_id: str
    detected: bool
    alarm_active_before_attack: bool
    attack_start_position: int
    attack_end_position: int
    first_eligible_alarm_onset_position: Optional[int]
    delay_rows: Optional[int]
    delay_seconds: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FalseAlarmEvent:
    event_index: int
    event_id: str
    segment_id: str
    start_position: int
    end_position: int
    duration_rows: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventMetricSummary:
    attack_events_total: int
    attack_events_detected: int
    attack_events_missed: int
    attack_detection_rate: float

    detected_event_delay_count: int
    mean_detection_delay_seconds: Optional[float]
    median_detection_delay_seconds: Optional[float]
    max_detection_delay_seconds: Optional[float]
    mean_detection_delay_rows: Optional[float]
    median_detection_delay_rows: Optional[float]
    max_detection_delay_rows: Optional[int]

    attack_events_with_preexisting_alarm: int

    false_alarm_rows: int
    valid_normal_rows: int
    normal_row_false_alarm_rate: float
    false_alarm_events: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def as_binary_labels(values: Sequence[Any], name: str = "labels") -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    numeric = arr.astype(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite values.")
    unique = set(np.unique(numeric).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"{name} must contain only binary 0/1 labels; found {sorted(unique)}.")
    return numeric.astype(np.int64)


def as_delta_t(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError("delta_t is empty.")
    arr = arr.astype(float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("delta_t contains non-finite values.")
    if np.any(arr < 0.0):
        raise ValueError("delta_t must be nonnegative.")
    return arr


def _duration_seconds(delta_t: np.ndarray, start: int, end: int) -> float:
    """
    Elapsed time from the event's first row to its last row.

    delta_t[t] is interpreted as elapsed time from row t-1 to row t, so the
    boundary transition into the first event row is not counted.
    """
    if end <= start:
        return 0.0
    return float(np.sum(delta_t[start + 1 : end + 1]))


def extract_attack_events(
    labels: Sequence[Any],
    segment_ids: Sequence[Any],
    delta_t: Sequence[Any],
) -> List[AttackEvent]:
    """
    Extract maximal contiguous label==1 intervals within each segment.

    No validity mask is accepted by design.
    """
    y = as_binary_labels(labels)
    segments = as_segment_ids(segment_ids)
    dt = as_delta_t(delta_t)
    validate_equal_lengths(labels=y, segment_ids=segments, delta_t=dt)

    events: List[AttackEvent] = []
    start: Optional[int] = None
    event_segment: Optional[str] = None

    def close_event(end_position: int) -> None:
        nonlocal start, event_segment
        if start is None or event_segment is None:
            return
        index = len(events) + 1
        events.append(
            AttackEvent(
                event_index=index,
                event_id=f"attack_{index:04d}",
                segment_id=event_segment,
                start_position=int(start),
                end_position=int(end_position),
                duration_rows=int(end_position - start + 1),
                duration_seconds=_duration_seconds(dt, start, end_position),
            )
        )
        start = None
        event_segment = None

    for position in range(y.size):
        segment = str(segments[position])
        is_attack = bool(y[position] == 1)

        if start is not None:
            segment_changed = segment != event_segment
            if segment_changed or not is_attack:
                close_event(position - 1)

        if start is None and is_attack:
            start = position
            event_segment = segment

    if start is not None:
        close_event(y.size - 1)

    return events


def extract_false_alarm_events(
    confirmed_alarm: Sequence[Any],
    labels: Sequence[Any],
    segment_ids: Sequence[Any],
    delta_t: Sequence[Any],
) -> List[FalseAlarmEvent]:
    """Extract maximal confirmed-alarm intervals occurring on normal rows."""
    alarm = as_binary_mask(confirmed_alarm, "confirmed_alarm")
    y = as_binary_labels(labels)
    segments = as_segment_ids(segment_ids)
    dt = as_delta_t(delta_t)
    validate_equal_lengths(
        confirmed_alarm=alarm,
        labels=y,
        segment_ids=segments,
        delta_t=dt,
    )

    condition = alarm & (y == 0)
    events: List[FalseAlarmEvent] = []
    start: Optional[int] = None
    event_segment: Optional[str] = None

    def close_event(end_position: int) -> None:
        nonlocal start, event_segment
        if start is None or event_segment is None:
            return
        index = len(events) + 1
        events.append(
            FalseAlarmEvent(
                event_index=index,
                event_id=f"false_alarm_{index:04d}",
                segment_id=event_segment,
                start_position=int(start),
                end_position=int(end_position),
                duration_rows=int(end_position - start + 1),
                duration_seconds=_duration_seconds(dt, start, end_position),
            )
        )
        start = None
        event_segment = None

    for position in range(condition.size):
        segment = str(segments[position])
        active = bool(condition[position])

        if start is not None:
            segment_changed = segment != event_segment
            if segment_changed or not active:
                close_event(position - 1)

        if start is None and active:
            start = position
            event_segment = segment

    if start is not None:
        close_event(condition.size - 1)

    return events


def compute_attack_event_detections(
    attack_events: Sequence[AttackEvent],
    confirmed_alarm: Sequence[Any],
    confirmed_alarm_onset: Sequence[Any],
    segment_ids: Sequence[Any],
    delta_t: Sequence[Any],
) -> List[AttackEventDetection]:
    """
    Match each attack to the first new confirmed-alarm onset inside that attack.

    A pre-existing alarm is recorded when the confirmed alarm was already active
    on the immediately preceding row in the same segment. Continuing alarm rows
    do not count as a new detection.
    """
    confirmed = as_binary_mask(confirmed_alarm, "confirmed_alarm")
    onset = as_binary_mask(confirmed_alarm_onset, "confirmed_alarm_onset")
    segments = as_segment_ids(segment_ids)
    dt = as_delta_t(delta_t)
    validate_equal_lengths(
        confirmed_alarm=confirmed,
        confirmed_alarm_onset=onset,
        segment_ids=segments,
        delta_t=dt,
    )

    detections: List[AttackEventDetection] = []

    for event in attack_events:
        start = int(event.start_position)
        end = int(event.end_position)

        if not (0 <= start <= end < confirmed.size):
            raise ValueError(f"Attack event positions are out of range: {event}")

        active_before = bool(
            start > 0
            and segments[start - 1] == event.segment_id
            and confirmed[start - 1]
        )

        eligible = np.flatnonzero(onset[start : end + 1]) + start
        first_onset = int(eligible[0]) if eligible.size else None

        if first_onset is None:
            detections.append(
                AttackEventDetection(
                    event_index=event.event_index,
                    event_id=event.event_id,
                    segment_id=event.segment_id,
                    detected=False,
                    alarm_active_before_attack=active_before,
                    attack_start_position=start,
                    attack_end_position=end,
                    first_eligible_alarm_onset_position=None,
                    delay_rows=None,
                    delay_seconds=None,
                )
            )
            continue

        delay_rows = int(first_onset - start)
        delay_seconds = (
            0.0
            if first_onset == start
            else float(np.sum(dt[start + 1 : first_onset + 1]))
        )

        detections.append(
            AttackEventDetection(
                event_index=event.event_index,
                event_id=event.event_id,
                segment_id=event.segment_id,
                detected=True,
                alarm_active_before_attack=active_before,
                attack_start_position=start,
                attack_end_position=end,
                first_eligible_alarm_onset_position=first_onset,
                delay_rows=delay_rows,
                delay_seconds=delay_seconds,
            )
        )

    return detections


def summarize_event_metrics(
    attack_events: Sequence[AttackEvent],
    detections: Sequence[AttackEventDetection],
    false_alarm_events: Sequence[FalseAlarmEvent],
    confirmed_alarm: Sequence[Any],
    labels: Sequence[Any],
    valid_mask: Sequence[Any],
) -> EventMetricSummary:
    confirmed = as_binary_mask(confirmed_alarm, "confirmed_alarm")
    y = as_binary_labels(labels)
    valid = as_binary_mask(valid_mask, "valid_mask")
    validate_equal_lengths(confirmed_alarm=confirmed, labels=y, valid_mask=valid)

    if len(attack_events) != len(detections):
        raise ValueError(
            f"Attack-event/detection count mismatch: {len(attack_events)} vs {len(detections)}."
        )
    for event, detection in zip(attack_events, detections):
        if event.event_id != detection.event_id:
            raise ValueError(
                f"Attack-event order mismatch: {event.event_id} vs {detection.event_id}."
            )

    detected = [item for item in detections if item.detected]
    delay_seconds = np.asarray(
        [float(item.delay_seconds) for item in detected if item.delay_seconds is not None],
        dtype=float,
    )
    delay_rows = np.asarray(
        [int(item.delay_rows) for item in detected if item.delay_rows is not None],
        dtype=float,
    )

    total = len(attack_events)
    detected_count = len(detected)
    missed = total - detected_count
    adr = float(detected_count / total) if total else 0.0

    normal_valid = valid & (y == 0)
    false_alarm_rows = int(np.sum(confirmed & normal_valid))
    valid_normal_rows = int(normal_valid.sum())
    normal_far = (
        float(false_alarm_rows / valid_normal_rows)
        if valid_normal_rows > 0
        else 0.0
    )

    return EventMetricSummary(
        attack_events_total=int(total),
        attack_events_detected=int(detected_count),
        attack_events_missed=int(missed),
        attack_detection_rate=adr,
        detected_event_delay_count=int(delay_seconds.size),
        mean_detection_delay_seconds=(
            float(delay_seconds.mean()) if delay_seconds.size else None
        ),
        median_detection_delay_seconds=(
            float(np.median(delay_seconds)) if delay_seconds.size else None
        ),
        max_detection_delay_seconds=(
            float(delay_seconds.max()) if delay_seconds.size else None
        ),
        mean_detection_delay_rows=(
            float(delay_rows.mean()) if delay_rows.size else None
        ),
        median_detection_delay_rows=(
            float(np.median(delay_rows)) if delay_rows.size else None
        ),
        max_detection_delay_rows=(
            int(delay_rows.max()) if delay_rows.size else None
        ),
        attack_events_with_preexisting_alarm=int(
            sum(item.alarm_active_before_attack for item in detections)
        ),
        false_alarm_rows=false_alarm_rows,
        valid_normal_rows=valid_normal_rows,
        normal_row_false_alarm_rate=normal_far,
        false_alarm_events=int(len(false_alarm_events)),
    )


def evaluate_event_level(
    labels: Sequence[Any],
    segment_ids: Sequence[Any],
    delta_t: Sequence[Any],
    valid_mask: Sequence[Any],
    confirmed_alarm: Sequence[Any],
    confirmed_alarm_onset: Sequence[Any],
) -> Tuple[
    List[AttackEvent],
    List[AttackEventDetection],
    List[FalseAlarmEvent],
    EventMetricSummary,
]:
    """Run the complete authoritative event-level evaluation."""
    attack_events = extract_attack_events(
        labels=labels,
        segment_ids=segment_ids,
        delta_t=delta_t,
    )
    detections = compute_attack_event_detections(
        attack_events=attack_events,
        confirmed_alarm=confirmed_alarm,
        confirmed_alarm_onset=confirmed_alarm_onset,
        segment_ids=segment_ids,
        delta_t=delta_t,
    )
    false_alarms = extract_false_alarm_events(
        confirmed_alarm=confirmed_alarm,
        labels=labels,
        segment_ids=segment_ids,
        delta_t=delta_t,
    )
    summary = summarize_event_metrics(
        attack_events=attack_events,
        detections=detections,
        false_alarm_events=false_alarms,
        confirmed_alarm=confirmed_alarm,
        labels=labels,
        valid_mask=valid_mask,
    )
    return attack_events, detections, false_alarms, summary


__all__ = [
    "AttackEvent",
    "AttackEventDetection",
    "EventMetricSummary",
    "FalseAlarmEvent",
    "as_binary_labels",
    "as_delta_t",
    "compute_attack_event_detections",
    "evaluate_event_level",
    "extract_attack_events",
    "extract_false_alarm_events",
    "summarize_event_metrics",
]
