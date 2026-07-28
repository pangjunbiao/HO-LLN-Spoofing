"""
Authoritative causal threshold and persistence alarm rules.

This module is intentionally independent of the legacy Step-10/13/15 alarm
implementations. New learning-based models should use this module only.

Locked semantics
----------------
1. raw_alarm[t] = probability[t] >= theta, but only on valid rows.
2. A confirmed alarm is active only after `persistence` consecutive valid,
   threshold-positive rows within the same segment.
3. Invalid rows break the consecutive-positive run.
4. Segment boundaries reset every alarm state.
5. `confirmed_alarm_onset[t]` marks a 0 -> 1 transition of the confirmed alarm
   inside a segment. It is the only signal eligible for event detection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AlarmSequenceResult:
    """Complete row-level output of the authoritative alarm rule."""

    theta: float
    persistence: int
    probabilities: np.ndarray
    valid_mask: np.ndarray
    segment_ids: np.ndarray
    raw_alarm: np.ndarray
    confirmed_alarm: np.ndarray
    confirmed_alarm_onset: np.ndarray

    def to_dict_summary(self) -> Dict[str, Any]:
        return {
            "theta": float(self.theta),
            "persistence": int(self.persistence),
            "rows": int(self.probabilities.size),
            "valid_rows": int(self.valid_mask.sum()),
            "raw_alarm_rows": int(self.raw_alarm.sum()),
            "confirmed_alarm_rows": int(self.confirmed_alarm.sum()),
            "confirmed_alarm_onsets": int(self.confirmed_alarm_onset.sum()),
            "segments": int(count_segment_blocks(self.segment_ids)),
        }


def _as_1d(values: Sequence[Any], name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def as_probabilities(values: Sequence[Any]) -> np.ndarray:
    """Return a strict finite probability vector in [0, 1]."""
    arr = _as_1d(values, "probabilities").astype(float)
    if not np.all(np.isfinite(arr)):
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"probabilities contains {bad} non-finite values.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        lo = float(arr.min())
        hi = float(arr.max())
        raise ValueError(
            f"probabilities must be within [0, 1]; observed range [{lo}, {hi}]."
        )
    return arr.astype(np.float64, copy=False)


def as_binary_mask(values: Sequence[Any], name: str = "valid_mask") -> np.ndarray:
    """Return a strict binary boolean mask."""
    arr = _as_1d(values, name)
    numeric = arr.astype(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite values.")
    unique = set(np.unique(numeric).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"{name} must contain only 0/1 values; found {sorted(unique)}.")
    return numeric.astype(bool)


def as_segment_ids(values: Sequence[Any]) -> np.ndarray:
    """Return non-empty string segment IDs and validate block contiguity."""
    arr = _as_1d(values, "segment_ids").astype(object)
    cleaned = np.empty(arr.size, dtype=object)
    for index, value in enumerate(arr):
        if value is None:
            raise ValueError(f"segment_ids[{index}] is None.")
        text = str(value).strip()
        if not text or text.lower() == "nan":
            raise ValueError(f"segment_ids[{index}] is empty/invalid.")
        cleaned[index] = text
    validate_contiguous_segment_blocks(cleaned)
    return cleaned


def validate_equal_lengths(**arrays: Sequence[Any]) -> int:
    """Validate that all supplied arrays have the same nonzero length."""
    lengths = {name: int(np.asarray(value).reshape(-1).size) for name, value in arrays.items()}
    if not lengths:
        raise ValueError("No arrays supplied for length validation.")
    unique = set(lengths.values())
    if len(unique) != 1:
        raise ValueError(f"Array length mismatch: {lengths}")
    length = next(iter(unique))
    if length <= 0:
        raise ValueError("Input arrays are empty.")
    return length


def validate_contiguous_segment_blocks(segment_ids: Sequence[Any]) -> None:
    """
    Require each segment to occupy exactly one contiguous block.

    This prevents accidental reordering such as A, B, A, which would make
    causal state resets and event boundaries ambiguous.
    """
    segments = _as_1d(segment_ids, "segment_ids").astype(str)
    seen = set()
    previous: Optional[str] = None
    for position, segment in enumerate(segments):
        if segment != previous:
            if segment in seen:
                raise ValueError(
                    f"Segment {segment!r} reappears at position {position}; "
                    "segments must form contiguous chronological blocks."
                )
            seen.add(segment)
            previous = segment


def count_segment_blocks(segment_ids: Sequence[Any]) -> int:
    segments = _as_1d(segment_ids, "segment_ids").astype(str)
    return int(1 + np.sum(segments[1:] != segments[:-1]))


def threshold_probabilities(
    probabilities: Sequence[Any],
    theta: float,
    valid_mask: Sequence[Any],
) -> np.ndarray:
    """Compute valid-row raw threshold decisions."""
    p = as_probabilities(probabilities)
    valid = as_binary_mask(valid_mask)
    validate_equal_lengths(probabilities=p, valid_mask=valid)

    theta = float(theta)
    if not np.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must be finite and within [0, 1], got {theta!r}.")

    return ((p >= theta) & valid).astype(np.int64)


def apply_persistence_by_segment(
    raw_alarm: Sequence[Any],
    segment_ids: Sequence[Any],
    valid_mask: Sequence[Any],
    persistence: int,
) -> np.ndarray:
    """
    Apply the locked persistence rule independently within every segment.

    The confirmed alarm is 1 from the N_p-th consecutive valid positive row
    until the run is broken by a negative row, an invalid row, or a segment
    boundary. It is not permanently latched.
    """
    raw = as_binary_mask(raw_alarm, "raw_alarm")
    valid = as_binary_mask(valid_mask)
    segments = as_segment_ids(segment_ids)
    validate_equal_lengths(raw_alarm=raw, valid_mask=valid, segment_ids=segments)

    persistence = int(persistence)
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence}.")

    confirmed = np.zeros(raw.size, dtype=np.int64)
    run_length = 0
    previous_segment: Optional[str] = None

    for position in range(raw.size):
        segment = str(segments[position])

        if segment != previous_segment:
            run_length = 0
            previous_segment = segment

        if not valid[position]:
            run_length = 0
            confirmed[position] = 0
            continue

        if raw[position]:
            run_length += 1
            confirmed[position] = int(run_length >= persistence)
        else:
            run_length = 0
            confirmed[position] = 0

    return confirmed


def find_confirmed_alarm_onsets(
    confirmed_alarm: Sequence[Any],
    segment_ids: Sequence[Any],
) -> np.ndarray:
    """Mark confirmed-alarm 0 -> 1 transitions inside each segment."""
    confirmed = as_binary_mask(confirmed_alarm, "confirmed_alarm")
    segments = as_segment_ids(segment_ids)
    validate_equal_lengths(confirmed_alarm=confirmed, segment_ids=segments)

    onset = np.zeros(confirmed.size, dtype=np.int64)
    for position in range(confirmed.size):
        if not confirmed[position]:
            continue
        if position == 0 or segments[position] != segments[position - 1]:
            onset[position] = 1
        elif not confirmed[position - 1]:
            onset[position] = 1
    return onset


def build_alarm_sequence(
    probabilities: Sequence[Any],
    segment_ids: Sequence[Any],
    valid_mask: Sequence[Any],
    theta: float,
    persistence: int,
) -> AlarmSequenceResult:
    """Build raw, confirmed, and onset alarm vectors under the locked rules."""
    p = as_probabilities(probabilities)
    valid = as_binary_mask(valid_mask)
    segments = as_segment_ids(segment_ids)
    validate_equal_lengths(probabilities=p, valid_mask=valid, segment_ids=segments)

    raw = threshold_probabilities(p, theta=theta, valid_mask=valid)
    confirmed = apply_persistence_by_segment(
        raw_alarm=raw,
        segment_ids=segments,
        valid_mask=valid,
        persistence=persistence,
    )
    onset = find_confirmed_alarm_onsets(
        confirmed_alarm=confirmed,
        segment_ids=segments,
    )

    return AlarmSequenceResult(
        theta=float(theta),
        persistence=int(persistence),
        probabilities=p,
        valid_mask=valid,
        segment_ids=segments,
        raw_alarm=raw,
        confirmed_alarm=confirmed,
        confirmed_alarm_onset=onset,
    )


__all__ = [
    "AlarmSequenceResult",
    "apply_persistence_by_segment",
    "as_binary_mask",
    "as_probabilities",
    "as_segment_ids",
    "build_alarm_sequence",
    "count_segment_blocks",
    "find_confirmed_alarm_onsets",
    "threshold_probabilities",
    "validate_contiguous_segment_blocks",
    "validate_equal_lengths",
]
