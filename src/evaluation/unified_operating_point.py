"""
Validation-only operating-point selection for the unified event engine.

Locked grid
-----------
theta       : 0.05, 0.10, ..., 0.95
persistence : 1, 2, ..., 10

Locked candidate ordering
-------------------------
1. higher confirmed-alarm F1
2. higher attack detection rate
3. lower mean detection delay
4. lower FPR
5. higher recall
6. lower persistence
7. higher threshold
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.unified_evaluator import (
    UnifiedEvaluationResult,
    UnifiedPredictionBundle,
    evaluate_prediction_bundle,
)


DEFAULT_THRESHOLD_GRID: Tuple[float, ...] = tuple(
    round(value, 2) for value in np.arange(0.05, 1.00, 0.05).tolist()
)
DEFAULT_PERSISTENCE_GRID: Tuple[int, ...] = tuple(range(1, 11))

_ALLOWED_VALIDATION_NAMES = {
    "val",
    "validation",
    "dataset1_val",
    "dataset1_validation",
    "d1_val",
    "d1_validation",
}


@dataclass(frozen=True)
class OperatingPointCandidate:
    theta: float
    persistence: int
    f1: float
    attack_detection_rate: float
    mean_detection_delay_seconds: Optional[float]
    fpr: float
    recall: float
    precision: float
    auprc: Optional[float]
    auroc: Optional[float]
    attack_events_total: int
    attack_events_detected: int
    false_alarm_events: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatingPointSelectionResult:
    model_name: str
    monitor_split: str
    theta: float
    persistence: int
    objective: str
    candidate_count: int
    selected_candidate: OperatingPointCandidate
    selected_evaluation: UnifiedEvaluationResult
    candidates: List[OperatingPointCandidate] = field(default_factory=list)

    def to_dict(self, include_candidates: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_name": self.model_name,
            "monitor_split": self.monitor_split,
            "theta": float(self.theta),
            "persistence": int(self.persistence),
            "objective": self.objective,
            "candidate_count": int(self.candidate_count),
            "selected_candidate": self.selected_candidate.to_dict(),
            "selected_evaluation": self.selected_evaluation.to_dict(
                include_row_arrays=False
            ),
        }
        if include_candidates:
            payload["candidates"] = [item.to_dict() for item in self.candidates]
        return payload


def _clean_threshold_grid(values: Sequence[Any]) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        number = float(value)
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"Invalid threshold-grid value: {value!r}")
        cleaned.append(round(number, 10))
    unique = sorted(set(cleaned))
    if not unique:
        raise ValueError("Threshold grid is empty.")
    return unique


def _clean_persistence_grid(values: Sequence[Any]) -> List[int]:
    cleaned: List[int] = []
    for value in values:
        number = int(value)
        if number < 1:
            raise ValueError(f"Invalid persistence-grid value: {value!r}")
        cleaned.append(number)
    unique = sorted(set(cleaned))
    if not unique:
        raise ValueError("Persistence grid is empty.")
    return unique


def _validate_validation_split(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    if normalized not in _ALLOWED_VALIDATION_NAMES:
        raise ValueError(
            f"Operating-point selection is validation-only; got split_name={split_name!r}. "
            f"Allowed names: {sorted(_ALLOWED_VALIDATION_NAMES)}"
        )
    return normalized


def candidate_from_evaluation(
    evaluation: UnifiedEvaluationResult,
) -> OperatingPointCandidate:
    sample = evaluation.sample_metrics
    event = evaluation.event_metrics
    ranking = evaluation.ranking_metrics
    return OperatingPointCandidate(
        theta=float(evaluation.theta),
        persistence=int(evaluation.persistence),
        f1=float(sample.f1),
        attack_detection_rate=float(event.attack_detection_rate),
        mean_detection_delay_seconds=event.mean_detection_delay_seconds,
        fpr=float(sample.fpr),
        recall=float(sample.recall),
        precision=float(sample.precision),
        auprc=ranking.auprc,
        auroc=ranking.auroc,
        attack_events_total=int(event.attack_events_total),
        attack_events_detected=int(event.attack_events_detected),
        false_alarm_events=int(event.false_alarm_events),
    )


def candidate_sort_key(candidate: OperatingPointCandidate) -> Tuple[float, ...]:
    """
    Return a tuple maximized by Python's `max`.

    Missing delay is treated as +infinity, i.e., worst possible delay.
    """
    delay = candidate.mean_detection_delay_seconds
    delay_for_minimization = float(delay) if delay is not None else float("inf")
    return (
        float(candidate.f1),
        float(candidate.attack_detection_rate),
        -delay_for_minimization,
        -float(candidate.fpr),
        float(candidate.recall),
        -float(candidate.persistence),
        float(candidate.theta),
    )


def select_operating_point(
    validation_bundle: UnifiedPredictionBundle,
    threshold_grid: Sequence[Any] = DEFAULT_THRESHOLD_GRID,
    persistence_grid: Sequence[Any] = DEFAULT_PERSISTENCE_GRID,
    include_candidates: bool = True,
) -> OperatingPointSelectionResult:
    """Select one model-specific operating point on Dataset-1 validation only."""
    bundle = validation_bundle.validated()
    monitor_split = _validate_validation_split(bundle.split_name)

    thresholds = _clean_threshold_grid(threshold_grid)
    persistences = _clean_persistence_grid(persistence_grid)

    candidates: List[OperatingPointCandidate] = []
    evaluations: Dict[Tuple[float, int], UnifiedEvaluationResult] = {}

    for theta in thresholds:
        for persistence in persistences:
            evaluation = evaluate_prediction_bundle(
                bundle=bundle,
                theta=theta,
                persistence=persistence,
            )
            candidate = candidate_from_evaluation(evaluation)
            candidates.append(candidate)
            evaluations[(candidate.theta, candidate.persistence)] = evaluation

    selected = max(candidates, key=candidate_sort_key)
    selected_evaluation = evaluations[(selected.theta, selected.persistence)]

    return OperatingPointSelectionResult(
        model_name=bundle.model_name,
        monitor_split=monitor_split,
        theta=selected.theta,
        persistence=selected.persistence,
        objective="confirmed_alarm_f1",
        candidate_count=len(candidates),
        selected_candidate=selected,
        selected_evaluation=selected_evaluation,
        candidates=candidates if include_candidates else [],
    )


__all__ = [
    "DEFAULT_PERSISTENCE_GRID",
    "DEFAULT_THRESHOLD_GRID",
    "OperatingPointCandidate",
    "OperatingPointSelectionResult",
    "candidate_from_evaluation",
    "candidate_sort_key",
    "select_operating_point",
]
