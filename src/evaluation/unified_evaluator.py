"""
Unified sample-, ranking-, alarm-, and event-level evaluator.

Every probability-producing learning model should be adapted to
`UnifiedPredictionBundle` and evaluated through `evaluate_prediction_bundle`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.unified_alarm_rules import (
    AlarmSequenceResult,
    as_binary_mask,
    as_probabilities,
    as_segment_ids,
    build_alarm_sequence,
    count_segment_blocks,
    validate_equal_lengths,
)
from src.evaluation.unified_event_metrics import (
    AttackEvent,
    AttackEventDetection,
    EventMetricSummary,
    FalseAlarmEvent,
    as_binary_labels,
    as_delta_t,
    evaluate_event_level,
)


@dataclass(frozen=True)
class UnifiedPredictionBundle:
    """Strict common prediction contract for all learning-based models."""

    split_name: str
    model_name: str
    probabilities: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    segment_ids: np.ndarray
    delta_t: np.ndarray
    row_indices: Optional[np.ndarray] = None
    logits: Optional[np.ndarray] = None
    checkpoint_path: Optional[str] = None

    def validated(self) -> "UnifiedPredictionBundle":
        p = as_probabilities(self.probabilities)
        y = as_binary_labels(self.labels)
        valid = as_binary_mask(self.valid_mask, "valid_mask")
        segments = as_segment_ids(self.segment_ids)
        dt = as_delta_t(self.delta_t)
        validate_equal_lengths(
            probabilities=p,
            labels=y,
            valid_mask=valid,
            segment_ids=segments,
            delta_t=dt,
        )

        rows: Optional[np.ndarray] = None
        if self.row_indices is not None:
            rows = np.asarray(self.row_indices)
            if rows.ndim == 0:
                rows = rows.reshape(1)
            rows = rows.reshape(-1)
            if rows.size != p.size:
                raise ValueError(
                    f"row_indices length mismatch: expected {p.size}, got {rows.size}."
                )
            if not np.all(np.isfinite(rows.astype(float))):
                raise ValueError("row_indices contains non-finite values.")
            rows = rows.astype(np.int64)

            keys = [(str(segment), int(row)) for segment, row in zip(segments, rows)]
            if len(keys) != len(set(keys)):
                raise ValueError(
                    "Duplicate (segment_id, row_index) keys found. Deduplicate "
                    "overlapping window predictions before unified evaluation."
                )

        logits: Optional[np.ndarray] = None
        if self.logits is not None:
            logits = np.asarray(self.logits, dtype=float).reshape(-1)
            if logits.size != p.size:
                raise ValueError(
                    f"logits length mismatch: expected {p.size}, got {logits.size}."
                )
            if not np.all(np.isfinite(logits)):
                raise ValueError("logits contains non-finite values.")

        split = str(self.split_name).strip()
        model = str(self.model_name).strip()
        if not split:
            raise ValueError("split_name is empty.")
        if not model:
            raise ValueError("model_name is empty.")

        return UnifiedPredictionBundle(
            split_name=split,
            model_name=model,
            probabilities=p,
            labels=y,
            valid_mask=valid,
            segment_ids=segments,
            delta_t=dt,
            row_indices=rows,
            logits=logits,
            checkpoint_path=(
                None if self.checkpoint_path is None else str(self.checkpoint_path)
            ),
        )

    def to_dict_summary(self) -> Dict[str, Any]:
        bundle = self.validated()
        valid_labels = bundle.labels[bundle.valid_mask]
        return {
            "split_name": bundle.split_name,
            "model_name": bundle.model_name,
            "rows": int(bundle.labels.size),
            "valid_rows": int(bundle.valid_mask.sum()),
            "invalid_rows": int((~bundle.valid_mask).sum()),
            "normal_valid_rows": int(np.sum(valid_labels == 0)),
            "attack_valid_rows": int(np.sum(valid_labels == 1)),
            "segments": int(count_segment_blocks(bundle.segment_ids)),
            "probability_min": float(bundle.probabilities.min()),
            "probability_max": float(bundle.probabilities.max()),
            "probability_mean": float(bundle.probabilities.mean()),
            "checkpoint_path": bundle.checkpoint_path,
        }


@dataclass(frozen=True)
class RankingMetrics:
    auprc: Optional[float]
    auroc: Optional[float]
    valid_rows: int
    positive_rows: int
    negative_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SampleMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    specificity: float
    tp: int
    fp: int
    tn: int
    fn: int
    valid_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnifiedEvaluationResult:
    model_name: str
    split_name: str
    theta: float
    persistence: int
    prediction_summary: Dict[str, Any]
    ranking_metrics: RankingMetrics
    sample_metrics: SampleMetrics
    event_metrics: EventMetricSummary
    attack_events: List[AttackEvent]
    attack_detections: List[AttackEventDetection]
    false_alarm_events: List[FalseAlarmEvent]
    alarm_sequence: AlarmSequenceResult

    def to_dict(self, include_row_arrays: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_name": self.model_name,
            "split_name": self.split_name,
            "theta": float(self.theta),
            "persistence": int(self.persistence),
            "prediction_summary": dict(self.prediction_summary),
            "ranking_metrics": self.ranking_metrics.to_dict(),
            "sample_metrics": self.sample_metrics.to_dict(),
            "event_metrics": self.event_metrics.to_dict(),
            "attack_events": [item.to_dict() for item in self.attack_events],
            "attack_detections": [item.to_dict() for item in self.attack_detections],
            "false_alarm_events": [item.to_dict() for item in self.false_alarm_events],
            "alarm_summary": self.alarm_sequence.to_dict_summary(),
        }
        if include_row_arrays:
            payload["row_arrays"] = {
                "raw_alarm": self.alarm_sequence.raw_alarm.astype(int).tolist(),
                "confirmed_alarm": self.alarm_sequence.confirmed_alarm.astype(int).tolist(),
                "confirmed_alarm_onset": (
                    self.alarm_sequence.confirmed_alarm_onset.astype(int).tolist()
                ),
            }
        return payload


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_sample_metrics(
    labels: Sequence[Any],
    confirmed_alarm: Sequence[Any],
    valid_mask: Sequence[Any],
) -> SampleMetrics:
    y = as_binary_labels(labels)
    pred = as_binary_mask(confirmed_alarm, "confirmed_alarm").astype(np.int64)
    valid = as_binary_mask(valid_mask, "valid_mask")
    validate_equal_lengths(labels=y, confirmed_alarm=pred, valid_mask=valid)

    y = y[valid]
    pred = pred[valid]

    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, tp + fp + tn + fn)
    fpr = _safe_divide(fp, fp + tn)
    specificity = _safe_divide(tn, tn + fp)

    return SampleMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        specificity=specificity,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        valid_rows=int(y.size),
    )


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = scores[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    distinct_end = np.r_[score_sorted[1:] != score_sorted[:-1], True]
    tp_at = tp[distinct_end].astype(float)
    fp_at = fp[distinct_end].astype(float)

    precision = tp_at / np.maximum(tp_at + fp_at, 1.0)
    recall = tp_at / positives
    recall_prev = np.r_[0.0, recall[:-1]]

    return float(np.sum((recall - recall_prev) * precision))


def _average_tied_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None

    ranks = _average_tied_ranks(scores)
    rank_sum_positive = float(ranks[y_true == 1].sum())
    u_statistic = rank_sum_positive - positives * (positives + 1) / 2.0
    return float(u_statistic / (positives * negatives))


def compute_ranking_metrics(
    labels: Sequence[Any],
    probabilities: Sequence[Any],
    valid_mask: Sequence[Any],
) -> RankingMetrics:
    y = as_binary_labels(labels)
    p = as_probabilities(probabilities)
    valid = as_binary_mask(valid_mask, "valid_mask")
    validate_equal_lengths(labels=y, probabilities=p, valid_mask=valid)

    y = y[valid]
    p = p[valid]
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    return RankingMetrics(
        auprc=_average_precision(y, p),
        auroc=_roc_auc(y, p),
        valid_rows=int(y.size),
        positive_rows=positives,
        negative_rows=negatives,
    )


def evaluate_prediction_bundle(
    bundle: UnifiedPredictionBundle,
    theta: float,
    persistence: int,
) -> UnifiedEvaluationResult:
    """Run the authoritative complete evaluation for one model/split."""
    bundle = bundle.validated()

    alarm = build_alarm_sequence(
        probabilities=bundle.probabilities,
        segment_ids=bundle.segment_ids,
        valid_mask=bundle.valid_mask,
        theta=theta,
        persistence=persistence,
    )
    ranking = compute_ranking_metrics(
        labels=bundle.labels,
        probabilities=bundle.probabilities,
        valid_mask=bundle.valid_mask,
    )
    sample = compute_sample_metrics(
        labels=bundle.labels,
        confirmed_alarm=alarm.confirmed_alarm,
        valid_mask=bundle.valid_mask,
    )
    events, detections, false_alarms, event_summary = evaluate_event_level(
        labels=bundle.labels,
        segment_ids=bundle.segment_ids,
        delta_t=bundle.delta_t,
        valid_mask=bundle.valid_mask,
        confirmed_alarm=alarm.confirmed_alarm,
        confirmed_alarm_onset=alarm.confirmed_alarm_onset,
    )

    return UnifiedEvaluationResult(
        model_name=bundle.model_name,
        split_name=bundle.split_name,
        theta=float(theta),
        persistence=int(persistence),
        prediction_summary=bundle.to_dict_summary(),
        ranking_metrics=ranking,
        sample_metrics=sample,
        event_metrics=event_summary,
        attack_events=events,
        attack_detections=detections,
        false_alarm_events=false_alarms,
        alarm_sequence=alarm,
    )


__all__ = [
    "RankingMetrics",
    "SampleMetrics",
    "UnifiedEvaluationResult",
    "UnifiedPredictionBundle",
    "compute_ranking_metrics",
    "compute_sample_metrics",
    "evaluate_prediction_bundle",
]
