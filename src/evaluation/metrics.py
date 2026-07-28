"""
Evaluation metrics for AV-GPS causal spoofing detection.

Step 10 purpose:
- implement locked primary metrics before model training,
- ensure every model, baseline, and ablation is evaluated identically,
- keep threshold-dependent metrics separate from validation-only threshold selection.

Primary locked metrics:
- AUPRC
- F1
- FPR
- Attack Detection Rate
- Detection Delay

Secondary metrics:
- AUROC
- Precision
- Recall
- Runtime
- Normal-Segment FAR

Important:
AUPRC/AUROC are computed from probabilities.
F1/FPR/Precision/Recall are computed from binary decisions or confirmed alarms.
Attack Detection Rate and Detection Delay are event-level metrics implemented in alarm_rules.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import math
import time

import numpy as np


EPS = 1.0e-12


@dataclass
class BinaryConfusionCounts:
    """Binary confusion matrix counts."""

    tp: int
    fp: int
    tn: int
    fn: int
    support: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdMetricResult:
    """Threshold-dependent time-step metric result."""

    threshold: float
    persistence: int

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
    support: int

    valid_rows_used: int
    positive_rate_prediction: float
    positive_rate_label: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankingMetricResult:
    """Threshold-free probability ranking metrics."""

    auprc: float
    auroc: float
    valid_rows_used: int
    positive_rows: int
    negative_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeMeter:
    """Simple runtime meter for training/evaluation timing."""

    start_time: float
    end_time: Optional[float] = None

    @classmethod
    def start(cls) -> "RuntimeMeter":
        return cls(start_time=time.perf_counter())

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        return self.elapsed_seconds()

    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return float(end - self.start_time)


def _as_numpy_1d(values: Sequence[Any], name: str) -> np.ndarray:
    """Convert input into a 1D numpy array."""
    arr = np.asarray(values)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    return arr


def _as_binary_labels(y_true: Sequence[Any]) -> np.ndarray:
    """Convert labels to int binary array."""
    y = _as_numpy_1d(y_true, "y_true")
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(y)
    if not np.all(finite):
        raise ValueError("y_true contains non-finite values.")

    y_bin = (y >= 0.5).astype(np.int64)
    return y_bin


def _as_probabilities(y_score: Sequence[Any]) -> np.ndarray:
    """Convert model scores to finite probability-like float array."""
    p = _as_numpy_1d(y_score, "y_score")
    p = np.asarray(p, dtype=float)

    finite = np.isfinite(p)
    if not np.all(finite):
        bad_count = int((~finite).sum())
        raise ValueError(f"y_score contains {bad_count} non-finite values.")

    # Keep scores in [0, 1] for probability-based evaluation.
    # This protects later code if logits are accidentally passed.
    p = np.clip(p, 0.0, 1.0)

    return p


def _as_optional_mask(mask: Optional[Sequence[Any]], length: int) -> np.ndarray:
    """Convert optional valid/loss mask to boolean array."""
    if mask is None:
        return np.ones(length, dtype=bool)

    arr = _as_numpy_1d(mask, "mask")

    if len(arr) != length:
        raise ValueError(
            f"Mask length mismatch. Expected {length}, got {len(arr)}."
        )

    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)

    return arr > 0.5


def filter_valid_rows(
    y_true: Sequence[Any],
    y_score: Optional[Sequence[Any]] = None,
    valid_mask: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Filter arrays by valid mask.

    Returns:
        y_valid, score_valid_or_none, mask_bool
    """
    y = _as_binary_labels(y_true)
    mask = _as_optional_mask(valid_mask, len(y))

    if y_score is None:
        return y[mask], None, mask

    p = _as_probabilities(y_score)

    if len(p) != len(y):
        raise ValueError(
            f"Length mismatch. y_true has {len(y)} rows, y_score has {len(p)} rows."
        )

    return y[mask], p[mask], mask


def compute_confusion_counts(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    valid_mask: Optional[Sequence[Any]] = None,
) -> BinaryConfusionCounts:
    """
    Compute TP/FP/TN/FN.

    y_pred can be a threshold decision or a confirmed alarm vector.
    """
    y = _as_binary_labels(y_true)
    pred = _as_binary_labels(y_pred)

    if len(y) != len(pred):
        raise ValueError(
            f"Length mismatch. y_true has {len(y)} rows, y_pred has {len(pred)} rows."
        )

    mask = _as_optional_mask(valid_mask, len(y))

    y = y[mask]
    pred = pred[mask]

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    return BinaryConfusionCounts(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        support=int(len(y)),
    )


def safe_divide(numerator: float, denominator: float) -> float:
    """Safe division returning 0.0 when denominator is zero."""
    denominator = float(denominator)
    if abs(denominator) <= EPS:
        return 0.0
    return float(numerator) / denominator


def f1_from_precision_recall(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    return safe_divide(2.0 * precision * recall, precision + recall)


def compute_metrics_from_counts(
    counts: BinaryConfusionCounts,
    threshold: float,
    persistence: int = 1,
) -> ThresholdMetricResult:
    """Compute threshold-dependent metrics from confusion counts."""
    tp = counts.tp
    fp = counts.fp
    tn = counts.tn
    fn = counts.fn
    support = counts.support

    accuracy = safe_divide(tp + tn, support)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = f1_from_precision_recall(precision, recall)
    fpr = safe_divide(fp, fp + tn)
    specificity = safe_divide(tn, tn + fp)

    pred_positive_rate = safe_divide(tp + fp, support)
    label_positive_rate = safe_divide(tp + fn, support)

    return ThresholdMetricResult(
        threshold=float(threshold),
        persistence=int(persistence),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fpr=float(fpr),
        specificity=float(specificity),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        support=int(support),
        valid_rows_used=int(support),
        positive_rate_prediction=float(pred_positive_rate),
        positive_rate_label=float(label_positive_rate),
    )


def threshold_scores(
    y_score: Sequence[Any],
    threshold: float,
    valid_mask: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """
    Convert probabilities to binary decisions c_t = I(p_hat_t >= theta).

    If valid_mask is provided, invalid rows are forced to 0.
    """
    p = _as_probabilities(y_score)
    pred = (p >= float(threshold)).astype(np.int64)

    if valid_mask is not None:
        mask = _as_optional_mask(valid_mask, len(p))
        pred = pred * mask.astype(np.int64)

    return pred


def compute_threshold_metrics(
    y_true: Sequence[Any],
    y_score: Optional[Sequence[Any]] = None,
    y_pred: Optional[Sequence[Any]] = None,
    threshold: float = 0.5,
    persistence: int = 1,
    valid_mask: Optional[Sequence[Any]] = None,
) -> ThresholdMetricResult:
    """
    Compute threshold-dependent metrics.

    Either y_score or y_pred must be provided.
    If y_score is provided and y_pred is None, predictions are thresholded.
    """
    if y_pred is None:
        if y_score is None:
            raise ValueError("Either y_score or y_pred must be provided.")
        y_pred = threshold_scores(
            y_score=y_score,
            threshold=threshold,
            valid_mask=valid_mask,
        )

    counts = compute_confusion_counts(
        y_true=y_true,
        y_pred=y_pred,
        valid_mask=valid_mask,
    )

    return compute_metrics_from_counts(
        counts=counts,
        threshold=threshold,
        persistence=persistence,
    )


def _compute_average_precision_fallback(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Small fallback implementation of average precision.

    sklearn is preferred when available.
    """
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    positives = int((y_sorted == 1).sum())
    if positives == 0:
        return float("nan")

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    precision = tp_cum / np.maximum(tp_cum + fp_cum, EPS)

    ap = float(np.sum(precision[y_sorted == 1]) / positives)
    return ap


def _compute_auroc_fallback(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Rank-based AUROC fallback.

    sklearn is preferred when available.
    """
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]

    n_pos = len(positives)
    n_neg = len(negatives)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Mann-Whitney interpretation with tie handling.
    comparisons = 0.0
    total = float(n_pos * n_neg)

    for pos_score in positives:
        comparisons += float((pos_score > negatives).sum())
        comparisons += 0.5 * float((pos_score == negatives).sum())

    return float(comparisons / total)


def compute_ranking_metrics(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    valid_mask: Optional[Sequence[Any]] = None,
) -> RankingMetricResult:
    """
    Compute threshold-free ranking metrics: AUPRC and AUROC.

    If only one class is present after masking, AUROC is NaN.
    """
    y, p, _mask = filter_valid_rows(
        y_true=y_true,
        y_score=y_score,
        valid_mask=valid_mask,
    )

    assert p is not None

    positive_rows = int((y == 1).sum())
    negative_rows = int((y == 0).sum())

    if len(y) == 0:
        return RankingMetricResult(
            auprc=float("nan"),
            auroc=float("nan"),
            valid_rows_used=0,
            positive_rows=0,
            negative_rows=0,
        )

    if positive_rows == 0:
        auprc = float("nan")
    else:
        try:
            from sklearn.metrics import average_precision_score

            auprc = float(average_precision_score(y, p))
        except Exception:
            auprc = _compute_average_precision_fallback(y, p)

    if positive_rows == 0 or negative_rows == 0:
        auroc = float("nan")
    else:
        try:
            from sklearn.metrics import roc_auc_score

            auroc = float(roc_auc_score(y, p))
        except Exception:
            auroc = _compute_auroc_fallback(y, p)

    return RankingMetricResult(
        auprc=float(auprc),
        auroc=float(auroc),
        valid_rows_used=int(len(y)),
        positive_rows=positive_rows,
        negative_rows=negative_rows,
    )


def compute_probability_summary(
    y_score: Sequence[Any],
    valid_mask: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Summarize probability distribution for diagnostics."""
    p = _as_probabilities(y_score)
    mask = _as_optional_mask(valid_mask, len(p))
    p = p[mask]

    if len(p) == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q50": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
        }

    return {
        "count": int(len(p)),
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "mean": float(np.mean(p)),
        "std": float(np.std(p)),
        "q01": float(np.quantile(p, 0.01)),
        "q05": float(np.quantile(p, 0.05)),
        "q50": float(np.quantile(p, 0.50)),
        "q95": float(np.quantile(p, 0.95)),
        "q99": float(np.quantile(p, 0.99)),
    }


def compute_time_step_metric_bundle(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    threshold: float,
    persistence: int = 1,
    valid_mask: Optional[Sequence[Any]] = None,
    y_pred: Optional[Sequence[Any]] = None,
    runtime_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute common time-step metric bundle.

    This does not compute event-level attack detection/delay. Those are added
    by alarm_rules.py / threshold_selection.py.
    """
    ranking = compute_ranking_metrics(
        y_true=y_true,
        y_score=y_score,
        valid_mask=valid_mask,
    )

    threshold_metrics = compute_threshold_metrics(
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred,
        threshold=threshold,
        persistence=persistence,
        valid_mask=valid_mask,
    )

    probability_summary = compute_probability_summary(
        y_score=y_score,
        valid_mask=valid_mask,
    )

    return {
        "ranking": ranking.to_dict(),
        "threshold_metrics": threshold_metrics.to_dict(),
        "probability_summary": probability_summary,
        "runtime_seconds": None
        if runtime_seconds is None
        else float(runtime_seconds),
        "primary_time_step_metrics": {
            "AUPRC": ranking.auprc,
            "F1": threshold_metrics.f1,
            "FPR": threshold_metrics.fpr,
        },
        "secondary_time_step_metrics": {
            "AUROC": ranking.auroc,
            "Precision": threshold_metrics.precision,
            "Recall": threshold_metrics.recall,
            "Runtime": None
            if runtime_seconds is None
            else float(runtime_seconds),
        },
    }


def nan_safe_float(value: Any, digits: Optional[int] = None) -> Optional[float]:
    """Convert value to JSON-safe float or None."""
    try:
        value_float = float(value)
    except Exception:
        return None

    if not math.isfinite(value_float):
        return None

    if digits is not None:
        return round(value_float, int(digits))

    return value_float


def compact_metric_row(
    metric_bundle: Mapping[str, Any],
    event_metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a compact row for results tables.

    Expected columns:
    Model | AUROC | AUPRC | F1 | Recall | FPR | Attack Detection Rate | Detection Delay
    """
    ranking = metric_bundle.get("ranking", {})
    thresh = metric_bundle.get("threshold_metrics", {})

    row = {
        "AUROC": ranking.get("auroc"),
        "AUPRC": ranking.get("auprc"),
        "F1": thresh.get("f1"),
        "Precision": thresh.get("precision"),
        "Recall": thresh.get("recall"),
        "FPR": thresh.get("fpr"),
        "Runtime": metric_bundle.get("runtime_seconds"),
    }

    if event_metrics is not None:
        row["Attack Detection Rate"] = event_metrics.get("attack_detection_rate")
        row["Detection Delay"] = event_metrics.get("mean_detection_delay_seconds")
        row["Normal-Segment FAR"] = event_metrics.get("normal_segment_far")

    return row