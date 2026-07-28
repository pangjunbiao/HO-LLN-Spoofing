"""
Dataset loading, candidate scoring, prediction, and artifact helpers for Step 5.

This module enforces:
- the exact 15-feature GPS-IDS contract for every classifier;
- identical row identities and segment assignments across all models;
- train-only fitting of imputation/scaling/model state;
- strict standardized prediction bundles;
- unified operating-point and event evaluation.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from external_baselines.gps_ids_reproduction.feature_contract import (
    GPS_IDS_MODEL_FEATURES,
    GPS_IDS_OUTPUT_METADATA_COLUMNS,
    build_gps_ids_feature_contract,
)
from external_baselines.gps_ids_reproduction.hyperparameters import (
    CandidateSpec,
)
from external_baselines.gps_ids_reproduction.models import (
    PipelineBuildSpec,
    build_pipeline,
)
from src.evaluation.artifact_manifest import save_strict_json
from src.evaluation.prediction_bundle_adapter import (
    SavedPredictionBundleArtifact,
    StandardizedPredictionBundle,
    save_standardized_prediction_bundle,
)
from src.evaluation.unified_evaluator import (
    UnifiedEvaluationResult,
    evaluate_prediction_bundle,
)


@dataclass(frozen=True)
class GPSIDSFeatureDataset:
    split_name: str
    source_path: str
    feature_names: Tuple[str, ...]
    feature_hash: str
    X: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    feature_complete_mask: np.ndarray
    target_yaw_missing: np.ndarray
    segment_ids: np.ndarray
    row_indices: np.ndarray
    within_segment_indices: np.ndarray
    delta_t: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.labels.size)

    @property
    def valid_rows(self) -> int:
        return int(self.valid_mask.sum())

    def identity_hash_input(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "rows": self.rows,
            "segment_ids": self.segment_ids.tolist(),
            "row_indices": self.row_indices.tolist(),
            "within_segment_indices": (
                self.within_segment_indices.tolist()
            ),
            "labels": self.labels.tolist(),
            "valid_mask": self.valid_mask.astype(int).tolist(),
            "delta_t": self.delta_t.tolist(),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "source_path": self.source_path,
            "rows": self.rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": int(self.rows - self.valid_rows),
            "segments": int(len(np.unique(self.segment_ids))),
            "normal_rows": int(np.sum(self.labels == 0)),
            "attack_rows": int(np.sum(self.labels == 1)),
            "feature_complete_rows": int(
                self.feature_complete_mask.sum()
            ),
            "feature_incomplete_rows": int(
                self.rows - self.feature_complete_mask.sum()
            ),
            "target_yaw_missing_rows": int(
                self.target_yaw_missing.sum()
            ),
            "feature_hash": self.feature_hash,
        }


@dataclass(frozen=True)
class CandidateSearchResult:
    candidate_id: str
    model_key: str
    status: str
    validation_auprc: Optional[float]
    validation_auroc: Optional[float]
    validation_log_loss: Optional[float]
    fit_seconds: Optional[float]
    predict_seconds: Optional[float]
    warning_messages: List[str]
    error_type: Optional[str]
    error_message: Optional[str]
    pipeline_spec: Dict[str, Any]
    complexity_order: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionOutput:
    probabilities: np.ndarray
    decision_scores: Optional[np.ndarray]
    decision_score_type: str
    predict_seconds: float


def _as_binary(values: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite values.")
    if not set(np.unique(numeric).tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"{name} must contain only 0/1.")
    return numeric.astype(np.int8)


def _as_integer(values: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite values.")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{name} must contain integer-valued entries.")
    integer = rounded.astype(np.int64)
    if np.any(integer < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return integer


def _validate_segment_order(
    segment_ids: np.ndarray,
    row_indices: np.ndarray,
    within_indices: np.ndarray,
) -> None:
    seen = set()
    previous: Optional[str] = None
    start = 0

    for position, segment in enumerate(segment_ids.astype(str)):
        if segment != previous:
            if segment in seen:
                raise ValueError(
                    f"Segment {segment!r} reappears at position {position}."
                )
            seen.add(segment)
            previous = segment

    while start < segment_ids.size:
        segment = segment_ids[start]
        end = start + 1
        while end < segment_ids.size and segment_ids[end] == segment:
            end += 1
        if end - start > 1:
            if np.any(np.diff(row_indices[start:end]) <= 0):
                raise ValueError(
                    f"row_index is not increasing inside segment {segment!r}."
                )
            if np.any(np.diff(within_indices[start:end]) <= 0):
                raise ValueError(
                    "within_segment_index is not increasing inside segment "
                    f"{segment!r}."
                )
        start = end

    keys = [
        (str(segment), int(row))
        for segment, row in zip(segment_ids, row_indices)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (segment_id, row_index) identities found.")


def load_gps_ids_feature_dataset(
    path: Path,
    expected_split: str,
) -> GPSIDSFeatureDataset:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    contract = build_gps_ids_feature_contract()
    frame = pd.read_csv(path, low_memory=False)
    expected_columns = [
        *GPS_IDS_OUTPUT_METADATA_COLUMNS,
        *contract.final_model_feature_names,
    ]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"Feature-file column contract mismatch for {path}.\n"
            f"Expected: {expected_columns}\n"
            f"Observed: {frame.columns.tolist()}"
        )
    if frame.empty:
        raise ValueError(f"Feature file is empty: {path}")

    split_values = frame["split"].astype(str).str.strip().unique().tolist()
    if split_values != [str(expected_split)]:
        raise ValueError(
            f"Expected split={expected_split!r} in {path}, "
            f"observed values={split_values}."
        )

    segment_ids = frame["segment_id"].astype(str).str.strip().to_numpy()
    if any(not value or value.lower() == "nan" for value in segment_ids):
        raise ValueError(f"Invalid segment IDs in {path}.")

    row_indices = _as_integer(frame["row_index"], "row_index")
    within_indices = _as_integer(
        frame["within_segment_index"],
        "within_segment_index",
    )
    _validate_segment_order(
        segment_ids,
        row_indices,
        within_indices,
    )

    delta_t = pd.to_numeric(
        frame["delta_t"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(delta_t)) or np.any(delta_t < 0.0):
        raise ValueError(f"delta_t must be finite and nonnegative in {path}.")

    labels = _as_binary(frame["label"], "label")
    valid_mask = _as_binary(frame["valid_mask"], "valid_mask").astype(bool)
    feature_complete = _as_binary(
        frame["feature_complete_mask"],
        "feature_complete_mask",
    ).astype(bool)
    target_yaw_missing = _as_binary(
        frame["target_yaw_missing"],
        "target_yaw_missing",
    ).astype(bool)

    X = frame.loc[
        :,
        list(contract.final_model_feature_names),
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if np.any(np.isinf(X)):
        raise ValueError(f"Feature matrix contains +/- infinity in {path}.")
    finite_by_row = np.all(np.isfinite(X), axis=1)
    if not np.array_equal(finite_by_row, feature_complete):
        mismatch = int(np.sum(finite_by_row != feature_complete))
        raise ValueError(
            f"feature_complete_mask disagrees with feature matrix on "
            f"{mismatch} rows in {path}."
        )

    target_index = list(contract.final_model_feature_names).index(
        "target_yaw_deg"
    )
    derived_target_missing = ~np.isfinite(X[:, target_index])
    if not np.array_equal(
        derived_target_missing,
        target_yaw_missing,
    ):
        mismatch = int(
            np.sum(derived_target_missing != target_yaw_missing)
        )
        raise ValueError(
            f"target_yaw_missing disagrees with target_yaw_deg on "
            f"{mismatch} rows in {path}."
        )

    return GPSIDSFeatureDataset(
        split_name=str(expected_split),
        source_path=str(path.resolve()),
        feature_names=tuple(contract.final_model_feature_names),
        feature_hash=contract.feature_hash,
        X=X,
        labels=labels,
        valid_mask=valid_mask,
        feature_complete_mask=feature_complete,
        target_yaw_missing=target_yaw_missing,
        segment_ids=segment_ids.astype(object),
        row_indices=row_indices,
        within_segment_indices=within_indices,
        delta_t=delta_t,
    )


def assert_common_feature_contract(
    datasets: Mapping[str, GPSIDSFeatureDataset],
) -> None:
    if not datasets:
        raise ValueError("No GPS-IDS datasets supplied.")
    first = next(iter(datasets.values()))
    for split_name, dataset in datasets.items():
        if dataset.feature_names != first.feature_names:
            raise ValueError(
                f"Feature-order mismatch for split {split_name}."
            )
        if dataset.feature_hash != first.feature_hash:
            raise ValueError(
                f"Feature-hash mismatch for split {split_name}."
            )


def _positive_probability(
    pipeline: Pipeline,
    X: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(
        pipeline.predict_proba(X),
        dtype=float,
    )
    model = pipeline.named_steps["model"]
    classes = np.asarray(model.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if positive_positions.size != 1:
        raise ValueError(
            f"Expected one positive class label 1, got classes={classes.tolist()}."
        )
    output = probabilities[:, int(positive_positions[0])]
    if not np.all(np.isfinite(output)):
        raise ValueError("Predicted probabilities contain non-finite values.")
    if np.any((output < -1.0e-12) | (output > 1.0 + 1.0e-12)):
        raise ValueError("Predicted probabilities are outside [0, 1].")
    return np.clip(output, 0.0, 1.0)


def predict_pipeline(
    pipeline: Pipeline,
    X: np.ndarray,
) -> PredictionOutput:
    start = time.perf_counter()
    probabilities = _positive_probability(pipeline, X)

    decision_scores: Optional[np.ndarray] = None
    decision_score_type = "probability_only"
    if hasattr(pipeline, "decision_function"):
        raw = np.asarray(pipeline.decision_function(X), dtype=float)
        if raw.ndim == 2:
            if raw.shape[1] != 1:
                raise ValueError(
                    "Binary decision_function returned unexpected shape "
                    f"{raw.shape}."
                )
            raw = raw[:, 0]
        raw = raw.reshape(-1)
        if raw.size != probabilities.size:
            raise ValueError("Decision-score length mismatch.")
        if not np.all(np.isfinite(raw)):
            raise ValueError("Decision scores contain non-finite values.")
        decision_scores = raw
        decision_score_type = "decision_function"

    elapsed = time.perf_counter() - start
    return PredictionOutput(
        probabilities=probabilities,
        decision_scores=decision_scores,
        decision_score_type=decision_score_type,
        predict_seconds=float(elapsed),
    )


def _safe_auroc(labels: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, probabilities))


def fit_and_score_candidate(
    *,
    candidate: CandidateSpec,
    train: GPSIDSFeatureDataset,
    validation: GPSIDSFeatureDataset,
    seed: int,
    n_jobs: int,
) -> Tuple[CandidateSearchResult, Optional[Pipeline]]:
    """
    Fit one candidate using train valid rows and score validation valid rows.

    Feature-incomplete rows remain eligible because missing values are handled
    by the train-fitted imputer.
    """
    if train.feature_names != validation.feature_names:
        raise ValueError("Train/validation feature contracts differ.")

    pipeline_spec: Dict[str, Any] = {
        "model_key": candidate.model_key,
        "imputer_strategy": candidate.imputer_strategy,
        "scaler_name": candidate.scaler_name,
        "model_parameters": candidate.to_dict()["model_parameters"],
        "active_seed": int(seed),
        "n_jobs": int(n_jobs),
    }
    warning_messages: List[str] = []
    try:
        pipeline, build_spec = build_pipeline(
            model_key=candidate.model_key,
            imputer_strategy=candidate.imputer_strategy,
            scaler_name=candidate.scaler_name,
            model_parameters=candidate.model_parameters,
            seed=seed,
            n_jobs=n_jobs,
        )
        pipeline_spec = build_spec.to_dict()

        train_mask = train.valid_mask
        if int(train_mask.sum()) == 0:
            raise ValueError("No valid Dataset-1 training rows.")

        fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipeline.fit(
                train.X[train_mask],
                train.labels[train_mask],
            )
        fit_seconds = time.perf_counter() - fit_start
        warning_messages = [
            f"{type(item.message).__name__}: {item.message}"
            for item in captured
        ]

        prediction = predict_pipeline(
            pipeline,
            validation.X,
        )
        valid = validation.valid_mask
        y_valid = validation.labels[valid]
        p_valid = prediction.probabilities[valid]
        if np.unique(y_valid).size < 2:
            raise ValueError(
                "Dataset-1 validation valid rows must contain both classes."
            )

        auprc = float(average_precision_score(y_valid, p_valid))
        auroc = _safe_auroc(y_valid, p_valid)
        validation_loss = float(
            log_loss(
                y_valid,
                np.column_stack([1.0 - p_valid, p_valid]),
                labels=[0, 1],
            )
        )

        result = CandidateSearchResult(
            candidate_id=candidate.candidate_id,
            model_key=candidate.model_key,
            status="PASSED",
            validation_auprc=auprc,
            validation_auroc=auroc,
            validation_log_loss=validation_loss,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(prediction.predict_seconds),
            warning_messages=warning_messages,
            error_type=None,
            error_message=None,
            pipeline_spec=pipeline_spec,
            complexity_order=int(candidate.complexity_order),
        )
        return result, pipeline

    except Exception as exc:
        result = CandidateSearchResult(
            candidate_id=candidate.candidate_id,
            model_key=candidate.model_key,
            status="FAILED",
            validation_auprc=None,
            validation_auroc=None,
            validation_log_loss=None,
            fit_seconds=None,
            predict_seconds=None,
            warning_messages=warning_messages,
            error_type=type(exc).__name__,
            error_message=str(exc),
            pipeline_spec=pipeline_spec,
            complexity_order=int(candidate.complexity_order),
        )
        return result, None


def candidate_selection_key(
    result: CandidateSearchResult,
) -> Tuple[float, float, float, float]:
    if result.status != "PASSED":
        return (
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
        )
    return (
        float(result.validation_auprc),
        (
            float(result.validation_auroc)
            if result.validation_auroc is not None
            else float("-inf")
        ),
        -float(result.validation_log_loss),
        -float(result.complexity_order),
    )


def select_best_candidate(
    results: Sequence[CandidateSearchResult],
) -> CandidateSearchResult:
    passed = [item for item in results if item.status == "PASSED"]
    if not passed:
        errors = [
            {
                "candidate_id": item.candidate_id,
                "error_type": item.error_type,
                "error_message": item.error_message,
            }
            for item in results
        ]
        raise RuntimeError(
            "Every hyperparameter candidate failed. "
            f"Failures: {errors}"
        )
    return max(passed, key=candidate_selection_key)


def build_standardized_bundle(
    *,
    dataset: GPSIDSFeatureDataset,
    model_name: str,
    checkpoint_path: Path,
    prediction: PredictionOutput,
) -> StandardizedPredictionBundle:
    return StandardizedPredictionBundle(
        split_name=dataset.split_name,
        model_name=str(model_name),
        probabilities=prediction.probabilities,
        decision_scores=prediction.decision_scores,
        decision_score_type=prediction.decision_score_type,
        labels=dataset.labels,
        valid_mask=dataset.valid_mask.astype(np.int8),
        segment_ids=dataset.segment_ids,
        row_indices=dataset.row_indices,
        within_segment_indices=dataset.within_segment_indices,
        delta_t=dataset.delta_t,
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        within_segment_index_source="provided",
    ).validated()


def save_prediction_and_evaluation(
    *,
    bundle: StandardizedPredictionBundle,
    theta: float,
    persistence: int,
    prediction_npz_path: Path,
    evaluation_json_path: Path,
) -> Tuple[
    SavedPredictionBundleArtifact,
    UnifiedEvaluationResult,
]:
    artifact = save_standardized_prediction_bundle(
        bundle=bundle,
        npz_path=prediction_npz_path,
    )
    evaluation = evaluate_prediction_bundle(
        bundle.to_unified_prediction_bundle(),
        theta=theta,
        persistence=persistence,
    )
    save_strict_json(
        evaluation.to_dict(include_row_arrays=False),
        evaluation_json_path,
    )
    return artifact, evaluation


def evaluation_row(
    *,
    model_key: str,
    model_name: str,
    reporting_role: str,
    selected_candidate_id: str,
    parameter_count: int,
    feature_hash: str,
    evaluation: UnifiedEvaluationResult,
) -> Dict[str, Any]:
    ranking = evaluation.ranking_metrics
    sample = evaluation.sample_metrics
    event = evaluation.event_metrics

    return {
        "model_key": model_key,
        "model_name": model_name,
        "reporting_role": reporting_role,
        "selected_candidate_id": selected_candidate_id,
        "split": evaluation.split_name,
        "theta": float(evaluation.theta),
        "persistence": int(evaluation.persistence),
        "auprc": ranking.auprc,
        "auroc": ranking.auroc,
        "accuracy": sample.accuracy,
        "precision": sample.precision,
        "recall": sample.recall,
        "f1": sample.f1,
        "fpr": sample.fpr,
        "specificity": sample.specificity,
        "tp": sample.tp,
        "fp": sample.fp,
        "tn": sample.tn,
        "fn": sample.fn,
        "attack_detection_rate": event.attack_detection_rate,
        "attack_events_total": event.attack_events_total,
        "attack_events_detected": event.attack_events_detected,
        "attack_events_missed": event.attack_events_missed,
        "mean_detection_delay_seconds": (
            event.mean_detection_delay_seconds
        ),
        "median_detection_delay_seconds": (
            event.median_detection_delay_seconds
        ),
        "max_detection_delay_seconds": (
            event.max_detection_delay_seconds
        ),
        "false_alarm_rows": event.false_alarm_rows,
        "false_alarm_events": event.false_alarm_events,
        "parameter_count": int(parameter_count),
        "gps_ids_contract_feature_hash": feature_hash,
    }


__all__ = [
    "CandidateSearchResult",
    "GPSIDSFeatureDataset",
    "PredictionOutput",
    "assert_common_feature_contract",
    "build_standardized_bundle",
    "candidate_selection_key",
    "evaluation_row",
    "fit_and_score_candidate",
    "load_gps_ids_feature_dataset",
    "predict_pipeline",
    "save_prediction_and_evaluation",
    "select_best_candidate",
]
