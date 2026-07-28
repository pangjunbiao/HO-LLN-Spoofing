"""
Validation-only threshold and persistence selection.

Step 10 purpose:
- choose theta and N_p using validation predictions only,
- evaluate all candidate threshold/persistence pairs with the same locked metrics,
- select the best validation operating point,
- save a transparent selection report,
- reuse the selected theta/N_p later for test, external, and online evaluation.

Critical rule:
Never select theta or N_p on train, test, Dataset-2 external, or Dataset-3 online.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import math

import numpy as np
import pandas as pd

from src.evaluation.alarm_rules import evaluate_alarm_rule, strip_large_frames
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_json


DEFAULT_THRESHOLD_GRID = [
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

DEFAULT_PERSISTENCE_GRID = [1, 2, 3, 4, 5]


@dataclass
class ThresholdSelectionConfig:
    """Configuration for validation-only threshold selection."""

    enabled: bool

    selection_split: str
    forbid_train_selection: bool
    forbid_test_selection: bool
    forbid_external_selection: bool
    forbid_online_selection: bool

    threshold_grid: List[float]
    persistence_grid: List[int]

    objective: str
    max_fpr_constraint: Optional[float]
    min_recall_constraint: Optional[float]
    min_attack_detection_rate_constraint: Optional[float]

    tie_breakers: List[str]

    save_candidate_table: bool
    save_selected_payload: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdCandidateResult:
    """One candidate threshold/persistence result on validation."""

    threshold: float
    persistence: int

    auprc: Optional[float]
    auroc: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    fpr: Optional[float]

    attack_detection_rate: Optional[float]
    mean_detection_delay_seconds: Optional[float]
    mean_detection_delay_rows: Optional[float]
    normal_segment_far: Optional[float]

    tp: int
    fp: int
    tn: int
    fn: int
    support: int

    feasible: bool
    feasibility_reason: str
    objective_value: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdSelectionResult:
    """Final validation-only threshold selection result."""

    selected_threshold: float
    selected_persistence: int
    selection_split: str
    objective: str

    selected_candidate: Dict[str, Any]
    candidate_count: int
    feasible_candidate_count: int

    threshold_grid: List[float]
    persistence_grid: List[int]

    validation_only_rule: Dict[str, Any]
    selection_config: Dict[str, Any]

    candidate_table: List[Dict[str, Any]]
    selected_evaluation_payload: Dict[str, Any]

    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to finite float or None."""
    try:
        value_float = float(value)
    except Exception:
        return None

    if not math.isfinite(value_float):
        return None

    return value_float


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int with fallback."""
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_list_float(values: Sequence[Any]) -> List[float]:
    """Convert sequence to clean sorted unique float list."""
    cleaned: List[float] = []

    for item in values:
        item_float = _safe_float(item)
        if item_float is None:
            continue
        item_float = float(np.clip(item_float, 0.0, 1.0))
        cleaned.append(item_float)

    unique = sorted(set(cleaned))

    if not unique:
        raise ValueError("Threshold grid is empty after cleaning.")

    return unique


def _as_list_int(values: Sequence[Any]) -> List[int]:
    """Convert sequence to clean sorted unique positive int list."""
    cleaned: List[int] = []

    for item in values:
        item_int = _safe_int(item, default=1)
        item_int = max(item_int, 1)
        cleaned.append(item_int)

    unique = sorted(set(cleaned))

    if not unique:
        raise ValueError("Persistence grid is empty after cleaning.")

    return unique


def _linspace_thresholds(start: float, stop: float, num: int) -> List[float]:
    """Build threshold grid from linspace settings."""
    num = max(int(num), 2)
    start = float(np.clip(start, 0.0, 1.0))
    stop = float(np.clip(stop, 0.0, 1.0))

    values = np.linspace(start, stop, num=num)
    return [round(float(value), 10) for value in values]


def get_threshold_selection_config(config: Mapping[str, Any]) -> ThresholdSelectionConfig:
    """
    Read threshold-selection configuration.

    Looks first under training.threshold_selection, then experiments.threshold_selection.
    """
    base_path = "training.threshold_selection"

    enabled = bool(get_by_path(config, f"{base_path}.enabled", True))

    threshold_grid_value = get_by_path(
        config,
        f"{base_path}.threshold_grid",
        None,
    )

    if threshold_grid_value is None:
        use_linspace = bool(
            get_by_path(config, f"{base_path}.use_threshold_linspace", False)
        )

        if use_linspace:
            threshold_grid = _linspace_thresholds(
                start=float(get_by_path(config, f"{base_path}.threshold_min", 0.05)),
                stop=float(get_by_path(config, f"{base_path}.threshold_max", 0.95)),
                num=int(get_by_path(config, f"{base_path}.threshold_num", 19)),
            )
        else:
            threshold_grid = list(DEFAULT_THRESHOLD_GRID)
    else:
        threshold_grid = _as_list_float(threshold_grid_value)

    persistence_grid_value = get_by_path(
        config,
        f"{base_path}.persistence_grid",
        DEFAULT_PERSISTENCE_GRID,
    )
    persistence_grid = _as_list_int(persistence_grid_value)

    tie_breakers = list(
        get_by_path(
            config,
            f"{base_path}.tie_breakers",
            [
                "higher_attack_detection_rate",
                "lower_detection_delay",
                "lower_fpr",
                "higher_recall",
                "lower_persistence",
                "higher_threshold",
            ],
        )
    )

    return ThresholdSelectionConfig(
        enabled=enabled,
        selection_split=str(
            get_by_path(config, f"{base_path}.selection_split", "val")
        ),
        forbid_train_selection=bool(
            get_by_path(config, f"{base_path}.forbid_train_selection", True)
        ),
        forbid_test_selection=bool(
            get_by_path(config, f"{base_path}.forbid_test_selection", True)
        ),
        forbid_external_selection=bool(
            get_by_path(config, f"{base_path}.forbid_external_selection", True)
        ),
        forbid_online_selection=bool(
            get_by_path(config, f"{base_path}.forbid_online_selection", True)
        ),
        threshold_grid=threshold_grid,
        persistence_grid=persistence_grid,
        objective=str(
            get_by_path(config, f"{base_path}.objective", "maximize_f1")
        ),
        max_fpr_constraint=_safe_float(
            get_by_path(config, f"{base_path}.max_fpr_constraint", None)
        ),
        min_recall_constraint=_safe_float(
            get_by_path(config, f"{base_path}.min_recall_constraint", None)
        ),
        min_attack_detection_rate_constraint=_safe_float(
            get_by_path(
                config,
                f"{base_path}.min_attack_detection_rate_constraint",
                None,
            )
        ),
        tie_breakers=[str(item) for item in tie_breakers],
        save_candidate_table=bool(
            get_by_path(config, f"{base_path}.save_candidate_table", True)
        ),
        save_selected_payload=bool(
            get_by_path(config, f"{base_path}.save_selected_payload", True)
        ),
    )


def validate_selection_split(
    split_name: str,
    cfg: ThresholdSelectionConfig,
) -> Dict[str, Any]:
    """
    Enforce validation-only threshold selection.

    Allowed selection split names:
    - val
    - validation
    """
    split = str(split_name).lower().strip()

    is_validation = split in {"val", "validation"}

    violations: List[str] = []

    if not is_validation:
        violations.append(
            f"Threshold selection must use validation only, got split='{split_name}'."
        )

    if cfg.forbid_train_selection and split in {"train", "training"}:
        violations.append("Train split cannot be used for threshold selection.")

    if cfg.forbid_test_selection and split in {"test", "internal_test"}:
        violations.append("Internal test split cannot be used for threshold selection.")

    if cfg.forbid_external_selection and split in {"external", "dataset2"}:
        violations.append("Dataset-2 external split cannot be used for threshold selection.")

    if cfg.forbid_online_selection and split in {"online", "dataset3"}:
        violations.append("Dataset-3 online split cannot be used for threshold selection.")

    passed = len(violations) == 0

    return {
        "passed": passed,
        "selection_split": split_name,
        "is_validation_split": is_validation,
        "violations": violations,
        "forbid_train_selection": cfg.forbid_train_selection,
        "forbid_test_selection": cfg.forbid_test_selection,
        "forbid_external_selection": cfg.forbid_external_selection,
        "forbid_online_selection": cfg.forbid_online_selection,
    }


def _get_nested(payload: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    """Get nested dict value."""
    current: Any = payload

    for key in path:
        if not isinstance(current, Mapping):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def _candidate_from_payload(
    threshold: float,
    persistence: int,
    payload: Mapping[str, Any],
    cfg: ThresholdSelectionConfig,
) -> ThresholdCandidateResult:
    """Extract compact candidate row from evaluation payload."""
    ranking = _get_nested(payload, ["time_step_metrics", "ranking"], {})
    threshold_metrics = _get_nested(payload, ["time_step_metrics", "threshold_metrics"], {})
    event_metrics = _get_nested(payload, ["event_metrics"], {})

    auprc = _safe_float(ranking.get("auprc"))
    auroc = _safe_float(ranking.get("auroc"))

    f1 = _safe_float(threshold_metrics.get("f1"))
    precision = _safe_float(threshold_metrics.get("precision"))
    recall = _safe_float(threshold_metrics.get("recall"))
    fpr = _safe_float(threshold_metrics.get("fpr"))

    attack_detection_rate = _safe_float(event_metrics.get("attack_detection_rate"))
    mean_delay_seconds = _safe_float(event_metrics.get("mean_detection_delay_seconds"))
    mean_delay_rows = _safe_float(event_metrics.get("mean_detection_delay_rows"))
    normal_segment_far = _safe_float(event_metrics.get("normal_segment_far"))

    tp = _safe_int(threshold_metrics.get("tp"), 0)
    fp = _safe_int(threshold_metrics.get("fp"), 0)
    tn = _safe_int(threshold_metrics.get("tn"), 0)
    fn = _safe_int(threshold_metrics.get("fn"), 0)
    support = _safe_int(threshold_metrics.get("support"), 0)

    feasible, reason = _check_candidate_feasibility(
        fpr=fpr,
        recall=recall,
        attack_detection_rate=attack_detection_rate,
        cfg=cfg,
    )

    objective_value = _candidate_objective_value(
        objective=cfg.objective,
        auprc=auprc,
        auroc=auroc,
        f1=f1,
        precision=precision,
        recall=recall,
        fpr=fpr,
        attack_detection_rate=attack_detection_rate,
        mean_detection_delay_seconds=mean_delay_seconds,
        normal_segment_far=normal_segment_far,
    )

    if not feasible:
        objective_value = None

    return ThresholdCandidateResult(
        threshold=float(threshold),
        persistence=int(persistence),
        auprc=auprc,
        auroc=auroc,
        f1=f1,
        precision=precision,
        recall=recall,
        fpr=fpr,
        attack_detection_rate=attack_detection_rate,
        mean_detection_delay_seconds=mean_delay_seconds,
        mean_detection_delay_rows=mean_delay_rows,
        normal_segment_far=normal_segment_far,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        support=support,
        feasible=bool(feasible),
        feasibility_reason=str(reason),
        objective_value=objective_value,
    )


def _check_candidate_feasibility(
    fpr: Optional[float],
    recall: Optional[float],
    attack_detection_rate: Optional[float],
    cfg: ThresholdSelectionConfig,
) -> Tuple[bool, str]:
    """Check optional validation constraints."""
    reasons: List[str] = []

    if cfg.max_fpr_constraint is not None:
        if fpr is None or fpr > float(cfg.max_fpr_constraint):
            reasons.append(
                f"fpr>{cfg.max_fpr_constraint}"
            )

    if cfg.min_recall_constraint is not None:
        if recall is None or recall < float(cfg.min_recall_constraint):
            reasons.append(
                f"recall<{cfg.min_recall_constraint}"
            )

    if cfg.min_attack_detection_rate_constraint is not None:
        if (
            attack_detection_rate is None
            or attack_detection_rate < float(cfg.min_attack_detection_rate_constraint)
        ):
            reasons.append(
                f"attack_detection_rate<{cfg.min_attack_detection_rate_constraint}"
            )

    if reasons:
        return False, "; ".join(reasons)

    return True, "passed_constraints"


def _candidate_objective_value(
    objective: str,
    auprc: Optional[float],
    auroc: Optional[float],
    f1: Optional[float],
    precision: Optional[float],
    recall: Optional[float],
    fpr: Optional[float],
    attack_detection_rate: Optional[float],
    mean_detection_delay_seconds: Optional[float],
    normal_segment_far: Optional[float],
) -> Optional[float]:
    """
    Compute scalar objective.

    Default is maximize_f1 because threshold/persistence directly control F1/FPR/delay.
    AUPRC/AUROC are threshold-free and mainly used for reporting.
    """
    objective = str(objective).lower().strip()

    values = {
        "maximize_f1": f1,
        "maximize_recall": recall,
        "maximize_precision": precision,
        "maximize_auprc": auprc,
        "maximize_auroc": auroc,
        "maximize_attack_detection_rate": attack_detection_rate,
        "minimize_fpr": None if fpr is None else -float(fpr),
        "minimize_normal_segment_far": None
        if normal_segment_far is None
        else -float(normal_segment_far),
        "minimize_detection_delay": None
        if mean_detection_delay_seconds is None
        else -float(mean_detection_delay_seconds),
    }

    if objective == "balanced_primary":
        # Balanced objective:
        # high F1, high event detection, low FPR, low normal-segment FAR.
        if f1 is None or attack_detection_rate is None or fpr is None:
            return None

        far_penalty = 0.0 if normal_segment_far is None else normal_segment_far
        delay_penalty = (
            0.0
            if mean_detection_delay_seconds is None
            else min(mean_detection_delay_seconds / 60.0, 1.0)
        )

        return float(
            0.45 * f1
            + 0.35 * attack_detection_rate
            - 0.15 * fpr
            - 0.03 * far_penalty
            - 0.02 * delay_penalty
        )

    if objective not in values:
        raise ValueError(
            f"Unknown threshold-selection objective '{objective}'. "
            "Supported: maximize_f1, maximize_recall, maximize_precision, "
            "maximize_auprc, maximize_auroc, maximize_attack_detection_rate, "
            "minimize_fpr, minimize_normal_segment_far, minimize_detection_delay, "
            "balanced_primary."
        )

    value = values[objective]

    if value is None:
        return None

    return float(value)


def _tie_breaker_value(
    candidate: ThresholdCandidateResult,
    tie_breaker: str,
) -> float:
    """Return sorting value for one tie-breaker. Higher is better."""
    tie_breaker = str(tie_breaker).lower().strip()

    large_missing_delay = 1.0e12

    mapping = {
        "higher_attack_detection_rate": candidate.attack_detection_rate,
        "lower_detection_delay": -(
            large_missing_delay
            if candidate.mean_detection_delay_seconds is None
            else candidate.mean_detection_delay_seconds
        ),
        "lower_detection_delay_rows": -(
            large_missing_delay
            if candidate.mean_detection_delay_rows is None
            else candidate.mean_detection_delay_rows
        ),
        "lower_fpr": -(1.0 if candidate.fpr is None else candidate.fpr),
        "lower_normal_segment_far": -(
            1.0
            if candidate.normal_segment_far is None
            else candidate.normal_segment_far
        ),
        "higher_recall": candidate.recall,
        "higher_precision": candidate.precision,
        "higher_f1": candidate.f1,
        "higher_auprc": candidate.auprc,
        "higher_auroc": candidate.auroc,
        "lower_persistence": -float(candidate.persistence),
        "higher_threshold": float(candidate.threshold),
        "lower_threshold": -float(candidate.threshold),
    }

    value = mapping.get(tie_breaker)

    if value is None:
        return -1.0e12

    value = _safe_float(value)

    if value is None:
        return -1.0e12

    return float(value)


def rank_candidates(
    candidates: Sequence[ThresholdCandidateResult],
    tie_breakers: Sequence[str],
) -> List[ThresholdCandidateResult]:
    """
    Rank candidates.

    Feasible candidates are ranked first. Higher objective is better.
    Tie-breakers are applied in order.
    """
    def sort_key(candidate: ThresholdCandidateResult) -> Tuple[Any, ...]:
        feasible_score = 1 if candidate.feasible else 0

        objective = (
            -1.0e12
            if candidate.objective_value is None
            else float(candidate.objective_value)
        )

        tie_values = tuple(
            _tie_breaker_value(candidate, tie)
            for tie in tie_breakers
        )

        return (feasible_score, objective) + tie_values

    return sorted(candidates, key=sort_key, reverse=True)


def _validate_prediction_inputs(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    segment_id: Sequence[Any],
    order_index: Optional[Sequence[Any]],
    delta_t: Optional[Sequence[Any]],
    valid_mask: Optional[Sequence[Any]],
) -> None:
    """Validate prediction arrays have compatible lengths."""
    n = len(np.asarray(y_true).reshape(-1))

    names_and_values = {
        "y_score": y_score,
        "segment_id": segment_id,
    }

    if order_index is not None:
        names_and_values["order_index"] = order_index

    if delta_t is not None:
        names_and_values["delta_t"] = delta_t

    if valid_mask is not None:
        names_and_values["valid_mask"] = valid_mask

    for name, values in names_and_values.items():
        length = len(np.asarray(values).reshape(-1))
        if length != n:
            raise ValueError(
                f"Length mismatch. y_true has {n} rows, {name} has {length} rows."
            )


def select_threshold_on_validation(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    segment_id: Sequence[Any],
    order_index: Optional[Sequence[Any]] = None,
    delta_t: Optional[Sequence[Any]] = None,
    valid_mask: Optional[Sequence[Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    selection_split: str = "val",
    threshold_grid: Optional[Sequence[float]] = None,
    persistence_grid: Optional[Sequence[int]] = None,
    runtime_seconds: Optional[float] = None,
) -> ThresholdSelectionResult:
    """
    Select threshold and persistence on validation predictions only.

    Parameters must come from validation split predictions.
    This function raises an error if selection_split is not validation.
    """
    cfg = (
        get_threshold_selection_config(config)
        if config is not None
        else ThresholdSelectionConfig(
            enabled=True,
            selection_split="val",
            forbid_train_selection=True,
            forbid_test_selection=True,
            forbid_external_selection=True,
            forbid_online_selection=True,
            threshold_grid=list(DEFAULT_THRESHOLD_GRID),
            persistence_grid=list(DEFAULT_PERSISTENCE_GRID),
            objective="maximize_f1",
            max_fpr_constraint=None,
            min_recall_constraint=None,
            min_attack_detection_rate_constraint=None,
            tie_breakers=[
                "higher_attack_detection_rate",
                "lower_detection_delay",
                "lower_fpr",
                "higher_recall",
                "lower_persistence",
                "higher_threshold",
            ],
            save_candidate_table=True,
            save_selected_payload=True,
        )
    )

    if threshold_grid is not None:
        cfg.threshold_grid = _as_list_float(threshold_grid)

    if persistence_grid is not None:
        cfg.persistence_grid = _as_list_int(persistence_grid)

    # Explicit user argument wins, but config still records expected split.
    split_to_validate = str(selection_split or cfg.selection_split)

    validation_rule = validate_selection_split(
        split_name=split_to_validate,
        cfg=cfg,
    )

    if not validation_rule["passed"]:
        raise RuntimeError(
            "Invalid threshold selection split. "
            f"Validation-only rule failed: {validation_rule}"
        )

    _validate_prediction_inputs(
        y_true=y_true,
        y_score=y_score,
        segment_id=segment_id,
        order_index=order_index,
        delta_t=delta_t,
        valid_mask=valid_mask,
    )

    candidates: List[ThresholdCandidateResult] = []
    payloads_by_key: Dict[str, Dict[str, Any]] = {}

    for threshold in cfg.threshold_grid:
        for persistence in cfg.persistence_grid:
            payload = evaluate_alarm_rule(
                y_true=y_true,
                y_score=y_score,
                segment_id=segment_id,
                order_index=order_index,
                delta_t=delta_t,
                valid_mask=valid_mask,
                threshold=float(threshold),
                persistence=int(persistence),
                runtime_seconds=runtime_seconds,
            )

            payload_small = strip_large_frames(payload)

            candidate = _candidate_from_payload(
                threshold=float(threshold),
                persistence=int(persistence),
                payload=payload_small,
                cfg=cfg,
            )

            key = f"theta={float(threshold):.10f}|Np={int(persistence)}"
            candidates.append(candidate)
            payloads_by_key[key] = payload_small

    if len(candidates) == 0:
        raise RuntimeError("No threshold candidates were evaluated.")

    ranked = rank_candidates(
        candidates=candidates,
        tie_breakers=cfg.tie_breakers,
    )

    best = ranked[0]

    if not best.feasible or best.objective_value is None:
        raise RuntimeError(
            "No feasible threshold candidate found. "
            "Relax validation constraints or inspect validation predictions."
        )

    selected_key = f"theta={float(best.threshold):.10f}|Np={int(best.persistence)}"
    selected_payload = payloads_by_key[selected_key]

    candidate_table = [candidate.to_dict() for candidate in ranked]

    feasible_count = int(sum(candidate.feasible for candidate in candidates))

    result = ThresholdSelectionResult(
        selected_threshold=float(best.threshold),
        selected_persistence=int(best.persistence),
        selection_split=str(split_to_validate),
        objective=str(cfg.objective),
        selected_candidate=best.to_dict(),
        candidate_count=int(len(candidates)),
        feasible_candidate_count=feasible_count,
        threshold_grid=list(cfg.threshold_grid),
        persistence_grid=list(cfg.persistence_grid),
        validation_only_rule=validation_rule,
        selection_config=cfg.to_dict(),
        candidate_table=candidate_table,
        selected_evaluation_payload=selected_payload,
        final_status="PASSED",
    )

    return result


def evaluate_with_selected_threshold(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    segment_id: Sequence[Any],
    selected_threshold: float,
    selected_persistence: int,
    order_index: Optional[Sequence[Any]] = None,
    delta_t: Optional[Sequence[Any]] = None,
    valid_mask: Optional[Sequence[Any]] = None,
    runtime_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate any split using already-selected threshold/persistence.

    Use this for train diagnostics, internal test, Dataset-2 external, and Dataset-3 online.
    Do not use it to select new threshold values.
    """
    payload = evaluate_alarm_rule(
        y_true=y_true,
        y_score=y_score,
        segment_id=segment_id,
        order_index=order_index,
        delta_t=delta_t,
        valid_mask=valid_mask,
        threshold=float(selected_threshold),
        persistence=int(selected_persistence),
        runtime_seconds=runtime_seconds,
    )

    return strip_large_frames(payload)


def get_threshold_selection_output_dir(config: Mapping[str, Any]) -> Path:
    """Resolve threshold-selection output directory."""
    value = get_by_path(
        config,
        "paths.threshold_selection_dir",
        "results/tables/threshold_selection",
    )
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_threshold_selection_report_path(
    config: Mapping[str, Any],
    model_name: str,
) -> Path:
    """Resolve threshold-selection report path."""
    output_dir = get_threshold_selection_output_dir(config)
    safe_name = str(model_name).replace(" ", "_").replace("/", "_")
    return output_dir / f"{safe_name}_threshold_selection.json"


def save_threshold_selection_result(
    result: ThresholdSelectionResult,
    config: Mapping[str, Any],
    model_name: str,
) -> Path:
    """Save threshold-selection result JSON."""
    path = get_threshold_selection_report_path(
        config=config,
        model_name=model_name,
    )
    save_json(result.to_dict(), path, indent=2)
    return path


def threshold_selection_result_from_json(path: str | Path) -> ThresholdSelectionResult:
    """
    Load a saved threshold-selection JSON.

    This helper is intentionally strict enough for later evaluation scripts.
    """
    import json

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Threshold selection JSON not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return ThresholdSelectionResult(
        selected_threshold=float(data["selected_threshold"]),
        selected_persistence=int(data["selected_persistence"]),
        selection_split=str(data["selection_split"]),
        objective=str(data["objective"]),
        selected_candidate=dict(data["selected_candidate"]),
        candidate_count=int(data["candidate_count"]),
        feasible_candidate_count=int(data["feasible_candidate_count"]),
        threshold_grid=[float(x) for x in data["threshold_grid"]],
        persistence_grid=[int(x) for x in data["persistence_grid"]],
        validation_only_rule=dict(data["validation_only_rule"]),
        selection_config=dict(data["selection_config"]),
        candidate_table=list(data["candidate_table"]),
        selected_evaluation_payload=dict(data["selected_evaluation_payload"]),
        final_status=str(data["final_status"]),
    )


def print_threshold_selection_summary(result: ThresholdSelectionResult) -> None:
    """Print concise threshold-selection summary."""
    selected = result.selected_candidate

    print("=" * 100)
    print("STEP 10 THRESHOLD SELECTION SUMMARY")
    print("=" * 100)
    print(f"Selection split                     : {result.selection_split}")
    print(f"Validation-only rule passed          : {result.validation_only_rule.get('passed')}")
    print(f"Objective                            : {result.objective}")
    print(f"Candidates evaluated                 : {result.candidate_count}")
    print(f"Feasible candidates                  : {result.feasible_candidate_count}")
    print(f"Selected theta                       : {result.selected_threshold}")
    print(f"Selected persistence N_p             : {result.selected_persistence}")
    print(f"Selected AUPRC                       : {selected.get('auprc')}")
    print(f"Selected AUROC                       : {selected.get('auroc')}")
    print(f"Selected F1                          : {selected.get('f1')}")
    print(f"Selected Precision                   : {selected.get('precision')}")
    print(f"Selected Recall                      : {selected.get('recall')}")
    print(f"Selected FPR                         : {selected.get('fpr')}")
    print(f"Selected Attack Detection Rate       : {selected.get('attack_detection_rate')}")
    print(f"Selected Mean Detection Delay seconds: {selected.get('mean_detection_delay_seconds')}")
    print(f"Selected Normal-Segment FAR          : {selected.get('normal_segment_far')}")
    print(f"Final status                         : {result.final_status}")
    print("=" * 100)


def candidate_table_to_dataframe(result: ThresholdSelectionResult) -> pd.DataFrame:
    """Convert candidate table to pandas DataFrame."""
    return pd.DataFrame(result.candidate_table)


def select_threshold_from_validation_dataframe(
    df: pd.DataFrame,
    score_column: str,
    config: Optional[Mapping[str, Any]] = None,
    selection_split: str = "val",
    label_column: str = "Data Type",
    segment_column: str = "segment_id",
    order_column: str = "within_segment_index",
    delta_t_column: str = "delta_t_seconds",
    valid_column: str = "xi_nu",
    threshold_grid: Optional[Sequence[float]] = None,
    persistence_grid: Optional[Sequence[int]] = None,
    runtime_seconds: Optional[float] = None,
) -> ThresholdSelectionResult:
    """
    Convenience wrapper for validation DataFrame predictions.

    The dataframe must contain labels, scores, segment IDs, order indices,
    delta_t, and valid/loss mask.
    """
    required = [
        score_column,
        label_column,
        segment_column,
        order_column,
        delta_t_column,
        valid_column,
    ]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(
            f"Validation dataframe is missing required threshold-selection columns: {missing}"
        )

    return select_threshold_on_validation(
        y_true=df[label_column].to_numpy(),
        y_score=df[score_column].to_numpy(),
        segment_id=df[segment_column].to_numpy(),
        order_index=df[order_column].to_numpy(),
        delta_t=df[delta_t_column].to_numpy(),
        valid_mask=df[valid_column].to_numpy(),
        config=config,
        selection_split=selection_split,
        threshold_grid=threshold_grid,
        persistence_grid=persistence_grid,
        runtime_seconds=runtime_seconds,
    )