"""
Run full proposed-model experiment for Step 13.

Step 13 responsibilities:
- train/reuse the full proposed model checkpoint,
- select real theta and N_p on Dataset-1 validation only,
- evaluate Dataset-1 internal test,
- evaluate Dataset-2 external/source-shift test,
- evaluate Dataset-3 online case study,
- save all comparison tables and summary artifacts.

Important:
- This is the first real full proposed-model experiment.
- The synthetic Step-10 theta=0.55 is never used here.
- Dataset-2 and Dataset-3 are never used for tuning or threshold selection.
- Baselines and ablations are not compared here yet; they come later.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.evaluation.evaluate_dataset1 import (
    DatasetEvaluationResult,
    run_dataset1_evaluation,
)
from src.evaluation.evaluate_dataset2 import run_dataset2_external_evaluation
from src.evaluation.evaluate_dataset3 import run_dataset3_online_evaluation
from src.evaluation.result_tables import (
    extract_primary_metrics,
    print_primary_metric_table,
)
from src.training.trainer import (
    proposed_best_checkpoint_exists,
    proposed_best_checkpoint_path,
    run_step12_training_protocol,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir


@dataclass
class ProposedExperimentConfig:
    """Configuration for Step-13 full proposed-model experiment."""

    model_name: str = "Proposed"

    retrain_policy: str = "reuse_if_exists"
    # Supported:
    # - "reuse_if_exists": train only when proposed_best.pt does not exist
    # - "always": always retrain full proposed model first
    # - "never": never train; fail if proposed_best.pt is missing

    evaluate_dataset1: bool = True
    evaluate_dataset2: bool = True
    evaluate_dataset3: bool = True

    checkpoint_path: Optional[str] = None

    summary_json: str = "results/tables/step13_proposed_experiment_summary.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProposedExperimentSummary:
    """Final Step-13 proposed experiment summary."""

    final_status: str
    active_seed: int
    model_name: str

    retrain_policy: str
    checkpoint_path: Optional[str]

    trained_in_step13: bool
    training_summary: Optional[Dict[str, Any]]

    dataset1_result: Optional[Dict[str, Any]]
    dataset2_result: Optional[Dict[str, Any]]
    dataset3_result: Optional[Dict[str, Any]]

    selected_threshold: Optional[float]
    selected_persistence: Optional[int]

    output_paths: Dict[str, str]
    runtime_seconds: float

    leakage_rules: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy/Path objects to JSON-safe values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    return value


def _save_json_safe(payload: Mapping[str, Any], output_path: Path) -> None:
    """Save JSON safely."""
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(dict(payload)), file, indent=2)


def _project_path(config: Mapping[str, Any], path_value: str) -> Path:
    """Resolve a project-relative path."""
    return resolve_project_path(config, path_value)


def _set_by_path(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set nested dictionary value by dotted path."""
    current: Dict[str, Any] = config

    keys = path.split(".")

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def build_proposed_experiment_config(config: Mapping[str, Any]) -> ProposedExperimentConfig:
    """Build Step-13 proposed experiment config."""
    return ProposedExperimentConfig(
        model_name=str(
            get_by_path(
                config,
                "experiments.proposed.model_name",
                get_by_path(config, "experiments.step13.model_name", "Proposed"),
            )
        ),
        retrain_policy=str(
            get_by_path(
                config,
                "experiments.proposed.retrain_policy",
                get_by_path(config, "experiments.step13.retrain_policy", "reuse_if_exists"),
            )
        ),
        evaluate_dataset1=bool(
            get_by_path(
                config,
                "experiments.proposed.evaluate_dataset1",
                get_by_path(config, "experiments.step13.evaluate_dataset1", True),
            )
        ),
        evaluate_dataset2=bool(
            get_by_path(
                config,
                "experiments.proposed.evaluate_dataset2",
                get_by_path(config, "experiments.step13.evaluate_dataset2", True),
            )
        ),
        evaluate_dataset3=bool(
            get_by_path(
                config,
                "experiments.proposed.evaluate_dataset3",
                get_by_path(config, "experiments.step13.evaluate_dataset3", True),
            )
        ),
        checkpoint_path=get_by_path(
            config,
            "experiments.proposed.checkpoint_path",
            get_by_path(config, "experiments.step13.checkpoint_path", None),
        ),
        summary_json=str(
            get_by_path(
                config,
                "paths.step13_proposed_experiment_summary_json",
                "results/tables/step13_proposed_experiment_summary.json",
            )
        ),
    )


def make_full_proposed_training_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return a config copy forced to train the full proposed model.

    This protects Step 13 from accidentally training an ablation if the user
    previously changed training.step12.variant_name.
    """
    cfg = copy.deepcopy(dict(config))

    _set_by_path(cfg, "training.step12.model_name", "proposed")
    _set_by_path(cfg, "training.step12.variant_name", "full")

    # Keep checkpoint names standard for first proposed experiment.
    _set_by_path(cfg, "training.checkpointing.best_checkpoint_name", "proposed_best.pt")
    _set_by_path(cfg, "training.checkpointing.last_checkpoint_name", "proposed_last.pt")

    return cfg


def resolve_step13_checkpoint_path(
    config: Mapping[str, Any],
    experiment_config: ProposedExperimentConfig,
) -> Path:
    """Resolve checkpoint path for Step 13."""
    if experiment_config.checkpoint_path is not None:
        return _project_path(config, str(experiment_config.checkpoint_path))

    return proposed_best_checkpoint_path(config)


def maybe_train_full_proposed_model(
    config: Mapping[str, Any],
    experiment_config: ProposedExperimentConfig,
    active_seed: int,
) -> tuple[bool, Optional[Dict[str, Any]], Path]:
    """
    Train full proposed model if required by retrain_policy.

    Returns:
        trained_now, training_summary_dict, checkpoint_path
    """
    retrain_policy = str(experiment_config.retrain_policy).lower().strip()

    if retrain_policy not in {"reuse_if_exists", "always", "never"}:
        raise ValueError(
            "Invalid experiments.proposed.retrain_policy. "
            "Expected one of: reuse_if_exists, always, never."
        )

    checkpoint_path = resolve_step13_checkpoint_path(
        config=config,
        experiment_config=experiment_config,
    )

    checkpoint_exists = checkpoint_path.exists()

    if retrain_policy == "never":
        if not checkpoint_exists:
            raise FileNotFoundError(
                f"retrain_policy='never', but checkpoint does not exist: {checkpoint_path}"
            )

        return False, None, checkpoint_path

    if retrain_policy == "reuse_if_exists" and checkpoint_exists:
        return False, None, checkpoint_path

    # retrain_policy == "always" OR checkpoint missing under reuse_if_exists.
    training_config = make_full_proposed_training_config(config)

    print("=" * 100)
    print("STEP 13 TRAINING DECISION")
    print("=" * 100)
    print(f"Retrain policy       : {retrain_policy}")
    print(f"Checkpoint exists    : {checkpoint_exists}")
    print(f"Training full model  : True")
    print(f"Checkpoint target    : {checkpoint_path}")
    print("=" * 100)

    training_summary = run_step12_training_protocol(
        config=training_config,
        active_seed=active_seed,
    )

    checkpoint_path_after = proposed_best_checkpoint_path(training_config)

    if not checkpoint_path_after.exists():
        raise RuntimeError(
            f"Training finished but best checkpoint was not found: {checkpoint_path_after}"
        )

    return True, training_summary.to_dict(), checkpoint_path_after


def _result_or_none(result: Optional[DatasetEvaluationResult]) -> Optional[Dict[str, Any]]:
    """Convert optional DatasetEvaluationResult to dict."""
    if result is None:
        return None

    return result.to_dict()


def _threshold_from_result(result: Optional[DatasetEvaluationResult]) -> Optional[float]:
    """Read selected theta from result."""
    if result is None:
        return None

    return float(result.threshold)


def _persistence_from_result(result: Optional[DatasetEvaluationResult]) -> Optional[int]:
    """Read selected persistence from result."""
    if result is None:
        return None

    return int(result.persistence)

def print_step13_threshold_config_debug(config: Mapping[str, Any]) -> None:
    """Print Step-13 threshold-selection config actually visible to run_proposed.py."""
    threshold_grid = get_by_path(
        config,
        "training.threshold_selection.threshold_grid",
        get_by_path(config, "training.threshold_selection.theta_grid", None),
    )

    theta_grid = get_by_path(
        config,
        "training.threshold_selection.theta_grid",
        None,
    )

    persistence_grid = get_by_path(
        config,
        "training.threshold_selection.persistence_grid",
        None,
    )

    persistence_values = get_by_path(
        config,
        "training.threshold_selection.persistence_values",
        None,
    )

    print("=" * 100)
    print("STEP 13 THRESHOLD CONFIG DEBUG")
    print("=" * 100)
    print(f"training.threshold_selection.threshold_grid     : {threshold_grid}")
    print(f"training.threshold_selection.theta_grid         : {theta_grid}")
    print(f"training.threshold_selection.persistence_grid   : {persistence_grid}")
    print(f"training.threshold_selection.persistence_values : {persistence_values}")

    if persistence_grid is not None:
        print(f"persistence_grid count                          : {len(list(persistence_grid))}")
    if persistence_values is not None:
        print(f"persistence_values count                        : {len(list(persistence_values))}")

    print("=" * 100)


def print_step13_combined_primary_metrics(
    dataset1_result: Optional[DatasetEvaluationResult],
    dataset2_result: Optional[DatasetEvaluationResult],
    dataset3_result: Optional[DatasetEvaluationResult],
    model_name: str,
) -> None:
    """Print combined locked primary metrics to console."""
    rows = []

    if dataset1_result is not None:
        rows.append(
            {
                "Model": f"{model_name} | Dataset-1 Test",
                **extract_primary_metrics(dataset1_result.metrics),
            }
        )

    if dataset2_result is not None:
        rows.append(
            {
                "Model": f"{model_name} | Dataset-2 External",
                **extract_primary_metrics(dataset2_result.metrics),
            }
        )

    if dataset3_result is not None:
        rows.append(
            {
                "Model": f"{model_name} | Dataset-3 Online",
                **extract_primary_metrics(dataset3_result.metrics),
            }
        )

    if rows:
        print_primary_metric_table(
            title="STEP 13 FULL PROPOSED MODEL — PRIMARY METRICS",
            rows=rows,
            model_key="Model",
        )


def run_full_proposed_experiment(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> ProposedExperimentSummary:
    """
    Run Step 13 full proposed-model experiment.

    Workflow:
    1. Reuse/train full proposed model checkpoint.
    2. Select threshold/persistence on Dataset-1 validation.
    3. Evaluate Dataset-1 internal test.
    4. Evaluate Dataset-2 external scenario.
    5. Evaluate Dataset-3 online case study.
    """
    start_time = time.perf_counter()

    experiment_config = build_proposed_experiment_config(config)

    device_info = setup_device_from_config(config, verbose=True)
    device = device_info.device

    print("=" * 100)
    print("STEP 13 FULL PROPOSED EXPERIMENT START")
    print("=" * 100)
    print(f"Model name             : {experiment_config.model_name}")
    print(f"Active seed            : {active_seed}")
    print(f"Device                 : {device}")
    print(f"Retrain policy         : {experiment_config.retrain_policy}")
    print(f"Evaluate Dataset-1     : {experiment_config.evaluate_dataset1}")
    print(f"Evaluate Dataset-2     : {experiment_config.evaluate_dataset2}")
    print(f"Evaluate Dataset-3     : {experiment_config.evaluate_dataset3}")
    print("Synthetic Step-10 theta=0.55 is not used as final model threshold.")
    print("=" * 100)

    print_step13_threshold_config_debug(config)

    trained_now, training_summary, checkpoint_path = maybe_train_full_proposed_model(
        config=config,
        experiment_config=experiment_config,
        active_seed=active_seed,
    )

    print("=" * 100)
    print("STEP 13 CHECKPOINT READY")
    print("=" * 100)
    print(f"Trained in Step 13     : {trained_now}")
    print(f"Checkpoint             : {checkpoint_path}")
    print("=" * 100)

    dataset1_result: Optional[DatasetEvaluationResult] = None
    dataset2_result: Optional[DatasetEvaluationResult] = None
    dataset3_result: Optional[DatasetEvaluationResult] = None

    if experiment_config.evaluate_dataset1:
        dataset1_result = run_dataset1_evaluation(
            config=config,
            active_seed=active_seed,
            checkpoint_path=str(checkpoint_path),
            model_name=experiment_config.model_name,
            device=device,
        )
    else:
        print("Dataset-1 evaluation skipped by config.")

    if experiment_config.evaluate_dataset2:
        dataset2_result = run_dataset2_external_evaluation(
            config=config,
            active_seed=active_seed,
            checkpoint_path=str(checkpoint_path),
            model_name=experiment_config.model_name,
            device=device,
        )
    else:
        print("Dataset-2 evaluation skipped by config.")

    if experiment_config.evaluate_dataset3:
        dataset3_result = run_dataset3_online_evaluation(
            config=config,
            active_seed=active_seed,
            checkpoint_path=str(checkpoint_path),
            method_name=experiment_config.model_name,
            device=device,
        )
    else:
        print("Dataset-3 evaluation skipped by config.")

    selected_theta = _threshold_from_result(dataset1_result)
    selected_persistence = _persistence_from_result(dataset1_result)

    if selected_theta is None and dataset2_result is not None:
        selected_theta = float(dataset2_result.threshold)
        selected_persistence = int(dataset2_result.persistence)

    if selected_theta is None and dataset3_result is not None:
        selected_theta = float(dataset3_result.threshold)
        selected_persistence = int(dataset3_result.persistence)

    summary_path = _project_path(config, experiment_config.summary_json)

    output_paths = {
        "step13_summary_json": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "dataset1_main_comparison_csv": str(
            _project_path(
                config,
                str(
                    get_by_path(
                        config,
                        "paths.dataset1_main_comparison_csv",
                        "results/tables/dataset1_main_comparison.csv",
                    )
                ),
            )
        ),
        "dataset2_external_comparison_csv": str(
            _project_path(
                config,
                str(
                    get_by_path(
                        config,
                        "paths.dataset2_external_comparison_csv",
                        "results/tables/dataset2_external_comparison.csv",
                    )
                ),
            )
        ),
        "dataset3_online_case_study_csv": str(
            _project_path(
                config,
                str(
                    get_by_path(
                        config,
                        "paths.dataset3_online_case_study_csv",
                        "results/tables/dataset3_online_case_study.csv",
                    )
                ),
            )
        ),
        "threshold_selection_json": str(
            _project_path(
                config,
                str(
                    get_by_path(
                        config,
                        "paths.proposed_threshold_selection_json",
                        "results/tables/proposed_threshold_selection.json",
                    )
                ),
            )
        ),
    }

    summary = ProposedExperimentSummary(
        final_status="PASSED",
        active_seed=int(active_seed),
        model_name=experiment_config.model_name,
        retrain_policy=experiment_config.retrain_policy,
        checkpoint_path=str(checkpoint_path),
        trained_in_step13=bool(trained_now),
        training_summary=training_summary,
        dataset1_result=_result_or_none(dataset1_result),
        dataset2_result=_result_or_none(dataset2_result),
        dataset3_result=_result_or_none(dataset3_result),
        selected_threshold=selected_theta,
        selected_persistence=selected_persistence,
        output_paths=output_paths,
        runtime_seconds=float(time.perf_counter() - start_time),
        leakage_rules={
            "dataset1_train_used_for_training": True,
            "dataset1_validation_used_for_threshold_selection": True,
            "dataset1_test_used_only_for_internal_test": True,
            "dataset2_used_only_for_external_test": True,
            "dataset3_used_only_for_online_case_study": True,
            "synthetic_step10_theta_not_used": True,
            "baseline_or_ablation_comparison_not_done_in_step13": True,
        },
    )

    _save_json_safe(summary.to_dict(), summary_path)

    print_step13_combined_primary_metrics(
        dataset1_result=dataset1_result,
        dataset2_result=dataset2_result,
        dataset3_result=dataset3_result,
        model_name=experiment_config.model_name,
    )

    print("=" * 100)
    print("STEP 13 FULL PROPOSED EXPERIMENT SUMMARY")
    print("=" * 100)
    print(f"Final status           : {summary.final_status}")
    print(f"Trained in Step 13     : {summary.trained_in_step13}")
    print(f"Checkpoint             : {summary.checkpoint_path}")
    print(f"Selected theta         : {summary.selected_threshold}")
    print(f"Selected persistence   : {summary.selected_persistence}")
    print(f"Runtime seconds        : {summary.runtime_seconds:.3f}")
    print("Saved outputs:")
    for key, value in summary.output_paths.items():
        print(f"  {key}: {value}")
    print("=" * 100)

    return summary


# Compatibility aliases for main.py.
run_step13_proposed_experiment = run_full_proposed_experiment
run_proposed_experiment = run_full_proposed_experiment


__all__ = [
    "ProposedExperimentConfig",
    "ProposedExperimentSummary",
    "build_proposed_experiment_config",
    "make_full_proposed_training_config",
    "resolve_step13_checkpoint_path",
    "maybe_train_full_proposed_model",
    "print_step13_combined_primary_metrics",
    "run_full_proposed_experiment",
    "run_step13_proposed_experiment",
    "run_proposed_experiment",
    "print_step13_threshold_config_debug",
]