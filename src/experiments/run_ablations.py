"""
Step 16: Official controlled ablation study.

Purpose
-------
Train and evaluate the full proposed model and each locked ablation under the
same protocol:

1. Build each variant separately.
2. Initialize each variant from scratch.
3. Train each variant from scratch on Dataset-1 train only.
4. Select theta and N_p on Dataset-1 validation only.
5. Evaluate Dataset-1 test, Dataset-2 external, and Dataset-3 online.
6. Save consolidated ablation tables and plots.

Locked variants
---------------
- full
- no_residual_evolution
- no_weak_accumulation
- no_kirchhoff_exchange
- no_third_order
- no_liquid_dynamics

Outputs
-------
- results/models/ablations/
- results/tables/ablation_results.csv
- results/tables/ablation_results_all_splits.csv
- results/tables/ablation_threshold_selection.csv
- results/tables/ablations/<variant>/
- results/figures/ablation_plots/
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.evaluation.evaluate_dataset1 import (
    DatasetEvaluationResult,
    apply_persistence_alarm,
    build_evaluation_dataloader,
    collect_model_predictions,
    compute_probability_metrics,
    evaluate_bundle_with_threshold,
    load_trained_model_for_evaluation,
    select_threshold_and_persistence,
)
from src.evaluation.result_tables import extract_primary_metrics, print_primary_metric_table
from src.models.model_factory import get_available_model_variants
from src.training.trainer import run_step12_training_protocol
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir


LOCKED_ABLATION_VARIANTS: Tuple[str, ...] = (
    "full",
    "no_residual_evolution",
    "no_weak_accumulation",
    "no_kirchhoff_exchange",
    "no_third_order",
    "no_liquid_dynamics",
)


# -------------------------------------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------------------------------------


@dataclass
class Step16AblationConfig:
    """Runtime config for Step 16."""

    enabled: bool = True
    experiment_name: str = "official_controlled_ablation_study_step16"

    variants: List[str] = field(default_factory=lambda: list(LOCKED_ABLATION_VARIANTS))

    # Official rule: train all ablations from scratch.
    retrain_policy: str = "always"

    evaluate_dataset1: bool = True
    evaluate_dataset2: bool = True
    evaluate_dataset3: bool = True

    models_dir: str = "results/models/ablations"
    tables_dir: str = "results/tables/ablations"
    plots_dir: str = "results/figures/ablation_plots"

    summary_json: str = "results/tables/ablation_summary.json"
    results_csv: str = "results/tables/ablation_results.csv"
    all_splits_csv: str = "results/tables/ablation_results_all_splits.csv"
    threshold_csv: str = "results/tables/ablation_threshold_selection.csv"

    save_plots: bool = True
    save_variant_artifacts: bool = True
    print_console_tables: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationVariantResult:
    """Result payload for one ablation variant."""

    variant_name: str
    display_name: str
    status: str

    trained_from_scratch: bool
    checkpoint_path: Optional[str]
    training_summary: Optional[Dict[str, Any]]

    selected_theta: Optional[float]
    selected_persistence: Optional[int]
    selected_validation_f1: Optional[float]
    selected_validation_auprc: Optional[float]
    selected_validation_auroc: Optional[float]
    selected_validation_fpr: Optional[float]

    dataset1_result: Optional[Dict[str, Any]]
    dataset2_result: Optional[Dict[str, Any]]
    dataset3_result: Optional[Dict[str, Any]]

    artifact_paths: Dict[str, str]
    runtime_seconds: float
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step16AblationSummary:
    """Final Step-16 summary."""

    final_status: str
    active_seed: int
    experiment_name: str
    variants: List[str]
    results: List[Dict[str, Any]]
    output_paths: Dict[str, str]
    runtime_seconds: float
    fairness_rules: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------------------------------------


def _project_path(config: Mapping[str, Any], path_value: str) -> Path:
    """Resolve project-relative path."""
    return resolve_project_path(config, str(path_value))


def _set_by_path(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set nested dictionary value by dotted path."""
    current: Dict[str, Any] = config
    keys = path.split(".")

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-safe Python types."""
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

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def _save_json_safe(payload: Mapping[str, Any], output_path: Path) -> None:
    """Save JSON with NumPy/Path conversion."""
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(dict(payload)), file, indent=2)


def _safe_float(value: Any) -> Optional[float]:
    """Convert to finite float or None."""
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def _variant_display_name(variant_name: str) -> str:
    """Human-readable variant name."""
    if variant_name == "full":
        return "Full"
    return str(variant_name).replace("_", " ")


def _metric(metrics: Optional[Mapping[str, Any]], key: str) -> Optional[float]:
    """Read metric from dict safely."""
    if not isinstance(metrics, Mapping):
        return None
    return _safe_float(metrics.get(key))


def _result_metrics(result_dict: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extract metrics dict from DatasetEvaluationResult-like dict."""
    if not isinstance(result_dict, Mapping):
        return {}
    metrics = result_dict.get("metrics", {})
    return dict(metrics) if isinstance(metrics, Mapping) else {}


# -------------------------------------------------------------------------------------------------
# Config builders
# -------------------------------------------------------------------------------------------------


def build_step16_ablation_config(config: Mapping[str, Any]) -> Step16AblationConfig:
    """Build Step-16 config from project config."""
    configured_variants = get_by_path(
        config,
        "experiments.step16.variants",
        get_by_path(config, "experiments.ablations.variants", list(LOCKED_ABLATION_VARIANTS)),
    )

    if configured_variants is None:
        variants = list(LOCKED_ABLATION_VARIANTS)
    else:
        variants = [str(item) for item in list(configured_variants)]

    return Step16AblationConfig(
        enabled=bool(
            get_by_path(
                config,
                "experiments.step16.enabled",
                get_by_path(config, "experiments.ablations.enabled", True),
            )
        ),
        experiment_name=str(
            get_by_path(
                config,
                "experiments.step16.experiment_name",
                get_by_path(
                    config,
                    "experiments.ablations.experiment_name",
                    "official_controlled_ablation_study_step16",
                ),
            )
        ),
        variants=variants,
        retrain_policy=str(
            get_by_path(
                config,
                "experiments.step16.retrain_policy",
                get_by_path(config, "experiments.ablations.retrain_policy", "always"),
            )
        ),
        evaluate_dataset1=bool(
            get_by_path(
                config,
                "experiments.step16.evaluate_dataset1",
                get_by_path(config, "experiments.ablations.evaluate_dataset1", True),
            )
        ),
        evaluate_dataset2=bool(
            get_by_path(
                config,
                "experiments.step16.evaluate_dataset2",
                get_by_path(config, "experiments.ablations.evaluate_dataset2", True),
            )
        ),
        evaluate_dataset3=bool(
            get_by_path(
                config,
                "experiments.step16.evaluate_dataset3",
                get_by_path(config, "experiments.ablations.evaluate_dataset3", True),
            )
        ),
        models_dir=str(
            get_by_path(
                config,
                "experiments.step16.models_dir",
                "results/models/ablations",
            )
        ),
        tables_dir=str(
            get_by_path(
                config,
                "experiments.step16.tables_dir",
                "results/tables/ablations",
            )
        ),
        plots_dir=str(
            get_by_path(
                config,
                "experiments.step16.plots_dir",
                "results/figures/ablation_plots",
            )
        ),
        summary_json=str(
            get_by_path(
                config,
                "experiments.step16.summary_json",
                "results/tables/ablation_summary.json",
            )
        ),
        results_csv=str(
            get_by_path(
                config,
                "experiments.step16.results_csv",
                "results/tables/ablation_results.csv",
            )
        ),
        all_splits_csv=str(
            get_by_path(
                config,
                "experiments.step16.all_splits_csv",
                "results/tables/ablation_results_all_splits.csv",
            )
        ),
        threshold_csv=str(
            get_by_path(
                config,
                "experiments.step16.threshold_csv",
                "results/tables/ablation_threshold_selection.csv",
            )
        ),
        save_plots=bool(
            get_by_path(config, "experiments.step16.save_plots", True)
        ),
        save_variant_artifacts=bool(
            get_by_path(config, "experiments.step16.save_variant_artifacts", True)
        ),
        print_console_tables=bool(
            get_by_path(config, "experiments.step16.print_console_tables", True)
        ),
    )


def validate_step16_variants(config: Mapping[str, Any], variants: Sequence[str]) -> None:
    """Validate requested variants against model factory."""
    available = get_available_model_variants(config)
    allowed = set(["full"])
    allowed.update(str(item) for item in available.get("official_ablations", []))

    missing = [variant for variant in variants if variant not in allowed]
    if missing:
        raise ValueError(
            "Unknown Step-16 ablation variant(s): "
            f"{missing}. Available official variants: {sorted(allowed)}"
        )


def make_variant_training_config(
    config: Mapping[str, Any],
    step16_config: Step16AblationConfig,
    variant_name: str,
) -> Dict[str, Any]:
    """
    Create a config copy for one variant.

    This forces:
    - training.step12.variant_name = variant_name
    - checkpoint output under results/models/ablations/
    - per-variant Step-12 history/summary artifacts
    - per-variant evaluation artifacts
    """
    cfg = copy.deepcopy(dict(config))

    variant_name = str(variant_name)
    variant_tables_dir = str(Path(step16_config.tables_dir) / variant_name)
    variant_models_dir = str(step16_config.models_dir)

    # Train selected ablation variant from scratch.
    _set_by_path(cfg, "training.step12.model_name", f"ablation_{variant_name}")
    _set_by_path(cfg, "training.step12.variant_name", variant_name)

    # Model checkpoints.
    _set_by_path(cfg, "paths.models_dir", variant_models_dir)
    _set_by_path(cfg, "training.checkpointing.best_checkpoint_name", f"{variant_name}_best.pt")
    _set_by_path(cfg, "training.checkpointing.last_checkpoint_name", f"{variant_name}_last.pt")

    # Step-12 per-variant training artifacts.
    _set_by_path(
        cfg,
        "paths.step12_training_history_csv",
        str(Path(variant_tables_dir) / "training_history.csv"),
    )
    _set_by_path(
        cfg,
        "paths.step12_training_history_json",
        str(Path(variant_tables_dir) / "training_history.json"),
    )
    _set_by_path(
        cfg,
        "paths.step12_training_summary_json",
        str(Path(variant_tables_dir) / "training_summary.json"),
    )
    _set_by_path(
        cfg,
        "paths.step12_validation_predictions_npz",
        str(Path(variant_tables_dir) / "step12_validation_predictions.npz"),
    )

    # Step-13-like per-variant evaluation artifacts.
    _set_by_path(
        cfg,
        "paths.dataset1_main_comparison_csv",
        str(Path(variant_tables_dir) / "dataset1_result.csv"),
    )
    _set_by_path(
        cfg,
        "paths.dataset1_proposed_summary_json",
        str(Path(variant_tables_dir) / "dataset1_summary.json"),
    )
    _set_by_path(
        cfg,
        "paths.proposed_threshold_selection_json",
        str(Path(variant_tables_dir) / "threshold_selection.json"),
    )
    _set_by_path(
        cfg,
        "paths.proposed_threshold_candidates_csv",
        str(Path(variant_tables_dir) / "threshold_candidates.csv"),
    )
    _set_by_path(
        cfg,
        "paths.dataset1_val_predictions_npz",
        str(Path(variant_tables_dir) / "dataset1_val_predictions.npz"),
    )
    _set_by_path(
        cfg,
        "paths.dataset1_test_predictions_npz",
        str(Path(variant_tables_dir) / "dataset1_test_predictions.npz"),
    )
    _set_by_path(
        cfg,
        "paths.dataset2_external_predictions_npz",
        str(Path(variant_tables_dir) / "dataset2_external_predictions.npz"),
    )
    _set_by_path(
        cfg,
        "paths.dataset3_online_predictions_npz",
        str(Path(variant_tables_dir) / "dataset3_online_predictions.npz"),
    )
    _set_by_path(
        cfg,
        "paths.dataset3_online_predictions_csv",
        str(Path(variant_tables_dir) / "dataset3_online_predictions.csv"),
    )

    return cfg


def resolve_variant_checkpoint_path(
    config: Mapping[str, Any],
    step16_config: Step16AblationConfig,
    variant_name: str,
) -> Path:
    """Resolve expected best checkpoint path for one variant."""
    return _project_path(
        config,
        str(Path(step16_config.models_dir) / f"{variant_name}_best.pt"),
    )


# -------------------------------------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------------------------------------


def _false_alarm_events(
    labels: np.ndarray,
    confirmed_alarm: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Count contiguous normal confirmed-alarm events."""
    details: List[Dict[str, Any]] = []
    in_event = False
    start_position = -1
    current_segment: Optional[str] = None
    event_id = 0

    for index in range(len(labels)):
        segment = str(segment_ids[index])
        segment_changed = current_segment is not None and segment != current_segment

        if segment_changed and in_event:
            details.append(
                {
                    "event_id": int(event_id),
                    "segment_id": str(current_segment),
                    "start_position": int(start_position),
                    "end_position": int(index - 1),
                    "duration_steps": int(index - start_position),
                }
            )
            event_id += 1
            in_event = False

        if current_segment is None or segment_changed:
            current_segment = segment

        is_false_alarm = bool(
            valid_mask[index] > 0.5
            and labels[index] == 0
            and confirmed_alarm[index] == 1
        )

        if is_false_alarm and not in_event:
            in_event = True
            start_position = int(index)

        if in_event and not is_false_alarm:
            details.append(
                {
                    "event_id": int(event_id),
                    "segment_id": str(current_segment),
                    "start_position": int(start_position),
                    "end_position": int(index - 1),
                    "duration_steps": int(index - start_position),
                }
            )
            event_id += 1
            in_event = False

    if in_event:
        details.append(
            {
                "event_id": int(event_id),
                "segment_id": str(current_segment),
                "start_position": int(start_position),
                "end_position": int(len(labels) - 1),
                "duration_steps": int(len(labels) - start_position),
            }
        )

    return int(len(details)), details


def evaluate_variant_split(
    config: Mapping[str, Any],
    variant_name: str,
    display_name: str,
    checkpoint_path: Path,
    split_name: str,
    output_split_name: str,
    theta: float,
    persistence: int,
    active_seed: int,
    device: Any,
    full_sequence: bool = False,
    prediction_npz_path: Optional[Path] = None,
    prediction_csv_path: Optional[Path] = None,
) -> DatasetEvaluationResult:
    """Evaluate one variant on one split with selected theta/Np."""
    model, checkpoint, checkpoint_metadata = load_trained_model_for_evaluation(
        config=config,
        checkpoint_path=str(checkpoint_path),
        device=device,
        variant_name=variant_name,
    )

    loader, dataset = build_evaluation_dataloader(
        config=config,
        split_name=split_name,
        active_seed=active_seed,
        full_sequence=full_sequence,
    )

    bundle = collect_model_predictions(
        model=model,
        dataloader=loader,
        device=device,
        split_name=split_name,
        checkpoint_path=str(checkpoint),
        model_name=display_name,
    )

    metrics = evaluate_bundle_with_threshold(
        bundle=bundle,
        theta=float(theta),
        persistence=int(persistence),
    )

    artifact_paths: Dict[str, str] = {}

    if prediction_npz_path is not None:
        bundle.save_npz(prediction_npz_path)
        artifact_paths[f"{split_name}_predictions_npz"] = str(prediction_npz_path)

    if prediction_csv_path is not None:
        ensure_dir(prediction_csv_path.parent)
        pd.DataFrame(
            {
                "row_index": bundle.row_indices,
                "segment_id": bundle.segment_ids.astype(str),
                "label": bundle.labels.astype(int),
                "valid_mask": bundle.valid_mask.astype(float),
                "probability": bundle.probabilities.astype(float),
            }
        ).to_csv(prediction_csv_path, index=False)
        artifact_paths[f"{split_name}_predictions_csv"] = str(prediction_csv_path)

    # Online-specific false-alarm event details.
    confirmed = apply_persistence_alarm(
        probabilities=bundle.probabilities,
        valid_mask=bundle.valid_mask,
        segment_ids=bundle.segment_ids,
        theta=float(theta),
        persistence=int(persistence),
    )

    false_alarm_event_count, false_alarm_event_details = _false_alarm_events(
        labels=bundle.labels,
        confirmed_alarm=confirmed,
        valid_mask=bundle.valid_mask,
        segment_ids=bundle.segment_ids,
    )

    metrics["row_level_false_alarms"] = int(metrics.get("false_alarms", 0))
    metrics["normal_alarm_event_count"] = int(false_alarm_event_count)
    metrics["false_alarm_event_count"] = int(false_alarm_event_count)
    metrics["false_alarm_events"] = int(false_alarm_event_count)
    metrics["false_alarm_event_details"] = false_alarm_event_details

    delays = metrics.get("detection_delays") or []
    if isinstance(delays, list):
        metrics["attack_1_delay"] = float(delays[0]) if len(delays) >= 1 else None
        metrics["attack_2_delay"] = float(delays[1]) if len(delays) >= 2 else None

    metrics["checkpoint_metadata"] = checkpoint_metadata

    return DatasetEvaluationResult(
        model_name=display_name,
        split_name=output_split_name,
        metrics=metrics,
        threshold=float(theta),
        persistence=int(persistence),
        checkpoint_path=str(checkpoint),
        prediction_summary=bundle.to_dict_summary(),
        artifact_paths=artifact_paths,
    )


def evaluate_variant_official_protocol(
    config: Mapping[str, Any],
    step16_config: Step16AblationConfig,
    variant_name: str,
    display_name: str,
    checkpoint_path: Path,
    active_seed: int,
    device: Any,
) -> Tuple[
    Optional[DatasetEvaluationResult],
    Optional[DatasetEvaluationResult],
    Optional[DatasetEvaluationResult],
    Dict[str, Any],
]:
    """
    Official Step-16 evaluation protocol.

    Dataset-1 validation selects theta/Np.
    Dataset-1 test, Dataset-2 external, and Dataset-3 online only apply it.
    """
    variant_tables_dir = _project_path(
        config,
        str(Path(step16_config.tables_dir) / variant_name),
    )
    ensure_dir(variant_tables_dir)

    # Load selected variant.
    model, checkpoint, checkpoint_metadata = load_trained_model_for_evaluation(
        config=config,
        checkpoint_path=str(checkpoint_path),
        device=device,
        variant_name=variant_name,
    )

    # Dataset-1 validation threshold selection.
    val_loader, val_dataset = build_evaluation_dataloader(
        config=config,
        split_name="val",
        active_seed=active_seed,
        full_sequence=False,
    )

    val_bundle = collect_model_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        split_name="val",
        checkpoint_path=str(checkpoint),
        model_name=display_name,
    )

    selection = select_threshold_and_persistence(
        validation_bundle=val_bundle,
        config=config,
    )

    threshold_json = variant_tables_dir / "threshold_selection.json"
    threshold_candidates_csv = variant_tables_dir / "threshold_candidates.csv"
    ensure_dir(threshold_json.parent)

    _save_json_safe(selection.to_dict(), threshold_json)
    pd.DataFrame(selection.candidates).to_csv(threshold_candidates_csv, index=False)

    val_npz = variant_tables_dir / "dataset1_val_predictions.npz"
    val_bundle.save_npz(val_npz)

    theta = float(selection.theta)
    persistence = int(selection.persistence)

    dataset1_result: Optional[DatasetEvaluationResult] = None
    dataset2_result: Optional[DatasetEvaluationResult] = None
    dataset3_result: Optional[DatasetEvaluationResult] = None

    if step16_config.evaluate_dataset1:
        dataset1_result = evaluate_variant_split(
            config=config,
            variant_name=variant_name,
            display_name=display_name,
            checkpoint_path=checkpoint_path,
            split_name="test",
            output_split_name="Dataset-1 Test",
            theta=theta,
            persistence=persistence,
            active_seed=active_seed,
            device=device,
            full_sequence=False,
            prediction_npz_path=variant_tables_dir / "dataset1_test_predictions.npz",
        )

    if step16_config.evaluate_dataset2:
        dataset2_result = evaluate_variant_split(
            config=config,
            variant_name=variant_name,
            display_name=display_name,
            checkpoint_path=checkpoint_path,
            split_name="external",
            output_split_name="Dataset-2 External",
            theta=theta,
            persistence=persistence,
            active_seed=active_seed,
            device=device,
            full_sequence=False,
            prediction_npz_path=variant_tables_dir / "dataset2_external_predictions.npz",
        )

    if step16_config.evaluate_dataset3:
        dataset3_result = evaluate_variant_split(
            config=config,
            variant_name=variant_name,
            display_name=display_name,
            checkpoint_path=checkpoint_path,
            split_name="online",
            output_split_name="Dataset-3 Online",
            theta=theta,
            persistence=persistence,
            active_seed=active_seed,
            device=device,
            full_sequence=True,
            prediction_npz_path=variant_tables_dir / "dataset3_online_predictions.npz",
            prediction_csv_path=variant_tables_dir / "dataset3_online_predictions.csv",
        )

    threshold_payload = {
        "theta": theta,
        "persistence": persistence,
        "objective": selection.objective,
        "monitor_split": selection.monitor_split,
        "selected_metric_value": selection.selected_metric_value,
        "selected_candidate": selection.selected_candidate,
        "candidate_count": selection.candidate_count,
        "validation_prediction_summary": val_bundle.to_dict_summary(),
        "validation_dataset_summary": val_dataset.summary(),
        "checkpoint_metadata": checkpoint_metadata,
        "artifact_paths": {
            "threshold_selection_json": str(threshold_json),
            "threshold_candidates_csv": str(threshold_candidates_csv),
            "dataset1_val_predictions_npz": str(val_npz),
        },
    }

    return dataset1_result, dataset2_result, dataset3_result, threshold_payload


# -------------------------------------------------------------------------------------------------
# Table construction
# -------------------------------------------------------------------------------------------------


def _split_result_to_row(
    variant_name: str,
    display_name: str,
    split_name: str,
    result: Optional[Mapping[str, Any]],
    threshold_info: Mapping[str, Any],
    training_summary: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Convert one split result into a flat CSV row."""
    metrics = _result_metrics(result)

    row = {
        "variant_name": variant_name,
        "display_name": display_name,
        "split": split_name,
        "theta": _safe_float(threshold_info.get("theta")),
        "persistence": int(threshold_info.get("persistence")) if threshold_info.get("persistence") is not None else None,
        "val_selected_f1": _metric(threshold_info.get("selected_candidate", {}), "f1"),
        "val_selected_auprc": _metric(threshold_info.get("selected_candidate", {}), "auprc"),
        "val_selected_auroc": _metric(threshold_info.get("selected_candidate", {}), "auroc"),
        "val_selected_fpr": _metric(threshold_info.get("selected_candidate", {}), "fpr"),
        "AUROC": _metric(metrics, "auroc"),
        "AUPRC": _metric(metrics, "auprc"),
        "F1": _metric(metrics, "f1"),
        "Precision": _metric(metrics, "precision"),
        "Recall": _metric(metrics, "recall"),
        "FPR": _metric(metrics, "fpr"),
        "Attack Detection Rate": _metric(metrics, "attack_detection_rate"),
        "Detection Delay": _metric(metrics, "mean_detection_delay"),
        "False Alarm Rows": _metric(metrics, "row_level_false_alarms"),
        "False Alarm Events": _metric(metrics, "false_alarm_events"),
        "Attack-1 Delay": _metric(metrics, "attack_1_delay"),
        "Attack-2 Delay": _metric(metrics, "attack_2_delay"),
        "Runtime": _metric(metrics, "runtime_seconds"),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "tn": metrics.get("tn"),
        "fn": metrics.get("fn"),
        "checkpoint_path": result.get("checkpoint_path") if isinstance(result, Mapping) else None,
        "best_epoch": None,
        "best_monitor": None,
        "best_monitor_value": None,
        "epochs_completed": None,
        "trained_from_scratch": True,
        "threshold_selected_on_dataset1_validation_only": True,
        "dataset2_used_for_tuning": False,
        "dataset3_used_for_tuning": False,
        "uses_same_xi_features": True,
        "raw_shortcut_columns_used": False,
    }

    if isinstance(training_summary, Mapping):
        row["best_epoch"] = training_summary.get("best_epoch")
        row["best_monitor"] = training_summary.get("best_monitor")
        row["best_monitor_value"] = training_summary.get("best_monitor_value")
        row["epochs_completed"] = training_summary.get("epochs_completed")

    return row


def build_ablation_tables(
    results: Sequence[AblationVariantResult],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build:
    - dataset1-only ablation_results.csv
    - all-splits ablation_results_all_splits.csv
    - threshold-selection table
    """
    dataset1_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []

    for item in results:
        payload = item.to_dict()

        training_summary = payload.get("training_summary")
        threshold_info = {
            "theta": item.selected_theta,
            "persistence": item.selected_persistence,
            "selected_candidate": {
                "f1": item.selected_validation_f1,
                "auprc": item.selected_validation_auprc,
                "auroc": item.selected_validation_auroc,
                "fpr": item.selected_validation_fpr,
            },
        }

        split_payloads = [
            ("Dataset-1 Test", item.dataset1_result),
            ("Dataset-2 External", item.dataset2_result),
            ("Dataset-3 Online", item.dataset3_result),
        ]

        for split_name, split_result in split_payloads:
            if split_result is None:
                continue

            row = _split_result_to_row(
                variant_name=item.variant_name,
                display_name=item.display_name,
                split_name=split_name,
                result=split_result,
                threshold_info=threshold_info,
                training_summary=training_summary,
            )
            all_rows.append(row)

            if split_name == "Dataset-1 Test":
                dataset1_rows.append(row)

        threshold_rows.append(
            {
                "variant_name": item.variant_name,
                "display_name": item.display_name,
                "theta": item.selected_theta,
                "persistence": item.selected_persistence,
                "val_f1": item.selected_validation_f1,
                "val_auprc": item.selected_validation_auprc,
                "val_auroc": item.selected_validation_auroc,
                "val_fpr": item.selected_validation_fpr,
                "checkpoint_path": item.checkpoint_path,
                "status": item.status,
            }
        )

    dataset1_df = pd.DataFrame(dataset1_rows)
    all_splits_df = pd.DataFrame(all_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    return dataset1_df, all_splits_df, threshold_df


# -------------------------------------------------------------------------------------------------
# Console and plot output
# -------------------------------------------------------------------------------------------------


def _format_metric(value: Any, precision: int = 4) -> str:
    """Format metric for console."""
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{precision}f}"


def print_ablation_console_tables(
    dataset1_df: pd.DataFrame,
    all_splits_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> None:
    """Print important Step-16 tables to console."""
    print("=" * 120)
    print("STEP 16 VALIDATION THRESHOLD SELECTION")
    print("=" * 120)

    if threshold_df.empty:
        print("No threshold rows available.")
    else:
        print(
            f"{'Variant':30s} | {'theta':>6s} | {'Np':>3s} | "
            f"{'Val F1':>8s} | {'Val AUPRC':>10s} | {'Val AUROC':>10s} | {'Val FPR':>8s}"
        )
        print("-" * 120)

        for _, row in threshold_df.iterrows():
            print(
                f"{str(row.get('display_name')):30s} | "
                f"{_format_metric(row.get('theta'), 2):>6s} | "
                f"{str(int(row.get('persistence')) if pd.notna(row.get('persistence')) else 'NA'):>3s} | "
                f"{_format_metric(row.get('val_f1')):>8s} | "
                f"{_format_metric(row.get('val_auprc')):>10s} | "
                f"{_format_metric(row.get('val_auroc')):>10s} | "
                f"{_format_metric(row.get('val_fpr')):>8s}"
            )

    print("=" * 120)
    print("STEP 16 DATASET-1 TEST ABLATION RESULTS")
    print("=" * 120)

    if dataset1_df.empty:
        print("No Dataset-1 rows available.")
    else:
        print(
            f"{'Variant':30s} | {'AUPRC':>8s} | {'AUROC':>8s} | {'F1':>8s} | "
            f"{'Prec.':>8s} | {'Recall':>8s} | {'FPR':>8s} | {'ADR':>8s} | {'Delay':>8s}"
        )
        print("-" * 120)

        for _, row in dataset1_df.iterrows():
            print(
                f"{str(row.get('display_name')):30s} | "
                f"{_format_metric(row.get('AUPRC')):>8s} | "
                f"{_format_metric(row.get('AUROC')):>8s} | "
                f"{_format_metric(row.get('F1')):>8s} | "
                f"{_format_metric(row.get('Precision')):>8s} | "
                f"{_format_metric(row.get('Recall')):>8s} | "
                f"{_format_metric(row.get('FPR')):>8s} | "
                f"{_format_metric(row.get('Attack Detection Rate')):>8s} | "
                f"{_format_metric(row.get('Detection Delay')):>8s}"
            )

    print("=" * 120)
    print("STEP 16 ALL-SPLITS PRIMARY METRICS")
    print("=" * 120)

    if all_splits_df.empty:
        print("No all-splits rows available.")
    else:
        print(
            f"{'Variant':30s} | {'Split':20s} | {'AUPRC':>8s} | {'F1':>8s} | "
            f"{'FPR':>8s} | {'ADR':>8s} | {'Delay':>8s}"
        )
        print("-" * 120)

        for _, row in all_splits_df.iterrows():
            print(
                f"{str(row.get('display_name')):30s} | "
                f"{str(row.get('split')):20s} | "
                f"{_format_metric(row.get('AUPRC')):>8s} | "
                f"{_format_metric(row.get('F1')):>8s} | "
                f"{_format_metric(row.get('FPR')):>8s} | "
                f"{_format_metric(row.get('Attack Detection Rate')):>8s} | "
                f"{_format_metric(row.get('Detection Delay')):>8s}"
            )

    print("=" * 120)


def save_ablation_plots(dataset1_df: pd.DataFrame, output_dir: Path) -> List[str]:
    """Save simple Dataset-1 ablation bar plots."""
    saved: List[str] = []

    if dataset1_df.empty:
        return saved

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping ablation plots because matplotlib is unavailable: {exc}")
        return saved

    ensure_dir(output_dir)

    plot_specs = [
        ("F1", "Dataset-1 F1 by ablation", "dataset1_f1"),
        ("AUPRC", "Dataset-1 AUPRC by ablation", "dataset1_auprc"),
        ("AUROC", "Dataset-1 AUROC by ablation", "dataset1_auroc"),
        ("FPR", "Dataset-1 FPR by ablation", "dataset1_fpr"),
        ("Detection Delay", "Dataset-1 detection delay by ablation", "dataset1_delay"),
    ]

    labels = dataset1_df["display_name"].astype(str).tolist()

    for metric, title, stem in plot_specs:
        if metric not in dataset1_df.columns:
            continue

        values = pd.to_numeric(dataset1_df[metric], errors="coerce").to_numpy(dtype=float)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xlabel("Ablation variant")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()

        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"

        fig.savefig(png_path, dpi=180)
        fig.savefig(pdf_path)
        plt.close(fig)

        saved.extend([str(png_path), str(pdf_path)])

    return saved


# -------------------------------------------------------------------------------------------------
# Main Step-16 runner
# -------------------------------------------------------------------------------------------------


def run_step16_ablation_study(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> Step16AblationSummary:
    """
    Run official controlled ablation study.

    Each variant is trained from scratch and evaluated under the same protocol.
    """
    start_time = time.perf_counter()

    step16_config = build_step16_ablation_config(config)
    variants = list(step16_config.variants)

    validate_step16_variants(config, variants)

    device_info = setup_device_from_config(config, verbose=True)
    device = device_info.device

    models_dir = _project_path(config, step16_config.models_dir)
    tables_dir = _project_path(config, step16_config.tables_dir)
    plots_dir = _project_path(config, step16_config.plots_dir)

    ensure_dir(models_dir)
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    print("=" * 120)
    print("STEP 16 OFFICIAL CONTROLLED ABLATION STUDY START")
    print("=" * 120)
    print(f"Experiment name       : {step16_config.experiment_name}")
    print(f"Active seed           : {active_seed}")
    print(f"Device                : {device}")
    print(f"Variants              : {variants}")
    print(f"Retrain policy        : {step16_config.retrain_policy}")
    print(f"Models dir            : {models_dir}")
    print(f"Tables dir            : {tables_dir}")
    print(f"Plots dir             : {plots_dir}")
    print("Official rule         : each variant is trained from scratch.")
    print("Threshold rule        : theta/Np selected on Dataset-1 validation only.")
    print("External/online rule  : Dataset-2 and Dataset-3 are never used for tuning.")
    print("=" * 120)

    if str(step16_config.retrain_policy).lower().strip() != "always":
        print(
            "WARNING: experiments.step16.retrain_policy is not 'always'. "
            "For the official paper ablation, use 'always'."
        )

    variant_results: List[AblationVariantResult] = []

    for variant_index, variant_name in enumerate(variants, start=1):
        variant_start = time.perf_counter()
        display_name = _variant_display_name(variant_name)

        print("=" * 120)
        print(f"STEP 16 VARIANT {variant_index}/{len(variants)}: {variant_name}")
        print("=" * 120)

        variant_config = make_variant_training_config(
            config=config,
            step16_config=step16_config,
            variant_name=variant_name,
        )

        variant_tables_dir = _project_path(
            variant_config,
            str(Path(step16_config.tables_dir) / variant_name),
        )
        ensure_dir(variant_tables_dir)

        checkpoint_path = resolve_variant_checkpoint_path(
            config=variant_config,
            step16_config=step16_config,
            variant_name=variant_name,
        )

        trained_from_scratch = False
        training_summary: Optional[Dict[str, Any]] = None

        try:
            retrain_policy = str(step16_config.retrain_policy).lower().strip()
            checkpoint_exists = checkpoint_path.exists()

            if retrain_policy == "reuse_if_exists" and checkpoint_exists:
                print(f"Reusing existing ablation checkpoint: {checkpoint_path}")
            elif retrain_policy == "never":
                if not checkpoint_exists:
                    raise FileNotFoundError(
                        f"retrain_policy='never' but checkpoint is missing: {checkpoint_path}"
                    )
                print(f"Using existing ablation checkpoint: {checkpoint_path}")
            else:
                print(f"Training from scratch: {variant_name}")
                print(f"Checkpoint target    : {checkpoint_path}")

                summary = run_step12_training_protocol(
                    config=variant_config,
                    active_seed=active_seed,
                )

                training_summary = summary.to_dict()
                trained_from_scratch = True

                best_path = training_summary.get("best_checkpoint_path")
                if best_path:
                    checkpoint_path = Path(str(best_path))

                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Expected best checkpoint not found after training: {checkpoint_path}"
                    )

            dataset1_result, dataset2_result, dataset3_result, threshold_payload = (
                evaluate_variant_official_protocol(
                    config=variant_config,
                    step16_config=step16_config,
                    variant_name=variant_name,
                    display_name=display_name,
                    checkpoint_path=checkpoint_path,
                    active_seed=active_seed,
                    device=device,
                )
            )

            variant_summary_path = variant_tables_dir / "variant_summary.json"

            selected_candidate = threshold_payload.get("selected_candidate", {})
            artifact_paths = {
                "variant_summary_json": str(variant_summary_path),
                "checkpoint_path": str(checkpoint_path),
                "variant_tables_dir": str(variant_tables_dir),
                "threshold_selection_json": str(variant_tables_dir / "threshold_selection.json"),
                "threshold_candidates_csv": str(variant_tables_dir / "threshold_candidates.csv"),
            }

            result_payload = AblationVariantResult(
                variant_name=variant_name,
                display_name=display_name,
                status="PASSED",
                trained_from_scratch=bool(trained_from_scratch),
                checkpoint_path=str(checkpoint_path),
                training_summary=training_summary,
                selected_theta=_safe_float(threshold_payload.get("theta")),
                selected_persistence=(
                    int(threshold_payload["persistence"])
                    if threshold_payload.get("persistence") is not None
                    else None
                ),
                selected_validation_f1=_safe_float(selected_candidate.get("f1")),
                selected_validation_auprc=_safe_float(selected_candidate.get("auprc")),
                selected_validation_auroc=_safe_float(selected_candidate.get("auroc")),
                selected_validation_fpr=_safe_float(selected_candidate.get("fpr")),
                dataset1_result=dataset1_result.to_dict() if dataset1_result is not None else None,
                dataset2_result=dataset2_result.to_dict() if dataset2_result is not None else None,
                dataset3_result=dataset3_result.to_dict() if dataset3_result is not None else None,
                artifact_paths=artifact_paths,
                runtime_seconds=float(time.perf_counter() - variant_start),
                message="",
            )

            _save_json_safe(result_payload.to_dict(), variant_summary_path)

            print("=" * 120)
            print(f"STEP 16 VARIANT SUMMARY: {variant_name}")
            print("=" * 120)
            print(f"Status              : {result_payload.status}")
            print(f"Checkpoint          : {result_payload.checkpoint_path}")
            print(f"Selected theta      : {result_payload.selected_theta}")
            print(f"Selected persistence: {result_payload.selected_persistence}")
            print(f"Validation F1       : {result_payload.selected_validation_f1}")
            print(f"Runtime seconds     : {result_payload.runtime_seconds:.3f}")
            print(f"Summary JSON        : {variant_summary_path}")

            split_rows = []

            if dataset1_result is not None:
                split_rows.append(
                    {
                        "Model": f"{display_name} | Dataset-1 Test",
                        **extract_primary_metrics(dataset1_result.metrics),
                    }
                )

            if dataset2_result is not None:
                split_rows.append(
                    {
                        "Model": f"{display_name} | Dataset-2 External",
                        **extract_primary_metrics(dataset2_result.metrics),
                    }
                )

            if dataset3_result is not None:
                split_rows.append(
                    {
                        "Model": f"{display_name} | Dataset-3 Online",
                        **extract_primary_metrics(dataset3_result.metrics),
                    }
                )

            if split_rows:
                print_primary_metric_table(
                    title=f"STEP 16 PRIMARY METRICS — {display_name}",
                    rows=split_rows,
                    model_key="Model",
                )

        except Exception as exc:
            result_payload = AblationVariantResult(
                variant_name=variant_name,
                display_name=display_name,
                status="FAILED",
                trained_from_scratch=bool(trained_from_scratch),
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
                training_summary=training_summary,
                selected_theta=None,
                selected_persistence=None,
                selected_validation_f1=None,
                selected_validation_auprc=None,
                selected_validation_auroc=None,
                selected_validation_fpr=None,
                dataset1_result=None,
                dataset2_result=None,
                dataset3_result=None,
                artifact_paths={},
                runtime_seconds=float(time.perf_counter() - variant_start),
                message=str(exc),
            )

            print("=" * 120)
            print(f"STEP 16 VARIANT FAILED: {variant_name}")
            print("=" * 120)
            print(str(exc))
            print("=" * 120)

        variant_results.append(result_payload)

    dataset1_df, all_splits_df, threshold_df = build_ablation_tables(variant_results)

    results_csv = _project_path(config, step16_config.results_csv)
    all_splits_csv = _project_path(config, step16_config.all_splits_csv)
    threshold_csv = _project_path(config, step16_config.threshold_csv)
    summary_json = _project_path(config, step16_config.summary_json)

    ensure_dir(results_csv.parent)
    ensure_dir(all_splits_csv.parent)
    ensure_dir(threshold_csv.parent)
    ensure_dir(summary_json.parent)

    dataset1_df.to_csv(results_csv, index=False)
    all_splits_df.to_csv(all_splits_csv, index=False)
    threshold_df.to_csv(threshold_csv, index=False)

    saved_plots: List[str] = []
    if step16_config.save_plots:
        saved_plots = save_ablation_plots(dataset1_df, plots_dir)

    if step16_config.print_console_tables:
        print_ablation_console_tables(
            dataset1_df=dataset1_df,
            all_splits_df=all_splits_df,
            threshold_df=threshold_df,
        )

    final_status = "PASSED" if all(item.status == "PASSED" for item in variant_results) else "FAILED"

    output_paths = {
        "models_dir": str(models_dir),
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
        "ablation_results_csv": str(results_csv),
        "ablation_results_all_splits_csv": str(all_splits_csv),
        "ablation_threshold_selection_csv": str(threshold_csv),
        "ablation_summary_json": str(summary_json),
    }

    for index, path in enumerate(saved_plots):
        output_paths[f"plot_{index:02d}"] = str(path)

    summary = Step16AblationSummary(
        final_status=final_status,
        active_seed=int(active_seed),
        experiment_name=step16_config.experiment_name,
        variants=list(variants),
        results=[item.to_dict() for item in variant_results],
        output_paths=output_paths,
        runtime_seconds=float(time.perf_counter() - start_time),
        fairness_rules={
            "all_variants_config_controlled": True,
            "all_variants_built_separately": True,
            "all_variants_initialized_from_scratch_when_retrain_policy_always": True,
            "all_variants_trained_on_dataset1_train_only": True,
            "theta_np_selected_on_dataset1_validation_only": True,
            "dataset1_test_not_used_for_threshold_selection": True,
            "dataset2_external_not_used_for_tuning": True,
            "dataset3_online_not_used_for_tuning": True,
            "same_9_scaled_xi_features": True,
            "raw_shortcut_columns_used": False,
            "synthetic_step10_theta_used": False,
        },
    )

    _save_json_safe(summary.to_dict(), summary_json)

    print("=" * 120)
    print("STEP 16 OFFICIAL CONTROLLED ABLATION STUDY SUMMARY")
    print("=" * 120)
    print(f"Final status       : {summary.final_status}")
    print(f"Active seed        : {summary.active_seed}")
    print(f"Variants           : {summary.variants}")
    print(f"Runtime seconds    : {summary.runtime_seconds:.3f}")
    print("Saved outputs:")
    for key, value in summary.output_paths.items():
        print(f"  {key}: {value}")
    print("=" * 120)

    if final_status != "PASSED":
        failed = [item.variant_name for item in variant_results if item.status != "PASSED"]
        raise RuntimeError(f"Step 16 ablation study failed for variants: {failed}")

    return summary


# Compatibility aliases for main.py.
run_step16_official_ablation_study = run_step16_ablation_study
run_official_ablation_study = run_step16_ablation_study
run_ablations_experiment = run_step16_ablation_study


__all__ = [
    "LOCKED_ABLATION_VARIANTS",
    "Step16AblationConfig",
    "AblationVariantResult",
    "Step16AblationSummary",
    "build_step16_ablation_config",
    "validate_step16_variants",
    "make_variant_training_config",
    "run_step16_ablation_study",
    "run_step16_official_ablation_study",
    "run_official_ablation_study",
    "run_ablations_experiment",
]