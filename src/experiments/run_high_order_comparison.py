"""
Step 17: High-order comparison experiment.

Purpose
-------
Answer the professor's question:

    "What is the contribution of feature high-order vs model high-order?"

This is a 2x2 factorial comparison:

    H0: no feature high-order, no model high-order
    H1: feature high-order only
    H2: model high-order only
    H3: feature high-order + model high-order

Interpretation:
    Feature high-order effect:
        H1 - H0
        H3 - H2

    Model high-order effect:
        H2 - H0
        H3 - H1

    Interaction effect:
        H3 - H2 - H1 + H0

Important:
    This is NOT the frozen component-intervention ablation.
    Each H-variant should be trained fairly under the same Step-12 protocol.

Outputs
-------
    results/tables/high_order_comparison.csv
    results/tables/high_order_comparison_all_splits.csv
    results/tables/high_order_effects.csv
    results/tables/high_order_threshold_selection.csv
    results/tables/high_order_comparison_summary.json
    results/figures/high_order_comparison_plots/
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

from src.evaluation.evaluate_dataset1 import (
    DatasetEvaluationResult,
    apply_persistence_alarm,
    build_evaluation_dataloader,
    collect_model_predictions,
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


# -------------------------------------------------------------------------------------------------
# Locked Step-17 variants
# -------------------------------------------------------------------------------------------------

HIGH_ORDER_COMPARISON_VARIANTS: Tuple[str, ...] = (
    "H0_no_feature_no_model_high_order",
    "H1_feature_high_order_only",
    "H2_model_high_order_only",
    "H3_full_feature_model_high_order",
)


HIGH_ORDER_VARIANT_METADATA: Dict[str, Dict[str, Any]] = {
    "H0_no_feature_no_model_high_order": {
        "short_name": "H0",
        "display_name": "H0: no feature HO + no model HO",
        "feature_high_order": False,
        "model_high_order": False,
        "liquid_temporal": False,
        "description": "Low-order features and low-order/non-liquid model.",
    },
    "H1_feature_high_order_only": {
        "short_name": "H1",
        "display_name": "H1: feature HO only",
        "feature_high_order": True,
        "model_high_order": False,
        "liquid_temporal": False,
        "description": "High-order engineered features, but no model high-order dynamics.",
    },
    "H2_model_high_order_only": {
        "short_name": "H2",
        "display_name": "H2: model HO only",
        "feature_high_order": False,
        "model_high_order": True,
        "liquid_temporal": True,
        "description": "Low-order features with model high-order fusion/dynamics.",
    },
    "H3_full_feature_model_high_order": {
        "short_name": "H3",
        "display_name": "H3: feature HO + model HO",
        "feature_high_order": True,
        "model_high_order": True,
        "liquid_temporal": True,
        "description": "Full proposed model with feature high-order and model high-order.",
    },
}


# -------------------------------------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------------------------------------


@dataclass
class Step17HighOrderComparisonConfig:
    """Runtime config for Step 17 high-order comparison."""

    enabled: bool = True
    experiment_name: str = "step17_high_order_feature_vs_model_comparison"

    variants: List[str] = field(default_factory=lambda: list(HIGH_ORDER_COMPARISON_VARIANTS))

    # This comparison should normally retrain all H variants fairly.
    retrain_policy: str = "always"

    evaluate_dataset1: bool = True
    evaluate_dataset2: bool = True
    evaluate_dataset3: bool = True

    models_dir: str = "results/models/high_order_comparison"
    tables_dir: str = "results/tables/high_order_comparison"
    plots_dir: str = "results/figures/high_order_comparison_plots"

    results_csv: str = "results/tables/high_order_comparison.csv"
    all_splits_csv: str = "results/tables/high_order_comparison_all_splits.csv"
    effects_csv: str = "results/tables/high_order_effects.csv"
    threshold_csv: str = "results/tables/high_order_threshold_selection.csv"
    summary_json: str = "results/tables/high_order_comparison_summary.json"

    save_plots: bool = True
    save_variant_artifacts: bool = True
    print_console_tables: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HighOrderVariantResult:
    """Result payload for one Step-17 high-order variant."""

    variant_name: str
    short_name: str
    display_name: str
    status: str

    feature_high_order: bool
    model_high_order: bool
    liquid_temporal: bool

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
class Step17HighOrderComparisonSummary:
    """Final Step-17 summary."""

    final_status: str
    active_seed: int
    experiment_name: str
    variants: List[str]
    results: List[Dict[str, Any]]
    output_paths: Dict[str, str]
    runtime_seconds: float
    interpretation_rules: Dict[str, Any]

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

    if isinstance(value, float) and not math.isfinite(value):
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


def _variant_metadata(variant_name: str) -> Dict[str, Any]:
    """Return metadata for a high-order comparison variant."""
    if variant_name not in HIGH_ORDER_VARIANT_METADATA:
        return {
            "short_name": str(variant_name),
            "display_name": str(variant_name),
            "feature_high_order": None,
            "model_high_order": None,
            "liquid_temporal": None,
            "description": "Custom high-order comparison variant.",
        }

    return dict(HIGH_ORDER_VARIANT_METADATA[variant_name])


# -------------------------------------------------------------------------------------------------
# Config builders
# -------------------------------------------------------------------------------------------------


def build_step17_high_order_comparison_config(
    config: Mapping[str, Any],
) -> Step17HighOrderComparisonConfig:
    """Build Step-17 config from project config."""
    configured_variants = get_by_path(
        config,
        "experiments.step17_high_order_comparison.variants",
        get_by_path(
            config,
            "experiments.high_order_comparison.variants",
            list(HIGH_ORDER_COMPARISON_VARIANTS),
        ),
    )

    variants = [str(item) for item in list(configured_variants)]

    return Step17HighOrderComparisonConfig(
        enabled=bool(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.enabled",
                get_by_path(config, "experiments.high_order_comparison.enabled", True),
            )
        ),
        experiment_name=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.experiment_name",
                get_by_path(
                    config,
                    "experiments.high_order_comparison.experiment_name",
                    "step17_high_order_feature_vs_model_comparison",
                ),
            )
        ),
        variants=variants,
        retrain_policy=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.retrain_policy",
                get_by_path(config, "experiments.high_order_comparison.retrain_policy", "always"),
            )
        ),
        evaluate_dataset1=bool(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.evaluate_dataset1",
                get_by_path(config, "experiments.high_order_comparison.evaluate_dataset1", True),
            )
        ),
        evaluate_dataset2=bool(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.evaluate_dataset2",
                get_by_path(config, "experiments.high_order_comparison.evaluate_dataset2", True),
            )
        ),
        evaluate_dataset3=bool(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.evaluate_dataset3",
                get_by_path(config, "experiments.high_order_comparison.evaluate_dataset3", True),
            )
        ),
        models_dir=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.models_dir",
                "results/models/high_order_comparison",
            )
        ),
        tables_dir=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.tables_dir",
                "results/tables/high_order_comparison",
            )
        ),
        plots_dir=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.plots_dir",
                "results/figures/high_order_comparison_plots",
            )
        ),
        results_csv=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.results_csv",
                "results/tables/high_order_comparison.csv",
            )
        ),
        all_splits_csv=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.all_splits_csv",
                "results/tables/high_order_comparison_all_splits.csv",
            )
        ),
        effects_csv=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.effects_csv",
                "results/tables/high_order_effects.csv",
            )
        ),
        threshold_csv=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.threshold_csv",
                "results/tables/high_order_threshold_selection.csv",
            )
        ),
        summary_json=str(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.summary_json",
                "results/tables/high_order_comparison_summary.json",
            )
        ),
        save_plots=bool(
            get_by_path(config, "experiments.step17_high_order_comparison.save_plots", True)
        ),
        save_variant_artifacts=bool(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.save_variant_artifacts",
                True,
            )
        ),
        print_console_tables=bool(
            get_by_path(
                config,
                "experiments.step17_high_order_comparison.print_console_tables",
                True,
            )
        ),
    )


def validate_step17_variants(config: Mapping[str, Any], variants: Sequence[str]) -> None:
    """Validate requested variants against model factory."""
    available = get_available_model_variants(config)

    allowed = set(HIGH_ORDER_COMPARISON_VARIANTS)
    allowed.update(str(item) for item in available.get("high_order_comparison", []))
    allowed.update(str(item) for item in available.get("professor_high_order_comparison", []))

    missing = [variant for variant in variants if variant not in allowed]

    if missing:
        raise ValueError(
            "Unknown Step-17 high-order comparison variant(s): "
            f"{missing}. Available high-order variants: {sorted(allowed)}"
        )


def make_high_order_variant_training_config(
    config: Mapping[str, Any],
    step17_config: Step17HighOrderComparisonConfig,
    variant_name: str,
) -> Dict[str, Any]:
    """
    Create a config copy for one Step-17 variant.

    This forces:
    - training.step12.variant_name = variant_name
    - checkpoints under results/models/high_order_comparison/
    - per-variant Step-12 and evaluation artifacts
    """
    cfg = copy.deepcopy(dict(config))

    variant_name = str(variant_name)
    variant_tables_dir = str(Path(step17_config.tables_dir) / variant_name)
    variant_models_dir = str(step17_config.models_dir)

    _set_by_path(cfg, "training.step12.model_name", f"high_order_{variant_name}")
    _set_by_path(cfg, "training.step12.variant_name", variant_name)

    _set_by_path(cfg, "paths.models_dir", variant_models_dir)
    _set_by_path(cfg, "training.checkpointing.best_checkpoint_name", f"{variant_name}_best.pt")
    _set_by_path(cfg, "training.checkpointing.last_checkpoint_name", f"{variant_name}_last.pt")

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


def resolve_high_order_variant_checkpoint_path(
    config: Mapping[str, Any],
    step17_config: Step17HighOrderComparisonConfig,
    variant_name: str,
) -> Path:
    """Resolve expected best checkpoint path for one Step-17 variant."""
    return _project_path(
        config,
        str(Path(step17_config.models_dir) / f"{variant_name}_best.pt"),
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


def evaluate_high_order_variant_split(
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
    """Evaluate one Step-17 variant on one split."""
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
        ensure_dir(prediction_npz_path.parent)
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
    metrics["checkpoint_metadata"] = checkpoint_metadata

    delays = metrics.get("detection_delays") or []
    if isinstance(delays, list):
        metrics["attack_1_delay"] = float(delays[0]) if len(delays) >= 1 else None
        metrics["attack_2_delay"] = float(delays[1]) if len(delays) >= 2 else None

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


def evaluate_high_order_variant_protocol(
    config: Mapping[str, Any],
    step17_config: Step17HighOrderComparisonConfig,
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
    Step-17 evaluation protocol.

    Dataset-1 validation selects theta/Np.
    Dataset-1 test, Dataset-2 external, and Dataset-3 online only apply it.
    """
    variant_tables_dir = _project_path(
        config,
        str(Path(step17_config.tables_dir) / variant_name),
    )
    ensure_dir(variant_tables_dir)

    model, checkpoint, checkpoint_metadata = load_trained_model_for_evaluation(
        config=config,
        checkpoint_path=str(checkpoint_path),
        device=device,
        variant_name=variant_name,
    )

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
    val_npz = variant_tables_dir / "dataset1_val_predictions.npz"

    ensure_dir(threshold_json.parent)
    _save_json_safe(selection.to_dict(), threshold_json)
    pd.DataFrame(selection.candidates).to_csv(threshold_candidates_csv, index=False)
    val_bundle.save_npz(val_npz)

    theta = float(selection.theta)
    persistence = int(selection.persistence)

    dataset1_result: Optional[DatasetEvaluationResult] = None
    dataset2_result: Optional[DatasetEvaluationResult] = None
    dataset3_result: Optional[DatasetEvaluationResult] = None

    if step17_config.evaluate_dataset1:
        dataset1_result = evaluate_high_order_variant_split(
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

    if step17_config.evaluate_dataset2:
        dataset2_result = evaluate_high_order_variant_split(
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

    if step17_config.evaluate_dataset3:
        dataset3_result = evaluate_high_order_variant_split(
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
    item: HighOrderVariantResult,
    split_name: str,
    result: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Convert one split result into a flat CSV row."""
    metrics = _result_metrics(result)

    return {
        "variant_name": item.variant_name,
        "short_name": item.short_name,
        "display_name": item.display_name,
        "feature_high_order": bool(item.feature_high_order),
        "model_high_order": bool(item.model_high_order),
        "liquid_temporal": bool(item.liquid_temporal),
        "split": split_name,
        "theta": item.selected_theta,
        "persistence": item.selected_persistence,
        "val_selected_f1": item.selected_validation_f1,
        "val_selected_auprc": item.selected_validation_auprc,
        "val_selected_auroc": item.selected_validation_auroc,
        "val_selected_fpr": item.selected_validation_fpr,
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
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "tn": metrics.get("tn"),
        "fn": metrics.get("fn"),
        "checkpoint_path": result.get("checkpoint_path") if isinstance(result, Mapping) else None,
        "trained_from_scratch": bool(item.trained_from_scratch),
        "threshold_selected_on_dataset1_validation_only": True,
        "dataset2_used_for_tuning": False,
        "dataset3_used_for_tuning": False,
        "uses_same_xi_framework": True,
        "raw_shortcut_columns_used": False,
    }


def build_high_order_comparison_tables(
    results: Sequence[HighOrderVariantResult],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Dataset-1 table, all-splits table, and threshold table."""
    dataset1_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []

    for item in results:
        split_payloads = [
            ("Dataset-1 Test", item.dataset1_result),
            ("Dataset-2 External", item.dataset2_result),
            ("Dataset-3 Online", item.dataset3_result),
        ]

        for split_name, split_result in split_payloads:
            if split_result is None:
                continue

            row = _split_result_to_row(
                item=item,
                split_name=split_name,
                result=split_result,
            )
            all_rows.append(row)

            if split_name == "Dataset-1 Test":
                dataset1_rows.append(row)

        threshold_rows.append(
            {
                "variant_name": item.variant_name,
                "short_name": item.short_name,
                "display_name": item.display_name,
                "feature_high_order": bool(item.feature_high_order),
                "model_high_order": bool(item.model_high_order),
                "liquid_temporal": bool(item.liquid_temporal),
                "theta": item.selected_theta,
                "persistence": item.selected_persistence,
                "val_f1": item.selected_validation_f1,
                "val_auprc": item.selected_validation_auprc,
                "val_auroc": item.selected_validation_auroc,
                "val_fpr": item.selected_validation_fpr,
                "checkpoint_path": item.checkpoint_path,
                "trained_from_scratch": item.trained_from_scratch,
                "status": item.status,
            }
        )

    return pd.DataFrame(dataset1_rows), pd.DataFrame(all_rows), pd.DataFrame(threshold_rows)


def build_high_order_effect_table(all_splits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build factorial effect table.

    For each split and metric:
        feature_effect_without_model = H1 - H0
        feature_effect_with_model    = H3 - H2
        model_effect_without_feature = H2 - H0
        model_effect_with_feature    = H3 - H1
        interaction_effect           = H3 - H2 - H1 + H0
    """
    metrics = [
        "AUPRC",
        "AUROC",
        "F1",
        "Precision",
        "Recall",
        "FPR",
        "Attack Detection Rate",
        "Detection Delay",
    ]

    needed = {
        "H0": "H0_no_feature_no_model_high_order",
        "H1": "H1_feature_high_order_only",
        "H2": "H2_model_high_order_only",
        "H3": "H3_full_feature_model_high_order",
    }

    rows: List[Dict[str, Any]] = []

    if all_splits_df.empty:
        return pd.DataFrame(rows)

    for split_name in sorted(all_splits_df["split"].dropna().unique()):
        split_df = all_splits_df[all_splits_df["split"] == split_name]

        by_variant: Dict[str, pd.Series] = {}
        for key, variant_name in needed.items():
            sub = split_df[split_df["variant_name"] == variant_name]
            if not sub.empty:
                by_variant[key] = sub.iloc[0]

        if set(by_variant.keys()) != set(needed.keys()):
            continue

        for metric in metrics:
            h0 = _safe_float(by_variant["H0"].get(metric))
            h1 = _safe_float(by_variant["H1"].get(metric))
            h2 = _safe_float(by_variant["H2"].get(metric))
            h3 = _safe_float(by_variant["H3"].get(metric))

            if None in {h0, h1, h2, h3}:
                continue

            rows.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "H0": h0,
                    "H1": h1,
                    "H2": h2,
                    "H3": h3,
                    "feature_effect_without_model_H1_minus_H0": h1 - h0,
                    "feature_effect_with_model_H3_minus_H2": h3 - h2,
                    "model_effect_without_feature_H2_minus_H0": h2 - h0,
                    "model_effect_with_feature_H3_minus_H1": h3 - h1,
                    "interaction_effect_H3_minus_H2_minus_H1_plus_H0": h3 - h2 - h1 + h0,
                    "higher_is_better": metric not in {"FPR", "Detection Delay"},
                }
            )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------------------
# Console and plots
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


def print_high_order_console_tables(
    dataset1_df: pd.DataFrame,
    all_splits_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    effects_df: pd.DataFrame,
) -> None:
    """Print important Step-17 tables to console."""
    print("=" * 120)
    print("STEP 17 HIGH-ORDER COMPARISON — THRESHOLD SELECTION")
    print("=" * 120)

    if threshold_df.empty:
        print("No threshold rows available.")
    else:
        print(
            f"{'Variant':34s} | {'Feature HO':>10s} | {'Model HO':>8s} | "
            f"{'theta':>6s} | {'Np':>3s} | {'Val F1':>8s} | {'Val AUPRC':>10s}"
        )
        print("-" * 120)

        for _, row in threshold_df.iterrows():
            print(
                f"{str(row.get('short_name')) + ' ' + str(row.get('display_name')):34s} | "
                f"{str(bool(row.get('feature_high_order'))):>10s} | "
                f"{str(bool(row.get('model_high_order'))):>8s} | "
                f"{_format_metric(row.get('theta'), 2):>6s} | "
                f"{str(int(row.get('persistence')) if pd.notna(row.get('persistence')) else 'NA'):>3s} | "
                f"{_format_metric(row.get('val_f1')):>8s} | "
                f"{_format_metric(row.get('val_auprc')):>10s}"
            )

    print("=" * 120)
    print("STEP 17 HIGH-ORDER COMPARISON — DATASET-1 TEST")
    print("=" * 120)

    if dataset1_df.empty:
        print("No Dataset-1 rows available.")
    else:
        print(
            f"{'Variant':34s} | {'AUPRC':>8s} | {'AUROC':>8s} | {'F1':>8s} | "
            f"{'FPR':>8s} | {'ADR':>8s} | {'Delay':>8s}"
        )
        print("-" * 120)

        for _, row in dataset1_df.iterrows():
            print(
                f"{str(row.get('short_name')) + ' ' + str(row.get('display_name')):34s} | "
                f"{_format_metric(row.get('AUPRC')):>8s} | "
                f"{_format_metric(row.get('AUROC')):>8s} | "
                f"{_format_metric(row.get('F1')):>8s} | "
                f"{_format_metric(row.get('FPR')):>8s} | "
                f"{_format_metric(row.get('Attack Detection Rate')):>8s} | "
                f"{_format_metric(row.get('Detection Delay')):>8s}"
            )

    print("=" * 120)
    print("STEP 17 HIGH-ORDER COMPARISON — ALL SPLITS")
    print("=" * 120)

    if all_splits_df.empty:
        print("No all-splits rows available.")
    else:
        print(
            f"{'Variant':34s} | {'Split':20s} | {'AUPRC':>8s} | {'AUROC':>8s} | "
            f"{'F1':>8s} | {'FPR':>8s} | {'ADR':>8s} | {'Delay':>8s}"
        )
        print("-" * 120)

        for _, row in all_splits_df.iterrows():
            print(
                f"{str(row.get('short_name')) + ' ' + str(row.get('display_name')):34s} | "
                f"{str(row.get('split')):20s} | "
                f"{_format_metric(row.get('AUPRC')):>8s} | "
                f"{_format_metric(row.get('AUROC')):>8s} | "
                f"{_format_metric(row.get('F1')):>8s} | "
                f"{_format_metric(row.get('FPR')):>8s} | "
                f"{_format_metric(row.get('Attack Detection Rate')):>8s} | "
                f"{_format_metric(row.get('Detection Delay')):>8s}"
            )

    if not effects_df.empty:
        print("=" * 120)
        print("STEP 17 FACTORIAL EFFECTS — AUPRC/F1 ONLY")
        print("=" * 120)
        view = effects_df[effects_df["metric"].isin(["AUPRC", "F1"])]

        for _, row in view.iterrows():
            print(
                f"{row['split']:20s} | {row['metric']:6s} | "
                f"Feature(no model)={_format_metric(row['feature_effect_without_model_H1_minus_H0'])} | "
                f"Feature(with model)={_format_metric(row['feature_effect_with_model_H3_minus_H2'])} | "
                f"Model(no feature)={_format_metric(row['model_effect_without_feature_H2_minus_H0'])} | "
                f"Model(with feature)={_format_metric(row['model_effect_with_feature_H3_minus_H1'])} | "
                f"Interaction={_format_metric(row['interaction_effect_H3_minus_H2_minus_H1_plus_H0'])}"
            )

    print("=" * 120)


def save_high_order_plots(all_splits_df: pd.DataFrame, output_dir: Path) -> List[str]:
    """Save simple high-order comparison plots."""
    saved: List[str] = []

    if all_splits_df.empty:
        return saved

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping Step-17 plots because matplotlib is unavailable: {exc}")
        return saved

    ensure_dir(output_dir)

    plot_specs = [
        ("AUPRC", "AUPRC by high-order comparison variant", "auprc"),
        ("AUROC", "AUROC by high-order comparison variant", "auroc"),
        ("F1", "F1 by high-order comparison variant", "f1"),
        ("FPR", "FPR by high-order comparison variant", "fpr"),
    ]

    for split_name in sorted(all_splits_df["split"].dropna().unique()):
        split_df = all_splits_df[all_splits_df["split"] == split_name].copy()
        split_df = split_df.sort_values("short_name")

        labels = split_df["short_name"].astype(str).tolist()
        safe_split = str(split_name).lower().replace(" ", "_").replace("-", "_")

        for metric, title, stem in plot_specs:
            if metric not in split_df.columns:
                continue

            values = pd.to_numeric(split_df[metric], errors="coerce").to_numpy(dtype=float)

            fig = plt.figure(figsize=(9, 5))
            ax = fig.add_subplot(111)
            ax.bar(labels, values)
            ax.set_title(f"{title} — {split_name}")
            ax.set_xlabel("High-order variant")
            ax.set_ylabel(metric)
            fig.tight_layout()

            png_path = output_dir / f"{safe_split}_{stem}.png"
            pdf_path = output_dir / f"{safe_split}_{stem}.pdf"

            fig.savefig(png_path, dpi=180)
            fig.savefig(pdf_path)
            plt.close(fig)

            saved.extend([str(png_path), str(pdf_path)])

    return saved


# -------------------------------------------------------------------------------------------------
# Main Step-17 runner
# -------------------------------------------------------------------------------------------------


def run_step17_high_order_comparison(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> Step17HighOrderComparisonSummary:
    """
    Run Step 17 high-order comparison.

    This trains/reuses four factorial variants and evaluates each under the same protocol.
    """
    start_time = time.perf_counter()

    step17_config = build_step17_high_order_comparison_config(config)
    variants = list(step17_config.variants)

    validate_step17_variants(config, variants)

    device_info = setup_device_from_config(config, verbose=True)
    device = device_info.device

    models_dir = _project_path(config, step17_config.models_dir)
    tables_dir = _project_path(config, step17_config.tables_dir)
    plots_dir = _project_path(config, step17_config.plots_dir)

    ensure_dir(models_dir)
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    print("=" * 120)
    print("STEP 17 HIGH-ORDER FEATURE VS MODEL COMPARISON START")
    print("=" * 120)
    print(f"Experiment name       : {step17_config.experiment_name}")
    print(f"Active seed           : {active_seed}")
    print(f"Device                : {device}")
    print(f"Variants              : {variants}")
    print(f"Retrain policy        : {step17_config.retrain_policy}")
    print(f"Models dir            : {models_dir}")
    print(f"Tables dir            : {tables_dir}")
    print("Question              : feature high-order vs model high-order?")
    print("Design                : 2x2 factorial H0/H1/H2/H3.")
    print("Training rule         : each variant trained on Dataset-1 train only.")
    print("Threshold rule        : theta/Np selected on Dataset-1 validation only.")
    print("External/online rule  : Dataset-2 and Dataset-3 are never used for tuning.")
    print("=" * 120)

    if str(step17_config.retrain_policy).lower().strip() != "always":
        print(
            "WARNING: Step 17 usually should use retrain_policy='always' for a fair "
            "feature/model high-order comparison."
        )

    variant_results: List[HighOrderVariantResult] = []

    for variant_index, variant_name in enumerate(variants, start=1):
        variant_start = time.perf_counter()
        meta = _variant_metadata(variant_name)

        short_name = str(meta["short_name"])
        display_name = str(meta["display_name"])

        print("=" * 120)
        print(f"STEP 17 VARIANT {variant_index}/{len(variants)}: {short_name} — {variant_name}")
        print("=" * 120)
        print(f"Display name       : {display_name}")
        print(f"Feature high-order : {meta['feature_high_order']}")
        print(f"Model high-order   : {meta['model_high_order']}")
        print(f"Liquid temporal    : {meta['liquid_temporal']}")
        print(f"Description        : {meta['description']}")
        print("=" * 120)

        variant_config = make_high_order_variant_training_config(
            config=config,
            step17_config=step17_config,
            variant_name=variant_name,
        )

        variant_tables_dir = _project_path(
            variant_config,
            str(Path(step17_config.tables_dir) / variant_name),
        )
        ensure_dir(variant_tables_dir)

        checkpoint_path = resolve_high_order_variant_checkpoint_path(
            config=variant_config,
            step17_config=step17_config,
            variant_name=variant_name,
        )

        trained_from_scratch = False
        training_summary: Optional[Dict[str, Any]] = None

        try:
            retrain_policy = str(step17_config.retrain_policy).lower().strip()
            checkpoint_exists = checkpoint_path.exists()

            if retrain_policy == "reuse_if_exists" and checkpoint_exists:
                print(f"Reusing existing Step-17 checkpoint: {checkpoint_path}")
            elif retrain_policy == "never":
                if not checkpoint_exists:
                    raise FileNotFoundError(
                        f"retrain_policy='never' but checkpoint is missing: {checkpoint_path}"
                    )
                print(f"Using existing Step-17 checkpoint: {checkpoint_path}")
            else:
                print(f"Training Step-17 variant from scratch: {variant_name}")
                print(f"Checkpoint target: {checkpoint_path}")

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
                evaluate_high_order_variant_protocol(
                    config=variant_config,
                    step17_config=step17_config,
                    variant_name=variant_name,
                    display_name=display_name,
                    checkpoint_path=checkpoint_path,
                    active_seed=active_seed,
                    device=device,
                )
            )

            selected_candidate = threshold_payload.get("selected_candidate", {})
            variant_summary_path = variant_tables_dir / "variant_summary.json"

            artifact_paths = {
                "variant_summary_json": str(variant_summary_path),
                "checkpoint_path": str(checkpoint_path),
                "variant_tables_dir": str(variant_tables_dir),
                "threshold_selection_json": str(variant_tables_dir / "threshold_selection.json"),
                "threshold_candidates_csv": str(variant_tables_dir / "threshold_candidates.csv"),
            }

            result_payload = HighOrderVariantResult(
                variant_name=variant_name,
                short_name=short_name,
                display_name=display_name,
                status="PASSED",
                feature_high_order=bool(meta["feature_high_order"]),
                model_high_order=bool(meta["model_high_order"]),
                liquid_temporal=bool(meta["liquid_temporal"]),
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
            print(f"STEP 17 VARIANT SUMMARY: {short_name} — {display_name}")
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
                        "Model": f"{short_name} | Dataset-1 Test",
                        **extract_primary_metrics(dataset1_result.metrics),
                    }
                )

            if dataset2_result is not None:
                split_rows.append(
                    {
                        "Model": f"{short_name} | Dataset-2 External",
                        **extract_primary_metrics(dataset2_result.metrics),
                    }
                )

            if dataset3_result is not None:
                split_rows.append(
                    {
                        "Model": f"{short_name} | Dataset-3 Online",
                        **extract_primary_metrics(dataset3_result.metrics),
                    }
                )

            if split_rows:
                print_primary_metric_table(
                    title=f"STEP 17 PRIMARY METRICS — {short_name}",
                    rows=split_rows,
                    model_key="Model",
                )

        except Exception as exc:
            result_payload = HighOrderVariantResult(
                variant_name=variant_name,
                short_name=short_name,
                display_name=display_name,
                status="FAILED",
                feature_high_order=bool(meta.get("feature_high_order", False)),
                model_high_order=bool(meta.get("model_high_order", False)),
                liquid_temporal=bool(meta.get("liquid_temporal", False)),
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
            print(f"STEP 17 VARIANT FAILED: {variant_name}")
            print("=" * 120)
            print(str(exc))
            print("=" * 120)

        variant_results.append(result_payload)

    dataset1_df, all_splits_df, threshold_df = build_high_order_comparison_tables(
        variant_results
    )
    effects_df = build_high_order_effect_table(all_splits_df)

    results_csv = _project_path(config, step17_config.results_csv)
    all_splits_csv = _project_path(config, step17_config.all_splits_csv)
    effects_csv = _project_path(config, step17_config.effects_csv)
    threshold_csv = _project_path(config, step17_config.threshold_csv)
    summary_json = _project_path(config, step17_config.summary_json)

    ensure_dir(results_csv.parent)
    ensure_dir(all_splits_csv.parent)
    ensure_dir(effects_csv.parent)
    ensure_dir(threshold_csv.parent)
    ensure_dir(summary_json.parent)

    dataset1_df.to_csv(results_csv, index=False)
    all_splits_df.to_csv(all_splits_csv, index=False)
    effects_df.to_csv(effects_csv, index=False)
    threshold_df.to_csv(threshold_csv, index=False)

    saved_plots: List[str] = []
    if step17_config.save_plots:
        saved_plots = save_high_order_plots(all_splits_df, plots_dir)

    if step17_config.print_console_tables:
        print_high_order_console_tables(
            dataset1_df=dataset1_df,
            all_splits_df=all_splits_df,
            threshold_df=threshold_df,
            effects_df=effects_df,
        )

    final_status = "PASSED" if all(item.status == "PASSED" for item in variant_results) else "FAILED"

    output_paths = {
        "models_dir": str(models_dir),
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
        "high_order_comparison_csv": str(results_csv),
        "high_order_comparison_all_splits_csv": str(all_splits_csv),
        "high_order_effects_csv": str(effects_csv),
        "high_order_threshold_selection_csv": str(threshold_csv),
        "high_order_comparison_summary_json": str(summary_json),
    }

    for index, path in enumerate(saved_plots):
        output_paths[f"plot_{index:02d}"] = str(path)

    summary = Step17HighOrderComparisonSummary(
        final_status=final_status,
        active_seed=int(active_seed),
        experiment_name=step17_config.experiment_name,
        variants=list(variants),
        results=[item.to_dict() for item in variant_results],
        output_paths=output_paths,
        runtime_seconds=float(time.perf_counter() - start_time),
        interpretation_rules={
            "question_answered": "feature high-order vs model high-order",
            "factorial_design": "2x2 H0/H1/H2/H3",
            "H0": "no feature high-order, no model high-order",
            "H1": "feature high-order only",
            "H2": "model high-order only",
            "H3": "feature high-order + model high-order",
            "feature_effect_without_model": "H1 - H0",
            "feature_effect_with_model": "H3 - H2",
            "model_effect_without_feature": "H2 - H0",
            "model_effect_with_feature": "H3 - H1",
            "interaction_effect": "H3 - H2 - H1 + H0",
            "all_variants_trained_on_dataset1_train_only": True,
            "theta_np_selected_on_dataset1_validation_only": True,
            "dataset1_test_not_used_for_threshold_selection": True,
            "dataset2_external_not_used_for_tuning": True,
            "dataset3_online_not_used_for_tuning": True,
            "same_xi_framework": True,
            "raw_shortcut_columns_used": False,
            "not_frozen_intervention_ablation": True,
        },
    )

    _save_json_safe(summary.to_dict(), summary_json)

    print("=" * 120)
    print("STEP 17 HIGH-ORDER FEATURE VS MODEL COMPARISON SUMMARY")
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
        raise RuntimeError(f"Step 17 high-order comparison failed for variants: {failed}")

    return summary


# Compatibility aliases for main.py.
run_high_order_comparison_experiment = run_step17_high_order_comparison
run_step17_high_order_experiment = run_step17_high_order_comparison


__all__ = [
    "HIGH_ORDER_COMPARISON_VARIANTS",
    "HIGH_ORDER_VARIANT_METADATA",
    "Step17HighOrderComparisonConfig",
    "HighOrderVariantResult",
    "Step17HighOrderComparisonSummary",
    "build_step17_high_order_comparison_config",
    "validate_step17_variants",
    "make_high_order_variant_training_config",
    "run_step17_high_order_comparison",
    "run_high_order_comparison_experiment",
    "run_step17_high_order_experiment",
]