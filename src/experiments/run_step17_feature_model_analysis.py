"""
Step 17 final feature/model high-order analysis.

This file contains the final professor-facing Step 17 design:

Step 17A — Feature-group intervention through the complete Proposed model
-----------------------------------------------------------------------
Question:
    Which engineered GNSS evidence features does the trained Proposed model
    actually use to detect GPS spoofing?

Protocol:
    - Use the official trained Proposed checkpoint: results/models/proposed_best.pt
    - Do not retrain.
    - Use the official Step-13 validation-selected theta/Np.
    - Mask one xi feature group at evaluation time.
    - Evaluate Dataset-1 test, Dataset-2 external, Dataset-3 online.

Important:
    The Full row in Step 17A should match the main Proposed result because it
    uses the same checkpoint and same theta/Np.

Step 17B — Kirchhoff/model high-order structure comparison
----------------------------------------------------------
Question:
    Given the same full xi feature evidence, why use Kirchhoff/high-order
    model structure?

Protocol:
    - All variants use full xi features.
    - K0/K1/K2 are trained fairly from scratch or reused depending on policy.
    - K3 reuses the official Proposed checkpoint: results/models/proposed_best.pt
    - K3 is not retrained, so it matches the main Proposed result.
    - Threshold/Np for K0/K1/K2 are selected on Dataset-1 validation only.
    - K3 uses the official Step-13 validation-selected theta/Np.
    - Dataset-2 and Dataset-3 are evaluation-only.

Outputs:
    results/tables/step17_feature_model_analysis/
    results/models/step17b_kirchhoff_structure/
    results/figures/step17_feature_model_analysis/
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
from torch import nn

from src.evaluation.evaluate_dataset1 import (
    DatasetEvaluationResult,
    apply_persistence_alarm,
    build_evaluation_dataloader,
    collect_model_predictions,
    evaluate_bundle_with_threshold,
    select_threshold_and_persistence,
)
from src.evaluation.result_tables import extract_primary_metrics, print_primary_metric_table
from src.models.model_factory import (
    FeatureHighOrderInputMaskWrapper,
    KIRCHHOFF_STRUCTURE_COMPARISON_NAMES,
    build_model,
)
from src.training.trainer import proposed_best_checkpoint_path, run_step12_training_protocol
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir


# ======================================================================================
# Locked variants
# ======================================================================================

STEP17A_FEATURE_VARIANTS: Tuple[str, ...] = (
    "full_features",
    "no_eta",
    "no_eta_dot",
    "no_eta_ddot",
    "no_q",
    "no_accum_log",
)

STEP17B_KIRCHHOFF_VARIANTS: Tuple[str, ...] = (
    "K0_full_features_simple_model",
    "K1_full_features_kirchhoff_only",
    "K2_full_features_kirchhoff_third_order",
    "K3_official_proposed",
)


# ======================================================================================
# Dataclasses
# ======================================================================================


@dataclass
class Step17AConfig:
    """Config for Step 17A feature-group intervention."""

    enabled: bool = True
    experiment_name: str = "step17a_feature_group_intervention"

    variants: List[str] = field(default_factory=lambda: list(STEP17A_FEATURE_VARIANTS))

    checkpoint_path: Optional[str] = None
    threshold_json: str = "results/tables/proposed_threshold_selection.json"

    evaluate_dataset1: bool = True
    evaluate_dataset2: bool = True
    evaluate_dataset3: bool = True

    tables_dir: str = "results/tables/step17_feature_model_analysis/feature_group_intervention"
    plots_dir: str = "results/figures/step17_feature_model_analysis/feature_group_intervention"

    results_csv: str = "results/tables/step17_feature_model_analysis/step17a_feature_group_intervention_dataset1.csv"
    all_splits_csv: str = "results/tables/step17_feature_model_analysis/step17a_feature_group_intervention_all_splits.csv"
    delta_csv: str = "results/tables/step17_feature_model_analysis/step17a_feature_group_intervention_deltas.csv"
    summary_json: str = "results/tables/step17_feature_model_analysis/step17a_feature_group_intervention_summary.json"

    save_variant_artifacts: bool = True
    save_plots: bool = True
    print_console_tables: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step17BConfig:
    """Config for Step 17B Kirchhoff/model high-order structure comparison."""

    enabled: bool = True
    experiment_name: str = "step17b_kirchhoff_structure_comparison"

    variants: List[str] = field(default_factory=lambda: list(STEP17B_KIRCHHOFF_VARIANTS))

    # K0/K1/K2 policy. K3 always reuses official Proposed checkpoint.
    retrain_policy: str = "always"

    k3_checkpoint_path: Optional[str] = None
    k3_threshold_json: str = "results/tables/proposed_threshold_selection.json"

    evaluate_dataset1: bool = True
    evaluate_dataset2: bool = True
    evaluate_dataset3: bool = True

    models_dir: str = "results/models/step17b_kirchhoff_structure"
    tables_dir: str = "results/tables/step17_feature_model_analysis/kirchhoff_structure"
    plots_dir: str = "results/figures/step17_feature_model_analysis/kirchhoff_structure"

    results_csv: str = "results/tables/step17_feature_model_analysis/step17b_kirchhoff_structure_dataset1.csv"
    all_splits_csv: str = "results/tables/step17_feature_model_analysis/step17b_kirchhoff_structure_all_splits.csv"
    threshold_csv: str = "results/tables/step17_feature_model_analysis/step17b_kirchhoff_structure_threshold_selection.csv"
    summary_json: str = "results/tables/step17_feature_model_analysis/step17b_kirchhoff_structure_summary.json"

    save_variant_artifacts: bool = True
    save_plots: bool = False
    print_console_tables: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step17VariantResult:
    """Result payload for one Step-17 variant."""

    variant_name: str
    display_name: str
    status: str

    analysis_type: str

    checkpoint_path: Optional[str]
    trained_from_scratch: bool
    reused_official_proposed_checkpoint: bool
    training_summary: Optional[Dict[str, Any]]

    selected_theta: Optional[float]
    selected_persistence: Optional[int]
    selected_validation_f1: Optional[float]
    selected_validation_auprc: Optional[float]
    selected_validation_auroc: Optional[float]
    selected_validation_fpr: Optional[float]
    threshold_source: str

    dataset1_result: Optional[Dict[str, Any]]
    dataset2_result: Optional[Dict[str, Any]]
    dataset3_result: Optional[Dict[str, Any]]

    metadata: Dict[str, Any]
    artifact_paths: Dict[str, str]
    runtime_seconds: float
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step17ExperimentSummary:
    """Summary for Step 17A, Step 17B, or combined Step 17."""

    final_status: str
    active_seed: int
    experiment_name: str
    analysis_type: str

    variants: List[str]
    results: List[Dict[str, Any]]

    output_paths: Dict[str, str]
    runtime_seconds: float

    fairness_rules: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step17CombinedSummary:
    """Combined Step 17A + Step 17B summary."""

    final_status: str
    active_seed: int
    step17a_summary: Optional[Dict[str, Any]]
    step17b_summary: Optional[Dict[str, Any]]
    runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ======================================================================================
# Generic utilities
# ======================================================================================


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
    """Convert values to JSON-safe Python objects."""
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
    """Save JSON safely."""
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
    """Read metric safely."""
    if not isinstance(metrics, Mapping):
        return None
    return _safe_float(metrics.get(key))


def _result_metrics(result_dict: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extract metrics from DatasetEvaluationResult.to_dict()."""
    if not isinstance(result_dict, Mapping):
        return {}
    metrics = result_dict.get("metrics", {})
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _result_or_none(result: Optional[DatasetEvaluationResult]) -> Optional[Dict[str, Any]]:
    """Convert optional DatasetEvaluationResult to dict."""
    if result is None:
        return None
    return result.to_dict()


def _format_metric(value: Any, precision: int = 4) -> str:
    """Format metric for console tables."""
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{precision}f}"


def _compute_safety_score(metrics: Mapping[str, Any]) -> Optional[float]:
    """
    Safety score used only as an operational summary metric.

    Formula:
        Safety Score = F1 * ADR * (1 - FPR)

    This rewards detection quality, event detection, and low false-alarm rate.
    It does not replace AUPRC/AUROC/F1; it summarizes deployment reliability.
    """
    f1 = _metric(metrics, "f1")
    fpr = _metric(metrics, "fpr")
    adr = _metric(metrics, "attack_detection_rate")

    if f1 is None or fpr is None or adr is None:
        return None

    return float(f1 * adr * max(0.0, 1.0 - fpr))


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


def _add_false_alarm_event_metrics(
    metrics: Dict[str, Any],
    bundle: Any,
    theta: float,
    persistence: int,
) -> Dict[str, Any]:
    """Add false-alarm event count/details to threshold metrics."""
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

    metrics["safety_score"] = _compute_safety_score(metrics)

    return metrics


def _load_threshold_json(config: Mapping[str, Any], threshold_json_path: str) -> Dict[str, Any]:
    """Load official Step-13 selected theta/Np JSON."""
    path = _project_path(config, threshold_json_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Official threshold JSON not found: {path}\n"
            "Run Step 13 first so proposed_threshold_selection.json exists."
        )

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    theta = (
        payload.get("theta")
        if payload.get("theta") is not None
        else payload.get("selected_threshold")
    )
    persistence = (
        payload.get("persistence")
        if payload.get("persistence") is not None
        else payload.get("selected_persistence")
    )

    if theta is None:
        selected_candidate = payload.get("selected_candidate", {})
        if isinstance(selected_candidate, Mapping):
            theta = selected_candidate.get("theta")

    if persistence is None:
        selected_candidate = payload.get("selected_candidate", {})
        if isinstance(selected_candidate, Mapping):
            persistence = selected_candidate.get("persistence")

    if theta is None or persistence is None:
        raise ValueError(
            f"Could not read theta/persistence from official threshold JSON: {path}"
        )

    return {
        "path": str(path),
        "theta": float(theta),
        "persistence": int(persistence),
        "payload": payload,
    }


def _resolve_official_checkpoint(config: Mapping[str, Any], checkpoint_path: Optional[str]) -> Path:
    """Resolve official Proposed checkpoint path."""
    if checkpoint_path is not None:
        return _project_path(config, str(checkpoint_path))

    return proposed_best_checkpoint_path(config)


def _load_state_dict_into_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: Any,
) -> Dict[str, Any]:
    """Load checkpoint into model with minor DataParallel fallback."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Common fallback if a checkpoint was saved under DataParallel.
        stripped = {}
        for key, value in state_dict.items():
            if str(key).startswith("module."):
                stripped[str(key)[7:]] = value
            else:
                stripped[str(key)] = value
        model.load_state_dict(stripped)

    model.to(device)
    model.eval()

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch") if isinstance(payload, Mapping) else None,
        "active_seed": payload.get("active_seed") if isinstance(payload, Mapping) else None,
    }


# ======================================================================================
# Config builders
# ======================================================================================


def build_step17a_config(config: Mapping[str, Any]) -> Step17AConfig:
    """Build Step 17A config."""
    configured_variants = get_by_path(
        config,
        "experiments.step17a_feature_group_intervention.variants",
        list(STEP17A_FEATURE_VARIANTS),
    )

    return Step17AConfig(
        enabled=bool(
            get_by_path(config, "experiments.step17a_feature_group_intervention.enabled", True)
        ),
        experiment_name=str(
            get_by_path(
                config,
                "experiments.step17a_feature_group_intervention.experiment_name",
                "step17a_feature_group_intervention",
            )
        ),
        variants=[str(item) for item in list(configured_variants)],
        checkpoint_path=get_by_path(
            config,
            "experiments.step17a_feature_group_intervention.checkpoint_path",
            None,
        ),
        threshold_json=str(
            get_by_path(
                config,
                "experiments.step17a_feature_group_intervention.threshold_json",
                "results/tables/proposed_threshold_selection.json",
            )
        ),
        evaluate_dataset1=bool(
            get_by_path(config, "experiments.step17a_feature_group_intervention.evaluate_dataset1", True)
        ),
        evaluate_dataset2=bool(
            get_by_path(config, "experiments.step17a_feature_group_intervention.evaluate_dataset2", True)
        ),
        evaluate_dataset3=bool(
            get_by_path(config, "experiments.step17a_feature_group_intervention.evaluate_dataset3", True)
        ),
        tables_dir=str(
            get_by_path(
                config,
                "experiments.step17a_feature_group_intervention.tables_dir",
                "results/tables/step17_feature_model_analysis/feature_group_intervention",
            )
        ),
        plots_dir=str(
            get_by_path(
                config,
                "experiments.step17a_feature_group_intervention.plots_dir",
                "results/figures/step17_feature_model_analysis/feature_group_intervention",
            )
        ),
        results_csv=str(
            get_by_path(
                config,
                "experiments.step17a_feature_group_intervention.results_csv",
                "results/tables/step17_feature_model_analysis/step17a_feature_group_intervention_dataset1.csv",
            )
        ),
        all_splits_csv=str(
            get_by_path(
                config,
                "experiments.step17a_feature_group_intervention.all_splits_csv",
                "results/tables/step17_feature_model_analysis/step17a_feature_group_intervention_all_splits.csv",
            )
        ),
        delta_csv=str(
            get_by_path(
                config,
                "experiments.step17a_feature_group_intervention.delta_csv",
                "results/tables/step17_feature_model_analysis/step17a_feature_group_intervention_deltas.csv",
            )
        ),
        summary_json=str(
            get_by_path(
                config,
                "experiments.step17a_feature_group_intervention.summary_json",
                "results/tables/step17_feature_model_analysis/step17a_feature_group_intervention_summary.json",
            )
        ),
        save_variant_artifacts=bool(
            get_by_path(config, "experiments.step17a_feature_group_intervention.save_variant_artifacts", True)
        ),
        save_plots=bool(
            get_by_path(config, "experiments.step17a_feature_group_intervention.save_plots", True)
        ),
        print_console_tables=bool(
            get_by_path(config, "experiments.step17a_feature_group_intervention.print_console_tables", True)
        ),
    )


def build_step17b_config(config: Mapping[str, Any]) -> Step17BConfig:
    """Build Step 17B config."""
    configured_variants = get_by_path(
        config,
        "experiments.step17b_kirchhoff_structure_comparison.variants",
        list(STEP17B_KIRCHHOFF_VARIANTS),
    )

    return Step17BConfig(
        enabled=bool(
            get_by_path(config, "experiments.step17b_kirchhoff_structure_comparison.enabled", True)
        ),
        experiment_name=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.experiment_name",
                "step17b_kirchhoff_structure_comparison",
            )
        ),
        variants=[str(item) for item in list(configured_variants)],
        retrain_policy=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.retrain_policy",
                "always",
            )
        ),
        k3_checkpoint_path=get_by_path(
            config,
            "experiments.step17b_kirchhoff_structure_comparison.k3_checkpoint_path",
            None,
        ),
        k3_threshold_json=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.k3_threshold_json",
                "results/tables/proposed_threshold_selection.json",
            )
        ),
        evaluate_dataset1=bool(
            get_by_path(config, "experiments.step17b_kirchhoff_structure_comparison.evaluate_dataset1", True)
        ),
        evaluate_dataset2=bool(
            get_by_path(config, "experiments.step17b_kirchhoff_structure_comparison.evaluate_dataset2", True)
        ),
        evaluate_dataset3=bool(
            get_by_path(config, "experiments.step17b_kirchhoff_structure_comparison.evaluate_dataset3", True)
        ),
        models_dir=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.models_dir",
                "results/models/step17b_kirchhoff_structure",
            )
        ),
        tables_dir=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.tables_dir",
                "results/tables/step17_feature_model_analysis/kirchhoff_structure",
            )
        ),
        plots_dir=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.plots_dir",
                "results/figures/step17_feature_model_analysis/kirchhoff_structure",
            )
        ),
        results_csv=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.results_csv",
                "results/tables/step17_feature_model_analysis/step17b_kirchhoff_structure_dataset1.csv",
            )
        ),
        all_splits_csv=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.all_splits_csv",
                "results/tables/step17_feature_model_analysis/step17b_kirchhoff_structure_all_splits.csv",
            )
        ),
        threshold_csv=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.threshold_csv",
                "results/tables/step17_feature_model_analysis/step17b_kirchhoff_structure_threshold_selection.csv",
            )
        ),
        summary_json=str(
            get_by_path(
                config,
                "experiments.step17b_kirchhoff_structure_comparison.summary_json",
                "results/tables/step17_feature_model_analysis/step17b_kirchhoff_structure_summary.json",
            )
        ),
        save_variant_artifacts=bool(
            get_by_path(config, "experiments.step17b_kirchhoff_structure_comparison.save_variant_artifacts", True)
        ),
        save_plots=bool(
            get_by_path(config, "experiments.step17b_kirchhoff_structure_comparison.save_plots", False)
        ),
        print_console_tables=bool(
            get_by_path(config, "experiments.step17b_kirchhoff_structure_comparison.print_console_tables", True)
        ),
    )


# ======================================================================================
# Variant metadata
# ======================================================================================


def step17a_variant_metadata(config: Mapping[str, Any], variant_name: str) -> Dict[str, Any]:
    """Metadata for Step 17A feature intervention variants."""
    defaults = {
        "full_features": {
            "display_name": "Full features",
            "feature_mask_mode": "all",
            "masked_group": "none",
            "description": "Official Proposed model with all 9 xi features.",
        },
        "no_eta": {
            "display_name": "no residual position eta",
            "feature_mask_mode": "no_eta",
            "masked_group": "eta",
            "description": "Masks residual position features xi_eta_east/north.",
        },
        "no_eta_dot": {
            "display_name": "no residual velocity eta_dot",
            "feature_mask_mode": "no_eta_dot",
            "masked_group": "eta_dot",
            "description": "Masks residual velocity / first derivative features.",
        },
        "no_eta_ddot": {
            "display_name": "no residual acceleration eta_ddot",
            "feature_mask_mode": "no_eta_ddot",
            "masked_group": "eta_ddot",
            "description": "Masks residual acceleration / second derivative features.",
        },
        "no_q": {
            "display_name": "no residual energy q",
            "feature_mask_mode": "no_q",
            "masked_group": "q",
            "description": "Masks Mahalanobis residual energy feature xi_q.",
        },
        "no_accum_log": {
            "display_name": "no accumulation",
            "feature_mask_mode": "no_accum_log",
            "masked_group": "accum_log",
            "description": "Masks weak evidence accumulation feature xi_accum_log.",
        },
    }

    base = dict(defaults.get(variant_name, {}))

    configured = get_by_path(
        config,
        f"model.feature_group_intervention.{variant_name}",
        {},
    )

    if isinstance(configured, Mapping):
        base.update(dict(configured))

    if not base:
        raise ValueError(f"Unknown Step 17A feature variant: {variant_name}")

    return base


def step17b_variant_metadata(variant_name: str) -> Dict[str, Any]:
    """Metadata for Step 17B Kirchhoff/model-structure variants."""
    metadata = {
        "K0_full_features_simple_model": {
            "display_name": "K0: full features + simple GRU",
            "short_name": "K0",
            "kirchhoff": False,
            "third_order": False,
            "liquid": False,
            "description": "Full xi features with simple GRU, no Kirchhoff, no third-order, no liquid dynamics.",
        },
        "K1_full_features_kirchhoff_only": {
            "display_name": "K1: Kirchhoff only",
            "short_name": "K1",
            "kirchhoff": True,
            "third_order": False,
            "liquid": False,
            "description": "Full xi features with Kirchhoff evidence coupling only.",
        },
        "K2_full_features_kirchhoff_third_order": {
            "display_name": "K2: Kirchhoff + third-order",
            "short_name": "K2",
            "kirchhoff": True,
            "third_order": True,
            "liquid": False,
            "description": "Full xi features with Kirchhoff exchange and third-order fusion, no liquid dynamics.",
        },
        "K3_official_proposed": {
            "display_name": "K3: official Proposed",
            "short_name": "K3",
            "kirchhoff": True,
            "third_order": True,
            "liquid": True,
            "description": "Official Proposed model with Kirchhoff, third-order fusion, and liquid dynamics.",
        },
    }

    if variant_name not in metadata:
        raise ValueError(f"Unknown Step 17B Kirchhoff variant: {variant_name}")

    return dict(metadata[variant_name])


# ======================================================================================
# Model loading and evaluation
# ======================================================================================


def load_official_full_model_with_feature_mask(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    feature_mask_mode: str,
    device: Any,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load official full Proposed checkpoint and optionally wrap it with a feature mask.

    Important:
        We load the unwrapped full model first, then apply the wrapper.
        This avoids checkpoint-key mismatch because proposed_best.pt was saved
        from the unwrapped full Proposed model.
    """
    base_config = copy.deepcopy(dict(config))

    # Protect against accidental global masking in config.
    _set_by_path(base_config, "model.proposed.use_feature_high_order", True)
    _set_by_path(base_config, "model.proposed.feature_mask_mode", "all")

    base_model, build_info, _variant_config = build_model(
        config=base_config,
        variant_name="full",
        device=device,
    )

    checkpoint_metadata = _load_state_dict_into_model(
        model=base_model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    mode = str(feature_mask_mode).lower().strip()

    if mode in {"", "all", "none", "full", "full_features", "use_all_features", "no_mask"}:
        model: nn.Module = base_model
    else:
        model = FeatureHighOrderInputMaskWrapper(
            base_model=base_model,
            feature_mask_mode=feature_mask_mode,
        )
        model.to(device)
        model.eval()

    metadata = {
        **checkpoint_metadata,
        "variant_name": "full",
        "model_build_info": build_info.to_dict() if hasattr(build_info, "to_dict") else str(build_info),
        "feature_mask_mode": feature_mask_mode,
        "loaded_as_official_full_then_wrapped": mode not in {"", "all", "none", "full", "full_features", "use_all_features", "no_mask"},
    }

    return model, metadata


def load_variant_model_from_checkpoint(
    config: Mapping[str, Any],
    variant_name: str,
    checkpoint_path: Path,
    device: Any,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build a model variant and load its checkpoint."""
    model, build_info, _variant_config = build_model(
        config=config,
        variant_name=variant_name,
        device=device,
    )

    checkpoint_metadata = _load_state_dict_into_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    metadata = {
        **checkpoint_metadata,
        "variant_name": variant_name,
        "model_build_info": build_info.to_dict() if hasattr(build_info, "to_dict") else str(build_info),
    }

    return model, metadata


def evaluate_loaded_model_split(
    model: nn.Module,
    config: Mapping[str, Any],
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
    """Evaluate an already-loaded model on one split."""
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
        checkpoint_path=str(checkpoint_path),
        model_name=display_name,
    )

    metrics = evaluate_bundle_with_threshold(
        bundle=bundle,
        theta=float(theta),
        persistence=int(persistence),
    )

    metrics = _add_false_alarm_event_metrics(
        metrics=metrics,
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

    metrics["dataset_summary"] = dataset.summary() if hasattr(dataset, "summary") else {}

    return DatasetEvaluationResult(
        model_name=display_name,
        split_name=output_split_name,
        metrics=metrics,
        threshold=float(theta),
        persistence=int(persistence),
        checkpoint_path=str(checkpoint_path),
        prediction_summary=bundle.to_dict_summary(),
        artifact_paths=artifact_paths,
    )


def collect_validation_selection_for_model(
    model: nn.Module,
    config: Mapping[str, Any],
    display_name: str,
    checkpoint_path: Path,
    active_seed: int,
    device: Any,
    output_dir: Path,
    fixed_theta: Optional[float] = None,
    fixed_persistence: Optional[int] = None,
    threshold_source: str = "dataset1_validation_selection",
) -> Dict[str, Any]:
    """
    Collect validation predictions and either:
    - select theta/Np on validation, or
    - evaluate a fixed official theta/Np on validation.
    """
    ensure_dir(output_dir)

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
        checkpoint_path=str(checkpoint_path),
        model_name=display_name,
    )

    val_npz = output_dir / "dataset1_val_predictions.npz"
    val_bundle.save_npz(val_npz)

    if fixed_theta is not None and fixed_persistence is not None:
        metrics = evaluate_bundle_with_threshold(
            bundle=val_bundle,
            theta=float(fixed_theta),
            persistence=int(fixed_persistence),
        )
        metrics = _add_false_alarm_event_metrics(
            metrics=metrics,
            bundle=val_bundle,
            theta=float(fixed_theta),
            persistence=int(fixed_persistence),
        )

        payload = {
            "theta": float(fixed_theta),
            "persistence": int(fixed_persistence),
            "objective": "fixed_official_step13_threshold",
            "monitor_split": "val",
            "selected_metric_value": _metric(metrics, "f1"),
            "selected_candidate": {
                "theta": float(fixed_theta),
                "persistence": int(fixed_persistence),
                **metrics,
            },
            "candidate_count": None,
            "candidates": [],
            "threshold_source": threshold_source,
            "validation_prediction_summary": val_bundle.to_dict_summary(),
            "validation_dataset_summary": val_dataset.summary() if hasattr(val_dataset, "summary") else {},
            "artifact_paths": {
                "dataset1_val_predictions_npz": str(val_npz),
            },
        }
    else:
        selection = select_threshold_and_persistence(
            validation_bundle=val_bundle,
            config=config,
        )

        threshold_json = output_dir / "threshold_selection.json"
        threshold_candidates_csv = output_dir / "threshold_candidates.csv"

        _save_json_safe(selection.to_dict(), threshold_json)
        pd.DataFrame(selection.candidates).to_csv(threshold_candidates_csv, index=False)

        payload = {
            "theta": float(selection.theta),
            "persistence": int(selection.persistence),
            "objective": selection.objective,
            "monitor_split": selection.monitor_split,
            "selected_metric_value": selection.selected_metric_value,
            "selected_candidate": selection.selected_candidate,
            "candidate_count": selection.candidate_count,
            "candidates": selection.candidates,
            "threshold_source": threshold_source,
            "validation_prediction_summary": val_bundle.to_dict_summary(),
            "validation_dataset_summary": val_dataset.summary() if hasattr(val_dataset, "summary") else {},
            "artifact_paths": {
                "threshold_selection_json": str(threshold_json),
                "threshold_candidates_csv": str(threshold_candidates_csv),
                "dataset1_val_predictions_npz": str(val_npz),
            },
        }

    threshold_json = output_dir / "threshold_selection.json"
    if not threshold_json.exists():
        _save_json_safe(payload, threshold_json)
        payload.setdefault("artifact_paths", {})["threshold_selection_json"] = str(threshold_json)

    return payload


def evaluate_loaded_model_official_splits(
    model: nn.Module,
    config: Mapping[str, Any],
    display_name: str,
    checkpoint_path: Path,
    theta: float,
    persistence: int,
    active_seed: int,
    device: Any,
    output_dir: Path,
    evaluate_dataset1: bool = True,
    evaluate_dataset2: bool = True,
    evaluate_dataset3: bool = True,
    save_artifacts: bool = True,
) -> Tuple[
    Optional[DatasetEvaluationResult],
    Optional[DatasetEvaluationResult],
    Optional[DatasetEvaluationResult],
]:
    """Evaluate Dataset-1 test, Dataset-2 external, and Dataset-3 online."""
    ensure_dir(output_dir)

    dataset1_result: Optional[DatasetEvaluationResult] = None
    dataset2_result: Optional[DatasetEvaluationResult] = None
    dataset3_result: Optional[DatasetEvaluationResult] = None

    if evaluate_dataset1:
        dataset1_result = evaluate_loaded_model_split(
            model=model,
            config=config,
            display_name=display_name,
            checkpoint_path=checkpoint_path,
            split_name="test",
            output_split_name="Dataset-1 Test",
            theta=theta,
            persistence=persistence,
            active_seed=active_seed,
            device=device,
            full_sequence=False,
            prediction_npz_path=output_dir / "dataset1_test_predictions.npz" if save_artifacts else None,
        )

    if evaluate_dataset2:
        dataset2_result = evaluate_loaded_model_split(
            model=model,
            config=config,
            display_name=display_name,
            checkpoint_path=checkpoint_path,
            split_name="external",
            output_split_name="Dataset-2 External",
            theta=theta,
            persistence=persistence,
            active_seed=active_seed,
            device=device,
            full_sequence=False,
            prediction_npz_path=output_dir / "dataset2_external_predictions.npz" if save_artifacts else None,
        )

    if evaluate_dataset3:
        dataset3_result = evaluate_loaded_model_split(
            model=model,
            config=config,
            display_name=display_name,
            checkpoint_path=checkpoint_path,
            split_name="online",
            output_split_name="Dataset-3 Online",
            theta=theta,
            persistence=persistence,
            active_seed=active_seed,
            device=device,
            full_sequence=True,
            prediction_npz_path=output_dir / "dataset3_online_predictions.npz" if save_artifacts else None,
            prediction_csv_path=output_dir / "dataset3_online_predictions.csv" if save_artifacts else None,
        )

    return dataset1_result, dataset2_result, dataset3_result


# ======================================================================================
# Table builders
# ======================================================================================


def _split_result_to_common_row(
    item: Step17VariantResult,
    split_name: str,
    split_result: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Convert split result into flat row."""
    metrics = _result_metrics(split_result)

    row = {
        "analysis_type": item.analysis_type,
        "variant_name": item.variant_name,
        "display_name": item.display_name,
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
        "False Alarm Rows": _metric(metrics, "row_level_false_alarms"),
        "False Alarm Events": _metric(metrics, "false_alarm_events"),
        "Attack Detection Rate": _metric(metrics, "attack_detection_rate"),
        "Detection Delay": _metric(metrics, "mean_detection_delay"),
        "Safety Score": _metric(metrics, "safety_score"),
        "Attack-1 Delay": _metric(metrics, "attack_1_delay"),
        "Attack-2 Delay": _metric(metrics, "attack_2_delay"),
        "Runtime": _metric(metrics, "runtime_seconds"),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "tn": metrics.get("tn"),
        "fn": metrics.get("fn"),
        "checkpoint_path": split_result.get("checkpoint_path") if isinstance(split_result, Mapping) else None,
        "trained_from_scratch": item.trained_from_scratch,
        "reused_official_proposed_checkpoint": item.reused_official_proposed_checkpoint,
        "threshold_source": item.threshold_source,
        "status": item.status,
    }

    row.update(item.metadata)

    return row


def build_step17_tables(
    results: Sequence[Step17VariantResult],
    baseline_variant: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build Dataset-1 table, all-splits table, and threshold table.
    """
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

            row = _split_result_to_common_row(
                item=item,
                split_name=split_name,
                split_result=split_result,
            )
            all_rows.append(row)

            if split_name == "Dataset-1 Test":
                dataset1_rows.append(row)

        threshold_rows.append(
            {
                "analysis_type": item.analysis_type,
                "variant_name": item.variant_name,
                "display_name": item.display_name,
                "theta": item.selected_theta,
                "persistence": item.selected_persistence,
                "val_f1": item.selected_validation_f1,
                "val_auprc": item.selected_validation_auprc,
                "val_auroc": item.selected_validation_auroc,
                "val_fpr": item.selected_validation_fpr,
                "threshold_source": item.threshold_source,
                "checkpoint_path": item.checkpoint_path,
                "trained_from_scratch": item.trained_from_scratch,
                "reused_official_proposed_checkpoint": item.reused_official_proposed_checkpoint,
                "status": item.status,
                **item.metadata,
            }
        )

    dataset1_df = pd.DataFrame(dataset1_rows)
    all_splits_df = pd.DataFrame(all_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    if baseline_variant is not None and not all_splits_df.empty:
        all_splits_df = add_delta_columns(
            all_splits_df,
            baseline_variant=baseline_variant,
        )

        dataset1_df = all_splits_df[all_splits_df["split"] == "Dataset-1 Test"].copy()

    return dataset1_df, all_splits_df, threshold_df


def add_delta_columns(df: pd.DataFrame, baseline_variant: str) -> pd.DataFrame:
    """Add metric deltas relative to a baseline variant within each split."""
    df = df.copy()

    metric_columns = [
        "AUPRC",
        "AUROC",
        "F1",
        "Precision",
        "Recall",
        "FPR",
        "False Alarm Rows",
        "False Alarm Events",
        "Attack Detection Rate",
        "Detection Delay",
        "Safety Score",
    ]

    for metric in metric_columns:
        df[f"delta_{metric}_vs_{baseline_variant}"] = np.nan
        df[f"drop_{metric}_vs_{baseline_variant}"] = np.nan

    for split_name in df["split"].dropna().unique():
        split_mask = df["split"] == split_name
        baseline_rows = df[split_mask & (df["variant_name"] == baseline_variant)]

        if baseline_rows.empty:
            continue

        baseline = baseline_rows.iloc[0]

        for metric in metric_columns:
            if metric not in df.columns:
                continue

            base_value = _safe_float(baseline.get(metric))
            if base_value is None:
                continue

            values = pd.to_numeric(df.loc[split_mask, metric], errors="coerce")
            df.loc[split_mask, f"delta_{metric}_vs_{baseline_variant}"] = values - base_value
            df.loc[split_mask, f"drop_{metric}_vs_{baseline_variant}"] = base_value - values

    return df


# ======================================================================================
# Console and plot helpers
# ======================================================================================


def print_step17_console_table(
    title: str,
    dataset1_df: pd.DataFrame,
    all_splits_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> None:
    """Print Step 17 tables."""
    print("=" * 132)
    print(f"{title} — THRESHOLD / OPERATING POINTS")
    print("=" * 132)

    if threshold_df.empty:
        print("No threshold rows available.")
    else:
        print(
            f"{'Variant':42s} | {'theta':>6s} | {'Np':>3s} | "
            f"{'Val F1':>8s} | {'Val AUPRC':>10s} | {'Val AUROC':>10s} | "
            f"{'Val FPR':>8s} | {'Source':22s}"
        )
        print("-" * 132)

        for _, row in threshold_df.iterrows():
            print(
                f"{str(row.get('display_name'))[:42]:42s} | "
                f"{_format_metric(row.get('theta'), 2):>6s} | "
                f"{str(int(row.get('persistence')) if pd.notna(row.get('persistence')) else 'NA'):>3s} | "
                f"{_format_metric(row.get('val_f1')):>8s} | "
                f"{_format_metric(row.get('val_auprc')):>10s} | "
                f"{_format_metric(row.get('val_auroc')):>10s} | "
                f"{_format_metric(row.get('val_fpr')):>8s} | "
                f"{str(row.get('threshold_source'))[:22]:22s}"
            )

    print("=" * 132)
    print(f"{title} — DATASET-1 TEST")
    print("=" * 132)

    if dataset1_df.empty:
        print("No Dataset-1 rows available.")
    else:
        print(
            f"{'Variant':42s} | {'AUPRC':>8s} | {'AUROC':>8s} | {'F1':>8s} | "
            f"{'Prec':>8s} | {'Rec':>8s} | {'FPR':>8s} | {'FAE':>5s} | "
            f"{'ADR':>8s} | {'Delay':>8s} | {'Safety':>8s}"
        )
        print("-" * 132)

        for _, row in dataset1_df.iterrows():
            print(
                f"{str(row.get('display_name'))[:42]:42s} | "
                f"{_format_metric(row.get('AUPRC')):>8s} | "
                f"{_format_metric(row.get('AUROC')):>8s} | "
                f"{_format_metric(row.get('F1')):>8s} | "
                f"{_format_metric(row.get('Precision')):>8s} | "
                f"{_format_metric(row.get('Recall')):>8s} | "
                f"{_format_metric(row.get('FPR')):>8s} | "
                f"{_format_metric(row.get('False Alarm Events'), 0):>5s} | "
                f"{_format_metric(row.get('Attack Detection Rate')):>8s} | "
                f"{_format_metric(row.get('Detection Delay')):>8s} | "
                f"{_format_metric(row.get('Safety Score')):>8s}"
            )

    print("=" * 132)
    print(f"{title} — ALL SPLITS")
    print("=" * 132)

    if all_splits_df.empty:
        print("No all-splits rows available.")
    else:
        print(
            f"{'Variant':42s} | {'Split':20s} | {'AUPRC':>8s} | {'AUROC':>8s} | "
            f"{'F1':>8s} | {'FPR':>8s} | {'FAE':>5s} | {'ADR':>8s} | "
            f"{'Delay':>8s} | {'Safety':>8s}"
        )
        print("-" * 132)

        for _, row in all_splits_df.iterrows():
            print(
                f"{str(row.get('display_name'))[:42]:42s} | "
                f"{str(row.get('split'))[:20]:20s} | "
                f"{_format_metric(row.get('AUPRC')):>8s} | "
                f"{_format_metric(row.get('AUROC')):>8s} | "
                f"{_format_metric(row.get('F1')):>8s} | "
                f"{_format_metric(row.get('FPR')):>8s} | "
                f"{_format_metric(row.get('False Alarm Events'), 0):>5s} | "
                f"{_format_metric(row.get('Attack Detection Rate')):>8s} | "
                f"{_format_metric(row.get('Detection Delay')):>8s} | "
                f"{_format_metric(row.get('Safety Score')):>8s}"
            )

    print("=" * 132)


def save_step17a_feature_waterfall_plots(
    all_splits_df: pd.DataFrame,
    output_dir: Path,
) -> List[str]:
    """Save feature-group intervention degradation plots."""
    saved: List[str] = []

    if all_splits_df.empty:
        return saved

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping Step 17A plots because matplotlib is unavailable: {exc}")
        return saved

    ensure_dir(output_dir)

    for split_name in sorted(all_splits_df["split"].dropna().unique()):
        split_df = all_splits_df[all_splits_df["split"] == split_name].copy()

        if "full_features" not in set(split_df["variant_name"].astype(str)):
            continue

        for metric in ["F1", "AUPRC", "Safety Score"]:
            if metric not in split_df.columns:
                continue

            baseline_row = split_df[split_df["variant_name"] == "full_features"].iloc[0]
            baseline_value = _safe_float(baseline_row.get(metric))

            if baseline_value is None:
                continue

            plot_df = split_df[split_df["variant_name"] != "full_features"].copy()
            plot_df[f"{metric} drop"] = baseline_value - pd.to_numeric(
                plot_df[metric],
                errors="coerce",
            )

            labels = plot_df["display_name"].astype(str).tolist()
            values = plot_df[f"{metric} drop"].to_numpy(dtype=float)

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.bar(labels, values)
            ax.axhline(0.0, linewidth=1)
            ax.set_title(f"{split_name}: {metric} drop after masking each feature group")
            ax.set_ylabel(f"Full - masked {metric}")
            ax.set_xlabel("Feature-group intervention")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()

            safe_split = str(split_name).lower().replace(" ", "_").replace("-", "_")
            safe_metric = str(metric).lower().replace(" ", "_")
            png_path = output_dir / f"{safe_split}_{safe_metric}_feature_drop.png"
            pdf_path = output_dir / f"{safe_split}_{safe_metric}_feature_drop.pdf"

            fig.savefig(png_path, dpi=180)
            fig.savefig(pdf_path)
            plt.close(fig)

            saved.extend([str(png_path), str(pdf_path)])

    return saved


# ======================================================================================
# Step 17A
# ======================================================================================


def run_step17a_feature_group_intervention(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> Step17ExperimentSummary:
    """
    Run Step 17A feature-group intervention through the complete trained model.
    """
    start_time = time.perf_counter()

    step17a_config = build_step17a_config(config)
    variants = list(step17a_config.variants)

    device_info = setup_device_from_config(config, verbose=True)
    device = device_info.device

    checkpoint_path = _resolve_official_checkpoint(
        config=config,
        checkpoint_path=step17a_config.checkpoint_path,
    )

    threshold_info = _load_threshold_json(
        config=config,
        threshold_json_path=step17a_config.threshold_json,
    )
    theta = float(threshold_info["theta"])
    persistence = int(threshold_info["persistence"])

    tables_dir = _project_path(config, step17a_config.tables_dir)
    plots_dir = _project_path(config, step17a_config.plots_dir)

    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    print("=" * 132)
    print("STEP 17A FEATURE-GROUP INTERVENTION START")
    print("=" * 132)
    print(f"Experiment name       : {step17a_config.experiment_name}")
    print(f"Active seed           : {active_seed}")
    print(f"Device                : {device}")
    print(f"Checkpoint            : {checkpoint_path}")
    print(f"Official theta        : {theta}")
    print(f"Official persistence  : {persistence}")
    print(f"Threshold source      : {threshold_info['path']}")
    print(f"Variants              : {variants}")
    print("Rule                  : do not retrain; mask feature group at evaluation time.")
    print("Full row rule         : Full features should match official Proposed.")
    print("=" * 132)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Official Proposed checkpoint not found: {checkpoint_path}\n"
            "Run Step 13 first."
        )

    results: List[Step17VariantResult] = []

    for index, variant_name in enumerate(variants, start=1):
        variant_start = time.perf_counter()
        metadata = step17a_variant_metadata(config, variant_name)

        display_name = str(metadata["display_name"])
        feature_mask_mode = str(metadata["feature_mask_mode"])
        variant_tables_dir = tables_dir / variant_name
        ensure_dir(variant_tables_dir)

        print("=" * 132)
        print(f"STEP 17A VARIANT {index}/{len(variants)}: {variant_name}")
        print("=" * 132)
        print(f"Display name       : {display_name}")
        print(f"Feature mask mode  : {feature_mask_mode}")
        print(f"Masked group       : {metadata.get('masked_group')}")
        print(f"Description        : {metadata.get('description')}")
        print("=" * 132)

        try:
            model, checkpoint_metadata = load_official_full_model_with_feature_mask(
                config=config,
                checkpoint_path=checkpoint_path,
                feature_mask_mode=feature_mask_mode,
                device=device,
            )

            threshold_payload = collect_validation_selection_for_model(
                model=model,
                config=config,
                display_name=display_name,
                checkpoint_path=checkpoint_path,
                active_seed=active_seed,
                device=device,
                output_dir=variant_tables_dir,
                fixed_theta=theta,
                fixed_persistence=persistence,
                threshold_source="official_step13_fixed_theta_np",
            )

            dataset1_result, dataset2_result, dataset3_result = evaluate_loaded_model_official_splits(
                model=model,
                config=config,
                display_name=display_name,
                checkpoint_path=checkpoint_path,
                theta=theta,
                persistence=persistence,
                active_seed=active_seed,
                device=device,
                output_dir=variant_tables_dir,
                evaluate_dataset1=step17a_config.evaluate_dataset1,
                evaluate_dataset2=step17a_config.evaluate_dataset2,
                evaluate_dataset3=step17a_config.evaluate_dataset3,
                save_artifacts=step17a_config.save_variant_artifacts,
            )

            selected_candidate = threshold_payload.get("selected_candidate", {})
            variant_summary_path = variant_tables_dir / "variant_summary.json"

            result_payload = Step17VariantResult(
                variant_name=variant_name,
                display_name=display_name,
                status="PASSED",
                analysis_type="step17a_feature_group_intervention",
                checkpoint_path=str(checkpoint_path),
                trained_from_scratch=False,
                reused_official_proposed_checkpoint=True,
                training_summary=None,
                selected_theta=theta,
                selected_persistence=persistence,
                selected_validation_f1=_safe_float(selected_candidate.get("f1")),
                selected_validation_auprc=_safe_float(selected_candidate.get("auprc")),
                selected_validation_auroc=_safe_float(selected_candidate.get("auroc")),
                selected_validation_fpr=_safe_float(selected_candidate.get("fpr")),
                threshold_source="official_step13_fixed_theta_np",
                dataset1_result=_result_or_none(dataset1_result),
                dataset2_result=_result_or_none(dataset2_result),
                dataset3_result=_result_or_none(dataset3_result),
                metadata={
                    "feature_mask_mode": feature_mask_mode,
                    "masked_group": metadata.get("masked_group"),
                    "same_checkpoint_as_main_proposed": True,
                    "same_threshold_as_main_proposed": True,
                    "model_retrained": False,
                    "checkpoint_metadata": checkpoint_metadata,
                },
                artifact_paths={
                    "variant_summary_json": str(variant_summary_path),
                    "variant_tables_dir": str(variant_tables_dir),
                    "checkpoint_path": str(checkpoint_path),
                    **threshold_payload.get("artifact_paths", {}),
                },
                runtime_seconds=float(time.perf_counter() - variant_start),
                message="",
            )

            _save_json_safe(result_payload.to_dict(), variant_summary_path)

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
                    title=f"STEP 17A PRIMARY METRICS — {display_name}",
                    rows=split_rows,
                    model_key="Model",
                )

        except Exception as exc:
            result_payload = Step17VariantResult(
                variant_name=variant_name,
                display_name=display_name,
                status="FAILED",
                analysis_type="step17a_feature_group_intervention",
                checkpoint_path=str(checkpoint_path),
                trained_from_scratch=False,
                reused_official_proposed_checkpoint=True,
                training_summary=None,
                selected_theta=None,
                selected_persistence=None,
                selected_validation_f1=None,
                selected_validation_auprc=None,
                selected_validation_auroc=None,
                selected_validation_fpr=None,
                threshold_source="official_step13_fixed_theta_np",
                dataset1_result=None,
                dataset2_result=None,
                dataset3_result=None,
                metadata={
                    "feature_mask_mode": feature_mask_mode,
                    "masked_group": metadata.get("masked_group"),
                },
                artifact_paths={},
                runtime_seconds=float(time.perf_counter() - variant_start),
                message=str(exc),
            )

            print("=" * 132)
            print(f"STEP 17A VARIANT FAILED: {variant_name}")
            print("=" * 132)
            print(str(exc))
            print("=" * 132)

        results.append(result_payload)

    dataset1_df, all_splits_df, threshold_df = build_step17_tables(
        results,
        baseline_variant="full_features",
    )

    results_csv = _project_path(config, step17a_config.results_csv)
    all_splits_csv = _project_path(config, step17a_config.all_splits_csv)
    delta_csv = _project_path(config, step17a_config.delta_csv)
    summary_json = _project_path(config, step17a_config.summary_json)

    ensure_dir(results_csv.parent)
    ensure_dir(all_splits_csv.parent)
    ensure_dir(delta_csv.parent)
    ensure_dir(summary_json.parent)

    dataset1_df.to_csv(results_csv, index=False)
    all_splits_df.to_csv(all_splits_csv, index=False)

    delta_columns = [
        column for column in all_splits_df.columns
        if column.startswith("delta_") or column.startswith("drop_")
    ]
    delta_base_columns = [
        "analysis_type",
        "variant_name",
        "display_name",
        "split",
        "feature_mask_mode",
        "masked_group",
    ]
    delta_df = all_splits_df[
        [column for column in delta_base_columns + delta_columns if column in all_splits_df.columns]
    ].copy()
    delta_df.to_csv(delta_csv, index=False)

    saved_plots: List[str] = []
    if step17a_config.save_plots:
        saved_plots = save_step17a_feature_waterfall_plots(
            all_splits_df=all_splits_df,
            output_dir=plots_dir,
        )

    if step17a_config.print_console_tables:
        print_step17_console_table(
            title="STEP 17A FEATURE-GROUP INTERVENTION",
            dataset1_df=dataset1_df,
            all_splits_df=all_splits_df,
            threshold_df=threshold_df,
        )

    final_status = "PASSED" if all(item.status == "PASSED" for item in results) else "FAILED"

    output_paths = {
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
        "step17a_results_csv": str(results_csv),
        "step17a_all_splits_csv": str(all_splits_csv),
        "step17a_delta_csv": str(delta_csv),
        "step17a_summary_json": str(summary_json),
    }

    for plot_index, plot_path in enumerate(saved_plots):
        output_paths[f"plot_{plot_index:02d}"] = str(plot_path)

    summary = Step17ExperimentSummary(
        final_status=final_status,
        active_seed=int(active_seed),
        experiment_name=step17a_config.experiment_name,
        analysis_type="step17a_feature_group_intervention",
        variants=list(variants),
        results=[item.to_dict() for item in results],
        output_paths=output_paths,
        runtime_seconds=float(time.perf_counter() - start_time),
        fairness_rules={
            "same_official_proposed_checkpoint_for_all_feature_interventions": True,
            "no_retraining": True,
            "same_theta_np_as_main_proposed": True,
            "same_9_dimensional_input_shape": True,
            "feature_group_zeroed_at_runtime_only": True,
            "dataset1_validation_threshold_from_step13": True,
            "dataset1_test_not_used_for_threshold_selection": True,
            "dataset2_external_not_used_for_tuning": True,
            "dataset3_online_not_used_for_tuning": True,
            "raw_shortcut_columns_used": False,
        },
    )

    _save_json_safe(summary.to_dict(), summary_json)

    print("=" * 132)
    print("STEP 17A FEATURE-GROUP INTERVENTION SUMMARY")
    print("=" * 132)
    print(f"Final status       : {summary.final_status}")
    print(f"Active seed        : {summary.active_seed}")
    print(f"Variants           : {summary.variants}")
    print(f"Runtime seconds    : {summary.runtime_seconds:.3f}")
    print("Saved outputs:")
    for key, value in summary.output_paths.items():
        print(f"  {key}: {value}")
    print("=" * 132)

    if final_status != "PASSED":
        failed = [item.variant_name for item in results if item.status != "PASSED"]
        raise RuntimeError(f"Step 17A failed for variants: {failed}")

    return summary


# ======================================================================================
# Step 17B
# ======================================================================================


def validate_step17b_variants(variants: Sequence[str]) -> None:
    """Validate Step 17B variants."""
    allowed = set(STEP17B_KIRCHHOFF_VARIANTS)
    allowed.update(str(item) for item in KIRCHHOFF_STRUCTURE_COMPARISON_NAMES)

    missing = [variant for variant in variants if variant not in allowed]

    if missing:
        raise ValueError(
            f"Unknown Step 17B variant(s): {missing}. "
            f"Allowed: {sorted(allowed)}"
        )


def make_step17b_training_config(
    config: Mapping[str, Any],
    step17b_config: Step17BConfig,
    variant_name: str,
) -> Dict[str, Any]:
    """Create per-variant Step-12 training config for K0/K1/K2."""
    cfg = copy.deepcopy(dict(config))

    variant_tables_dir = str(Path(step17b_config.tables_dir) / variant_name)
    variant_models_dir = str(Path(step17b_config.models_dir))

    _set_by_path(cfg, "training.step12.model_name", f"step17b_{variant_name}")
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

    return cfg


def resolve_step17b_checkpoint_path(
    config: Mapping[str, Any],
    step17b_config: Step17BConfig,
    variant_name: str,
) -> Path:
    """Resolve expected checkpoint for K0/K1/K2."""
    return _project_path(
        config,
        str(Path(step17b_config.models_dir) / f"{variant_name}_best.pt"),
    )


def run_step17b_kirchhoff_structure_comparison(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> Step17ExperimentSummary:
    """
    Run Step 17B Kirchhoff/model high-order structure comparison.
    """
    start_time = time.perf_counter()

    step17b_config = build_step17b_config(config)
    variants = list(step17b_config.variants)
    validate_step17b_variants(variants)

    device_info = setup_device_from_config(config, verbose=True)
    device = device_info.device

    models_dir = _project_path(config, step17b_config.models_dir)
    tables_dir = _project_path(config, step17b_config.tables_dir)
    plots_dir = _project_path(config, step17b_config.plots_dir)

    ensure_dir(models_dir)
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    k3_checkpoint_path = _resolve_official_checkpoint(
        config=config,
        checkpoint_path=step17b_config.k3_checkpoint_path,
    )

    k3_threshold_info = _load_threshold_json(
        config=config,
        threshold_json_path=step17b_config.k3_threshold_json,
    )

    print("=" * 132)
    print("STEP 17B KIRCHHOFF / MODEL HIGH-ORDER STRUCTURE COMPARISON START")
    print("=" * 132)
    print(f"Experiment name       : {step17b_config.experiment_name}")
    print(f"Active seed           : {active_seed}")
    print(f"Device                : {device}")
    print(f"Variants              : {variants}")
    print(f"Retrain policy K0-K2  : {step17b_config.retrain_policy}")
    print(f"K3 checkpoint         : {k3_checkpoint_path}")
    print(f"K3 threshold JSON     : {k3_threshold_info['path']}")
    print(f"Models dir            : {models_dir}")
    print(f"Tables dir            : {tables_dir}")
    print("Rule                  : all variants use full xi features.")
    print("K3 rule               : reuse official Proposed checkpoint; do not retrain.")
    print("Threshold rule        : K0-K2 select on Dataset-1 validation; K3 uses official Step-13 theta/Np.")
    print("=" * 132)

    if not k3_checkpoint_path.exists():
        raise FileNotFoundError(
            f"K3 official Proposed checkpoint not found: {k3_checkpoint_path}\n"
            "Run Step 13 first."
        )

    retrain_policy = str(step17b_config.retrain_policy).lower().strip()
    if retrain_policy not in {"always", "reuse_if_exists", "never"}:
        raise ValueError(
            "Invalid Step 17B retrain_policy. Expected: always, reuse_if_exists, never."
        )

    results: List[Step17VariantResult] = []

    for index, variant_name in enumerate(variants, start=1):
        variant_start = time.perf_counter()
        metadata = step17b_variant_metadata(variant_name)
        display_name = str(metadata["display_name"])

        variant_tables_dir = tables_dir / variant_name
        ensure_dir(variant_tables_dir)

        print("=" * 132)
        print(f"STEP 17B VARIANT {index}/{len(variants)}: {variant_name}")
        print("=" * 132)
        print(f"Display name : {display_name}")
        print(f"Kirchhoff    : {metadata['kirchhoff']}")
        print(f"Third-order  : {metadata['third_order']}")
        print(f"Liquid       : {metadata['liquid']}")
        print(f"Description  : {metadata['description']}")
        print("=" * 132)

        trained_from_scratch = False
        reused_official = False
        training_summary: Optional[Dict[str, Any]] = None

        try:
            if variant_name == "K3_official_proposed":
                checkpoint_path = k3_checkpoint_path
                variant_eval_config = copy.deepcopy(dict(config))
                reused_official = True

                theta = float(k3_threshold_info["theta"])
                persistence = int(k3_threshold_info["persistence"])

                model, checkpoint_metadata = load_variant_model_from_checkpoint(
                    config=variant_eval_config,
                    variant_name=variant_name,
                    checkpoint_path=checkpoint_path,
                    device=device,
                )

                threshold_payload = collect_validation_selection_for_model(
                    model=model,
                    config=variant_eval_config,
                    display_name=display_name,
                    checkpoint_path=checkpoint_path,
                    active_seed=active_seed,
                    device=device,
                    output_dir=variant_tables_dir,
                    fixed_theta=theta,
                    fixed_persistence=persistence,
                    threshold_source="official_step13_fixed_theta_np",
                )

            else:
                variant_eval_config = make_step17b_training_config(
                    config=config,
                    step17b_config=step17b_config,
                    variant_name=variant_name,
                )

                checkpoint_path = resolve_step17b_checkpoint_path(
                    config=variant_eval_config,
                    step17b_config=step17b_config,
                    variant_name=variant_name,
                )

                checkpoint_exists = checkpoint_path.exists()

                if retrain_policy == "reuse_if_exists" and checkpoint_exists:
                    print(f"Reusing existing Step 17B checkpoint: {checkpoint_path}")
                elif retrain_policy == "never":
                    if not checkpoint_exists:
                        raise FileNotFoundError(
                            f"retrain_policy='never' but checkpoint missing: {checkpoint_path}"
                        )
                    print(f"Using existing Step 17B checkpoint: {checkpoint_path}")
                else:
                    print(f"Training Step 17B variant from scratch: {variant_name}")
                    print(f"Checkpoint target: {checkpoint_path}")

                    summary = run_step12_training_protocol(
                        config=variant_eval_config,
                        active_seed=active_seed,
                    )

                    training_summary = summary.to_dict()
                    trained_from_scratch = True

                    best_path = training_summary.get("best_checkpoint_path")
                    if best_path:
                        checkpoint_path = Path(str(best_path))

                    if not checkpoint_path.exists():
                        raise FileNotFoundError(
                            f"Expected Step 17B checkpoint not found after training: {checkpoint_path}"
                        )

                model, checkpoint_metadata = load_variant_model_from_checkpoint(
                    config=variant_eval_config,
                    variant_name=variant_name,
                    checkpoint_path=checkpoint_path,
                    device=device,
                )

                threshold_payload = collect_validation_selection_for_model(
                    model=model,
                    config=variant_eval_config,
                    display_name=display_name,
                    checkpoint_path=checkpoint_path,
                    active_seed=active_seed,
                    device=device,
                    output_dir=variant_tables_dir,
                    fixed_theta=None,
                    fixed_persistence=None,
                    threshold_source="dataset1_validation_selection",
                )

                theta = float(threshold_payload["theta"])
                persistence = int(threshold_payload["persistence"])

            dataset1_result, dataset2_result, dataset3_result = evaluate_loaded_model_official_splits(
                model=model,
                config=variant_eval_config,
                display_name=display_name,
                checkpoint_path=checkpoint_path,
                theta=theta,
                persistence=persistence,
                active_seed=active_seed,
                device=device,
                output_dir=variant_tables_dir,
                evaluate_dataset1=step17b_config.evaluate_dataset1,
                evaluate_dataset2=step17b_config.evaluate_dataset2,
                evaluate_dataset3=step17b_config.evaluate_dataset3,
                save_artifacts=step17b_config.save_variant_artifacts,
            )

            selected_candidate = threshold_payload.get("selected_candidate", {})
            variant_summary_path = variant_tables_dir / "variant_summary.json"

            result_payload = Step17VariantResult(
                variant_name=variant_name,
                display_name=display_name,
                status="PASSED",
                analysis_type="step17b_kirchhoff_structure_comparison",
                checkpoint_path=str(checkpoint_path),
                trained_from_scratch=bool(trained_from_scratch),
                reused_official_proposed_checkpoint=bool(reused_official),
                training_summary=training_summary,
                selected_theta=float(theta),
                selected_persistence=int(persistence),
                selected_validation_f1=_safe_float(selected_candidate.get("f1")),
                selected_validation_auprc=_safe_float(selected_candidate.get("auprc")),
                selected_validation_auroc=_safe_float(selected_candidate.get("auroc")),
                selected_validation_fpr=_safe_float(selected_candidate.get("fpr")),
                threshold_source=str(threshold_payload.get("threshold_source")),
                dataset1_result=_result_or_none(dataset1_result),
                dataset2_result=_result_or_none(dataset2_result),
                dataset3_result=_result_or_none(dataset3_result),
                metadata={
                    "short_name": metadata["short_name"],
                    "feature_set": "full_xi",
                    "kirchhoff": bool(metadata["kirchhoff"]),
                    "third_order": bool(metadata["third_order"]),
                    "liquid": bool(metadata["liquid"]),
                    "same_full_xi_features": True,
                    "checkpoint_metadata": checkpoint_metadata,
                    "k3_matches_main_proposed": variant_name == "K3_official_proposed",
                },
                artifact_paths={
                    "variant_summary_json": str(variant_summary_path),
                    "variant_tables_dir": str(variant_tables_dir),
                    "checkpoint_path": str(checkpoint_path),
                    **threshold_payload.get("artifact_paths", {}),
                },
                runtime_seconds=float(time.perf_counter() - variant_start),
                message="",
            )

            _save_json_safe(result_payload.to_dict(), variant_summary_path)

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
                    title=f"STEP 17B PRIMARY METRICS — {display_name}",
                    rows=split_rows,
                    model_key="Model",
                )

        except Exception as exc:
            result_payload = Step17VariantResult(
                variant_name=variant_name,
                display_name=display_name,
                status="FAILED",
                analysis_type="step17b_kirchhoff_structure_comparison",
                checkpoint_path=None,
                trained_from_scratch=bool(trained_from_scratch),
                reused_official_proposed_checkpoint=bool(reused_official),
                training_summary=training_summary,
                selected_theta=None,
                selected_persistence=None,
                selected_validation_f1=None,
                selected_validation_auprc=None,
                selected_validation_auroc=None,
                selected_validation_fpr=None,
                threshold_source="",
                dataset1_result=None,
                dataset2_result=None,
                dataset3_result=None,
                metadata={
                    "short_name": metadata["short_name"],
                    "feature_set": "full_xi",
                    "kirchhoff": bool(metadata["kirchhoff"]),
                    "third_order": bool(metadata["third_order"]),
                    "liquid": bool(metadata["liquid"]),
                },
                artifact_paths={},
                runtime_seconds=float(time.perf_counter() - variant_start),
                message=str(exc),
            )

            print("=" * 132)
            print(f"STEP 17B VARIANT FAILED: {variant_name}")
            print("=" * 132)
            print(str(exc))
            print("=" * 132)

        results.append(result_payload)

    dataset1_df, all_splits_df, threshold_df = build_step17_tables(
        results,
        baseline_variant="K3_official_proposed",
    )

    results_csv = _project_path(config, step17b_config.results_csv)
    all_splits_csv = _project_path(config, step17b_config.all_splits_csv)
    threshold_csv = _project_path(config, step17b_config.threshold_csv)
    summary_json = _project_path(config, step17b_config.summary_json)

    ensure_dir(results_csv.parent)
    ensure_dir(all_splits_csv.parent)
    ensure_dir(threshold_csv.parent)
    ensure_dir(summary_json.parent)

    dataset1_df.to_csv(results_csv, index=False)
    all_splits_df.to_csv(all_splits_csv, index=False)
    threshold_df.to_csv(threshold_csv, index=False)

    if step17b_config.print_console_tables:
        print_step17_console_table(
            title="STEP 17B KIRCHHOFF STRUCTURE COMPARISON",
            dataset1_df=dataset1_df,
            all_splits_df=all_splits_df,
            threshold_df=threshold_df,
        )

    final_status = "PASSED" if all(item.status == "PASSED" for item in results) else "FAILED"

    output_paths = {
        "models_dir": str(models_dir),
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
        "step17b_results_csv": str(results_csv),
        "step17b_all_splits_csv": str(all_splits_csv),
        "step17b_threshold_csv": str(threshold_csv),
        "step17b_summary_json": str(summary_json),
    }

    summary = Step17ExperimentSummary(
        final_status=final_status,
        active_seed=int(active_seed),
        experiment_name=step17b_config.experiment_name,
        analysis_type="step17b_kirchhoff_structure_comparison",
        variants=list(variants),
        results=[item.to_dict() for item in results],
        output_paths=output_paths,
        runtime_seconds=float(time.perf_counter() - start_time),
        fairness_rules={
            "all_variants_use_full_xi_features": True,
            "k0_k1_k2_train_on_dataset1_train_only": True,
            "k0_k1_k2_threshold_selected_on_dataset1_validation_only": True,
            "k3_reuses_official_proposed_checkpoint": True,
            "k3_reuses_official_step13_theta_np": True,
            "dataset1_test_not_used_for_threshold_selection": True,
            "dataset2_external_not_used_for_tuning": True,
            "dataset3_online_not_used_for_tuning": True,
            "raw_shortcut_columns_used": False,
            "safety_score_formula": "F1 * Attack Detection Rate * (1 - FPR)",
        },
    )

    _save_json_safe(summary.to_dict(), summary_json)

    print("=" * 132)
    print("STEP 17B KIRCHHOFF STRUCTURE COMPARISON SUMMARY")
    print("=" * 132)
    print(f"Final status       : {summary.final_status}")
    print(f"Active seed        : {summary.active_seed}")
    print(f"Variants           : {summary.variants}")
    print(f"Runtime seconds    : {summary.runtime_seconds:.3f}")
    print("Saved outputs:")
    for key, value in summary.output_paths.items():
        print(f"  {key}: {value}")
    print("=" * 132)

    if final_status != "PASSED":
        failed = [item.variant_name for item in results if item.status != "PASSED"]
        raise RuntimeError(f"Step 17B failed for variants: {failed}")

    return summary


# ======================================================================================
# Combined Step 17
# ======================================================================================


def run_step17_feature_model_analysis(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> Step17CombinedSummary:
    """
    Run combined final Step 17 analysis:
    - Step 17A feature-group intervention,
    - Step 17B Kirchhoff/model high-order structure comparison.
    """
    start_time = time.perf_counter()

    step17a_config = build_step17a_config(config)
    step17b_config = build_step17b_config(config)

    print("=" * 132)
    print("STEP 17 FINAL FEATURE/MODEL HIGH-ORDER ANALYSIS START")
    print("=" * 132)
    print(f"Active seed     : {active_seed}")
    print(f"Run Step 17A    : {step17a_config.enabled}")
    print(f"Run Step 17B    : {step17b_config.enabled}")
    print("=" * 132)

    step17a_summary: Optional[Step17ExperimentSummary] = None
    step17b_summary: Optional[Step17ExperimentSummary] = None

    if step17a_config.enabled:
        step17a_summary = run_step17a_feature_group_intervention(
            config=config,
            active_seed=active_seed,
        )

    if step17b_config.enabled:
        step17b_summary = run_step17b_kirchhoff_structure_comparison(
            config=config,
            active_seed=active_seed,
        )

    statuses = []
    if step17a_summary is not None:
        statuses.append(step17a_summary.final_status)
    if step17b_summary is not None:
        statuses.append(step17b_summary.final_status)

    final_status = "PASSED" if statuses and all(status == "PASSED" for status in statuses) else "FAILED"

    summary = Step17CombinedSummary(
        final_status=final_status,
        active_seed=int(active_seed),
        step17a_summary=step17a_summary.to_dict() if step17a_summary is not None else None,
        step17b_summary=step17b_summary.to_dict() if step17b_summary is not None else None,
        runtime_seconds=float(time.perf_counter() - start_time),
    )

    combined_summary_path = _project_path(
        config,
        str(
            get_by_path(
                config,
                "experiments.step17_feature_model_analysis.summary_json",
                "results/tables/step17_feature_model_analysis/step17_feature_model_analysis_summary.json",
            )
        ),
    )
    _save_json_safe(summary.to_dict(), combined_summary_path)

    print("=" * 132)
    print("STEP 17 FINAL FEATURE/MODEL HIGH-ORDER ANALYSIS SUMMARY")
    print("=" * 132)
    print(f"Final status    : {summary.final_status}")
    print(f"Active seed     : {summary.active_seed}")
    print(f"Runtime seconds : {summary.runtime_seconds:.3f}")
    print(f"Summary JSON    : {combined_summary_path}")
    print("=" * 132)

    if final_status != "PASSED":
        raise RuntimeError(f"Step 17 final feature/model analysis failed: {final_status}")

    return summary


# Compatibility aliases for main.py.
run_step17_final_feature_model_analysis = run_step17_feature_model_analysis
run_step17a_features = run_step17a_feature_group_intervention
run_step17b_kirchhoff = run_step17b_kirchhoff_structure_comparison


__all__ = [
    "STEP17A_FEATURE_VARIANTS",
    "STEP17B_KIRCHHOFF_VARIANTS",
    "Step17AConfig",
    "Step17BConfig",
    "Step17VariantResult",
    "Step17ExperimentSummary",
    "Step17CombinedSummary",
    "build_step17a_config",
    "build_step17b_config",
    "run_step17a_feature_group_intervention",
    "run_step17b_kirchhoff_structure_comparison",
    "run_step17_feature_model_analysis",
    "run_step17_final_feature_model_analysis",
    "run_step17a_features",
    "run_step17b_kirchhoff",
]