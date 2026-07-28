"""
Dataset-2 external/source-shift evaluation for the full proposed model.

Step 13 responsibilities:
- load trained full proposed checkpoint,
- load real validation-selected theta and N_p from Dataset-1 validation,
- evaluate Dataset-2 external/source-shift split,
- save Dataset-2 external comparison table,
- save prediction artifacts and summary JSON.

Important:
- Dataset-2 is external only.
- Dataset-2 must never be used for training, validation, threshold selection, or tuning.
- This evaluator only applies the threshold selected on Dataset-1 validation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from src.evaluation.evaluate_dataset1 import (
    DatasetEvaluationResult,
    EvaluationPredictionBundle,
    build_evaluation_dataloader,
    collect_model_predictions,
    evaluate_bundle_with_threshold,
    load_trained_model_for_evaluation,
)
from src.evaluation.result_tables import (
    extract_primary_metrics,
    metrics_to_dataset2_row,
    print_primary_metric_table,
    save_dataset2_external_comparison_row,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir, save_json


@dataclass
class SelectedThreshold:
    """Validation-selected threshold and persistence."""

    theta: float
    persistence: int
    source: str
    objective: Optional[str] = None
    selected_metric_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _project_path(config: Mapping[str, Any], path_value: str) -> Path:
    """Resolve project-relative path."""
    return resolve_project_path(config, path_value)


def get_dataset2_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    """Resolve Dataset-2 Step-13 output paths."""
    return {
        "dataset2_table": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.dataset2_external_comparison_csv",
                    "results/tables/dataset2_external_comparison.csv",
                )
            ),
        ),
        "dataset2_summary_json": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.dataset2_proposed_summary_json",
                    "results/tables/dataset2_proposed_summary.json",
                )
            ),
        ),
        "dataset2_predictions_npz": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.dataset2_external_predictions_npz",
                    "results/tables/dataset2_external_predictions.npz",
                )
            ),
        ),
        "threshold_selection_json": _project_path(
            config,
            str(
                get_by_path(
                    config,
                    "paths.proposed_threshold_selection_json",
                    "results/tables/proposed_threshold_selection.json",
                )
            ),
        ),
    }


def load_selected_threshold(
    config: Mapping[str, Any],
    theta: Optional[float] = None,
    persistence: Optional[int] = None,
) -> SelectedThreshold:
    """
    Load theta and N_p selected from Dataset-1 validation.

    Priority:
    1. explicit theta and persistence arguments,
    2. paths.proposed_threshold_selection_json,
    3. error if neither is available.

    This function must never select a threshold on Dataset-2 or Dataset-3.
    """
    if theta is not None and persistence is not None:
        return SelectedThreshold(
            theta=float(theta),
            persistence=int(persistence),
            source="explicit_arguments",
            objective=None,
            selected_metric_value=None,
        )

    paths = get_dataset2_paths(config)
    threshold_path = paths["threshold_selection_json"]

    if not threshold_path.exists():
        raise FileNotFoundError(
            "Validation-selected threshold file was not found. "
            f"Expected: {threshold_path}. "
            "Run Dataset-1 validation threshold selection first."
        )

    with open(threshold_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    # Expected Part-1 threshold file format:
    # {
    #   "theta": ...,
    #   "persistence": ...,
    #   "objective": ...,
    #   "selected_metric_value": ...
    # }
    if "theta" in payload and "persistence" in payload:
        return SelectedThreshold(
            theta=float(payload["theta"]),
            persistence=int(payload["persistence"]),
            source=str(threshold_path),
            objective=payload.get("objective"),
            selected_metric_value=payload.get("selected_metric_value"),
        )

    # Robust fallback if user points to dataset1 summary JSON instead.
    if "threshold_selection" in payload:
        nested = payload["threshold_selection"]

        if "theta" in nested and "persistence" in nested:
            return SelectedThreshold(
                theta=float(nested["theta"]),
                persistence=int(nested["persistence"]),
                source=str(threshold_path),
                objective=nested.get("objective"),
                selected_metric_value=nested.get("selected_metric_value"),
            )

    raise KeyError(
        f"Could not find theta and persistence inside threshold file: {threshold_path}"
    )


def save_dataset2_prediction_bundle(
    bundle: EvaluationPredictionBundle,
    config: Mapping[str, Any],
) -> Path:
    """Save Dataset-2 prediction bundle."""
    paths = get_dataset2_paths(config)
    return bundle.save_npz(paths["dataset2_predictions_npz"])


def run_dataset2_external_evaluation(
    config: Mapping[str, Any],
    active_seed: int = 42,
    checkpoint_path: Optional[str] = None,
    theta: Optional[float] = None,
    persistence: Optional[int] = None,
    model_name: str = "Proposed",
    device: Optional[Any] = None,
) -> DatasetEvaluationResult:
    """
    Evaluate full proposed model on Dataset-2 external/source-shift test.

    Returns:
        DatasetEvaluationResult for Dataset-2 external test.
    """
    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    start_time = time.perf_counter()

    selected_threshold = load_selected_threshold(
        config=config,
        theta=theta,
        persistence=persistence,
    )

    model, checkpoint, checkpoint_metadata = load_trained_model_for_evaluation(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        variant_name="full",
    )

    external_loader, external_dataset = build_evaluation_dataloader(
        config=config,
        split_name="external",
        active_seed=active_seed,
        full_sequence=False,
    )

    print("=" * 100)
    print("STEP 13 DATASET-2 EXTERNAL EVALUATION")
    print("=" * 100)
    print(f"Model                    : {model_name}")
    print(f"Checkpoint               : {checkpoint}")
    print(f"External rows/windows    : {external_dataset.summary()['rows']} / {external_dataset.summary()['windows']}")
    print(f"Using validation theta   : {selected_threshold.theta}")
    print(f"Using validation N_p     : {selected_threshold.persistence}")
    print(f"Threshold source         : {selected_threshold.source}")
    print("Dataset-2 is external only: no tuning, no threshold selection.")
    print("=" * 100)

    external_bundle = collect_model_predictions(
        model=model,
        dataloader=external_loader,
        device=device,
        split_name="external",
        checkpoint_path=str(checkpoint),
        model_name=model_name,
    )

    metrics = evaluate_bundle_with_threshold(
        bundle=external_bundle,
        theta=selected_threshold.theta,
        persistence=selected_threshold.persistence,
    )
    metrics["runtime_seconds"] = float(time.perf_counter() - start_time)

    paths = get_dataset2_paths(config)
    predictions_npz = save_dataset2_prediction_bundle(
        bundle=external_bundle,
        config=config,
    )

    table_row = metrics_to_dataset2_row(
        model_name=model_name,
        metrics=metrics,
        threshold=selected_threshold.theta,
        persistence=selected_threshold.persistence,
        checkpoint_path=str(checkpoint),
        notes=(
            "Full proposed model; theta and N_p selected on Dataset-1 validation only; "
            "Dataset-2 external/source-shift evaluation only."
        ),
    )

    save_dataset2_external_comparison_row(
        output_path=paths["dataset2_table"],
        row=table_row,
    )

    result = DatasetEvaluationResult(
        model_name=model_name,
        split_name="Dataset-2 External",
        metrics=metrics,
        threshold=selected_threshold.theta,
        persistence=selected_threshold.persistence,
        checkpoint_path=str(checkpoint),
        prediction_summary=external_bundle.to_dict_summary(),
        artifact_paths={
            "dataset2_table": str(paths["dataset2_table"]),
            "dataset2_summary_json": str(paths["dataset2_summary_json"]),
            "dataset2_predictions_npz": str(predictions_npz),
            "threshold_selection_json": str(paths["threshold_selection_json"]),
        },
    )

    summary_payload = {
        "result": result.to_dict(),
        "external_prediction_summary": external_bundle.to_dict_summary(),
        "selected_threshold": selected_threshold.to_dict(),
        "checkpoint_metadata": checkpoint_metadata,
        "external_dataset_summary": external_dataset.summary(),
        "leakage_rules": {
            "dataset2_used_for_training": False,
            "dataset2_used_for_validation": False,
            "dataset2_used_for_threshold_selection": False,
            "threshold_selected_on_dataset1_validation_only": True,
            "synthetic_step10_theta_not_used": True,
        },
    }

    save_json(summary_payload, paths["dataset2_summary_json"], indent=2)

    primary_row = {
        "Model": model_name,
        **extract_primary_metrics(metrics),
    }
    print_primary_metric_table(
        title="STEP 13 DATASET-2 EXTERNAL PRIMARY METRICS",
        rows=[primary_row],
        model_key="Model",
    )

    print("Saved Dataset-2 artifacts:")
    for key, value in result.artifact_paths.items():
        print(f"  {key}: {value}")
    print("=" * 100)

    return result


__all__ = [
    "SelectedThreshold",
    "get_dataset2_paths",
    "load_selected_threshold",
    "save_dataset2_prediction_bundle",
    "run_dataset2_external_evaluation",
]