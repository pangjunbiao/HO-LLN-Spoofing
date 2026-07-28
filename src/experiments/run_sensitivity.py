"""
Step 20: Hybrid sensitivity analysis for the proposed AV-GPS spoofing detector.

Scientific protocol
-------------------
This file implements three reviewer-safe sensitivity analyses:

1. hidden_dim sensitivity
   - true model-capacity sensitivity,
   - retrains one full proposed model per hidden_dim value,
   - selects theta and N_p on Dataset-1 validation only,
   - evaluates Dataset-1 test only.

2. theta sensitivity
   - operating-threshold sensitivity,
   - does not retrain because theta is not inside the model,
   - reuses official saved Dataset-1 test probabilities,
   - keeps official persistence fixed.

3. persistence sensitivity
   - causal alarm-confirmation sensitivity,
   - does not retrain because N_p is not inside the model,
   - reuses official saved Dataset-1 test probabilities,
   - keeps official theta fixed.

Excluded
--------
rho is intentionally excluded because it is not currently defined in
model.yaml or training.yaml. It should not be faked.

Outputs
-------
- results/tables/sensitivity_results.csv
- results/tables/sensitivity_summary.json
- results/models/sensitivity/hidden_dim_<value>/
- results/tables/sensitivity/hidden_dim_<value>/
"""

from __future__ import annotations

import copy
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from src.evaluation.evaluate_dataset1 import (
    EvaluationPredictionBundle,
    evaluate_bundle_with_threshold,
    run_dataset1_evaluation,
)
from src.evaluation.result_tables import (
    metrics_to_sensitivity_row,
    print_sensitivity_table,
    save_sensitivity_results_table,
)
from src.training.trainer import (
    proposed_best_checkpoint_path,
    run_step12_training_protocol,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_json


# -------------------------------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------------------------------


def _project_path(config: Mapping[str, Any], path_value: str | Path) -> Path:
    """Resolve project-relative path."""
    return resolve_project_path(config, str(path_value))


def _set_by_path(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set nested dictionary value by dotted path."""
    current: Dict[str, Any] = config

    keys = str(path).split(".")
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def _json_safe(value: Any) -> Any:
    """Convert common numpy/pandas/path objects into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass

    return value


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to finite float."""
    if value is None:
        return default

    try:
        item = float(value)
    except Exception:
        return default

    return item if math.isfinite(item) else default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely convert value to int."""
    if value is None:
        return default

    try:
        return int(value)
    except Exception:
        return default


def _same_float(a: float, b: float, atol: float = 1.0e-12) -> bool:
    """Compare two floats safely."""
    return abs(float(a) - float(b)) <= float(atol)


def _sanitize_token(value: Any) -> str:
    """Return filesystem-safe value token."""
    text = str(value).strip()
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(" ", "_")
    return text


def _as_float_list(values: Sequence[Any], name: str) -> List[float]:
    """Convert sequence to float list."""
    output: List[float] = []

    for value in values:
        item = _safe_float(value)
        if item is None:
            raise ValueError(f"{name} contains non-float value: {value!r}")
        output.append(float(item))

    return output


def _as_int_list(values: Sequence[Any], name: str) -> List[int]:
    """Convert sequence to int list."""
    output: List[int] = []

    for value in values:
        item = _safe_int(value)
        if item is None:
            raise ValueError(f"{name} contains non-int value: {value!r}")
        output.append(int(item))

    return output


def _npz_optional(payload: Any, key: str) -> Any:
    """Safely read optional key from npz payload."""
    if hasattr(payload, "files") and key in payload.files:
        return payload[key]
    return None


def _first_npz_scalar(payload: Any, key: str, default: Any = None) -> Any:
    """Read first scalar from one-element npz array."""
    value = _npz_optional(payload, key)
    if value is None:
        return default

    arr = np.asarray(value)
    if arr.size == 0:
        return default

    item = arr.reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def _cleanup_torch_memory() -> None:
    """Release Python and CUDA memory between sensitivity runs."""
    gc.collect()

    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------------------------------------------------------------------------------
# Step-20 path helpers
# -------------------------------------------------------------------------------------------------


def get_step20_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    """Resolve Step-20 output paths."""
    results_csv = get_by_path(
        config,
        "sensitivity.outputs.results_csv",
        get_by_path(
            config,
            "paths.sensitivity_results_csv",
            "results/tables/sensitivity_results.csv",
        ),
    )

    summary_json = get_by_path(
        config,
        "sensitivity.outputs.summary_json",
        get_by_path(
            config,
            "paths.sensitivity_summary_json",
            "results/tables/sensitivity_summary.json",
        ),
    )

    models_dir = get_by_path(
        config,
        "sensitivity.outputs.models_dir",
        get_by_path(
            config,
            "paths.sensitivity_models_dir",
            "results/models/sensitivity",
        ),
    )

    artifacts_dir = get_by_path(
        config,
        "sensitivity.outputs.artifacts_dir",
        get_by_path(
            config,
            "paths.sensitivity_artifacts_dir",
            "results/tables/sensitivity",
        ),
    )

    return {
        "results_csv": _project_path(config, str(results_csv)),
        "summary_json": _project_path(config, str(summary_json)),
        "models_dir": _project_path(config, str(models_dir)),
        "artifacts_dir": _project_path(config, str(artifacts_dir)),
    }


# -------------------------------------------------------------------------------------------------
# Official prediction / threshold loading for theta and persistence sensitivity
# -------------------------------------------------------------------------------------------------


def load_prediction_bundle_from_npz(
    npz_path: str | Path,
    fallback_split_name: str = "test",
) -> EvaluationPredictionBundle:
    """Load a saved Step-13 Dataset-1 prediction bundle."""
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Dataset-1 test prediction bundle not found: {npz_path}. "
            "Run Step 13 first or point paths.dataset1_test_predictions_npz to the correct official file."
        )

    payload = np.load(npz_path, allow_pickle=True)

    split_name = str(_first_npz_scalar(payload, "split_name", fallback_split_name))
    checkpoint_path = str(_first_npz_scalar(payload, "checkpoint_path", ""))
    model_name = str(_first_npz_scalar(payload, "model_name", "Proposed"))

    return EvaluationPredictionBundle(
        split_name=split_name,
        probabilities=np.asarray(payload["probabilities"], dtype=np.float64).reshape(-1),
        logits=np.asarray(payload["logits"], dtype=np.float64).reshape(-1),
        labels=np.asarray(payload["labels"], dtype=np.int64).reshape(-1),
        valid_mask=np.asarray(payload["valid_mask"], dtype=np.float64).reshape(-1),
        segment_ids=np.asarray(payload["segment_ids"]).astype(str).reshape(-1),
        row_indices=np.asarray(payload["row_indices"], dtype=np.int64).reshape(-1),
        delta_t=np.asarray(payload["delta_t"], dtype=np.float64).reshape(-1),
        checkpoint_path=checkpoint_path if checkpoint_path else None,
        model_name=model_name,
    )


def load_official_selected_operating_point(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Load official theta and persistence from Step-13 threshold-selection JSON.

    If the JSON is missing, fall back to sensitivity YAML defaults.
    """
    threshold_path = _project_path(
        config,
        str(
            get_by_path(
                config,
                "paths.proposed_threshold_selection_json",
                "results/tables/proposed_threshold_selection.json",
            )
        ),
    )

    fallback_theta = _safe_float(
        get_by_path(config, "sensitivity.theta.default", 0.8),
        default=0.8,
    )
    fallback_persistence = _safe_int(
        get_by_path(config, "sensitivity.persistence.default", 7),
        default=7,
    )

    if not threshold_path.exists():
        return {
            "theta": float(fallback_theta),
            "persistence": int(fallback_persistence),
            "source": str(threshold_path),
            "loaded_from_json": False,
            "warning": "Threshold-selection JSON missing; used sensitivity YAML defaults.",
        }

    with open(threshold_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    theta = (
        payload.get("theta")
        or payload.get("selected_theta")
        or payload.get("best_theta")
    )
    persistence = (
        payload.get("persistence")
        or payload.get("selected_persistence")
        or payload.get("best_persistence")
    )

    selected_candidate = payload.get("selected_candidate")
    if isinstance(selected_candidate, Mapping):
        theta = selected_candidate.get("theta", theta)
        persistence = selected_candidate.get("persistence", persistence)

    theta = _safe_float(theta, default=fallback_theta)
    persistence = _safe_int(persistence, default=fallback_persistence)

    if theta is None or persistence is None:
        raise ValueError(
            f"Could not read theta/persistence from threshold JSON: {threshold_path}"
        )

    return {
        "theta": float(theta),
        "persistence": int(persistence),
        "source": str(threshold_path),
        "loaded_from_json": True,
        "warning": None,
    }


def load_official_test_bundle(config: Mapping[str, Any]) -> EvaluationPredictionBundle:
    """Load official saved Dataset-1 test prediction bundle."""
    prediction_path = _project_path(
        config,
        str(
            get_by_path(
                config,
                "sensitivity.inputs.dataset1_test_predictions_npz",
                get_by_path(
                    config,
                    "paths.dataset1_test_predictions_npz",
                    "results/tables/dataset1_test_predictions.npz",
                ),
            )
        ),
    )

    return load_prediction_bundle_from_npz(
        npz_path=prediction_path,
        fallback_split_name="test",
    )


# -------------------------------------------------------------------------------------------------
# YAML sensitivity readers
# -------------------------------------------------------------------------------------------------


def _sensitivity_enabled(config: Mapping[str, Any]) -> bool:
    """Return top-level sensitivity.enabled."""
    return bool(get_by_path(config, "sensitivity.enabled", False))


def _group_enabled(config: Mapping[str, Any], group_name: str) -> bool:
    """Return sensitivity.<group>.enabled."""
    return bool(get_by_path(config, f"sensitivity.{group_name}.enabled", False))


def _group_values(
    config: Mapping[str, Any],
    group_name: str,
    fallback_values: Sequence[Any],
) -> List[Any]:
    """Return configured sensitivity values."""
    values = get_by_path(config, f"sensitivity.{group_name}.values", list(fallback_values))

    if values is None:
        return list(fallback_values)

    if not isinstance(values, list):
        raise TypeError(f"sensitivity.{group_name}.values must be a YAML list.")

    return list(values)


# -------------------------------------------------------------------------------------------------
# Operating-point sensitivity: theta and persistence
# -------------------------------------------------------------------------------------------------


def evaluate_selected_official_row(
    bundle: EvaluationPredictionBundle,
    selected_theta: float,
    selected_persistence: int,
) -> Dict[str, Any]:
    """Evaluate official selected operating point."""
    metrics = evaluate_bundle_with_threshold(
        bundle=bundle,
        theta=float(selected_theta),
        persistence=int(selected_persistence),
    )

    return metrics_to_sensitivity_row(
        sensitivity_parameter="selected",
        sensitivity_value="official",
        metrics=metrics,
        theta=float(selected_theta),
        persistence=int(selected_persistence),
        model_name="Proposed",
        split="Dataset-1 Test",
        is_official_selected=True,
        status="PASSED",
        notes="Official Step-13 validation-selected operating point evaluated on Dataset-1 test.",
    )


def run_theta_sensitivity(
    config: Mapping[str, Any],
    bundle: EvaluationPredictionBundle,
    selected_theta: float,
    selected_persistence: int,
) -> List[Dict[str, Any]]:
    """Run theta sensitivity with persistence fixed."""
    if not _group_enabled(config, "theta"):
        return []

    theta_values = _as_float_list(
        _group_values(config, "theta", [0.5, 0.6, 0.7, 0.8, 0.9]),
        name="sensitivity.theta.values",
    )

    rows: List[Dict[str, Any]] = []

    for theta in theta_values:
        metrics = evaluate_bundle_with_threshold(
            bundle=bundle,
            theta=float(theta),
            persistence=int(selected_persistence),
        )

        rows.append(
            metrics_to_sensitivity_row(
                sensitivity_parameter="theta",
                sensitivity_value=float(theta),
                metrics=metrics,
                theta=float(theta),
                persistence=int(selected_persistence),
                model_name="Proposed",
                split="Dataset-1 Test",
                is_official_selected=_same_float(theta, selected_theta),
                status="PASSED",
                notes=(
                    "Operating-threshold sensitivity. "
                    f"Persistence fixed at official N_p={selected_persistence}. "
                    "No retraining because theta is applied after model probabilities."
                ),
            )
        )

    return rows


def run_persistence_sensitivity(
    config: Mapping[str, Any],
    bundle: EvaluationPredictionBundle,
    selected_theta: float,
    selected_persistence: int,
) -> List[Dict[str, Any]]:
    """Run persistence sensitivity with theta fixed."""
    if not _group_enabled(config, "persistence"):
        return []

    persistence_values = _as_int_list(
        _group_values(config, "persistence", [3, 5, 7, 9, 11]),
        name="sensitivity.persistence.values",
    )

    rows: List[Dict[str, Any]] = []

    for persistence in persistence_values:
        metrics = evaluate_bundle_with_threshold(
            bundle=bundle,
            theta=float(selected_theta),
            persistence=int(persistence),
        )

        rows.append(
            metrics_to_sensitivity_row(
                sensitivity_parameter="persistence",
                sensitivity_value=int(persistence),
                metrics=metrics,
                theta=float(selected_theta),
                persistence=int(persistence),
                model_name="Proposed",
                split="Dataset-1 Test",
                is_official_selected=int(persistence) == int(selected_persistence),
                status="PASSED",
                notes=(
                    "Causal alarm-confirmation sensitivity. "
                    f"Theta fixed at official theta={selected_theta}. "
                    "No retraining because persistence is applied after model probabilities."
                ),
            )
        )

    return rows


# -------------------------------------------------------------------------------------------------
# True model retraining sensitivity: hidden_dim
# -------------------------------------------------------------------------------------------------


def _hidden_dim_scaled_values(hidden_dim: int) -> Dict[str, int]:
    """
    Return consistent internal dimensions for hidden_dim sensitivity.

    The official model has hidden_dim=64 and branch/module dims near 32.
    Therefore branch/module dims are scaled as max(16, hidden_dim // 2).
    """
    hidden = int(hidden_dim)
    branch = max(16, int(hidden // 2))

    return {
        "hidden": hidden,
        "velocity": hidden,
        "fusion": hidden,
        "head": hidden,
        "module_state": branch,
        "branch": branch,
        "conductance": branch,
    }


def make_hidden_dim_sensitivity_config(
    config: Mapping[str, Any],
    hidden_dim: int,
    step20_paths: Mapping[str, Path],
) -> Dict[str, Any]:
    """
    Create an isolated config for one hidden_dim sensitivity run.

    This prevents Step 20 from overwriting official Step-13 artifacts.
    """
    cfg = copy.deepcopy(dict(config))

    hidden_dim = int(hidden_dim)
    token = f"hidden_dim_{_sanitize_token(hidden_dim)}"

    model_dir = Path(step20_paths["models_dir"]) / token
    artifact_dir = Path(step20_paths["artifacts_dir"]) / token

    ensure_dir(model_dir)
    ensure_dir(artifact_dir)

    dims = _hidden_dim_scaled_values(hidden_dim)

    # Full proposed model.
    _set_by_path(cfg, "training.step12.model_name", "proposed")
    _set_by_path(cfg, "training.step12.variant_name", "full")

    # Force isolated checkpoint directory.
    _set_by_path(cfg, "paths.models_dir", str(model_dir))
    _set_by_path(cfg, "training.checkpointing.best_checkpoint_name", "proposed_best.pt")
    _set_by_path(cfg, "training.checkpointing.last_checkpoint_name", "proposed_last.pt")

    # Isolate Step-12 artifacts.
    _set_by_path(
        cfg,
        "paths.step12_training_history_csv",
        str(artifact_dir / "step12_training_history.csv"),
    )
    _set_by_path(
        cfg,
        "paths.step12_training_history_json",
        str(artifact_dir / "step12_training_history.json"),
    )
    _set_by_path(
        cfg,
        "paths.step12_training_summary_json",
        str(artifact_dir / "step12_training_summary.json"),
    )
    _set_by_path(
        cfg,
        "paths.step12_validation_predictions_npz",
        str(artifact_dir / "step12_validation_predictions.npz"),
    )

    # Isolate Step-13 Dataset-1 evaluation artifacts for this retrained model.
    _set_by_path(
        cfg,
        "paths.dataset1_main_comparison_csv",
        str(artifact_dir / "dataset1_main_comparison.csv"),
    )
    _set_by_path(
        cfg,
        "paths.dataset1_proposed_summary_json",
        str(artifact_dir / "dataset1_proposed_summary.json"),
    )
    _set_by_path(
        cfg,
        "paths.proposed_threshold_selection_json",
        str(artifact_dir / "proposed_threshold_selection.json"),
    )
    _set_by_path(
        cfg,
        "paths.proposed_threshold_candidates_csv",
        str(artifact_dir / "proposed_threshold_candidates.csv"),
    )
    _set_by_path(
        cfg,
        "paths.dataset1_val_predictions_npz",
        str(artifact_dir / "dataset1_val_predictions.npz"),
    )
    _set_by_path(
        cfg,
        "paths.dataset1_test_predictions_npz",
        str(artifact_dir / "dataset1_test_predictions.npz"),
    )

    # Main proposed capacity setting.
    _set_by_path(cfg, "model.proposed.hidden_dim", dims["hidden"])
    _set_by_path(cfg, "model.proposed.module_state_dim", dims["module_state"])

    # Temporal block dimensions.
    _set_by_path(cfg, "model.proposed.liquid_second_order.hidden_dim", dims["hidden"])
    _set_by_path(cfg, "model.proposed.liquid_second_order.velocity_dim", dims["velocity"])
    _set_by_path(cfg, "model.proposed.gru.hidden_dim", dims["hidden"])
    _set_by_path(cfg, "model.proposed.simple_first_order.hidden_dim", dims["hidden"])

    # Output head.
    _set_by_path(cfg, "model.proposed.output_head.hidden_dim", dims["head"])

    # Kirchhoff/high-order internal dimensions scaled consistently.
    _set_by_path(cfg, "model.proposed.kirchhoff_high_order.instantaneous_branch_dim", dims["branch"])
    _set_by_path(cfg, "model.proposed.kirchhoff_high_order.evolution_branch_dim", dims["branch"])
    _set_by_path(cfg, "model.proposed.kirchhoff_high_order.persistence_branch_dim", dims["branch"])
    _set_by_path(cfg, "model.proposed.kirchhoff_high_order.conductance_hidden_dim", dims["conductance"])
    _set_by_path(cfg, "model.proposed.kirchhoff_high_order.fusion_dim", dims["fusion"])

    # Store metadata for the summary.
    _set_by_path(cfg, "sensitivity.active_parameter", "hidden_dim")
    _set_by_path(cfg, "sensitivity.active_value", hidden_dim)
    _set_by_path(cfg, "sensitivity.active_model_dir", str(model_dir))
    _set_by_path(cfg, "sensitivity.active_artifact_dir", str(artifact_dir))

    return cfg


def run_one_hidden_dim_sensitivity(
    config: Mapping[str, Any],
    hidden_dim: int,
    active_seed: int,
    step20_paths: Mapping[str, Path],
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate one hidden_dim setting.

    Returns one sensitivity table row plus run metadata.
    """
    hidden_dim = int(hidden_dim)
    cfg = make_hidden_dim_sensitivity_config(
        config=config,
        hidden_dim=hidden_dim,
        step20_paths=step20_paths,
    )

    model_dir = _project_path(cfg, str(get_by_path(cfg, "paths.models_dir", "results/models")))
    artifact_dir = Path(str(get_by_path(cfg, "sensitivity.active_artifact_dir", "")))

    print("=" * 120)
    print(f"STEP 20 hidden_dim RETRAINING RUN: hidden_dim={hidden_dim}")
    print("=" * 120)
    print(f"Model directory    : {model_dir}")
    print(f"Artifact directory : {artifact_dir}")
    print("Training           : Dataset-1 train only")
    print("Threshold selection: Dataset-1 validation only")
    print("Evaluation         : Dataset-1 test only")
    print("=" * 120)

    if logger is not None:
        logger.info("Step 20 hidden_dim=%s retraining started.", hidden_dim)
        logger.info("Step 20 hidden_dim=%s model dir: %s", hidden_dim, model_dir)

    training_start = time.perf_counter()

    training_summary = run_step12_training_protocol(
        config=cfg,
        active_seed=int(active_seed),
    )

    checkpoint_path = proposed_best_checkpoint_path(cfg)

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Hidden-dim sensitivity training completed but checkpoint is missing: {checkpoint_path}"
        )

    dataset1_result = run_dataset1_evaluation(
        config=cfg,
        active_seed=int(active_seed),
        checkpoint_path=str(checkpoint_path),
        model_name=f"Proposed-hdim{hidden_dim}",
        device=None,
    )

    runtime_seconds = float(time.perf_counter() - training_start)

    metrics = dataset1_result.metrics

    row = metrics_to_sensitivity_row(
        sensitivity_parameter="hidden_dim",
        sensitivity_value=int(hidden_dim),
        metrics=metrics,
        theta=float(dataset1_result.threshold),
        persistence=int(dataset1_result.persistence),
        model_name="Proposed",
        split="Dataset-1 Test",
        is_official_selected=False,
        status="PASSED",
        notes=(
            "True model-capacity sensitivity. "
            "A separate full proposed model was trained from scratch for this hidden_dim; "
            "theta and N_p were selected on Dataset-1 validation only."
        ),
    )

    run_payload = {
        "parameter": "hidden_dim",
        "value": int(hidden_dim),
        "status": "PASSED",
        "row": row,
        "runtime_seconds": runtime_seconds,
        "checkpoint_path": str(checkpoint_path),
        "model_dir": str(model_dir),
        "artifact_dir": str(artifact_dir),
        "training_summary": training_summary.to_dict()
        if hasattr(training_summary, "to_dict")
        else training_summary,
        "dataset1_result": dataset1_result.to_dict()
        if hasattr(dataset1_result, "to_dict")
        else dataset1_result,
    }

    _cleanup_torch_memory()

    if logger is not None:
        logger.info("Step 20 hidden_dim=%s completed.", hidden_dim)

    return run_payload


def run_hidden_dim_sensitivity(
    config: Mapping[str, Any],
    active_seed: int,
    step20_paths: Mapping[str, Path],
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run hidden_dim retraining sensitivity for all configured values."""
    if not _group_enabled(config, "hidden_dim"):
        return []

    values = _as_int_list(
        _group_values(config, "hidden_dim", [32, 64, 128]),
        name="sensitivity.hidden_dim.values",
    )

    continue_on_error = bool(
        get_by_path(config, "sensitivity.continue_on_error", False)
    )

    runs: List[Dict[str, Any]] = []

    for hidden_dim in values:
        try:
            runs.append(
                run_one_hidden_dim_sensitivity(
                    config=config,
                    hidden_dim=int(hidden_dim),
                    active_seed=int(active_seed),
                    step20_paths=step20_paths,
                    logger=logger,
                )
            )
        except Exception as exc:
            _cleanup_torch_memory()

            failure_payload = {
                "parameter": "hidden_dim",
                "value": int(hidden_dim),
                "status": "FAILED",
                "error": repr(exc),
                "row": {
                    "Model": "Proposed",
                    "Split": "Dataset-1 Test",
                    "Sensitivity Parameter": "hidden_dim",
                    "Sensitivity Value": int(hidden_dim),
                    "theta": None,
                    "persistence": None,
                    "AUROC": None,
                    "AUPRC": None,
                    "F1": None,
                    "Precision": None,
                    "Recall": None,
                    "FPR": None,
                    "Attack Detection Rate": None,
                    "Detection Delay": None,
                    "False Alarms": None,
                    "is_official_selected": False,
                    "status": "FAILED",
                    "notes": f"Hidden-dim retraining failed: {repr(exc)}",
                },
            }

            runs.append(failure_payload)

            if logger is not None:
                logger.exception("Step 20 hidden_dim=%s failed.", hidden_dim)

            if not continue_on_error:
                raise

    return runs


# -------------------------------------------------------------------------------------------------
# Main Step-20 runner
# -------------------------------------------------------------------------------------------------


def run_step20_sensitivity_analysis(
    config: Mapping[str, Any],
    active_seed: int = 42,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run Step 20 hybrid sensitivity analysis.

    This is the public function called by main.py.
    """
    start_time = time.perf_counter()

    if not _sensitivity_enabled(config):
        raise RuntimeError(
            "sensitivity.enabled is false. Set sensitivity.enabled: true in experiments.yaml "
            "before running Step 20."
        )

    step20_paths = get_step20_paths(config)

    ensure_dir(step20_paths["results_csv"].parent)
    ensure_dir(step20_paths["summary_json"].parent)
    ensure_dir(step20_paths["models_dir"])
    ensure_dir(step20_paths["artifacts_dir"])

    print("=" * 120)
    print("STEP 20 HYBRID SENSITIVITY ANALYSIS")
    print("=" * 120)
    print("Included sensitivity groups:")
    print(f"  theta       : enabled={_group_enabled(config, 'theta')} | retrain=False")
    print(f"  persistence : enabled={_group_enabled(config, 'persistence')} | retrain=False")
    print(f"  hidden_dim  : enabled={_group_enabled(config, 'hidden_dim')} | retrain=True")
    print("Excluded:")
    print("  rho         : excluded because it is not currently defined in model.yaml/training.yaml")
    print("=" * 120)

    if logger is not None:
        logger.info("Step 20 hybrid sensitivity analysis started.")
        logger.info("Step 20 results CSV: %s", step20_paths["results_csv"])
        logger.info("Step 20 summary JSON: %s", step20_paths["summary_json"])

    # ---------------------------------------------------------------------------------------------
    # Part A: operating-point sensitivity from official saved predictions.
    # ---------------------------------------------------------------------------------------------

    official_bundle = load_official_test_bundle(config)
    selected = load_official_selected_operating_point(config)

    selected_theta = float(selected["theta"])
    selected_persistence = int(selected["persistence"])

    rows: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []

    selected_row = evaluate_selected_official_row(
        bundle=official_bundle,
        selected_theta=selected_theta,
        selected_persistence=selected_persistence,
    )
    rows.append(selected_row)

    theta_rows = run_theta_sensitivity(
        config=config,
        bundle=official_bundle,
        selected_theta=selected_theta,
        selected_persistence=selected_persistence,
    )
    rows.extend(theta_rows)

    persistence_rows = run_persistence_sensitivity(
        config=config,
        bundle=official_bundle,
        selected_theta=selected_theta,
        selected_persistence=selected_persistence,
    )
    rows.extend(persistence_rows)

    # ---------------------------------------------------------------------------------------------
    # Part B: true retraining sensitivity for hidden_dim.
    # ---------------------------------------------------------------------------------------------

    hidden_runs = run_hidden_dim_sensitivity(
        config=config,
        active_seed=int(active_seed),
        step20_paths=step20_paths,
        logger=logger,
    )
    runs.extend(hidden_runs)

    for run in hidden_runs:
        row = run.get("row")
        if isinstance(row, Mapping):
            rows.append(dict(row))

    # ---------------------------------------------------------------------------------------------
    # Save final table and summary.
    # ---------------------------------------------------------------------------------------------

    result_df = save_sensitivity_results_table(
        output_path=step20_paths["results_csv"],
        rows=rows,
    )

    final_status = "PASSED"
    if any(str(run.get("status")) == "FAILED" for run in runs):
        final_status = "FAILED"

    summary = {
        "final_status": final_status,
        "step": "step20_hybrid_sensitivity_analysis",
        "active_seed": int(active_seed),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "protocol": {
            "theta_sensitivity": {
                "enabled": bool(_group_enabled(config, "theta")),
                "retraining_used": False,
                "description": "Operating-threshold sensitivity using official saved Dataset-1 test probabilities.",
            },
            "persistence_sensitivity": {
                "enabled": bool(_group_enabled(config, "persistence")),
                "retraining_used": False,
                "description": "Causal alarm-confirmation sensitivity using official saved Dataset-1 test probabilities.",
            },
            "hidden_dim_sensitivity": {
                "enabled": bool(_group_enabled(config, "hidden_dim")),
                "retraining_used": True,
                "description": "True model-capacity sensitivity; one model retrained per hidden_dim value.",
            },
            "rho_sensitivity": {
                "enabled": False,
                "included": False,
                "reason": "rho is not currently defined in model.yaml or training.yaml.",
            },
        },
        "official_operating_point": {
            "theta": selected_theta,
            "persistence": selected_persistence,
            "source": selected.get("source"),
            "loaded_from_json": bool(selected.get("loaded_from_json")),
            "warning": selected.get("warning"),
        },
        "input_artifacts": {
            "official_test_bundle_rows": int(len(official_bundle.labels)),
            "official_test_bundle_checkpoint": official_bundle.checkpoint_path,
        },
        "configured_values": {
            "theta": _group_values(config, "theta", [0.5, 0.6, 0.7, 0.8, 0.9])
            if _group_enabled(config, "theta")
            else [],
            "persistence": _group_values(config, "persistence", [3, 5, 7, 9, 11])
            if _group_enabled(config, "persistence")
            else [],
            "hidden_dim": _group_values(config, "hidden_dim", [32, 64, 128])
            if _group_enabled(config, "hidden_dim")
            else [],
        },
        "row_counts": {
            "total_rows": int(len(result_df)),
            "selected_rows": int((result_df["Sensitivity Parameter"] == "selected").sum())
            if "Sensitivity Parameter" in result_df.columns
            else 0,
            "theta_rows": int((result_df["Sensitivity Parameter"] == "theta").sum())
            if "Sensitivity Parameter" in result_df.columns
            else 0,
            "persistence_rows": int((result_df["Sensitivity Parameter"] == "persistence").sum())
            if "Sensitivity Parameter" in result_df.columns
            else 0,
            "hidden_dim_rows": int((result_df["Sensitivity Parameter"] == "hidden_dim").sum())
            if "Sensitivity Parameter" in result_df.columns
            else 0,
        },
        "hidden_dim_runs": runs,
        "output_paths": {
            "sensitivity_results_csv": str(step20_paths["results_csv"]),
            "sensitivity_summary_json": str(step20_paths["summary_json"]),
            "sensitivity_models_dir": str(step20_paths["models_dir"]),
            "sensitivity_artifacts_dir": str(step20_paths["artifacts_dir"]),
        },
        "leakage_rules": {
            "dataset1_train_used_for_hidden_dim_training": True,
            "dataset1_validation_used_for_threshold_selection": True,
            "dataset1_test_used_only_for_evaluation": True,
            "dataset2_used_for_sensitivity": False,
            "dataset3_used_for_sensitivity": False,
            "theta_not_inside_model": True,
            "persistence_not_inside_model": True,
            "rho_not_evaluated_without_config_path": True,
        },
    }

    save_json(
        _json_safe(summary),
        step20_paths["summary_json"],
        indent=2,
    )

    print_sensitivity_table(
        title="STEP 20 HYBRID SENSITIVITY RESULTS",
        rows=rows,
    )

    print("=" * 120)
    print("STEP 20 SAVED ARTIFACTS")
    print("=" * 120)
    print(f"Results CSV     : {step20_paths['results_csv']}")
    print(f"Summary JSON    : {step20_paths['summary_json']}")
    print(f"Models directory: {step20_paths['models_dir']}")
    print(f"Artifacts dir   : {step20_paths['artifacts_dir']}")
    print("=" * 120)

    if logger is not None:
        logger.info("Step 20 final status: %s", final_status)
        logger.info("Step 20 results CSV: %s", step20_paths["results_csv"])
        logger.info("Step 20 summary JSON: %s", step20_paths["summary_json"])

    if final_status != "PASSED":
        raise RuntimeError("Step 20 sensitivity analysis failed. See sensitivity_summary.json.")

    return summary


# Backward-compatible aliases expected by main.py or older runners.
run_sensitivity_analysis = run_step20_sensitivity_analysis
run_step20 = run_step20_sensitivity_analysis


__all__ = [
    "get_step20_paths",
    "load_prediction_bundle_from_npz",
    "load_official_selected_operating_point",
    "load_official_test_bundle",
    "run_theta_sensitivity",
    "run_persistence_sensitivity",
    "run_hidden_dim_sensitivity",
    "run_step20_sensitivity_analysis",
    "run_sensitivity_analysis",
    "run_step20",
]