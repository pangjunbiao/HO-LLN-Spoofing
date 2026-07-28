"""
Feature-importance diagnostics for the trained full proposed model.

Step 14 purpose:
- Estimate the diagnostic contribution of eta, eta_dot, eta_ddot, q,
  accumulated weak evidence, and nu.
- Use input masking/occlusion at inference time only.
- Save CSV/JSON results under results/figures/module_usage/.

Important:
- This is diagnostic occlusion only.
- This is not the official ablation.
- Official ablations must still be retrained from scratch later.
- This file uses the Step-13 validation-selected theta and persistence.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.diagnostics.module_usage import (
    FeatureGroupSpec,
    Step14DiagnosticsContext,
    build_step14_paths,
    load_step14_context,
    save_json_safe,
)
from src.evaluation.evaluate_dataset1 import (
    EvaluationPredictionBundle,
    build_evaluation_dataloader,
    evaluate_bundle_with_threshold,
)
from src.utils.config import get_by_path
from src.utils.device import move_to_device
from src.utils.io import ensure_dir


@dataclass
class FeatureImportanceResult:
    """Feature-importance result for one split and one feature group."""

    split: str
    group_name: str
    columns: List[str]
    indices: List[int]
    replacement_strategy: str

    baseline_auprc: Optional[float]
    occluded_auprc: Optional[float]
    delta_auprc: Optional[float]

    baseline_f1: Optional[float]
    occluded_f1: Optional[float]
    delta_f1: Optional[float]

    baseline_fpr: Optional[float]
    occluded_fpr: Optional[float]
    delta_fpr: Optional[float]

    baseline_attack_detection_rate: Optional[float]
    occluded_attack_detection_rate: Optional[float]
    delta_attack_detection_rate: Optional[float]

    baseline_detection_delay: Optional[float]
    occluded_detection_delay: Optional[float]
    delta_detection_delay: Optional[float]

    mean_abs_probability_change: Optional[float]
    mean_signed_probability_change: Optional[float]
    max_abs_probability_change: Optional[float]

    valid_rows: int
    runtime_seconds: float

    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def _delta(occluded: Any, baseline: Any) -> Optional[float]:
    """Return occluded - baseline for finite values."""
    a = _safe_float(occluded)
    b = _safe_float(baseline)

    if a is None or b is None:
        return None

    return float(a - b)


def build_reference_values(
    context: Step14DiagnosticsContext,
    replacement_strategy: str = "neutral",
) -> Tensor:
    """
    Build reference values for feature occlusion.

    neutral:
    - scaled continuous xi features -> 0.0
    - xi_nu -> 1.0, so we remove variation in nu without pretending rows are invalid

    zero:
    - all features -> 0.0

    one:
    - all features -> 1.0
    """
    replacement_strategy = str(replacement_strategy).lower().strip()
    feature_columns = list(context.feature_columns)

    if replacement_strategy == "neutral":
        values = np.zeros(len(feature_columns), dtype=np.float32)

        for i, column in enumerate(feature_columns):
            if column == "xi_nu":
                values[i] = 1.0

        return torch.tensor(values, dtype=torch.float32, device=context.device)

    if replacement_strategy == "zero":
        return torch.zeros(len(feature_columns), dtype=torch.float32, device=context.device)

    if replacement_strategy == "one":
        return torch.ones(len(feature_columns), dtype=torch.float32, device=context.device)

    raise ValueError(
        f"Unsupported feature replacement strategy: {replacement_strategy}. "
        "Use one of: neutral, zero, one."
    )


def occlude_batch_features(
    batch: Mapping[str, Any],
    feature_indices: Sequence[int],
    reference_values: Tensor,
) -> Dict[str, Any]:
    """
    Return batch copy with selected feature indices replaced.

    Only x is modified.
    y, loss_mask, padding_mask, segment ids, and delta_t are untouched.
    """
    transformed: Dict[str, Any] = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            transformed[key] = value.clone()
        else:
            transformed[key] = value

    x = transformed["x"].clone()

    for feature_index in feature_indices:
        x[:, :, int(feature_index)] = reference_values[int(feature_index)]

    transformed["x"] = x

    return transformed


@torch.no_grad()
def collect_predictions_with_optional_occlusion(
    context: Step14DiagnosticsContext,
    split_name: str,
    group: Optional[FeatureGroupSpec] = None,
    replacement_strategy: str = "neutral",
) -> EvaluationPredictionBundle:
    """
    Collect model predictions with optional input feature occlusion.
    """
    loader, _dataset = build_evaluation_dataloader(
        config=context.config,
        split_name=split_name,
        active_seed=context.active_seed,
        full_sequence=(split_name == "online"),
    )

    reference_values = build_reference_values(
        context=context,
        replacement_strategy=replacement_strategy,
    )

    probabilities: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []
    segment_ids: List[str] = []
    row_indices: List[int] = []
    delta_t_values: List[float] = []

    context.model.eval()

    for batch in loader:
        batch = move_to_device(batch, context.device)

        if group is not None:
            batch = occlude_batch_features(
                batch=batch,
                feature_indices=group.indices,
                reference_values=reference_values,
            )

        output = context.model(batch)

        batch_probs = output.probabilities.detach().cpu().numpy()
        batch_logits = output.logits.detach().cpu().numpy()
        batch_labels = batch["y"].detach().cpu().numpy()
        batch_valid = (batch["loss_mask"] * batch["padding_mask"]).detach().cpu().numpy()
        batch_delta = batch["delta_t"].detach().cpu().numpy()
        real_lengths = batch["real_length"].detach().cpu().numpy().astype(int)
        start_indices = batch["start_index"].detach().cpu().numpy().astype(int)
        batch_segment_ids = [str(item) for item in batch["segment_id"]]

        for i in range(batch_probs.shape[0]):
            real_len = int(real_lengths[i])
            start = int(start_indices[i])
            seg_id = batch_segment_ids[i]

            probabilities.append(batch_probs[i, :real_len].reshape(-1))
            logits.append(batch_logits[i, :real_len].reshape(-1))
            labels.append(batch_labels[i, :real_len].reshape(-1))
            valid_masks.append(batch_valid[i, :real_len].reshape(-1))

            segment_ids.extend([seg_id] * real_len)
            row_indices.extend(list(range(start, start + real_len)))
            delta_t_values.extend(batch_delta[i, :real_len].reshape(-1).astype(float).tolist())

    if probabilities:
        p = np.concatenate(probabilities).astype(np.float32)
        z = np.concatenate(logits).astype(np.float32)
        y = np.concatenate(labels).astype(np.int64)
        m = np.concatenate(valid_masks).astype(np.float32)
    else:
        p = np.asarray([], dtype=np.float32)
        z = np.asarray([], dtype=np.float32)
        y = np.asarray([], dtype=np.int64)
        m = np.asarray([], dtype=np.float32)

    bundle = EvaluationPredictionBundle(
        split_name=split_name,
        probabilities=p,
        logits=z,
        labels=y,
        valid_mask=m,
        segment_ids=np.asarray(segment_ids, dtype=object),
        row_indices=np.asarray(row_indices, dtype=np.int64),
        delta_t=np.asarray(delta_t_values, dtype=np.float32),
        checkpoint_path=str(context.checkpoint_path),
        model_name=context.diagnostics_config.model_name,
    )

    return _sort_and_deduplicate_bundle(bundle)


def _sort_and_deduplicate_bundle(bundle: EvaluationPredictionBundle) -> EvaluationPredictionBundle:
    """Sort and average duplicate row predictions if overlapping windows exist."""
    if len(bundle.row_indices) == 0:
        return bundle

    df = pd.DataFrame(
        {
            "row_index": bundle.row_indices,
            "probability": bundle.probabilities,
            "logit": bundle.logits,
            "label": bundle.labels,
            "valid_mask": bundle.valid_mask,
            "segment_id": bundle.segment_ids.astype(str),
            "delta_t": bundle.delta_t,
        }
    )

    grouped = (
        df.groupby("row_index", sort=True)
        .agg(
            probability=("probability", "mean"),
            logit=("logit", "mean"),
            label=("label", "first"),
            valid_mask=("valid_mask", "max"),
            segment_id=("segment_id", "first"),
            delta_t=("delta_t", "first"),
        )
        .reset_index()
    )

    return EvaluationPredictionBundle(
        split_name=bundle.split_name,
        probabilities=grouped["probability"].to_numpy(dtype=np.float32),
        logits=grouped["logit"].to_numpy(dtype=np.float32),
        labels=grouped["label"].to_numpy(dtype=np.int64),
        valid_mask=grouped["valid_mask"].to_numpy(dtype=np.float32),
        segment_ids=grouped["segment_id"].to_numpy(dtype=object),
        row_indices=grouped["row_index"].to_numpy(dtype=np.int64),
        delta_t=grouped["delta_t"].to_numpy(dtype=np.float32),
        checkpoint_path=bundle.checkpoint_path,
        model_name=bundle.model_name,
    )


def compute_probability_change_summary(
    baseline_bundle: EvaluationPredictionBundle,
    occluded_bundle: EvaluationPredictionBundle,
) -> Dict[str, Optional[float]]:
    """Compute probability-change summary between baseline and occluded outputs."""
    if len(baseline_bundle.probabilities) != len(occluded_bundle.probabilities):
        raise ValueError(
            "Baseline and occluded prediction lengths do not match: "
            f"{len(baseline_bundle.probabilities)} vs {len(occluded_bundle.probabilities)}"
        )

    keep = baseline_bundle.valid_mask > 0.5

    if keep.sum() == 0:
        return {
            "mean_abs_probability_change": None,
            "mean_signed_probability_change": None,
            "max_abs_probability_change": None,
            "valid_rows": 0,
        }

    delta = occluded_bundle.probabilities[keep] - baseline_bundle.probabilities[keep]

    return {
        "mean_abs_probability_change": float(np.mean(np.abs(delta))),
        "mean_signed_probability_change": float(np.mean(delta)),
        "max_abs_probability_change": float(np.max(np.abs(delta))),
        "valid_rows": int(keep.sum()),
    }


def compute_feature_importance_for_group(
    context: Step14DiagnosticsContext,
    split_name: str,
    baseline_bundle: EvaluationPredictionBundle,
    baseline_metrics: Mapping[str, Any],
    group: FeatureGroupSpec,
    replacement_strategy: str = "neutral",
) -> FeatureImportanceResult:
    """Compute diagnostic feature importance for one feature group."""
    start_time = time.perf_counter()

    occluded_bundle = collect_predictions_with_optional_occlusion(
        context=context,
        split_name=split_name,
        group=group,
        replacement_strategy=replacement_strategy,
    )

    occluded_metrics = evaluate_bundle_with_threshold(
        bundle=occluded_bundle,
        theta=context.selected_threshold.theta,
        persistence=context.selected_threshold.persistence,
    )

    probability_change = compute_probability_change_summary(
        baseline_bundle=baseline_bundle,
        occluded_bundle=occluded_bundle,
    )

    return FeatureImportanceResult(
        split=split_name,
        group_name=group.name,
        columns=list(group.columns),
        indices=list(group.indices),
        replacement_strategy=replacement_strategy,
        baseline_auprc=_safe_float(baseline_metrics.get("auprc")),
        occluded_auprc=_safe_float(occluded_metrics.get("auprc")),
        delta_auprc=_delta(occluded_metrics.get("auprc"), baseline_metrics.get("auprc")),
        baseline_f1=_safe_float(baseline_metrics.get("f1")),
        occluded_f1=_safe_float(occluded_metrics.get("f1")),
        delta_f1=_delta(occluded_metrics.get("f1"), baseline_metrics.get("f1")),
        baseline_fpr=_safe_float(baseline_metrics.get("fpr")),
        occluded_fpr=_safe_float(occluded_metrics.get("fpr")),
        delta_fpr=_delta(occluded_metrics.get("fpr"), baseline_metrics.get("fpr")),
        baseline_attack_detection_rate=_safe_float(
            baseline_metrics.get("attack_detection_rate")
        ),
        occluded_attack_detection_rate=_safe_float(
            occluded_metrics.get("attack_detection_rate")
        ),
        delta_attack_detection_rate=_delta(
            occluded_metrics.get("attack_detection_rate"),
            baseline_metrics.get("attack_detection_rate"),
        ),
        baseline_detection_delay=_safe_float(
            baseline_metrics.get("mean_detection_delay")
        ),
        occluded_detection_delay=_safe_float(
            occluded_metrics.get("mean_detection_delay")
        ),
        delta_detection_delay=_delta(
            occluded_metrics.get("mean_detection_delay"),
            baseline_metrics.get("mean_detection_delay"),
        ),
        mean_abs_probability_change=probability_change["mean_abs_probability_change"],
        mean_signed_probability_change=probability_change["mean_signed_probability_change"],
        max_abs_probability_change=probability_change["max_abs_probability_change"],
        valid_rows=int(probability_change["valid_rows"]),
        runtime_seconds=float(time.perf_counter() - start_time),
        description=group.description,
    )


def run_feature_importance_for_split(
    context: Step14DiagnosticsContext,
    split_name: str = "test",
    replacement_strategy: str = "neutral",
) -> List[FeatureImportanceResult]:
    """Run feature-importance diagnostics for one split."""
    print("=" * 100)
    print(f"STEP 14 FEATURE IMPORTANCE DIAGNOSTICS | split={split_name}")
    print("=" * 100)
    print(f"Replacement strategy : {replacement_strategy}")
    print(f"Selected theta/N_p   : {context.selected_threshold.theta} / {context.selected_threshold.persistence}")
    print("Occlusion is diagnostic only; official ablation will retrain from scratch.")
    print("=" * 100)

    baseline_bundle = collect_predictions_with_optional_occlusion(
        context=context,
        split_name=split_name,
        group=None,
        replacement_strategy=replacement_strategy,
    )

    baseline_metrics = evaluate_bundle_with_threshold(
        bundle=baseline_bundle,
        theta=context.selected_threshold.theta,
        persistence=context.selected_threshold.persistence,
    )

    results: List[FeatureImportanceResult] = []

    for group in context.feature_groups:
        result = compute_feature_importance_for_group(
            context=context,
            split_name=split_name,
            baseline_bundle=baseline_bundle,
            baseline_metrics=baseline_metrics,
            group=group,
            replacement_strategy=replacement_strategy,
        )
        results.append(result)

        print(
            f"{group.name:<32} | "
            f"ΔAUPRC={result.delta_auprc} | "
            f"ΔF1={result.delta_f1} | "
            f"ΔFPR={result.delta_fpr} | "
            f"ΔADR={result.delta_attack_detection_rate} | "
            f"ΔDelay={result.delta_detection_delay} | "
            f"mean|Δp|={result.mean_abs_probability_change}"
        )

    print("=" * 100)

    return results


def save_feature_importance_results(
    context: Step14DiagnosticsContext,
    results: Sequence[FeatureImportanceResult],
) -> Dict[str, str]:
    """Save feature-importance results."""
    rows = [result.to_dict() for result in results]

    output_paths: Dict[str, str] = {}

    if context.diagnostics_config.save_csv:
        ensure_dir(context.paths.feature_importance_csv.parent)
        pd.DataFrame(rows).to_csv(context.paths.feature_importance_csv, index=False)
        output_paths["feature_importance_csv"] = str(context.paths.feature_importance_csv)

    if context.diagnostics_config.save_json:
        save_json_safe(
            {
                "context": context.to_dict(),
                "results": rows,
                "interpretation_note": (
                    "These are diagnostic occlusion results only. "
                    "Official ablations must be retrained from scratch."
                ),
            },
            context.paths.feature_importance_json,
        )
        output_paths["feature_importance_json"] = str(context.paths.feature_importance_json)

    return output_paths


def run_feature_importance_diagnostics(
    config: Mapping[str, Any],
    active_seed: int = 42,
    context: Optional[Step14DiagnosticsContext] = None,
) -> Dict[str, Any]:
    """
    Run Step-14 feature-importance diagnostics for configured splits.
    """
    if context is None:
        context = load_step14_context(config=config, active_seed=active_seed)

    replacement_strategy = str(
        get_by_path(
            config,
            "experiments.step14.feature_importance.replacement_strategy",
            "neutral",
        )
    )

    all_results: List[FeatureImportanceResult] = []

    for split_name in context.diagnostics_config.diagnostic_splits:
        split_results = run_feature_importance_for_split(
            context=context,
            split_name=str(split_name),
            replacement_strategy=replacement_strategy,
        )
        all_results.extend(split_results)

    artifact_paths = save_feature_importance_results(
        context=context,
        results=all_results,
    )

    return {
        "status": "PASSED",
        "result_count": len(all_results),
        "diagnostic_splits": list(context.diagnostics_config.diagnostic_splits),
        "replacement_strategy": replacement_strategy,
        "artifact_paths": artifact_paths,
    }


__all__ = [
    "FeatureImportanceResult",
    "build_reference_values",
    "occlude_batch_features",
    "collect_predictions_with_optional_occlusion",
    "compute_probability_change_summary",
    "compute_feature_importance_for_group",
    "run_feature_importance_for_split",
    "save_feature_importance_results",
    "run_feature_importance_diagnostics",
]