"""
Step 16 Frozen Component-Intervention Ablation Study.

Purpose
-------
Evaluate module importance using one trained Full Proposed checkpoint.

Important difference from retrained Step 16:
    - This runner does NOT call Step 12 training.
    - It loads results/models/proposed_best.pt for every variant.
    - It applies model.set_runtime_intervention(variant_name) during evaluation.
    - It selects theta/Np on Dataset-1 validation only.
    - It evaluates Dataset-1 test, Dataset-2 external, and Dataset-3 online.

Scientific interpretation:
    Retrained ablation:
        Can another model relearn the task without this module?

    Frozen intervention ablation:
        Does the trained Proposed model depend on this module?
"""

from __future__ import annotations

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
from src.experiments.run_ablations import (
    LOCKED_ABLATION_VARIANTS,
    AblationVariantResult,
    Step16AblationSummary,
    _false_alarm_events,
    _format_metric,
    _json_safe,
    _project_path,
    _result_metrics,
    _safe_float,
    _save_json_safe,
    _variant_display_name,
    print_ablation_console_tables,
    save_ablation_plots,
)
from src.utils.config import get_by_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir


@dataclass
class Step16FrozenAblationConfig:
    """Runtime config for frozen Step-16 intervention ablation."""

    enabled: bool = True
    experiment_name: str = "frozen_component_intervention_ablation_step16"

    checkpoint_path: str = "results/models/proposed_best.pt"
    retrain_policy: str = "never"

    variants: List[str] = field(default_factory=lambda: list(LOCKED_ABLATION_VARIANTS))

    threshold_policy: str = "select_per_variant_on_validation"

    evaluate_dataset1: bool = True
    evaluate_dataset2: bool = True
    evaluate_dataset3: bool = True

    tables_dir: str = "results/tables/frozen_ablations"
    plots_dir: str = "results/figures/frozen_ablation_plots"

    results_csv: str = "results/tables/frozen_ablations/frozen_ablation_results.csv"
    all_splits_csv: str = "results/tables/frozen_ablations/frozen_ablation_results_all_splits.csv"
    threshold_csv: str = "results/tables/frozen_ablations/frozen_ablation_threshold_selection.csv"
    summary_json: str = "results/tables/frozen_ablations/frozen_ablation_summary.json"

    save_plots: bool = True
    save_variant_artifacts: bool = True
    print_console_tables: bool = True

    note: str = (
        "Frozen intervention ablation: same trained full checkpoint, "
        "one module disabled during evaluation only."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_step16_frozen_ablation_config(
    config: Mapping[str, Any],
) -> Step16FrozenAblationConfig:
    """Build frozen Step-16 config from experiments.step16_frozen."""
    configured_variants = get_by_path(
        config,
        "experiments.step16_frozen.variants",
        list(LOCKED_ABLATION_VARIANTS),
    )

    return Step16FrozenAblationConfig(
        enabled=bool(get_by_path(config, "experiments.step16_frozen.enabled", True)),
        experiment_name=str(
            get_by_path(
                config,
                "experiments.step16_frozen.experiment_name",
                "frozen_component_intervention_ablation_step16",
            )
        ),
        checkpoint_path=str(
            get_by_path(
                config,
                "experiments.step16_frozen.checkpoint_path",
                "results/models/proposed_best.pt",
            )
        ),
        retrain_policy=str(
            get_by_path(config, "experiments.step16_frozen.retrain_policy", "never")
        ),
        variants=[str(item) for item in list(configured_variants)],
        threshold_policy=str(
            get_by_path(
                config,
                "experiments.step16_frozen.threshold_policy",
                "select_per_variant_on_validation",
            )
        ),
        evaluate_dataset1=bool(
            get_by_path(config, "experiments.step16_frozen.evaluate_dataset1", True)
        ),
        evaluate_dataset2=bool(
            get_by_path(config, "experiments.step16_frozen.evaluate_dataset2", True)
        ),
        evaluate_dataset3=bool(
            get_by_path(config, "experiments.step16_frozen.evaluate_dataset3", True)
        ),
        tables_dir=str(
            get_by_path(
                config,
                "experiments.step16_frozen.tables_dir",
                "results/tables/frozen_ablations",
            )
        ),
        plots_dir=str(
            get_by_path(
                config,
                "experiments.step16_frozen.plots_dir",
                "results/figures/frozen_ablation_plots",
            )
        ),
        results_csv=str(
            get_by_path(
                config,
                "experiments.step16_frozen.results_csv",
                "results/tables/frozen_ablations/frozen_ablation_results.csv",
            )
        ),
        all_splits_csv=str(
            get_by_path(
                config,
                "experiments.step16_frozen.all_splits_csv",
                "results/tables/frozen_ablations/frozen_ablation_results_all_splits.csv",
            )
        ),
        threshold_csv=str(
            get_by_path(
                config,
                "experiments.step16_frozen.threshold_csv",
                "results/tables/frozen_ablations/frozen_ablation_threshold_selection.csv",
            )
        ),
        summary_json=str(
            get_by_path(
                config,
                "experiments.step16_frozen.summary_json",
                "results/tables/frozen_ablations/frozen_ablation_summary.json",
            )
        ),
        save_plots=bool(
            get_by_path(config, "experiments.step16_frozen.save_plots", True)
        ),
        save_variant_artifacts=bool(
            get_by_path(config, "experiments.step16_frozen.save_variant_artifacts", True)
        ),
        print_console_tables=bool(
            get_by_path(config, "experiments.step16_frozen.print_console_tables", True)
        ),
        note=str(
            get_by_path(
                config,
                "experiments.step16_frozen.note",
                "Frozen intervention ablation.",
            )
        ),
    )


def validate_frozen_variants(variants: Sequence[str]) -> None:
    """Validate frozen intervention variants."""
    allowed = set(LOCKED_ABLATION_VARIANTS)
    missing = [str(item) for item in variants if str(item) not in allowed]

    if missing:
        raise ValueError(
            f"Unknown frozen ablation variant(s): {missing}. "
            f"Allowed variants: {sorted(allowed)}"
        )


def evaluate_frozen_split(
    config: Mapping[str, Any],
    model: Any,
    checkpoint_path: Path,
    checkpoint_metadata: Mapping[str, Any],
    variant_name: str,
    display_name: str,
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
    """Evaluate one frozen-intervention variant on one split."""
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
    metrics["frozen_intervention_variant"] = str(variant_name)
    metrics["checkpoint_metadata"] = dict(checkpoint_metadata)

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
        checkpoint_path=str(checkpoint_path),
        prediction_summary=bundle.to_dict_summary(),
        artifact_paths=artifact_paths,
    )


def evaluate_frozen_variant_protocol(
    config: Mapping[str, Any],
    frozen_config: Step16FrozenAblationConfig,
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
    Frozen intervention protocol.

    Dataset-1 validation selects theta/Np for this intervention.
    Dataset-1 test, Dataset-2 external, and Dataset-3 online only apply it.
    """
    variant_tables_dir = _project_path(
        config,
        str(Path(frozen_config.tables_dir) / variant_name),
    )
    ensure_dir(variant_tables_dir)

    # Always load the FULL trained model checkpoint.
    model, checkpoint, checkpoint_metadata = load_trained_model_for_evaluation(
        config=config,
        checkpoint_path=str(checkpoint_path),
        device=device,
        variant_name="full",
    )

    if not hasattr(model, "set_runtime_intervention"):
        raise AttributeError(
            "Loaded Proposed model does not have set_runtime_intervention(). "
            "Patch src/models/proposed_model.py first."
        )

    model.set_runtime_intervention(variant_name)
    model.eval()

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

    if frozen_config.evaluate_dataset1:
        dataset1_result = evaluate_frozen_split(
            config=config,
            model=model,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            variant_name=variant_name,
            display_name=display_name,
            split_name="test",
            output_split_name="Dataset-1 Test",
            theta=theta,
            persistence=persistence,
            active_seed=active_seed,
            device=device,
            full_sequence=False,
            prediction_npz_path=variant_tables_dir / "dataset1_test_predictions.npz",
        )

    if frozen_config.evaluate_dataset2:
        dataset2_result = evaluate_frozen_split(
            config=config,
            model=model,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            variant_name=variant_name,
            display_name=display_name,
            split_name="external",
            output_split_name="Dataset-2 External",
            theta=theta,
            persistence=persistence,
            active_seed=active_seed,
            device=device,
            full_sequence=False,
            prediction_npz_path=variant_tables_dir / "dataset2_external_predictions.npz",
        )

    if frozen_config.evaluate_dataset3:
        dataset3_result = evaluate_frozen_split(
            config=config,
            model=model,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            variant_name=variant_name,
            display_name=display_name,
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
        "threshold_policy": frozen_config.threshold_policy,
        "validation_prediction_summary": val_bundle.to_dict_summary(),
        "validation_dataset_summary": val_dataset.summary(),
        "checkpoint_metadata": checkpoint_metadata,
        "frozen_intervention_variant": str(variant_name),
        "artifact_paths": {
            "threshold_selection_json": str(threshold_json),
            "threshold_candidates_csv": str(threshold_candidates_csv),
            "dataset1_val_predictions_npz": str(val_npz),
        },
    }

    return dataset1_result, dataset2_result, dataset3_result, threshold_payload


def _split_result_to_frozen_row(
    variant_name: str,
    display_name: str,
    split_name: str,
    result: Optional[Mapping[str, Any]],
    threshold_info: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert one split result into a flat frozen-ablation CSV row."""
    metrics = _result_metrics(result)

    return {
        "variant_name": variant_name,
        "display_name": display_name,
        "split": split_name,
        "theta": _safe_float(threshold_info.get("theta")),
        "persistence": (
            int(threshold_info.get("persistence"))
            if threshold_info.get("persistence") is not None
            else None
        ),
        "val_selected_f1": _safe_float(
            (threshold_info.get("selected_candidate") or {}).get("f1")
        ),
        "val_selected_auprc": _safe_float(
            (threshold_info.get("selected_candidate") or {}).get("auprc")
        ),
        "val_selected_auroc": _safe_float(
            (threshold_info.get("selected_candidate") or {}).get("auroc")
        ),
        "val_selected_fpr": _safe_float(
            (threshold_info.get("selected_candidate") or {}).get("fpr")
        ),
        "AUROC": _safe_float(metrics.get("auroc")),
        "AUPRC": _safe_float(metrics.get("auprc")),
        "F1": _safe_float(metrics.get("f1")),
        "Precision": _safe_float(metrics.get("precision")),
        "Recall": _safe_float(metrics.get("recall")),
        "FPR": _safe_float(metrics.get("fpr")),
        "Attack Detection Rate": _safe_float(metrics.get("attack_detection_rate")),
        "Detection Delay": _safe_float(metrics.get("mean_detection_delay")),
        "False Alarm Rows": _safe_float(metrics.get("row_level_false_alarms")),
        "False Alarm Events": _safe_float(metrics.get("false_alarm_events")),
        "Attack-1 Delay": _safe_float(metrics.get("attack_1_delay")),
        "Attack-2 Delay": _safe_float(metrics.get("attack_2_delay")),
        "Runtime": _safe_float(metrics.get("runtime_seconds")),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "tn": metrics.get("tn"),
        "fn": metrics.get("fn"),
        "checkpoint_path": result.get("checkpoint_path") if isinstance(result, Mapping) else None,
        "trained_from_scratch": False,
        "frozen_intervention": True,
        "same_full_checkpoint_for_all_variants": True,
        "threshold_selected_on_dataset1_validation_only": True,
        "dataset2_used_for_tuning": False,
        "dataset3_used_for_tuning": False,
        "uses_same_9_scaled_xi_features": True,
        "raw_shortcut_columns_used": False,
    }


def build_frozen_ablation_tables(
    results: Sequence[AblationVariantResult],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build frozen ablation result tables."""
    dataset1_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []

    for item in results:
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

            row = _split_result_to_frozen_row(
                variant_name=item.variant_name,
                display_name=item.display_name,
                split_name=split_name,
                result=split_result,
                threshold_info=threshold_info,
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
                "trained_from_scratch": False,
                "frozen_intervention": True,
                "status": item.status,
            }
        )

    return pd.DataFrame(dataset1_rows), pd.DataFrame(all_rows), pd.DataFrame(threshold_rows)


def run_step16_frozen_intervention_ablation_study(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> Step16AblationSummary:
    """
    Run frozen intervention ablation study.

    This uses one trained Full checkpoint and disables one module at a time
    during evaluation.
    """
    start_time = time.perf_counter()

    frozen_config = build_step16_frozen_ablation_config(config)
    variants = list(frozen_config.variants)
    validate_frozen_variants(variants)

    if str(frozen_config.threshold_policy) != "select_per_variant_on_validation":
        raise ValueError(
            "Only threshold_policy='select_per_variant_on_validation' is currently supported."
        )

    if str(frozen_config.retrain_policy).lower().strip() != "never":
        raise ValueError(
            "Frozen ablation must use retrain_policy='never'. "
            "Do not train ablations in this runner."
        )

    device_info = setup_device_from_config(config, verbose=True)
    device = device_info.device

    checkpoint_path = _project_path(config, frozen_config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Frozen ablation checkpoint not found: {checkpoint_path}. "
            "Run Step 13 first to create results/models/proposed_best.pt."
        )

    tables_dir = _project_path(config, frozen_config.tables_dir)
    plots_dir = _project_path(config, frozen_config.plots_dir)
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    print("=" * 120)
    print("STEP 16 FROZEN COMPONENT-INTERVENTION ABLATION STUDY START")
    print("=" * 120)
    print(f"Experiment name       : {frozen_config.experiment_name}")
    print(f"Active seed           : {active_seed}")
    print(f"Device                : {device}")
    print(f"Full checkpoint       : {checkpoint_path}")
    print(f"Variants              : {variants}")
    print(f"Retrain policy        : {frozen_config.retrain_policy}")
    print(f"Tables dir            : {tables_dir}")
    print("Frozen rule           : same Full checkpoint for every variant.")
    print("Intervention rule     : one module disabled during evaluation only.")
    print("Threshold rule        : theta/Np selected on Dataset-1 validation only.")
    print("External/online rule  : Dataset-2 and Dataset-3 are never used for tuning.")
    print("=" * 120)

    variant_results: List[AblationVariantResult] = []

    for variant_index, variant_name in enumerate(variants, start=1):
        variant_start = time.perf_counter()
        display_name = _variant_display_name(variant_name)
        variant_tables_dir = _project_path(
            config,
            str(Path(frozen_config.tables_dir) / variant_name),
        )
        ensure_dir(variant_tables_dir)

        print("=" * 120)
        print(f"STEP 16 FROZEN VARIANT {variant_index}/{len(variants)}: {variant_name}")
        print("=" * 120)

        try:
            dataset1_result, dataset2_result, dataset3_result, threshold_payload = (
                evaluate_frozen_variant_protocol(
                    config=config,
                    frozen_config=frozen_config,
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
                trained_from_scratch=False,
                checkpoint_path=str(checkpoint_path),
                training_summary=None,
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
            print(f"STEP 16 FROZEN VARIANT SUMMARY: {variant_name}")
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
                    title=f"STEP 16 FROZEN PRIMARY METRICS — {display_name}",
                    rows=split_rows,
                    model_key="Model",
                )

        except Exception as exc:
            result_payload = AblationVariantResult(
                variant_name=variant_name,
                display_name=display_name,
                status="FAILED",
                trained_from_scratch=False,
                checkpoint_path=str(checkpoint_path),
                training_summary=None,
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
            print(f"STEP 16 FROZEN VARIANT FAILED: {variant_name}")
            print("=" * 120)
            print(str(exc))
            print("=" * 120)

        variant_results.append(result_payload)

    dataset1_df, all_splits_df, threshold_df = build_frozen_ablation_tables(variant_results)

    results_csv = _project_path(config, frozen_config.results_csv)
    all_splits_csv = _project_path(config, frozen_config.all_splits_csv)
    threshold_csv = _project_path(config, frozen_config.threshold_csv)
    summary_json = _project_path(config, frozen_config.summary_json)

    ensure_dir(results_csv.parent)
    ensure_dir(all_splits_csv.parent)
    ensure_dir(threshold_csv.parent)
    ensure_dir(summary_json.parent)

    dataset1_df.to_csv(results_csv, index=False)
    all_splits_df.to_csv(all_splits_csv, index=False)
    threshold_df.to_csv(threshold_csv, index=False)

    saved_plots: List[str] = []
    if frozen_config.save_plots:
        saved_plots = save_ablation_plots(dataset1_df, plots_dir)

    if frozen_config.print_console_tables:
        print_ablation_console_tables(
            dataset1_df=dataset1_df,
            all_splits_df=all_splits_df,
            threshold_df=threshold_df,
        )

    final_status = "PASSED" if all(item.status == "PASSED" for item in variant_results) else "FAILED"

    output_paths = {
        "checkpoint_path": str(checkpoint_path),
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
        "frozen_ablation_results_csv": str(results_csv),
        "frozen_ablation_results_all_splits_csv": str(all_splits_csv),
        "frozen_ablation_threshold_selection_csv": str(threshold_csv),
        "frozen_ablation_summary_json": str(summary_json),
    }

    for index, path in enumerate(saved_plots):
        output_paths[f"plot_{index:02d}"] = str(path)

    summary = Step16AblationSummary(
        final_status=final_status,
        active_seed=int(active_seed),
        experiment_name=frozen_config.experiment_name,
        variants=list(variants),
        results=[item.to_dict() for item in variant_results],
        output_paths=output_paths,
        runtime_seconds=float(time.perf_counter() - start_time),
        fairness_rules={
            "frozen_intervention_ablation": True,
            "all_variants_use_same_full_checkpoint": True,
            "checkpoint_path": str(checkpoint_path),
            "no_variant_training_in_this_runner": True,
            "trained_from_scratch": False,
            "runtime_intervention_only": True,
            "theta_np_selected_on_dataset1_validation_only": True,
            "dataset1_test_not_used_for_threshold_selection": True,
            "dataset2_external_not_used_for_tuning": True,
            "dataset3_online_not_used_for_tuning": True,
            "same_9_scaled_xi_features": True,
            "raw_shortcut_columns_used": False,
            "scientific_interpretation": (
                "Tests whether the trained Full Proposed model depends on each module."
            ),
        },
    )

    _save_json_safe(summary.to_dict(), summary_json)

    print("=" * 120)
    print("STEP 16 FROZEN COMPONENT-INTERVENTION ABLATION STUDY SUMMARY")
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
        raise RuntimeError(f"Frozen Step 16 ablation study failed for variants: {failed}")

    return summary


# Compatibility aliases for main.py.
run_step16_frozen_ablation_study = run_step16_frozen_intervention_ablation_study
run_frozen_ablation_study = run_step16_frozen_intervention_ablation_study


__all__ = [
    "Step16FrozenAblationConfig",
    "build_step16_frozen_ablation_config",
    "run_step16_frozen_intervention_ablation_study",
    "run_step16_frozen_ablation_study",
    "run_frozen_ablation_study",
]