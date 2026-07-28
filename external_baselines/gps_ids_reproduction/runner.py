"""
End-to-end GPS-IDS classifier-suite runner.

Protocol
--------
For every classifier:
1. load the exact same 15-feature GPS-IDS matrix and segment split;
2. fit candidate preprocessing/model pipelines on Dataset-1 training valid rows;
3. select the candidate by Dataset-1 validation AUPRC/AUROC/log-loss;
4. refit the selected pipeline on Dataset-1 training valid rows only;
5. generate validation probabilities and select theta/persistence using the
   authoritative unified selector;
6. freeze the complete pipeline and operating point;
7. evaluate Dataset-1 test, Dataset-2, and Dataset-3;
8. save standardized prediction bundles, evaluations, manifests, and tables.

GPS-IDS–MLP is marked as the primary published-method baseline. The remaining
six classifiers are retained for reproduction completeness and supplementary
comparison.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from external_baselines.gps_ids_reproduction.feature_contract import (
    build_gps_ids_feature_contract,
)
from external_baselines.gps_ids_reproduction.hyperparameters import (
    CandidateSpec,
    get_candidate_specs,
)
from external_baselines.gps_ids_reproduction.models import (
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    MODEL_REPORTING_ROLES,
    build_pipeline,
    count_learned_parameters,
    dependency_versions,
    extract_preprocessing_state,
    require_xgboost,
    save_pipeline,
    validate_model_key,
)
from external_baselines.gps_ids_reproduction.prediction import (
    CandidateSearchResult,
    GPSIDSFeatureDataset,
    assert_common_feature_contract,
    build_standardized_bundle,
    evaluation_row,
    fit_and_score_candidate,
    load_gps_ids_feature_dataset,
    predict_pipeline,
    save_prediction_and_evaluation,
    select_best_candidate,
)
from src.evaluation.artifact_manifest import (
    build_artifact_manifest,
    detect_code_version,
    save_artifact_manifest,
    save_strict_json,
    verify_artifact_manifest,
)
from src.evaluation.prediction_bundle_adapter import (
    sha256_file,
    verify_saved_prediction_bundle,
)
from src.evaluation.unified_operating_point import (
    DEFAULT_PERSISTENCE_GRID,
    DEFAULT_THRESHOLD_GRID,
    select_operating_point,
)
from src.utils.config import (
    get_by_path,
    resolve_project_path,
)


SPLIT_ORDER: Tuple[str, ...] = (
    "validation",
    "test",
    "dataset2",
    "dataset3",
)


def _required_path(
    config: Mapping[str, Any],
    key_path: str,
) -> Path:
    value = get_by_path(config, key_path, None)
    if value is None or not str(value).strip():
        raise KeyError(f"Missing required config path: {key_path}")
    return resolve_project_path(config, str(value))


def _plain_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return json.loads(json.dumps(config))


def _parse_model_keys(
    configured: Sequence[Any],
    override: Optional[Sequence[str]],
) -> List[str]:
    raw = list(override) if override else list(configured)
    if not raw:
        raw = list(MODEL_ORDER)

    normalized = [validate_model_key(item) for item in raw]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Duplicate model keys requested: {normalized}")

    # Preserve the published classifier order regardless of CLI order.
    return [key for key in MODEL_ORDER if key in set(normalized)]


def _assert_output_isolation(
    output_root: Path,
    project_root: Path,
) -> None:
    output_root = output_root.resolve()
    project_root = project_root.resolve()
    allowed = (
        project_root
        / "results"
        / "extended_comparison"
        / "gps_ids_reproduction"
    ).resolve()
    if output_root != allowed and allowed not in output_root.parents:
        raise AssertionError(
            f"Unsafe GPS-IDS classifier output root: {output_root}\n"
            f"Expected a path under {allowed}."
        )

    forbidden_roots = [
        (project_root / "results" / "models").resolve(),
        (project_root / "results" / "tables").resolve(),
        (project_root / "results" / "figures").resolve(),
    ]
    for forbidden in forbidden_roots:
        if output_root == forbidden or forbidden in output_root.parents:
            raise AssertionError(
                f"GPS-IDS branch would overwrite a legacy result root: "
                f"{output_root}"
            )


def _load_datasets(
    config: Mapping[str, Any],
) -> Dict[str, GPSIDSFeatureDataset]:
    prefix = "gps_ids_reproduction.classifier_suite.feature_files"
    paths = {
        "train": _required_path(config, f"{prefix}.train"),
        "validation": _required_path(
            config, f"{prefix}.validation"
        ),
        "test": _required_path(config, f"{prefix}.test"),
        "dataset2": _required_path(config, f"{prefix}.dataset2"),
        "dataset3": _required_path(config, f"{prefix}.dataset3"),
    }

    datasets = {
        split: load_gps_ids_feature_dataset(path, split)
        for split, path in paths.items()
    }
    assert_common_feature_contract(datasets)

    # No segment overlap is allowed among Dataset-1 splits.
    segment_sets = {
        split: set(datasets[split].segment_ids.astype(str).tolist())
        for split in ("train", "validation", "test")
    }
    intersections = {
        "train_validation": (
            segment_sets["train"] & segment_sets["validation"]
        ),
        "train_test": segment_sets["train"] & segment_sets["test"],
        "validation_test": (
            segment_sets["validation"] & segment_sets["test"]
        ),
    }
    if any(intersections.values()):
        raise ValueError(
            f"Dataset-1 segment leakage detected: {intersections}"
        )
    return datasets


def _save_candidate_results(
    results: Sequence[CandidateSearchResult],
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [item.to_dict() for item in results]

    json_path = output_dir / "candidate_search.json"
    csv_path = output_dir / "candidate_search.csv"
    save_strict_json(
        {
            "selection_metric_order": [
                "higher_validation_auprc",
                "higher_validation_auroc",
                "lower_validation_log_loss",
                "lower_candidate_order",
            ],
            "candidates": records,
        },
        json_path,
    )
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return {
        "candidate_search_json": str(json_path.resolve()),
        "candidate_search_csv": str(csv_path.resolve()),
    }


def _fit_selected_pipeline(
    *,
    candidate: CandidateSpec,
    train: GPSIDSFeatureDataset,
    seed: int,
    n_jobs: int,
) -> Tuple[Any, Dict[str, Any], float]:
    pipeline, build_spec = build_pipeline(
        model_key=candidate.model_key,
        imputer_strategy=candidate.imputer_strategy,
        scaler_name=candidate.scaler_name,
        model_parameters=candidate.model_parameters,
        seed=seed,
        n_jobs=n_jobs,
    )
    train_mask = train.valid_mask
    start = time.perf_counter()
    pipeline.fit(
        train.X[train_mask],
        train.labels[train_mask],
    )
    fit_seconds = time.perf_counter() - start
    return pipeline, build_spec.to_dict(), float(fit_seconds)


def _processed_paths_for_split(
    datasets: Mapping[str, GPSIDSFeatureDataset],
    split_name: str,
) -> List[Path]:
    names = ["train", "validation"]
    if split_name not in names:
        names.append(split_name)
    output: List[Path] = []
    seen = set()
    for name in names:
        path = Path(datasets[name].source_path).resolve()
        if path not in seen:
            seen.add(path)
            output.append(path)
    return output


def _model_output_paths(
    run_root: Path,
    model_key: str,
) -> Dict[str, Path]:
    model_root = run_root / "models" / model_key
    return {
        "model_root": model_root,
        "checkpoint": model_root / "checkpoint" / "pipeline.joblib",
        "preprocessing_manifest": (
            model_root / "checkpoint" / "preprocessing_manifest.json"
        ),
        "candidate_dir": model_root / "search",
        "operating_point": model_root / "operating_point.json",
        "summary": model_root / "model_summary.json",
        "completion": model_root / "COMPLETED.json",
        "predictions": model_root / "predictions",
        "evaluations": model_root / "evaluations",
        "manifests": model_root / "manifests",
    }


def _candidate_lookup(
    candidates: Sequence[CandidateSpec],
) -> Dict[str, CandidateSpec]:
    return {item.candidate_id: item for item in candidates}


def run_one_classifier(
    *,
    config: Mapping[str, Any],
    run_root: Path,
    datasets: Mapping[str, GPSIDSFeatureDataset],
    model_key: str,
    search_profile: str,
    seed: int,
    training_seed_list: Sequence[int],
    n_jobs: int,
    resolved_config_path: Path,
    code_version: str,
    overwrite: bool,
    fail_on_candidate_error: bool,
) -> Dict[str, Any]:
    paths = _model_output_paths(run_root, model_key)
    completion_path = paths["completion"]

    if completion_path.exists() and not overwrite:
        raise FileExistsError(
            f"Completed output already exists for {model_key}: "
            f"{completion_path}. Use --overwrite to replace this isolated "
            "model directory."
        )
    if paths["model_root"].exists() and overwrite:
        shutil.rmtree(paths["model_root"])

    for key, path in paths.items():
        if key in {"checkpoint", "preprocessing_manifest", "operating_point",
                   "summary", "completion"}:
            path.parent.mkdir(parents=True, exist_ok=True)
        elif key != "model_root":
            path.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_DISPLAY_NAMES[model_key]
    reporting_role = MODEL_REPORTING_ROLES[model_key]
    candidates = get_candidate_specs(model_key, search_profile)

    print("-" * 132)
    print(
        f"MODEL {model_name} | role={reporting_role} | "
        f"candidates={len(candidates)}"
    )
    print("-" * 132)

    search_results: List[CandidateSearchResult] = []
    for position, candidate in enumerate(candidates, start=1):
        print(
            f"[{model_key}] candidate {position}/{len(candidates)} "
            f"{candidate.candidate_id} | "
            f"imputer={candidate.imputer_strategy} "
            f"scaler={candidate.scaler_name}"
        )
        result, _ = fit_and_score_candidate(
            candidate=candidate,
            train=datasets["train"],
            validation=datasets["validation"],
            seed=seed,
            n_jobs=n_jobs,
        )
        search_results.append(result)

        if result.status == "PASSED":
            print(
                f"  PASSED | AUPRC={result.validation_auprc:.6f} "
                f"AUROC={result.validation_auroc:.6f} "
                f"logloss={result.validation_log_loss:.6f} "
                f"fit={result.fit_seconds:.2f}s "
                f"predict={result.predict_seconds:.2f}s"
            )
        else:
            print(
                f"  FAILED | {result.error_type}: "
                f"{result.error_message}"
            )
            if fail_on_candidate_error:
                raise RuntimeError(
                    f"Candidate {candidate.candidate_id} failed and "
                    "fail_on_candidate_error=true."
                )

    search_paths = _save_candidate_results(
        search_results,
        paths["candidate_dir"],
    )
    selected_result = select_best_candidate(search_results)
    candidate_by_id = _candidate_lookup(candidates)
    selected_candidate = candidate_by_id[
        selected_result.candidate_id
    ]

    print(
        f"[{model_key}] SELECTED {selected_candidate.candidate_id} | "
        f"AUPRC={selected_result.validation_auprc:.6f} "
        f"AUROC={selected_result.validation_auroc:.6f}"
    )

    pipeline, pipeline_spec, final_fit_seconds = _fit_selected_pipeline(
        candidate=selected_candidate,
        train=datasets["train"],
        seed=seed,
        n_jobs=n_jobs,
    )
    checkpoint_path = save_pipeline(
        pipeline,
        paths["checkpoint"],
    )
    parameter_count = count_learned_parameters(pipeline)

    preprocessing_state = extract_preprocessing_state(
        pipeline,
        datasets["train"].feature_names,
    )
    preprocessing_manifest = {
        "status": "PASSED",
        "model_key": model_key,
        "model_name": model_name,
        "reporting_role": reporting_role,
        "fit_scope": "dataset1_train_valid_rows_only",
        "training_rows_total": datasets["train"].rows,
        "training_rows_used": datasets["train"].valid_rows,
        "training_rows_excluded_by_valid_mask": int(
            datasets["train"].rows - datasets["train"].valid_rows
        ),
        "feature_incomplete_rows_in_training": int(
            datasets["train"].rows
            - datasets["train"].feature_complete_mask.sum()
        ),
        "missing_values_imputed": True,
        "feature_complete_mask_used_as_feature": False,
        "target_yaw_missing_used_as_feature": False,
        "gps_ids_contract_feature_hash": (
            datasets["train"].feature_hash
        ),
        "pipeline_spec": pipeline_spec,
        "fitted_preprocessing_state": preprocessing_state,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": int(parameter_count),
    }
    save_strict_json(
        preprocessing_manifest,
        paths["preprocessing_manifest"],
    )

    validation_prediction = predict_pipeline(
        pipeline,
        datasets["validation"].X,
    )
    validation_bundle = build_standardized_bundle(
        dataset=datasets["validation"],
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        prediction=validation_prediction,
    )
    operating_point = select_operating_point(
        validation_bundle.to_unified_prediction_bundle(),
        threshold_grid=DEFAULT_THRESHOLD_GRID,
        persistence_grid=DEFAULT_PERSISTENCE_GRID,
        include_candidates=True,
    )
    save_strict_json(
        operating_point.to_dict(include_candidates=True),
        paths["operating_point"],
    )
    theta = float(operating_point.theta)
    persistence = int(operating_point.persistence)

    print(
        f"[{model_key}] operating point | theta={theta:.2f} "
        f"persistence={persistence} "
        f"validation_F1={operating_point.selected_candidate.f1:.6f} "
        f"ADR={operating_point.selected_candidate.attack_detection_rate:.6f}"
    )

    metric_rows: List[Dict[str, Any]] = []
    split_artifacts: Dict[str, Any] = {}
    artifacts_by_split: Dict[str, Any] = {}

    for split_name in SPLIT_ORDER:
        dataset = datasets[split_name]
        prediction = (
            validation_prediction
            if split_name == "validation"
            else predict_pipeline(pipeline, dataset.X)
        )
        bundle = build_standardized_bundle(
            dataset=dataset,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            prediction=prediction,
        )

        prediction_path = (
            paths["predictions"] / f"{split_name}_predictions.npz"
        )
        evaluation_path = (
            paths["evaluations"] / f"{split_name}_evaluation.json"
        )
        artifact, evaluation = save_prediction_and_evaluation(
            bundle=bundle,
            theta=theta,
            persistence=persistence,
            prediction_npz_path=prediction_path,
            evaluation_json_path=evaluation_path,
        )
        verification = verify_saved_prediction_bundle(artifact)

        manifest = build_artifact_manifest(
            bundle=bundle,
            prediction_artifact=artifact,
            model_family=model_key,
            input_representation=(
                "GPS-IDS intended 15-feature vehicle-behavior representation"
            ),
            feature_names=list(dataset.feature_names),
            processed_data_paths=_processed_paths_for_split(
                datasets,
                split_name,
            ),
            checkpoint_path=checkpoint_path,
            resolved_config=resolved_config_path,
            active_seed=seed,
            training_seed_list=list(training_seed_list),
            threshold_source=(
                "dataset1_validation_unified_operating_point_selector"
            ),
            theta=theta,
            persistence=persistence,
            parameter_count=parameter_count,
            code_version=code_version,
            project_root=Path(
                get_by_path(config, "project.root", ".")
            ),
        )
        manifest_path = (
            paths["manifests"]
            / f"{split_name}_artifact_manifest.json"
        )
        save_artifact_manifest(manifest, manifest_path)
        manifest_verification = verify_artifact_manifest(
            manifest_path
        )

        row = evaluation_row(
            model_key=model_key,
            model_name=model_name,
            reporting_role=reporting_role,
            selected_candidate_id=selected_candidate.candidate_id,
            parameter_count=parameter_count,
            feature_hash=dataset.feature_hash,
            evaluation=evaluation,
        )
        row["prediction_seconds"] = float(
            prediction.predict_seconds
        )
        metric_rows.append(row)

        split_artifacts[split_name] = {
            "prediction_artifact": artifact.to_dict(),
            "prediction_verification": verification,
            "evaluation_path": str(evaluation_path.resolve()),
            "artifact_manifest_path": str(manifest_path.resolve()),
            "artifact_manifest_verification": (
                manifest_verification
            ),
        }

        print(
            f"[{model_key}] {split_name:<10} | "
            f"AUPRC={evaluation.ranking_metrics.auprc:.6f} "
            f"AUROC={evaluation.ranking_metrics.auroc:.6f} "
            f"F1={evaluation.sample_metrics.f1:.6f} "
            f"P={evaluation.sample_metrics.precision:.6f} "
            f"R={evaluation.sample_metrics.recall:.6f} "
            f"FPR={evaluation.sample_metrics.fpr:.6f} "
            f"ADR={evaluation.event_metrics.attack_detection_rate:.6f} "
            f"Delay={evaluation.event_metrics.mean_detection_delay_seconds}"
        )

    model_metrics_path = paths["model_root"] / "metrics.csv"
    pd.DataFrame(metric_rows).to_csv(
        model_metrics_path,
        index=False,
    )

    model_summary = {
        "status": "PASSED",
        "model_key": model_key,
        "model_name": model_name,
        "reporting_role": reporting_role,
        "search_profile": search_profile,
        "selection_rule": [
            "higher_validation_auprc",
            "higher_validation_auroc",
            "lower_validation_log_loss",
            "lower_candidate_order",
        ],
        "selected_candidate": selected_candidate.to_dict(),
        "selected_candidate_validation_result": (
            selected_result.to_dict()
        ),
        "pipeline_spec": pipeline_spec,
        "final_fit_seconds": final_fit_seconds,
        "parameter_count": int(parameter_count),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "preprocessing_manifest_path": str(
            paths["preprocessing_manifest"].resolve()
        ),
        "candidate_search_paths": search_paths,
        "operating_point_path": str(
            paths["operating_point"].resolve()
        ),
        "theta": theta,
        "persistence": persistence,
        "threshold_source": (
            "dataset1_validation_unified_operating_point_selector"
        ),
        "gps_ids_contract_feature_hash": (
            datasets["train"].feature_hash
        ),
        "feature_names": list(datasets["train"].feature_names),
        "training_rows_total": datasets["train"].rows,
        "training_rows_used": datasets["train"].valid_rows,
        "split_artifacts": split_artifacts,
        "metrics_path": str(model_metrics_path.resolve()),
    }
    save_strict_json(model_summary, paths["summary"])

    completion = {
        "status": "PASSED",
        "model_key": model_key,
        "model_summary_path": str(paths["summary"].resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "theta": theta,
        "persistence": persistence,
    }
    save_strict_json(completion, completion_path)

    return {
        "model_summary": model_summary,
        "metric_rows": metric_rows,
    }


def _print_final_table(metrics: pd.DataFrame) -> None:
    print("=" * 160)
    print("STEP 5 GPS-IDS CLASSIFIER-SUITE FINAL METRICS")
    print("=" * 160)
    columns = [
        "model_name",
        "reporting_role",
        "split",
        "auprc",
        "auroc",
        "precision",
        "recall",
        "f1",
        "fpr",
        "attack_detection_rate",
        "mean_detection_delay_seconds",
        "theta",
        "persistence",
    ]
    print(metrics.loc[:, columns].to_string(index=False))
    print("=" * 160)


def run_gps_ids_classifier_suite(
    *,
    config: Mapping[str, Any],
    model_keys_override: Optional[Sequence[str]] = None,
    search_profile_override: Optional[str] = None,
    overwrite_override: Optional[bool] = None,
) -> Dict[str, Any]:
    require_xgboost()

    project_root = Path(
        get_by_path(config, "project.root", ".")
    ).resolve()
    run_id = str(
        get_by_path(
            config,
            "gps_ids_reproduction.classifier_suite.run_id",
            "gps_ids_suite_v1_seed42",
        )
    ).strip()
    if not run_id:
        raise ValueError("GPS-IDS classifier run_id is empty.")

    base_output_root = _required_path(
        config,
        "gps_ids_reproduction.classifier_suite.output_root",
    )
    run_root = (base_output_root / run_id).resolve()
    _assert_output_isolation(run_root, project_root)

    configured_models = get_by_path(
        config,
        "gps_ids_reproduction.classifier_suite.models",
        list(MODEL_ORDER),
    )
    model_keys = _parse_model_keys(
        configured_models,
        model_keys_override,
    )

    search_profile = str(
        search_profile_override
        or get_by_path(
            config,
            "gps_ids_reproduction.classifier_suite.search_profile",
            "standard",
        )
    ).strip().lower()
    seed = int(
        get_by_path(
            config,
            "gps_ids_reproduction.classifier_suite.active_seed",
            42,
        )
    )
    training_seed_list = [
        int(item)
        for item in get_by_path(
            config,
            "gps_ids_reproduction.classifier_suite.training_seed_list",
            [seed],
        )
    ]
    if len(training_seed_list) != len(set(training_seed_list)):
        raise ValueError("training_seed_list contains duplicates.")
    if seed not in training_seed_list:
        raise ValueError(
            "active_seed must be included in training_seed_list."
        )

    n_jobs = int(
        get_by_path(
            config,
            "gps_ids_reproduction.classifier_suite.n_jobs",
            -1,
        )
    )
    overwrite = (
        bool(overwrite_override)
        if overwrite_override is not None
        else bool(
            get_by_path(
                config,
                "gps_ids_reproduction.classifier_suite.overwrite",
                False,
            )
        )
    )
    fail_on_candidate_error = bool(
        get_by_path(
            config,
            "gps_ids_reproduction.classifier_suite.fail_on_candidate_error",
            False,
        )
    )

    if run_root.exists() and overwrite:
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    datasets = _load_datasets(config)
    contract = build_gps_ids_feature_contract()
    if datasets["train"].feature_hash != contract.feature_hash:
        raise ValueError(
            "Step-4 feature hash does not match the imported feature contract."
        )

    resolved_config_path = run_root / "resolved_config.json"
    save_strict_json(_plain_config(config), resolved_config_path)

    requested_code_paths = [
        project_root
        / "external_baselines"
        / "gps_ids_reproduction",
        project_root
        / "src"
        / "evaluation",
        project_root
        / "src"
        / "experiments"
        / "run_gps_ids_reproduction.py",
        project_root / "configs" / "gps_ids_classifiers.yaml",
    ]
    existing_code_paths = [
        path for path in requested_code_paths if path.exists()
    ]
    code_version = detect_code_version(
        project_root=project_root,
        code_paths=(existing_code_paths or None),
    )

    preflight = {
        "status": "PASSED",
        "run_id": run_id,
        "run_root": str(run_root),
        "model_keys": model_keys,
        "all_seven_classifiers_requested": (
            model_keys == list(MODEL_ORDER)
        ),
        "search_profile": search_profile,
        "active_seed": seed,
        "training_seed_list": training_seed_list,
        "n_jobs": n_jobs,
        "overwrite": overwrite,
        "fail_on_candidate_error": fail_on_candidate_error,
        "feature_names": list(contract.final_model_feature_names),
        "gps_ids_contract_feature_hash": contract.feature_hash,
        "dataset_summaries": {
            key: value.summary()
            for key, value in datasets.items()
        },
        "dependency_versions": dependency_versions(),
        "code_version": code_version,
        "resolved_config_path": str(
            resolved_config_path.resolve()
        ),
        "resolved_config_sha256": sha256_file(
            resolved_config_path
        ),
        "protocol_assertions": {
            "same_feature_order_all_models": True,
            "same_segment_split_all_models": True,
            "model_specific_feature_substitution": False,
            "preprocessing_fit_on_dataset1_train_only": True,
            "hyperparameters_selected_on_dataset1_validation_only": True,
            "operating_point_selected_on_dataset1_validation_only": True,
            "dataset1_test_dataset2_dataset3_excluded_from_selection": True,
            "feature_complete_mask_in_model_input": False,
            "target_yaw_missing_in_model_input": False,
        },
    }
    save_strict_json(preflight, run_root / "preflight.json")

    print("=" * 132)
    print("STEP 5 GPS-IDS CLASSIFIER-SUITE REIMPLEMENTATION")
    print("=" * 132)
    print(
        "Reproduction status : protocol-controlled reimplementation "
        "(not exact reproduction)"
    )
    print(f"Run ID              : {run_id}")
    print(f"Output root         : {run_root}")
    print(f"Search profile      : {search_profile}")
    print(f"Active seed         : {seed}")
    print(f"Models              : {', '.join(model_keys)}")
    print(f"Locked features     : {len(contract.final_model_feature_names)}")
    print(f"Feature hash        : {contract.feature_hash}")
    print(f"Code version        : {code_version}")
    print("-" * 132)
    for split_name, dataset in datasets.items():
        print(
            f"{split_name:<10} rows={dataset.rows:<7} "
            f"valid={dataset.valid_rows:<7} "
            f"segments={len(np.unique(dataset.segment_ids)):<4} "
            f"incomplete={dataset.rows - int(dataset.feature_complete_mask.sum()):<7} "
            f"target_yaw_missing={int(dataset.target_yaw_missing.sum())}"
        )
    print("=" * 132)

    all_metric_rows: List[Dict[str, Any]] = []
    model_summaries: Dict[str, Any] = {}

    for model_key in model_keys:
        result = run_one_classifier(
            config=config,
            run_root=run_root,
            datasets=datasets,
            model_key=model_key,
            search_profile=search_profile,
            seed=seed,
            training_seed_list=training_seed_list,
            n_jobs=n_jobs,
            resolved_config_path=resolved_config_path,
            code_version=code_version,
            overwrite=overwrite,
            fail_on_candidate_error=fail_on_candidate_error,
        )
        model_summaries[model_key] = result["model_summary"]
        all_metric_rows.extend(result["metric_rows"])

        # Persist progress after every completed model.
        save_strict_json(
            {
                "status": "IN_PROGRESS",
                "completed_models": list(model_summaries),
                "requested_models": model_keys,
                "model_summaries": model_summaries,
            },
            run_root / "run_progress.json",
        )

    metrics = pd.DataFrame(all_metric_rows)
    comparison_dir = run_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    all_metrics_path = comparison_dir / "all_metrics_long.csv"
    metrics.to_csv(all_metrics_path, index=False)

    for split_name in SPLIT_ORDER:
        metrics.loc[metrics["split"] == split_name].to_csv(
            comparison_dir / f"{split_name}_metrics.csv",
            index=False,
        )

    # Primary published-method baseline summary.
    primary_summary: Optional[Dict[str, Any]] = None
    if "mlp" in model_summaries:
        primary_rows = metrics.loc[metrics["model_key"] == "mlp"]
        primary_summary = {
            "model_key": "mlp",
            "model_name": MODEL_DISPLAY_NAMES["mlp"],
            "reporting_role": MODEL_REPORTING_ROLES["mlp"],
            "model_summary_path": str(
                (
                    run_root
                    / "models"
                    / "mlp"
                    / "model_summary.json"
                ).resolve()
            ),
            "metrics": primary_rows.to_dict(orient="records"),
        }
        save_strict_json(
            primary_summary,
            comparison_dir / "primary_gps_ids_mlp_summary.json",
        )

    suite_completed = model_keys == list(MODEL_ORDER)
    final_report = {
        "status": "PASSED",
        "run_id": run_id,
        "run_root": str(run_root),
        "models_requested": model_keys,
        "models_completed": list(model_summaries),
        "all_seven_classifiers_completed": suite_completed,
        "primary_published_method_baseline": (
            "GPS-IDS–MLP" if "mlp" in model_summaries else None
        ),
        "supplementary_classifiers": [
            MODEL_DISPLAY_NAMES[key]
            for key in model_keys
            if key != "mlp"
        ],
        "search_profile": search_profile,
        "active_seed": seed,
        "training_seed_list": training_seed_list,
        "gps_ids_contract_feature_hash": contract.feature_hash,
        "feature_names": list(contract.final_model_feature_names),
        "threshold_source": (
            "dataset1_validation_unified_operating_point_selector"
        ),
        "metrics_path": str(all_metrics_path.resolve()),
        "primary_summary": primary_summary,
        "model_summaries": model_summaries,
        "preflight_path": str(
            (run_root / "preflight.json").resolve()
        ),
        "resolved_config_path": str(
            resolved_config_path.resolve()
        ),
        "code_version": code_version,
    }
    save_strict_json(
        final_report,
        run_root / "gps_ids_classifier_suite_report.json",
    )
    save_strict_json(
        {
            "status": "PASSED",
            "all_seven_classifiers_completed": suite_completed,
            "report_path": str(
                (
                    run_root
                    / "gps_ids_classifier_suite_report.json"
                ).resolve()
            ),
        },
        run_root / "COMPLETED.json",
    )

    _print_final_table(metrics)
    print(
        "Final report: "
        f"{(run_root / 'gps_ids_classifier_suite_report.json').resolve()}"
    )
    print(
        "Metrics     : "
        f"{all_metrics_path.resolve()}"
    )
    print("=" * 160)

    return final_report


__all__ = [
    "run_gps_ids_classifier_suite",
    "run_one_classifier",
]
