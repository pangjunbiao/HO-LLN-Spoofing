"""
Synthetic validation runner for Step 3 prediction/provenance artifacts.

Run from the project root:

    python -m src.experiments.run_prediction_provenance_validation

No YAML changes and no trained model are required.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.evaluation.artifact_manifest import (
    build_artifact_manifest,
    save_artifact_manifest,
    save_strict_json,
    verify_artifact_manifest,
)
from src.evaluation.prediction_bundle_adapter import (
    StandardizedPredictionBundle,
    adapt_prediction_mapping,
    load_standardized_prediction_bundle,
    save_standardized_prediction_bundle,
    verify_saved_prediction_bundle,
)
from src.evaluation.unified_evaluator import evaluate_prediction_bundle


LOCKED_XI_FEATURES = [
    "xi_eta_east_scaled",
    "xi_eta_north_scaled",
    "xi_eta_dot_east_scaled",
    "xi_eta_dot_north_scaled",
    "xi_eta_ddot_east_scaled",
    "xi_eta_ddot_north_scaled",
    "xi_q_scaled",
    "xi_accum_log_scaled",
    "xi_nu",
]


def _expect_rejection(name: str, function) -> Dict[str, Any]:
    try:
        function()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        return {
            "name": name,
            "status": "PASSED",
            "rejection_type": type(exc).__name__,
            "message": str(exc),
        }
    raise AssertionError(f"{name} was expected to reject malformed input.")


def _valid_mapping(checkpoint_path: Path) -> Dict[str, Any]:
    return {
        "probability": np.asarray(
            [0.05, 0.10, 0.85, 0.90, 0.20, 0.95],
            dtype=float,
        ),
        "logit": np.asarray(
            [-2.94, -2.20, 1.73, 2.20, -1.39, 2.94],
            dtype=float,
        ),
        "label": np.asarray([0, 0, 1, 1, 0, 1], dtype=int),
        "valid_mask": np.asarray([1, 1, 1, 1, 1, 1], dtype=int),
        "segment_id": np.asarray(
            ["segment_a"] * 4 + ["segment_b"] * 2,
            dtype=object,
        ),
        "row_index": np.asarray([10, 11, 12, 13, 20, 21], dtype=int),
        "within_segment_index": np.asarray(
            [0, 1, 2, 3, 0, 1],
            dtype=int,
        ),
        "delta_t": np.asarray([0.0, 1.0, 1.5, 0.5, 0.0, 2.0]),
        "split": "validation",
        "model_name": "synthetic_step3_model",
        "checkpoint_path": str(checkpoint_path.resolve()),
    }


def run_step3_validation(
    output_root: Path | str = (
        "results/extended_comparison/"
        "prediction_provenance_engine"
    ),
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    support_dir = output_root / "synthetic_support"
    support_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = support_dir / "synthetic_model.pt"
    checkpoint_path.write_bytes(b"synthetic checkpoint for step 3 validation\n")

    processed_path = support_dir / "synthetic_validation_xi.csv"
    processed_path.write_text(
        "segment_id,row_index,within_segment_index,Data Type\n"
        "segment_a,10,0,0\n"
        "segment_a,11,1,0\n"
        "segment_a,12,2,1\n"
        "segment_a,13,3,1\n"
        "segment_b,20,0,0\n"
        "segment_b,21,1,1\n",
        encoding="utf-8",
    )

    resolved_config_path = support_dir / "resolved_config.json"
    resolved_config = {
        "seed": {"active_seed": 42, "training_seed_list": [42]},
        "features": LOCKED_XI_FEATURES,
        "unified_evaluation": {
            "threshold_source": "dataset1_validation",
            "theta": 0.5,
            "persistence": 2,
        },
    }
    save_strict_json(resolved_config, resolved_config_path)

    bundle = adapt_prediction_mapping(
        _valid_mapping(checkpoint_path),
        decision_score_type="logit",
    )

    prediction_path = output_root / "synthetic_validation_predictions.npz"
    prediction_artifact = save_standardized_prediction_bundle(
        bundle=bundle,
        npz_path=prediction_path,
    )
    prediction_verification = verify_saved_prediction_bundle(
        prediction_artifact
    )

    loaded_bundle = load_standardized_prediction_bundle(prediction_path)
    if loaded_bundle.content_hash() != bundle.content_hash():
        raise AssertionError("Save/load bundle hash changed.")

    evaluation = evaluate_prediction_bundle(
        bundle.to_unified_prediction_bundle(),
        theta=0.5,
        persistence=2,
    )

    manifest = build_artifact_manifest(
        bundle=bundle,
        prediction_artifact=prediction_artifact,
        model_family="synthetic_validation_model",
        input_representation="nine-dimensional scaled causal xi_t",
        feature_names=LOCKED_XI_FEATURES,
        processed_data_paths=[processed_path],
        checkpoint_path=checkpoint_path,
        resolved_config=resolved_config_path,
        active_seed=42,
        training_seed_list=[42],
        threshold_source="dataset1_validation",
        theta=0.5,
        persistence=2,
        parameter_count=12345,
        code_version="step3-synthetic-validation",
        project_root=Path.cwd(),
    )

    manifest_path = output_root / "synthetic_result_manifest.json"
    save_artifact_manifest(manifest, manifest_path)
    manifest_verification = verify_artifact_manifest(manifest_path)

    tests: List[Dict[str, Any]] = [
        {
            "name": "valid_bundle_save_load_evaluate",
            "status": "PASSED",
            "row_count": bundle.row_count,
            "bundle_hash": bundle.content_hash(),
            "split_hash": bundle.split_identity_hash(),
            "confirmed_f1": evaluation.sample_metrics.f1,
        },
        {
            "name": "manifest_build_and_verify",
            "status": "PASSED",
            "manifest_path": str(manifest_path.resolve()),
        },
    ]

    base = _valid_mapping(checkpoint_path)

    tests.append(
        _expect_rejection(
            "length_mismatch",
            lambda: adapt_prediction_mapping(
                {**base, "label": np.asarray([0, 1])},
                decision_score_type="logit",
            ),
        )
    )

    tests.append(
        _expect_rejection(
            "duplicate_row_identity",
            lambda: adapt_prediction_mapping(
                {
                    **base,
                    "row_index": np.asarray(
                        [10, 10, 12, 13, 20, 21]
                    ),
                },
                decision_score_type="logit",
            ),
        )
    )

    tests.append(
        _expect_rejection(
            "noncontiguous_segment_reappearance",
            lambda: adapt_prediction_mapping(
                {
                    **base,
                    "segment_id": np.asarray(
                        ["a", "a", "b", "b", "a", "a"],
                        dtype=object,
                    ),
                },
                decision_score_type="logit",
            ),
        )
    )

    tests.append(
        _expect_rejection(
            "invalid_nonbinary_label",
            lambda: adapt_prediction_mapping(
                {
                    **base,
                    "label": np.asarray([0, 0, 2, 1, 0, 1]),
                },
                decision_score_type="logit",
            ),
        )
    )

    tests.append(
        _expect_rejection(
            "invalid_nonbinary_mask",
            lambda: adapt_prediction_mapping(
                {
                    **base,
                    "valid_mask": np.asarray(
                        [1, 1, 0.5, 1, 1, 1]
                    ),
                },
                decision_score_type="logit",
            ),
        )
    )

    tests.append(
        _expect_rejection(
            "nonmonotonic_within_segment_index",
            lambda: adapt_prediction_mapping(
                {
                    **base,
                    "within_segment_index": np.asarray(
                        [0, 2, 1, 3, 0, 1]
                    ),
                },
                decision_score_type="logit",
            ),
        )
    )

    tests.append(
        _expect_rejection(
            "nonmonotonic_row_index",
            lambda: adapt_prediction_mapping(
                {
                    **base,
                    "row_index": np.asarray(
                        [10, 12, 11, 13, 20, 21]
                    ),
                },
                decision_score_type="logit",
            ),
        )
    )

    tests.append(
        _expect_rejection(
            "missing_within_segment_index_without_permission",
            lambda: adapt_prediction_mapping(
                {
                    key: value
                    for key, value in base.items()
                    if key != "within_segment_index"
                },
                decision_score_type="logit",
            ),
        )
    )

    inferred_bundle = adapt_prediction_mapping(
        {
            key: value
            for key, value in base.items()
            if key != "within_segment_index"
        },
        allow_infer_within_segment_index=True,
        decision_score_type="logit",
    )
    if inferred_bundle.within_segment_index_source != "inferred":
        raise AssertionError("Inference provenance was not recorded.")
    tests.append(
        {
            "name": "explicit_within_segment_inference",
            "status": "PASSED",
            "inferred_indices": (
                inferred_bundle.within_segment_indices.tolist()
            ),
        }
    )

    original_processed = processed_path.read_bytes()
    processed_path.write_bytes(original_processed + b"# tampered\n")
    tests.append(
        _expect_rejection(
            "processed_data_tamper_detection",
            lambda: verify_artifact_manifest(manifest_path),
        )
    )
    processed_path.write_bytes(original_processed)

    final_manifest_check = verify_artifact_manifest(manifest_path)
    if final_manifest_check["status"] != "PASSED":
        raise AssertionError("Manifest did not verify after file restoration.")

    report = {
        "status": "PASSED",
        "tests_passed": len(tests),
        "tests": tests,
        "prediction_artifact": prediction_artifact.to_dict(),
        "prediction_verification": prediction_verification,
        "manifest_verification": manifest_verification,
        "final_manifest_verification": final_manifest_check,
        "locked_contract": {
            "row_fields": [
                "probability",
                "decision_score",
                "label",
                "valid_mask",
                "segment_id",
                "row_index",
                "within_segment_index",
                "delta_t",
            ],
            "metadata_fields": [
                "split",
                "model_name",
                "checkpoint_path",
            ],
            "manifest_fields": [
                "model_family",
                "input_representation",
                "feature_names_and_order",
                "feature_hash",
                "split_hash",
                "processed_data_hash",
                "checkpoint_hash",
                "resolved_config_hash",
                "active_seed",
                "training_seed_list",
                "threshold_source",
                "theta",
                "persistence",
                "code_version",
                "parameter_count",
            ],
            "evaluator_rejects_inconsistent_lengths": True,
            "evaluator_rejects_duplicate_row_ids": True,
            "evaluator_rejects_noncontiguous_segments": True,
            "evaluator_rejects_invalid_labels": True,
            "evaluator_rejects_invalid_masks": True,
            "evaluator_rejects_nonchronological_indices": True,
        },
    }

    report_path = output_root / "step3_synthetic_validation.json"
    save_strict_json(report, report_path)

    report["report_path"] = str(report_path.resolve())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the AV–GPS standardized prediction and provenance "
            "artifact contract."
        )
    )
    parser.add_argument(
        "--output-root",
        default=(
            "results/extended_comparison/"
            "prediction_provenance_engine"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_step3_validation(args.output_root)

    print("=" * 100)
    print("STEP 3 STANDARDIZED PREDICTION AND PROVENANCE ARTIFACTS")
    print("=" * 100)
    print(f"Status           : {report['status']}")
    print(f"Tests passed     : {report['tests_passed']}")
    print(
        "Prediction NPZ   : "
        f"{report['prediction_artifact']['npz_path']}"
    )
    print(
        "Prediction meta  : "
        f"{report['prediction_artifact']['metadata_path']}"
    )
    print(
        "Manifest         : "
        f"{report['manifest_verification']['manifest_path']}"
    )
    print(f"Validation report: {report['report_path']}")
    print("-" * 100)
    for test in report["tests"]:
        print(f"{test['status']:<7} | {test['name']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
