"""
Step-4 runner: reconstruct and lock the GPS-IDS feature contract.

Run from the project root:

    python -m external_baselines.gps_ids_reproduction.preprocessing \
        --config-dir configs \
        --gps-ids-config gps_ids_features.yaml

This runner is isolated. It does not modify legacy preprocessing, xi datasets,
model checkpoints, or existing paper tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from external_baselines.gps_ids_reproduction.feature_builder import (
    GPSIDSFeatureBuildReport,
    build_gps_ids_feature_files,
    sha256_file,
)
from external_baselines.gps_ids_reproduction.feature_contract import (
    build_gps_ids_feature_contract,
    save_gps_ids_feature_contract,
    save_strict_json,
)
from src.utils.config import (
    get_by_path,
    load_project_config,
    resolve_project_path,
)


BASE_CONFIG_FILES = ("dataset.yaml",)


def _resolve_required_path(
    config: Mapping[str, Any],
    key_path: str,
) -> Path:
    value = get_by_path(config, key_path, None)
    if value is None or str(value).strip() == "":
        raise KeyError(f"Missing required GPS-IDS config path: {key_path}")
    return resolve_project_path(config, str(value))


def _load_step4_paths(
    config: Mapping[str, Any],
) -> Dict[str, Dict[str, Path] | Path]:
    prefix = "gps_ids_reproduction.feature_contract"

    source_paths = {
        "dataset1": _resolve_required_path(
            config, f"{prefix}.source_files.dataset1"
        ),
        "dataset2": _resolve_required_path(
            config, f"{prefix}.source_files.dataset2"
        ),
        "dataset3": _resolve_required_path(
            config, f"{prefix}.source_files.dataset3"
        ),
    }
    split_paths = {
        "train": _resolve_required_path(
            config, f"{prefix}.split_files.train"
        ),
        "validation": _resolve_required_path(
            config, f"{prefix}.split_files.validation"
        ),
        "test": _resolve_required_path(
            config, f"{prefix}.split_files.test"
        ),
    }
    output_paths = {
        "train": _resolve_required_path(
            config, f"{prefix}.output_files.train"
        ),
        "validation": _resolve_required_path(
            config, f"{prefix}.output_files.validation"
        ),
        "test": _resolve_required_path(
            config, f"{prefix}.output_files.test"
        ),
        "dataset2": _resolve_required_path(
            config, f"{prefix}.output_files.dataset2"
        ),
        "dataset3": _resolve_required_path(
            config, f"{prefix}.output_files.dataset3"
        ),
    }

    return {
        "source_paths": source_paths,
        "split_paths": split_paths,
        "output_paths": output_paths,
        "contract_json": _resolve_required_path(
            config, f"{prefix}.contract_json"
        ),
        "mapping_csv": _resolve_required_path(
            config, f"{prefix}.mapping_csv"
        ),
        "build_report_json": _resolve_required_path(
            config, f"{prefix}.build_report_json"
        ),
        "artifact_index_json": _resolve_required_path(
            config, f"{prefix}.artifact_index_json"
        ),
    }


def _assert_output_isolation(
    config: Mapping[str, Any],
    paths: Dict[str, Dict[str, Path] | Path],
) -> None:
    allowed_data_root = _resolve_required_path(
        config,
        "gps_ids_reproduction.feature_contract.allowed_data_root",
    )
    allowed_result_root = _resolve_required_path(
        config,
        "gps_ids_reproduction.feature_contract.allowed_result_root",
    )

    writable = [
        *paths["output_paths"].values(),
        paths["contract_json"],
        paths["mapping_csv"],
        paths["build_report_json"],
        paths["artifact_index_json"],
    ]

    for path in writable:
        path = Path(path).resolve()
        inside_data = (
            path == allowed_data_root
            or allowed_data_root in path.parents
        )
        inside_results = (
            path == allowed_result_root
            or allowed_result_root in path.parents
        )
        if not (inside_data or inside_results):
            raise AssertionError(
                f"Unsafe Step-4 output path: {path}\n"
                f"Allowed roots: {allowed_data_root}, {allowed_result_root}"
            )


def run_gps_ids_feature_contract_step(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    paths = _load_step4_paths(config)
    _assert_output_isolation(config, paths)

    contract = build_gps_ids_feature_contract()
    contract_paths = save_gps_ids_feature_contract(
        contract=contract,
        contract_json_path=paths["contract_json"],
        mapping_csv_path=paths["mapping_csv"],
    )

    report = build_gps_ids_feature_files(
        contract=contract,
        source_paths=paths["source_paths"],
        split_paths=paths["split_paths"],
        output_paths=paths["output_paths"],
    )
    save_strict_json(
        report.to_dict(),
        paths["build_report_json"],
    )

    artifacts = {
        "gps_ids_feature_contract.json": Path(
            contract_paths["contract_json_path"]
        ),
        "gps_ids_feature_mapping.csv": Path(
            contract_paths["mapping_csv_path"]
        ),
        "gps_ids_feature_build_report.json": Path(
            paths["build_report_json"]
        ),
        **{
            Path(path).name: Path(path)
            for path in paths["output_paths"].values()
        },
    }
    artifact_index = {
        "status": "PASSED",
        "step": 4,
        "branch": "gps_ids_reproduction",
        "reproduction_status": (
            "protocol-controlled reimplementation_not_exact_reproduction"
        ),
        "feature_hash": contract.feature_hash,
        "feature_count": contract.final_model_feature_count,
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(artifacts.items())
        },
        "legacy_project_modified": False,
        "xi_files_used_as_input": False,
        "classifier_training_performed": False,
    }
    save_strict_json(
        artifact_index,
        paths["artifact_index_json"],
    )

    return {
        "status": "PASSED",
        "contract": contract.to_dict(),
        "report": report.to_dict(),
        "contract_paths": contract_paths,
        "build_report_path": str(
            Path(paths["build_report_json"]).resolve()
        ),
        "artifact_index_path": str(
            Path(paths["artifact_index_json"]).resolve()
        ),
    }


def print_step4_report(result: Mapping[str, Any]) -> None:
    report = result["report"]
    contract = result["contract"]

    print("=" * 118)
    print("STEP 4 GPS-IDS FEATURE-CONTRACT RECONSTRUCTION")
    print("=" * 118)
    print(f"Status                    : {result['status']}")
    print(
        "Reproduction status       : "
        "protocol-controlled reimplementation (not exact reproduction)"
    )
    print(f"Locked model features     : {contract['final_model_feature_count']}")
    print(f"Feature hash              : {contract['feature_hash']}")
    print("Source representation     : segmented raw AV-GPS data before pruning")
    print("Xi inputs used            : False")
    print("Classifier training       : Not performed in Step 4")
    print("-" * 118)
    print(
        f"{'Split':<14} {'Rows':>9} {'Segments':>10} "
        f"{'Normal':>9} {'Attack':>9} {'Valid':>9} "
        f"{'Invalid':>9} {'Complete':>10} {'Incomplete':>11} "
        f"{'YawMissing':>11}  Output"
    )
    print("-" * 148)

    for split_name in (
        "train",
        "validation",
        "test",
        "dataset2",
        "dataset3",
    ):
        summary = report["file_summaries"][split_name]
        print(
            f"{split_name:<14} "
            f"{summary['rows']:>9} "
            f"{summary['segments']:>10} "
            f"{summary['normal_rows']:>9} "
            f"{summary['attack_rows']:>9} "
            f"{summary['valid_rows']:>9} "
            f"{summary['invalid_rows']:>9} "
            f"{summary['feature_complete_rows']:>10} "
            f"{summary['feature_incomplete_rows']:>11} "
            f"{summary['target_yaw_missing_rows']:>11}  "
            f"{summary['output_path']}"
        )

    print("-" * 148)
    split_check = report["dataset1_split_leakage_check"]
    print(
        "Dataset-1 split check     : "
        f"{'PASSED' if split_check['passed'] else 'FAILED'}"
    )
    print(
        "Contract JSON             : "
        f"{result['contract_paths']['contract_json_path']}"
    )
    print(
        "Mapping CSV               : "
        f"{result['contract_paths']['mapping_csv_path']}"
    )
    print(f"Build report              : {result['build_report_path']}")
    print(f"Artifact index            : {result['artifact_index_path']}")
    print("=" * 118)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the locked GPS-IDS behavior-feature contract and "
            "segment-controlled feature CSVs."
        )
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--gps-ids-config",
        default="gps_ids_features.yaml",
        help=(
            "GPS-IDS Step-4 YAML loaded after dataset.yaml. "
            "Do not add it to the default project config order."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_files = [
        *BASE_CONFIG_FILES,
        str(args.gps_ids_config),
    ]
    config = load_project_config(
        config_dir=args.config_dir,
        config_files=config_files,
        allow_missing_optional=False,
    )
    result = run_gps_ids_feature_contract_step(config)
    print_step4_report(result)


if __name__ == "__main__":
    main()
