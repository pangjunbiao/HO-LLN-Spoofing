"""
Run the isolated corrected-evidence branch without modifying legacy artifacts.

Execution:
    python -m src.experiments.run_corrected_evidence_branch \
        --config-dir configs \
        --override-config extended_preprocessing.yaml

The override YAML is loaded last only for this command. Do not add it to the
project's DEFAULT_CONFIG_ORDER.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from src.data.clean_columns import run_shortcut_column_exclusion
from src.data.segment_trajectories import run_trajectory_segmentation
from src.extended_preprocessing.label_independence_audit import (
    run_label_independence_audit,
)
from src.preprocessing.evidence_builder import (
    DEFAULT_DATASET_KEYS,
    get_residual_file_path,
    run_evidence_builder_step,
)
from src.preprocessing.motion_model import run_coordinate_motion_model_step
from src.preprocessing.normal_statistics import (
    load_training_normal_statistics_for_xi,
    run_residual_and_normal_statistics_step,
)
from src.utils.config import (
    DEFAULT_CONFIG_ORDER,
    get_by_path,
    load_project_config,
    resolve_project_path,
)
from src.utils.io import ensure_dir


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _save_strict_json(payload: Mapping[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, allow_nan=False)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_corrected_flag(config: Mapping[str, Any]) -> None:
    value = bool(
        get_by_path(
            config,
            "preprocessing.validity.invalidate_attack_to_normal_boundary",
            True,
        )
    )
    if value:
        raise AssertionError(
            "The isolated corrected branch requires "
            "preprocessing.validity.invalidate_attack_to_normal_boundary=false."
        )


def _assert_isolated_paths(config: Mapping[str, Any]) -> Dict[str, str]:
    """
    Verify every writable path is inside the extended-comparison branch.

    paths.splits_dir is intentionally excluded because it is reused read-only.
    """
    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    allowed_data_root = resolve_project_path(
        config,
        str(
            get_by_path(
                config,
                "extended_preprocessing.data_root",
                "data/extended_comparison/corrected_evidence_v1",
            )
        ),
    )
    allowed_result_root = resolve_project_path(
        config,
        str(
            get_by_path(
                config,
                "extended_preprocessing.result_root",
                "results/extended_comparison/corrected_evidence_v1",
            )
        ),
    )

    writable_path_keys = [
        "paths.interim_data_dir",
        "paths.processed_data_dir",
        "paths.step3_segmentation_json",
        "paths.step5_clean_columns_json",
        "paths.step6_physical_model_json",
        "paths.step7_residual_summary_json",
        "paths.step7_normal_statistics_json",
        "paths.step7_residual_normal_statistics_json",
        "paths.step8_evidence_summary_json",
        "paths.step8_energy_diagnostics_json",
        "paths.step8_xi_scaler_json",
        "paths.step8_xi_feature_spec_json",
        "paths.step8_normal_statistics_validation_json",
        "paths.extended_preprocessing_manifest_json",
        "paths.extended_preprocessing_audit_json",
    ]

    resolved: Dict[str, str] = {}
    for key in writable_path_keys:
        raw = get_by_path(config, key, None)
        if raw is None:
            raise KeyError(f"Missing required isolated output path: {key}")

        path = resolve_project_path(config, str(raw))
        resolved[key] = str(path)

        inside_data = path == allowed_data_root or allowed_data_root in path.parents
        inside_results = path == allowed_result_root or allowed_result_root in path.parents

        if not (inside_data or inside_results):
            raise AssertionError(
                f"Unsafe output path for {key}: {path}\n"
                f"Expected it under {allowed_data_root} or {allowed_result_root}."
            )

    legacy_roots = [
        (project_root / "data" / "interim").resolve(),
        (project_root / "data" / "processed").resolve(),
        (project_root / "results" / "models").resolve(),
        (project_root / "results" / "tables").resolve(),
        (project_root / "results" / "figures").resolve(),
    ]
    for path_text in resolved.values():
        path = Path(path_text)
        for legacy_root in legacy_roots:
            if path == legacy_root or legacy_root in path.parents:
                raise AssertionError(
                    f"Isolated branch would write into legacy root: {path}"
                )

    return {
        "project_root": str(project_root),
        "allowed_data_root": str(allowed_data_root),
        "allowed_result_root": str(allowed_result_root),
        **resolved,
    }


def _load_segmented_frames(
    segmentation_report: Any,
) -> Dict[str, pd.DataFrame]:
    """
    Load the isolated Step-3 outputs from the report.

    The report's dataset summaries expose output_path in the reviewed project.
    """
    frames: Dict[str, pd.DataFrame] = {}
    summaries = getattr(segmentation_report, "dataset_summaries", {})

    for dataset_key, summary in summaries.items():
        path = Path(getattr(summary, "output_path"))
        frames[str(dataset_key)] = pd.read_csv(path, low_memory=False)

    if not frames:
        raise RuntimeError("No isolated segmented frames were produced.")

    return frames


def _load_residual_frames(
    config: Mapping[str, Any],
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for dataset_key in DEFAULT_DATASET_KEYS:
        path = get_residual_file_path(config, dataset_key)
        if not path.exists():
            raise FileNotFoundError(
                f"Expected isolated residual file is missing: {path}"
            )
        frames[dataset_key] = pd.read_csv(path, low_memory=False)
    return frames


def run_corrected_evidence_branch(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Regenerate corrected xi artifacts entirely inside the isolated branch."""
    _assert_corrected_flag(config)
    isolation = _assert_isolated_paths(config)

    segmentation_report = run_trajectory_segmentation(
        config=config,
        bundle=None,
        save_outputs=True,
    )
    clean_report = run_shortcut_column_exclusion(
        config=config,
        dataset_keys=None,
        save_outputs=True,
    )
    physical_report = run_coordinate_motion_model_step(
        config=config,
        dataset_keys=None,
        save_outputs=True,
    )
    residual_statistics_report = run_residual_and_normal_statistics_step(
        config=config,
        dataset_keys=None,
        save_outputs=True,
    )
    evidence_report = run_evidence_builder_step(
        config=config,
        dataset_keys=None,
        save_outputs=True,
    )

    segmented_frames = _load_segmented_frames(segmentation_report)
    residual_frames = _load_residual_frames(config)
    normal_stats = load_training_normal_statistics_for_xi(
        config=config,
        save_validation_report=True,
    )

    audit = run_label_independence_audit(
        segmented_frames=segmented_frames,
        residual_frames=residual_frames,
        normal_stats=normal_stats,
        config=config,
    )
    if audit.get("status") != "PASSED":
        raise AssertionError(
            "Corrected-evidence label-independence audit failed. "
            "Inspect the audit JSON before training any model."
        )

    audit_path = resolve_project_path(
        config,
        str(
            get_by_path(
                config,
                "paths.extended_preprocessing_audit_json",
                "results/extended_comparison/corrected_evidence_v1/"
                "manifests/label_independence_audit.json",
            )
        ),
    )
    _save_strict_json(audit, audit_path)

    output_files = []
    for directory_key in ["paths.interim_data_dir", "paths.processed_data_dir"]:
        directory = resolve_project_path(
            config,
            str(get_by_path(config, directory_key)),
        )
        if directory.exists():
            output_files.extend(
                path for path in directory.rglob("*") if path.is_file()
            )

    hashes = {
        str(path): {
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in sorted(output_files)
    }

    manifest = {
        "status": "PASSED",
        "branch_name": str(
            get_by_path(
                config,
                "extended_preprocessing.branch_name",
                "corrected_evidence_v1",
            )
        ),
        "legacy_project_modified": False,
        "isolation": isolation,
        "configuration_claims": {
            "invalidate_attack_to_normal_boundary": False,
            "label_used_for_per_row_validity": False,
            "label_used_for_samplewise_evidence_after_reference_fit": False,
            "training_labels_used_to_fit_normal_reference_statistics": True,
            "validation_test_dataset2_dataset3_labels_used_for_evidence": False,
            "labels_used_for_supervised_training_and_evaluation": True,
        },
        "reports": {
            "segmentation": _json_safe(segmentation_report),
            "clean_columns": _json_safe(clean_report),
            "coordinate_motion": _json_safe(physical_report),
            "residual_and_normal_statistics": _json_safe(
                residual_statistics_report
            ),
            "evidence": _json_safe(evidence_report),
        },
        "audit_path": str(audit_path),
        "generated_file_hashes": hashes,
    }

    manifest_path = resolve_project_path(
        config,
        str(
            get_by_path(
                config,
                "paths.extended_preprocessing_manifest_json",
                "results/extended_comparison/corrected_evidence_v1/"
                "manifests/corrected_evidence_manifest.json",
            )
        ),
    )
    _save_strict_json(manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated corrected AV–GPS evidence preprocessing."
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--override-config",
        default="extended_preprocessing.yaml",
        help="Isolated override YAML loaded last for this command only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_files = [
        *DEFAULT_CONFIG_ORDER,
        str(args.override_config),
    ]
    config = load_project_config(
        config_dir=args.config_dir,
        config_files=config_files,
        allow_missing_optional=False,
    )

    manifest = run_corrected_evidence_branch(config)
    print("=" * 100)
    print("ISOLATED CORRECTED-EVIDENCE BRANCH COMPLETE")
    print("=" * 100)
    print(f"Status        : {manifest['status']}")
    print(f"Manifest      : {manifest['manifest_path']}")
    print(f"Legacy changed: {manifest['legacy_project_modified']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
