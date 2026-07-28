"""
Strict provenance manifest for standardized prediction/evaluation artifacts.

Every production result should record:

- model family
- input representation
- feature names and order
- feature hash
- split hash
- processed-data hash
- checkpoint hash
- resolved-config hash
- active seed
- training seed list
- threshold source
- theta
- persistence
- code version
- parameter count

The module uses strict JSON (`allow_nan=False`) and can verify all saved hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.evaluation.prediction_bundle_adapter import (
    SavedPredictionBundleArtifact,
    StandardizedPredictionBundle,
    sha256_file,
)


ARTIFACT_MANIFEST_SCHEMA_VERSION = "av_gps_result_manifest_v1"


def _strict_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_strict_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _strict_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _strict_json_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return _strict_json_value(vars(value))
    return str(value)


def save_strict_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Save portable JSON and reject NaN/Infinity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _strict_json_value(dict(payload)),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return path


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def canonical_json_hash(value: Any) -> str:
    """Hash a canonical strict-JSON representation."""
    encoded = json.dumps(
        _strict_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_contract_hash(feature_names: Sequence[str]) -> str:
    features = [str(name).strip() for name in feature_names]
    if not features or any(not name for name in features):
        raise ValueError("feature_names must be a nonempty list of names.")
    if len(features) != len(set(features)):
        raise ValueError("feature_names contains duplicates.")
    return canonical_json_hash(
        {
            "ordered_feature_names": features,
            "feature_count": len(features),
        }
    )


@dataclass(frozen=True)
class HashedFile:
    logical_name: str
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hash_files(
    paths: Sequence[Union[str, Path]],
    *,
    require_nonempty: bool = True,
) -> Tuple[List[HashedFile], str]:
    """
    Hash a deterministic ordered file set.

    The combined hash uses logical order, basename, size, and content hash. It
    does not depend on machine-specific absolute directory prefixes.
    """
    if require_nonempty and not paths:
        raise ValueError("At least one file path is required.")

    records: List[HashedFile] = []
    for index, raw_path in enumerate(paths):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            HashedFile(
                logical_name=f"file_{index:04d}:{path.name}",
                path=str(path),
                size_bytes=int(path.stat().st_size),
                sha256=sha256_file(path),
            )
        )

    combined = canonical_json_hash(
        [
            {
                "logical_name": record.logical_name,
                "basename": Path(record.path).name,
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
            }
            for record in records
        ]
    )
    return records, combined


def resolved_config_hash(
    resolved_config: Union[Mapping[str, Any], str, Path],
) -> Tuple[str, Optional[str]]:
    """
    Hash either an already merged configuration mapping or a saved config file.
    """
    if isinstance(resolved_config, Mapping):
        return canonical_json_hash(dict(resolved_config)), None

    path = Path(resolved_config).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path), str(path)


def _hash_source_tree(
    project_root: Path,
    code_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> str:
    project_root = Path(project_root).resolve()

    if code_paths is None:
        candidates: List[Path] = []
        for relative in ["src", "external_baselines", "configs"]:
            base = project_root / relative
            if not base.exists():
                continue
            candidates.extend(
                path
                for path in base.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".py", ".yaml", ".yml", ".json"}
                and "__pycache__" not in path.parts
            )
    else:
        candidates = []
        for raw in code_paths:
            path = Path(raw)
            if not path.is_absolute():
                path = project_root / path
            if path.is_file():
                candidates.append(path.resolve())
            elif path.is_dir():
                candidates.extend(
                    child
                    for child in path.rglob("*")
                    if child.is_file()
                    and child.suffix.lower() in {".py", ".yaml", ".yml", ".json"}
                    and "__pycache__" not in child.parts
                )
            else:
                raise FileNotFoundError(path)

    candidates = sorted(set(candidates), key=lambda item: str(item))
    if not candidates:
        return "source-tree-sha256:unavailable"

    records = []
    for path in candidates:
        try:
            relative = path.relative_to(project_root).as_posix()
        except ValueError:
            relative = path.name
        records.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    return f"source-tree-sha256:{canonical_json_hash(records)}"


def detect_code_version(
    project_root: Union[str, Path] = ".",
    code_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> str:
    """
    Prefer a Git commit identifier; fall back to a source-tree hash.
    """
    root = Path(project_root).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        state = "dirty" if status else "clean"
        if commit:
            return f"git:{commit}:{state}"
    except (OSError, subprocess.SubprocessError):
        pass

    return _hash_source_tree(root, code_paths=code_paths)


def _validate_seed_contract(
    active_seed: int,
    training_seed_list: Sequence[int],
) -> Tuple[int, List[int]]:
    active = int(active_seed)
    seeds = [int(seed) for seed in training_seed_list]
    if not seeds:
        raise ValueError("training_seed_list must not be empty.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("training_seed_list contains duplicate seeds.")
    if active not in seeds:
        raise ValueError(
            f"active_seed={active} is not present in training_seed_list={seeds}."
        )
    return active, seeds


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    created_at_utc: str

    model_name: str
    model_family: str
    input_representation: str
    split_name: str

    feature_names: List[str]
    feature_count: int
    feature_hash: str
    split_hash: str

    processed_data_files: List[HashedFile]
    processed_data_hash: str

    checkpoint_path: str
    checkpoint_hash: str

    resolved_config_path: Optional[str]
    resolved_config_hash: str

    active_seed: int
    training_seed_list: List[int]

    threshold_source: str
    theta: float
    persistence: int

    code_version: str
    parameter_count: int

    prediction_npz_path: str
    prediction_npz_hash: str
    prediction_metadata_path: str
    prediction_metadata_hash: str
    prediction_bundle_hash: str

    decision_score_type: str
    within_segment_index_source: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["processed_data_files"] = [
            item.to_dict() for item in self.processed_data_files
        ]
        return payload


def build_artifact_manifest(
    *,
    bundle: StandardizedPredictionBundle,
    prediction_artifact: SavedPredictionBundleArtifact,
    model_family: str,
    input_representation: str,
    feature_names: Sequence[str],
    processed_data_paths: Sequence[Union[str, Path]],
    checkpoint_path: Union[str, Path],
    resolved_config: Union[Mapping[str, Any], str, Path],
    active_seed: int,
    training_seed_list: Sequence[int],
    threshold_source: str,
    theta: float,
    persistence: int,
    parameter_count: int,
    code_version: Optional[str] = None,
    project_root: Union[str, Path] = ".",
    code_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> ArtifactManifest:
    """Build a complete strict manifest for one model/split result."""
    bundle = bundle.validated()

    model_family = str(model_family).strip()
    input_representation = str(input_representation).strip()
    threshold_source = str(threshold_source).strip()
    if not model_family:
        raise ValueError("model_family is empty.")
    if not input_representation:
        raise ValueError("input_representation is empty.")
    if not threshold_source:
        raise ValueError("threshold_source is empty.")

    features = [str(name).strip() for name in feature_names]
    feature_hash = feature_contract_hash(features)

    processed_files, processed_hash = hash_files(
        processed_data_paths,
        require_nonempty=True,
    )

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256_file(checkpoint)

    config_hash, config_path = resolved_config_hash(resolved_config)

    active, seeds = _validate_seed_contract(
        active_seed=active_seed,
        training_seed_list=training_seed_list,
    )

    theta = float(theta)
    persistence = int(persistence)
    if not math.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must be within [0, 1], got {theta}.")
    if persistence < 1:
        raise ValueError(
            f"persistence must be >= 1, got {persistence}."
        )

    parameter_count = int(parameter_count)
    if parameter_count < 0:
        raise ValueError("parameter_count must be nonnegative.")

    if Path(prediction_artifact.npz_path).resolve() != Path(
        prediction_artifact.npz_path
    ).resolve():
        raise AssertionError("Unexpected prediction path normalization failure.")

    if prediction_artifact.bundle_content_hash != bundle.content_hash():
        raise ValueError(
            "prediction_artifact bundle hash does not match supplied bundle."
        )
    if prediction_artifact.split_hash != bundle.split_identity_hash():
        raise ValueError(
            "prediction_artifact split hash does not match supplied bundle."
        )
    if bundle.checkpoint_path is not None:
        bundle_checkpoint = Path(bundle.checkpoint_path).resolve()
        if bundle_checkpoint != checkpoint:
            raise ValueError(
                "Bundle checkpoint_path does not match manifest checkpoint_path."
            )

    if code_version is None:
        code_version = detect_code_version(
            project_root=project_root,
            code_paths=code_paths,
        )
    code_version = str(code_version).strip()
    if not code_version:
        raise ValueError("code_version is empty.")

    return ArtifactManifest(
        schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        created_at_utc=datetime.now(timezone.utc).isoformat(),

        model_name=bundle.model_name,
        model_family=model_family,
        input_representation=input_representation,
        split_name=bundle.split_name,

        feature_names=features,
        feature_count=len(features),
        feature_hash=feature_hash,
        split_hash=bundle.split_identity_hash(),

        processed_data_files=processed_files,
        processed_data_hash=processed_hash,

        checkpoint_path=str(checkpoint),
        checkpoint_hash=checkpoint_hash,

        resolved_config_path=config_path,
        resolved_config_hash=config_hash,

        active_seed=active,
        training_seed_list=seeds,

        threshold_source=threshold_source,
        theta=theta,
        persistence=persistence,

        code_version=code_version,
        parameter_count=parameter_count,

        prediction_npz_path=str(
            Path(prediction_artifact.npz_path).resolve()
        ),
        prediction_npz_hash=prediction_artifact.npz_sha256,
        prediction_metadata_path=str(
            Path(prediction_artifact.metadata_path).resolve()
        ),
        prediction_metadata_hash=prediction_artifact.metadata_sha256,
        prediction_bundle_hash=prediction_artifact.bundle_content_hash,

        decision_score_type=bundle.decision_score_type,
        within_segment_index_source=(
            bundle.within_segment_index_source
        ),
    )


def save_artifact_manifest(
    manifest: ArtifactManifest,
    path: Union[str, Path],
) -> Path:
    path = Path(path)
    save_strict_json(manifest.to_dict(), path)
    return path


_REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "created_at_utc",
    "model_name",
    "model_family",
    "input_representation",
    "split_name",
    "feature_names",
    "feature_count",
    "feature_hash",
    "split_hash",
    "processed_data_files",
    "processed_data_hash",
    "checkpoint_path",
    "checkpoint_hash",
    "resolved_config_hash",
    "active_seed",
    "training_seed_list",
    "threshold_source",
    "theta",
    "persistence",
    "code_version",
    "parameter_count",
    "prediction_npz_path",
    "prediction_npz_hash",
    "prediction_metadata_path",
    "prediction_metadata_hash",
    "prediction_bundle_hash",
}


def verify_artifact_manifest(
    manifest_path: Union[str, Path],
    *,
    verify_files: bool = True,
) -> Dict[str, Any]:
    """Validate required fields and recompute all available hashes."""
    manifest_path = Path(manifest_path)
    payload = load_json(manifest_path)

    missing = sorted(_REQUIRED_MANIFEST_KEYS - set(payload))
    if missing:
        raise KeyError(f"Manifest is missing required fields: {missing}")

    if payload["schema_version"] != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema: {payload['schema_version']!r}."
        )

    if int(payload["feature_count"]) != len(payload["feature_names"]):
        raise ValueError("feature_count does not match feature_names.")
    if feature_contract_hash(payload["feature_names"]) != payload["feature_hash"]:
        raise ValueError("Feature contract hash mismatch.")

    _validate_seed_contract(
        active_seed=int(payload["active_seed"]),
        training_seed_list=payload["training_seed_list"],
    )

    if not verify_files:
        return {
            "status": "PASSED",
            "manifest_path": str(manifest_path.resolve()),
            "files_verified": False,
        }

    prediction_npz = Path(payload["prediction_npz_path"])
    prediction_metadata = Path(payload["prediction_metadata_path"])
    checkpoint = Path(payload["checkpoint_path"])

    if sha256_file(prediction_npz) != payload["prediction_npz_hash"]:
        raise ValueError("Prediction NPZ hash mismatch.")
    if sha256_file(prediction_metadata) != payload["prediction_metadata_hash"]:
        raise ValueError("Prediction metadata hash mismatch.")
    if sha256_file(checkpoint) != payload["checkpoint_hash"]:
        raise ValueError("Checkpoint hash mismatch.")

    processed_paths = [
        Path(record["path"])
        for record in payload["processed_data_files"]
    ]
    processed_records, processed_hash = hash_files(
        processed_paths,
        require_nonempty=True,
    )
    if processed_hash != payload["processed_data_hash"]:
        raise ValueError("Processed-data combined hash mismatch.")

    for expected, actual in zip(
        payload["processed_data_files"],
        processed_records,
    ):
        if expected["sha256"] != actual.sha256:
            raise ValueError(
                f"Processed-data file hash mismatch: {actual.path}"
            )

    config_path = payload.get("resolved_config_path")
    if config_path:
        if sha256_file(Path(config_path)) != payload["resolved_config_hash"]:
            raise ValueError("Resolved-config file hash mismatch.")

    return {
        "status": "PASSED",
        "manifest_path": str(manifest_path.resolve()),
        "files_verified": True,
        "feature_hash": payload["feature_hash"],
        "split_hash": payload["split_hash"],
        "processed_data_hash": payload["processed_data_hash"],
        "checkpoint_hash": payload["checkpoint_hash"],
        "resolved_config_hash": payload["resolved_config_hash"],
        "prediction_bundle_hash": payload["prediction_bundle_hash"],
    }


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "ArtifactManifest",
    "HashedFile",
    "build_artifact_manifest",
    "canonical_json_hash",
    "detect_code_version",
    "feature_contract_hash",
    "hash_files",
    "load_json",
    "resolved_config_hash",
    "save_artifact_manifest",
    "save_strict_json",
    "verify_artifact_manifest",
]
