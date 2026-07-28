"""
Standard prediction-bundle contract and adapters for all learning-based models.

This module does not modify the legacy prediction classes. It creates a strict
artifact contract that can adapt existing PyTorch/XGBoost bundles, future
Causal Transformer/first-order liquid bundles, and GPS-IDS classifier outputs.

Required row-level fields
-------------------------
- probability
- logit or decision score (optional only when a model exposes probability alone)
- label
- valid_mask
- segment_id
- row_index
- within_segment_index
- delta_t

Required metadata
-----------------
- split
- model_name
- checkpoint_path

The contract rejects inconsistent lengths, invalid labels/masks, nonfinite
values, duplicate row identities, repeated noncontiguous segments, and
nonchronological row/index ordering.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.unified_alarm_rules import (
    as_binary_mask,
    as_probabilities,
    as_segment_ids,
    validate_equal_lengths,
)
from src.evaluation.unified_event_metrics import (
    as_binary_labels,
    as_delta_t,
)
from src.evaluation.unified_evaluator import UnifiedPredictionBundle


PREDICTION_BUNDLE_SCHEMA_VERSION = "av_gps_prediction_bundle_v1"

_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "probabilities": ("probability", "probabilities", "probs", "y_probability"),
    "decision_scores": (
        "decision_score",
        "decision_scores",
        "logit",
        "logits",
        "score",
        "scores",
    ),
    "labels": ("label", "labels", "target", "targets", "y_true"),
    "valid_mask": ("valid_mask", "validity_mask", "mask", "xi_nu"),
    "segment_ids": ("segment_id", "segment_ids", "segments"),
    "row_indices": ("row_index", "row_indices", "rows"),
    "within_segment_indices": (
        "within_segment_index",
        "within_segment_indices",
        "sequence_index",
        "sequence_indices",
    ),
    "delta_t": ("delta_t", "delta_t_seconds", "time_delta"),
    "split_name": ("split", "split_name", "xi_split"),
    "model_name": ("model_name", "method", "method_name"),
    "checkpoint_path": ("checkpoint_path", "checkpoint", "model_path"),
}


def _strict_json_value(value: Any) -> Any:
    """Convert a value into strict JSON-compatible content."""
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


def _save_strict_json(payload: Mapping[str, Any], path: Path) -> Path:
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


def _as_integer_vector(
    values: Sequence[Any],
    name: str,
    *,
    nonnegative: bool,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    numeric = arr.astype(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite values.")
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded, atol=0.0, rtol=0.0):
        raise ValueError(f"{name} must contain integer-valued entries.")
    integer = rounded.astype(np.int64)
    if nonnegative and np.any(integer < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return integer


def _as_decision_scores(
    values: Optional[Sequence[Any]],
    expected_length: int,
) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = arr.reshape(-1)
    if arr.size != expected_length:
        raise ValueError(
            "decision_scores length mismatch: "
            f"expected {expected_length}, got {arr.size}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("decision_scores contains non-finite values.")
    return arr.astype(np.float64, copy=False)


def _validate_chronological_identity(
    segment_ids: np.ndarray,
    row_indices: np.ndarray,
    within_segment_indices: np.ndarray,
) -> None:
    """Validate unique and strictly increasing identities within each segment."""
    row_keys = [
        (str(segment), int(row))
        for segment, row in zip(segment_ids, row_indices)
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError(
            "Duplicate (segment_id, row_index) identities found. "
            "Overlapping-window predictions must be deduplicated before "
            "standardization."
        )

    within_keys = [
        (str(segment), int(index))
        for segment, index in zip(segment_ids, within_segment_indices)
    ]
    if len(within_keys) != len(set(within_keys)):
        raise ValueError(
            "Duplicate (segment_id, within_segment_index) identities found."
        )

    start = 0
    while start < segment_ids.size:
        segment = segment_ids[start]
        end = start + 1
        while end < segment_ids.size and segment_ids[end] == segment:
            end += 1

        segment_rows = row_indices[start:end]
        segment_within = within_segment_indices[start:end]

        if segment_rows.size > 1 and np.any(np.diff(segment_rows) <= 0):
            raise ValueError(
                f"row_index is not strictly increasing inside segment {segment!r}."
            )
        if segment_within.size > 1 and np.any(np.diff(segment_within) <= 0):
            raise ValueError(
                "within_segment_index is not strictly increasing inside "
                f"segment {segment!r}."
            )

        start = end


def infer_within_segment_indices(segment_ids: Sequence[Any]) -> np.ndarray:
    """
    Infer zero-based chronological positions inside contiguous segment blocks.

    This should be used only for already-deduplicated prediction bundles whose
    ordering is known to be chronological. The adapter records that inference
    occurred.
    """
    segments = as_segment_ids(segment_ids)
    output = np.zeros(segments.size, dtype=np.int64)
    current = 0
    previous: Optional[str] = None

    for position, segment in enumerate(segments):
        segment = str(segment)
        if segment != previous:
            current = 0
            previous = segment
        output[position] = current
        current += 1

    return output


def _update_hash_with_array(
    digest: "hashlib._Hash",
    name: str,
    array: np.ndarray,
) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")

    arr = np.asarray(array)
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(b"\0")

    if arr.dtype.kind in {"O", "U", "S"}:
        for item in arr.reshape(-1):
            encoded = str(item).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="little"))
            digest.update(encoded)
    else:
        contiguous = np.ascontiguousarray(arr)
        digest.update(contiguous.tobytes(order="C"))


@dataclass(frozen=True)
class StandardizedPredictionBundle:
    """Strict model-agnostic prediction artifact."""

    split_name: str
    model_name: str
    probabilities: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    segment_ids: np.ndarray
    row_indices: np.ndarray
    within_segment_indices: np.ndarray
    delta_t: np.ndarray

    decision_scores: Optional[np.ndarray] = None
    decision_score_type: str = "probability_only"
    checkpoint_path: Optional[str] = None
    within_segment_index_source: str = "provided"

    def validated(self) -> "StandardizedPredictionBundle":
        split_name = str(self.split_name).strip()
        model_name = str(self.model_name).strip()
        if not split_name:
            raise ValueError("split_name is empty.")
        if not model_name:
            raise ValueError("model_name is empty.")

        probabilities = as_probabilities(self.probabilities)
        labels = as_binary_labels(self.labels)
        valid_mask = as_binary_mask(self.valid_mask, "valid_mask")
        segment_ids = as_segment_ids(self.segment_ids)
        row_indices = _as_integer_vector(
            self.row_indices,
            "row_indices",
            nonnegative=True,
        )
        within_segment_indices = _as_integer_vector(
            self.within_segment_indices,
            "within_segment_indices",
            nonnegative=True,
        )
        delta_t = as_delta_t(self.delta_t)

        validate_equal_lengths(
            probabilities=probabilities,
            labels=labels,
            valid_mask=valid_mask,
            segment_ids=segment_ids,
            row_indices=row_indices,
            within_segment_indices=within_segment_indices,
            delta_t=delta_t,
        )

        decision_scores = _as_decision_scores(
            self.decision_scores,
            expected_length=probabilities.size,
        )

        decision_score_type = str(self.decision_score_type).strip()
        if decision_scores is None:
            decision_score_type = "probability_only"
        elif not decision_score_type or decision_score_type == "probability_only":
            raise ValueError(
                "A nonempty decision_score_type such as 'logit' or "
                "'decision_function' is required when decision_scores are supplied."
            )

        index_source = str(self.within_segment_index_source).strip().lower()
        if index_source not in {"provided", "inferred"}:
            raise ValueError(
                "within_segment_index_source must be 'provided' or 'inferred'."
            )

        checkpoint = (
            None
            if self.checkpoint_path is None
            else str(self.checkpoint_path).strip()
        )
        if checkpoint == "":
            checkpoint = None

        _validate_chronological_identity(
            segment_ids=segment_ids,
            row_indices=row_indices,
            within_segment_indices=within_segment_indices,
        )

        return StandardizedPredictionBundle(
            split_name=split_name,
            model_name=model_name,
            probabilities=probabilities,
            labels=labels,
            valid_mask=valid_mask,
            segment_ids=segment_ids,
            row_indices=row_indices,
            within_segment_indices=within_segment_indices,
            delta_t=delta_t,
            decision_scores=decision_scores,
            decision_score_type=decision_score_type,
            checkpoint_path=checkpoint,
            within_segment_index_source=index_source,
        )

    @property
    def row_count(self) -> int:
        return int(self.validated().probabilities.size)

    def content_hash(self) -> str:
        """Hash metadata and every standardized row-level array."""
        bundle = self.validated()
        digest = hashlib.sha256()

        for name, value in [
            ("schema_version", PREDICTION_BUNDLE_SCHEMA_VERSION),
            ("split_name", bundle.split_name),
            ("model_name", bundle.model_name),
            ("decision_score_type", bundle.decision_score_type),
            ("checkpoint_path", bundle.checkpoint_path or ""),
            (
                "within_segment_index_source",
                bundle.within_segment_index_source,
            ),
        ]:
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")

        arrays = bundle.to_array_mapping()
        for name in sorted(arrays):
            _update_hash_with_array(digest, name, arrays[name])

        return digest.hexdigest()

    def split_identity_hash(self) -> str:
        """
        Hash the exact split composition and chronological identities.

        Labels and validity are included because changing either changes the
        evaluable split definition.
        """
        bundle = self.validated()
        digest = hashlib.sha256()
        digest.update(bundle.split_name.encode("utf-8"))
        digest.update(b"\0")

        arrays = {
            "segment_id": bundle.segment_ids,
            "row_index": bundle.row_indices,
            "within_segment_index": bundle.within_segment_indices,
            "label": bundle.labels,
            "valid_mask": bundle.valid_mask.astype(np.int8),
            "delta_t": bundle.delta_t,
        }
        for name in sorted(arrays):
            _update_hash_with_array(digest, name, arrays[name])
        return digest.hexdigest()

    def to_array_mapping(self) -> Dict[str, np.ndarray]:
        bundle = self.validated()
        arrays: Dict[str, np.ndarray] = {
            "probability": bundle.probabilities.astype(np.float64),
            "label": bundle.labels.astype(np.int8),
            "valid_mask": bundle.valid_mask.astype(np.int8),
            "segment_id": bundle.segment_ids.astype(str),
            "row_index": bundle.row_indices.astype(np.int64),
            "within_segment_index": (
                bundle.within_segment_indices.astype(np.int64)
            ),
            "delta_t": bundle.delta_t.astype(np.float64),
        }
        if bundle.decision_scores is not None:
            arrays["decision_score"] = bundle.decision_scores.astype(np.float64)
        return arrays

    def metadata_dict(self) -> Dict[str, Any]:
        bundle = self.validated()
        return {
            "schema_version": PREDICTION_BUNDLE_SCHEMA_VERSION,
            "split": bundle.split_name,
            "model_name": bundle.model_name,
            "checkpoint_path": bundle.checkpoint_path,
            "decision_score_type": bundle.decision_score_type,
            "within_segment_index_source": (
                bundle.within_segment_index_source
            ),
            "row_count": int(bundle.probabilities.size),
            "valid_row_count": int(bundle.valid_mask.sum()),
            "segment_count": int(
                1
                + np.sum(
                    bundle.segment_ids[1:] != bundle.segment_ids[:-1]
                )
            ),
            "bundle_content_hash": bundle.content_hash(),
            "split_hash": bundle.split_identity_hash(),
            "field_contract": [
                "probability",
                "decision_score",
                "label",
                "valid_mask",
                "segment_id",
                "row_index",
                "within_segment_index",
                "delta_t",
                "split",
                "model_name",
                "checkpoint_path",
            ],
        }

    def to_unified_prediction_bundle(self) -> UnifiedPredictionBundle:
        """Convert into the Step-2 evaluator contract."""
        bundle = self.validated()
        return UnifiedPredictionBundle(
            split_name=bundle.split_name,
            model_name=bundle.model_name,
            probabilities=bundle.probabilities,
            labels=bundle.labels,
            valid_mask=bundle.valid_mask,
            segment_ids=bundle.segment_ids,
            delta_t=bundle.delta_t,
            row_indices=bundle.row_indices,
            logits=bundle.decision_scores,
            checkpoint_path=bundle.checkpoint_path,
        )


def _resolve_mapping_alias(
    mapping: Mapping[str, Any],
    canonical_name: str,
    *,
    required: bool,
) -> Any:
    aliases = _FIELD_ALIASES[canonical_name]
    present = [name for name in aliases if name in mapping]

    if len(present) > 1:
        raise ValueError(
            f"Ambiguous aliases for {canonical_name}: {present}. "
            "Supply exactly one canonical/alias field."
        )
    if not present:
        if required:
            raise KeyError(
                f"Missing required field {canonical_name}; accepted aliases: "
                f"{aliases}."
            )
        return None
    return mapping[present[0]]


def adapt_prediction_mapping(
    mapping: Mapping[str, Any],
    *,
    split_name: Optional[str] = None,
    model_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    within_segment_indices: Optional[Sequence[Any]] = None,
    allow_infer_within_segment_index: bool = False,
    decision_score_type: str = "logit",
) -> StandardizedPredictionBundle:
    """
    Adapt a dictionary-like model output into the strict standard contract.
    """
    probabilities = _resolve_mapping_alias(
        mapping,
        "probabilities",
        required=True,
    )
    labels = _resolve_mapping_alias(mapping, "labels", required=True)
    valid_mask = _resolve_mapping_alias(
        mapping,
        "valid_mask",
        required=True,
    )
    segment_ids = _resolve_mapping_alias(
        mapping,
        "segment_ids",
        required=True,
    )
    row_indices = _resolve_mapping_alias(
        mapping,
        "row_indices",
        required=True,
    )
    delta_t = _resolve_mapping_alias(mapping, "delta_t", required=True)

    decision_scores = _resolve_mapping_alias(
        mapping,
        "decision_scores",
        required=False,
    )

    if split_name is None:
        split_name = _resolve_mapping_alias(
            mapping,
            "split_name",
            required=True,
        )
    if model_name is None:
        model_name = _resolve_mapping_alias(
            mapping,
            "model_name",
            required=True,
        )
    if checkpoint_path is None:
        checkpoint_path = _resolve_mapping_alias(
            mapping,
            "checkpoint_path",
            required=False,
        )

    index_source = "provided"
    if within_segment_indices is None:
        within_segment_indices = _resolve_mapping_alias(
            mapping,
            "within_segment_indices",
            required=False,
        )

    if within_segment_indices is None:
        if not allow_infer_within_segment_index:
            raise KeyError(
                "within_segment_index is required. Set "
                "allow_infer_within_segment_index=True only for a verified, "
                "chronologically ordered, deduplicated source bundle."
            )
        within_segment_indices = infer_within_segment_indices(segment_ids)
        index_source = "inferred"

    if decision_scores is None:
        decision_score_type = "probability_only"

    return StandardizedPredictionBundle(
        split_name=str(split_name),
        model_name=str(model_name),
        probabilities=np.asarray(probabilities),
        decision_scores=(
            None
            if decision_scores is None
            else np.asarray(decision_scores)
        ),
        decision_score_type=str(decision_score_type),
        labels=np.asarray(labels),
        valid_mask=np.asarray(valid_mask),
        segment_ids=np.asarray(segment_ids),
        row_indices=np.asarray(row_indices),
        within_segment_indices=np.asarray(within_segment_indices),
        delta_t=np.asarray(delta_t),
        checkpoint_path=checkpoint_path,
        within_segment_index_source=index_source,
    ).validated()


def adapt_existing_prediction_bundle(
    source_bundle: Any,
    *,
    split_name: str,
    within_segment_indices: Optional[Sequence[Any]] = None,
    allow_infer_within_segment_index: bool = False,
    decision_score_type: str = "logit",
) -> StandardizedPredictionBundle:
    """
    Adapt the project's existing EvaluationPredictionBundle-like objects.

    Expected source attributes include probabilities, labels, valid_mask,
    segment_ids, row_indices, delta_t, model_name, checkpoint_path, and
    optionally logits.
    """
    if isinstance(source_bundle, Mapping):
        mapping = dict(source_bundle)
    elif hasattr(source_bundle, "__dict__"):
        mapping = vars(source_bundle)
    else:
        attribute_names = {
            alias
            for aliases in _FIELD_ALIASES.values()
            for alias in aliases
        }
        mapping = {
            name: getattr(source_bundle, name)
            for name in attribute_names
            if hasattr(source_bundle, name)
        }

    mapping = dict(mapping)
    mapping["split_name"] = split_name

    return adapt_prediction_mapping(
        mapping,
        within_segment_indices=within_segment_indices,
        allow_infer_within_segment_index=(
            allow_infer_within_segment_index
        ),
        decision_score_type=decision_score_type,
    )


@dataclass(frozen=True)
class SavedPredictionBundleArtifact:
    npz_path: str
    metadata_path: str
    npz_sha256: str
    metadata_sha256: str
    bundle_content_hash: str
    split_hash: str
    row_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_standardized_prediction_bundle(
    bundle: StandardizedPredictionBundle,
    npz_path: Path,
    metadata_path: Optional[Path] = None,
) -> SavedPredictionBundleArtifact:
    """Save a strict no-pickle NPZ plus strict JSON metadata."""
    bundle = bundle.validated()
    npz_path = Path(npz_path)
    if npz_path.suffix.lower() != ".npz":
        raise ValueError(f"Prediction bundle path must end with .npz: {npz_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata_path is None:
        metadata_path = npz_path.with_suffix(".metadata.json")
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(npz_path, **bundle.to_array_mapping())
    _save_strict_json(bundle.metadata_dict(), metadata_path)

    artifact = SavedPredictionBundleArtifact(
        npz_path=str(npz_path.resolve()),
        metadata_path=str(metadata_path.resolve()),
        npz_sha256=sha256_file(npz_path),
        metadata_sha256=sha256_file(metadata_path),
        bundle_content_hash=bundle.content_hash(),
        split_hash=bundle.split_identity_hash(),
        row_count=int(bundle.probabilities.size),
    )
    return artifact


def load_standardized_prediction_bundle(
    npz_path: Path,
    metadata_path: Optional[Path] = None,
    *,
    verify_metadata_hashes: bool = True,
) -> StandardizedPredictionBundle:
    """Load and validate a standard prediction artifact without pickle."""
    npz_path = Path(npz_path)
    if metadata_path is None:
        metadata_path = npz_path.with_suffix(".metadata.json")
    metadata_path = Path(metadata_path)

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if metadata.get("schema_version") != PREDICTION_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported prediction-bundle schema: "
            f"{metadata.get('schema_version')!r}."
        )

    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "probability",
            "label",
            "valid_mask",
            "segment_id",
            "row_index",
            "within_segment_index",
            "delta_t",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"Prediction NPZ missing arrays: {missing}")

        decision_scores = (
            archive["decision_score"]
            if "decision_score" in archive.files
            else None
        )

        bundle = StandardizedPredictionBundle(
            split_name=str(metadata["split"]),
            model_name=str(metadata["model_name"]),
            probabilities=archive["probability"],
            decision_scores=decision_scores,
            decision_score_type=str(
                metadata.get(
                    "decision_score_type",
                    "probability_only",
                )
            ),
            labels=archive["label"],
            valid_mask=archive["valid_mask"],
            segment_ids=archive["segment_id"],
            row_indices=archive["row_index"],
            within_segment_indices=archive["within_segment_index"],
            delta_t=archive["delta_t"],
            checkpoint_path=metadata.get("checkpoint_path"),
            within_segment_index_source=str(
                metadata.get(
                    "within_segment_index_source",
                    "provided",
                )
            ),
        ).validated()

    if int(metadata.get("row_count", -1)) != bundle.row_count:
        raise ValueError(
            "Prediction metadata row_count does not match loaded arrays."
        )

    if verify_metadata_hashes:
        expected_bundle_hash = str(
            metadata.get("bundle_content_hash", "")
        )
        expected_split_hash = str(metadata.get("split_hash", ""))
        if expected_bundle_hash != bundle.content_hash():
            raise ValueError(
                "Prediction bundle content hash mismatch; the artifact or "
                "metadata may have been altered."
            )
        if expected_split_hash != bundle.split_identity_hash():
            raise ValueError(
                "Prediction split hash mismatch; split identity may have "
                "changed."
            )

    return bundle


def verify_saved_prediction_bundle(
    artifact: SavedPredictionBundleArtifact,
) -> Dict[str, Any]:
    """Recompute file/content hashes and load the artifact strictly."""
    npz_path = Path(artifact.npz_path)
    metadata_path = Path(artifact.metadata_path)

    if sha256_file(npz_path) != artifact.npz_sha256:
        raise ValueError("Prediction NPZ file hash mismatch.")
    if sha256_file(metadata_path) != artifact.metadata_sha256:
        raise ValueError("Prediction metadata file hash mismatch.")

    bundle = load_standardized_prediction_bundle(
        npz_path=npz_path,
        metadata_path=metadata_path,
        verify_metadata_hashes=True,
    )
    if bundle.content_hash() != artifact.bundle_content_hash:
        raise ValueError("Prediction bundle content hash mismatch.")
    if bundle.split_identity_hash() != artifact.split_hash:
        raise ValueError("Prediction split hash mismatch.")

    return {
        "status": "PASSED",
        "npz_path": str(npz_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "row_count": bundle.row_count,
        "bundle_content_hash": bundle.content_hash(),
        "split_hash": bundle.split_identity_hash(),
    }


__all__ = [
    "PREDICTION_BUNDLE_SCHEMA_VERSION",
    "SavedPredictionBundleArtifact",
    "StandardizedPredictionBundle",
    "adapt_existing_prediction_bundle",
    "adapt_prediction_mapping",
    "infer_within_segment_indices",
    "load_standardized_prediction_bundle",
    "save_standardized_prediction_bundle",
    "sha256_file",
    "verify_saved_prediction_bundle",
]
