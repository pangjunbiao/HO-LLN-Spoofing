"""
Baseline factory for Step 15.

Step 15 purpose:
- Provide a single interface for training/loading/predicting all official baselines:
  1. XGBoost-xi
  2. MLP-xi
  3. LSTM-xi
  4. GRU-xi
  5. Causal-TCN-xi

Important:
- All baselines use the same reconstructed xi_t feature columns.
- No baseline uses raw shortcut columns.
- No baseline uses EKF Detector as input.
- Threshold and persistence selection is handled later by run_baselines.py using
  Dataset-1 validation only.
- Official proposed model comparison remains fair only if all baselines use the
  same validation threshold-selection protocol.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

from src.baselines.gru_baseline import (
    GRUBaselineArtifact,
    collect_gru_predictions,
    load_gru_baseline,
    train_gru_baseline,
)
from src.baselines.lstm_baseline import (
    LSTMBaselineArtifact,
    collect_lstm_predictions,
    load_lstm_baseline,
    train_lstm_baseline,
)
from src.baselines.mlp_baseline import (
    MLPBaselineArtifact,
    collect_mlp_predictions,
    load_mlp_baseline,
    train_mlp_baseline,
)
from src.baselines.tcn_baseline import (
    TCNBaselineArtifact,
    collect_tcn_predictions,
    load_tcn_baseline,
    train_tcn_baseline,
)
from src.baselines.xgboost_baseline import (
    XGBoostBaselineArtifact,
    collect_xgboost_predictions,
    get_baseline_feature_columns,
    train_xgboost_baseline,
    load_xgboost_baseline,
)
from src.evaluation.evaluate_dataset1 import EvaluationPredictionBundle
from src.utils.config import get_by_path, resolve_project_path


SUPPORTED_BASELINE_KEYS = [
    "xgboost_xi",
    "mlp_xi",
    "lstm_xi",
    "gru_xi",
    "tcn_xi",
]


BASELINE_DISPLAY_NAMES = {
    "xgboost_xi": "XGBoost-xi",
    "mlp_xi": "MLP-xi",
    "lstm_xi": "LSTM-xi",
    "gru_xi": "GRU-xi",
    "tcn_xi": "Causal-TCN-xi",
}


DEFAULT_BASELINE_SPLITS = [
    "val",
    "test",
    "external",
    "online",
]


@dataclass
class BaselineFactorySpec:
    """One baseline factory specification."""

    key: str
    display_name: str
    train_function_name: str
    load_function_name: str
    collect_function_name: str
    checkpoint_config_path: str
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineRuntimeRecord:
    """Runtime record for one trained/loaded baseline."""

    key: str
    display_name: str
    action: str
    status: str
    runtime_seconds: float
    model_path: Optional[str]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselinePredictionRecord:
    """Prediction collection record for one baseline and one split."""

    key: str
    display_name: str
    split_name: str
    status: str
    prediction_count: int
    valid_count: int
    attack_valid_count: int
    normal_valid_count: int
    runtime_seconds: float
    checkpoint_path: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve project-relative path."""
    return resolve_project_path(config, str(value))


def normalize_baseline_key(name: str) -> str:
    """Normalize baseline aliases to canonical keys."""
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "xgboost": "xgboost_xi",
        "xgb": "xgboost_xi",
        "xgb_xi": "xgboost_xi",
        "xgboost_xi": "xgboost_xi",
        "mlp": "mlp_xi",
        "mlp_xi": "mlp_xi",
        "lstm": "lstm_xi",
        "lstm_xi": "lstm_xi",
        "gru": "gru_xi",
        "gru_xi": "gru_xi",
        "tcn": "tcn_xi",
        "causal_tcn": "tcn_xi",
        "causal_tcn_xi": "tcn_xi",
        "tcn_xi": "tcn_xi",
    }

    if text not in aliases:
        raise KeyError(
            f"Unsupported baseline key '{name}'. "
            f"Supported baselines: {SUPPORTED_BASELINE_KEYS}"
        )

    return aliases[text]


def get_baseline_checkpoint_path(
    config: Mapping[str, Any],
    baseline_key: str,
) -> Path:
    """Return configured checkpoint/model path for a baseline."""
    baseline_key = normalize_baseline_key(baseline_key)

    default_paths = {
        "xgboost_xi": "results/models/xgboost_xi.pkl",
        "mlp_xi": "results/models/mlp_xi.pt",
        "lstm_xi": "results/models/lstm_xi.pt",
        "gru_xi": "results/models/gru_xi.pt",
        "tcn_xi": "results/models/tcn_xi.pt",
    }

    value = get_by_path(
        config,
        f"baselines.{baseline_key}.output_model_path",
        default_paths[baseline_key],
    )

    return _project_path(config, value)


def make_baseline_factory_specs(
    config: Mapping[str, Any],
) -> Dict[str, BaselineFactorySpec]:
    """Build baseline factory specifications."""
    specs: Dict[str, BaselineFactorySpec] = {}

    for key in SUPPORTED_BASELINE_KEYS:
        enabled = bool(get_by_path(config, f"baselines.{key}.enabled", True))

        specs[key] = BaselineFactorySpec(
            key=key,
            display_name=str(
                get_by_path(
                    config,
                    f"baselines.{key}.model_name",
                    BASELINE_DISPLAY_NAMES[key],
                )
            ),
            train_function_name=f"train_{key}_baseline",
            load_function_name=f"load_{key}_baseline",
            collect_function_name=f"collect_{key}_predictions",
            checkpoint_config_path=f"baselines.{key}.output_model_path",
            enabled=enabled,
        )

    return specs


def get_enabled_baseline_keys(config: Mapping[str, Any]) -> List[str]:
    """
    Return enabled baseline keys.

    Priority:
    1. baselines.enabled_models if explicitly set.
    2. baselines.step15.enabled_models if explicitly set.
    3. all supported baselines with per-model enabled=true.
    """
    explicit = get_by_path(config, "baselines.enabled_models", None)

    if explicit is None:
        explicit = get_by_path(config, "baselines.step15.enabled_models", None)

    specs = make_baseline_factory_specs(config)

    if explicit is not None:
        keys = [normalize_baseline_key(item) for item in list(explicit)]
        return [key for key in keys if specs[key].enabled]

    return [key for key in SUPPORTED_BASELINE_KEYS if specs[key].enabled]


def get_step15_retrain_policy(config: Mapping[str, Any]) -> str:
    """
    Return Step-15 retrain policy.

    Supported:
    - reuse_if_exists
    - always
    - never
    """
    policy = str(
        get_by_path(
            config,
            "experiments.step15.retrain_policy",
            get_by_path(config, "baselines.step15.retrain_policy", "reuse_if_exists"),
        )
    ).strip().lower()

    if policy not in {"reuse_if_exists", "always", "never"}:
        raise ValueError(
            f"Unsupported Step-15 retrain_policy='{policy}'. "
            "Use one of: reuse_if_exists, always, never."
        )

    return policy


def get_step15_evaluation_splits(config: Mapping[str, Any]) -> List[str]:
    """Return Step-15 prediction/evaluation splits."""
    splits = list(
        get_by_path(
            config,
            "experiments.step15.evaluation_splits",
            get_by_path(config, "baselines.step15.evaluation_splits", DEFAULT_BASELINE_SPLITS),
        )
    )

    normalized = []
    for split in splits:
        split = str(split)
        if split == "validation":
            split = "val"
        normalized.append(split)

    return normalized


def baseline_checkpoint_exists(
    config: Mapping[str, Any],
    baseline_key: str,
) -> bool:
    """Return True if saved baseline model exists."""
    return get_baseline_checkpoint_path(config, baseline_key).exists()


def train_baseline(
    baseline_key: str,
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
) -> Any:
    """Train one baseline."""
    baseline_key = normalize_baseline_key(baseline_key)

    if baseline_key == "xgboost_xi":
        return train_xgboost_baseline(
            config=config,
            active_seed=active_seed,
        )

    if baseline_key == "mlp_xi":
        return train_mlp_baseline(
            config=config,
            active_seed=active_seed,
            device=device,
        )

    if baseline_key == "lstm_xi":
        return train_lstm_baseline(
            config=config,
            active_seed=active_seed,
            device=device,
        )

    if baseline_key == "gru_xi":
        return train_gru_baseline(
            config=config,
            active_seed=active_seed,
            device=device,
        )

    if baseline_key == "tcn_xi":
        return train_tcn_baseline(
            config=config,
            active_seed=active_seed,
            device=device,
        )

    raise KeyError(f"Unsupported baseline key: {baseline_key}")


def load_baseline(
    baseline_key: str,
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
) -> Any:
    """Load one saved baseline."""
    baseline_key = normalize_baseline_key(baseline_key)

    if baseline_key == "xgboost_xi":
        return load_xgboost_baseline(
            config=config,
            active_seed=active_seed,
        )

    if baseline_key == "mlp_xi":
        return load_mlp_baseline(
            config=config,
            active_seed=active_seed,
            device=device,
        )

    if baseline_key == "lstm_xi":
        return load_lstm_baseline(
            config=config,
            active_seed=active_seed,
            device=device,
        )

    if baseline_key == "gru_xi":
        return load_gru_baseline(
            config=config,
            active_seed=active_seed,
            device=device,
        )

    if baseline_key == "tcn_xi":
        return load_tcn_baseline(
            config=config,
            active_seed=active_seed,
            device=device,
        )

    raise KeyError(f"Unsupported baseline key: {baseline_key}")


def train_or_load_baseline(
    baseline_key: str,
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
    retrain_policy: Optional[str] = None,
) -> tuple[Any, BaselineRuntimeRecord]:
    """
    Train or load one baseline according to retrain policy.

    Returns:
        artifact, runtime_record
    """
    baseline_key = normalize_baseline_key(baseline_key)
    display_name = BASELINE_DISPLAY_NAMES[baseline_key]
    model_path = get_baseline_checkpoint_path(config, baseline_key)

    if retrain_policy is None:
        retrain_policy = get_step15_retrain_policy(config)

    retrain_policy = str(retrain_policy).strip().lower()

    start = time.perf_counter()

    if retrain_policy == "always":
        artifact = train_baseline(
            baseline_key=baseline_key,
            config=config,
            active_seed=active_seed,
            device=device,
        )
        action = "trained"

    elif retrain_policy == "reuse_if_exists":
        if model_path.exists():
            artifact = load_baseline(
                baseline_key=baseline_key,
                config=config,
                active_seed=active_seed,
                device=device,
            )
            action = "loaded_existing"
        else:
            artifact = train_baseline(
                baseline_key=baseline_key,
                config=config,
                active_seed=active_seed,
                device=device,
            )
            action = "trained_missing"

    elif retrain_policy == "never":
        if not model_path.exists():
            raise FileNotFoundError(
                f"Baseline checkpoint missing and retrain_policy='never': {model_path}"
            )

        artifact = load_baseline(
            baseline_key=baseline_key,
            config=config,
            active_seed=active_seed,
            device=device,
        )
        action = "loaded_existing"

    else:
        raise ValueError(
            f"Unsupported retrain_policy='{retrain_policy}'. "
            "Use one of: reuse_if_exists, always, never."
        )

    runtime = float(time.perf_counter() - start)

    record = BaselineRuntimeRecord(
        key=baseline_key,
        display_name=getattr(artifact, "model_name", display_name),
        action=action,
        status="PASSED",
        runtime_seconds=runtime,
        model_path=str(getattr(artifact, "model_path", model_path)),
        message="",
    )

    return artifact, record


def collect_baseline_predictions(
    baseline_key: str,
    artifact: Any,
    config: Mapping[str, Any],
    split_name: str,
    device: Optional[torch.device] = None,
) -> EvaluationPredictionBundle:
    """Collect predictions from one baseline on one split."""
    baseline_key = normalize_baseline_key(baseline_key)

    if baseline_key == "xgboost_xi":
        return collect_xgboost_predictions(
            config=config,
            artifact=artifact,
            split_name=split_name,
        )

    if baseline_key == "mlp_xi":
        return collect_mlp_predictions(
            config=config,
            artifact=artifact,
            split_name=split_name,
            device=device,
        )

    if baseline_key == "lstm_xi":
        return collect_lstm_predictions(
            config=config,
            artifact=artifact,
            split_name=split_name,
            device=device,
        )

    if baseline_key == "gru_xi":
        return collect_gru_predictions(
            config=config,
            artifact=artifact,
            split_name=split_name,
            device=device,
        )

    if baseline_key == "tcn_xi":
        return collect_tcn_predictions(
            config=config,
            artifact=artifact,
            split_name=split_name,
            device=device,
        )

    raise KeyError(f"Unsupported baseline key: {baseline_key}")


def collect_baseline_predictions_for_splits(
    baseline_key: str,
    artifact: Any,
    config: Mapping[str, Any],
    split_names: Sequence[str],
    device: Optional[torch.device] = None,
) -> tuple[Dict[str, EvaluationPredictionBundle], List[BaselinePredictionRecord]]:
    """
    Collect predictions for one baseline over multiple splits.

    Returns:
        bundles_by_split, records
    """
    baseline_key = normalize_baseline_key(baseline_key)
    display_name = getattr(artifact, "model_name", BASELINE_DISPLAY_NAMES[baseline_key])

    bundles: Dict[str, EvaluationPredictionBundle] = {}
    records: List[BaselinePredictionRecord] = []

    for split_name in split_names:
        split_name = str(split_name)
        start = time.perf_counter()

        bundle = collect_baseline_predictions(
            baseline_key=baseline_key,
            artifact=artifact,
            config=config,
            split_name=split_name,
            device=device,
        )

        runtime = float(time.perf_counter() - start)
        valid = bundle.valid_mask > 0.5

        record = BaselinePredictionRecord(
            key=baseline_key,
            display_name=display_name,
            split_name=split_name,
            status="PASSED",
            prediction_count=int(len(bundle.probabilities)),
            valid_count=int(valid.sum()),
            attack_valid_count=int(((bundle.labels == 1) & valid).sum()),
            normal_valid_count=int(((bundle.labels == 0) & valid).sum()),
            runtime_seconds=runtime,
            checkpoint_path=str(bundle.checkpoint_path),
            message="",
        )

        bundles[split_name] = bundle
        records.append(record)

    return bundles, records


def train_or_load_enabled_baselines(
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
    retrain_policy: Optional[str] = None,
) -> tuple[Dict[str, Any], List[BaselineRuntimeRecord]]:
    """Train/load all enabled baselines."""
    baseline_keys = get_enabled_baseline_keys(config)

    artifacts: Dict[str, Any] = {}
    records: List[BaselineRuntimeRecord] = []

    print("=" * 100)
    print("STEP 15 BASELINE FACTORY")
    print("=" * 100)
    print(f"Enabled baselines : {baseline_keys}")
    print(f"Retrain policy    : {retrain_policy or get_step15_retrain_policy(config)}")
    print(f"Feature columns   : {get_baseline_feature_columns(config)}")
    print("All baselines use same reconstructed xi_t features only.")
    print("=" * 100)

    for baseline_key in baseline_keys:
        artifact, record = train_or_load_baseline(
            baseline_key=baseline_key,
            config=config,
            active_seed=active_seed,
            device=device,
            retrain_policy=retrain_policy,
        )

        artifacts[baseline_key] = artifact
        records.append(record)

        print(
            f"{record.display_name:<18} | "
            f"action={record.action:<16} | "
            f"status={record.status:<8} | "
            f"runtime={record.runtime_seconds:.3f}s | "
            f"path={record.model_path}"
        )

    print("=" * 100)

    return artifacts, records


def collect_predictions_for_enabled_baselines(
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
    split_names: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
) -> tuple[
    Dict[str, Dict[str, EvaluationPredictionBundle]],
    List[BaselinePredictionRecord],
]:
    """Collect prediction bundles for all enabled baseline artifacts."""
    if split_names is None:
        split_names = get_step15_evaluation_splits(config)

    all_bundles: Dict[str, Dict[str, EvaluationPredictionBundle]] = {}
    all_records: List[BaselinePredictionRecord] = []

    print("=" * 100)
    print("STEP 15 BASELINE PREDICTION COLLECTION")
    print("=" * 100)
    print(f"Splits: {list(split_names)}")
    print("=" * 100)

    for baseline_key, artifact in artifacts.items():
        bundles, records = collect_baseline_predictions_for_splits(
            baseline_key=baseline_key,
            artifact=artifact,
            config=config,
            split_names=split_names,
            device=device,
        )

        all_bundles[baseline_key] = bundles
        all_records.extend(records)

        for record in records:
            print(
                f"{record.display_name:<18} | "
                f"split={record.split_name:<8} | "
                f"valid={record.valid_count:<6} | "
                f"attack={record.attack_valid_count:<6} | "
                f"normal={record.normal_valid_count:<6} | "
                f"runtime={record.runtime_seconds:.3f}s"
            )

    print("=" * 100)

    return all_bundles, all_records


def baseline_artifact_summary(artifact: Any) -> Dict[str, Any]:
    """Return JSON-safe artifact summary."""
    if hasattr(artifact, "summary"):
        return dict(artifact.summary())

    return {
        "model_name": getattr(artifact, "model_name", "unknown"),
        "model_path": str(getattr(artifact, "model_path", "")),
        "feature_columns": list(getattr(artifact, "feature_columns", [])),
    }


__all__ = [
    "SUPPORTED_BASELINE_KEYS",
    "BASELINE_DISPLAY_NAMES",
    "DEFAULT_BASELINE_SPLITS",
    "BaselineFactorySpec",
    "BaselineRuntimeRecord",
    "BaselinePredictionRecord",
    "normalize_baseline_key",
    "get_baseline_checkpoint_path",
    "make_baseline_factory_specs",
    "get_enabled_baseline_keys",
    "get_step15_retrain_policy",
    "get_step15_evaluation_splits",
    "baseline_checkpoint_exists",
    "train_baseline",
    "load_baseline",
    "train_or_load_baseline",
    "collect_baseline_predictions",
    "collect_baseline_predictions_for_splits",
    "train_or_load_enabled_baselines",
    "collect_predictions_for_enabled_baselines",
    "baseline_artifact_summary",
]