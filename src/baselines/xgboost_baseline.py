"""
XGBoost-xi baseline for causal GNSS spoofing detection.

Step 15 purpose:
- Train a standard time-step baseline on the same reconstructed xi_t features.
- Use Dataset-1 train only for fitting.
- Use Dataset-1 validation only for threshold/persistence selection later.
- Evaluate Dataset-1 test, Dataset-2 external, and Dataset-3 online later.
- Do not use raw shortcut columns.
- Do not use EKF Detector, Clock, Date, Time, GPS raw MGRS columns, or labels as input.

Important:
- This baseline is intentionally fair and standard, not artificially weakened.
- It is a time-step model, so it does not receive hidden recurrent state.
- Temporal event metrics are still computed after prediction using the same alarm rule.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:  # pragma: no cover
    HistGradientBoostingClassifier = None

from src.evaluation.evaluate_dataset1 import EvaluationPredictionBundle
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir


DEFAULT_BASELINE_FEATURE_COLUMNS = [
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


FORBIDDEN_BASELINE_COLUMNS = {
    "Data Type",
    "Spoofing",
    "is_attack",
    "label",
    "Label",
    "Clock",
    "Date",
    "Time",
    "GPS MGRS",
    "GPS Easting",
    "GPS Northing",
    "GPS Latitude",
    "GPS Longitude",
    "EKF Detector",
}


@dataclass
class BaselineSplitData:
    """Flattened time-step data for tabular baselines."""

    split_name: str
    csv_path: Path
    dataframe: pd.DataFrame

    x_all: np.ndarray
    y_all: np.ndarray
    valid_mask_all: np.ndarray

    x_valid: np.ndarray
    y_valid: np.ndarray

    segment_ids: np.ndarray
    row_indices: np.ndarray
    delta_t: np.ndarray

    feature_columns: List[str]
    label_column: str
    validity_column: str
    segment_column: str
    order_column: str
    delta_t_column: str

    def summary(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "csv_path": str(self.csv_path),
            "rows": int(len(self.dataframe)),
            "valid_rows": int(self.valid_mask_all.sum()),
            "invalid_rows": int(len(self.valid_mask_all) - self.valid_mask_all.sum()),
            "attack_valid_rows": int(((self.y_all == 1) & (self.valid_mask_all > 0.5)).sum()),
            "normal_valid_rows": int(((self.y_all == 0) & (self.valid_mask_all > 0.5)).sum()),
            "segments": int(pd.Series(self.segment_ids).nunique()),
            "feature_dim": int(self.x_all.shape[1]) if self.x_all.ndim == 2 else 0,
            "feature_columns": list(self.feature_columns),
        }


@dataclass
class XGBoostBaselineConfig:
    """Configuration for XGBoost-xi baseline."""

    model_name: str = "XGBoost-xi"

    n_estimators: int = 250
    max_depth: int = 3
    learning_rate: float = 0.03
    subsample: float = 0.90
    colsample_bytree: float = 0.90
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0

    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    tree_method: str = "hist"

    n_jobs: int = 4
    random_state: int = 42

    use_train_class_weight: bool = True

    allow_sklearn_fallback: bool = False
    fallback_max_iter: int = 250
    fallback_learning_rate: float = 0.05
    fallback_max_leaf_nodes: int = 31
    fallback_l2_regularization: float = 0.10

    output_model_path: str = "results/models/xgboost_xi.pkl"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class XGBoostBaselineArtifact:
    """Trained XGBoost baseline artifact."""

    model_name: str
    backend: str
    model: Any
    config: XGBoostBaselineConfig
    model_path: Path
    feature_columns: List[str]

    train_summary: Dict[str, Any]
    val_summary: Dict[str, Any]
    fit_runtime_seconds: float
    active_seed: int

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "model_path": str(self.model_path),
            "config": self.config.to_dict(),
            "feature_columns": list(self.feature_columns),
            "train_summary": self.train_summary,
            "val_summary": self.val_summary,
            "fit_runtime_seconds": float(self.fit_runtime_seconds),
            "active_seed": int(self.active_seed),
        }


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to finite float or None."""
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


def _project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve project-relative path."""
    value = str(value)
    return resolve_project_path(config, value)


def save_json_safe(payload: Mapping[str, Any], output_path: Path | str) -> Path:
    """Save JSON with numpy-safe conversion."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    def convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, tuple):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(convert(dict(payload)), file, indent=2)

    return output_path


def get_baseline_feature_columns(config: Mapping[str, Any]) -> List[str]:
    """Return the official shared xi feature columns."""
    columns = list(
        get_by_path(
            config,
            "baselines.feature_columns",
            get_by_path(
                config,
                "training.dataset.feature_columns",
                get_by_path(
                    config,
                    "model.input.recommended_model_input_columns",
                    DEFAULT_BASELINE_FEATURE_COLUMNS,
                ),
            ),
        )
    )

    if columns != DEFAULT_BASELINE_FEATURE_COLUMNS:
        if len(columns) != 9:
            raise ValueError(
                f"Baseline feature contract expects 9 xi columns, got {len(columns)}: {columns}"
            )

    forbidden = [column for column in columns if column in FORBIDDEN_BASELINE_COLUMNS]
    if forbidden:
        raise ValueError(f"Forbidden shortcut columns found in baseline features: {forbidden}")

    return columns


def get_baseline_label_column(config: Mapping[str, Any]) -> str:
    """Return label column name."""
    return str(
        get_by_path(
            config,
            "training.dataset.label_column",
            get_by_path(config, "model.input.label_column", "Data Type"),
        )
    )


def get_baseline_validity_column(config: Mapping[str, Any]) -> str:
    """Return validity column name."""
    return str(
        get_by_path(
            config,
            "training.dataset.validity_column",
            get_by_path(config, "model.input.validity_column", "xi_nu"),
        )
    )


def get_baseline_segment_column(config: Mapping[str, Any]) -> str:
    """Return segment id column."""
    return str(
        get_by_path(
            config,
            "training.dataset.segment_column",
            get_by_path(config, "model.input.segment_column", "segment_id"),
        )
    )


def get_baseline_order_column(config: Mapping[str, Any]) -> str:
    """Return within-segment order column."""
    return str(
        get_by_path(
            config,
            "training.dataset.order_column",
            get_by_path(config, "model.input.order_column", "within_segment_index"),
        )
    )


def get_baseline_delta_t_column(config: Mapping[str, Any]) -> str:
    """Return delta_t column."""
    return str(
        get_by_path(
            config,
            "training.dataset.delta_t_column",
            get_by_path(config, "model.input.delta_t_column", "delta_t_seconds"),
        )
    )


def get_baseline_split_csv_path(config: Mapping[str, Any], split_name: str) -> Path:
    """Resolve split CSV path."""
    split_name = str(split_name)

    default_paths = {
        "train": "data/processed/train_xi.csv",
        "val": "data/processed/val_xi.csv",
        "validation": "data/processed/val_xi.csv",
        "test": "data/processed/test_xi.csv",
        "external": "data/processed/external_xi.csv",
        "online": "data/processed/online_xi.csv",
    }

    key = "val" if split_name == "validation" else split_name

    path_value = get_by_path(
        config,
        f"training.dataset.xi_split_files.{key}",
        get_by_path(config, f"paths.{key}_xi_csv", default_paths.get(key)),
    )

    if path_value is None:
        raise KeyError(f"No CSV path configured for baseline split: {split_name}")

    return _project_path(config, str(path_value))


def labels_to_binary(labels: Sequence[Any]) -> np.ndarray:
    """Convert labels to binary attack labels."""
    series = pd.Series(labels)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        return (values > 0).astype(np.int64).to_numpy()

    normalized = series.astype(str).str.strip().str.lower()

    attack_values = {
        "1",
        "attack",
        "spoof",
        "spoofing",
        "attacked",
        "malicious",
        "true",
        "yes",
    }

    normal_values = {
        "0",
        "normal",
        "benign",
        "clean",
        "false",
        "no",
    }

    output = np.zeros(len(series), dtype=np.int64)

    for i, value in enumerate(normalized):
        if value in attack_values or "attack" in value or "spoof" in value:
            output[i] = 1
        elif value in normal_values:
            output[i] = 0
        else:
            output[i] = 0

    return output


def _sort_dataframe_for_causal_evaluation(
    df: pd.DataFrame,
    segment_column: str,
    order_column: str,
) -> pd.DataFrame:
    """Sort by segment and within-segment time while preserving original row id."""
    df = df.copy()

    if "_baseline_original_row_index" not in df.columns:
        df["_baseline_original_row_index"] = np.arange(len(df), dtype=np.int64)

    if segment_column in df.columns and order_column in df.columns:
        df = df.sort_values([segment_column, order_column], kind="mergesort").reset_index(drop=True)
    elif segment_column in df.columns:
        df = df.sort_values([segment_column, "_baseline_original_row_index"], kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def load_baseline_split_data(
    config: Mapping[str, Any],
    split_name: str,
    feature_columns: Optional[Sequence[str]] = None,
) -> BaselineSplitData:
    """Load one split as flattened time-step arrays."""
    if feature_columns is None:
        feature_columns = get_baseline_feature_columns(config)

    feature_columns = list(feature_columns)

    label_column = get_baseline_label_column(config)
    validity_column = get_baseline_validity_column(config)
    segment_column = get_baseline_segment_column(config)
    order_column = get_baseline_order_column(config)
    delta_t_column = get_baseline_delta_t_column(config)

    csv_path = get_baseline_split_csv_path(config, split_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline split file not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df = _sort_dataframe_for_causal_evaluation(
        df=df,
        segment_column=segment_column,
        order_column=order_column,
    )

    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise KeyError(f"Missing baseline feature columns in {csv_path}: {missing_features}")

    if label_column not in df.columns:
        raise KeyError(f"Missing label column '{label_column}' in {csv_path}")

    x_all = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)

    y_all = labels_to_binary(df[label_column].to_numpy())

    if validity_column in df.columns:
        valid_mask = pd.to_numeric(df[validity_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        valid_mask = (valid_mask > 0.5).astype(np.float32)
    else:
        valid_mask = np.ones(len(df), dtype=np.float32)

    if segment_column in df.columns:
        segment_ids = df[segment_column].astype(str).to_numpy(dtype=object)
    else:
        segment_ids = np.asarray([f"{split_name}_segment_0"] * len(df), dtype=object)

    row_indices = np.arange(len(df), dtype=np.int64)

    if delta_t_column in df.columns:
        delta_t = pd.to_numeric(df[delta_t_column], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    else:
        delta_t = np.ones(len(df), dtype=np.float32)

    valid_bool = valid_mask > 0.5

    return BaselineSplitData(
        split_name=str(split_name),
        csv_path=csv_path,
        dataframe=df,
        x_all=x_all,
        y_all=y_all.astype(np.int64),
        valid_mask_all=valid_mask.astype(np.float32),
        x_valid=x_all[valid_bool],
        y_valid=y_all[valid_bool].astype(np.int64),
        segment_ids=segment_ids,
        row_indices=row_indices,
        delta_t=delta_t,
        feature_columns=feature_columns,
        label_column=label_column,
        validity_column=validity_column,
        segment_column=segment_column,
        order_column=order_column,
        delta_t_column=delta_t_column,
    )


def make_prediction_bundle_from_probabilities(
    split_data: BaselineSplitData,
    probabilities: np.ndarray,
    logits: Optional[np.ndarray],
    checkpoint_path: str,
    model_name: str,
) -> EvaluationPredictionBundle:
    """Build EvaluationPredictionBundle from flat probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)

    if len(probabilities) != len(split_data.y_all):
        raise ValueError(
            f"Prediction length mismatch for {split_data.split_name}: "
            f"{len(probabilities)} probabilities vs {len(split_data.y_all)} labels."
        )

    if logits is None:
        eps = 1.0e-7
        clipped = np.clip(probabilities, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped))

    logits = np.asarray(logits, dtype=np.float32).reshape(-1)

    return EvaluationPredictionBundle(
        split_name=split_data.split_name,
        probabilities=probabilities,
        logits=logits,
        labels=split_data.y_all.astype(np.int64),
        valid_mask=split_data.valid_mask_all.astype(np.float32),
        segment_ids=split_data.segment_ids.astype(object),
        row_indices=split_data.row_indices.astype(np.int64),
        delta_t=split_data.delta_t.astype(np.float32),
        checkpoint_path=str(checkpoint_path),
        model_name=str(model_name),
    )


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute XGBoost scale_pos_weight from train labels."""
    y = np.asarray(y).astype(int)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    if positives <= 0 or negatives <= 0:
        return 1.0

    return float(negatives / positives)


def build_xgboost_baseline_config(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> XGBoostBaselineConfig:
    """Build XGBoost baseline config from project config."""
    base = "baselines.xgboost_xi"

    return XGBoostBaselineConfig(
        model_name=str(get_by_path(config, f"{base}.model_name", "XGBoost-xi")),
        n_estimators=int(get_by_path(config, f"{base}.n_estimators", 250)),
        max_depth=int(get_by_path(config, f"{base}.max_depth", 3)),
        learning_rate=float(get_by_path(config, f"{base}.learning_rate", 0.03)),
        subsample=float(get_by_path(config, f"{base}.subsample", 0.90)),
        colsample_bytree=float(get_by_path(config, f"{base}.colsample_bytree", 0.90)),
        min_child_weight=float(get_by_path(config, f"{base}.min_child_weight", 1.0)),
        reg_lambda=float(get_by_path(config, f"{base}.reg_lambda", 1.0)),
        reg_alpha=float(get_by_path(config, f"{base}.reg_alpha", 0.0)),
        gamma=float(get_by_path(config, f"{base}.gamma", 0.0)),
        objective=str(get_by_path(config, f"{base}.objective", "binary:logistic")),
        eval_metric=str(get_by_path(config, f"{base}.eval_metric", "logloss")),
        tree_method=str(get_by_path(config, f"{base}.tree_method", "hist")),
        n_jobs=int(get_by_path(config, f"{base}.n_jobs", 4)),
        random_state=int(get_by_path(config, f"{base}.random_state", active_seed)),
        use_train_class_weight=bool(get_by_path(config, f"{base}.use_train_class_weight", True)),
        allow_sklearn_fallback=bool(get_by_path(config, f"{base}.allow_sklearn_fallback", False)),
        fallback_max_iter=int(get_by_path(config, f"{base}.fallback_max_iter", 250)),
        fallback_learning_rate=float(get_by_path(config, f"{base}.fallback_learning_rate", 0.05)),
        fallback_max_leaf_nodes=int(get_by_path(config, f"{base}.fallback_max_leaf_nodes", 31)),
        fallback_l2_regularization=float(
            get_by_path(config, f"{base}.fallback_l2_regularization", 0.10)
        ),
        output_model_path=str(
            get_by_path(config, f"{base}.output_model_path", "results/models/xgboost_xi.pkl")
        ),
    )


def _build_xgboost_model(
    cfg: XGBoostBaselineConfig,
    scale_pos_weight: float,
) -> Tuple[Any, str]:
    """Build XGBoost model or optional sklearn fallback."""
    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            gamma=cfg.gamma,
            objective=cfg.objective,
            eval_metric=cfg.eval_metric,
            tree_method=cfg.tree_method,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            scale_pos_weight=scale_pos_weight,
        )
        return model, "xgboost.XGBClassifier"

    if cfg.allow_sklearn_fallback and HistGradientBoostingClassifier is not None:
        model = HistGradientBoostingClassifier(
            max_iter=cfg.fallback_max_iter,
            learning_rate=cfg.fallback_learning_rate,
            max_leaf_nodes=cfg.fallback_max_leaf_nodes,
            l2_regularization=cfg.fallback_l2_regularization,
            random_state=cfg.random_state,
        )
        return model, "sklearn.HistGradientBoostingClassifier_fallback"

    raise ImportError(
        "xgboost is not installed. Install it with: pip install xgboost\n"
        "For a non-paper fallback, set baselines.xgboost_xi.allow_sklearn_fallback=true."
    )


def save_pickle_artifact(obj: Any, path: Path | str) -> Path:
    """Save Python object as pickle."""
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "wb") as file:
        pickle.dump(obj, file)

    return path


def load_pickle_artifact(path: Path | str) -> Any:
    """Load Python object from pickle."""
    with open(path, "rb") as file:
        return pickle.load(file)


def train_xgboost_baseline(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> XGBoostBaselineArtifact:
    """Train XGBoost-xi baseline on Dataset-1 train valid rows."""
    start_time = time.perf_counter()

    cfg = build_xgboost_baseline_config(config=config, active_seed=active_seed)
    feature_columns = get_baseline_feature_columns(config)

    train_data = load_baseline_split_data(
        config=config,
        split_name="train",
        feature_columns=feature_columns,
    )
    val_data = load_baseline_split_data(
        config=config,
        split_name="val",
        feature_columns=feature_columns,
    )

    scale_pos_weight = (
        compute_scale_pos_weight(train_data.y_valid)
        if cfg.use_train_class_weight
        else 1.0
    )

    model, backend = _build_xgboost_model(
        cfg=cfg,
        scale_pos_weight=scale_pos_weight,
    )

    print("=" * 100)
    print("STEP 15 BASELINE TRAINING: XGBoost-xi")
    print("=" * 100)
    print(f"Backend              : {backend}")
    print(f"Train rows/valid     : {len(train_data.y_all)} / {len(train_data.y_valid)}")
    print(f"Val rows/valid       : {len(val_data.y_all)} / {len(val_data.y_valid)}")
    print(f"Feature dim          : {len(feature_columns)}")
    print(f"Scale pos weight     : {scale_pos_weight:.6f}")
    print("Uses same xi_t features only. No raw shortcut columns.")
    print("=" * 100)

    fit_kwargs = {}

    if backend.startswith("xgboost"):
        fit_kwargs["eval_set"] = [(val_data.x_valid, val_data.y_valid)]
        fit_kwargs["verbose"] = False

    try:
        model.fit(train_data.x_valid, train_data.y_valid, **fit_kwargs)
    except TypeError:
        model.fit(train_data.x_valid, train_data.y_valid)

    fit_runtime = float(time.perf_counter() - start_time)

    model_path = _project_path(config, cfg.output_model_path)
    save_pickle_artifact(model, model_path)

    artifact = XGBoostBaselineArtifact(
        model_name=cfg.model_name,
        backend=backend,
        model=model,
        config=cfg,
        model_path=model_path,
        feature_columns=feature_columns,
        train_summary=train_data.summary(),
        val_summary=val_data.summary(),
        fit_runtime_seconds=fit_runtime,
        active_seed=int(active_seed),
    )

    summary_path = model_path.with_suffix(".summary.json")
    save_json_safe(artifact.summary(), summary_path)

    print("XGBoost-xi training completed.")
    print(f"Model path           : {model_path}")
    print(f"Summary path         : {summary_path}")
    print(f"Runtime seconds      : {fit_runtime:.3f}")
    print("=" * 100)

    return artifact


def predict_xgboost_probabilities(
    artifact: XGBoostBaselineArtifact,
    x: np.ndarray,
) -> np.ndarray:
    """Predict positive-class probabilities."""
    model = artifact.model

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[:, 1]
        return np.asarray(probabilities, dtype=np.float32)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        return np.asarray(probabilities, dtype=np.float32)

    raw = model.predict(x)
    return np.asarray(raw, dtype=np.float32).reshape(-1)


def collect_xgboost_predictions(
    config: Mapping[str, Any],
    artifact: XGBoostBaselineArtifact,
    split_name: str,
) -> EvaluationPredictionBundle:
    """Collect predictions for one split."""
    split_data = load_baseline_split_data(
        config=config,
        split_name=split_name,
        feature_columns=artifact.feature_columns,
    )

    probabilities = predict_xgboost_probabilities(
        artifact=artifact,
        x=split_data.x_all,
    )

    return make_prediction_bundle_from_probabilities(
        split_data=split_data,
        probabilities=probabilities,
        logits=None,
        checkpoint_path=str(artifact.model_path),
        model_name=artifact.model_name,
    )


def load_xgboost_baseline(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> XGBoostBaselineArtifact:
    """Load saved XGBoost baseline artifact."""
    cfg = build_xgboost_baseline_config(config=config, active_seed=active_seed)
    feature_columns = get_baseline_feature_columns(config)
    model_path = _project_path(config, cfg.output_model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Saved XGBoost baseline not found: {model_path}")

    model = load_pickle_artifact(model_path)

    return XGBoostBaselineArtifact(
        model_name=cfg.model_name,
        backend="loaded",
        model=model,
        config=cfg,
        model_path=model_path,
        feature_columns=feature_columns,
        train_summary={},
        val_summary={},
        fit_runtime_seconds=0.0,
        active_seed=int(active_seed),
    )


__all__ = [
    "DEFAULT_BASELINE_FEATURE_COLUMNS",
    "FORBIDDEN_BASELINE_COLUMNS",
    "BaselineSplitData",
    "XGBoostBaselineConfig",
    "XGBoostBaselineArtifact",
    "get_baseline_feature_columns",
    "get_baseline_label_column",
    "get_baseline_validity_column",
    "get_baseline_segment_column",
    "get_baseline_order_column",
    "get_baseline_delta_t_column",
    "get_baseline_split_csv_path",
    "labels_to_binary",
    "load_baseline_split_data",
    "make_prediction_bundle_from_probabilities",
    "compute_scale_pos_weight",
    "build_xgboost_baseline_config",
    "save_pickle_artifact",
    "load_pickle_artifact",
    "train_xgboost_baseline",
    "predict_xgboost_probabilities",
    "collect_xgboost_predictions",
    "load_xgboost_baseline",
]