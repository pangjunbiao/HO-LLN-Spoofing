"""
Train-only scaling utilities for causal xi evidence features.

Step 8 purpose:
- fit feature scaling on Dataset-1 TRAIN split only,
- apply the same scaler to train/val/test/external/online xi datasets,
- avoid validation, test, Dataset-2, Dataset-3, and Dataset-1-normal leakage.

Important:
Scaling is useful for neural sequence models, but it must be fair:
proposed model, baselines, and ablations will all receive the same scaled xi
feature representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import json

import numpy as np
import pandas as pd

from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_json


@dataclass
class XiScalerConfig:
    """Configuration for xi feature scaling."""

    enabled: bool
    method: str
    continuous_feature_columns: List[str]
    passthrough_feature_columns: List[str]
    scaled_suffix: str
    clip_scaled_features: bool
    clip_value: float
    epsilon: float
    scaler_json_path: str
    feature_spec_json_path: str


@dataclass
class XiScalerParameters:
    """Fitted scaler parameters."""

    method: str
    continuous_feature_columns: List[str]
    passthrough_feature_columns: List[str]
    scaled_feature_columns: List[str]
    center: Dict[str, float]
    scale: Dict[str, float]
    clip_scaled_features: bool
    clip_value: float
    epsilon: float
    fitted_on: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class XiScalerSummary:
    """Scaler fitting/application summary."""

    enabled: bool
    method: str
    fitted_on_split: str
    train_rows_used: int
    continuous_feature_columns: List[str]
    passthrough_feature_columns: List[str]
    scaled_feature_columns: List[str]
    output_scaler_json: str
    output_feature_spec_json: str
    leakage_rule: Dict[str, Any]
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONTINUOUS_XI_COLUMNS = [
    "xi_eta_east",
    "xi_eta_north",
    "xi_eta_dot_east",
    "xi_eta_dot_north",
    "xi_eta_ddot_east",
    "xi_eta_ddot_north",
    "xi_q",
    "xi_accum_log",
]

DEFAULT_PASSTHROUGH_XI_COLUMNS = [
    "xi_nu",
]


def get_processed_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/processed directory."""
    value = get_by_path(config, "paths.processed_data_dir", "data/processed")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_xi_scaler_config(config: Mapping[str, Any]) -> XiScalerConfig:
    """
    Read xi scaler configuration.
    """
    continuous_cols = get_by_path(
        config,
        "preprocessing.xi_scaler.continuous_feature_columns",
        DEFAULT_CONTINUOUS_XI_COLUMNS,
    )
    passthrough_cols = get_by_path(
        config,
        "preprocessing.xi_scaler.passthrough_feature_columns",
        DEFAULT_PASSTHROUGH_XI_COLUMNS,
    )

    return XiScalerConfig(
        enabled=bool(get_by_path(config, "preprocessing.xi_scaler.enabled", True)),
        method=str(get_by_path(config, "preprocessing.xi_scaler.method", "robust")),
        continuous_feature_columns=list(continuous_cols),
        passthrough_feature_columns=list(passthrough_cols),
        scaled_suffix=str(get_by_path(config, "preprocessing.xi_scaler.scaled_suffix", "_scaled")),
        clip_scaled_features=bool(
            get_by_path(config, "preprocessing.xi_scaler.clip_scaled_features", True)
        ),
        clip_value=float(get_by_path(config, "preprocessing.xi_scaler.clip_value", 20.0)),
        epsilon=float(get_by_path(config, "preprocessing.xi_scaler.epsilon", 1e-8)),
        scaler_json_path=str(
            get_by_path(
                config,
                "paths.step8_xi_scaler_json",
                "results/tables/step8_xi_scaler.json",
            )
        ),
        feature_spec_json_path=str(
            get_by_path(
                config,
                "paths.step8_xi_feature_spec_json",
                "results/tables/step8_xi_feature_spec.json",
            )
        ),
    )


def get_xi_scaler_json_path(config: Mapping[str, Any]) -> Path:
    """Resolve scaler JSON path."""
    value = get_by_path(
        config,
        "paths.step8_xi_scaler_json",
        "results/tables/step8_xi_scaler.json",
    )
    return resolve_project_path(config, value)


def get_xi_feature_spec_json_path(config: Mapping[str, Any]) -> Path:
    """Resolve xi feature spec JSON path."""
    value = get_by_path(
        config,
        "paths.step8_xi_feature_spec_json",
        "results/tables/step8_xi_feature_spec.json",
    )
    return resolve_project_path(config, value)


def _safe_float(value: Any, digits: int = 12) -> float:
    """Convert to JSON-safe float."""
    if value is None:
        return 0.0
    if pd.isna(value):
        return 0.0
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return round(value, digits)


def _validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    context: str,
) -> None:
    """Check that required feature columns exist."""
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing xi feature columns for {context}: {missing}")


def _robust_center_scale(
    train_df: pd.DataFrame,
    columns: Sequence[str],
    epsilon: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit robust scaler using median and IQR.

    This is preferred for xi because residual energy can contain a small number
    of extreme normal outliers.
    """
    center: Dict[str, float] = {}
    scale: Dict[str, float] = {}

    for col in columns:
        x = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = x.dropna()

        if len(finite) == 0:
            center[col] = 0.0
            scale[col] = 1.0
            continue

        q25 = float(finite.quantile(0.25))
        q50 = float(finite.quantile(0.50))
        q75 = float(finite.quantile(0.75))
        iqr = q75 - q25

        if not np.isfinite(iqr) or abs(iqr) < epsilon:
            iqr = float(finite.std(ddof=0))

        if not np.isfinite(iqr) or abs(iqr) < epsilon:
            iqr = 1.0

        center[col] = _safe_float(q50)
        scale[col] = _safe_float(iqr)

    return center, scale


def _standard_center_scale(
    train_df: pd.DataFrame,
    columns: Sequence[str],
    epsilon: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Fit standard scaler using mean and standard deviation."""
    center: Dict[str, float] = {}
    scale: Dict[str, float] = {}

    for col in columns:
        x = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = x.dropna()

        if len(finite) == 0:
            center[col] = 0.0
            scale[col] = 1.0
            continue

        mean = float(finite.mean())
        std = float(finite.std(ddof=0))

        if not np.isfinite(std) or abs(std) < epsilon:
            std = 1.0

        center[col] = _safe_float(mean)
        scale[col] = _safe_float(std)

    return center, scale


def get_scaled_feature_columns(
    continuous_feature_columns: Sequence[str],
    passthrough_feature_columns: Sequence[str],
    scaled_suffix: str = "_scaled",
) -> List[str]:
    """
    Return final scaled model-feature columns.

    Continuous xi columns receive suffix.
    Passthrough columns, especially xi_nu, keep their original name.
    """
    scaled = [f"{col}{scaled_suffix}" for col in continuous_feature_columns]
    scaled.extend(list(passthrough_feature_columns))
    return scaled


def fit_xi_scaler(
    train_df: pd.DataFrame,
    config: Mapping[str, Any],
    fitted_on: Optional[Dict[str, Any]] = None,
) -> XiScalerParameters:
    """
    Fit xi scaler using training split only.
    """
    cfg = get_xi_scaler_config(config)

    if not cfg.enabled:
        scaled_cols = list(cfg.continuous_feature_columns) + list(cfg.passthrough_feature_columns)
        return XiScalerParameters(
            method="identity",
            continuous_feature_columns=cfg.continuous_feature_columns,
            passthrough_feature_columns=cfg.passthrough_feature_columns,
            scaled_feature_columns=scaled_cols,
            center={col: 0.0 for col in cfg.continuous_feature_columns},
            scale={col: 1.0 for col in cfg.continuous_feature_columns},
            clip_scaled_features=False,
            clip_value=cfg.clip_value,
            epsilon=cfg.epsilon,
            fitted_on=fitted_on or {},
        )

    _validate_feature_columns(
        train_df,
        cfg.continuous_feature_columns + cfg.passthrough_feature_columns,
        context="fit_xi_scaler(train)",
    )

    method = cfg.method.lower().strip()

    if method == "robust":
        center, scale = _robust_center_scale(
            train_df=train_df,
            columns=cfg.continuous_feature_columns,
            epsilon=cfg.epsilon,
        )
    elif method == "standard":
        center, scale = _standard_center_scale(
            train_df=train_df,
            columns=cfg.continuous_feature_columns,
            epsilon=cfg.epsilon,
        )
    elif method in {"none", "identity"}:
        method = "identity"
        center = {col: 0.0 for col in cfg.continuous_feature_columns}
        scale = {col: 1.0 for col in cfg.continuous_feature_columns}
    else:
        raise ValueError(
            f"Unsupported xi scaler method: {cfg.method}. "
            "Use one of: robust, standard, identity."
        )

    scaled_cols = get_scaled_feature_columns(
        continuous_feature_columns=cfg.continuous_feature_columns,
        passthrough_feature_columns=cfg.passthrough_feature_columns,
        scaled_suffix=cfg.scaled_suffix,
    )

    return XiScalerParameters(
        method=method,
        continuous_feature_columns=cfg.continuous_feature_columns,
        passthrough_feature_columns=cfg.passthrough_feature_columns,
        scaled_feature_columns=scaled_cols,
        center=center,
        scale=scale,
        clip_scaled_features=cfg.clip_scaled_features,
        clip_value=cfg.clip_value,
        epsilon=cfg.epsilon,
        fitted_on=fitted_on or {},
    )


def transform_xi_dataframe(
    df: pd.DataFrame,
    scaler: XiScalerParameters,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Apply fitted xi scaler to a dataframe.

    Continuous xi columns are transformed into new columns with suffix.
    Passthrough columns remain unchanged.
    """
    out = df.copy() if copy else df

    _validate_feature_columns(
        out,
        scaler.continuous_feature_columns + scaler.passthrough_feature_columns,
        context="transform_xi_dataframe",
    )

    suffix = "_scaled"

    for scaled_col in scaler.scaled_feature_columns:
        if scaled_col.endswith("_scaled"):
            suffix = "_scaled"
            break

    # Recover suffix from first scaled continuous feature when possible.
    if scaler.continuous_feature_columns:
        first = scaler.continuous_feature_columns[0]
        candidates = [
            col for col in scaler.scaled_feature_columns
            if col.startswith(first) and col != first
        ]
        if candidates:
            suffix = candidates[0].replace(first, "", 1)

    for col in scaler.continuous_feature_columns:
        x = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0.0)

        center = float(scaler.center.get(col, 0.0))
        scale = float(scaler.scale.get(col, 1.0))

        if not np.isfinite(scale) or abs(scale) < scaler.epsilon:
            scale = 1.0

        z = (x - center) / scale

        if scaler.clip_scaled_features:
            z = z.clip(lower=-float(scaler.clip_value), upper=float(scaler.clip_value))

        out[f"{col}{suffix}"] = z.astype(float)

    for col in scaler.passthrough_feature_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(float)

    return out


def transform_xi_frames(
    frames: Mapping[str, pd.DataFrame],
    scaler: XiScalerParameters,
) -> Dict[str, pd.DataFrame]:
    """
    Apply xi scaler to multiple named dataframes.
    """
    return {
        name: transform_xi_dataframe(df, scaler=scaler, copy=True)
        for name, df in frames.items()
    }


def save_xi_scaler(
    scaler: XiScalerParameters,
    config: Mapping[str, Any],
) -> Path:
    """
    Save fitted scaler parameters.
    """
    path = get_xi_scaler_json_path(config)
    save_json(scaler.to_dict(), path, indent=2)
    return path


def load_xi_scaler(path: Path) -> XiScalerParameters:
    """
    Load fitted scaler parameters from JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Xi scaler JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return XiScalerParameters(
        method=str(data["method"]),
        continuous_feature_columns=list(data["continuous_feature_columns"]),
        passthrough_feature_columns=list(data["passthrough_feature_columns"]),
        scaled_feature_columns=list(data["scaled_feature_columns"]),
        center=dict(data["center"]),
        scale=dict(data["scale"]),
        clip_scaled_features=bool(data["clip_scaled_features"]),
        clip_value=float(data["clip_value"]),
        epsilon=float(data["epsilon"]),
        fitted_on=dict(data.get("fitted_on", {})),
    )


def save_xi_feature_spec(
    config: Mapping[str, Any],
    raw_feature_columns: Sequence[str],
    scaled_feature_columns: Sequence[str],
    label_column: str,
    group_columns: Sequence[str],
    time_columns: Sequence[str],
) -> Path:
    """
    Save model feature specification for Step 9+.
    """
    path = get_xi_feature_spec_json_path(config)

    spec = {
        "raw_xi_feature_columns": list(raw_feature_columns),
        "scaled_xi_feature_columns": list(scaled_feature_columns),
        "recommended_model_input_columns": list(scaled_feature_columns),
        "label_column": label_column,
        "group_columns": list(group_columns),
        "time_columns": list(time_columns),
        "usage_rule": {
            "proposed_model_uses_same_xi_as_baselines": True,
            "baselines_use_same_scaled_xi_columns": True,
            "raw_shortcut_columns_not_used": True,
            "scaler_fit_on_dataset1_train_only": True,
            "validation_test_external_not_used_for_scaler_fit": True,
        },
    }

    save_json(spec, path, indent=2)
    return path


def summarize_scaler_application(
    scaler: XiScalerParameters,
    train_df: pd.DataFrame,
    config: Mapping[str, Any],
    scaler_path: Path,
    feature_spec_path: Path,
) -> XiScalerSummary:
    """
    Build scaler summary.
    """
    return XiScalerSummary(
        enabled=True,
        method=scaler.method,
        fitted_on_split="dataset1_train",
        train_rows_used=int(len(train_df)),
        continuous_feature_columns=list(scaler.continuous_feature_columns),
        passthrough_feature_columns=list(scaler.passthrough_feature_columns),
        scaled_feature_columns=list(scaler.scaled_feature_columns),
        output_scaler_json=str(scaler_path),
        output_feature_spec_json=str(feature_spec_path),
        leakage_rule={
            "fit_dataset": "dataset1",
            "fit_split": "train",
            "fit_uses_labels": False,
            "validation_used_for_scaler_fit": False,
            "internal_test_used_for_scaler_fit": False,
            "dataset1_normal_used_for_scaler_fit": False,
            "dataset2_used_for_scaler_fit": False,
            "dataset3_used_for_scaler_fit": False,
        },
        final_status="PASSED",
    )