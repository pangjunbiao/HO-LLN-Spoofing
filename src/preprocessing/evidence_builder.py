"""
Build final causal xi evidence vectors for AV-GPS spoofing detection.

Step 8 purpose:
- load Step-7 residual files,
- load training-only normal residual statistics,
- compute normalized residual evidence eta_t,
- compute residual evolution dot_eta_t and ddot_eta_t,
- compute baseline-compensated evidence q_t,
- compute weak accumulation log(1 + a_t),
- create final xi vector:
    xi_t = [eta_t, dot_eta_t, ddot_eta_t, q_t, accum_log_t, nu_t]
- split Dataset-1 into train/val/internal-test by Step-4 segment split,
- map Dataset-2 to external_xi and Dataset-3 to online_xi,
- fit scaler on Dataset-1 train only and apply to all xi files.

This is the representation on which proposed model, baselines, and ablations
must compete.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import json

import numpy as np
import pandas as pd

from src.preprocessing.scaler import (
    XiScalerSummary,
    fit_xi_scaler,
    save_xi_feature_spec,
    save_xi_scaler,
    summarize_scaler_application,
    transform_xi_frames,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_csv, save_json
from src.preprocessing.normal_statistics import (
    load_training_normal_statistics_for_xi,
    print_training_normal_statistics_for_xi_summary,
)


DEFAULT_DATASET_KEYS = [
    "dataset1",
    "dataset1_normal",
    "dataset2",
    "dataset3",
]

DEFAULT_RESIDUAL_FILES = {
    "dataset1": "dataset1_residual.csv",
    "dataset1_normal": "dataset1_normal_residual.csv",
    "dataset2": "dataset2_residual.csv",
    "dataset3": "dataset3_residual.csv",
}

DEFAULT_XI_FILES = {
    "dataset1": "dataset1_xi.csv",
    "dataset1_normal": "dataset1_normal_xi.csv",
    "dataset2": "dataset2_xi.csv",
    "dataset3": "dataset3_xi.csv",
}

RAW_XI_FEATURE_COLUMNS = [
    "xi_eta_east",
    "xi_eta_north",
    "xi_eta_dot_east",
    "xi_eta_dot_north",
    "xi_eta_ddot_east",
    "xi_eta_ddot_north",
    "xi_q",
    "xi_accum_log",
    "xi_nu",
]


@dataclass
class EvidenceBuilderConfig:
    """Configuration for xi evidence construction."""

    segment_column: str
    order_column: str
    label_column: str
    delta_t_column: str

    residual_east_column: str
    residual_north_column: str
    residual_valid_column: str

    rho: float
    kappa: float
    max_delta_seconds: float

    eta_east_column: str
    eta_north_column: str
    eta_dot_east_column: str
    eta_dot_north_column: str
    eta_ddot_east_column: str
    eta_ddot_north_column: str
    q_column: str
    accum_raw_column: str
    accum_log_column: str
    nu_column: str

    residual_energy_column: str
    q_raw_column: str
    dot_valid_column: str
    ddot_valid_column: str
    xi_valid_column: str


@dataclass
class XiDatasetSummary:
    """Summary for one xi dataset."""

    dataset_key: str
    input_path: str
    output_path: str

    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int

    segments: int
    valid_nu_rows: int
    invalid_nu_rows: int
    dot_valid_rows: int
    ddot_valid_rows: int

    normal_rows: int
    attack_rows: int
    valid_normal_rows: int
    valid_attack_rows: int

    residual_energy_quantiles_valid: Dict[str, float]
    q_quantiles_valid: Dict[str, float]
    accum_log_quantiles: Dict[str, float]

    required_columns_missing: List[str]
    output_columns_added: List[str]
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class XiSplitSummary:
    """Summary for train/val/test/external/online xi split files."""

    split_name: str
    source_dataset: str
    output_path: str
    rows: int
    segments: int
    normal_rows: int
    attack_rows: int
    valid_nu_rows: int
    invalid_nu_rows: int
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FullEvidenceBuilderReport:
    """Full Step-8 evidence-builder report."""

    dataset_summaries: Dict[str, XiDatasetSummary]
    split_summaries: Dict[str, XiSplitSummary]
    scaler_summary: Dict[str, Any]
    normal_statistics_source: Dict[str, Any]
    evidence_builder_config: Dict[str, Any]
    raw_xi_feature_columns: List[str]
    scaled_xi_feature_columns: List[str]
    final_step8_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_summaries": {
                key: value.to_dict()
                for key, value in self.dataset_summaries.items()
            },
            "split_summaries": {
                key: value.to_dict()
                for key, value in self.split_summaries.items()
            },
            "scaler_summary": self.scaler_summary,
            "normal_statistics_source": self.normal_statistics_source,
            "evidence_builder_config": self.evidence_builder_config,
            "raw_xi_feature_columns": self.raw_xi_feature_columns,
            "scaled_xi_feature_columns": self.scaled_xi_feature_columns,
            "final_step8_status": self.final_step8_status,
        }


def get_interim_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/interim directory."""
    value = get_by_path(config, "paths.interim_data_dir", "data/interim")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_processed_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/processed directory."""
    value = get_by_path(config, "paths.processed_data_dir", "data/processed")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_splits_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/splits directory."""
    value = get_by_path(config, "paths.splits_dir", "data/splits")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_step8_summary_path(config: Mapping[str, Any]) -> Path:
    """Resolve Step-8 summary JSON path."""
    value = get_by_path(
        config,
        "paths.step8_evidence_summary_json",
        "results/tables/step8_evidence_summary.json",
    )
    return resolve_project_path(config, value)


def get_step8_energy_diagnostics_path(config: Mapping[str, Any]) -> Path:
    """Resolve Step-8 energy diagnostics JSON path."""
    value = get_by_path(
        config,
        "paths.step8_energy_diagnostics_json",
        "results/tables/step8_energy_diagnostics.json",
    )
    return resolve_project_path(config, value)


def get_residual_file_path(config: Mapping[str, Any], dataset_key: str) -> Path:
    """Resolve Step-7 residual input path."""
    file_name = get_by_path(
        config,
        f"dataset.residual_files.{dataset_key}",
        DEFAULT_RESIDUAL_FILES.get(dataset_key, f"{dataset_key}_residual.csv"),
    )
    return (get_interim_dir(config) / str(file_name)).resolve()


def get_xi_file_path(config: Mapping[str, Any], dataset_key: str) -> Path:
    """Resolve dataset-level xi output path."""
    file_name = get_by_path(
        config,
        f"dataset.xi_files.{dataset_key}",
        DEFAULT_XI_FILES.get(dataset_key, f"{dataset_key}_xi.csv"),
    )
    return (get_processed_dir(config) / str(file_name)).resolve()


def get_split_xi_path(config: Mapping[str, Any], split_name: str) -> Path:
    """Resolve split-level xi output path."""
    defaults = {
        "train": "train_xi.csv",
        "val": "val_xi.csv",
        "test": "test_xi.csv",
        "external": "external_xi.csv",
        "online": "online_xi.csv",
        "normal_reference": "normal_reference_xi.csv",
    }

    file_name = get_by_path(
        config,
        f"dataset.xi_split_files.{split_name}",
        defaults.get(split_name, f"{split_name}_xi.csv"),
    )
    return (get_processed_dir(config) / str(file_name)).resolve()


def get_evidence_builder_config(config: Mapping[str, Any]) -> EvidenceBuilderConfig:
    """Read evidence-builder configuration."""
    return EvidenceBuilderConfig(
        segment_column=str(
            get_by_path(config, "preprocessing.evidence.segment_column", "segment_id")
        ),
        order_column=str(
            get_by_path(config, "preprocessing.evidence.order_column", "within_segment_index")
        ),
        label_column=str(
            get_by_path(config, "dataset.label_column", "Data Type")
        ),
        delta_t_column=str(
            get_by_path(config, "preprocessing.evidence.delta_t_column", "delta_t_seconds")
        ),

        residual_east_column=str(
            get_by_path(config, "preprocessing.residual.residual_east_column", "residual_east_m")
        ),
        residual_north_column=str(
            get_by_path(config, "preprocessing.residual.residual_north_column", "residual_north_m")
        ),
        residual_valid_column=str(
            get_by_path(config, "preprocessing.residual.residual_valid_column", "nu")
        ),

        rho=float(get_by_path(config, "preprocessing.evidence.cusum_rho", 0.98)),
        kappa=float(get_by_path(config, "preprocessing.evidence.kappa", 0.0)),
        max_delta_seconds=float(
            get_by_path(config, "preprocessing.evidence.max_delta_seconds", 5.0)
        ),

        eta_east_column=str(
            get_by_path(config, "preprocessing.evidence.eta_east_column", "xi_eta_east")
        ),
        eta_north_column=str(
            get_by_path(config, "preprocessing.evidence.eta_north_column", "xi_eta_north")
        ),
        eta_dot_east_column=str(
            get_by_path(config, "preprocessing.evidence.eta_dot_east_column", "xi_eta_dot_east")
        ),
        eta_dot_north_column=str(
            get_by_path(config, "preprocessing.evidence.eta_dot_north_column", "xi_eta_dot_north")
        ),
        eta_ddot_east_column=str(
            get_by_path(config, "preprocessing.evidence.eta_ddot_east_column", "xi_eta_ddot_east")
        ),
        eta_ddot_north_column=str(
            get_by_path(config, "preprocessing.evidence.eta_ddot_north_column", "xi_eta_ddot_north")
        ),
        q_column=str(
            get_by_path(config, "preprocessing.evidence.q_column", "xi_q")
        ),
        accum_raw_column=str(
            get_by_path(config, "preprocessing.evidence.accum_raw_column", "xi_accum_raw")
        ),
        accum_log_column=str(
            get_by_path(config, "preprocessing.evidence.accum_log_column", "xi_accum_log")
        ),
        nu_column=str(
            get_by_path(config, "preprocessing.evidence.nu_column", "xi_nu")
        ),

        residual_energy_column=str(
            get_by_path(config, "preprocessing.evidence.residual_energy_column", "xi_residual_energy")
        ),
        q_raw_column=str(
            get_by_path(config, "preprocessing.evidence.q_raw_column", "xi_q_raw")
        ),
        dot_valid_column=str(
            get_by_path(config, "preprocessing.evidence.dot_valid_column", "xi_dot_valid")
        ),
        ddot_valid_column=str(
            get_by_path(config, "preprocessing.evidence.ddot_valid_column", "xi_ddot_valid")
        ),
        xi_valid_column=str(
            get_by_path(config, "preprocessing.evidence.xi_valid_column", "xi_valid")
        ),
    )


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_normal_statistics_path(config: Mapping[str, Any]) -> Path:
    """Resolve Step-7 normal statistics JSON path."""
    value = get_by_path(
        config,
        "paths.step7_normal_statistics_json",
        "results/tables/step7_normal_statistics.json",
    )
    return resolve_project_path(config, value)


def load_training_normal_statistics(config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Load training-only normal statistics from Step 7.

    Required keys:
    - residual_inverse_sqrt_covariance
    - residual_inverse_covariance
    - train_normal_energy_median_mu_e
    """
    path = get_normal_statistics_path(config)
    data = load_json_file(path)

    required = [
        "residual_inverse_sqrt_covariance",
        "residual_inverse_covariance",
        "train_normal_energy_median_mu_e",
    ]

    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            f"Step-7 normal statistics JSON missing keys: {missing}. "
            f"Path: {path}"
        )

    data["_normal_statistics_path"] = str(path)
    return data


def load_residual_dataset(
    config: Mapping[str, Any],
    dataset_key: str,
) -> pd.DataFrame:
    """Load one Step-7 residual dataset."""
    path = get_residual_file_path(config, dataset_key)

    if not path.exists():
        raise FileNotFoundError(
            f"Residual file not found for {dataset_key}: {path}\n"
            "Run Step 7 first."
        )

    return pd.read_csv(path, low_memory=False)


def _safe_float(value: Any, digits: int = 10) -> float:
    """Convert numeric value to safe rounded JSON float."""
    if value is None:
        return 0.0
    if pd.isna(value):
        return 0.0

    value = float(value)
    if not np.isfinite(value):
        return 0.0

    return round(value, digits)


def _quantiles(values: Any) -> Dict[str, float]:
    """Return robust diagnostics quantiles."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.dropna()

    if len(x) == 0:
        return {
            "count": 0,
            "q00": 0.0,
            "q01": 0.0,
            "q05": 0.0,
            "q10": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "q100": 0.0,
        }

    quantile_map = {
        "q00": 0.00,
        "q01": 0.01,
        "q05": 0.05,
        "q10": 0.10,
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q90": 0.90,
        "q95": 0.95,
        "q99": 0.99,
        "q100": 1.00,
    }

    out = {"count": int(len(x))}
    for name, q in quantile_map.items():
        out[name] = _safe_float(x.quantile(q))

    return out


def _required_columns(cfg: EvidenceBuilderConfig) -> List[str]:
    """Columns required to build xi."""
    return [
        cfg.segment_column,
        cfg.order_column,
        cfg.label_column,
        cfg.delta_t_column,
        cfg.residual_east_column,
        cfg.residual_north_column,
        cfg.residual_valid_column,
    ]


def _sort_for_causal_processing(
    df: pd.DataFrame,
    cfg: EvidenceBuilderConfig,
) -> pd.DataFrame:
    """
    Sort by segment and within-segment order.

    Causal derivatives and CUSUM require correct temporal order.
    """
    return df.sort_values(
        by=[cfg.segment_column, cfg.order_column],
        kind="mergesort",
    ).reset_index(drop=True)


def _compute_eta(
    residual_matrix: np.ndarray,
    valid_mask: np.ndarray,
    inverse_sqrt_covariance: np.ndarray,
) -> np.ndarray:
    """
    Compute eta_t = Sigma^{-1/2} r_t.

    Invalid rows are set to zero.
    """
    eta = residual_matrix @ inverse_sqrt_covariance.T
    eta[~valid_mask, :] = 0.0
    eta[~np.isfinite(eta)] = 0.0
    return eta


def _valid_delta(delta_t: Any, max_delta_seconds: float) -> bool:
    """Check positive finite delta-t."""
    try:
        value = float(delta_t)
    except Exception:
        return False

    return bool(np.isfinite(value) and value > 0.0 and value <= float(max_delta_seconds))


def _compute_evolution_and_accumulation(
    out: pd.DataFrame,
    cfg: EvidenceBuilderConfig,
) -> pd.DataFrame:
    """
    Compute dot_eta, ddot_eta, one-sided leaky CUSUM, and log accumulation.

    All sequential states reset at segment start.
    """
    eta_cols = [cfg.eta_east_column, cfg.eta_north_column]

    out[cfg.eta_dot_east_column] = 0.0
    out[cfg.eta_dot_north_column] = 0.0
    out[cfg.eta_ddot_east_column] = 0.0
    out[cfg.eta_ddot_north_column] = 0.0
    out[cfg.dot_valid_column] = 0
    out[cfg.ddot_valid_column] = 0
    out[cfg.accum_raw_column] = 0.0
    out[cfg.accum_log_column] = 0.0

    for _, group_index in out.groupby(cfg.segment_column, sort=False).groups.items():
        idx = list(group_index)

        previous_accum = 0.0

        previous_dot = np.array([0.0, 0.0], dtype=float)
        previous_dot_valid = False
        previous_delta_t = np.nan

        for position, row_idx in enumerate(idx):
            nu = int(out.at[row_idx, cfg.nu_column]) == 1
            delta_t = out.at[row_idx, cfg.delta_t_column]
            delta_valid = _valid_delta(delta_t, cfg.max_delta_seconds)

            eta_current = out.loc[row_idx, eta_cols].to_numpy(dtype=float)

            # Residual evolution: dot_eta_t.
            dot_current = np.array([0.0, 0.0], dtype=float)
            dot_valid = False

            if position > 0 and nu and delta_valid:
                prev_idx = idx[position - 1]
                prev_nu = int(out.at[prev_idx, cfg.nu_column]) == 1

                if prev_nu:
                    eta_prev = out.loc[prev_idx, eta_cols].to_numpy(dtype=float)
                    dot_current = (eta_current - eta_prev) / float(delta_t)
                    dot_valid = bool(np.all(np.isfinite(dot_current)))

            if dot_valid:
                out.at[row_idx, cfg.eta_dot_east_column] = float(dot_current[0])
                out.at[row_idx, cfg.eta_dot_north_column] = float(dot_current[1])
                out.at[row_idx, cfg.dot_valid_column] = 1

            # Residual acceleration: ddot_eta_t.
            ddot_current = np.array([0.0, 0.0], dtype=float)
            ddot_valid = False

            if position > 1 and dot_valid and previous_dot_valid:
                if _valid_delta(previous_delta_t, cfg.max_delta_seconds):
                    avg_delta = 0.5 * (float(delta_t) + float(previous_delta_t))

                    if avg_delta > 0.0 and np.isfinite(avg_delta):
                        ddot_current = (dot_current - previous_dot) / avg_delta
                        ddot_valid = bool(np.all(np.isfinite(ddot_current)))

            if ddot_valid:
                out.at[row_idx, cfg.eta_ddot_east_column] = float(ddot_current[0])
                out.at[row_idx, cfg.eta_ddot_north_column] = float(ddot_current[1])
                out.at[row_idx, cfg.ddot_valid_column] = 1

            # Weak accumulation.
            q_value = float(out.at[row_idx, cfg.q_column])

            if nu:
                previous_accum = max(0.0, float(cfg.rho) * previous_accum + q_value)
            else:
                previous_accum = float(cfg.rho) * previous_accum

            if not np.isfinite(previous_accum):
                previous_accum = 0.0

            out.at[row_idx, cfg.accum_raw_column] = previous_accum
            out.at[row_idx, cfg.accum_log_column] = float(np.log1p(max(previous_accum, 0.0)))

            previous_dot = dot_current.copy()
            previous_dot_valid = dot_valid
            previous_delta_t = delta_t

    return out


def build_xi_for_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    normal_stats: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, XiDatasetSummary]:
    """
    Build xi evidence dataframe for one dataset.
    """
    cfg = get_evidence_builder_config(config)

    required = _required_columns(cfg)
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"{dataset_key} missing required evidence columns: {missing}")

    input_rows = int(len(df))
    input_columns = int(df.shape[1])

    out = _sort_for_causal_processing(df=df, cfg=cfg)

    inv_sqrt = np.asarray(
        normal_stats["residual_inverse_sqrt_covariance"],
        dtype=float,
    )

    if inv_sqrt.shape != (2, 2):
        raise ValueError(f"Expected inverse sqrt covariance shape (2, 2), got {inv_sqrt.shape}")

    mu_e = float(normal_stats["train_normal_energy_median_mu_e"])

    residual_east = pd.to_numeric(out[cfg.residual_east_column], errors="coerce").fillna(0.0)
    residual_north = pd.to_numeric(out[cfg.residual_north_column], errors="coerce").fillna(0.0)
    residual_matrix = np.stack([residual_east.to_numpy(), residual_north.to_numpy()], axis=1)

    valid_mask = out[cfg.residual_valid_column].fillna(0).astype(int).to_numpy() == 1

    eta = _compute_eta(
        residual_matrix=residual_matrix,
        valid_mask=valid_mask,
        inverse_sqrt_covariance=inv_sqrt,
    )

    energy = np.sum(eta * eta, axis=1)
    energy[~valid_mask] = 0.0
    energy[~np.isfinite(energy)] = 0.0

    q_raw = energy - mu_e - float(cfg.kappa)
    q = q_raw.copy()

    # Invalid rows must not inject artificial negative evidence.
    q[~valid_mask] = 0.0
    q[~np.isfinite(q)] = 0.0

    out[cfg.eta_east_column] = eta[:, 0]
    out[cfg.eta_north_column] = eta[:, 1]
    out[cfg.residual_energy_column] = energy
    out[cfg.q_raw_column] = q_raw
    out[cfg.q_column] = q
    out[cfg.nu_column] = valid_mask.astype(int)
    out[cfg.xi_valid_column] = 1

    out = _compute_evolution_and_accumulation(out=out, cfg=cfg)

    label_col = cfg.label_column
    normal_label = int(get_by_path(config, "dataset.normal_label", 0))
    attack_label = int(get_by_path(config, "dataset.attack_label", 1))

    labels = out[label_col] if label_col in out.columns else pd.Series([], dtype=int)

    normal_rows = int((labels == normal_label).sum()) if label_col in out.columns else 0
    attack_rows = int((labels == attack_label).sum()) if label_col in out.columns else 0
    valid_normal_rows = int(((labels == normal_label) & valid_mask).sum()) if label_col in out.columns else 0
    valid_attack_rows = int(((labels == attack_label) & valid_mask).sum()) if label_col in out.columns else 0

    valid_energy = out.loc[out[cfg.nu_column].astype(int) == 1, cfg.residual_energy_column]
    valid_q = out.loc[out[cfg.nu_column].astype(int) == 1, cfg.q_column]
    accum_log = out[cfg.accum_log_column]

    output_columns_added = [
        cfg.eta_east_column,
        cfg.eta_north_column,
        cfg.residual_energy_column,
        cfg.q_raw_column,
        cfg.q_column,
        cfg.nu_column,
        cfg.xi_valid_column,
        cfg.eta_dot_east_column,
        cfg.eta_dot_north_column,
        cfg.eta_ddot_east_column,
        cfg.eta_ddot_north_column,
        cfg.dot_valid_column,
        cfg.ddot_valid_column,
        cfg.accum_raw_column,
        cfg.accum_log_column,
    ]

    summary = XiDatasetSummary(
        dataset_key=dataset_key,
        input_path=str(get_residual_file_path(config, dataset_key)),
        output_path=str(get_xi_file_path(config, dataset_key)),
        input_rows=input_rows,
        output_rows=int(len(out)),
        input_columns=input_columns,
        output_columns=int(out.shape[1]),
        segments=int(out[cfg.segment_column].astype(str).nunique()),
        valid_nu_rows=int(valid_mask.sum()),
        invalid_nu_rows=int((~valid_mask).sum()),
        dot_valid_rows=int((out[cfg.dot_valid_column].astype(int) == 1).sum()),
        ddot_valid_rows=int((out[cfg.ddot_valid_column].astype(int) == 1).sum()),
        normal_rows=normal_rows,
        attack_rows=attack_rows,
        valid_normal_rows=valid_normal_rows,
        valid_attack_rows=valid_attack_rows,
        residual_energy_quantiles_valid=_quantiles(valid_energy),
        q_quantiles_valid=_quantiles(valid_q),
        accum_log_quantiles=_quantiles(accum_log),
        required_columns_missing=missing,
        output_columns_added=output_columns_added,
        final_status="PASSED" if int(valid_mask.sum()) > 0 else "FAILED_NO_VALID_XI_ROWS",
    )

    return out, summary


def get_split_segments_path(config: Mapping[str, Any], split_name: str) -> Path:
    """Resolve train/val/test segment split JSON path."""
    file_key = {
        "train": "train_segments",
        "val": "val_segments",
        "test": "test_segments",
    }[split_name]

    file_name = get_by_path(
        config,
        f"dataset.split_files.{file_key}",
        f"dataset1_{split_name}_segments.json",
    )

    return (get_splits_dir(config) / str(file_name)).resolve()


def load_split_segment_ids(
    config: Mapping[str, Any],
    split_name: str,
) -> List[str]:
    """Load Step-4 segment IDs for train/val/test."""
    path = get_split_segments_path(config, split_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Split segment file not found: {path}\n"
            "Run Step 4 first."
        )

    data = load_json_file(path)
    segments = data.get("segments", [])

    if not segments:
        raise ValueError(f"No segments found in split file: {path}")

    return [str(seg) for seg in segments]


def make_dataset1_split_frame(
    dataset1_xi: pd.DataFrame,
    config: Mapping[str, Any],
    split_name: str,
) -> pd.DataFrame:
    """Create train/val/test xi dataframe from Dataset-1 segment IDs."""
    cfg = get_evidence_builder_config(config)
    segment_ids = set(load_split_segment_ids(config, split_name))

    out = dataset1_xi.loc[
        dataset1_xi[cfg.segment_column].astype(str).isin(segment_ids)
    ].copy()

    out["xi_split"] = split_name
    out["xi_source_dataset"] = "dataset1"

    return out


def summarize_split_frame(
    df: pd.DataFrame,
    split_name: str,
    source_dataset: str,
    output_path: Path,
    config: Mapping[str, Any],
) -> XiSplitSummary:
    """Summarize one xi split dataframe."""
    cfg = get_evidence_builder_config(config)

    normal_label = int(get_by_path(config, "dataset.normal_label", 0))
    attack_label = int(get_by_path(config, "dataset.attack_label", 1))

    label_col = cfg.label_column

    normal_rows = int((df[label_col] == normal_label).sum()) if label_col in df.columns else 0
    attack_rows = int((df[label_col] == attack_label).sum()) if label_col in df.columns else 0

    valid = df[cfg.nu_column].fillna(0).astype(int) == 1

    return XiSplitSummary(
        split_name=split_name,
        source_dataset=source_dataset,
        output_path=str(output_path),
        rows=int(len(df)),
        segments=int(df[cfg.segment_column].astype(str).nunique()) if cfg.segment_column in df.columns else 0,
        normal_rows=normal_rows,
        attack_rows=attack_rows,
        valid_nu_rows=int(valid.sum()),
        invalid_nu_rows=int((~valid).sum()),
        final_status="PASSED" if len(df) > 0 else "FAILED_EMPTY_SPLIT",
    )


def _print_dataset_summary(summary: XiDatasetSummary) -> None:
    """Print one dataset-level xi summary."""
    print("=" * 100)
    print(f"STEP 8 XI DATASET SUMMARY | {summary.dataset_key}")
    print("=" * 100)
    print(f"Input path                         : {summary.input_path}")
    print(f"Output path                        : {summary.output_path}")
    print(f"Rows                               : {summary.input_rows} -> {summary.output_rows}")
    print(f"Columns                            : {summary.input_columns} -> {summary.output_columns}")
    print(f"Segments                            : {summary.segments}")
    print(f"Required columns missing            : {summary.required_columns_missing}")
    print(f"Final nu=1 rows                     : {summary.valid_nu_rows}")
    print(f"Final nu=0 rows                     : {summary.invalid_nu_rows}")
    print(f"dot_eta valid rows                  : {summary.dot_valid_rows}")
    print(f"ddot_eta valid rows                 : {summary.ddot_valid_rows}")
    print(f"Normal rows                         : {summary.normal_rows}")
    print(f"Attack rows                         : {summary.attack_rows}")
    print(f"Valid normal rows                   : {summary.valid_normal_rows}")
    print(f"Valid attack rows                   : {summary.valid_attack_rows}")
    print(f"Energy quantiles valid              : {summary.residual_energy_quantiles_valid}")
    print(f"q quantiles valid                   : {summary.q_quantiles_valid}")
    print(f"accum_log quantiles                 : {summary.accum_log_quantiles}")
    print(f"Final status                        : {summary.final_status}")
    print("=" * 100)


def _print_split_summary(summary: XiSplitSummary) -> None:
    """Print one split-level xi summary."""
    print("-" * 100)
    print(f"STEP 8 XI SPLIT SUMMARY | {summary.split_name}")
    print("-" * 100)
    print(f"Source dataset                      : {summary.source_dataset}")
    print(f"Output path                         : {summary.output_path}")
    print(f"Rows                                : {summary.rows}")
    print(f"Segments                            : {summary.segments}")
    print(f"Normal rows                         : {summary.normal_rows}")
    print(f"Attack rows                         : {summary.attack_rows}")
    print(f"Final nu=1 rows                     : {summary.valid_nu_rows}")
    print(f"Final nu=0 rows                     : {summary.invalid_nu_rows}")
    print(f"Final status                        : {summary.final_status}")


def _save_frames(
    frames: Mapping[str, pd.DataFrame],
    paths: Mapping[str, Path],
) -> None:
    """Save multiple CSV frames."""
    for name, df in frames.items():
        save_csv(df, paths[name], index=False)


def run_evidence_builder_step(
    config: Mapping[str, Any],
    dataset_keys: Optional[Sequence[str]] = None,
    save_outputs: bool = True,
) -> FullEvidenceBuilderReport:
    """
    Main Step-8 entry point.

    Saves:
    - data/processed/dataset1_xi.csv
    - data/processed/dataset1_normal_xi.csv
    - data/processed/dataset2_xi.csv
    - data/processed/dataset3_xi.csv
    - data/processed/train_xi.csv
    - data/processed/val_xi.csv
    - data/processed/test_xi.csv
    - data/processed/external_xi.csv
    - data/processed/online_xi.csv
    - results/tables/step8_evidence_summary.json
    - results/tables/step8_energy_diagnostics.json
    - results/tables/step8_xi_scaler.json
    - results/tables/step8_xi_feature_spec.json
    """
    keys = list(dataset_keys or DEFAULT_DATASET_KEYS)

    normal_stats = load_training_normal_statistics_for_xi(
        config=config,
        save_validation_report=True,
    )
    print_training_normal_statistics_for_xi_summary(normal_stats)
    evidence_cfg = get_evidence_builder_config(config)

    dataset_frames: Dict[str, pd.DataFrame] = {}
    dataset_summaries: Dict[str, XiDatasetSummary] = {}

    for dataset_key in keys:
        residual_df = load_residual_dataset(config, dataset_key)

        xi_df, summary = build_xi_for_dataset(
            df=residual_df,
            dataset_key=dataset_key,
            normal_stats=normal_stats,
            config=config,
        )

        dataset_frames[dataset_key] = xi_df
        dataset_summaries[dataset_key] = summary

    dataset1_xi = dataset_frames["dataset1"]

    split_frames: Dict[str, pd.DataFrame] = {
        "train": make_dataset1_split_frame(dataset1_xi, config, "train"),
        "val": make_dataset1_split_frame(dataset1_xi, config, "val"),
        "test": make_dataset1_split_frame(dataset1_xi, config, "test"),
        "external": dataset_frames["dataset2"].copy(),
        "online": dataset_frames["dataset3"].copy(),
        "normal_reference": dataset_frames["dataset1_normal"].copy(),
    }

    split_frames["external"]["xi_split"] = "external"
    split_frames["external"]["xi_source_dataset"] = "dataset2"

    split_frames["online"]["xi_split"] = "online"
    split_frames["online"]["xi_source_dataset"] = "dataset3"

    split_frames["normal_reference"]["xi_split"] = "normal_reference"
    split_frames["normal_reference"]["xi_source_dataset"] = "dataset1_normal"

    fitted_on = {
        "dataset": "dataset1",
        "split": "train",
        "rows": int(len(split_frames["train"])),
        "uses_labels": False,
        "leakage_safe": True,
    }

    scaler = fit_xi_scaler(
        train_df=split_frames["train"],
        config=config,
        fitted_on=fitted_on,
    )

    all_frames_for_scaling: Dict[str, pd.DataFrame] = {}
    all_frames_for_scaling.update({f"dataset::{k}": v for k, v in dataset_frames.items()})
    all_frames_for_scaling.update({f"split::{k}": v for k, v in split_frames.items()})

    scaled_all_frames = transform_xi_frames(
        frames=all_frames_for_scaling,
        scaler=scaler,
    )

    dataset_frames = {
        key.replace("dataset::", ""): value
        for key, value in scaled_all_frames.items()
        if key.startswith("dataset::")
    }
    split_frames = {
        key.replace("split::", ""): value
        for key, value in scaled_all_frames.items()
        if key.startswith("split::")
    }

    dataset_paths = {
        key: get_xi_file_path(config, key)
        for key in dataset_frames.keys()
    }

    split_paths = {
        key: get_split_xi_path(config, key)
        for key in split_frames.keys()
    }

    if save_outputs:
        _save_frames(dataset_frames, dataset_paths)
        _save_frames(split_frames, split_paths)

    scaler_path = save_xi_scaler(scaler, config) if save_outputs else Path("")
    feature_spec_path = (
        save_xi_feature_spec(
            config=config,
            raw_feature_columns=RAW_XI_FEATURE_COLUMNS,
            scaled_feature_columns=scaler.scaled_feature_columns,
            label_column=evidence_cfg.label_column,
            group_columns=[
                evidence_cfg.segment_column,
                "xi_split",
                "xi_source_dataset",
            ],
            time_columns=[
                evidence_cfg.order_column,
                evidence_cfg.delta_t_column,
            ],
        )
        if save_outputs
        else Path("")
    )

    scaler_summary: XiScalerSummary = summarize_scaler_application(
        scaler=scaler,
        train_df=split_frames["train"],
        config=config,
        scaler_path=scaler_path,
        feature_spec_path=feature_spec_path,
    )

    split_summaries: Dict[str, XiSplitSummary] = {
        key: summarize_split_frame(
            df=frame,
            split_name=key,
            source_dataset=str(frame["xi_source_dataset"].iloc[0]) if len(frame) else "unknown",
            output_path=split_paths[key],
            config=config,
        )
        for key, frame in split_frames.items()
    }

    # Refresh output paths after final saving.
    for key, summary in dataset_summaries.items():
        summary.output_path = str(dataset_paths[key])
        summary.output_columns = int(dataset_frames[key].shape[1])

    all_dataset_passed = all(s.final_status == "PASSED" for s in dataset_summaries.values())
    all_split_passed = all(s.final_status == "PASSED" for s in split_summaries.values())

    final_status = "PASSED" if all_dataset_passed and all_split_passed else "FAILED_STEP8_CHECK"

    normal_statistics_source = {
        "path": normal_stats.get("_normal_statistics_path", ""),
        "dataset_key": normal_stats.get("dataset_key", "dataset1"),
        "split_name": normal_stats.get("split_name", "train"),
        "train_normal_valid_count": normal_stats.get("train_normal_valid_count", None),
        "mu_e": normal_stats.get("train_normal_energy_median_mu_e", None),
        "residual_covariance_regularized": normal_stats.get("residual_covariance_regularized", None),
        "leakage_rule": normal_stats.get("leakage_rule", {}),
    }

    report = FullEvidenceBuilderReport(
        dataset_summaries=dataset_summaries,
        split_summaries=split_summaries,
        scaler_summary=scaler_summary.to_dict(),
        normal_statistics_source=normal_statistics_source,
        evidence_builder_config=asdict(evidence_cfg),
        raw_xi_feature_columns=RAW_XI_FEATURE_COLUMNS,
        scaled_xi_feature_columns=scaler.scaled_feature_columns,
        final_step8_status=final_status,
    )

    energy_diagnostics = {
        "normal_statistics_source": normal_statistics_source,
        "dataset_energy_diagnostics": {
            key: {
                "residual_energy_quantiles_valid": value.residual_energy_quantiles_valid,
                "q_quantiles_valid": value.q_quantiles_valid,
                "accum_log_quantiles": value.accum_log_quantiles,
            }
            for key, value in dataset_summaries.items()
        },
        "split_energy_diagnostics": {
            key: {
                "rows": int(len(frame)),
                "valid_nu_rows": int((frame[evidence_cfg.nu_column].astype(int) == 1).sum()),
                "energy_quantiles_valid": _quantiles(
                    frame.loc[
                        frame[evidence_cfg.nu_column].astype(int) == 1,
                        evidence_cfg.residual_energy_column,
                    ]
                ),
                "q_quantiles_valid": _quantiles(
                    frame.loc[
                        frame[evidence_cfg.nu_column].astype(int) == 1,
                        evidence_cfg.q_column,
                    ]
                ),
                "accum_log_quantiles": _quantiles(frame[evidence_cfg.accum_log_column]),
            }
            for key, frame in split_frames.items()
        },
        "note": (
            "Extreme q100 values are diagnostics, not automatic failures. "
            "Models should use train-fitted robust scaled xi columns."
        ),
    }

    if save_outputs:
        summary_path = get_step8_summary_path(config)
        diagnostics_path = get_step8_energy_diagnostics_path(config)

        save_json(report.to_dict(), summary_path, indent=2)
        save_json(energy_diagnostics, diagnostics_path, indent=2)

        print(f"Saved Step 8 evidence summary JSON: {summary_path}")
        print(f"Saved Step 8 energy diagnostics JSON: {diagnostics_path}")
        print(f"Saved Step 8 xi scaler JSON: {scaler_path}")
        print(f"Saved Step 8 xi feature spec JSON: {feature_spec_path}")

    print("=" * 100)
    print("STEP 8 CAUSAL XI EVIDENCE BUILDER REPORT")
    print("=" * 100)

    for summary in dataset_summaries.values():
        _print_dataset_summary(summary)

    for summary in split_summaries.values():
        _print_split_summary(summary)

    print("=" * 100)
    print("STEP 8 SCALER SUMMARY")
    print("=" * 100)
    print(f"Scaler method                       : {scaler_summary.method}")
    print(f"Fitted on split                     : {scaler_summary.fitted_on_split}")
    print(f"Train rows used                     : {scaler_summary.train_rows_used}")
    print(f"Continuous xi columns               : {scaler_summary.continuous_feature_columns}")
    print(f"Passthrough xi columns              : {scaler_summary.passthrough_feature_columns}")
    print(f"Scaled model input columns          : {scaler_summary.scaled_feature_columns}")
    print(f"Leakage rule                        : {scaler_summary.leakage_rule}")
    print(f"Final scaler status                 : {scaler_summary.final_status}")

    print("=" * 100)
    print("STEP 8 FINAL STATUS")
    print("=" * 100)
    print(f"Final Step 8 status                 : {report.final_step8_status}")
    print("=" * 100)

    if report.final_step8_status != "PASSED":
        raise RuntimeError(f"Step 8 failed with status: {report.final_step8_status}")

    return report