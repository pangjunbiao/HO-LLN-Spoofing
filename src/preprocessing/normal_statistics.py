"""
Training-only normal residual statistics for AV-GPS causal spoofing detection.

Step 7 purpose:
- compute Sigma_r^tr from Dataset-1 TRAIN normal valid residuals only,
- compute inverse covariance and inverse square-root covariance,
- compute training-normal residual energy distribution,
- compute mu_e as the median training-normal residual energy,
- save diagnostics so we can inspect leakage safety and residual quality.

Critical rule:
No validation, internal-test, Dataset-2, Dataset-3, or Dataset-1-Normal samples
are allowed to estimate residual normal statistics.

Dataset-1-Normal may be diagnostic/reference later, but not an independent
supervised training source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import json

import numpy as np
import pandas as pd

from src.preprocessing.residual_builder import (
    FullResidualBuilderReport,
    get_residual_file_path,
    run_residual_builder_step,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_csv, save_json


@dataclass
class NormalStatisticsConfig:
    """Configuration for training-only normal residual statistics."""

    dataset_key: str
    split_name: str
    segment_column: str
    label_column: str
    normal_label: int

    residual_east_column: str
    residual_north_column: str
    residual_valid_column: str

    covariance_epsilon: float
    min_train_normal_samples: int
    center_covariance_estimation: bool
    subtract_residual_mean_for_energy: bool

    train_segments_file: str
    normal_statistics_json: str
    train_normal_samples_csv: str
    combined_step7_json: str


@dataclass
class NormalStatisticsSummary:
    """Summary of training-only normal statistics."""

    dataset_key: str
    split_name: str
    residual_input_path: str
    train_segments_path: str

    total_dataset1_rows: int
    total_dataset1_segments: int

    train_segment_count: int
    train_row_count: int
    train_normal_valid_count: int
    train_attack_valid_count: int

    covariance_epsilon: float
    residual_mean: List[float]
    residual_covariance_raw: List[List[float]]
    residual_covariance_regularized: List[List[float]]
    residual_inverse_covariance: List[List[float]]
    residual_inverse_sqrt_covariance: List[List[float]]

    train_normal_energy_count: int
    train_normal_energy_median_mu_e: float
    train_normal_energy_mean: float
    train_normal_energy_std: float
    train_normal_energy_quantiles: Dict[str, float]

    leakage_rule: Dict[str, Any]
    output_statistics_path: str
    output_train_normal_samples_path: str
    final_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FullStep7ResidualStatisticsReport:
    """Full Step-7 report containing residual builder + normal statistics."""

    residual_builder_report: Dict[str, Any]
    normal_statistics_summary: Dict[str, Any]
    final_step7_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residual_builder_report": self.residual_builder_report,
            "normal_statistics_summary": self.normal_statistics_summary,
            "final_step7_status": self.final_step7_status,
        }


def get_splits_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/splits directory."""
    value = get_by_path(config, "paths.splits_dir", "data/splits")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_processed_dir(config: Mapping[str, Any]) -> Path:
    """Resolve data/processed directory."""
    value = get_by_path(config, "paths.processed_data_dir", "data/processed")
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_step7_normal_statistics_path(config: Mapping[str, Any]) -> Path:
    """Resolve Step-7 normal statistics JSON path."""
    value = get_by_path(
        config,
        "paths.step7_normal_statistics_json",
        "results/tables/step7_normal_statistics.json",
    )
    return resolve_project_path(config, value)


def get_step7_combined_summary_path(config: Mapping[str, Any]) -> Path:
    """Resolve combined Step-7 summary JSON path."""
    value = get_by_path(
        config,
        "paths.step7_residual_normal_statistics_json",
        "results/tables/step7_residual_normal_statistics_summary.json",
    )
    return resolve_project_path(config, value)


def get_normal_statistics_config(config: Mapping[str, Any]) -> NormalStatisticsConfig:
    """
    Read normal statistics config.
    """
    return NormalStatisticsConfig(
        dataset_key=str(
            get_by_path(config, "preprocessing.normal_statistics.dataset_key", "dataset1")
        ),
        split_name=str(
            get_by_path(config, "preprocessing.normal_statistics.split_name", "train")
        ),
        segment_column=str(
            get_by_path(config, "preprocessing.normal_statistics.segment_column", "segment_id")
        ),
        label_column=str(
            get_by_path(config, "dataset.label_column", "Data Type")
        ),
        normal_label=int(
            get_by_path(config, "dataset.normal_label", 0)
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

        covariance_epsilon=float(
            get_by_path(config, "preprocessing.normal_statistics.covariance_epsilon", 1e-6)
        ),
        min_train_normal_samples=int(
            get_by_path(config, "preprocessing.normal_statistics.min_train_normal_samples", 100)
        ),
        center_covariance_estimation=bool(
            get_by_path(config, "preprocessing.normal_statistics.center_covariance_estimation", True)
        ),
        subtract_residual_mean_for_energy=bool(
            get_by_path(config, "preprocessing.normal_statistics.subtract_residual_mean_for_energy", False)
        ),

        train_segments_file=str(
            get_by_path(
                config,
                "dataset.split_files.train_segments",
                "dataset1_train_segments.json",
            )
        ),
        normal_statistics_json=str(
            get_by_path(
                config,
                "paths.step7_normal_statistics_json",
                "results/tables/step7_normal_statistics.json",
            )
        ),
        train_normal_samples_csv=str(
            get_by_path(
                config,
                "preprocessing.normal_statistics.train_normal_samples_csv",
                "dataset1_train_normal_residual_samples.csv",
            )
        ),
        combined_step7_json=str(
            get_by_path(
                config,
                "paths.step7_residual_normal_statistics_json",
                "results/tables/step7_residual_normal_statistics_summary.json",
            )
        ),
    )


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_train_segments_path(config: Mapping[str, Any], stats_cfg: NormalStatisticsConfig) -> Path:
    """Resolve train segment JSON path."""
    return (get_splits_dir(config) / stats_cfg.train_segments_file).resolve()


def load_train_segment_ids(config: Mapping[str, Any], stats_cfg: NormalStatisticsConfig) -> List[str]:
    """
    Load Dataset-1 train segment IDs from Step 4.
    """
    path = get_train_segments_path(config, stats_cfg)

    if not path.exists():
        raise FileNotFoundError(
            f"Train segment split file not found: {path}\n"
            "Run Step 4 first."
        )

    data = load_json_file(path)
    segments = data.get("segments", [])

    if not segments:
        raise ValueError(f"No train segments found in {path}")

    return [str(seg) for seg in segments]


def load_dataset1_residuals_for_statistics(
    config: Mapping[str, Any],
    stats_cfg: NormalStatisticsConfig,
) -> pd.DataFrame:
    """
    Load Dataset-1 residual file.
    """
    path = get_residual_file_path(config, stats_cfg.dataset_key)

    if not path.exists():
        raise FileNotFoundError(
            f"Residual file not found: {path}\n"
            "Run residual builder first."
        )

    return pd.read_csv(path, low_memory=False)


def select_train_normal_valid_residuals(
    df: pd.DataFrame,
    train_segment_ids: Sequence[str],
    stats_cfg: NormalStatisticsConfig,
) -> pd.DataFrame:
    """
    Select Dataset-1 train normal valid residual samples only.

    This is the only data allowed for Sigma_r^tr and mu_e.
    """
    required = [
        stats_cfg.segment_column,
        stats_cfg.label_column,
        stats_cfg.residual_east_column,
        stats_cfg.residual_north_column,
        stats_cfg.residual_valid_column,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for normal statistics: {missing}")

    train_set = set(str(seg) for seg in train_segment_ids)

    residual_east = pd.to_numeric(df[stats_cfg.residual_east_column], errors="coerce")
    residual_north = pd.to_numeric(df[stats_cfg.residual_north_column], errors="coerce")

    mask = (
        df[stats_cfg.segment_column].astype(str).isin(train_set)
        & (df[stats_cfg.label_column] == stats_cfg.normal_label)
        & (df[stats_cfg.residual_valid_column].astype(int) == 1)
        & residual_east.notna()
        & residual_north.notna()
        & np.isfinite(residual_east)
        & np.isfinite(residual_north)
    )

    out = df.loc[mask].copy()
    out[stats_cfg.residual_east_column] = residual_east.loc[mask]
    out[stats_cfg.residual_north_column] = residual_north.loc[mask]

    return out


def _symmetrize_2x2(matrix: np.ndarray) -> np.ndarray:
    """Return a strictly symmetric 2x2 matrix without calling np.linalg."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (2, 2):
        raise ValueError(f"Expected shape (2, 2), got {m.shape}")

    b = 0.5 * (m[0, 1] + m[1, 0])
    return np.array(
        [
            [float(m[0, 0]), float(b)],
            [float(b), float(m[1, 1])],
        ],
        dtype=float,
    )


def _eigenvalues_2x2_symmetric(matrix: np.ndarray) -> tuple[float, float]:
    """
    Analytic eigenvalues for a real symmetric 2x2 matrix.

    Avoids np.linalg to prevent OpenMP duplicate-runtime crashes on some
    Windows/Conda/PyTorch/MKL environments.
    """
    m = _symmetrize_2x2(matrix)
    a = float(m[0, 0])
    b = float(m[0, 1])
    d = float(m[1, 1])

    trace_half = 0.5 * (a + d)
    radius = float(np.sqrt(((a - d) * 0.5) ** 2 + b ** 2))

    lambda_max = trace_half + radius
    lambda_min = trace_half - radius

    return float(lambda_min), float(lambda_max)


def compute_covariance_matrix(
    residual_matrix: np.ndarray,
    epsilon: float,
    center_covariance_estimation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute raw and regularized residual covariance.

    This implementation avoids np.linalg and BLAS-heavy matrix products because
    the residual covariance is only 2x2. That also avoids OpenMP duplicate DLL
    crashes in some Windows/Conda environments.
    """
    x = np.asarray(residual_matrix, dtype=float)

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(
            f"Expected residual matrix shape (n, 2), got {x.shape}"
        )

    n = int(x.shape[0])

    if n < 2:
        raise ValueError("Need at least two residual samples to compute covariance.")

    x0 = x[:, 0].astype(float)
    x1 = x[:, 1].astype(float)

    if center_covariance_estimation:
        x0 = x0 - float(np.mean(x0))
        x1 = x1 - float(np.mean(x1))

    denom = max(float(n - 1), 1.0)

    cov00 = float(np.sum(x0 * x0) / denom)
    cov01 = float(np.sum(x0 * x1) / denom)
    cov11 = float(np.sum(x1 * x1) / denom)

    raw_cov = np.array(
        [
            [cov00, cov01],
            [cov01, cov11],
        ],
        dtype=float,
    )

    raw_cov = _symmetrize_2x2(raw_cov)

    regularized = raw_cov + float(epsilon) * np.eye(2, dtype=float)
    regularized = _symmetrize_2x2(regularized)

    min_eig, _ = _eigenvalues_2x2_symmetric(regularized)

    if min_eig <= 0:
        regularized = regularized + (abs(min_eig) + float(epsilon)) * np.eye(2, dtype=float)
        regularized = _symmetrize_2x2(regularized)

    return raw_cov, regularized

def matrix_inverse_and_inverse_sqrt(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute inverse covariance and inverse square-root covariance for a 2x2 SPD matrix.

    Uses closed-form 2x2 symmetric eigendecomposition instead of np.linalg.eigh
    to avoid OpenMP runtime conflicts.
    """
    cov = _symmetrize_2x2(covariance)

    a = float(cov[0, 0])
    b = float(cov[0, 1])
    d = float(cov[1, 1])

    lambda_min, lambda_max = _eigenvalues_2x2_symmetric(cov)

    lambda_min = max(float(lambda_min), 1e-12)
    lambda_max = max(float(lambda_max), 1e-12)

    # Analytic rotation angle for symmetric 2x2 matrix.
    theta = 0.5 * float(np.arctan2(2.0 * b, a - d))
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    # Eigenvector matrix columns correspond to lambda_max then lambda_min.
    q = np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=float,
    )

    inv_diag = np.diag([1.0 / lambda_max, 1.0 / lambda_min])
    inv_sqrt_diag = np.diag([1.0 / np.sqrt(lambda_max), 1.0 / np.sqrt(lambda_min)])

    inv_cov = q @ inv_diag @ q.T
    inv_sqrt = q @ inv_sqrt_diag @ q.T

    inv_cov = _symmetrize_2x2(inv_cov)
    inv_sqrt = _symmetrize_2x2(inv_sqrt)

    return inv_cov, inv_sqrt

def compute_mahalanobis_energy(
    residual_matrix: np.ndarray,
    inv_covariance: np.ndarray,
    residual_mean: np.ndarray,
    subtract_mean: bool,
) -> np.ndarray:
    """
    Compute residual energy e_t = r_t^T Sigma^{-1} r_t.

    Uses explicit 2D formula instead of np.einsum/BLAS.
    """
    x = np.asarray(residual_matrix, dtype=float)

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected residual matrix shape (n, 2), got {x.shape}")

    if subtract_mean:
        mean = np.asarray(residual_mean, dtype=float).reshape(1, 2)
        x = x - mean

    inv = _symmetrize_2x2(inv_covariance)

    x0 = x[:, 0]
    x1 = x[:, 1]

    energy = (
        float(inv[0, 0]) * x0 * x0
        + 2.0 * float(inv[0, 1]) * x0 * x1
        + float(inv[1, 1]) * x1 * x1
    )

    return np.asarray(energy, dtype=float)

def _safe_float(value: Any, digits: int = 10) -> float:
    """Safe rounded float for JSON."""
    if value is None:
        return 0.0

    if pd.isna(value):
        return 0.0

    return round(float(value), digits)


def _energy_quantiles(energy: np.ndarray) -> Dict[str, float]:
    """Compute energy quantiles for diagnostics."""
    quantile_values = {
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

    return {
        name: _safe_float(np.quantile(energy, q))
        for name, q in quantile_values.items()
    }


def compute_training_only_normal_statistics(
    config: Mapping[str, Any],
    save_outputs: bool = True,
) -> NormalStatisticsSummary:
    """
    Compute training-only normal residual statistics.
    """
    stats_cfg = get_normal_statistics_config(config)

    residual_path = get_residual_file_path(config, stats_cfg.dataset_key)
    train_segments_path = get_train_segments_path(config, stats_cfg)

    df = load_dataset1_residuals_for_statistics(config, stats_cfg)
    train_segments = load_train_segment_ids(config, stats_cfg)

    train_set = set(train_segments)

    train_mask = df[stats_cfg.segment_column].astype(str).isin(train_set)
    train_df = df.loc[train_mask].copy()

    train_normal_valid = select_train_normal_valid_residuals(
        df=df,
        train_segment_ids=train_segments,
        stats_cfg=stats_cfg,
    )

    train_attack_valid_count = int(
        (
            train_mask
            & (df[stats_cfg.label_column] != stats_cfg.normal_label)
            & (df[stats_cfg.residual_valid_column].astype(int) == 1)
        ).sum()
    )

    if len(train_normal_valid) < stats_cfg.min_train_normal_samples:
        raise ValueError(
            f"Not enough train normal valid residuals. "
            f"Need at least {stats_cfg.min_train_normal_samples}, got {len(train_normal_valid)}."
        )

    residual_matrix = train_normal_valid[
        [stats_cfg.residual_east_column, stats_cfg.residual_north_column]
    ].to_numpy(dtype=float)

    residual_mean = residual_matrix.mean(axis=0)

    raw_cov, reg_cov = compute_covariance_matrix(
        residual_matrix=residual_matrix,
        epsilon=stats_cfg.covariance_epsilon,
        center_covariance_estimation=stats_cfg.center_covariance_estimation,
    )

    inv_cov, inv_sqrt = matrix_inverse_and_inverse_sqrt(reg_cov)

    energy = compute_mahalanobis_energy(
        residual_matrix=residual_matrix,
        inv_covariance=inv_cov,
        residual_mean=residual_mean,
        subtract_mean=stats_cfg.subtract_residual_mean_for_energy,
    )

    train_normal_valid = train_normal_valid.copy()
    train_normal_valid["train_normal_mahalanobis_energy"] = energy

    mu_e = float(np.median(energy))

    total_segments = int(df[stats_cfg.segment_column].astype(str).nunique())

    leakage_rule = {
        "statistics_use_dataset1_only": True,
        "statistics_use_train_split_only": True,
        "statistics_use_normal_labels_only": True,
        "statistics_use_valid_residuals_only": True,
        "validation_split_used": False,
        "internal_test_split_used": False,
        "dataset1_normal_used_as_independent_training_data": False,
        "dataset2_used": False,
        "dataset3_used": False,
        "train_segment_count": len(train_segments),
        "train_segments_preview": train_segments[:10],
    }

    output_statistics_path = get_step7_normal_statistics_path(config)
    output_train_normal_samples_path = (
        get_processed_dir(config) / stats_cfg.train_normal_samples_csv
    ).resolve()

    summary = NormalStatisticsSummary(
        dataset_key=stats_cfg.dataset_key,
        split_name=stats_cfg.split_name,
        residual_input_path=str(residual_path),
        train_segments_path=str(train_segments_path),

        total_dataset1_rows=int(len(df)),
        total_dataset1_segments=total_segments,

        train_segment_count=int(len(train_segments)),
        train_row_count=int(len(train_df)),
        train_normal_valid_count=int(len(train_normal_valid)),
        train_attack_valid_count=train_attack_valid_count,

        covariance_epsilon=float(stats_cfg.covariance_epsilon),
        residual_mean=[
            _safe_float(residual_mean[0]),
            _safe_float(residual_mean[1]),
        ],
        residual_covariance_raw=raw_cov.round(10).tolist(),
        residual_covariance_regularized=reg_cov.round(10).tolist(),
        residual_inverse_covariance=inv_cov.round(10).tolist(),
        residual_inverse_sqrt_covariance=inv_sqrt.round(10).tolist(),

        train_normal_energy_count=int(len(energy)),
        train_normal_energy_median_mu_e=_safe_float(mu_e),
        train_normal_energy_mean=_safe_float(float(np.mean(energy))),
        train_normal_energy_std=_safe_float(float(np.std(energy, ddof=1))),
        train_normal_energy_quantiles=_energy_quantiles(energy),

        leakage_rule=leakage_rule,
        output_statistics_path=str(output_statistics_path),
        output_train_normal_samples_path=str(output_train_normal_samples_path),
        final_status="PASSED",
    )

    if save_outputs:
        save_json(summary.to_dict(), output_statistics_path, indent=2)

        # Save a compact diagnostic CSV for inspection, not for training models.
        keep_cols = [
            col for col in [
                "source_key",
                "source_file",
                stats_cfg.segment_column,
                "within_segment_index",
                stats_cfg.label_column,
                stats_cfg.residual_valid_column,
                stats_cfg.residual_east_column,
                stats_cfg.residual_north_column,
                "residual_norm_m",
                "delta_t_seconds",
                "train_normal_mahalanobis_energy",
            ]
            if col in train_normal_valid.columns
        ]
        save_csv(train_normal_valid.loc[:, keep_cols], output_train_normal_samples_path, index=False)

        print(f"Saved Step 7 normal statistics JSON: {output_statistics_path}")
        print(f"Saved Step 7 train-normal residual samples CSV: {output_train_normal_samples_path}")

    return summary


def print_normal_statistics_summary(summary: NormalStatisticsSummary) -> None:
    """Print training-only normal statistics summary."""
    print("=" * 100)
    print("STEP 7 TRAINING-ONLY NORMAL STATISTICS SUMMARY")
    print("=" * 100)
    print(f"Dataset key                         : {summary.dataset_key}")
    print(f"Split used                          : {summary.split_name}")
    print(f"Residual input path                 : {summary.residual_input_path}")
    print(f"Train segments path                 : {summary.train_segments_path}")
    print(f"Total Dataset-1 rows                : {summary.total_dataset1_rows}")
    print(f"Total Dataset-1 segments            : {summary.total_dataset1_segments}")
    print(f"Train segment count                 : {summary.train_segment_count}")
    print(f"Train row count                     : {summary.train_row_count}")
    print(f"Train normal valid count            : {summary.train_normal_valid_count}")
    print(f"Train attack valid count            : {summary.train_attack_valid_count}")
    print(f"Residual mean [east, north]          : {summary.residual_mean}")
    print(f"Residual covariance regularized      : {summary.residual_covariance_regularized}")
    print(f"mu_e median normal energy            : {summary.train_normal_energy_median_mu_e}")
    print(f"Energy quantiles                     : {summary.train_normal_energy_quantiles}")
    print(f"Leakage rule                         : {summary.leakage_rule}")
    print(f"Statistics JSON saved to             : {summary.output_statistics_path}")
    print(f"Train-normal samples saved to        : {summary.output_train_normal_samples_path}")
    print(f"Final status                         : {summary.final_status}")
    print("=" * 100)


def run_residual_and_normal_statistics_step(
    config: Mapping[str, Any],
    dataset_keys: Optional[Sequence[str]] = None,
    save_outputs: bool = True,
) -> FullStep7ResidualStatisticsReport:
    """
    Main Step-7 entry point.

    This runs:
    1. residual construction for all datasets,
    2. training-only normal statistics from Dataset-1 train normal residuals,
    3. combined Step-7 summary saving.

    Saves:
    - data/interim/*_residual.csv
    - results/tables/step7_residual_summary.json
    - results/tables/step7_normal_statistics.json
    - results/tables/step7_residual_normal_statistics_summary.json
    - data/processed/dataset1_train_normal_residual_samples.csv
    """
    residual_report: FullResidualBuilderReport = run_residual_builder_step(
        config=config,
        dataset_keys=dataset_keys,
        save_outputs=save_outputs,
    )

    normal_summary = compute_training_only_normal_statistics(
        config=config,
        save_outputs=save_outputs,
    )

    final_status = (
        "PASSED"
        if residual_report.final_residual_builder_status == "PASSED"
        and normal_summary.final_status == "PASSED"
        else "FAILED_STEP7_CHECK"
    )

    combined_report = FullStep7ResidualStatisticsReport(
        residual_builder_report=residual_report.to_dict(),
        normal_statistics_summary=normal_summary.to_dict(),
        final_step7_status=final_status,
    )

    if save_outputs:
        combined_path = get_step7_combined_summary_path(config)
        save_json(combined_report.to_dict(), combined_path, indent=2)
        print(f"Saved combined Step 7 summary JSON: {combined_path}")

    print_normal_statistics_summary(normal_summary)

    print("=" * 100)
    print("STEP 7 RESIDUAL + NORMAL STATISTICS FINAL STATUS")
    print("=" * 100)
    print(f"Final Step 7 status                  : {combined_report.final_step7_status}")
    print("=" * 100)

    if combined_report.final_step7_status != "PASSED":
        raise RuntimeError(
            f"Step 7 failed with status: {combined_report.final_step7_status}"
        )

    return combined_report

# =============================================================================
# Step 8 helpers: validated loading of training-only normal statistics
# =============================================================================


def _matrix_2x2_from_json(value: Any, key_name: str) -> np.ndarray:
    """
    Convert a JSON matrix value into a finite 2x2 numpy array.

    Used by Step 8 before building eta_t = Sigma^{-1/2} r_t.
    """
    matrix = np.asarray(value, dtype=float)

    if matrix.shape != (2, 2):
        raise ValueError(
            f"Normal-statistics key '{key_name}' must have shape (2, 2), "
            f"got {matrix.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"Normal-statistics key '{key_name}' contains non-finite values."
        )

    return _symmetrize_2x2(matrix)


def validate_training_normal_statistics_for_xi(
    stats: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate Step-7 normal statistics before Step-8 xi construction.

    This is a safety gate for Step 8:
    - confirms required matrices exist,
    - confirms covariance is positive definite after regularization,
    - confirms mu_e is finite,
    - confirms statistics came from Dataset-1 train normal valid residuals only,
    - reports energy outlier diagnostics without failing automatically.

    Returns a JSON-safe validation report.
    """
    config = config or {}

    required_keys = [
        "dataset_key",
        "split_name",
        "train_normal_valid_count",
        "residual_covariance_regularized",
        "residual_inverse_covariance",
        "residual_inverse_sqrt_covariance",
        "train_normal_energy_median_mu_e",
        "train_normal_energy_quantiles",
        "leakage_rule",
        "final_status",
    ]

    missing_keys = [key for key in required_keys if key not in stats]

    if missing_keys:
        return {
            "final_status": "FAILED_MISSING_KEYS",
            "missing_keys": missing_keys,
        }

    expected_dataset = str(
        get_by_path(config, "preprocessing.normal_statistics.dataset_key", "dataset1")
    )
    expected_split = str(
        get_by_path(config, "preprocessing.normal_statistics.split_name", "train")
    )

    min_train_normal_samples = int(
        get_by_path(config, "preprocessing.normal_statistics.min_train_normal_samples", 100)
    )

    dataset_key = str(stats.get("dataset_key", ""))
    split_name = str(stats.get("split_name", ""))
    final_status = str(stats.get("final_status", ""))

    train_normal_valid_count = int(stats.get("train_normal_valid_count", 0))

    covariance = _matrix_2x2_from_json(
        stats.get("residual_covariance_regularized"),
        "residual_covariance_regularized",
    )
    inverse_covariance = _matrix_2x2_from_json(
        stats.get("residual_inverse_covariance"),
        "residual_inverse_covariance",
    )
    inverse_sqrt_covariance = _matrix_2x2_from_json(
        stats.get("residual_inverse_sqrt_covariance"),
        "residual_inverse_sqrt_covariance",
    )

    cov_min_eig, cov_max_eig = _eigenvalues_2x2_symmetric(covariance)
    inv_cov_min_eig, inv_cov_max_eig = _eigenvalues_2x2_symmetric(inverse_covariance)
    inv_sqrt_min_eig, inv_sqrt_max_eig = _eigenvalues_2x2_symmetric(inverse_sqrt_covariance)

    mu_e = float(stats.get("train_normal_energy_median_mu_e", np.nan))

    leakage_rule = dict(stats.get("leakage_rule", {}))

    leakage_checks = {
        "statistics_use_dataset1_only": leakage_rule.get("statistics_use_dataset1_only") is True,
        "statistics_use_train_split_only": leakage_rule.get("statistics_use_train_split_only") is True,
        "statistics_use_normal_labels_only": leakage_rule.get("statistics_use_normal_labels_only") is True,
        "statistics_use_valid_residuals_only": leakage_rule.get("statistics_use_valid_residuals_only") is True,
        "validation_split_used_false": leakage_rule.get("validation_split_used") is False,
        "internal_test_split_used_false": leakage_rule.get("internal_test_split_used") is False,
        "dataset1_normal_used_false": leakage_rule.get("dataset1_normal_used_as_independent_training_data") is False,
        "dataset2_used_false": leakage_rule.get("dataset2_used") is False,
        "dataset3_used_false": leakage_rule.get("dataset3_used") is False,
    }

    energy_quantiles = dict(stats.get("train_normal_energy_quantiles", {}))

    q50 = float(energy_quantiles.get("q50", 0.0))
    q95 = float(energy_quantiles.get("q95", 0.0))
    q99 = float(energy_quantiles.get("q99", 0.0))
    q100 = float(energy_quantiles.get("q100", 0.0))

    energy_outlier_warning = bool(
        np.isfinite(q100)
        and np.isfinite(q99)
        and (
            q100 > 1000.0
            or (q99 > 0.0 and q100 / q99 > 100.0)
        )
    )

    covariance_condition_estimate = (
        float(cov_max_eig / cov_min_eig)
        if cov_min_eig > 0.0
        else float("inf")
    )

    checks = {
        "dataset_key_is_expected": dataset_key == expected_dataset,
        "split_name_is_expected": split_name == expected_split,
        "step7_final_status_passed": final_status == "PASSED",
        "enough_train_normal_samples": train_normal_valid_count >= min_train_normal_samples,
        "mu_e_is_finite_nonnegative": bool(np.isfinite(mu_e) and mu_e >= 0.0),
        "covariance_positive_definite": bool(cov_min_eig > 0.0 and np.isfinite(cov_min_eig)),
        "inverse_covariance_positive_definite": bool(inv_cov_min_eig > 0.0 and np.isfinite(inv_cov_min_eig)),
        "inverse_sqrt_covariance_positive_definite": bool(inv_sqrt_min_eig > 0.0 and np.isfinite(inv_sqrt_min_eig)),
        "all_leakage_checks_passed": all(leakage_checks.values()),
    }

    final_validation_status = (
        "PASSED"
        if all(checks.values())
        else "FAILED_NORMAL_STATISTICS_VALIDATION"
    )

    return {
        "final_status": final_validation_status,
        "checks": checks,
        "leakage_checks": leakage_checks,
        "dataset_key": dataset_key,
        "split_name": split_name,
        "train_normal_valid_count": train_normal_valid_count,
        "mu_e": _safe_float(mu_e),
        "covariance_min_eigenvalue": _safe_float(cov_min_eig),
        "covariance_max_eigenvalue": _safe_float(cov_max_eig),
        "covariance_condition_estimate": _safe_float(covariance_condition_estimate),
        "inverse_covariance_min_eigenvalue": _safe_float(inv_cov_min_eig),
        "inverse_sqrt_covariance_min_eigenvalue": _safe_float(inv_sqrt_min_eig),
        "energy_quantiles": {
            "q50": _safe_float(q50),
            "q95": _safe_float(q95),
            "q99": _safe_float(q99),
            "q100": _safe_float(q100),
        },
        "energy_outlier_warning": energy_outlier_warning,
        "energy_outlier_note": (
            "Extreme training-normal residual energy exists. This is not an automatic "
            "failure because Step 8 uses median mu_e and robust train-only xi scaling."
            if energy_outlier_warning
            else "No severe energy outlier warning."
        ),
    }


def get_step8_normal_statistics_validation_path(config: Mapping[str, Any]) -> Path:
    """
    Resolve Step-8 normal-statistics validation JSON path.
    """
    value = get_by_path(
        config,
        "paths.step8_normal_statistics_validation_json",
        "results/tables/step8_normal_statistics_validation.json",
    )
    return resolve_project_path(config, value)


def load_training_normal_statistics_for_xi(
    config: Mapping[str, Any],
    save_validation_report: bool = True,
) -> Dict[str, Any]:
    """
    Load and validate Step-7 normal statistics for Step-8 xi construction.

    This is the preferred Step-8 loader.

    It prevents Step 8 from silently using:
    - wrong dataset statistics,
    - wrong split statistics,
    - non-normal statistics,
    - leaked external/test statistics,
    - invalid covariance matrices.
    """
    path = get_step7_normal_statistics_path(config)

    if not path.exists():
        raise FileNotFoundError(
            f"Step-7 normal statistics JSON not found: {path}\n"
            "Run Step 7 before Step 8."
        )

    stats = load_json_file(path)
    validation_report = validate_training_normal_statistics_for_xi(
        stats=stats,
        config=config,
    )

    stats["_normal_statistics_path"] = str(path)
    stats["_xi_validation_report"] = validation_report

    if save_validation_report:
        validation_path = get_step8_normal_statistics_validation_path(config)
        save_json(validation_report, validation_path, indent=2)
        stats["_xi_validation_report_path"] = str(validation_path)

    if validation_report.get("final_status") != "PASSED":
        raise RuntimeError(
            "Step-7 normal statistics failed Step-8 validation. "
            f"Validation report: {validation_report}"
        )

    return dict(stats)


def print_training_normal_statistics_for_xi_summary(
    stats: Mapping[str, Any],
) -> None:
    """
    Print a compact Step-8 normal-statistics validation summary.
    """
    validation = dict(stats.get("_xi_validation_report", {}))

    print("=" * 100)
    print("STEP 8 NORMAL-STATISTICS VALIDATION SUMMARY")
    print("=" * 100)
    print(f"Statistics path                      : {stats.get('_normal_statistics_path', '')}")
    print(f"Dataset key                          : {stats.get('dataset_key', '')}")
    print(f"Split name                           : {stats.get('split_name', '')}")
    print(f"Train normal valid count             : {stats.get('train_normal_valid_count', '')}")
    print(f"mu_e                                 : {stats.get('train_normal_energy_median_mu_e', '')}")
    print(f"Validation status                    : {validation.get('final_status', '')}")
    print(f"Leakage checks                       : {validation.get('leakage_checks', {})}")
    print(f"Covariance condition estimate         : {validation.get('covariance_condition_estimate', '')}")
    print(f"Energy outlier warning               : {validation.get('energy_outlier_warning', '')}")
    print(f"Energy outlier note                  : {validation.get('energy_outlier_note', '')}")
    print("=" * 100)