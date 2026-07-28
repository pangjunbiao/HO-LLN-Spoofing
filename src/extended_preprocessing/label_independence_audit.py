"""
Label-independence audit for the isolated corrected-evidence branch.

This module does not modify the legacy project. It verifies two narrower,
scientifically accurate claims:

1. Per-row validity does not depend on attack labels when
   invalidate_attack_to_normal_boundary=False.
2. After the training-normal reference statistics have been fitted and frozen,
   the sample-wise xi evidence transformation does not depend on the row label.

Training labels are still used legitimately to identify Dataset-1 training-normal
samples when fitting the residual covariance and energy reference. Validation,
test, Dataset-2, and Dataset-3 labels must not be used to construct xi values.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from src.preprocessing.evidence_builder import (
    RAW_XI_FEATURE_COLUMNS,
    build_xi_for_dataset,
    get_evidence_builder_config,
)
from src.preprocessing.validity_mask import build_preliminary_validity_mask
from src.utils.config import get_by_path


@dataclass
class ValidityIndependenceResult:
    dataset_key: str
    rows: int
    label_column: str
    compared_columns: list[str]
    changed_rows_by_column: Dict[str, int]
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LegacyRuleDifferenceResult:
    dataset_key: str
    rows: int
    attack_to_normal_transition_rows: int
    nu_prelim_rows_changed: int
    valid_transition_rows_changed: int
    reason_rows_changed: int
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceIndependenceResult:
    dataset_key: str
    rows: int
    label_column: str
    compared_columns: list[str]
    maximum_absolute_difference: Dict[str, float]
    mismatched_rows_by_column: Dict[str, int]
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _plain_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a mutable deep copy of a Config/mapping."""
    if hasattr(config, "to_dict"):
        return copy.deepcopy(config.to_dict())
    return copy.deepcopy(dict(config))


def _set_nested(config: Dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    current = config
    for key in keys[:-1]:
        node = current.get(key)
        if not isinstance(node, dict):
            node = {}
            current[key] = node
        current = node
    current[keys[-1]] = value


def _mutate_binary_labels(
    df: pd.DataFrame,
    label_column: str,
    normal_label: int,
    attack_label: int,
) -> pd.DataFrame:
    """
    Flip binary labels while leaving every physical and temporal column unchanged.

    Unknown/nonbinary labels are preserved so the audit never invents a new class.
    """
    if label_column not in df.columns:
        raise KeyError(f"Missing label column for audit: {label_column}")

    out = df.copy()
    labels = pd.to_numeric(out[label_column], errors="coerce")

    flipped = labels.copy()
    flipped.loc[labels == normal_label] = attack_label
    flipped.loc[labels == attack_label] = normal_label

    out[label_column] = flipped.where(labels.notna(), out[label_column])
    return out


def _series_changed_count(left: pd.Series, right: pd.Series) -> int:
    if len(left) != len(right):
        raise ValueError(f"Series length mismatch: {len(left)} vs {len(right)}")

    left_obj = left.astype(object).where(left.notna(), "<NA>")
    right_obj = right.astype(object).where(right.notna(), "<NA>")
    return int((left_obj.to_numpy() != right_obj.to_numpy()).sum())


def audit_preliminary_validity_label_independence(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    dataset_key: str,
) -> ValidityIndependenceResult:
    """
    Verify that changing labels alone does not change preliminary validity.

    The audit intentionally allows the diagnostic transition columns
    normal_to_attack_transition and attack_to_normal_transition to change.
    They are metadata; they must not affect nu_prelim under the corrected config.
    """
    flag = bool(
        get_by_path(
            config,
            "preprocessing.validity.invalidate_attack_to_normal_boundary",
            True,
        )
    )
    if flag:
        raise AssertionError(
            "Corrected branch requires "
            "preprocessing.validity.invalidate_attack_to_normal_boundary=false."
        )

    label_column = str(get_by_path(config, "dataset.label_column", "Data Type"))
    normal_label = int(get_by_path(config, "dataset.normal_label", 0))
    attack_label = int(get_by_path(config, "dataset.attack_label", 1))
    segment_column = str(
        get_by_path(config, "preprocessing.segmentation.segment_column", "segment_id")
    )

    original = build_preliminary_validity_mask(
        df=df,
        config=config,
        segment_col=segment_column,
        copy=True,
    )
    mutated_input = _mutate_binary_labels(
        df=df,
        label_column=label_column,
        normal_label=normal_label,
        attack_label=attack_label,
    )
    mutated = build_preliminary_validity_mask(
        df=mutated_input,
        config=config,
        segment_col=segment_column,
        copy=True,
    )

    compared_columns = [
        "valid_transition_prelim",
        "nu_prelim",
        "invalid_transition_reason",
    ]
    missing = [
        column
        for column in compared_columns
        if column not in original.columns or column not in mutated.columns
    ]
    if missing:
        raise KeyError(f"Validity audit missing output columns: {missing}")

    changed = {
        column: _series_changed_count(original[column], mutated[column])
        for column in compared_columns
    }
    passed = all(value == 0 for value in changed.values())

    return ValidityIndependenceResult(
        dataset_key=str(dataset_key),
        rows=int(len(df)),
        label_column=label_column,
        compared_columns=compared_columns,
        changed_rows_by_column=changed,
        passed=bool(passed),
    )


def compare_legacy_and_corrected_validity_rules(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    dataset_key: str,
) -> LegacyRuleDifferenceResult:
    """Measure exactly how many rows differ between the legacy and corrected rule."""
    corrected_config = _plain_config(config)
    legacy_config = _plain_config(config)

    _set_nested(
        corrected_config,
        "preprocessing.validity.invalidate_attack_to_normal_boundary",
        False,
    )
    _set_nested(
        legacy_config,
        "preprocessing.validity.invalidate_attack_to_normal_boundary",
        True,
    )

    segment_column = str(
        get_by_path(config, "preprocessing.segmentation.segment_column", "segment_id")
    )

    corrected = build_preliminary_validity_mask(
        df=df,
        config=corrected_config,
        segment_col=segment_column,
        copy=True,
    )
    legacy = build_preliminary_validity_mask(
        df=df,
        config=legacy_config,
        segment_col=segment_column,
        copy=True,
    )

    transition_rows = (
        int(corrected["attack_to_normal_transition"].fillna(False).astype(bool).sum())
        if "attack_to_normal_transition" in corrected.columns
        else 0
    )
    nu_changed = _series_changed_count(
        legacy["nu_prelim"], corrected["nu_prelim"]
    )
    valid_changed = _series_changed_count(
        legacy["valid_transition_prelim"],
        corrected["valid_transition_prelim"],
    )
    reason_changed = _series_changed_count(
        legacy["invalid_transition_reason"],
        corrected["invalid_transition_reason"],
    )

    expected_upper_bound = transition_rows
    passed = (
        nu_changed <= expected_upper_bound
        and valid_changed <= expected_upper_bound
        and reason_changed <= expected_upper_bound
    )

    return LegacyRuleDifferenceResult(
        dataset_key=str(dataset_key),
        rows=int(len(df)),
        attack_to_normal_transition_rows=transition_rows,
        nu_prelim_rows_changed=nu_changed,
        valid_transition_rows_changed=valid_changed,
        reason_rows_changed=reason_changed,
        passed=bool(passed),
    )


def _evidence_columns(config: Mapping[str, Any]) -> list[str]:
    cfg = get_evidence_builder_config(config)
    ordered = [
        cfg.eta_east_column,
        cfg.eta_north_column,
        cfg.eta_dot_east_column,
        cfg.eta_dot_north_column,
        cfg.eta_ddot_east_column,
        cfg.eta_ddot_north_column,
        cfg.q_column,
        cfg.accum_log_column,
        cfg.nu_column,
        cfg.residual_energy_column,
        cfg.q_raw_column,
        cfg.dot_valid_column,
        cfg.ddot_valid_column,
        cfg.accum_raw_column,
        cfg.xi_valid_column,
    ]

    # Preserve order while removing duplicates.
    return list(dict.fromkeys([*RAW_XI_FEATURE_COLUMNS, *ordered]))


def audit_evidence_label_independence(
    residual_df: pd.DataFrame,
    normal_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    dataset_key: str,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-8,
) -> EvidenceIndependenceResult:
    """
    Verify sample-wise xi construction with frozen normal statistics.

    The normal-reference statistics are intentionally kept fixed. The only
    changed input is the label column.
    """
    label_column = str(get_by_path(config, "dataset.label_column", "Data Type"))
    normal_label = int(get_by_path(config, "dataset.normal_label", 0))
    attack_label = int(get_by_path(config, "dataset.attack_label", 1))

    original_xi, _ = build_xi_for_dataset(
        df=residual_df,
        dataset_key=dataset_key,
        normal_stats=normal_stats,
        config=config,
    )
    mutated_residual = _mutate_binary_labels(
        df=residual_df,
        label_column=label_column,
        normal_label=normal_label,
        attack_label=attack_label,
    )
    mutated_xi, _ = build_xi_for_dataset(
        df=mutated_residual,
        dataset_key=dataset_key,
        normal_stats=normal_stats,
        config=config,
    )

    columns = _evidence_columns(config)
    missing = [
        column
        for column in columns
        if column not in original_xi.columns or column not in mutated_xi.columns
    ]
    if missing:
        raise KeyError(f"Evidence audit missing columns: {missing}")

    maximum_absolute_difference: Dict[str, float] = {}
    mismatched_rows: Dict[str, int] = {}

    for column in columns:
        left = pd.to_numeric(original_xi[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(mutated_xi[column], errors="coerce").to_numpy(dtype=float)

        if left.shape != right.shape:
            raise ValueError(
                f"Evidence audit shape mismatch for {column}: "
                f"{left.shape} vs {right.shape}"
            )

        finite_pair = np.isfinite(left) & np.isfinite(right)
        both_nonfinite = ~np.isfinite(left) & ~np.isfinite(right)
        close = np.zeros(left.shape, dtype=bool)
        close[finite_pair] = np.isclose(
            left[finite_pair],
            right[finite_pair],
            atol=float(atol),
            rtol=float(rtol),
        )
        close[both_nonfinite] = True

        differences = np.abs(left[finite_pair] - right[finite_pair])
        max_difference = float(differences.max()) if differences.size else 0.0

        maximum_absolute_difference[column] = (
            max_difference if math.isfinite(max_difference) else float("inf")
        )
        mismatched_rows[column] = int((~close).sum())

    passed = all(value == 0 for value in mismatched_rows.values())

    return EvidenceIndependenceResult(
        dataset_key=str(dataset_key),
        rows=int(len(residual_df)),
        label_column=label_column,
        compared_columns=columns,
        maximum_absolute_difference=maximum_absolute_difference,
        mismatched_rows_by_column=mismatched_rows,
        passed=bool(passed),
    )


def run_label_independence_audit(
    segmented_frames: Mapping[str, pd.DataFrame],
    residual_frames: Mapping[str, pd.DataFrame],
    normal_stats: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run all corrected-branch label-independence checks."""
    atol = float(
        get_by_path(
            config,
            "extended_preprocessing.audit.numeric_atol",
            1.0e-10,
        )
    )
    rtol = float(
        get_by_path(
            config,
            "extended_preprocessing.audit.numeric_rtol",
            1.0e-8,
        )
    )

    validity_results: Dict[str, Any] = {}
    legacy_differences: Dict[str, Any] = {}
    evidence_results: Dict[str, Any] = {}

    for dataset_key, frame in segmented_frames.items():
        validity_results[dataset_key] = (
            audit_preliminary_validity_label_independence(
                df=frame,
                config=config,
                dataset_key=dataset_key,
            ).to_dict()
        )
        legacy_differences[dataset_key] = (
            compare_legacy_and_corrected_validity_rules(
                df=frame,
                config=config,
                dataset_key=dataset_key,
            ).to_dict()
        )

    for dataset_key, frame in residual_frames.items():
        evidence_results[dataset_key] = (
            audit_evidence_label_independence(
                residual_df=frame,
                normal_stats=normal_stats,
                config=config,
                dataset_key=dataset_key,
                atol=atol,
                rtol=rtol,
            ).to_dict()
        )

    passed = (
        all(item["passed"] for item in validity_results.values())
        and all(item["passed"] for item in legacy_differences.values())
        and all(item["passed"] for item in evidence_results.values())
    )

    return {
        "status": "PASSED" if passed else "FAILED",
        "scientific_scope": {
            "label_used_for_per_row_validity": False,
            "label_used_for_samplewise_evidence_after_reference_fit": False,
            "training_labels_used_to_fit_normal_reference_statistics": True,
            "validation_test_dataset2_dataset3_labels_used_for_evidence": False,
            "labels_used_for_supervised_training_and_evaluation": True,
        },
        "validity_label_independence": validity_results,
        "legacy_vs_corrected_rule": legacy_differences,
        "samplewise_evidence_label_independence": evidence_results,
    }
