"""
Diagnostic occlusion tests for the trained full proposed model.

Step 14 purpose:
- Run inference-time masking tests for important evidence groups.
- Measure how much metrics change when eta, eta_dot, eta_ddot, q,
  accumulated weak evidence, nu, or branch-level groups are neutralized.
- Diagnose module contribution before official retrained ablation.

Important:
- These are diagnostic occlusion tests only.
- They do not retrain the model.
- They are not official ablations.
- Official ablations later must be trained from scratch using the same protocol.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.diagnostics.feature_importance import (
    collect_predictions_with_optional_occlusion,
)
from src.diagnostics.module_usage import (
    FeatureGroupSpec,
    Step14DiagnosticsContext,
    load_step14_context,
    save_json_safe,
)
from src.evaluation.evaluate_dataset1 import (
    EvaluationPredictionBundle,
    evaluate_bundle_with_threshold,
)
from src.utils.config import get_by_path
from src.utils.io import ensure_dir


@dataclass
class OcclusionScenario:
    """One diagnostic occlusion scenario."""

    name: str
    columns: List[str]
    indices: List[int]
    scenario_type: str
    description: str

    def to_feature_group(self) -> FeatureGroupSpec:
        return FeatureGroupSpec(
            name=self.name,
            columns=list(self.columns),
            indices=list(self.indices),
            description=self.description,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OcclusionTestResult:
    """Metric change caused by one diagnostic occlusion scenario."""

    split: str
    scenario_name: str
    scenario_type: str
    columns: List[str]
    indices: List[int]
    replacement_strategy: str

    baseline_auprc: Optional[float]
    occluded_auprc: Optional[float]
    delta_auprc: Optional[float]
    relative_delta_auprc: Optional[float]

    baseline_f1: Optional[float]
    occluded_f1: Optional[float]
    delta_f1: Optional[float]
    relative_delta_f1: Optional[float]

    baseline_precision: Optional[float]
    occluded_precision: Optional[float]
    delta_precision: Optional[float]

    baseline_recall: Optional[float]
    occluded_recall: Optional[float]
    delta_recall: Optional[float]

    baseline_fpr: Optional[float]
    occluded_fpr: Optional[float]
    delta_fpr: Optional[float]

    baseline_attack_detection_rate: Optional[float]
    occluded_attack_detection_rate: Optional[float]
    delta_attack_detection_rate: Optional[float]

    baseline_detection_delay: Optional[float]
    occluded_detection_delay: Optional[float]
    delta_detection_delay: Optional[float]

    baseline_tp: int
    baseline_fp: int
    baseline_tn: int
    baseline_fn: int

    occluded_tp: int
    occluded_fp: int
    occluded_tn: int
    occluded_fn: int

    delta_tp: int
    delta_fp: int
    delta_tn: int
    delta_fn: int

    mean_abs_probability_change: Optional[float]
    mean_signed_probability_change: Optional[float]
    max_abs_probability_change: Optional[float]

    valid_rows: int
    runtime_seconds: float
    diagnostic_only: bool
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> Optional[float]:
    """Convert to finite float or None."""
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


def _delta(occluded: Any, baseline: Any) -> Optional[float]:
    """Return occluded - baseline for finite values."""
    a = _safe_float(occluded)
    b = _safe_float(baseline)

    if a is None or b is None:
        return None

    return float(a - b)


def _relative_delta(occluded: Any, baseline: Any) -> Optional[float]:
    """Return relative delta: (occluded - baseline) / abs(baseline)."""
    a = _safe_float(occluded)
    b = _safe_float(baseline)

    if a is None or b is None:
        return None

    if abs(b) < 1.0e-12:
        return None

    return float((a - b) / abs(b))


def _int_metric(metrics: Mapping[str, Any], key: str) -> int:
    """Read integer metric safely."""
    value = metrics.get(key, 0)

    if value is None:
        return 0

    try:
        return int(value)
    except Exception:
        return 0


def _probability_change(
    baseline_bundle: EvaluationPredictionBundle,
    occluded_bundle: EvaluationPredictionBundle,
) -> Dict[str, Any]:
    """Compute valid-row probability-change diagnostics."""
    if len(baseline_bundle.probabilities) != len(occluded_bundle.probabilities):
        raise ValueError(
            "Baseline and occluded bundles have different lengths: "
            f"{len(baseline_bundle.probabilities)} vs {len(occluded_bundle.probabilities)}"
        )

    keep = baseline_bundle.valid_mask > 0.5

    if keep.sum() == 0:
        return {
            "mean_abs_probability_change": None,
            "mean_signed_probability_change": None,
            "max_abs_probability_change": None,
            "valid_rows": 0,
        }

    delta = occluded_bundle.probabilities[keep] - baseline_bundle.probabilities[keep]

    return {
        "mean_abs_probability_change": float(np.mean(np.abs(delta))),
        "mean_signed_probability_change": float(np.mean(delta)),
        "max_abs_probability_change": float(np.max(np.abs(delta))),
        "valid_rows": int(keep.sum()),
    }


def _index_map(context: Step14DiagnosticsContext) -> Dict[str, int]:
    """Map feature column name to index."""
    return {name: index for index, name in enumerate(context.feature_columns)}


def _make_scenario(
    context: Step14DiagnosticsContext,
    name: str,
    columns: Sequence[str],
    scenario_type: str,
    description: str,
) -> OcclusionScenario:
    """Build one occlusion scenario from column names."""
    index_map = _index_map(context)

    missing = [column for column in columns if column not in index_map]

    if missing:
        raise KeyError(f"Missing feature columns for occlusion scenario {name}: {missing}")

    return OcclusionScenario(
        name=name,
        columns=list(columns),
        indices=[int(index_map[column]) for column in columns],
        scenario_type=str(scenario_type),
        description=str(description),
    )


def build_occlusion_scenarios(context: Step14DiagnosticsContext) -> List[OcclusionScenario]:
    """
    Build diagnostic occlusion scenarios.

    Includes:
    - individual theoretical groups,
    - branch-level groups,
    - combined groups for module-level interpretation.
    """
    scenarios: List[OcclusionScenario] = []

    # Individual theoretical evidence components.
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_eta",
            columns=["xi_eta_east_scaled", "xi_eta_north_scaled"],
            scenario_type="individual_component",
            description="Neutralize instantaneous residual eta_t.",
        )
    )
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_eta_dot",
            columns=["xi_eta_dot_east_scaled", "xi_eta_dot_north_scaled"],
            scenario_type="individual_component",
            description="Neutralize first residual derivative eta_dot_t.",
        )
    )
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_eta_ddot",
            columns=["xi_eta_ddot_east_scaled", "xi_eta_ddot_north_scaled"],
            scenario_type="individual_component",
            description="Neutralize second residual derivative eta_ddot_t.",
        )
    )
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_q",
            columns=["xi_q_scaled"],
            scenario_type="individual_component",
            description="Neutralize Mahalanobis residual energy q_t.",
        )
    )
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_accum_log",
            columns=["xi_accum_log_scaled"],
            scenario_type="individual_component",
            description="Neutralize weak accumulated evidence feature.",
        )
    )
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_nu",
            columns=["xi_nu"],
            scenario_type="individual_component",
            description=(
                "Neutralize validity feature nu_t by setting it to neutral reference. "
                "Loss/evaluation masks are not changed."
            ),
        )
    )

    # Branch-level diagnostic groups.
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_instantaneous_branch",
            columns=["xi_eta_east_scaled", "xi_eta_north_scaled", "xi_q_scaled"],
            scenario_type="branch_level",
            description="Neutralize instantaneous residual branch eta_t + q_t.",
        )
    )
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_residual_evolution_branch",
            columns=[
                "xi_eta_dot_east_scaled",
                "xi_eta_dot_north_scaled",
                "xi_eta_ddot_east_scaled",
                "xi_eta_ddot_north_scaled",
            ],
            scenario_type="branch_level",
            description="Neutralize residual evolution branch eta_dot_t + eta_ddot_t.",
        )
    )
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_persistence_branch",
            columns=["xi_accum_log_scaled", "xi_nu"],
            scenario_type="branch_level",
            description="Neutralize persistence branch accum_log + nu.",
        )
    )

    # Combined module-level diagnostic groups.
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_all_residual_vector_features",
            columns=[
                "xi_eta_east_scaled",
                "xi_eta_north_scaled",
                "xi_eta_dot_east_scaled",
                "xi_eta_dot_north_scaled",
                "xi_eta_ddot_east_scaled",
                "xi_eta_ddot_north_scaled",
            ],
            scenario_type="combined_evidence",
            description="Neutralize eta_t, eta_dot_t, and eta_ddot_t together.",
        )
    )
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_all_residual_energy_features",
            columns=[
                "xi_eta_east_scaled",
                "xi_eta_north_scaled",
                "xi_q_scaled",
                "xi_accum_log_scaled",
            ],
            scenario_type="combined_evidence",
            description="Neutralize instantaneous residual evidence, q_t, and accumulated evidence.",
        )
    )
    scenarios.append(
        _make_scenario(
            context=context,
            name="mask_all_evidence_except_nu",
            columns=[
                "xi_eta_east_scaled",
                "xi_eta_north_scaled",
                "xi_eta_dot_east_scaled",
                "xi_eta_dot_north_scaled",
                "xi_eta_ddot_east_scaled",
                "xi_eta_ddot_north_scaled",
                "xi_q_scaled",
                "xi_accum_log_scaled",
            ],
            scenario_type="stress_test",
            description="Neutralize all evidence features except nu_t.",
        )
    )

    return scenarios


def compute_occlusion_result(
    context: Step14DiagnosticsContext,
    split_name: str,
    scenario: OcclusionScenario,
    baseline_bundle: EvaluationPredictionBundle,
    baseline_metrics: Mapping[str, Any],
    replacement_strategy: str = "neutral",
) -> OcclusionTestResult:
    """Run and summarize one occlusion scenario."""
    start_time = time.perf_counter()

    occluded_bundle = collect_predictions_with_optional_occlusion(
        context=context,
        split_name=split_name,
        group=scenario.to_feature_group(),
        replacement_strategy=replacement_strategy,
    )

    occluded_metrics = evaluate_bundle_with_threshold(
        bundle=occluded_bundle,
        theta=context.selected_threshold.theta,
        persistence=context.selected_threshold.persistence,
    )

    probability_change = _probability_change(
        baseline_bundle=baseline_bundle,
        occluded_bundle=occluded_bundle,
    )

    baseline_tp = _int_metric(baseline_metrics, "tp")
    baseline_fp = _int_metric(baseline_metrics, "fp")
    baseline_tn = _int_metric(baseline_metrics, "tn")
    baseline_fn = _int_metric(baseline_metrics, "fn")

    occluded_tp = _int_metric(occluded_metrics, "tp")
    occluded_fp = _int_metric(occluded_metrics, "fp")
    occluded_tn = _int_metric(occluded_metrics, "tn")
    occluded_fn = _int_metric(occluded_metrics, "fn")

    return OcclusionTestResult(
        split=split_name,
        scenario_name=scenario.name,
        scenario_type=scenario.scenario_type,
        columns=list(scenario.columns),
        indices=list(scenario.indices),
        replacement_strategy=replacement_strategy,
        baseline_auprc=_safe_float(baseline_metrics.get("auprc")),
        occluded_auprc=_safe_float(occluded_metrics.get("auprc")),
        delta_auprc=_delta(occluded_metrics.get("auprc"), baseline_metrics.get("auprc")),
        relative_delta_auprc=_relative_delta(
            occluded_metrics.get("auprc"),
            baseline_metrics.get("auprc"),
        ),
        baseline_f1=_safe_float(baseline_metrics.get("f1")),
        occluded_f1=_safe_float(occluded_metrics.get("f1")),
        delta_f1=_delta(occluded_metrics.get("f1"), baseline_metrics.get("f1")),
        relative_delta_f1=_relative_delta(
            occluded_metrics.get("f1"),
            baseline_metrics.get("f1"),
        ),
        baseline_precision=_safe_float(baseline_metrics.get("precision")),
        occluded_precision=_safe_float(occluded_metrics.get("precision")),
        delta_precision=_delta(
            occluded_metrics.get("precision"),
            baseline_metrics.get("precision"),
        ),
        baseline_recall=_safe_float(baseline_metrics.get("recall")),
        occluded_recall=_safe_float(occluded_metrics.get("recall")),
        delta_recall=_delta(
            occluded_metrics.get("recall"),
            baseline_metrics.get("recall"),
        ),
        baseline_fpr=_safe_float(baseline_metrics.get("fpr")),
        occluded_fpr=_safe_float(occluded_metrics.get("fpr")),
        delta_fpr=_delta(occluded_metrics.get("fpr"), baseline_metrics.get("fpr")),
        baseline_attack_detection_rate=_safe_float(
            baseline_metrics.get("attack_detection_rate")
        ),
        occluded_attack_detection_rate=_safe_float(
            occluded_metrics.get("attack_detection_rate")
        ),
        delta_attack_detection_rate=_delta(
            occluded_metrics.get("attack_detection_rate"),
            baseline_metrics.get("attack_detection_rate"),
        ),
        baseline_detection_delay=_safe_float(
            baseline_metrics.get("mean_detection_delay")
        ),
        occluded_detection_delay=_safe_float(
            occluded_metrics.get("mean_detection_delay")
        ),
        delta_detection_delay=_delta(
            occluded_metrics.get("mean_detection_delay"),
            baseline_metrics.get("mean_detection_delay"),
        ),
        baseline_tp=baseline_tp,
        baseline_fp=baseline_fp,
        baseline_tn=baseline_tn,
        baseline_fn=baseline_fn,
        occluded_tp=occluded_tp,
        occluded_fp=occluded_fp,
        occluded_tn=occluded_tn,
        occluded_fn=occluded_fn,
        delta_tp=int(occluded_tp - baseline_tp),
        delta_fp=int(occluded_fp - baseline_fp),
        delta_tn=int(occluded_tn - baseline_tn),
        delta_fn=int(occluded_fn - baseline_fn),
        mean_abs_probability_change=probability_change["mean_abs_probability_change"],
        mean_signed_probability_change=probability_change["mean_signed_probability_change"],
        max_abs_probability_change=probability_change["max_abs_probability_change"],
        valid_rows=int(probability_change["valid_rows"]),
        runtime_seconds=float(time.perf_counter() - start_time),
        diagnostic_only=True,
        description=scenario.description,
    )


def run_occlusion_tests_for_split(
    context: Step14DiagnosticsContext,
    split_name: str = "test",
    replacement_strategy: str = "neutral",
) -> List[OcclusionTestResult]:
    """Run diagnostic occlusion tests for one split."""
    scenarios = build_occlusion_scenarios(context)

    print("=" * 100)
    print(f"STEP 14 DIAGNOSTIC OCCLUSION TESTS | split={split_name}")
    print("=" * 100)
    print(f"Scenario count       : {len(scenarios)}")
    print(f"Replacement strategy : {replacement_strategy}")
    print(f"Selected theta/N_p   : {context.selected_threshold.theta} / {context.selected_threshold.persistence}")
    print("These are inference-time diagnostics only; official ablations retrain from scratch.")
    print("=" * 100)

    baseline_bundle = collect_predictions_with_optional_occlusion(
        context=context,
        split_name=split_name,
        group=None,
        replacement_strategy=replacement_strategy,
    )

    baseline_metrics = evaluate_bundle_with_threshold(
        bundle=baseline_bundle,
        theta=context.selected_threshold.theta,
        persistence=context.selected_threshold.persistence,
    )

    results: List[OcclusionTestResult] = []

    for scenario in scenarios:
        result = compute_occlusion_result(
            context=context,
            split_name=split_name,
            scenario=scenario,
            baseline_bundle=baseline_bundle,
            baseline_metrics=baseline_metrics,
            replacement_strategy=replacement_strategy,
        )
        results.append(result)

        print(
            f"{scenario.name:<38} | "
            f"ΔAUPRC={result.delta_auprc} | "
            f"ΔF1={result.delta_f1} | "
            f"ΔFPR={result.delta_fpr} | "
            f"ΔADR={result.delta_attack_detection_rate} | "
            f"ΔDelay={result.delta_detection_delay} | "
            f"mean|Δp|={result.mean_abs_probability_change}"
        )

    print("=" * 100)

    return results


def save_occlusion_test_results(
    context: Step14DiagnosticsContext,
    results: Sequence[OcclusionTestResult],
) -> Dict[str, str]:
    """Save occlusion-test results."""
    rows = [result.to_dict() for result in results]

    output_paths: Dict[str, str] = {}

    if context.diagnostics_config.save_csv:
        ensure_dir(context.paths.occlusion_csv.parent)
        pd.DataFrame(rows).to_csv(context.paths.occlusion_csv, index=False)
        output_paths["occlusion_csv"] = str(context.paths.occlusion_csv)

    if context.diagnostics_config.save_json:
        save_json_safe(
            {
                "context": context.to_dict(),
                "results": rows,
                "interpretation_note": (
                    "These occlusion tests are diagnostic inference-time masking tests only. "
                    "They do not replace official ablations, which must be retrained from scratch."
                ),
            },
            context.paths.occlusion_json,
        )
        output_paths["occlusion_json"] = str(context.paths.occlusion_json)

    return output_paths


def summarize_most_sensitive_occlusions(
    results: Sequence[OcclusionTestResult],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Return the most sensitive occlusions.

    Ranking:
    - most negative delta_f1 first,
    - then largest mean absolute probability change.
    """
    rows = [result.to_dict() for result in results]

    def sort_key(row: Mapping[str, Any]) -> tuple:
        delta_f1 = row.get("delta_f1")
        mean_abs_change = row.get("mean_abs_probability_change")

        delta_f1_value = float(delta_f1) if delta_f1 is not None else 0.0
        change_value = float(mean_abs_change) if mean_abs_change is not None else 0.0

        return (delta_f1_value, -change_value)

    rows = sorted(rows, key=sort_key)

    return rows[: int(top_k)]


def run_occlusion_tests(
    config: Mapping[str, Any],
    active_seed: int = 42,
    context: Optional[Step14DiagnosticsContext] = None,
) -> Dict[str, Any]:
    """Run Step-14 diagnostic occlusion tests for configured splits."""
    if context is None:
        context = load_step14_context(config=config, active_seed=active_seed)

    replacement_strategy = str(
        get_by_path(
            config,
            "experiments.step14.occlusion_tests.replacement_strategy",
            get_by_path(
                config,
                "experiments.step14.feature_importance.replacement_strategy",
                "neutral",
            ),
        )
    )

    all_results: List[OcclusionTestResult] = []

    for split_name in context.diagnostics_config.diagnostic_splits:
        split_results = run_occlusion_tests_for_split(
            context=context,
            split_name=str(split_name),
            replacement_strategy=replacement_strategy,
        )
        all_results.extend(split_results)

    artifact_paths = save_occlusion_test_results(
        context=context,
        results=all_results,
    )

    top_sensitive = summarize_most_sensitive_occlusions(
        results=all_results,
        top_k=int(
            get_by_path(
                config,
                "experiments.step14.occlusion_tests.top_k_console",
                10,
            )
        ),
    )

    print("=" * 100)
    print("STEP 14 MOST SENSITIVE DIAGNOSTIC OCCLUSIONS")
    print("=" * 100)
    for row in top_sensitive:
        print(
            f"{row['split']:<10} | {row['scenario_name']:<38} | "
            f"ΔF1={row['delta_f1']} | "
            f"ΔAUPRC={row['delta_auprc']} | "
            f"mean|Δp|={row['mean_abs_probability_change']}"
        )
    print("=" * 100)

    return {
        "status": "PASSED",
        "result_count": len(all_results),
        "diagnostic_splits": list(context.diagnostics_config.diagnostic_splits),
        "replacement_strategy": replacement_strategy,
        "artifact_paths": artifact_paths,
        "most_sensitive_occlusions": top_sensitive,
        "diagnostic_only": True,
    }


__all__ = [
    "OcclusionScenario",
    "OcclusionTestResult",
    "build_occlusion_scenarios",
    "compute_occlusion_result",
    "run_occlusion_tests_for_split",
    "save_occlusion_test_results",
    "summarize_most_sensitive_occlusions",
    "run_occlusion_tests",
]