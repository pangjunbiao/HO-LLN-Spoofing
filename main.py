"""
Main entry point for the AV-GPS causal spoofing detection project.

Implemented steps:

Step 1:
- load YAML configs,
- create standard project directories,
- select GPU/CPU device,
- set single-seed or multi-seed plan,
- initialize logging,
- verify sensitivity config is readable,
- write run logs and experiment history.

Step 2:
- check all four AV-GPS raw CSV files exist,
- load all raw datasets,
- validate basic schema,
- confirm labels and EKF Detector rule,
- inspect missing values, duplicate rows, shortcut columns, and core columns,
- save raw inspection JSON report.

Later steps will extend this main.py gradually.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from typing import Any, Dict, Mapping

from src.data.inspect_dataset import run_raw_dataset_inspection
from src.data.load_raw import load_and_validate_raw_datasets
from src.utils.config import (
    apply_overrides,
    get_by_path,
    load_project_config,
    print_config_summary,
)
from src.utils.device import assert_gpu_if_required, setup_device_from_config
from src.utils.io import ensure_standard_project_dirs, print_directory_summary
from src.utils.logging_utils import make_run_id, managed_run_logger
from src.utils.seed import print_seed_plan, resolve_seed_list, setup_seed_from_config
from src.data.segment_trajectories import run_trajectory_segmentation
from src.data.split_segments import run_dataset1_source_aware_split
from src.data.clean_columns import run_shortcut_column_exclusion
from src.preprocessing.motion_model import run_coordinate_motion_model_step
from src.preprocessing.normal_statistics import run_residual_and_normal_statistics_step
from src.preprocessing.evidence_builder import run_evidence_builder_step
from src.data.dataset_objects import run_dataset_objects_step
import numpy as np

from src.evaluation.threshold_selection import (
    evaluate_with_selected_threshold,
    print_threshold_selection_summary,
    save_threshold_selection_result,
    select_threshold_on_validation,
)
from pathlib import Path
from typing import Any, Mapping

from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import save_json
import torch

from src.models.model_factory import run_model_factory_sanity_check
from src.training.trainer import run_step12_training_protocol
from src.experiments.run_proposed import run_step13_proposed_experiment
from src.diagnostics.module_usage import (
    collect_module_activation_summaries,
    load_step14_context,
    save_json_safe,
    save_module_activation_summaries,
)
from src.diagnostics.feature_importance import run_feature_importance_diagnostics
from src.diagnostics.conductance_analysis import run_conductance_analysis
from src.diagnostics.third_order_analysis import run_third_order_analysis
from src.diagnostics.liquid_state_analysis import run_liquid_state_analysis
from src.diagnostics.occlusion_tests import run_occlusion_tests
from src.visualization.plot_module_usage import run_module_usage_plots
from src.experiments.run_baselines import run_step15_baselines_experiment
from src.experiments.run_ablations import run_step16_official_ablation_study
from src.experiments.run_frozen_ablations import (
    run_step16_frozen_intervention_ablation_study,
)
from src.experiments.run_high_order_comparison import (
    run_step17_high_order_comparison,
)
from src.experiments.run_step17_feature_model_analysis import (
    run_step17a_feature_group_intervention as run_step17a_feature_group_intervention_experiment,
    run_step17b_kirchhoff_structure_comparison as run_step17b_kirchhoff_structure_comparison_experiment,
    run_step17_feature_model_analysis as run_step17_feature_model_analysis_experiment,
)
from src.experiments.run_step19_dataset3_case_study import (
    run_step19_dataset3_case_study as run_step19_dataset3_case_study_impl,
)

from src.experiments.run_sensitivity import (
    run_step20_sensitivity_analysis as run_step20_sensitivity_analysis_impl,
)
from src.experiments.run_multiseed_proposed import run_multiseed_proposed

STEP21_MULTI_SEED_MODES = {
    "step21",
    "step21_multiseed",
    "step21_multi_seed",
    "step21_multiseed_robustness",
    "multiseed",
    "multi_seed",
    "proposed_multiseed",
    "multiseed_proposed",
    "seed_robustness",
    "training_robustness",
}
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AV-GPS causal spoofing detection project"
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing YAML config files.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help=(
            "Optional run mode override. Current supported modes: "
            "step1, setup, "
            "step2, inspect, raw_inspection, "
            "step3, segment, segmentation, trajectory_segmentation, "
            "step4, split, split_segments, source_aware_split, "
            "step5, clean, clean_columns, shortcut_exclusion, "
            "step6, physical, coordinate_motion, motion_model, "
            "step7, residual, residuals, normal_stats, residual_statistics, "
            "step8, evidence, xi, build_xi, evidence_builder, "
            "step9, dataset_objects, sequence_batching, batching, xi_dataset, "
            "step10, evaluation, metrics, threshold_selection, alarm_rules, "
            "step11, model, models, proposed_model, model_factory, "
            "step12, train, training, trainer, fit, "
            "step13, proposed, run_proposed, full_proposed, proposed_experiment, evaluate_proposed, "
            "step14, module_usage, module_diagnostics, full_model_diagnostics, "
            "diagnostics, feature_importance, occlusion_diagnostics, "
            "step15, baselines, baseline, run_baselines, train_baselines, "
            "evaluate_baselines, baseline_comparison, official_baselines, "
            "step16, ablations, ablation, official_ablations, official_ablation, "
            "controlled_ablations, controlled_ablation, ablation_study, "
            "run_ablations, train_ablations, evaluate_ablations, "
            "step16_frozen, frozen_ablations, frozen_ablation, "
            "frozen_intervention_ablations, run_frozen_ablations, "
            "evaluate_frozen_ablations."
            "step17, step17_high_order, high_order_comparison, "
            "feature_vs_model_high_order, model_vs_feature_high_order, "
            "run_high_order_comparison, professor_high_order_comparison."
            "step19, dataset3_case_study, dataset3_ekf, "
            "dataset3_ekf_case_study, online_case_study."
            "step20, sensitivity, sensitivity_analysis, "
            "operating_point_sensitivity, threshold_sensitivity."
            "step21, step21_multiseed, multiseed, multi_seed, "
            "proposed_multiseed, multiseed_proposed, "
            "seed_robustness, training_robustness."
        ),
    )

    parser.add_argument(
        "--seed-mode",
        type=str,
        default=None,
        choices=["single", "multi"],
        help="Optional seed mode override.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Optional device preference override.",
    )

    return parser.parse_args()


def build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert CLI arguments into config overrides."""
    overrides: Dict[str, Any] = {}

    if args.mode is not None:
        overrides["run.mode"] = args.mode

    if args.seed_mode is not None:
        overrides["seed.mode"] = args.seed_mode

    if args.device is not None:
        overrides["device.preference"] = args.device

    return overrides


def print_sensitivity_plan(config: Mapping[str, Any]) -> None:
    """
    Print sensitivity settings from config.

    Step 1/2 only verifies that sensitivity control exists.
    Actual sensitivity experiments are implemented later.
    """
    enabled = bool(get_by_path(config, "sensitivity.enabled", False))

    print("=" * 100)
    print("SENSITIVITY PLAN")
    print("=" * 100)
    print(f"Sensitivity enabled: {enabled}")

    groups = ["theta", "persistence", "rho", "hidden_dim"]

    for group in groups:
        group_enabled = bool(
            get_by_path(config, f"sensitivity.{group}.enabled", False)
        )
        default_value = get_by_path(config, f"sensitivity.{group}.default", "N/A")
        values = get_by_path(config, f"sensitivity.{group}.values", [])

        print(
            f"{group:12s} | enabled={group_enabled} | "
            f"default={default_value} | values={values}"
        )

    print("=" * 100)


def prepare_common_runtime(
    config: Mapping[str, Any],
    active_seed: int,
) -> Any:
    """
    Common runtime setup used by every step.

    This performs:
    - directory creation,
    - seed setup,
    - GPU/CPU setup.

    Returns:
        DeviceInfo object.
    """
    project_root = Path(get_by_path(config, "project.root", ".")).resolve()

    ensure_standard_project_dirs(project_root)
    print_directory_summary(project_root)

    setup_seed_from_config(config, seed=active_seed, verbose=True)

    assert_gpu_if_required(config)
    device_info = setup_device_from_config(config, verbose=True)

    return device_info


def run_step1_check(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step-1-only system check.

    Confirms:
    - config loaded,
    - directories exist,
    - seed is set,
    - device is selected,
    - logging works,
    - sensitivity config is readable.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step1"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step1_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 1 system check is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        logger.info("Step 1 completed successfully.")
        logger.info("Logs were written to the logs directory.")


def run_step2_raw_inspection(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 2: raw dataset loading and inspection.

    This step:
    - checks all expected raw CSV files,
    - loads all four AV-GPS files,
    - validates labels and EKF Detector rule,
    - inspects schema, labels, missing values, duplicates, shortcut columns,
    - saves inspection JSON to results/tables/.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step2"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step2_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 2 raw dataset loading and inspection is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        bundle = load_and_validate_raw_datasets(config=config)
        logger.info("Raw datasets loaded successfully.")

        for key, info in bundle.summary().items():
            logger.info(
                "Loaded %s | rows=%s | columns=%s | role=%s | path=%s",
                key,
                info["rows"],
                info["columns"],
                info["role"],
                info["path"],
            )

        report = run_raw_dataset_inspection(
            bundle=bundle,
            config=config,
            save_report=True,
        )

        logger.info("Step 2 inspection status: %s", report.final_step2_status)

        if report.final_step2_status != "PASSED":
            logger.warning(
                "Step 2 completed with warnings. Check printed report and JSON output."
            )

        logger.info("Step 2 completed successfully.")

def run_step3_trajectory_segmentation(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 3: trajectory segmentation.

    This step:
    - loads all raw AV-GPS files,
    - parses time columns,
    - detects resets, date/session changes, and large gaps,
    - creates segment_id and within_segment_index,
    - preserves Dataset-3 as one online scenario sequence,
    - marks invalid first rows and segment-boundary transitions,
    - saves segmented CSV files to data/interim/.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step3"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step3_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 3 trajectory segmentation is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        report = run_trajectory_segmentation(
            config=config,
            bundle=None,
            save_outputs=True,
        )

        logger.info("Step 3 segmentation status: %s", report.final_step3_status)

        for key, summary in report.dataset_summaries.items():
            logger.info(
                "Segmented %s | rows=%s | segments=%s | min_len=%s | "
                "median_len=%s | max_len=%s | output=%s",
                key,
                summary.rows,
                summary.segment_count,
                summary.min_segment_length,
                summary.median_segment_length,
                summary.max_segment_length,
                summary.output_path,
            )

        if report.final_step3_status != "PASSED":
            logger.warning(
                "Step 3 completed with warnings. Check segmentation summary JSON."
            )

        logger.info("Step 3 completed successfully.")

def run_step4_source_aware_split(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 4: source-aware Dataset-1 segment split.

    This step:
    - uses Dataset-1 only for main development,
    - splits by segment_id, not rows,
    - prevents same-segment leakage,
    - saves train/val/test segment IDs,
    - keeps Dataset-2 and Dataset-3 untouched.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step4"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step4_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 4 source-aware Dataset-1 split is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        report = run_dataset1_source_aware_split(
            config=config,
            save_outputs=True,
        )

        logger.info("Step 4 split status: %s", report.final_step4_status)

        for split_name, split_file in report.split_files.items():
            logger.info(
                "Split %s | segments=%s | rows=%s | normal=%s | attack=%s | attack_rate=%s | output=%s",
                split_name,
                split_file.segment_count,
                split_file.row_count,
                split_file.normal_count,
                split_file.attack_count,
                split_file.attack_rate,
                split_file.output_path,
            )

        logger.info("Leakage check passed: %s", report.leakage_check["passed"])
        logger.info("Dataset-2 untouched: %s", report.external_datasets_untouched["dataset2_untouched_for_external_source_shift_test"])
        logger.info("Dataset-3 untouched: %s", report.external_datasets_untouched["dataset3_untouched_for_online_case_study"])

        if report.final_step4_status != "PASSED":
            logger.warning(
                "Step 4 completed with warnings. Check data/splits/split_summary.json."
            )

        logger.info("Step 4 completed successfully.")

def run_step5_clean_columns(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 5: shortcut-column exclusion and clean causal column selection.

    This step:
    - loads segmented intermediate files,
    - keeps only causal source columns needed for evidence construction,
    - removes shortcut-prone raw columns,
    - does not create final model inputs yet,
    - saves cleaned intermediate files to data/interim/.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step5"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step5_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 5 shortcut-column exclusion is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        report = run_shortcut_column_exclusion(
            config=config,
            dataset_keys=None,
            save_outputs=True,
        )

        logger.info("Step 5 clean-column status: %s", report.final_step5_status)

        for key, summary in report.dataset_summaries.items():
            logger.info(
                "Cleaned %s | rows=%s -> %s | columns=%s -> %s | "
                "forbidden_remaining=%s | output=%s",
                key,
                summary.input_rows,
                summary.output_rows,
                summary.input_columns,
                summary.output_columns,
                summary.forbidden_columns_remaining,
                summary.output_path,
            )

        if report.final_step5_status != "PASSED":
            logger.warning(
                "Step 5 completed with warnings. Check step5_clean_columns_summary.json."
            )

        logger.info("Step 5 completed successfully.")

def run_step6_coordinate_motion_model(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 6: coordinate transform and causal motion model.

    This step:
    - loads cleaned intermediate files from Step 5,
    - computes local GNSS east/north coordinates,
    - computes causal GNSS displacement Delta p_g,
    - computes causal onboard-motion displacement Delta p_u,
    - saves physical intermediate files to data/interim/.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step6"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step6_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 6 coordinate transform and motion model is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        report = run_coordinate_motion_model_step(
            config=config,
            dataset_keys=None,
            save_outputs=True,
        )

        logger.info("Step 6 physical-model status: %s", report.final_step6_status)

        for key, summary in report.dataset_summaries.items():
            coordinate = summary.coordinate_summary
            motion = summary.motion_summary

            logger.info(
                "Physical %s | rows=%s -> %s | columns=%s -> %s | "
                "valid_gnss_disp=%s | valid_motion_disp=%s | output=%s",
                key,
                summary.input_rows,
                summary.output_rows,
                summary.input_columns,
                summary.output_columns,
                coordinate["valid_gnss_displacement_rows"],
                motion["valid_motion_displacement_rows"],
                summary.output_path,
            )

        if report.final_step6_status != "PASSED":
            logger.warning(
                "Step 6 completed with warnings. Check step6_coordinate_motion_summary.json."
            )

        logger.info("Step 6 completed successfully.")


def run_step7_residual_and_normal_statistics(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 7: residual construction and training-only normal statistics.

    This step:
    - loads Step-6 physical intermediate files,
    - computes residual r_t = Delta p_g_t - Delta p_u_t,
    - builds final residual validity nu_t,
    - saves residual files for all datasets,
    - computes Sigma_r^tr and mu_e from Dataset-1 TRAIN NORMAL valid residuals only,
    - saves residual/statistics reports for inspection.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step7"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step7_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 7 residual and normal-statistics construction is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        report = run_residual_and_normal_statistics_step(
            config=config,
            dataset_keys=None,
            save_outputs=True,
        )

        logger.info("Step 7 final status: %s", report.final_step7_status)

        residual_report = report.residual_builder_report
        dataset_summaries = residual_report.get("dataset_summaries", {})

        for key, summary in dataset_summaries.items():
            logger.info(
                "Residual %s | rows=%s -> %s | valid_nu=%s | invalid_nu=%s | "
                "valid_normal=%s | valid_attack=%s | output=%s",
                key,
                summary.get("input_rows"),
                summary.get("output_rows"),
                summary.get("valid_residual_rows"),
                summary.get("invalid_residual_rows"),
                summary.get("valid_normal_residual_rows"),
                summary.get("valid_attack_residual_rows"),
                summary.get("output_path"),
            )

        normal_summary = report.normal_statistics_summary

        logger.info(
            "Normal stats | dataset=%s | split=%s | train_segments=%s | "
            "train_normal_valid=%s | mu_e=%s | stats_path=%s",
            normal_summary.get("dataset_key"),
            normal_summary.get("split_name"),
            normal_summary.get("train_segment_count"),
            normal_summary.get("train_normal_valid_count"),
            normal_summary.get("train_normal_energy_median_mu_e"),
            normal_summary.get("output_statistics_path"),
        )

        logger.info("Leakage rule: %s", normal_summary.get("leakage_rule"))

        if report.final_step7_status != "PASSED":
            logger.warning(
                "Step 7 completed with warnings. Check Step-7 JSON outputs."
            )

        logger.info("Step 7 completed successfully.")

def run_step8_evidence_builder(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 8: build final causal xi evidence dataset.

    This step:
    - loads Step-7 residual files,
    - validates Step-7 training-only normal statistics,
    - computes eta_t, dot_eta_t, ddot_eta_t, q_t, log accumulation, and xi_nu,
    - saves dataset-level xi files,
    - saves train/val/test/external/online xi files,
    - fits xi scaler on Dataset-1 train only,
    - applies the same scaler to all xi outputs,
    - saves diagnostics for energy distribution and feature specification.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step8"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step8_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 8 causal xi evidence builder is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        report = run_evidence_builder_step(
            config=config,
            dataset_keys=None,
            save_outputs=True,
        )

        logger.info("Step 8 final status: %s", report.final_step8_status)

        for key, summary in report.dataset_summaries.items():
            logger.info(
                "Xi dataset %s | rows=%s -> %s | valid_nu=%s | invalid_nu=%s | "
                "dot_valid=%s | ddot_valid=%s | output=%s",
                key,
                summary.input_rows,
                summary.output_rows,
                summary.valid_nu_rows,
                summary.invalid_nu_rows,
                summary.dot_valid_rows,
                summary.ddot_valid_rows,
                summary.output_path,
            )

        for key, summary in report.split_summaries.items():
            logger.info(
                "Xi split %s | source=%s | rows=%s | segments=%s | "
                "normal=%s | attack=%s | valid_nu=%s | output=%s",
                key,
                summary.source_dataset,
                summary.rows,
                summary.segments,
                summary.normal_rows,
                summary.attack_rows,
                summary.valid_nu_rows,
                summary.output_path,
            )

        logger.info("Normal statistics source: %s", report.normal_statistics_source)
        logger.info("Raw xi feature columns: %s", report.raw_xi_feature_columns)
        logger.info("Scaled xi feature columns: %s", report.scaled_xi_feature_columns)
        logger.info("Scaler summary: %s", report.scaler_summary)

        if report.final_step8_status != "PASSED":
            logger.warning(
                "Step 8 completed with warnings. Check Step-8 JSON outputs."
            )

        logger.info("Step 8 completed successfully.")

def run_step9_dataset_objects(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 9: dataset objects and sequence batching.

    This step:
    - loads Step-8 train/val/test/external/online xi files,
    - validates the official scaled xi feature columns,
    - groups rows by segment,
    - builds train/eval sequence windows,
    - preserves Dataset-3 online order,
    - creates flattened valid-row arrays for XGBoost/MLP,
    - saves Step-9 summary and sequence manifest.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step9"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step9_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 9 dataset objects and sequence batching is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        report = run_dataset_objects_step(
            config=config,
            split_names=None,
            save_outputs=True,
        )

        logger.info("Step 9 final status: %s", report.final_step9_status)
        logger.info("Step 9 feature columns: %s", report.feature_columns)
        logger.info("Step 9 fairness rules: %s", report.fairness_rules)

        for split_name, summary in report.split_summaries.items():
            logger.info(
                "Step 9 split %s | rows=%s | segments=%s | feature_dim=%s | "
                "normal=%s | attack=%s | valid=%s | invalid=%s | "
                "train_windows=%s | eval_windows=%s | flat_valid_rows=%s | status=%s",
                split_name,
                summary.rows,
                summary.segments,
                summary.feature_dim,
                summary.normal_rows,
                summary.attack_rows,
                summary.valid_rows,
                summary.invalid_rows,
                summary.sequence_windows_train_mode,
                summary.sequence_windows_eval_mode,
                summary.flattened_rows_valid_only,
                summary.final_status,
            )

        logger.info("Saved outputs: %s", report.saved_outputs)

        if report.final_step9_status != "PASSED":
            logger.warning(
                "Step 9 completed with warnings. Check Step-9 JSON outputs."
            )

        logger.info("Step 9 completed successfully.")

def run_step10_evaluation_framework(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 10: evaluation metrics, threshold selection, and alarm-rule sanity check.

    This step does not select the final model threshold yet.
    Final theta and N_p will be selected later using each trained model's
    validation predictions only.

    This step verifies:
    - AUPRC/AUROC/F1/FPR metric code runs,
    - c_t = I(p_hat_t >= theta) works,
    - confirmed alarm after N_p consecutive positives works,
    - event-level Attack Detection Rate and Detection Delay work,
    - validation-only threshold-selection guard works.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step10"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step10_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 10 evaluation framework sanity check is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        # ------------------------------------------------------------------
        # Synthetic validation-like sequence.
        # This is only a framework sanity check, not a real model result.
        # ------------------------------------------------------------------
        y_true = np.array(
            [
                # segment 1: one attack event from positions 5 to 8
                0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                # segment 2: all-normal segment for normal-segment FAR sanity
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            dtype=np.int64,
        )

        y_score = np.array(
            [
                0.02, 0.05, 0.04, 0.06, 0.10, 0.55, 0.72, 0.81, 0.77, 0.20, 0.08, 0.05,
                0.03, 0.04, 0.08, 0.11, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
            ],
            dtype=np.float64,
        )

        segment_id = np.array(
            ["sanity_attack_segment"] * 12 + ["sanity_normal_segment"] * 10,
            dtype=object,
        )

        order_index = np.array(
            list(range(12)) + list(range(10)),
            dtype=np.int64,
        )

        delta_t = np.ones_like(y_score, dtype=np.float64)
        valid_mask = np.ones_like(y_true, dtype=np.int64)

        # ------------------------------------------------------------------
        # Validation-only selection should pass for split='val'.
        # ------------------------------------------------------------------
        selection_result = select_threshold_on_validation(
            y_true=y_true,
            y_score=y_score,
            segment_id=segment_id,
            order_index=order_index,
            delta_t=delta_t,
            valid_mask=valid_mask,
            config=config,
            selection_split="val",
            runtime_seconds=0.0,
        )

        report_path = save_threshold_selection_result(
            result=selection_result,
            config=config,
            model_name="step10_sanity_check",
        )

        print_threshold_selection_summary(selection_result)

        logger.info("Step 10 sanity threshold report saved to: %s", report_path)
        logger.info(
            "Step 10 selected theta=%s, N_p=%s",
            selection_result.selected_threshold,
            selection_result.selected_persistence,
        )

        # ------------------------------------------------------------------
        # Evaluate using the already-selected threshold.
        # This mirrors later test/external/online use.
        # ------------------------------------------------------------------
        selected_eval_payload = evaluate_with_selected_threshold(
            y_true=y_true,
            y_score=y_score,
            segment_id=segment_id,
            order_index=order_index,
            delta_t=delta_t,
            valid_mask=valid_mask,
            selected_threshold=selection_result.selected_threshold,
            selected_persistence=selection_result.selected_persistence,
            runtime_seconds=0.0,
        )

        # ------------------------------------------------------------------
        # Validation-only guard should reject non-validation selection.
        # ------------------------------------------------------------------
        validation_only_guard_passed = False
        validation_only_guard_error = ""

        try:
            _ = select_threshold_on_validation(
                y_true=y_true,
                y_score=y_score,
                segment_id=segment_id,
                order_index=order_index,
                delta_t=delta_t,
                valid_mask=valid_mask,
                config=config,
                selection_split="test",
                runtime_seconds=0.0,
            )
        except RuntimeError as exc:
            validation_only_guard_passed = True
            validation_only_guard_error = str(exc)

        final_status = (
            "PASSED"
            if selection_result.final_status == "PASSED"
            and validation_only_guard_passed
            else "FAILED"
        )

        summary = {
            "step": "step10",
            "purpose": "evaluation metrics, threshold selection, and alarm-rule sanity check",
            "note": (
                "This is a synthetic framework sanity check only. "
                "Final model thresholds will be selected later using real validation predictions."
            ),
            "selected_threshold": selection_result.selected_threshold,
            "selected_persistence": selection_result.selected_persistence,
            "selection_report_path": str(report_path),
            "selected_candidate": selection_result.selected_candidate,
            "selected_eval_primary_metrics": selected_eval_payload.get("primary_metrics", {}),
            "selected_eval_secondary_metrics": selected_eval_payload.get("secondary_metrics", {}),
            "validation_only_guard_passed": validation_only_guard_passed,
            "validation_only_guard_error": validation_only_guard_error,
            "locked_primary_metrics": [
                "AUPRC",
                "F1",
                "FPR",
                "Attack Detection Rate",
                "Detection Delay",
            ],
            "secondary_metrics": [
                "AUROC",
                "Precision",
                "Recall",
                "Runtime",
                "Normal-Segment FAR",
            ],
            "final_step10_status": final_status,
        }

        summary_path_value = get_by_path(
            config,
            "paths.step10_evaluation_framework_summary_json",
            "results/tables/step10_evaluation_framework_summary.json",
        )
        summary_path = resolve_project_path(config, summary_path_value)

        save_json(summary, summary_path, indent=2)

        print("=" * 100)
        print("STEP 10 EVALUATION FRAMEWORK SUMMARY")
        print("=" * 100)
        print(f"Selected theta                      : {selection_result.selected_threshold}")
        print(f"Selected persistence N_p            : {selection_result.selected_persistence}")
        print(f"Validation-only guard passed         : {validation_only_guard_passed}")
        print(f"Threshold report saved               : {report_path}")
        print(f"Step 10 summary saved                : {summary_path}")
        print(f"Final Step 10 status                 : {final_status}")
        print("=" * 100)

        logger.info("Step 10 summary saved to: %s", summary_path)
        logger.info("Step 10 validation-only guard passed: %s", validation_only_guard_passed)
        logger.info("Step 10 final status: %s", final_status)

        if final_status != "PASSED":
            raise RuntimeError(f"Step 10 failed with status: {final_status}")

        logger.info("Step 10 completed successfully.")

def run_step11_model_factory(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 11: proposed model modules and model factory sanity check.

    This step:
    - builds the full proposed model,
    - builds all official ablation architectures,
    - builds professor high-order comparison variants,
    - checks the locked 9-column scaled xi input contract,
    - runs synthetic forward sanity checks,
    - saves architecture summaries.

    This step does not train models.
    Step 12 will train the full proposed model.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step11"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step11_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 11 proposed model modules/model factory sanity check is running.")
        logger.info("Project root: %s", project_root)

        print_config_summary(config)
        print_sensitivity_plan(config)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        logger.info("Step 11 model sanity-check device: %s", device)

        include_ablations = bool(
            get_by_path(
                config,
                "experiments.step11_model_factory.build_official_ablations",
                True,
            )
        )
        include_high_order = bool(
            get_by_path(
                config,
                "experiments.step11_model_factory.build_high_order_comparison",
                True,
            )
        )
        save_outputs = bool(
            get_by_path(
                config,
                "experiments.step11_model_factory.save_architecture_summaries",
                True,
            )
        )

        report = run_model_factory_sanity_check(
            config=config,
            device=device,
            include_ablations=include_ablations,
            include_high_order_comparison=include_high_order,
            save_outputs=save_outputs,
        )

        logger.info("Step 11 final status: %s", report.final_step11_status)
        logger.info("Step 11 fairness rules: %s", report.fairness_rules)
        logger.info("Step 11 saved outputs: %s", report.saved_outputs)

        for variant_name, check in report.forward_sanity_checks.items():
            logger.info(
                "Step 11 forward check | variant=%s | status=%s | prob_shape=%s | "
                "prob_min=%s | prob_max=%s | finite=%s",
                variant_name,
                check.get("status"),
                check.get("actual_probability_shape"),
                check.get("probability_min"),
                check.get("probability_max"),
                check.get("finite_probabilities"),
            )

        if report.final_step11_status != "PASSED":
            raise RuntimeError(
                f"Step 11 failed with status: {report.final_step11_status}"
            )

        logger.info("Step 11 completed successfully.")

def run_step12_training_loop(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 12: common PyTorch training loop.

    This step:
    - loads train/validation xi splits,
    - builds sequence windows,
    - builds the selected PyTorch model variant,
    - computes class weights from TRAIN only,
    - trains with weighted BCE,
    - monitors validation loss,
    - applies early stopping,
    - saves best checkpoint,
    - saves validation probabilities for later threshold selection.

    This step does not select final theta or N_p.
    The synthetic Step-10 theta=0.55 is not used here.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step12"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step12_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 12 training loop is running.")
        logger.info("Project root: %s", project_root)
        logger.info("Active seed: %s", active_seed)
        logger.info(
            "Important: Step 12 does not use synthetic Step-10 theta=0.55 as final threshold."
        )

        print_config_summary(config)
        print_sensitivity_plan(config)

        summary = run_step12_training_protocol(
            config=config,
            active_seed=active_seed,
        )

        logger.info("Step 12 final status: %s", summary.final_status)
        logger.info("Step 12 best epoch: %s", summary.best_epoch)
        logger.info("Step 12 best monitor value: %s", summary.best_monitor_value)
        logger.info("Step 12 best checkpoint: %s", summary.best_checkpoint_path)
        logger.info("Step 12 last checkpoint: %s", summary.last_checkpoint_path)

        if summary.final_status != "PASSED":
            raise RuntimeError(f"Step 12 failed with status: {summary.final_status}")

        logger.info("Step 12 completed successfully.")


def run_step13_full_proposed_experiment(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 13: train/reuse and evaluate full proposed model.

    This step:
    - reuses or trains results/models/proposed_best.pt,
    - selects real theta and N_p on Dataset-1 validation only,
    - evaluates Dataset-1 internal test,
    - evaluates Dataset-2 external/source-shift test,
    - evaluates Dataset-3 online case study,
    - saves comparison tables and summaries.

    This step does not compare baselines or ablations yet.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step13"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step13_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 13 full proposed experiment is running.")
        logger.info("Project root: %s", project_root)
        logger.info("Active seed: %s", active_seed)
        logger.info(
            "Important: Step 13 selects theta and N_p on Dataset-1 validation only."
        )
        logger.info(
            "Important: synthetic Step-10 theta=0.55 is not used as final threshold."
        )

        print_config_summary(config)
        print_sensitivity_plan(config)

        summary = run_step13_proposed_experiment(
            config=config,
            active_seed=active_seed,
        )

        logger.info("Step 13 final status: %s", summary.final_status)
        logger.info("Step 13 checkpoint: %s", summary.checkpoint_path)
        logger.info("Step 13 selected theta: %s", summary.selected_threshold)
        logger.info("Step 13 selected persistence: %s", summary.selected_persistence)
        logger.info("Step 13 summary JSON: %s", summary.output_paths.get("step13_summary_json"))

        if summary.final_status != "PASSED":
            raise RuntimeError(f"Step 13 failed with status: {summary.final_status}")

        logger.info("Step 13 completed successfully.")

def run_step14_full_model_module_usage_diagnostics(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 14: full-model module-usage diagnostics.

    This step diagnoses whether the trained full proposed model actually uses:
    - eta, eta_dot, eta_ddot, q, accumulated evidence, and nu,
    - Kirchhoff/conductance/exchange modules,
    - third-order/high-order fusion,
    - liquid second-order temporal states,
    - diagnostic occlusion sensitivity.

    Important:
    - This step does not retrain the model.
    - This step does not select a new threshold.
    - This step uses the Step-13 validation-selected theta and persistence.
    - Occlusion is diagnostic only; official ablations still retrain from scratch.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step14"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step14_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 14 full-model module-usage diagnostics are running.")
        logger.info("Project root: %s", project_root)
        logger.info("Active seed: %s", active_seed)
        logger.info("Important: Step 14 uses Step-13 validation-selected theta/N_p.")
        logger.info("Important: Step 14 occlusion is diagnostic only.")
        logger.info("Important: official ablations must still retrain from scratch.")

        print_config_summary(config)
        print_sensitivity_plan(config)

        print("=" * 100)
        print("STEP 14 FULL-MODEL MODULE-USAGE DIAGNOSTICS START")
        print("=" * 100)
        print(f"Project root     : {project_root}")
        print(f"Active seed      : {active_seed}")
        print("Uses Step-13 selected theta and persistence.")
        print("Does not retrain. Does not select a new threshold.")
        print("Occlusion is diagnostic only; official ablation comes later.")
        print("=" * 100)

        context = load_step14_context(
            config=config,
            active_seed=active_seed,
        )

        logger.info("Loaded Step 14 context.")
        logger.info("Checkpoint: %s", context.checkpoint_path)
        logger.info("Selected theta: %s", context.selected_threshold.theta)
        logger.info("Selected persistence: %s", context.selected_threshold.persistence)
        logger.info("Diagnostic splits: %s", context.diagnostics_config.diagnostic_splits)

        # ------------------------------------------------------------------
        # 1. Module activation usage diagnostics
        # ------------------------------------------------------------------
        all_module_summaries = []

        for split_name in context.diagnostics_config.diagnostic_splits:
            split_summaries = collect_module_activation_summaries(
                context=context,
                split_name=str(split_name),
            )
            all_module_summaries.extend(split_summaries)

        module_artifact_paths = save_module_activation_summaries(
            context=context,
            summaries=all_module_summaries,
        )

        module_usage_result = {
            "status": "PASSED",
            "result_count": len(all_module_summaries),
            "diagnostic_splits": list(context.diagnostics_config.diagnostic_splits),
            "artifact_paths": module_artifact_paths,
        }

        # ------------------------------------------------------------------
        # 2. Feature importance / simple diagnostic masking
        # ------------------------------------------------------------------
        feature_importance_result = run_feature_importance_diagnostics(
            config=config,
            active_seed=active_seed,
            context=context,
        )

        # ------------------------------------------------------------------
        # 3. Kirchhoff/conductance/exchange diagnostics
        # ------------------------------------------------------------------
        conductance_result = run_conductance_analysis(
            config=config,
            active_seed=active_seed,
            context=context,
        )

        # ------------------------------------------------------------------
        # 4. Third-order/high-order diagnostics
        # ------------------------------------------------------------------
        third_order_result = run_third_order_analysis(
            config=config,
            active_seed=active_seed,
            context=context,
        )

        # ------------------------------------------------------------------
        # 5. Liquid/temporal-state diagnostics
        # ------------------------------------------------------------------
        liquid_state_result = run_liquid_state_analysis(
            config=config,
            active_seed=active_seed,
            context=context,
        )

        # ------------------------------------------------------------------
        # 6. Broader diagnostic occlusion tests
        # ------------------------------------------------------------------
        occlusion_result = run_occlusion_tests(
            config=config,
            active_seed=active_seed,
            context=context,
        )

        # ------------------------------------------------------------------
        # 7. Plots
        # ------------------------------------------------------------------
        plot_result = run_module_usage_plots(
            config=config,
            active_seed=active_seed,
        )

        final_summary = {
            "final_status": "PASSED",
            "active_seed": int(active_seed),
            "checkpoint_path": str(context.checkpoint_path),
            "selected_threshold": context.selected_threshold.theta,
            "selected_persistence": context.selected_threshold.persistence,
            "diagnostic_splits": list(context.diagnostics_config.diagnostic_splits),
            "context": context.to_dict(),
            "module_usage_result": module_usage_result,
            "feature_importance_result": feature_importance_result,
            "conductance_result": conductance_result,
            "third_order_result": third_order_result,
            "liquid_state_result": liquid_state_result,
            "occlusion_result": occlusion_result,
            "plot_result": plot_result,
            "important_interpretation_rules": {
                "diagnostic_only": True,
                "does_not_retrain_model": True,
                "does_not_select_new_threshold": True,
                "uses_step13_validation_selected_threshold": True,
                "official_ablation_must_retrain_from_scratch": True,
                "dataset2_row_level_strong_but_event_level_weaker_should_be_reported_later": True,
                "persistence_5_was_grid_maximum_should_be_checked_before_baselines": True,
            },
            "output_directory": str(context.paths.output_dir),
            "summary_json": str(context.paths.summary_json),
        }

        save_json_safe(
            payload=final_summary,
            output_path=context.paths.summary_json,
            indent=2,
        )

        logger.info("Step 14 final status: PASSED")
        logger.info("Step 14 output directory: %s", context.paths.output_dir)
        logger.info("Step 14 summary JSON: %s", context.paths.summary_json)

        print("=" * 100)
        print("STEP 14 FULL-MODEL MODULE-USAGE DIAGNOSTICS SUMMARY")
        print("=" * 100)
        print("Final status       : PASSED")
        print(f"Checkpoint         : {context.checkpoint_path}")
        print(f"Selected theta     : {context.selected_threshold.theta}")
        print(f"Selected N_p       : {context.selected_threshold.persistence}")
        print(f"Diagnostic splits  : {context.diagnostics_config.diagnostic_splits}")
        print(f"Output directory   : {context.paths.output_dir}")
        print(f"Summary JSON       : {context.paths.summary_json}")
        print("Main result files:")
        print(f"  module usage      : {context.paths.module_usage_csv}")
        print(f"  feature importance: {context.paths.feature_importance_csv}")
        print(f"  conductance       : {context.paths.conductance_csv}")
        print(f"  third order       : {context.paths.third_order_csv}")
        print(f"  liquid state      : {context.paths.liquid_state_csv}")
        print(f"  occlusion tests   : {context.paths.occlusion_csv}")
        print(f"  plots directory   : {context.paths.output_dir / 'plots'}")
        print("=" * 100)

        logger.info("Step 14 completed successfully.")

def run_step15_official_baseline_comparison(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 15: official fair baseline comparison.

    This step:
    - trains or reuses XGBoost-xi, MLP-xi, LSTM-xi, GRU-xi, and Causal-TCN-xi,
    - uses Dataset-1 train only for fitting,
    - selects theta and persistence on Dataset-1 validation only,
    - evaluates Dataset-1 test, Dataset-2 external, and Dataset-3 online,
    - uses the same reconstructed xi_t features as the proposed model,
    - never uses raw shortcut columns,
    - never uses Dataset-2 or Dataset-3 for tuning.

    This step does not run official ablations. Ablations are Step 16.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step15"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step15_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 15 official baseline comparison is running.")
        logger.info("Project root: %s", project_root)
        logger.info("Active seed: %s", active_seed)
        logger.info("Important: baselines use same reconstructed xi_t features.")
        logger.info("Important: no raw shortcut columns are used.")
        logger.info("Important: threshold selection uses Dataset-1 validation only.")
        logger.info("Important: Dataset-2 and Dataset-3 are not tuned.")
        logger.info("Important: Step 15 does not run official ablations.")

        print_config_summary(config)
        print_sensitivity_plan(config)

        print("=" * 120)
        print("STEP 15 OFFICIAL BASELINE COMPARISON START")
        print("=" * 120)
        print(f"Project root : {project_root}")
        print(f"Active seed  : {active_seed}")
        print("Baselines    : XGBoost-xi, MLP-xi, LSTM-xi, GRU-xi, Causal-TCN-xi")
        print("Features     : same reconstructed xi_t features as proposed model")
        print("Training     : Dataset-1 train only")
        print("Threshold    : Dataset-1 validation only")
        print("Testing      : Dataset-1 test, Dataset-2 external, Dataset-3 online")
        print("No shortcut  : no raw GPS/time/date/EKF Detector input")
        print("No ablation  : official ablations are Step 16")
        print("=" * 120)

        summary = run_step15_baselines_experiment(
            config=config,
            active_seed=active_seed,
        )

        logger.info("Step 15 final status: %s", summary.final_status)
        logger.info("Step 15 enabled baselines: %s", summary.enabled_baselines)
        logger.info("Step 15 summary JSON: %s", summary.output_paths.get("summary_json"))
        logger.info("Step 15 comparison CSV: %s", summary.output_paths.get("all_baselines_comparison_csv"))

        if summary.final_status != "PASSED":
            raise RuntimeError(f"Step 15 failed with status: {summary.final_status}")

        print("=" * 120)
        print("STEP 15 OFFICIAL BASELINE COMPARISON COMPLETED")
        print("=" * 120)
        print(f"Final status      : {summary.final_status}")
        print(f"Enabled baselines : {summary.enabled_baselines}")
        print(f"Summary JSON      : {summary.output_paths.get('summary_json')}")
        print(f"Comparison CSV    : {summary.output_paths.get('all_baselines_comparison_csv')}")
        print(f"Dataset1 table    : {summary.output_paths.get('dataset1_main_comparison_csv')}")
        print(f"Dataset2 table    : {summary.output_paths.get('dataset2_external_comparison_csv')}")
        print(f"Dataset3 table    : {summary.output_paths.get('dataset3_online_case_study_csv')}")
        print("=" * 120)

        logger.info("Step 15 completed successfully.")

def run_step16_official_controlled_ablation_study(
    config,
    active_seed: int = 42,
    logger=None,
):
    """
    Step 16: Official controlled ablation study.

    This trains each locked ablation from scratch, selects theta/Np on
    Dataset-1 validation only, and evaluates Dataset-1 test, Dataset-2 external,
    and Dataset-3 online.
    """
    if logger is not None:
        logger.info("Step 16 official controlled ablation study is running.")
        logger.info("Each ablation variant will be trained from scratch.")
        logger.info("Threshold/persistence selection uses Dataset-1 validation only.")

    summary = run_step16_official_ablation_study(
        config=config,
        active_seed=active_seed,
    )

    final_status = getattr(summary, "final_status", None)
    output_paths = getattr(summary, "output_paths", {})

    if logger is not None:
        logger.info(f"Step 16 final status: {final_status}")
        logger.info(f"Step 16 summary JSON: {output_paths.get('ablation_summary_json')}")
        logger.info(f"Step 16 Dataset-1 ablation CSV: {output_paths.get('ablation_results_csv')}")
        logger.info(f"Step 16 all-splits ablation CSV: {output_paths.get('ablation_results_all_splits_csv')}")

    return summary

def run_step16_frozen_component_intervention_ablation_study(
    config,
    active_seed: int = 42,
    logger=None,
):
    """
    Step 16 Frozen Component-Intervention Ablation Study.

    This does not retrain ablations.
    It loads the same full proposed checkpoint and disables one module
    at evaluation time.
    """
    if logger is not None:
        logger.info("Step 16 frozen component-intervention ablation study is running.")
        logger.info("No ablation variant will be trained.")
        logger.info("All variants use the same full proposed checkpoint.")
        logger.info("One module is disabled at evaluation time only.")
        logger.info("Threshold/persistence selection uses Dataset-1 validation only.")

    summary = run_step16_frozen_intervention_ablation_study(
        config=config,
        active_seed=active_seed,
    )

    final_status = getattr(summary, "final_status", None)
    output_paths = getattr(summary, "output_paths", {})

    if logger is not None:
        logger.info(f"Frozen Step 16 final status: {final_status}")
        logger.info(
            f"Frozen Step 16 summary JSON: "
            f"{output_paths.get('frozen_ablation_summary_json')}"
        )
        logger.info(
            f"Frozen Step 16 Dataset-1 CSV: "
            f"{output_paths.get('frozen_ablation_results_csv')}"
        )
        logger.info(
            f"Frozen Step 16 all-splits CSV: "
            f"{output_paths.get('frozen_ablation_results_all_splits_csv')}"
        )

    return summary

def run_step17_high_order_feature_vs_model_comparison(
    config,
    active_seed: int = 42,
    logger=None,
):
    """
    Step 17: High-order feature-vs-model comparison.

    This answers:
        feature high-order vs model high-order?

    It trains/evaluates the four H variants:
        H0: no feature high-order, no model high-order
        H1: feature high-order only
        H2: model high-order only
        H3: feature high-order + model high-order
    """
    if logger is not None:
        logger.info("Step 17 high-order feature-vs-model comparison is running.")
        logger.info("Question: feature high-order vs model high-order?")
        logger.info("Design: 2x2 H0/H1/H2/H3 factorial comparison.")
        logger.info("Each H variant is trained on Dataset-1 train only.")
        logger.info("Threshold/persistence selection uses Dataset-1 validation only.")
        logger.info("Dataset-2 and Dataset-3 are evaluation-only.")

    summary = run_step17_high_order_comparison(
        config=config,
        active_seed=active_seed,
    )

    final_status = getattr(summary, "final_status", None)
    output_paths = getattr(summary, "output_paths", {})

    if logger is not None:
        logger.info(f"Step 17 final status: {final_status}")
        logger.info(
            f"Step 17 summary JSON: "
            f"{output_paths.get('high_order_comparison_summary_json')}"
        )
        logger.info(
            f"Step 17 Dataset-1 CSV: "
            f"{output_paths.get('high_order_comparison_csv')}"
        )
        logger.info(
            f"Step 17 all-splits CSV: "
            f"{output_paths.get('high_order_comparison_all_splits_csv')}"
        )
        logger.info(
            f"Step 17 effects CSV: "
            f"{output_paths.get('high_order_effects_csv')}"
        )

    return summary

def run_step17a_feature_group_intervention_analysis(
    config,
    active_seed: int = 42,
    logger=None,
):
    """
    Run Step 17A:
    feature-group intervention through the complete trained Proposed model.

    This uses:
    - official results/models/proposed_best.pt,
    - official Step-13 theta/Np,
    - no retraining,
    - feature-group masking at evaluation time.
    """
    if logger is not None:
        logger.info("Running Step 17A feature-group intervention analysis.")

    summary = run_step17a_feature_group_intervention_experiment(
        config=config,
        active_seed=active_seed,
    )

    if logger is not None:
        logger.info("Step 17A final status: %s", summary.final_status)
        logger.info("Step 17A saved outputs: %s", summary.output_paths)

    if summary.final_status != "PASSED":
        raise RuntimeError(f"Step 17A failed with status: {summary.final_status}")

    return summary


def run_step17b_kirchhoff_structure_analysis(
    config,
    active_seed: int = 42,
    logger=None,
):
    """
    Run Step 17B:
    Kirchhoff / model high-order structure comparison.

    This uses:
    - full xi features for K0/K1/K2/K3,
    - K0/K1/K2 trained fairly from scratch,
    - K3 reused from official proposed_best.pt,
    - K3 official Step-13 theta/Np.
    """
    if logger is not None:
        logger.info("Running Step 17B Kirchhoff/model high-order structure analysis.")

    summary = run_step17b_kirchhoff_structure_comparison_experiment(
        config=config,
        active_seed=active_seed,
    )

    if logger is not None:
        logger.info("Step 17B final status: %s", summary.final_status)
        logger.info("Step 17B saved outputs: %s", summary.output_paths)

    if summary.final_status != "PASSED":
        raise RuntimeError(f"Step 17B failed with status: {summary.final_status}")

    return summary


def run_step17_final_feature_model_high_order_analysis(
    config,
    active_seed: int = 42,
    logger=None,
):
    """
    Run final Step 17:
    Step 17A + Step 17B combined.

    Step 17A:
      feature-group contribution through complete trained Proposed model.

    Step 17B:
      full-feature Kirchhoff/model high-order structure comparison.
    """
    if logger is not None:
        logger.info("Running final Step 17 feature/model high-order analysis.")

    summary = run_step17_feature_model_analysis_experiment(
        config=config,
        active_seed=active_seed,
    )

    if logger is not None:
        logger.info("Final Step 17 status: %s", summary.final_status)

    if summary.final_status != "PASSED":
        raise RuntimeError(f"Final Step 17 failed with status: {summary.final_status}")

    return summary

def run_step19_dataset3_ekf_case_study(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 19: Dataset-3 online Proposed-vs-EKF comparison and case-study plot.

    This step:
    - does not train,
    - does not retune theta/N_p,
    - reuses saved Step-13 Dataset-3 Proposed predictions,
    - evaluates EKF Detector with the same event-level online alarm rule,
    - creates the Dataset-3 case-study figure.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step19"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step19_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 19 Dataset-3 EKF case-study comparison is running.")
        logger.info("Project root: %s", project_root)
        logger.info("Active seed: %s", active_seed)
        logger.info("Important: Step 19 does not train or retune thresholds.")
        logger.info("Important: Step 19 reuses Step-13 Dataset-3 predictions.")
        logger.info("Important: EKF Detector is used only for comparison, not as model input.")

        print_config_summary(config)
        print_sensitivity_plan(config)

        summary = run_step19_dataset3_case_study_impl(
            config=config,
            active_seed=active_seed,
            logger=logger,
        )

        final_status = summary.get("final_status", None)
        output_paths = summary.get("output_paths", {})

        logger.info("Step 19 final status: %s", final_status)
        logger.info("Step 19 comparison CSV: %s", output_paths.get("comparison_csv"))
        logger.info("Step 19 summary JSON: %s", output_paths.get("summary_json"))
        logger.info("Step 19 figure directory: %s", output_paths.get("figure_dir"))

        if final_status != "PASSED":
            raise RuntimeError(f"Step 19 failed with status: {final_status}")

        logger.info("Step 19 completed successfully.")

def run_step20_table_only_sensitivity_analysis(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 20: table-only sensitivity analysis.

    This step:
    - does not train,
    - does not retune theta/N_p,
    - reuses saved Step-13 Dataset-1 test predictions,
    - evaluates theta and persistence sensitivity,
    - skips rho and hidden_dim unless a retraining protocol is implemented.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step20"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step20_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 20 table-only sensitivity analysis is running.")
        logger.info("Project root: %s", project_root)
        logger.info("Active seed: %s", active_seed)
        logger.info("Important: Step 20 reuses saved Step-13 predictions.")
        logger.info("Important: Step 20 does not train.")
        logger.info("Important: Step 20 does not retune thresholds.")
        logger.info("Important: rho/hidden_dim sensitivity requires retraining and is skipped unless separately implemented.")

        print_config_summary(config)
        print_sensitivity_plan(config)

        summary = run_step20_sensitivity_analysis_impl(
            config=config,
            active_seed=active_seed,
            logger=logger,
        )

        final_status = summary.get("final_status", None)
        output_paths = summary.get("output_paths", {})

        logger.info("Step 20 final status: %s", final_status)
        logger.info("Step 20 sensitivity CSV: %s", output_paths.get("sensitivity_results_csv"))
        logger.info("Step 20 summary JSON: %s", output_paths.get("sensitivity_summary_json"))

        if final_status != "PASSED":
            raise RuntimeError(f"Step 20 failed with status: {final_status}")

        logger.info("Step 20 completed successfully.")

def run_step21_proposed_multiseed_robustness(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Run Step 21: Proposed-model multi-seed robustness experiment.

    This step:
    - retrains the full Proposed model across multiple random seeds,
    - selects theta and N_p on Dataset-1 validation only for each seed,
    - evaluates Dataset-1 test, Dataset-2 external, and Dataset-3 online,
    - keeps all seed outputs isolated under results/models/multiseed/ and results/tables/multiseed/,
    - writes aggregate mean ± std robustness tables.

    Important:
    - This step is not the official single-checkpoint Step 13 result.
    - This step is robustness evidence for the paper.
    - The actual seed list comes from experiments.multiseed_proposed.seeds.
    - Do not run this inside the normal global seed loop; main() handles it once.
    """
    device_info = prepare_common_runtime(config=config, active_seed=active_seed)

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    mode = str(get_by_path(config, "run.mode", "step21"))
    seed_mode = str(get_by_path(config, "seed.mode", "single"))
    run_id = make_run_id(prefix=f"step21_seed{active_seed}")

    with managed_run_logger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info.as_dict(),
        config_path=str(Path(config_dir).resolve()),
    ) as run_logger:
        logger = run_logger.logger

        logger.info("Step 21 Proposed multi-seed robustness experiment is running.")
        logger.info("Project root: %s", project_root)
        logger.info("Outer active seed: %s", active_seed)
        logger.info("Important: Step 21 manages its own seed list internally.")
        logger.info("Important: Step 21 keeps outputs isolated under results/models/multiseed and results/tables/multiseed.")

        print_config_summary(config)
        print_sensitivity_plan(config)

        enabled = bool(get_by_path(config, "experiments.multiseed_proposed.enabled", False))
        seeds = get_by_path(config, "experiments.multiseed_proposed.seeds", [])

        if not enabled:
            raise RuntimeError(
                "Step 21 requested, but experiments.multiseed_proposed.enabled is false or missing. "
                "Set experiments.multiseed_proposed.enabled: true in configs/experiments.yaml."
            )

        if not seeds:
            raise RuntimeError(
                "Step 21 requested, but experiments.multiseed_proposed.seeds is empty or missing. "
                "Set experiments.multiseed_proposed.seeds: [42, 43, 44, 45, 46]."
            )

        print("=" * 120)
        print("STEP 21 PROPOSED MULTI-SEED ROBUSTNESS START")
        print("=" * 120)
        print(f"Configured seeds : {seeds}")
        print("Output root      : results/tables/multiseed")
        print("=" * 120)

        summary = run_multiseed_proposed(
            config=config,
            seed_override=None,
            dry_run=False,
        )

        if isinstance(summary, Mapping):
            final_status = summary.get("final_status", summary.get("status"))
        else:
            final_status = getattr(summary, "final_status", None)

        logger.info("Step 21 final status: %s", final_status)
        logger.info("Step 21 expected all-seeds CSV: results/tables/multiseed/proposed_multiseed_all_seeds.csv")
        logger.info("Step 21 expected mean/std CSV: results/tables/multiseed/proposed_multiseed_mean_std.csv")
        logger.info("Step 21 expected paper table CSV: results/tables/multiseed/proposed_multiseed_paper_table.csv")
        logger.info("Step 21 expected summary JSON: results/tables/multiseed/proposed_multiseed_summary.json")

        if final_status != "PASSED":
            raise RuntimeError(f"Step 21 failed with status: {final_status}")

        print("=" * 120)
        print("STEP 21 PROPOSED MULTI-SEED ROBUSTNESS COMPLETE")
        print("=" * 120)

        logger.info("Step 21 completed successfully.")
def run_selected_mode(
    config: Mapping[str, Any],
    active_seed: int,
    config_dir: str,
) -> None:
    """
    Route execution based on run.mode.
    """
    mode = str(get_by_path(config, "run.mode", "step1")).lower().strip()

    if mode in {"step1", "setup"}:
        run_step1_check(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step2", "inspect", "raw_inspection"}:
        run_step2_raw_inspection(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step3", "segment", "segmentation", "trajectory_segmentation"}:
        run_step3_trajectory_segmentation(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step4", "split", "split_segments", "source_aware_split"}:
        run_step4_source_aware_split(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step5", "clean", "clean_columns", "shortcut_exclusion"}:
        run_step5_clean_columns(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step6", "physical", "coordinate_motion", "motion_model"}:
        run_step6_coordinate_motion_model(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step7", "residual", "residuals", "normal_stats", "residual_statistics"}:
        run_step7_residual_and_normal_statistics(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step8", "evidence", "xi", "build_xi", "evidence_builder"}:
        run_step8_evidence_builder(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step9", "dataset_objects", "sequence_batching", "batching", "xi_dataset"}:
        run_step9_dataset_objects(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step10", "evaluation", "metrics", "threshold_selection", "alarm_rules"}:
        run_step10_evaluation_framework(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step11", "model", "models", "proposed_model", "model_factory"}:
        run_step11_model_factory(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {"step12", "train", "training", "trainer", "fit"}:
        run_step12_training_loop(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {
        "step13",
        "proposed",
        "run_proposed",
        "full_proposed",
        "proposed_experiment",
        "evaluate_proposed",
    }:
        run_step13_full_proposed_experiment(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {
        "step14",
        "module_usage",
        "module_diagnostics",
        "full_model_diagnostics",
        "diagnostics",
        "feature_importance",
        "occlusion_diagnostics",
    }:
        run_step14_full_model_module_usage_diagnostics(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {
        "step15",
        "baselines",
        "baseline",
        "run_baselines",
        "train_baselines",
        "evaluate_baselines",
        "baseline_comparison",
        "official_baselines",
    }:
        run_step15_official_baseline_comparison(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {
        "step16_frozen",
        "frozen_ablations",
        "frozen_ablation",
        "frozen_intervention_ablations",
        "frozen_intervention_ablation",
        "component_intervention_ablation",
        "component_intervention_ablations",
        "run_frozen_ablations",
        "evaluate_frozen_ablations",
    }:
        run_step16_frozen_component_intervention_ablation_study(
            config=config,
            active_seed=active_seed,
            logger=None,
        )
        return

    if mode in {
        "step16",
        "ablations",
        "ablation",
        "official_ablations",
        "official_ablation",
        "controlled_ablations",
        "controlled_ablation",
        "ablation_study",
        "run_ablations",
        "train_ablations",
        "evaluate_ablations",
    }:
        run_step16_official_controlled_ablation_study(
            config=config,
            active_seed=active_seed,
            logger=None,
        )
        return

    if mode in {
        "step17",
        "step17_high_order",
        "high_order_comparison",
        "feature_vs_model_high_order",
        "model_vs_feature_high_order",
        "run_high_order_comparison",
        "professor_high_order_comparison",
    }:
        run_step17_high_order_feature_vs_model_comparison(
            config=config,
            active_seed=active_seed,
            logger=None,
        )
        return
    # ------------------------------------------------------------------
    # Step 17A — Feature-group intervention through complete Proposed model
    # ------------------------------------------------------------------
    if mode in {
        "step17a",
        "step17a_features",
        "step17a_feature_group",
        "step17a_feature_group_intervention",
        "feature_group_intervention",
        "feature_contribution",
        "feature_group_contribution",
    }:
        run_step17a_feature_group_intervention_analysis(
            config=config,
            active_seed=active_seed,
            logger=None,
        )
        return

    # ------------------------------------------------------------------
    # Step 17B — Kirchhoff / model high-order structure comparison
    # ------------------------------------------------------------------
    if mode in {
        "step17b",
        "step17b_kirchhoff",
        "step17b_kirchhoff_structure",
        "step17b_kirchhoff_structure_comparison",
        "kirchhoff_structure_comparison",
        "model_high_order_structure",
        "kirchhoff_high_order",
    }:
        run_step17b_kirchhoff_structure_analysis(
            config=config,
            active_seed=active_seed,
            logger=None,
        )
        return

    # ------------------------------------------------------------------
    # Final Step 17 — Step 17A + Step 17B combined
    # ------------------------------------------------------------------
    if mode in {
        "step17_final",
        "step17_feature_model_analysis",
        "step17_final_feature_model_analysis",
        "step17_feature_model_high_order",
        "final_feature_model_high_order",
        "feature_model_high_order_analysis",
    }:
        run_step17_final_feature_model_high_order_analysis(
            config=config,
            active_seed=active_seed,
            logger=None,
        )
        return

    if mode in {
        "step19",
        "dataset3_case_study",
        "dataset3_ekf",
        "dataset3_ekf_case_study",
        "online_case_study",
        "ekf_case_study",
        "proposed_vs_ekf",
    }:
        run_step19_dataset3_ekf_case_study(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in {
        "step20",
        "sensitivity",
        "sensitivity_analysis",
        "run_sensitivity",
        "operating_point_sensitivity",
        "threshold_sensitivity",
        "theta_sensitivity",
        "persistence_sensitivity",
    }:
        run_step20_table_only_sensitivity_analysis(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    if mode in STEP21_MULTI_SEED_MODES:
        run_step21_proposed_multiseed_robustness(
            config=config,
            active_seed=active_seed,
            config_dir=config_dir,
        )
        return

    raise NotImplementedError(
        f"run.mode='{mode}' is not implemented yet. "
        "Currently supported modes: step1, setup, step2, inspect, "
        "raw_inspection, step3, segment, segmentation, trajectory_segmentation, "
        "step4, split, split_segments, source_aware_split, "
        "step5, clean, clean_columns, shortcut_exclusion, "
        "step6, physical, coordinate_motion, motion_model, "
        "step7, residual, residuals, normal_stats, residual_statistics, "
        "step8, evidence, xi, build_xi, evidence_builder, "
        "step9, dataset_objects, sequence_batching, batching, xi_dataset, "
        "step10, evaluation, metrics, threshold_selection, alarm_rules, "
        "step11, model, models, proposed_model, model_factory, "
        "step12, train, training, trainer, fit, "
        "step13, proposed, run_proposed, full_proposed, proposed_experiment, "
        "evaluate_proposed, "
        "step14, module_usage, module_diagnostics, full_model_diagnostics, "
        "diagnostics, feature_importance, occlusion_diagnostics, "
        "step15, baselines, baseline, run_baselines, train_baselines, "
        "evaluate_baselines, baseline_comparison, official_baselines, "
        "step16, ablations, ablation, official_ablations, official_ablation, "
        "controlled_ablations, controlled_ablation, ablation_study, "
        "run_ablations, train_ablations, evaluate_ablations, "
        "step16_frozen, frozen_ablations, frozen_ablation, "
        "frozen_intervention_ablations, run_frozen_ablations, "
        "evaluate_frozen_ablations."
        "step17, step17_high_order, high_order_comparison, "
        "feature_vs_model_high_order, model_vs_feature_high_order, "
        "run_high_order_comparison, professor_high_order_comparison."
        "step19, dataset3_case_study, dataset3_ekf, "
        "dataset3_ekf_case_study, online_case_study, "
        "ekf_case_study, proposed_vs_ekf."
        "step20, sensitivity, sensitivity_analysis, run_sensitivity, "
        "operating_point_sensitivity, threshold_sensitivity, "
        "theta_sensitivity, persistence_sensitivity."
        "step21/step21_multiseed/multiseed/proposed_multiseed/multiseed_proposed/seed_robustness/training_robustness."
    )

def main() -> None:
    """Main function."""
    args = parse_args()

    config = load_project_config(config_dir=args.config_dir)
    overrides = build_cli_overrides(args)
    config = apply_overrides(config, overrides)

    print_config_summary(config)
    print_seed_plan(config)

    seed_list = resolve_seed_list(config)

    # for active_seed in seed_list:
    #     run_selected_mode(
    #         config=config,
    #         active_seed=active_seed,
    #         config_dir=args.config_dir,
    #     )
    seed_list = resolve_seed_list(config)

    mode = str(get_by_path(config, "run.mode", "step1")).lower().strip()

    # Step 21 manages its own seed list internally through
    # experiments.multiseed_proposed.seeds. Therefore, do not run it once per
    # global seed-mode seed.
    if mode in STEP21_MULTI_SEED_MODES:
        active_seed = int(seed_list[0]) if seed_list else int(get_by_path(config, "seed.single_seed", 42))
        run_selected_mode(
            config=config,
            active_seed=active_seed,
            config_dir=args.config_dir,
        )
        return

    for active_seed in seed_list:
        run_selected_mode(
            config=config,
            active_seed=active_seed,
            config_dir=args.config_dir,
        )


if __name__ == "__main__":
    main()