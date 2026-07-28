"""
Multi-seed robustness experiment for the official Proposed model.

Purpose:
- Train the full Proposed model from scratch for multiple random seeds.
- For each seed, select theta/Np on Dataset-1 validation only.
- Evaluate Dataset-1 test, Dataset-2 external, and Dataset-3 online.
- Save every seed in its own isolated directory so the official Step-13
  checkpoint and tables are not overwritten.

Run:
    python -m src.experiments.run_multiseed_proposed

Optional one-seed test:
    python -m src.experiments.run_multiseed_proposed --seeds 42
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

# Keep the same safe environment behavior as the main project runner.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from src.experiments.run_proposed import run_full_proposed_experiment
from src.utils.config import get_by_path, load_project_config, resolve_project_path
from src.utils.io import ensure_dir


# ---------------------------------------------------------------------
# Small config/path helpers
# ---------------------------------------------------------------------


def _set_by_path(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set nested dictionary value using a dotted path."""
    current: Dict[str, Any] = config
    parts = path.split(".")

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def _project_path(config: Mapping[str, Any], path_value: str | Path) -> Path:
    """Resolve path relative to project root."""
    return resolve_project_path(config, str(path_value))


def _json_safe(value: Any) -> Any:
    """Convert numpy/path/pandas values into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if pd.isna(value) if isinstance(value, (float, np.floating)) else False:
        return None

    return value


def _save_json(payload: Mapping[str, Any], path: Path) -> None:
    """Save JSON with parent directory creation."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(dict(payload)), f, indent=2, ensure_ascii=False)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save CSV with parent directory creation."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def setup_multiseed_seed(config: Mapping[str, Any], active_seed: int) -> None:
    """
    Set Python, NumPy, and PyTorch seeds for one multi-seed run.

    This does not regenerate Dataset-1 train/val/test splits.
    It only controls model initialization, dataloader shuffling, and
    training-time randomness.
    """
    deterministic = bool(get_by_path(config, "seed.deterministic", True))
    warn_only = bool(get_by_path(config, "seed.deterministic_warn_only", True))
    benchmark = bool(get_by_path(config, "seed.benchmark", False))
    cublas_config = str(
        get_by_path(config, "seed.cublas_workspace_config", ":4096:8")
    )
    cuda_tf32 = bool(get_by_path(config, "seed.cuda_tf32", False))

    os.environ["PYTHONHASHSEED"] = str(active_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config

    random.seed(active_seed)
    np.random.seed(active_seed)

    torch.manual_seed(active_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(active_seed)
        torch.cuda.manual_seed_all(active_seed)

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
    torch.backends.cudnn.allow_tf32 = cuda_tf32

    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    except TypeError:
        # Older PyTorch versions may not support warn_only.
        torch.use_deterministic_algorithms(deterministic)

    print("=" * 100)
    print("MULTI-SEED SEED SETUP")
    print("=" * 100)
    print(f"Active seed              : {active_seed}")
    print(f"Deterministic             : {deterministic}")
    print(f"Deterministic warn only   : {warn_only}")
    print(f"cuDNN benchmark           : {benchmark}")
    print(f"CUBLAS_WORKSPACE_CONFIG   : {cublas_config}")
    print(f"CUDA TF32 allowed         : {cuda_tf32}")
    print("=" * 100)


# ---------------------------------------------------------------------
# Multi-seed config preparation
# ---------------------------------------------------------------------


def _parse_seed_list(value: Any) -> List[int]:
    """Parse seed list from YAML or CLI string."""
    if value is None:
        return []

    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]

    if isinstance(value, Iterable):
        return [int(x) for x in value]

    return [int(value)]


def build_seed_run_config(
    base_config: Mapping[str, Any],
    seed: int,
    base_models_dir: str,
    base_tables_dir: str,
    retrain_policy: str,
    evaluate_dataset1: bool,
    evaluate_dataset2: bool,
    evaluate_dataset3: bool,
) -> Dict[str, Any]:
    """
    Build an isolated Step-13 config for one seed.

    This is the key safety function. It redirects model checkpoints,
    Step-12 outputs, Step-13 tables, predictions, and threshold artifacts
    to the seed-specific directories.
    """
    cfg: Dict[str, Any] = copy.deepcopy(dict(base_config))

    seed_name = f"seed_{int(seed)}"
    seed_models_dir = str(Path(base_models_dir) / seed_name)
    seed_tables_dir = str(Path(base_tables_dir) / seed_name)

    # Seed state for this run.
    _set_by_path(cfg, "seed.mode", "single")
    _set_by_path(cfg, "seed.single_seed", int(seed))

    # Force GPU for multi-seed training.
    # If CUDA is not available, fail instead of silently falling back to CPU.
    _set_by_path(cfg, "device.preference", "cuda")
    _set_by_path(cfg, "device.gpu_index", 0)
    _set_by_path(cfg, "device.allow_cpu_fallback", False)
    _set_by_path(cfg, "device.require_gpu", True)

    # Force full Proposed model, not ablation/high-order variants.
    _set_by_path(cfg, "training.step12.model_name", "proposed")
    _set_by_path(cfg, "training.step12.variant_name", "full")
    _set_by_path(cfg, "training.checkpointing.best_checkpoint_name", "proposed_best.pt")
    _set_by_path(cfg, "training.checkpointing.last_checkpoint_name", "proposed_last.pt")

    # Important: Step 13 reads experiments.proposed first, then experiments.step13.
    # Set both so there is no ambiguity.
    _set_by_path(cfg, "experiments.proposed.model_name", "Proposed")
    _set_by_path(cfg, "experiments.proposed.retrain_policy", retrain_policy)
    _set_by_path(cfg, "experiments.proposed.checkpoint_path", None)
    _set_by_path(cfg, "experiments.proposed.evaluate_dataset1", evaluate_dataset1)
    _set_by_path(cfg, "experiments.proposed.evaluate_dataset2", evaluate_dataset2)
    _set_by_path(cfg, "experiments.proposed.evaluate_dataset3", evaluate_dataset3)

    _set_by_path(cfg, "experiments.step13.model_name", "Proposed")
    _set_by_path(cfg, "experiments.step13.retrain_policy", retrain_policy)
    _set_by_path(cfg, "experiments.step13.checkpoint_path", None)
    _set_by_path(cfg, "experiments.step13.evaluate_dataset1", evaluate_dataset1)
    _set_by_path(cfg, "experiments.step13.evaluate_dataset2", evaluate_dataset2)
    _set_by_path(cfg, "experiments.step13.evaluate_dataset3", evaluate_dataset3)

    # Step-12 checkpoint directory.
    _set_by_path(cfg, "paths.models_dir", seed_models_dir)

    # Step-12 training artifacts.
    _set_by_path(
        cfg,
        "paths.step12_training_history_csv",
        str(Path(seed_tables_dir) / "step12_training_history.csv"),
    )
    _set_by_path(
        cfg,
        "paths.step12_training_history_json",
        str(Path(seed_tables_dir) / "step12_training_history.json"),
    )
    _set_by_path(
        cfg,
        "paths.step12_training_summary_json",
        str(Path(seed_tables_dir) / "step12_training_summary.json"),
    )
    _set_by_path(
        cfg,
        "paths.step12_validation_predictions_npz",
        str(Path(seed_tables_dir) / "step12_validation_predictions.npz"),
    )

    # Step-13 summary.
    _set_by_path(
        cfg,
        "paths.step13_proposed_experiment_summary_json",
        str(Path(seed_tables_dir) / "step13_proposed_experiment_summary.json"),
    )

    # Dataset-1 artifacts.
    _set_by_path(
        cfg,
        "paths.dataset1_main_comparison_csv",
        str(Path(seed_tables_dir) / "dataset1_main_comparison.csv"),
    )
    _set_by_path(
        cfg,
        "paths.dataset1_proposed_summary_json",
        str(Path(seed_tables_dir) / "dataset1_proposed_summary.json"),
    )
    _set_by_path(
        cfg,
        "paths.dataset1_val_predictions_npz",
        str(Path(seed_tables_dir) / "dataset1_val_predictions.npz"),
    )
    _set_by_path(
        cfg,
        "paths.dataset1_test_predictions_npz",
        str(Path(seed_tables_dir) / "dataset1_test_predictions.npz"),
    )

    # Threshold-selection artifacts.
    _set_by_path(
        cfg,
        "paths.proposed_threshold_selection_json",
        str(Path(seed_tables_dir) / "proposed_threshold_selection.json"),
    )
    _set_by_path(
        cfg,
        "paths.proposed_threshold_candidates_csv",
        str(Path(seed_tables_dir) / "proposed_threshold_candidates.csv"),
    )

    # Dataset-2 artifacts.
    _set_by_path(
        cfg,
        "paths.dataset2_external_comparison_csv",
        str(Path(seed_tables_dir) / "dataset2_external_comparison.csv"),
    )
    _set_by_path(
        cfg,
        "paths.dataset2_proposed_summary_json",
        str(Path(seed_tables_dir) / "dataset2_proposed_summary.json"),
    )
    _set_by_path(
        cfg,
        "paths.dataset2_external_predictions_npz",
        str(Path(seed_tables_dir) / "dataset2_external_predictions.npz"),
    )

    # Dataset-3 artifacts.
    _set_by_path(
        cfg,
        "paths.dataset3_online_case_study_csv",
        str(Path(seed_tables_dir) / "dataset3_online_case_study.csv"),
    )
    _set_by_path(
        cfg,
        "paths.dataset3_proposed_summary_json",
        str(Path(seed_tables_dir) / "dataset3_proposed_summary.json"),
    )
    _set_by_path(
        cfg,
        "paths.dataset3_online_predictions_npz",
        str(Path(seed_tables_dir) / "dataset3_online_predictions.npz"),
    )
    _set_by_path(
        cfg,
        "paths.dataset3_online_predictions_csv",
        str(Path(seed_tables_dir) / "dataset3_online_predictions.csv"),
    )

    return cfg


def assert_seed_config_is_safe(
    base_config: Mapping[str, Any],
    seed_config: Mapping[str, Any],
) -> None:
    """
    Prevent accidental overwriting of official Step-13 files.

    This compares official paths from the base config against the seed-specific
    paths after redirection.
    """
    official_models_dir = _project_path(
        base_config,
        str(get_by_path(base_config, "paths.models_dir", "results/models")),
    )
    seed_models_dir = _project_path(
        seed_config,
        str(get_by_path(seed_config, "paths.models_dir", "results/models")),
    )

    official_checkpoint = official_models_dir / "proposed_best.pt"
    seed_checkpoint = seed_models_dir / "proposed_best.pt"

    if official_checkpoint.resolve() == seed_checkpoint.resolve():
        raise RuntimeError(
            "Unsafe multi-seed config: seed checkpoint path equals official checkpoint path:\n"
            f"{official_checkpoint}"
        )

    output_keys = [
        "paths.dataset1_main_comparison_csv",
        "paths.dataset2_external_comparison_csv",
        "paths.dataset3_online_case_study_csv",
        "paths.proposed_threshold_selection_json",
        "paths.step13_proposed_experiment_summary_json",
    ]

    for key in output_keys:
        official_value = get_by_path(base_config, key, None)
        seed_value = get_by_path(seed_config, key, None)

        if official_value is None or seed_value is None:
            continue

        official_path = _project_path(base_config, str(official_value)).resolve()
        seed_path = _project_path(seed_config, str(seed_value)).resolve()

        if official_path == seed_path:
            raise RuntimeError(
                f"Unsafe multi-seed config: {key} would overwrite official file:\n"
                f"{official_path}"
            )


# ---------------------------------------------------------------------
# Result extraction and aggregation
# ---------------------------------------------------------------------


METRIC_ALIASES: Dict[str, List[str]] = {
    "AUROC": ["auroc", "AUROC"],
    "AUPRC": ["auprc", "AUPRC"],
    "F1": ["f1", "F1"],
    "Precision": ["precision", "Precision"],
    "Recall": ["recall", "Recall"],
    "FPR": ["fpr", "FPR"],
    "ADR": [
        "attack_detection_rate",
        "Attack Detection Rate",
        "ADR",
        "Attack_Detection_Rate",
    ],
    "Delay": [
        "mean_detection_delay",
        "Detection Delay",
        "Mean Delay",
        "detection_delay",
        "Delay",
    ],
}


def _metric(metrics: Mapping[str, Any], canonical_name: str) -> Optional[float]:
    """Get one metric using alias lookup."""
    aliases = METRIC_ALIASES.get(canonical_name, [canonical_name])

    for key in aliases:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except Exception:
                return None

    # Case-insensitive fallback.
    lower_map = {str(k).lower(): v for k, v in metrics.items()}
    for key in aliases:
        value = lower_map.get(str(key).lower())
        if value is not None:
            try:
                return float(value)
            except Exception:
                return None

    return None


def flatten_seed_summary(seed: int, summary: Any) -> List[Dict[str, Any]]:
    """Convert ProposedExperimentSummary into one row per dataset."""
    if hasattr(summary, "to_dict"):
        payload = summary.to_dict()
    elif isinstance(summary, Mapping):
        payload = dict(summary)
    else:
        raise TypeError(f"Unsupported summary type: {type(summary)!r}")

    dataset_items = [
        ("Dataset-1 Test", payload.get("dataset1_result")),
        ("Dataset-2 External", payload.get("dataset2_result")),
        ("Dataset-3 Online", payload.get("dataset3_result")),
    ]

    rows: List[Dict[str, Any]] = []

    for dataset_name, result in dataset_items:
        if not result:
            continue

        metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}

        row = {
            "seed": int(seed),
            "dataset": dataset_name,
            "model": result.get("model_name", "Proposed"),
            "theta": result.get("threshold", metrics.get("theta")),
            "persistence": result.get("persistence", metrics.get("persistence")),
            "checkpoint_path": result.get("checkpoint_path"),
        }

        for canonical_name in METRIC_ALIASES:
            row[canonical_name] = _metric(metrics, canonical_name)

        rows.append(row)

    return rows


def aggregate_multiseed_rows(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std per dataset."""
    metric_cols = [
        "AUROC",
        "AUPRC",
        "F1",
        "Precision",
        "Recall",
        "FPR",
        "ADR",
        "Delay",
        "theta",
        "persistence",
    ]

    output_rows: List[Dict[str, Any]] = []

    for dataset_name, group in all_rows.groupby("dataset", sort=False):
        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(x) for x in sorted(group["seed"].unique())),
        }

        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
            else:
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

        output_rows.append(row)

    return pd.DataFrame(output_rows)


def make_paper_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create mean ± std formatted table for paper drafting."""
    metrics = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "FPR", "ADR", "Delay"]

    rows: List[Dict[str, Any]] = []

    for _, item in summary_df.iterrows():
        row: Dict[str, Any] = {
            "Dataset": item["dataset"],
            "Seeds": int(item["n_seeds"]),
        }

        for metric in metrics:
            mean_value = item.get(f"{metric}_mean")
            std_value = item.get(f"{metric}_std")

            if pd.isna(mean_value):
                row[metric] = ""
            else:
                row[metric] = f"{float(mean_value):.4f} ± {float(std_value):.4f}"

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------


def run_multiseed_proposed(
    config: Mapping[str, Any],
    seed_override: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the multi-seed Proposed robustness experiment."""
    block = get_by_path(config, "experiments.multiseed_proposed", {})

    enabled = bool(block.get("enabled", False))
    if not enabled:
        raise RuntimeError(
            "experiments.multiseed_proposed.enabled is false or missing. "
            "Enable it before running multi-seed."
        )

    experiment_name = str(block.get("experiment_name", "proposed_multiseed_robustness"))
    seeds = seed_override if seed_override is not None else _parse_seed_list(block.get("seeds", []))

    if not seeds:
        raise ValueError("No seeds provided for multi-seed run.")

    retrain_policy = str(block.get("retrain_policy", "always")).lower().strip()
    if retrain_policy != "always":
        raise ValueError(
            "For multi-seed robustness, retrain_policy should be 'always'. "
            f"Got: {retrain_policy!r}"
        )

    evaluate_dataset1 = bool(block.get("evaluate_dataset1", True))
    evaluate_dataset2 = bool(block.get("evaluate_dataset2", True))
    evaluate_dataset3 = bool(block.get("evaluate_dataset3", True))

    base_models_dir = str(block.get("base_models_dir", "results/models/multiseed"))
    base_tables_dir = str(block.get("base_tables_dir", "results/tables/multiseed"))
    continue_on_error = bool(block.get("continue_on_error", False))

    base_tables_path = _project_path(config, base_tables_dir)
    ensure_dir(base_tables_path)

    all_seed_rows: List[Dict[str, Any]] = []
    seed_summaries: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    start_time = time.perf_counter()

    print("=" * 120)
    print("PROPOSED MULTI-SEED ROBUSTNESS START")
    print("=" * 120)
    print(f"Experiment name : {experiment_name}")
    print(f"Seeds           : {seeds}")
    print(f"Models root     : {_project_path(config, base_models_dir)}")
    print(f"Tables root     : {base_tables_path}")
    print(f"Dry run         : {dry_run}")
    print("=" * 120)

    for index, seed in enumerate(seeds, start=1):
        print("=" * 120)
        print(f"MULTI-SEED RUN {index}/{len(seeds)} | seed={seed}")
        print("=" * 120)

        seed_config = build_seed_run_config(
            base_config=config,
            seed=seed,
            base_models_dir=base_models_dir,
            base_tables_dir=base_tables_dir,
            retrain_policy=retrain_policy,
            evaluate_dataset1=evaluate_dataset1,
            evaluate_dataset2=evaluate_dataset2,
            evaluate_dataset3=evaluate_dataset3,
        )

        assert_seed_config_is_safe(base_config=config, seed_config=seed_config)

        seed_models_dir = _project_path(
            seed_config,
            str(get_by_path(seed_config, "paths.models_dir")),
        )
        seed_tables_dir = _project_path(
            seed_config,
            str(Path(base_tables_dir) / f"seed_{seed}"),
        )

        ensure_dir(seed_models_dir)
        ensure_dir(seed_tables_dir)

        print(f"Seed models dir : {seed_models_dir}")
        print(f"Seed tables dir : {seed_tables_dir}")

        if dry_run:
            print("Dry run only; skipping training/evaluation.")
            continue

        try:
            setup_multiseed_seed(seed_config, int(seed))

            summary = run_full_proposed_experiment(
                config=seed_config,
                active_seed=int(seed),
            )

            rows = flatten_seed_summary(seed=int(seed), summary=summary)
            all_seed_rows.extend(rows)

            summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)
            seed_summaries.append(
                {
                    "seed": int(seed),
                    "status": "PASSED",
                    "selected_threshold": summary_dict.get("selected_threshold"),
                    "selected_persistence": summary_dict.get("selected_persistence"),
                    "checkpoint_path": summary_dict.get("checkpoint_path"),
                    "summary_json": get_by_path(
                        seed_config,
                        "paths.step13_proposed_experiment_summary_json",
                        str(seed_tables_dir / "step13_proposed_experiment_summary.json"),
                    ),
                }
            )

        except Exception as exc:
            error = {
                "seed": int(seed),
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            error_rows.append(error)
            seed_summaries.append(error)

            print("=" * 120)
            print(f"MULTI-SEED RUN FAILED | seed={seed}")
            print("=" * 120)
            print(f"{type(exc).__name__}: {exc}")

            if not continue_on_error:
                raise

    all_rows_df = pd.DataFrame(all_seed_rows)
    seed_summary_df = pd.DataFrame(seed_summaries)
    error_df = pd.DataFrame(error_rows)

    all_rows_csv = base_tables_path / "proposed_multiseed_all_seeds.csv"
    seed_summary_csv = base_tables_path / "proposed_multiseed_seed_summary.csv"
    mean_std_csv = base_tables_path / "proposed_multiseed_mean_std.csv"
    paper_csv = base_tables_path / "proposed_multiseed_paper_table.csv"
    summary_json = base_tables_path / "proposed_multiseed_summary.json"

    if not all_rows_df.empty:
        mean_std_df = aggregate_multiseed_rows(all_rows_df)
        paper_df = make_paper_summary_table(mean_std_df)
    else:
        mean_std_df = pd.DataFrame()
        paper_df = pd.DataFrame()

    _save_csv(all_rows_df, all_rows_csv)
    _save_csv(seed_summary_df, seed_summary_csv)
    _save_csv(mean_std_df, mean_std_csv)
    _save_csv(paper_df, paper_csv)

    if not error_df.empty:
        _save_csv(error_df, base_tables_path / "proposed_multiseed_errors.csv")

    final_payload = {
        "experiment_name": experiment_name,
        "final_status": "PASSED" if not error_rows else "FAILED_PARTIAL",
        "seeds": [int(s) for s in seeds],
        "runtime_seconds": float(time.perf_counter() - start_time),
        "output_paths": {
            "all_seed_rows_csv": str(all_rows_csv),
            "seed_summary_csv": str(seed_summary_csv),
            "mean_std_csv": str(mean_std_csv),
            "paper_table_csv": str(paper_csv),
            "summary_json": str(summary_json),
            "models_root": str(_project_path(config, base_models_dir)),
            "tables_root": str(base_tables_path),
        },
        "seed_summaries": seed_summaries,
        "errors": error_rows,
        "leakage_rules": {
            "dataset1_train_used_for_training": True,
            "dataset1_validation_used_for_threshold_selection": True,
            "dataset1_test_used_only_for_internal_test": True,
            "dataset2_used_only_for_external_test": bool(evaluate_dataset2),
            "dataset3_used_only_for_online_case_study": bool(evaluate_dataset3),
            "official_step13_checkpoint_not_overwritten": True,
            "per_seed_output_directories": True,
        },
    }

    _save_json(final_payload, summary_json)

    print("=" * 120)
    print("PROPOSED MULTI-SEED ROBUSTNESS SUMMARY")
    print("=" * 120)
    print(f"Final status      : {final_payload['final_status']}")
    print(f"Seeds             : {seeds}")
    print(f"All seeds CSV     : {all_rows_csv}")
    print(f"Mean/std CSV      : {mean_std_csv}")
    print(f"Paper table CSV   : {paper_csv}")
    print(f"Summary JSON      : {summary_json}")

    if not paper_df.empty:
        print("=" * 120)
        print("PAPER-FORMATTED MEAN ± STD TABLE")
        print("=" * 120)
        print(paper_df.to_string(index=False))

    print("=" * 120)

    return final_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Proposed model multi-seed robustness experiment."
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing YAML config files.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed override, e.g. '42' or '42,43,44'.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved seed directories without training/evaluation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_project_config(config_dir=args.config_dir)

    seed_override = _parse_seed_list(args.seeds) if args.seeds is not None else None

    run_multiseed_proposed(
        config=config,
        seed_override=seed_override,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()