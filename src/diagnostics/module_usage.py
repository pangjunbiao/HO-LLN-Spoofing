"""
Module-usage diagnostics for the trained full proposed model.

Step 14 purpose:
- Inspect whether the trained full model actually activates its internal modules.
- Summarize module outputs using forward hooks.
- Save diagnostic CSV/JSON files under results/figures/module_usage/.
- Provide shared Step-14 utilities used by feature importance, conductance analysis,
  third-order analysis, liquid-state analysis, and occlusion tests.

Important:
- This file is diagnostic only.
- It does not retrain the model.
- It does not select a new threshold.
- It uses the Step-13 validation-selected theta and persistence.
- Official ablation remains retraining from scratch in later steps.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.evaluation.evaluate_dataset1 import (
    build_evaluation_dataloader,
    load_trained_model_for_evaluation,
)
from src.evaluation.evaluate_dataset2 import SelectedThreshold, load_selected_threshold
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import move_to_device, setup_device_from_config
from src.utils.io import ensure_dir


DEFAULT_STEP14_FEATURE_COLUMNS = [
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


DEFAULT_MODULE_KEYWORDS = [
    "evidence",
    "encoder",
    "kirchhoff",
    "exchange",
    "conductance",
    "third",
    "fusion",
    "liquid",
    "temporal",
    "output",
    "head",
]


@dataclass
class FeatureGroupSpec:
    """Feature-group specification for Step-14 diagnostics."""

    name: str
    columns: List[str]
    indices: List[int]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step14DiagnosticsConfig:
    """Step-14 diagnostic configuration."""

    output_dir: str = "results/figures/module_usage"
    table_dir: str = "results/tables"

    checkpoint_path: Optional[str] = None
    model_name: str = "Proposed"
    variant_name: str = "full"

    diagnostic_splits: List[str] = field(
        default_factory=lambda: ["test", "external", "online"]
    )

    module_keywords: List[str] = field(default_factory=lambda: list(DEFAULT_MODULE_KEYWORDS))
    max_modules: int = 80
    max_batches_per_split: Optional[int] = None

    save_json: bool = True
    save_csv: bool = True
    print_console_summary: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step14DiagnosticsPaths:
    """Resolved Step-14 output paths."""

    output_dir: Path
    table_dir: Path

    module_usage_json: Path
    module_usage_csv: Path

    feature_importance_json: Path
    feature_importance_csv: Path

    conductance_json: Path
    conductance_csv: Path

    third_order_json: Path
    third_order_csv: Path

    liquid_state_json: Path
    liquid_state_csv: Path

    occlusion_json: Path
    occlusion_csv: Path

    summary_json: Path

    def to_dict(self) -> Dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass
class Step14DiagnosticsContext:
    """Loaded model, threshold, and runtime context."""

    config: Mapping[str, Any]
    diagnostics_config: Step14DiagnosticsConfig
    paths: Step14DiagnosticsPaths
    active_seed: int
    device: torch.device
    model: nn.Module
    checkpoint_path: Path
    checkpoint_metadata: Dict[str, Any]
    selected_threshold: SelectedThreshold
    feature_columns: List[str]
    feature_groups: List[FeatureGroupSpec]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_seed": int(self.active_seed),
            "device": str(self.device),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_metadata": _json_safe(self.checkpoint_metadata),
            "selected_threshold": self.selected_threshold.to_dict(),
            "feature_columns": list(self.feature_columns),
            "feature_groups": [group.to_dict() for group in self.feature_groups],
            "diagnostics_config": self.diagnostics_config.to_dict(),
            "paths": self.paths.to_dict(),
        }


@dataclass
class ActivationSummary:
    """Summary statistics for one module activation."""

    split: str
    module_name: str
    module_class: str

    tensor_count: int
    element_count: int

    mean: Optional[float]
    std: Optional[float]
    mean_abs: Optional[float]
    max_abs: Optional[float]
    nonzero_fraction: Optional[float]

    shape_examples: List[str]
    runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActivationAccumulator:
    """Streaming accumulator for module output tensors."""

    def __init__(self, split: str, module_name: str, module_class: str) -> None:
        self.split = str(split)
        self.module_name = str(module_name)
        self.module_class = str(module_class)

        self.tensor_count = 0
        self.element_count = 0

        self.sum_value = 0.0
        self.sum_square = 0.0
        self.sum_abs = 0.0
        self.max_abs = 0.0
        self.nonzero_count = 0

        self.shape_examples: List[str] = []
        self.start_time = time.perf_counter()

    def update(self, tensor: Tensor) -> None:
        """Update accumulator with one tensor."""
        if tensor is None:
            return

        if not torch.is_tensor(tensor):
            return

        arr = tensor.detach().float().cpu()

        if arr.numel() == 0:
            return

        self.tensor_count += 1
        self.element_count += int(arr.numel())

        self.sum_value += float(arr.sum().item())
        self.sum_square += float((arr * arr).sum().item())
        self.sum_abs += float(arr.abs().sum().item())
        self.max_abs = max(self.max_abs, float(arr.abs().max().item()))
        self.nonzero_count += int((arr.abs() > 1.0e-12).sum().item())

        if len(self.shape_examples) < 5:
            self.shape_examples.append(str(tuple(arr.shape)))

    def summary(self) -> ActivationSummary:
        """Return JSON-safe summary."""
        if self.element_count <= 0:
            mean = None
            std = None
            mean_abs = None
            max_abs = None
            nonzero_fraction = None
        else:
            mean = self.sum_value / self.element_count
            second = self.sum_square / self.element_count
            variance = max(second - mean * mean, 0.0)

            std = math.sqrt(variance)
            mean_abs = self.sum_abs / self.element_count
            max_abs = self.max_abs
            nonzero_fraction = self.nonzero_count / self.element_count

        return ActivationSummary(
            split=self.split,
            module_name=self.module_name,
            module_class=self.module_class,
            tensor_count=int(self.tensor_count),
            element_count=int(self.element_count),
            mean=_safe_float(mean),
            std=_safe_float(std),
            mean_abs=_safe_float(mean_abs),
            max_abs=_safe_float(max_abs),
            nonzero_fraction=_safe_float(nonzero_fraction),
            shape_examples=list(self.shape_examples),
            runtime_seconds=float(time.perf_counter() - self.start_time),
        )


def _safe_float(value: Any) -> Optional[float]:
    """Return finite float or None."""
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


def _json_safe(value: Any) -> Any:
    """Recursively convert values to JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    return value


def save_json_safe(payload: Mapping[str, Any], output_path: Path | str, indent: int = 2) -> Path:
    """Save JSON safely."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(dict(payload)), file, indent=indent)

    return output_path


def _project_path(config: Mapping[str, Any], value: str) -> Path:
    """Resolve project-relative path."""
    return resolve_project_path(config, value)


def build_step14_diagnostics_config(config: Mapping[str, Any]) -> Step14DiagnosticsConfig:
    """Build Step14DiagnosticsConfig from project config."""
    diagnostic_splits = list(
        get_by_path(
            config,
            "experiments.step14.diagnostic_splits",
            get_by_path(config, "experiments.module_usage.diagnostic_splits", ["test", "external", "online"]),
        )
    )

    module_keywords = list(
        get_by_path(
            config,
            "experiments.step14.module_usage.module_keywords",
            get_by_path(config, "experiments.module_usage.module_keywords", DEFAULT_MODULE_KEYWORDS),
        )
    )

    return Step14DiagnosticsConfig(
        output_dir=str(
            get_by_path(
                config,
                "paths.module_usage_figures_dir",
                get_by_path(config, "experiments.step14.output_dir", "results/figures/module_usage"),
            )
        ),
        table_dir=str(
            get_by_path(
                config,
                "paths.tables_dir",
                "results/tables",
            )
        ),
        checkpoint_path=get_by_path(
            config,
            "experiments.step14.checkpoint_path",
            None,
        ),
        model_name=str(
            get_by_path(
                config,
                "experiments.step14.model_name",
                "Proposed",
            )
        ),
        variant_name=str(
            get_by_path(
                config,
                "experiments.step14.variant_name",
                "full",
            )
        ),
        diagnostic_splits=diagnostic_splits,
        module_keywords=module_keywords,
        max_modules=int(
            get_by_path(config, "experiments.step14.module_usage.max_modules", 80)
        ),
        max_batches_per_split=get_by_path(
            config,
            "experiments.step14.module_usage.max_batches_per_split",
            None,
        ),
        save_json=bool(get_by_path(config, "experiments.step14.save_json", True)),
        save_csv=bool(get_by_path(config, "experiments.step14.save_csv", True)),
        print_console_summary=bool(
            get_by_path(config, "experiments.step14.print_console_summary", True)
        ),
    )


def build_step14_paths(
    config: Mapping[str, Any],
    diagnostics_config: Optional[Step14DiagnosticsConfig] = None,
) -> Step14DiagnosticsPaths:
    """Build resolved Step-14 output paths."""
    if diagnostics_config is None:
        diagnostics_config = build_step14_diagnostics_config(config)

    output_dir = _project_path(config, diagnostics_config.output_dir)
    table_dir = _project_path(config, diagnostics_config.table_dir)

    ensure_dir(output_dir)
    ensure_dir(table_dir)

    return Step14DiagnosticsPaths(
        output_dir=output_dir,
        table_dir=table_dir,
        module_usage_json=output_dir / "module_activation_summary.json",
        module_usage_csv=output_dir / "module_activation_summary.csv",
        feature_importance_json=output_dir / "feature_importance_summary.json",
        feature_importance_csv=output_dir / "feature_importance_summary.csv",
        conductance_json=output_dir / "conductance_summary.json",
        conductance_csv=output_dir / "conductance_summary.csv",
        third_order_json=output_dir / "third_order_summary.json",
        third_order_csv=output_dir / "third_order_summary.csv",
        liquid_state_json=output_dir / "liquid_state_summary.json",
        liquid_state_csv=output_dir / "liquid_state_summary.csv",
        occlusion_json=output_dir / "occlusion_summary.json",
        occlusion_csv=output_dir / "occlusion_summary.csv",
        summary_json=output_dir / "step14_module_usage_diagnostics_summary.json",
    )


def get_step14_feature_columns(config: Mapping[str, Any]) -> List[str]:
    """Return the 9 final model input columns."""
    columns = list(
        get_by_path(
            config,
            "training.dataset.feature_columns",
            get_by_path(
                config,
                "model.input.recommended_model_input_columns",
                DEFAULT_STEP14_FEATURE_COLUMNS,
            ),
        )
    )

    if len(columns) != 9:
        raise ValueError(f"Step 14 expects exactly 9 xi input columns, got {len(columns)}.")

    return columns


def build_default_feature_groups(feature_columns: Sequence[str]) -> List[FeatureGroupSpec]:
    """Build canonical Step-14 feature groups."""
    columns = list(feature_columns)
    index_map = {name: index for index, name in enumerate(columns)}

    def idx(names: Sequence[str]) -> List[int]:
        missing = [name for name in names if name not in index_map]
        if missing:
            raise KeyError(f"Missing feature columns for Step-14 group: {missing}")
        return [index_map[name] for name in names]

    groups = [
        FeatureGroupSpec(
            name="eta",
            columns=["xi_eta_east_scaled", "xi_eta_north_scaled"],
            indices=idx(["xi_eta_east_scaled", "xi_eta_north_scaled"]),
            description="Instantaneous residual vector eta_t.",
        ),
        FeatureGroupSpec(
            name="eta_dot",
            columns=["xi_eta_dot_east_scaled", "xi_eta_dot_north_scaled"],
            indices=idx(["xi_eta_dot_east_scaled", "xi_eta_dot_north_scaled"]),
            description="First residual derivative eta_dot_t.",
        ),
        FeatureGroupSpec(
            name="eta_ddot",
            columns=["xi_eta_ddot_east_scaled", "xi_eta_ddot_north_scaled"],
            indices=idx(["xi_eta_ddot_east_scaled", "xi_eta_ddot_north_scaled"]),
            description="Second residual derivative eta_ddot_t.",
        ),
        FeatureGroupSpec(
            name="q",
            columns=["xi_q_scaled"],
            indices=idx(["xi_q_scaled"]),
            description="Mahalanobis residual energy q_t.",
        ),
        FeatureGroupSpec(
            name="accum_log",
            columns=["xi_accum_log_scaled"],
            indices=idx(["xi_accum_log_scaled"]),
            description="Weak evidence accumulation log feature.",
        ),
        FeatureGroupSpec(
            name="nu",
            columns=["xi_nu"],
            indices=idx(["xi_nu"]),
            description="Validity indicator nu_t used as an input feature only for diagnostics.",
        ),
        FeatureGroupSpec(
            name="instantaneous_residual_branch",
            columns=["xi_eta_east_scaled", "xi_eta_north_scaled", "xi_q_scaled"],
            indices=idx(["xi_eta_east_scaled", "xi_eta_north_scaled", "xi_q_scaled"]),
            description="Instantaneous branch: eta_t and q_t.",
        ),
        FeatureGroupSpec(
            name="residual_evolution_branch",
            columns=[
                "xi_eta_dot_east_scaled",
                "xi_eta_dot_north_scaled",
                "xi_eta_ddot_east_scaled",
                "xi_eta_ddot_north_scaled",
            ],
            indices=idx(
                [
                    "xi_eta_dot_east_scaled",
                    "xi_eta_dot_north_scaled",
                    "xi_eta_ddot_east_scaled",
                    "xi_eta_ddot_north_scaled",
                ]
            ),
            description="Residual evolution branch: eta_dot_t and eta_ddot_t.",
        ),
        FeatureGroupSpec(
            name="persistence_branch",
            columns=["xi_accum_log_scaled", "xi_nu"],
            indices=idx(["xi_accum_log_scaled", "xi_nu"]),
            description="Persistence/validity branch: accumulated evidence and nu_t.",
        ),
    ]

    return groups


def load_step14_context(
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[Any] = None,
) -> Step14DiagnosticsContext:
    """Load model, checkpoint, threshold, paths, and feature groups for Step 14."""
    diagnostics_config = build_step14_diagnostics_config(config)
    paths = build_step14_paths(config, diagnostics_config)

    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    model, checkpoint_path, checkpoint_metadata = load_trained_model_for_evaluation(
        config=config,
        checkpoint_path=diagnostics_config.checkpoint_path,
        device=device,
        variant_name=diagnostics_config.variant_name,
    )

    selected_threshold = load_selected_threshold(config=config)

    feature_columns = get_step14_feature_columns(config)
    feature_groups = build_default_feature_groups(feature_columns)

    return Step14DiagnosticsContext(
        config=config,
        diagnostics_config=diagnostics_config,
        paths=paths,
        active_seed=int(active_seed),
        device=device,
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        selected_threshold=selected_threshold,
        feature_columns=feature_columns,
        feature_groups=feature_groups,
    )


def _extract_tensors_from_output(output: Any) -> List[Tensor]:
    """Extract tensors recursively from module output."""
    tensors: List[Tensor] = []

    if torch.is_tensor(output):
        tensors.append(output)
        return tensors

    if isinstance(output, Mapping):
        for value in output.values():
            tensors.extend(_extract_tensors_from_output(value))
        return tensors

    if isinstance(output, (list, tuple)):
        for value in output:
            tensors.extend(_extract_tensors_from_output(value))
        return tensors

    if is_dataclass(output):
        tensors.extend(_extract_tensors_from_output(asdict(output)))
        return tensors

    if hasattr(output, "__dict__"):
        try:
            tensors.extend(_extract_tensors_from_output(vars(output)))
        except Exception:
            pass

    return tensors


def select_modules_for_hooks(
    model: nn.Module,
    keywords: Sequence[str],
    max_modules: int = 80,
) -> List[Tuple[str, nn.Module]]:
    """
    Select modules to inspect with forward hooks.

    Uses name keywords so the function remains robust across small architecture changes.
    """
    keyword_set = [str(item).lower() for item in keywords]

    selected: List[Tuple[str, nn.Module]] = []
    seen_ids = set()

    for name, module in model.named_modules():
        if name == "":
            continue

        lower_name = name.lower()
        class_name = module.__class__.__name__.lower()

        matched = any(keyword in lower_name or keyword in class_name for keyword in keyword_set)

        if not matched:
            continue

        if id(module) in seen_ids:
            continue

        selected.append((name, module))
        seen_ids.add(id(module))

        if len(selected) >= int(max_modules):
            break

    return selected


def collect_module_activation_summaries(
    context: Step14DiagnosticsContext,
    split_name: str = "test",
) -> List[ActivationSummary]:
    """
    Collect module activation summaries for one split using forward hooks.
    """
    loader, dataset = build_evaluation_dataloader(
        config=context.config,
        split_name=split_name,
        active_seed=context.active_seed,
        full_sequence=(split_name == "online"),
    )

    modules = select_modules_for_hooks(
        model=context.model,
        keywords=context.diagnostics_config.module_keywords,
        max_modules=context.diagnostics_config.max_modules,
    )

    if not modules:
        raise RuntimeError(
            "No modules were selected for Step-14 module-usage diagnostics. "
            "Check experiments.step14.module_usage.module_keywords."
        )

    accumulators: Dict[str, ActivationAccumulator] = {}
    handles = []

    for module_name, module in modules:
        accumulators[module_name] = ActivationAccumulator(
            split=split_name,
            module_name=module_name,
            module_class=module.__class__.__name__,
        )

        def make_hook(name: str) -> Callable[[nn.Module, Tuple[Any, ...], Any], None]:
            def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                tensors = _extract_tensors_from_output(output)
                for tensor in tensors:
                    accumulators[name].update(tensor)

            return hook

        handles.append(module.register_forward_hook(make_hook(module_name)))

    context.model.eval()

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                batch = move_to_device(batch, context.device)
                _ = context.model(batch)

                max_batches = context.diagnostics_config.max_batches_per_split
                if max_batches is not None and batch_index >= int(max_batches):
                    break
    finally:
        for handle in handles:
            handle.remove()

    summaries = [acc.summary() for acc in accumulators.values()]

    print("=" * 100)
    print(f"STEP 14 MODULE ACTIVATION DIAGNOSTICS | split={split_name}")
    print("=" * 100)
    print(f"Rows/windows       : {dataset.summary()['rows']} / {dataset.summary()['windows']}")
    print(f"Hooked modules     : {len(modules)}")
    print(f"Selected theta/N_p : {context.selected_threshold.theta} / {context.selected_threshold.persistence}")
    print("=" * 100)

    for item in summaries[:20]:
        print(
            f"{item.module_name:<45} | "
            f"{item.module_class:<24} | "
            f"mean_abs={item.mean_abs} | "
            f"std={item.std} | "
            f"nonzero={item.nonzero_fraction}"
        )

    if len(summaries) > 20:
        print(f"... {len(summaries) - 20} more modules saved to file.")

    print("=" * 100)

    return summaries


def save_module_activation_summaries(
    context: Step14DiagnosticsContext,
    summaries: Sequence[ActivationSummary],
) -> Dict[str, str]:
    """Save module activation summaries to CSV and JSON."""
    rows = [summary.to_dict() for summary in summaries]

    output_paths: Dict[str, str] = {}

    if context.diagnostics_config.save_csv:
        ensure_dir(context.paths.module_usage_csv.parent)
        pd.DataFrame(rows).to_csv(context.paths.module_usage_csv, index=False)
        output_paths["module_usage_csv"] = str(context.paths.module_usage_csv)

    if context.diagnostics_config.save_json:
        save_json_safe(
            {
                "context": context.to_dict(),
                "module_activation_summaries": rows,
            },
            context.paths.module_usage_json,
        )
        output_paths["module_usage_json"] = str(context.paths.module_usage_json)

    return output_paths


def run_module_usage_diagnostics(
    config: Mapping[str, Any],
    active_seed: int = 42,
    split_name: str = "test",
    context: Optional[Step14DiagnosticsContext] = None,
) -> Dict[str, Any]:
    """
    Run module activation diagnostics for one split.

    This is one component of Step 14.
    """
    if context is None:
        context = load_step14_context(config=config, active_seed=active_seed)

    summaries = collect_module_activation_summaries(
        context=context,
        split_name=split_name,
    )

    artifact_paths = save_module_activation_summaries(
        context=context,
        summaries=summaries,
    )

    payload = {
        "status": "PASSED",
        "split_name": split_name,
        "summary_count": len(summaries),
        "artifact_paths": artifact_paths,
        "context": context.to_dict(),
    }

    return payload


__all__ = [
    "DEFAULT_STEP14_FEATURE_COLUMNS",
    "DEFAULT_MODULE_KEYWORDS",
    "FeatureGroupSpec",
    "Step14DiagnosticsConfig",
    "Step14DiagnosticsPaths",
    "Step14DiagnosticsContext",
    "ActivationSummary",
    "ActivationAccumulator",
    "save_json_safe",
    "build_step14_diagnostics_config",
    "build_step14_paths",
    "get_step14_feature_columns",
    "build_default_feature_groups",
    "load_step14_context",
    "select_modules_for_hooks",
    "collect_module_activation_summaries",
    "save_module_activation_summaries",
    "run_module_usage_diagnostics",
]