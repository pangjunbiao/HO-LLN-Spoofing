"""
Conductance and Kirchhoff-exchange diagnostics for the trained full proposed model.

Step 14 purpose:
- Inspect whether the Kirchhoff/conductance/exchange part of the full model is active.
- Summarize conductance-like tensors if exposed by the model.
- Summarize exchange-like tensors if exposed by the model.
- Compare activation magnitudes on normal vs attack rows where tensor shape allows it.

Important:
- This is diagnostic only.
- This does not retrain the model.
- This does not replace official ablations.
- Official ablation no_kirchhoff_exchange still must be retrained from scratch later.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.diagnostics.module_usage import (
    Step14DiagnosticsContext,
    load_step14_context,
    save_json_safe,
)
from src.evaluation.evaluate_dataset1 import build_evaluation_dataloader
from src.utils.config import get_by_path
from src.utils.device import move_to_device
from src.utils.io import ensure_dir


DEFAULT_CONDUCTANCE_KEYWORDS = [
    "kirchhoff",
    "conductance",
    "exchange",
    "cie",
    "cip",
    "cep",
]


@dataclass
class TensorDiagnosticSummary:
    """Summary for one captured diagnostic tensor."""

    split: str
    module_name: str
    module_class: str
    tensor_name: str
    tensor_role: str

    tensor_count: int
    element_count: int
    shape_examples: List[str]

    mean: Optional[float]
    std: Optional[float]
    mean_abs: Optional[float]
    max_abs: Optional[float]
    nonzero_fraction: Optional[float]

    attack_mean_abs: Optional[float]
    normal_mean_abs: Optional[float]
    attack_minus_normal_mean_abs: Optional[float]

    runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TensorDiagnosticAccumulator:
    """Streaming accumulator for tensor diagnostics."""

    def __init__(
        self,
        split: str,
        module_name: str,
        module_class: str,
        tensor_name: str,
        tensor_role: str,
    ) -> None:
        self.split = str(split)
        self.module_name = str(module_name)
        self.module_class = str(module_class)
        self.tensor_name = str(tensor_name)
        self.tensor_role = str(tensor_role)

        self.tensor_count = 0
        self.element_count = 0

        self.sum_value = 0.0
        self.sum_square = 0.0
        self.sum_abs = 0.0
        self.max_abs = 0.0
        self.nonzero_count = 0

        self.attack_abs_sum = 0.0
        self.attack_count = 0
        self.normal_abs_sum = 0.0
        self.normal_count = 0

        self.shape_examples: List[str] = []
        self.start_time = time.perf_counter()

    def update(
        self,
        tensor: Tensor,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> None:
        """Update accumulator with one tensor."""
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

        self._update_attack_normal(arr=arr, labels=labels, mask=mask)

    def _update_attack_normal(
        self,
        arr: Tensor,
        labels: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> None:
        """
        Update attack/normal mean-absolute statistics when shape is alignable.

        Supported:
        - arr shape [B, T, D]
        - arr shape [B, T]
        - arr shape [B, T, ...]
        """
        if labels is None or mask is None:
            return

        if not torch.is_tensor(labels) or not torch.is_tensor(mask):
            return

        labels_cpu = labels.detach().cpu().long()
        mask_cpu = mask.detach().cpu().float()

        if labels_cpu.ndim != 2 or mask_cpu.ndim != 2:
            return

        if arr.ndim < 2:
            return

        if arr.shape[0] != labels_cpu.shape[0] or arr.shape[1] != labels_cpu.shape[1]:
            return

        abs_arr = arr.abs()

        if abs_arr.ndim > 2:
            # Reduce feature/channel dimensions so each time-step has one magnitude.
            reduce_dims = tuple(range(2, abs_arr.ndim))
            abs_timestep = abs_arr.mean(dim=reduce_dims)
        else:
            abs_timestep = abs_arr

        valid = mask_cpu > 0.5
        attack = (labels_cpu == 1) & valid
        normal = (labels_cpu == 0) & valid

        if attack.any():
            self.attack_abs_sum += float(abs_timestep[attack].sum().item())
            self.attack_count += int(attack.sum().item())

        if normal.any():
            self.normal_abs_sum += float(abs_timestep[normal].sum().item())
            self.normal_count += int(normal.sum().item())

    def summary(self) -> TensorDiagnosticSummary:
        """Return summary payload."""
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

        attack_mean_abs = (
            self.attack_abs_sum / self.attack_count
            if self.attack_count > 0
            else None
        )
        normal_mean_abs = (
            self.normal_abs_sum / self.normal_count
            if self.normal_count > 0
            else None
        )

        if attack_mean_abs is None or normal_mean_abs is None:
            attack_minus_normal = None
        else:
            attack_minus_normal = attack_mean_abs - normal_mean_abs

        return TensorDiagnosticSummary(
            split=self.split,
            module_name=self.module_name,
            module_class=self.module_class,
            tensor_name=self.tensor_name,
            tensor_role=self.tensor_role,
            tensor_count=int(self.tensor_count),
            element_count=int(self.element_count),
            shape_examples=list(self.shape_examples),
            mean=_safe_float(mean),
            std=_safe_float(std),
            mean_abs=_safe_float(mean_abs),
            max_abs=_safe_float(max_abs),
            nonzero_fraction=_safe_float(nonzero_fraction),
            attack_mean_abs=_safe_float(attack_mean_abs),
            normal_mean_abs=_safe_float(normal_mean_abs),
            attack_minus_normal_mean_abs=_safe_float(attack_minus_normal),
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


def _infer_tensor_role(module_name: str, tensor_name: str) -> str:
    """Infer diagnostic tensor role from name."""
    text = f"{module_name}.{tensor_name}".lower()

    if "cie" in text or "c_ie" in text or "instant" in text:
        return "C_IE_or_instantaneous_exchange"

    if "cip" in text or "c_ip" in text or "persist" in text:
        return "C_IP_or_persistence_exchange"

    if "cep" in text or "c_ep" in text or "evolution" in text:
        return "C_EP_or_evolution_exchange"

    if "conductance" in text or "gate" in text or "alpha" in text:
        return "conductance_or_gate"

    if "exchange" in text or "kirchhoff" in text or "flow" in text:
        return "exchange_magnitude"

    return "conductance_exchange_related"


def _extract_named_tensors(output: Any, prefix: str = "output") -> List[Tuple[str, Tensor]]:
    """Recursively extract named tensors from module output."""
    tensors: List[Tuple[str, Tensor]] = []

    if torch.is_tensor(output):
        tensors.append((prefix, output))
        return tensors

    if isinstance(output, Mapping):
        for key, value in output.items():
            tensors.extend(_extract_named_tensors(value, f"{prefix}.{key}"))
        return tensors

    if isinstance(output, (list, tuple)):
        for index, value in enumerate(output):
            tensors.extend(_extract_named_tensors(value, f"{prefix}.{index}"))
        return tensors

    if is_dataclass(output):
        tensors.extend(_extract_named_tensors(asdict(output), prefix))
        return tensors

    if hasattr(output, "__dict__"):
        try:
            tensors.extend(_extract_named_tensors(vars(output), prefix))
        except Exception:
            pass

    return tensors


def select_conductance_modules(
    model: nn.Module,
    keywords: Sequence[str],
    max_modules: int = 60,
) -> List[Tuple[str, nn.Module]]:
    """Select conductance/Kirchhoff/exchange modules by name/class keyword."""
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


def collect_conductance_summaries_for_split(
    context: Step14DiagnosticsContext,
    split_name: str = "test",
) -> List[TensorDiagnosticSummary]:
    """
    Collect conductance/exchange summaries for one split.
    """
    keywords = list(
        get_by_path(
            context.config,
            "experiments.step14.conductance_analysis.module_keywords",
            DEFAULT_CONDUCTANCE_KEYWORDS,
        )
    )
    max_modules = int(
        get_by_path(
            context.config,
            "experiments.step14.conductance_analysis.max_modules",
            60,
        )
    )

    loader, dataset = build_evaluation_dataloader(
        config=context.config,
        split_name=split_name,
        active_seed=context.active_seed,
        full_sequence=(split_name == "online"),
    )

    modules = select_conductance_modules(
        model=context.model,
        keywords=keywords,
        max_modules=max_modules,
    )

    if not modules:
        raise RuntimeError(
            "No conductance/Kirchhoff/exchange modules were found. "
            "Check module names or experiments.step14.conductance_analysis.module_keywords."
        )

    accumulators: Dict[str, TensorDiagnosticAccumulator] = {}
    handles = []
    current_batch: Dict[str, Optional[Tensor]] = {
        "labels": None,
        "mask": None,
    }

    def get_accumulator(
        module_name: str,
        module_class: str,
        tensor_name: str,
    ) -> TensorDiagnosticAccumulator:
        key = f"{module_name}::{tensor_name}"

        if key not in accumulators:
            accumulators[key] = TensorDiagnosticAccumulator(
                split=split_name,
                module_name=module_name,
                module_class=module_class,
                tensor_name=tensor_name,
                tensor_role=_infer_tensor_role(module_name, tensor_name),
            )

        return accumulators[key]

    for module_name, module in modules:
        module_class = module.__class__.__name__

        def make_hook(name: str, cls_name: str):
            def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                tensors = _extract_named_tensors(output)

                for tensor_name, tensor in tensors:
                    accumulator = get_accumulator(
                        module_name=name,
                        module_class=cls_name,
                        tensor_name=tensor_name,
                    )
                    accumulator.update(
                        tensor=tensor,
                        labels=current_batch["labels"],
                        mask=current_batch["mask"],
                    )

            return hook

        handles.append(module.register_forward_hook(make_hook(module_name, module_class)))

    context.model.eval()

    try:
        with torch.no_grad():
            for batch in loader:
                batch = move_to_device(batch, context.device)

                current_batch["labels"] = batch["y"].detach()
                current_batch["mask"] = (batch["loss_mask"] * batch["padding_mask"]).detach()

                _ = context.model(batch)
    finally:
        for handle in handles:
            handle.remove()

    summaries = [acc.summary() for acc in accumulators.values()]

    print("=" * 100)
    print(f"STEP 14 CONDUCTANCE / KIRCHHOFF EXCHANGE ANALYSIS | split={split_name}")
    print("=" * 100)
    print(f"Rows/windows       : {dataset.summary()['rows']} / {dataset.summary()['windows']}")
    print(f"Selected modules   : {len(modules)}")
    print(f"Captured tensors   : {len(summaries)}")
    print(f"Selected theta/N_p : {context.selected_threshold.theta} / {context.selected_threshold.persistence}")
    print("=" * 100)

    sorted_summaries = sorted(
        summaries,
        key=lambda item: item.mean_abs if item.mean_abs is not None else -1.0,
        reverse=True,
    )

    for item in sorted_summaries[:20]:
        print(
            f"{item.module_name:<42} | "
            f"{item.tensor_role:<32} | "
            f"mean_abs={item.mean_abs} | "
            f"attack-normal={item.attack_minus_normal_mean_abs} | "
            f"nonzero={item.nonzero_fraction}"
        )

    if len(sorted_summaries) > 20:
        print(f"... {len(sorted_summaries) - 20} more conductance/exchange tensors saved.")

    print("=" * 100)

    return summaries


def save_conductance_summaries(
    context: Step14DiagnosticsContext,
    summaries: Sequence[TensorDiagnosticSummary],
) -> Dict[str, str]:
    """Save conductance/exchange summaries."""
    rows = [summary.to_dict() for summary in summaries]

    output_paths: Dict[str, str] = {}

    if context.diagnostics_config.save_csv:
        ensure_dir(context.paths.conductance_csv.parent)
        pd.DataFrame(rows).to_csv(context.paths.conductance_csv, index=False)
        output_paths["conductance_csv"] = str(context.paths.conductance_csv)

    if context.diagnostics_config.save_json:
        save_json_safe(
            {
                "context": context.to_dict(),
                "results": rows,
                "interpretation_note": (
                    "Conductance/exchange summaries are diagnostic forward-hook statistics. "
                    "They show whether Kirchhoff-related modules produce non-trivial tensors, "
                    "but they are not a substitute for retrained ablation."
                ),
            },
            context.paths.conductance_json,
        )
        output_paths["conductance_json"] = str(context.paths.conductance_json)

    return output_paths


def run_conductance_analysis(
    config: Mapping[str, Any],
    active_seed: int = 42,
    context: Optional[Step14DiagnosticsContext] = None,
) -> Dict[str, Any]:
    """Run conductance/Kirchhoff-exchange diagnostics for configured splits."""
    if context is None:
        context = load_step14_context(config=config, active_seed=active_seed)

    all_summaries: List[TensorDiagnosticSummary] = []

    for split_name in context.diagnostics_config.diagnostic_splits:
        split_summaries = collect_conductance_summaries_for_split(
            context=context,
            split_name=str(split_name),
        )
        all_summaries.extend(split_summaries)

    artifact_paths = save_conductance_summaries(
        context=context,
        summaries=all_summaries,
    )

    return {
        "status": "PASSED",
        "result_count": len(all_summaries),
        "diagnostic_splits": list(context.diagnostics_config.diagnostic_splits),
        "artifact_paths": artifact_paths,
    }


__all__ = [
    "DEFAULT_CONDUCTANCE_KEYWORDS",
    "TensorDiagnosticSummary",
    "TensorDiagnosticAccumulator",
    "select_conductance_modules",
    "collect_conductance_summaries_for_split",
    "save_conductance_summaries",
    "run_conductance_analysis",
]