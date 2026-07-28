"""
Optimizer utilities for Step 12 training.

Purpose:
- build optimizers consistently for full model and ablations,
- support AdamW / Adam / SGD / RMSprop,
- apply weight decay correctly,
- optionally exclude bias and normalization parameters from weight decay,
- support gradient clipping,
- expose JSON-safe optimizer summaries.

This module does not train by itself. The trainer uses it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math

import torch
from torch import Tensor, nn


@dataclass
class OptimizerConfig:
    """Configuration for optimizer construction."""

    name: str = "adamw"

    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8

    momentum: float = 0.9
    nesterov: bool = False

    rmsprop_alpha: float = 0.99

    exclude_bias_and_norm_from_weight_decay: bool = True
    no_decay_keywords: Tuple[str, ...] = (
        "bias",
        "norm",
        "layernorm",
        "layer_norm",
        "batchnorm",
        "batch_norm",
        "bn",
    )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["betas"] = list(self.betas)
        payload["no_decay_keywords"] = list(self.no_decay_keywords)
        return payload


@dataclass
class GradientClippingConfig:
    """Configuration for gradient clipping."""

    enabled: bool = True
    max_norm: float = 1.0
    norm_type: float = 2.0
    error_if_nonfinite: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerBuildSummary:
    """JSON-safe optimizer build summary."""

    optimizer_name: str
    learning_rate: float
    weight_decay: float

    trainable_parameter_count: int
    total_parameter_count: int

    parameter_group_count: int
    decayed_parameter_count: int
    no_decay_parameter_count: int

    exclude_bias_and_norm_from_weight_decay: bool

    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GradientClipResult:
    """Result of one gradient-clipping operation."""

    enabled: bool
    total_norm_before_clip: Optional[float]
    max_norm: Optional[float]
    clipped: bool
    finite: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_by_path(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Small local config helper."""
    current: Any = config

    for key in path.split("."):
        if not isinstance(current, Mapping):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def _as_tuple_float(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    """Convert config value to pair of floats."""
    if value is None:
        return default

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])

    raise ValueError(f"Expected pair of floats, got {value!r}.")


def build_optimizer_config(config: Optional[Mapping[str, Any]] = None) -> OptimizerConfig:
    """Build OptimizerConfig from project config."""
    if config is None:
        return OptimizerConfig()

    return OptimizerConfig(
        name=str(_get_by_path(config, "training.optimizer.name", "adamw")),
        learning_rate=float(_get_by_path(config, "training.optimizer.learning_rate", 1.0e-3)),
        weight_decay=float(_get_by_path(config, "training.optimizer.weight_decay", 1.0e-4)),
        betas=_as_tuple_float(
            _get_by_path(config, "training.optimizer.betas", [0.9, 0.999]),
            default=(0.9, 0.999),
        ),
        eps=float(_get_by_path(config, "training.optimizer.eps", 1.0e-8)),
        momentum=float(_get_by_path(config, "training.optimizer.momentum", 0.9)),
        nesterov=bool(_get_by_path(config, "training.optimizer.nesterov", False)),
        rmsprop_alpha=float(_get_by_path(config, "training.optimizer.rmsprop_alpha", 0.99)),
        exclude_bias_and_norm_from_weight_decay=bool(
            _get_by_path(
                config,
                "training.optimizer.exclude_bias_and_norm_from_weight_decay",
                True,
            )
        ),
        no_decay_keywords=tuple(
            str(item).lower()
            for item in _get_by_path(
                config,
                "training.optimizer.no_decay_keywords",
                [
                    "bias",
                    "norm",
                    "layernorm",
                    "layer_norm",
                    "batchnorm",
                    "batch_norm",
                    "bn",
                ],
            )
        ),
    )


def build_gradient_clipping_config(
    config: Optional[Mapping[str, Any]] = None,
) -> GradientClippingConfig:
    """Build GradientClippingConfig from project config."""
    if config is None:
        return GradientClippingConfig()

    return GradientClippingConfig(
        enabled=bool(_get_by_path(config, "training.gradient_clipping.enabled", True)),
        max_norm=float(_get_by_path(config, "training.gradient_clipping.max_norm", 1.0)),
        norm_type=float(_get_by_path(config, "training.gradient_clipping.norm_type", 2.0)),
        error_if_nonfinite=bool(
            _get_by_path(config, "training.gradient_clipping.error_if_nonfinite", False)
        ),
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    return {
        "total_parameter_count": int(total),
        "trainable_parameter_count": int(trainable),
        "frozen_parameter_count": int(total - trainable),
    }


def _is_no_decay_parameter(
    name: str,
    parameter: nn.Parameter,
    no_decay_keywords: Sequence[str],
) -> bool:
    """Decide whether a parameter should be excluded from weight decay."""
    if parameter.ndim <= 1:
        return True

    lower_name = str(name).lower()
    return any(keyword in lower_name for keyword in no_decay_keywords)


def split_weight_decay_parameter_groups(
    model: nn.Module,
    weight_decay: float,
    no_decay_keywords: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Split trainable parameters into decay/no-decay groups.

    Biases and norm parameters should usually not receive weight decay.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    decay_count = 0
    no_decay_count = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if _is_no_decay_parameter(name, parameter, no_decay_keywords):
            no_decay_params.append(parameter)
            no_decay_count += int(parameter.numel())
        else:
            decay_params.append(parameter)
            decay_count += int(parameter.numel())

    groups: List[Dict[str, Any]] = []

    if decay_params:
        groups.append(
            {
                "params": decay_params,
                "weight_decay": float(weight_decay),
                "group_name": "decay",
            }
        )

    if no_decay_params:
        groups.append(
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "group_name": "no_decay",
            }
        )

    stats = {
        "decayed_parameter_count": int(decay_count),
        "no_decay_parameter_count": int(no_decay_count),
        "parameter_group_count": int(len(groups)),
    }

    return groups, stats


def get_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return trainable parameters."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def create_optimizer(
    model: nn.Module,
    config: Optional[OptimizerConfig] = None,
) -> Tuple[torch.optim.Optimizer, OptimizerBuildSummary]:
    """
    Create optimizer for model.

    Returns:
        optimizer, summary
    """
    cfg = config or OptimizerConfig()

    if cfg.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    if cfg.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")

    counts = count_parameters(model)

    if counts["trainable_parameter_count"] <= 0:
        raise ValueError("Model has no trainable parameters.")

    name = str(cfg.name).lower().strip()

    if cfg.exclude_bias_and_norm_from_weight_decay and cfg.weight_decay > 0.0:
        params, group_stats = split_weight_decay_parameter_groups(
            model=model,
            weight_decay=cfg.weight_decay,
            no_decay_keywords=cfg.no_decay_keywords,
        )
    else:
        params = [
            {
                "params": get_trainable_parameters(model),
                "weight_decay": float(cfg.weight_decay),
                "group_name": "all_trainable",
            }
        ]
        group_stats = {
            "decayed_parameter_count": counts["trainable_parameter_count"],
            "no_decay_parameter_count": 0,
            "parameter_group_count": 1,
        }

    if name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=float(cfg.learning_rate),
            betas=tuple(cfg.betas),
            eps=float(cfg.eps),
        )

    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=float(cfg.learning_rate),
            betas=tuple(cfg.betas),
            eps=float(cfg.eps),
        )

    elif name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=float(cfg.learning_rate),
            momentum=float(cfg.momentum),
            nesterov=bool(cfg.nesterov),
        )

    elif name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params,
            lr=float(cfg.learning_rate),
            alpha=float(cfg.rmsprop_alpha),
            eps=float(cfg.eps),
            momentum=float(cfg.momentum),
        )

    else:
        raise ValueError(
            f"Unknown optimizer name='{cfg.name}'. "
            "Supported: adamw, adam, sgd, rmsprop."
        )

    summary = OptimizerBuildSummary(
        optimizer_name=name,
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        trainable_parameter_count=int(counts["trainable_parameter_count"]),
        total_parameter_count=int(counts["total_parameter_count"]),
        parameter_group_count=int(group_stats["parameter_group_count"]),
        decayed_parameter_count=int(group_stats["decayed_parameter_count"]),
        no_decay_parameter_count=int(group_stats["no_decay_parameter_count"]),
        exclude_bias_and_norm_from_weight_decay=bool(
            cfg.exclude_bias_and_norm_from_weight_decay
        ),
        status="PASSED",
    )

    return optimizer, summary


def create_optimizer_from_project_config(
    model: nn.Module,
    config: Mapping[str, Any],
) -> Tuple[torch.optim.Optimizer, OptimizerBuildSummary]:
    """Create optimizer from full project config."""
    return create_optimizer(
        model=model,
        config=build_optimizer_config(config),
    )


def get_current_lrs(optimizer: torch.optim.Optimizer) -> List[float]:
    """Return current learning rates for all parameter groups."""
    return [float(group.get("lr", 0.0)) for group in optimizer.param_groups]


def set_optimizer_lrs(optimizer: torch.optim.Optimizer, learning_rates: Sequence[float]) -> None:
    """Set optimizer learning rates for parameter groups."""
    if len(learning_rates) != len(optimizer.param_groups):
        raise ValueError(
            f"learning_rates length mismatch. Expected {len(optimizer.param_groups)}, "
            f"got {len(learning_rates)}."
        )

    for group, lr in zip(optimizer.param_groups, learning_rates):
        group["lr"] = float(lr)


def compute_gradient_norm(
    parameters: Iterable[nn.Parameter],
    norm_type: float = 2.0,
) -> Optional[float]:
    """Compute total gradient norm."""
    grads = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]

    if not grads:
        return None

    norm_type = float(norm_type)

    if math.isinf(norm_type):
        total_norm = max(float(grad.abs().max().item()) for grad in grads)
    else:
        norms = torch.stack(
            [torch.linalg.vector_norm(grad, ord=norm_type) for grad in grads]
        )
        total_norm = float(torch.linalg.vector_norm(norms, ord=norm_type).item())

    return total_norm


def clip_gradients(
    model: nn.Module,
    config: Optional[GradientClippingConfig] = None,
) -> GradientClipResult:
    """
    Clip gradients if enabled.

    Call after loss.backward() and before optimizer.step().
    """
    cfg = config or GradientClippingConfig()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if not cfg.enabled:
        norm = compute_gradient_norm(parameters, norm_type=cfg.norm_type)
        finite = True if norm is None else math.isfinite(float(norm))

        return GradientClipResult(
            enabled=False,
            total_norm_before_clip=None if norm is None else float(norm),
            max_norm=None,
            clipped=False,
            finite=bool(finite),
        )

    if cfg.max_norm <= 0.0:
        raise ValueError("Gradient clipping max_norm must be positive.")

    total_norm_tensor = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=float(cfg.max_norm),
        norm_type=float(cfg.norm_type),
        error_if_nonfinite=bool(cfg.error_if_nonfinite),
    )

    total_norm = float(total_norm_tensor.detach().cpu().item())
    finite = math.isfinite(total_norm)
    clipped = finite and total_norm > float(cfg.max_norm)

    return GradientClipResult(
        enabled=True,
        total_norm_before_clip=total_norm,
        max_norm=float(cfg.max_norm),
        clipped=bool(clipped),
        finite=bool(finite),
    )


def optimizer_state_summary(
    optimizer: torch.optim.Optimizer,
    summary: Optional[OptimizerBuildSummary] = None,
) -> Dict[str, Any]:
    """Return JSON-safe optimizer state summary."""
    payload: Dict[str, Any] = {
        "optimizer_class": optimizer.__class__.__name__,
        "parameter_group_count": int(len(optimizer.param_groups)),
        "learning_rates": get_current_lrs(optimizer),
        "group_weight_decays": [
            float(group.get("weight_decay", 0.0))
            for group in optimizer.param_groups
        ],
        "group_names": [
            str(group.get("group_name", f"group_{idx}"))
            for idx, group in enumerate(optimizer.param_groups)
        ],
    }

    if summary is not None:
        payload["build_summary"] = summary.to_dict()

    return payload


__all__ = [
    "OptimizerConfig",
    "GradientClippingConfig",
    "OptimizerBuildSummary",
    "GradientClipResult",
    "build_optimizer_config",
    "build_gradient_clipping_config",
    "count_parameters",
    "split_weight_decay_parameter_groups",
    "get_trainable_parameters",
    "create_optimizer",
    "create_optimizer_from_project_config",
    "get_current_lrs",
    "set_optimizer_lrs",
    "compute_gradient_norm",
    "clip_gradients",
    "optimizer_state_summary",
]