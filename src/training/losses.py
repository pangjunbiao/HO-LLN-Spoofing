"""
Training losses for causal GNSS spoofing detection.

Step 12:
- weighted binary cross entropy,
- class weights computed from TRAIN split only,
- loss masking for valid xi rows,
- same loss protocol for full model and ablations.

Canonical training objective:

    L_det = -1/|I_tr| sum_t [
        alpha_1 y_t log(p_t + eps)
        + alpha_0 (1-y_t) log(1-p_t + eps)
    ]

Implementation detail:
- We train from logits using BCEWithLogits for numerical stability.
- Class weights alpha_0 and alpha_1 are sample weights.
- loss_mask is usually xi_nu and padding_mask combined.
- Thresholding is not performed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


EPS = 1e-12


@dataclass
class ClassWeightConfig:
    """Configuration for class-weight computation."""

    enabled: bool = True
    strategy: str = "balanced"  # balanced, none, manual
    normal_label: int = 0
    attack_label: int = 1

    manual_alpha_0: Optional[float] = None
    manual_alpha_1: Optional[float] = None

    min_weight: float = 0.05
    max_weight: float = 50.0

    train_only: bool = True
    valid_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassWeightSummary:
    """Summary of computed class weights."""

    strategy: str

    normal_count: int
    attack_count: int
    total_count: int

    normal_fraction: float
    attack_fraction: float

    alpha_0: float
    alpha_1: float

    train_only: bool
    valid_only: bool

    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeightedBCELossConfig:
    """Configuration for weighted BCE loss."""

    enabled: bool = True
    from_logits: bool = True

    epsilon: float = 1e-12
    reduction: str = "mean"

    normalization: str = "valid_count"  # valid_count or weight_sum

    use_loss_mask: bool = True
    combine_padding_and_loss_mask: bool = True

    alpha_0: float = 1.0
    alpha_1: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LossComputationResult:
    """Container for loss computation details."""

    loss: Tensor
    unweighted_loss: Tensor
    weighted_loss_sum: Tensor

    valid_count: int
    positive_count: int
    negative_count: int

    alpha_0: float
    alpha_1: float

    from_logits: bool
    normalization: str

    def scalar_summary(self) -> Dict[str, Any]:
        return {
            "loss": float(self.loss.detach().cpu().item()),
            "unweighted_loss": float(self.unweighted_loss.detach().cpu().item()),
            "weighted_loss_sum": float(self.weighted_loss_sum.detach().cpu().item()),
            "valid_count": int(self.valid_count),
            "positive_count": int(self.positive_count),
            "negative_count": int(self.negative_count),
            "alpha_0": float(self.alpha_0),
            "alpha_1": float(self.alpha_1),
            "from_logits": bool(self.from_logits),
            "normalization": str(self.normalization),
        }


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


def build_class_weight_config(config: Optional[Mapping[str, Any]] = None) -> ClassWeightConfig:
    """Build class-weight config from project config."""
    if config is None:
        return ClassWeightConfig()

    return ClassWeightConfig(
        enabled=bool(_get_by_path(config, "training.loss.class_weights.enabled", True)),
        strategy=str(_get_by_path(config, "training.loss.class_weights.strategy", "balanced")),
        normal_label=int(_get_by_path(config, "training.loss.class_weights.normal_label", 0)),
        attack_label=int(_get_by_path(config, "training.loss.class_weights.attack_label", 1)),
        manual_alpha_0=_get_by_path(config, "training.loss.class_weights.manual_alpha_0", None),
        manual_alpha_1=_get_by_path(config, "training.loss.class_weights.manual_alpha_1", None),
        min_weight=float(_get_by_path(config, "training.loss.class_weights.min_weight", 0.05)),
        max_weight=float(_get_by_path(config, "training.loss.class_weights.max_weight", 50.0)),
        train_only=bool(_get_by_path(config, "training.loss.class_weights.train_only", True)),
        valid_only=bool(_get_by_path(config, "training.loss.class_weights.valid_only", True)),
    )


def build_weighted_bce_config(
    config: Optional[Mapping[str, Any]] = None,
    alpha_0: float = 1.0,
    alpha_1: float = 1.0,
) -> WeightedBCELossConfig:
    """Build weighted BCE config from project config."""
    if config is None:
        return WeightedBCELossConfig(alpha_0=float(alpha_0), alpha_1=float(alpha_1))

    return WeightedBCELossConfig(
        enabled=bool(_get_by_path(config, "training.loss.enabled", True)),
        from_logits=bool(_get_by_path(config, "training.loss.from_logits", True)),
        epsilon=float(_get_by_path(config, "training.loss.epsilon", 1e-12)),
        reduction=str(_get_by_path(config, "training.loss.reduction", "mean")),
        normalization=str(_get_by_path(config, "training.loss.normalization", "valid_count")),
        use_loss_mask=bool(_get_by_path(config, "training.loss.use_loss_mask", True)),
        combine_padding_and_loss_mask=bool(
            _get_by_path(config, "training.loss.combine_padding_and_loss_mask", True)
        ),
        alpha_0=float(alpha_0),
        alpha_1=float(alpha_1),
    )


def labels_to_binary_tensor(
    labels: Union[Tensor, np.ndarray, Sequence[Any]],
    device: Optional[torch.device | str] = None,
) -> Tensor:
    """
    Convert labels to binary float tensor.

    Numeric:
        0 -> normal
        1 -> attack

    String:
        attack/spoof/spoofing/malicious/anomaly -> 1
        everything else -> 0
    """
    if torch.is_tensor(labels):
        y = labels.detach().clone()
        if device is not None:
            y = y.to(device)
        return (y.float() >= 0.5).float()

    arr = np.asarray(labels)

    if arr.dtype.kind in {"U", "S", "O"}:
        attack_words = {"attack", "attacked", "spoof", "spoofing", "malicious", "anomaly", "1", "true"}
        flat = arr.reshape(-1)
        out = np.array(
            [1.0 if str(item).strip().lower() in attack_words else 0.0 for item in flat],
            dtype=np.float32,
        ).reshape(arr.shape)
    else:
        out = (arr.astype(np.float32) >= 0.5).astype(np.float32)

    tensor = torch.as_tensor(out, dtype=torch.float32)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def mask_to_float_tensor(
    mask: Optional[Union[Tensor, np.ndarray, Sequence[Any]]],
    reference: Tensor,
) -> Optional[Tensor]:
    """Convert optional mask to float tensor matching reference device/dtype."""
    if mask is None:
        return None

    if torch.is_tensor(mask):
        out = mask.to(device=reference.device, dtype=reference.dtype)
    else:
        out = torch.as_tensor(mask, dtype=reference.dtype, device=reference.device)

    if out.ndim == 1 and reference.ndim == 2:
        out = out.unsqueeze(1)

    if out.shape != reference.shape:
        raise ValueError(
            f"Mask shape mismatch. Expected {tuple(reference.shape)}, got {tuple(out.shape)}."
        )

    return (out > 0.5).to(dtype=reference.dtype)


def combine_masks(
    reference: Tensor,
    loss_mask: Optional[Union[Tensor, np.ndarray, Sequence[Any]]] = None,
    padding_mask: Optional[Union[Tensor, np.ndarray, Sequence[Any]]] = None,
    use_loss_mask: bool = True,
    combine_padding_and_loss_mask: bool = True,
) -> Tensor:
    """
    Combine loss mask and padding mask.

    Convention:
    - 1 means included in loss.
    - 0 means ignored.
    """
    mask = torch.ones_like(reference, dtype=reference.dtype, device=reference.device)

    if use_loss_mask and loss_mask is not None:
        loss_mask_tensor = mask_to_float_tensor(loss_mask, reference)
        mask = mask * loss_mask_tensor

    if combine_padding_and_loss_mask and padding_mask is not None:
        padding_mask_tensor = mask_to_float_tensor(padding_mask, reference)
        mask = mask * padding_mask_tensor

    return (mask > 0.5).to(dtype=reference.dtype)


def compute_binary_class_counts(
    labels: Union[Tensor, np.ndarray, Sequence[Any]],
    mask: Optional[Union[Tensor, np.ndarray, Sequence[Any]]] = None,
) -> Tuple[int, int, int]:
    """Compute normal/attack/total counts under optional mask."""
    y = labels_to_binary_tensor(labels)

    if mask is not None:
        mask_tensor = mask_to_float_tensor(mask, y)
        y = y[mask_tensor > 0.5]
    else:
        y = y.reshape(-1)

    total_count = int(y.numel())

    if total_count == 0:
        return 0, 0, 0

    attack_count = int((y >= 0.5).sum().item())
    normal_count = int(total_count - attack_count)

    return normal_count, attack_count, total_count


def _clip_weight(value: float, min_weight: float, max_weight: float) -> float:
    """Clip class weight safely."""
    if not math.isfinite(value):
        return float(max_weight)

    return float(max(float(min_weight), min(float(max_weight), value)))


def compute_balanced_class_weights(
    labels: Union[Tensor, np.ndarray, Sequence[Any]],
    mask: Optional[Union[Tensor, np.ndarray, Sequence[Any]]] = None,
    config: Optional[ClassWeightConfig] = None,
) -> ClassWeightSummary:
    """
    Compute alpha_0 and alpha_1 from labels.

    Balanced weights:
        alpha_0 = N / (2 * N_0)
        alpha_1 = N / (2 * N_1)

    This must be called on TRAIN split only by the trainer.
    """
    cfg = config or ClassWeightConfig()

    normal_count, attack_count, total_count = compute_binary_class_counts(
        labels=labels,
        mask=mask if cfg.valid_only else None,
    )

    if total_count <= 0:
        raise ValueError("Cannot compute class weights: no valid training labels.")

    strategy = cfg.strategy.lower().strip()

    if not cfg.enabled or strategy == "none":
        alpha_0 = 1.0
        alpha_1 = 1.0

    elif strategy == "manual":
        if cfg.manual_alpha_0 is None or cfg.manual_alpha_1 is None:
            raise ValueError(
                "Manual class weights require manual_alpha_0 and manual_alpha_1."
            )

        alpha_0 = float(cfg.manual_alpha_0)
        alpha_1 = float(cfg.manual_alpha_1)

    elif strategy == "balanced":
        if normal_count <= 0 or attack_count <= 0:
            raise ValueError(
                "Balanced class weights require both normal and attack examples "
                f"in the TRAIN split. Got normal={normal_count}, attack={attack_count}."
            )

        alpha_0 = total_count / (2.0 * normal_count)
        alpha_1 = total_count / (2.0 * attack_count)

    else:
        raise ValueError(
            f"Unknown class-weight strategy='{cfg.strategy}'. "
            "Supported: balanced, none, manual."
        )

    alpha_0 = _clip_weight(alpha_0, cfg.min_weight, cfg.max_weight)
    alpha_1 = _clip_weight(alpha_1, cfg.min_weight, cfg.max_weight)

    return ClassWeightSummary(
        strategy=strategy,
        normal_count=int(normal_count),
        attack_count=int(attack_count),
        total_count=int(total_count),
        normal_fraction=float(normal_count / max(total_count, 1)),
        attack_fraction=float(attack_count / max(total_count, 1)),
        alpha_0=float(alpha_0),
        alpha_1=float(alpha_1),
        train_only=bool(cfg.train_only),
        valid_only=bool(cfg.valid_only),
        status="PASSED",
    )


def _extract_batch_field(batch: Any, keys: Sequence[str]) -> Optional[Any]:
    """Extract first matching field from a dataloader batch."""
    if isinstance(batch, Mapping):
        for key in keys:
            if key in batch:
                return batch[key]

    return None


def compute_class_weights_from_dataloader(
    dataloader: Iterable[Any],
    config: Optional[ClassWeightConfig] = None,
) -> ClassWeightSummary:
    """
    Compute class weights from a TRAIN dataloader.

    Expected batch keys:
    - y or labels
    - loss_mask or valid_mask
    - padding_mask

    The trainer must call this only on the training dataloader.
    """
    labels_list = []
    masks_list = []

    for batch in dataloader:
        y = _extract_batch_field(batch, ["y", "labels", "label"])
        if y is None:
            raise KeyError("Could not find labels in batch. Expected key 'y' or 'labels'.")

        loss_mask = _extract_batch_field(batch, ["loss_mask", "valid_mask", "mask"])
        padding_mask = _extract_batch_field(batch, ["padding_mask"])

        y_tensor = labels_to_binary_tensor(y)
        mask_tensor = combine_masks(
            reference=y_tensor,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            use_loss_mask=True,
            combine_padding_and_loss_mask=True,
        )

        labels_list.append(y_tensor.detach().cpu().reshape(-1))
        masks_list.append(mask_tensor.detach().cpu().reshape(-1))

    if not labels_list:
        raise ValueError("Cannot compute class weights from empty dataloader.")

    all_labels = torch.cat(labels_list, dim=0)
    all_masks = torch.cat(masks_list, dim=0)

    return compute_balanced_class_weights(
        labels=all_labels,
        mask=all_masks,
        config=config or ClassWeightConfig(),
    )


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted binary cross entropy for sequence spoofing detection.

    Inputs:
        logits/probabilities: [B,T] or [B,T,1]
        labels:              [B,T]
        loss_mask:           [B,T], 1 included, 0 ignored
        padding_mask:        [B,T], 1 real row, 0 padded row
    """

    def __init__(self, config: Optional[WeightedBCELossConfig] = None) -> None:
        super().__init__()
        self.config = config or WeightedBCELossConfig()

        if self.config.reduction != "mean":
            raise ValueError("Only reduction='mean' is supported for reviewer-stable logging.")

        if self.config.normalization not in {"valid_count", "weight_sum"}:
            raise ValueError("normalization must be 'valid_count' or 'weight_sum'.")

    @classmethod
    def from_project_config(
        cls,
        config: Mapping[str, Any],
        class_weight_summary: Optional[ClassWeightSummary] = None,
    ) -> "WeightedBCEWithLogitsLoss":
        """Construct loss from project config and optional class weights."""
        alpha_0 = 1.0
        alpha_1 = 1.0

        if class_weight_summary is not None:
            alpha_0 = float(class_weight_summary.alpha_0)
            alpha_1 = float(class_weight_summary.alpha_1)

        loss_config = build_weighted_bce_config(
            config=config,
            alpha_0=alpha_0,
            alpha_1=alpha_1,
        )
        return cls(loss_config)

    def forward(
        self,
        predictions: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> LossComputationResult:
        """Compute masked weighted BCE."""
        if predictions.ndim == 3 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        if labels.ndim == 3 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)

        if predictions.shape != labels.shape:
            raise ValueError(
                f"Prediction/label shape mismatch. "
                f"predictions={tuple(predictions.shape)}, labels={tuple(labels.shape)}."
            )

        y = labels_to_binary_tensor(labels, device=predictions.device).to(dtype=predictions.dtype)

        mask = combine_masks(
            reference=y,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            use_loss_mask=self.config.use_loss_mask,
            combine_padding_and_loss_mask=self.config.combine_padding_and_loss_mask,
        )

        if self.config.from_logits:
            element_loss = F.binary_cross_entropy_with_logits(
                predictions,
                y,
                reduction="none",
            )
        else:
            p = torch.clamp(
                predictions,
                min=float(self.config.epsilon),
                max=1.0 - float(self.config.epsilon),
            )
            element_loss = F.binary_cross_entropy(
                p,
                y,
                reduction="none",
            )

        class_weights = (
            float(self.config.alpha_1) * y
            + float(self.config.alpha_0) * (1.0 - y)
        )

        weighted_element_loss = element_loss * class_weights * mask
        unweighted_element_loss = element_loss * mask

        valid_count_tensor = torch.clamp(mask.sum(), min=1.0)
        weight_sum_tensor = torch.clamp((class_weights * mask).sum(), min=1.0)

        if self.config.normalization == "valid_count":
            denominator = valid_count_tensor
        else:
            denominator = weight_sum_tensor

        loss = weighted_element_loss.sum() / denominator
        unweighted_loss = unweighted_element_loss.sum() / valid_count_tensor

        positive_count = int(((y >= 0.5) * (mask > 0.5)).sum().detach().cpu().item())
        valid_count = int((mask > 0.5).sum().detach().cpu().item())
        negative_count = int(valid_count - positive_count)

        return LossComputationResult(
            loss=loss,
            unweighted_loss=unweighted_loss,
            weighted_loss_sum=weighted_element_loss.sum(),
            valid_count=valid_count,
            positive_count=positive_count,
            negative_count=negative_count,
            alpha_0=float(self.config.alpha_0),
            alpha_1=float(self.config.alpha_1),
            from_logits=bool(self.config.from_logits),
            normalization=str(self.config.normalization),
        )


def compute_detection_loss(
    model_output: Any,
    batch: Mapping[str, Any],
    criterion: WeightedBCEWithLogitsLoss,
) -> LossComputationResult:
    """
    Compute detection loss from model output and Step-9 batch.

    Expected model output:
    - logits attribute preferred.
    - probabilities allowed only if criterion.config.from_logits=False.

    Expected batch:
    - y
    - loss_mask
    - padding_mask
    """
    if hasattr(model_output, "logits"):
        predictions = model_output.logits
    elif hasattr(model_output, "probabilities"):
        predictions = model_output.probabilities
    elif torch.is_tensor(model_output):
        predictions = model_output
    else:
        raise TypeError(
            "model_output must have .logits, .probabilities, or be a tensor."
        )

    if "y" in batch:
        labels = batch["y"]
    elif "labels" in batch:
        labels = batch["labels"]
    else:
        raise KeyError("Batch must contain 'y' or 'labels'.")

    loss_mask = batch.get("loss_mask")
    padding_mask = batch.get("padding_mask")

    return criterion(
        predictions=predictions,
        labels=labels,
        loss_mask=loss_mask,
        padding_mask=padding_mask,
    )


__all__ = [
    "ClassWeightConfig",
    "ClassWeightSummary",
    "WeightedBCELossConfig",
    "LossComputationResult",
    "build_class_weight_config",
    "build_weighted_bce_config",
    "labels_to_binary_tensor",
    "mask_to_float_tensor",
    "combine_masks",
    "compute_binary_class_counts",
    "compute_balanced_class_weights",
    "compute_class_weights_from_dataloader",
    "WeightedBCEWithLogitsLoss",
    "compute_detection_loss",
]