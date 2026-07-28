"""
Output head for the proposed AV-GPS spoofing detector.

Step 11 module:
6. Output head

Methodology:
    p_hat_t = sigmoid(w_o^T [h_t, v_t] + b_o)

This implementation supports:
- hidden_and_velocity: full proposed output head
- hidden_only: optional diagnostic/ablation-compatible mode
- velocity_only: diagnostic only

Default is hidden_and_velocity.

Important:
- This module produces probabilities only.
- Thresholding and N_p alarm confirmation are handled by Step 10 evaluation code.
- The synthetic Step-10 theta is not used here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class OutputHeadConfig:
    """Configuration for output head."""

    hidden_dim: int = 64
    velocity_dim: int = 64

    input_mode: str = "hidden_and_velocity"

    head_hidden_dim: int = 64
    dropout: float = 0.10
    use_layer_norm: bool = False

    output_dim: int = 1
    probability_activation: str = "sigmoid"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputHeadResult:
    """Output container for output head."""

    logits: Tensor
    probabilities: Tensor

    head_input: Tensor
    hidden_sequence: Tensor
    velocity_sequence: Tensor

    padding_mask: Optional[Tensor]
    config: Dict[str, Any]

    def prediction_tensor(self) -> Tensor:
        return self.probabilities

    def logits_tensor(self) -> Tensor:
        return self.logits

    def tensor_dict(self) -> Dict[str, Tensor]:
        return {
            "logits": self.logits,
            "probabilities": self.probabilities,
            "head_input": self.head_input,
            "hidden_sequence": self.hidden_sequence,
            "velocity_sequence": self.velocity_sequence,
        }


def _get_by_path(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Small local get_by_path fallback for model configs."""
    current: Any = config

    for key in path.split("."):
        if not isinstance(current, Mapping):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def build_output_head_config(
    config: Optional[Mapping[str, Any]] = None,
    hidden_dim: Optional[int] = None,
    velocity_dim: Optional[int] = None,
) -> OutputHeadConfig:
    """Build OutputHeadConfig from full project config."""
    if config is None:
        h = 64 if hidden_dim is None else int(hidden_dim)
        v = h if velocity_dim is None else int(velocity_dim)

        return OutputHeadConfig(
            hidden_dim=h,
            velocity_dim=v,
        )

    inferred_hidden_dim = hidden_dim
    if inferred_hidden_dim is None:
        inferred_hidden_dim = int(
            _get_by_path(config, "model.proposed.liquid_second_order.hidden_dim", 64)
        )

    inferred_velocity_dim = velocity_dim
    if inferred_velocity_dim is None:
        inferred_velocity_dim = int(
            _get_by_path(
                config,
                "model.proposed.liquid_second_order.velocity_dim",
                inferred_hidden_dim,
            )
        )

    return OutputHeadConfig(
        hidden_dim=int(inferred_hidden_dim),
        velocity_dim=int(inferred_velocity_dim),
        input_mode=str(
            _get_by_path(config, "model.proposed.output_head.input", "hidden_and_velocity")
        ),
        head_hidden_dim=int(
            _get_by_path(config, "model.proposed.output_head.hidden_dim", inferred_hidden_dim)
        ),
        dropout=float(_get_by_path(config, "model.proposed.dropout", 0.10)),
        use_layer_norm=bool(
            _get_by_path(config, "model.proposed.output_head.use_layer_norm", False)
        ),
        output_dim=1,
        probability_activation=str(
            _get_by_path(config, "model.proposed.output_head.activation", "sigmoid")
        ),
    )


def _ensure_sequence_tensor(x: Tensor, name: str) -> Tensor:
    """Ensure tensor shape [B,T,D]."""
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if x.ndim == 2:
        return x.unsqueeze(1)

    if x.ndim == 3:
        return x

    raise ValueError(f"{name} must have shape [B,D] or [B,T,D], got {tuple(x.shape)}.")


def _ensure_padding_mask(
    reference: Tensor,
    padding_mask: Optional[Tensor],
) -> Optional[Tensor]:
    """Normalize padding mask to [B,T]."""
    if padding_mask is None:
        return None

    if not torch.is_tensor(padding_mask):
        raise TypeError("padding_mask must be a torch.Tensor or None.")

    if padding_mask.ndim == 1:
        padding_mask = padding_mask.unsqueeze(1)

    if padding_mask.ndim != 2:
        raise ValueError(f"padding_mask must have shape [B,T], got {tuple(padding_mask.shape)}.")

    expected = (reference.shape[0], reference.shape[1])
    got = tuple(padding_mask.shape)

    if got != expected:
        raise ValueError(f"padding_mask shape mismatch. Expected {expected}, got {got}.")

    return padding_mask.to(device=reference.device, dtype=reference.dtype)


def _mask_sequence(x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
    """Zero padded positions."""
    if padding_mask is None:
        return x

    return x * padding_mask.unsqueeze(-1)


class OutputHead(nn.Module):
    """
    Output head producing per-time-step spoofing probability.

    Full proposed formula:
        p_hat_t = sigmoid(w_o^T [h_t, v_t] + b_o)

    Input:
        hidden_sequence:   [B,T,H]
        velocity_sequence: [B,T,V]

    Output:
        logits:        [B,T]
        probabilities: [B,T]
    """

    def __init__(self, config: Optional[OutputHeadConfig] = None) -> None:
        super().__init__()

        self.config = config or OutputHeadConfig()

        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if self.config.velocity_dim <= 0:
            raise ValueError("velocity_dim must be positive.")

        self.input_mode = str(self.config.input_mode).lower().strip()

        if self.input_mode == "hidden_and_velocity":
            head_input_dim = int(self.config.hidden_dim + self.config.velocity_dim)
        elif self.input_mode == "hidden_only":
            head_input_dim = int(self.config.hidden_dim)
        elif self.input_mode == "velocity_only":
            head_input_dim = int(self.config.velocity_dim)
        else:
            raise ValueError(
                f"Unknown output_head input_mode='{self.config.input_mode}'. "
                "Supported: hidden_and_velocity, hidden_only, velocity_only."
            )

        self.head_input_dim = head_input_dim

        layers = []

        if self.config.use_layer_norm:
            layers.append(nn.LayerNorm(head_input_dim))

        layers.extend(
            [
                nn.Dropout(p=float(self.config.dropout)),
                nn.Linear(head_input_dim, int(self.config.head_hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=float(self.config.dropout)),
                nn.Linear(int(self.config.head_hidden_dim), int(self.config.output_dim)),
            ]
        )

        self.net = nn.Sequential(*layers)

        if self.config.probability_activation.lower() != "sigmoid":
            raise ValueError("OutputHead currently supports probability_activation='sigmoid' only.")

        self.activation = nn.Sigmoid()

        self.reset_parameters()

    @classmethod
    def from_project_config(
        cls,
        config: Mapping[str, Any],
        hidden_dim: Optional[int] = None,
        velocity_dim: Optional[int] = None,
    ) -> "OutputHead":
        """Construct from full project config."""
        return cls(
            build_output_head_config(
                config=config,
                hidden_dim=hidden_dim,
                velocity_dim=velocity_dim,
            )
        )

    def reset_parameters(self) -> None:
        """Initialize linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def validate_inputs(
        self,
        hidden_sequence: Tensor,
        velocity_sequence: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Validate hidden/velocity inputs."""
        hidden = _ensure_sequence_tensor(hidden_sequence, "hidden_sequence")
        velocity = _ensure_sequence_tensor(velocity_sequence, "velocity_sequence")

        if hidden.shape[0] != velocity.shape[0] or hidden.shape[1] != velocity.shape[1]:
            raise ValueError(
                "OutputHead hidden/velocity batch-time mismatch. "
                f"hidden={tuple(hidden.shape)}, velocity={tuple(velocity.shape)}."
            )

        if hidden.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch. Expected {self.config.hidden_dim}, got {hidden.shape[-1]}."
            )

        if velocity.shape[-1] != self.config.velocity_dim:
            raise ValueError(
                f"velocity_dim mismatch. Expected {self.config.velocity_dim}, got {velocity.shape[-1]}."
            )

        return hidden, velocity

    def build_head_input(
        self,
        hidden_sequence: Tensor,
        velocity_sequence: Tensor,
    ) -> Tensor:
        """Build output head input according to input_mode."""
        if self.input_mode == "hidden_and_velocity":
            return torch.cat([hidden_sequence, velocity_sequence], dim=-1)

        if self.input_mode == "hidden_only":
            return hidden_sequence

        if self.input_mode == "velocity_only":
            return velocity_sequence

        raise RuntimeError(f"Unsupported input_mode='{self.input_mode}'.")

    def forward(
        self,
        hidden_sequence: Tensor,
        velocity_sequence: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> OutputHeadResult:
        """
        Forward pass.

        Args:
            hidden_sequence:
                [B,T,H]
            velocity_sequence:
                [B,T,V]
            padding_mask:
                Optional [B,T], 1 real row, 0 padded row.

        Returns:
            OutputHeadResult with logits/probabilities [B,T].
        """
        hidden, velocity = self.validate_inputs(
            hidden_sequence=hidden_sequence,
            velocity_sequence=velocity_sequence,
        )

        padding_mask = _ensure_padding_mask(hidden, padding_mask)

        hidden = _mask_sequence(hidden, padding_mask)
        velocity = _mask_sequence(velocity, padding_mask)

        head_input = self.build_head_input(
            hidden_sequence=hidden,
            velocity_sequence=velocity,
        )
        head_input = _mask_sequence(head_input, padding_mask)

        logits = self.net(head_input).squeeze(-1)
        probabilities = self.activation(logits)

        if padding_mask is not None:
            logits = logits * padding_mask
            probabilities = probabilities * padding_mask

        return OutputHeadResult(
            logits=logits,
            probabilities=probabilities,
            head_input=head_input,
            hidden_sequence=hidden,
            velocity_sequence=velocity,
            padding_mask=padding_mask,
            config=self.config.to_dict(),
        )

    @torch.no_grad()
    def output_statistics(self, output: OutputHeadResult) -> Dict[str, Any]:
        """Return JSON-safe output diagnostics."""
        stats: Dict[str, Any] = {}

        tensors = {
            "logits": output.logits,
            "probabilities": output.probabilities,
            "head_input": output.head_input,
        }

        for name, tensor in tensors.items():
            values = tensor.detach()

            if name in {"logits", "probabilities"}:
                if output.padding_mask is not None:
                    mask = output.padding_mask.detach().bool()
                    values = values[mask]
            else:
                if output.padding_mask is not None:
                    mask = output.padding_mask.detach().bool().unsqueeze(-1)
                    values = values[mask.expand_as(values)]

            if values.numel() == 0:
                stats[name] = {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                }
                continue

            stats[name] = {
                "count": int(values.numel()),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
            }

        return stats

    def extra_repr(self) -> str:
        return (
            f"input_mode={self.config.input_mode}, "
            f"head_input_dim={self.head_input_dim}, "
            f"hidden_dim={self.config.hidden_dim}, "
            f"velocity_dim={self.config.velocity_dim}, "
            f"head_hidden_dim={self.config.head_hidden_dim}"
        )

    def module_summary(self) -> Dict[str, Any]:
        """JSON-safe module summary for Step-11 inspection."""
        return {
            "module": "OutputHead",
            "config": self.config.to_dict(),
            "head_input_dim": int(self.head_input_dim),
            "formula": "p_hat_t = sigmoid(w_o^T [h_t, v_t] + b_o)",
            "threshold_applied_here": False,
            "alarm_rule_applied_here": False,
            "note": "Threshold and N_p alarm confirmation are handled by Step 10 evaluation code.",
        }


__all__ = [
    "OutputHeadConfig",
    "OutputHeadResult",
    "OutputHead",
    "build_output_head_config",
]