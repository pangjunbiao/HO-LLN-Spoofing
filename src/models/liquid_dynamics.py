"""
Second-order liquid dynamics module for the proposed AV-GPS spoofing detector.

Step 11 module 5:
This module implements the causal second-order liquid temporal block:

candidate:
    h_tilde_t = tanh(W_c zeta_t + b_c)

adaptive constants:
    gamma_t = 1 - exp(-delta_t / tau_h_t)
    beta_t  = exp(-delta_t / tau_v_t)

second-order update:
    v_t = beta_t * v_{t-1} + (1 - beta_t) * (h_tilde_t - h_{t-1})
    h_t = h_{t-1} + gamma_t * v_t

Output:
    h_t and v_t for each time step.

Important:
- This module is causal: it loops forward from t=1 to T.
- No future time step is used.
- Segment reset is handled by starting each sequence/window with h0=v0=0,
  or by using reset_state flags in later training code.
- Padding rows do not update states.
- no_liquid_dynamics ablation will be handled in temporal_blocks.py by replacing
  this module with GRU/simple first-order temporal block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class LiquidDynamicsConfig:
    """Configuration for second-order liquid dynamics."""

    input_dim: int = 64
    hidden_dim: int = 64
    velocity_dim: int = 64

    tau_h_min: float = 0.05
    tau_h_max: float = 10.0
    tau_v_min: float = 0.05
    tau_v_max: float = 10.0

    min_delta_t: float = 0.0
    max_delta_t: float = 5.0

    dropout: float = 0.10

    use_delta_t: bool = True
    reset_h0_at_segment_start: bool = True
    reset_v0_at_segment_start: bool = True

    return_all_states: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiquidDynamicsOutput:
    """Output container for liquid dynamics."""

    hidden_sequence: Tensor
    velocity_sequence: Tensor

    final_hidden: Tensor
    final_velocity: Tensor

    candidate_sequence: Tensor
    gamma_sequence: Tensor
    beta_sequence: Tensor
    tau_h_sequence: Tensor
    tau_v_sequence: Tensor

    padding_mask: Optional[Tensor]
    delta_t: Tensor
    config: Dict[str, Any]

    def output_tuple(self) -> Tuple[Tensor, Tensor]:
        return self.hidden_sequence, self.velocity_sequence

    def tensor_dict(self) -> Dict[str, Tensor]:
        return {
            "hidden_sequence": self.hidden_sequence,
            "velocity_sequence": self.velocity_sequence,
            "final_hidden": self.final_hidden,
            "final_velocity": self.final_velocity,
            "candidate_sequence": self.candidate_sequence,
            "gamma_sequence": self.gamma_sequence,
            "beta_sequence": self.beta_sequence,
            "tau_h_sequence": self.tau_h_sequence,
            "tau_v_sequence": self.tau_v_sequence,
            "delta_t": self.delta_t,
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


def build_liquid_dynamics_config(
    config: Optional[Mapping[str, Any]] = None,
    input_dim: Optional[int] = None,
) -> LiquidDynamicsConfig:
    """Build LiquidDynamicsConfig from full project config."""
    if config is None:
        return LiquidDynamicsConfig(
            input_dim=64 if input_dim is None else int(input_dim)
        )

    inferred_input_dim = input_dim
    if inferred_input_dim is None:
        inferred_input_dim = int(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.fusion_dim",
                64,
            )
        )

    hidden_dim = int(
        _get_by_path(config, "model.proposed.liquid_second_order.hidden_dim", 64)
    )

    velocity_dim = int(
        _get_by_path(config, "model.proposed.liquid_second_order.velocity_dim", hidden_dim)
    )

    if velocity_dim != hidden_dim:
        raise ValueError(
            "This second-order liquid implementation requires velocity_dim == hidden_dim. "
            f"Got velocity_dim={velocity_dim}, hidden_dim={hidden_dim}."
        )

    return LiquidDynamicsConfig(
        input_dim=int(inferred_input_dim),
        hidden_dim=hidden_dim,
        velocity_dim=velocity_dim,
        tau_h_min=float(
            _get_by_path(config, "model.proposed.liquid_second_order.tau_h_min", 0.05)
        ),
        tau_h_max=float(
            _get_by_path(config, "model.proposed.liquid_second_order.tau_h_max", 10.0)
        ),
        tau_v_min=float(
            _get_by_path(config, "model.proposed.liquid_second_order.tau_v_min", 0.05)
        ),
        tau_v_max=float(
            _get_by_path(config, "model.proposed.liquid_second_order.tau_v_max", 10.0)
        ),
        min_delta_t=0.0,
        max_delta_t=float(
            _get_by_path(config, "preprocessing.evidence.max_delta_seconds", 5.0)
        ),
        dropout=float(_get_by_path(config, "model.proposed.dropout", 0.10)),
        use_delta_t=bool(
            _get_by_path(config, "model.proposed.liquid_second_order.use_delta_t", True)
        ),
        reset_h0_at_segment_start=bool(
            _get_by_path(
                config,
                "model.proposed.liquid_second_order.reset_h0_at_segment_start",
                True,
            )
        ),
        reset_v0_at_segment_start=bool(
            _get_by_path(
                config,
                "model.proposed.liquid_second_order.reset_v0_at_segment_start",
                True,
            )
        ),
        return_all_states=True,
    )


def _ensure_sequence_tensor(x: Tensor, name: str) -> Tensor:
    """Ensure tensor is [B,T,D]."""
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if x.ndim == 2:
        return x.unsqueeze(1)

    if x.ndim == 3:
        return x

    raise ValueError(
        f"{name} must have shape [B,D] or [B,T,D], got {tuple(x.shape)}."
    )


def _ensure_padding_mask(
    reference: Tensor,
    padding_mask: Optional[Tensor],
) -> Optional[Tensor]:
    """Normalize padding mask to [B,T] float tensor."""
    if padding_mask is None:
        return None

    if not torch.is_tensor(padding_mask):
        raise TypeError("padding_mask must be a torch.Tensor or None.")

    if padding_mask.ndim == 1:
        padding_mask = padding_mask.unsqueeze(1)

    if padding_mask.ndim != 2:
        raise ValueError(
            f"padding_mask must have shape [B,T], got {tuple(padding_mask.shape)}."
        )

    expected = (reference.shape[0], reference.shape[1])
    got = tuple(padding_mask.shape)

    if got != expected:
        raise ValueError(
            f"padding_mask shape mismatch. Expected {expected}, got {got}."
        )

    return padding_mask.to(device=reference.device, dtype=reference.dtype)


def _ensure_delta_t(
    reference: Tensor,
    delta_t: Optional[Tensor],
    use_delta_t: bool,
    min_delta_t: float,
    max_delta_t: float,
) -> Tensor:
    """Normalize delta_t to [B,T]."""
    batch_size, time_steps = reference.shape[0], reference.shape[1]

    if delta_t is None or not use_delta_t:
        dt = torch.ones(
            batch_size,
            time_steps,
            dtype=reference.dtype,
            device=reference.device,
        )
    else:
        if not torch.is_tensor(delta_t):
            raise TypeError("delta_t must be a torch.Tensor or None.")

        if delta_t.ndim == 1:
            delta_t = delta_t.unsqueeze(1)

        if delta_t.ndim == 3 and delta_t.shape[-1] == 1:
            delta_t = delta_t.squeeze(-1)

        if delta_t.ndim != 2:
            raise ValueError(
                f"delta_t must have shape [B,T], got {tuple(delta_t.shape)}."
            )

        expected = (batch_size, time_steps)
        got = tuple(delta_t.shape)

        if got != expected:
            raise ValueError(
                f"delta_t shape mismatch. Expected {expected}, got {got}."
            )

        dt = delta_t.to(device=reference.device, dtype=reference.dtype)

    dt = torch.nan_to_num(dt, nan=0.0, posinf=max_delta_t, neginf=0.0)
    dt = torch.clamp(dt, min=float(min_delta_t), max=float(max_delta_t))

    return dt


def _initial_state(
    batch_size: int,
    hidden_dim: int,
    reference: Tensor,
    provided: Optional[Tensor],
    name: str,
) -> Tensor:
    """Create or validate initial state [B,H]."""
    if provided is None:
        return torch.zeros(
            batch_size,
            hidden_dim,
            dtype=reference.dtype,
            device=reference.device,
        )

    if not torch.is_tensor(provided):
        raise TypeError(f"{name} must be a torch.Tensor or None.")

    if provided.ndim != 2:
        raise ValueError(
            f"{name} must have shape [B,H], got {tuple(provided.shape)}."
        )

    expected = (batch_size, hidden_dim)
    got = tuple(provided.shape)

    if got != expected:
        raise ValueError(f"{name} shape mismatch. Expected {expected}, got {got}.")

    return provided.to(device=reference.device, dtype=reference.dtype)


def _apply_reset_state(
    h0: Tensor,
    v0: Tensor,
    reset_state: Optional[Tensor],
    reset_h: bool,
    reset_v: bool,
) -> Tuple[Tensor, Tensor]:
    """
    Apply reset_state flag.

    reset_state convention:
    - 1 means reset initial state to zero
    - 0 means keep provided initial state
    """
    if reset_state is None:
        return h0, v0

    if not torch.is_tensor(reset_state):
        raise TypeError("reset_state must be a torch.Tensor or None.")

    if reset_state.ndim == 0:
        reset_state = reset_state.reshape(1)

    if reset_state.ndim > 1:
        reset_state = reset_state.reshape(reset_state.shape[0])

    if reset_state.shape[0] != h0.shape[0]:
        raise ValueError(
            f"reset_state length mismatch. Expected {h0.shape[0]}, got {reset_state.shape[0]}."
        )

    mask = (reset_state.to(device=h0.device, dtype=h0.dtype) >= 0.5).unsqueeze(-1)

    if reset_h:
        h0 = torch.where(mask, torch.zeros_like(h0), h0)

    if reset_v:
        v0 = torch.where(mask, torch.zeros_like(v0), v0)

    return h0, v0


class LiquidSecondOrderDynamics(nn.Module):
    """
    Causal second-order liquid dynamics.

    Input:
        zeta_t: [B,T,F]
        delta_t: [B,T]

    Output:
        hidden_sequence: [B,T,H]
        velocity_sequence: [B,T,H]
    """

    def __init__(self, config: Optional[LiquidDynamicsConfig] = None) -> None:
        super().__init__()

        self.config = config or LiquidDynamicsConfig()

        if self.config.input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if self.config.velocity_dim != self.config.hidden_dim:
            raise ValueError("velocity_dim must equal hidden_dim.")

        if self.config.tau_h_min <= 0.0 or self.config.tau_v_min <= 0.0:
            raise ValueError("tau minimum values must be positive.")

        if self.config.tau_h_max <= self.config.tau_h_min:
            raise ValueError("tau_h_max must be greater than tau_h_min.")

        if self.config.tau_v_max <= self.config.tau_v_min:
            raise ValueError("tau_v_max must be greater than tau_v_min.")

        self.dropout = nn.Dropout(p=float(self.config.dropout))

        self.candidate_layer = nn.Linear(
            self.config.input_dim,
            self.config.hidden_dim,
        )

        self.tau_h_layer = nn.Linear(
            self.config.input_dim,
            self.config.hidden_dim,
        )

        self.tau_v_layer = nn.Linear(
            self.config.input_dim,
            self.config.hidden_dim,
        )

        self.candidate_activation = nn.Tanh()

        self.reset_parameters()

    @classmethod
    def from_project_config(
        cls,
        config: Mapping[str, Any],
        input_dim: Optional[int] = None,
    ) -> "LiquidSecondOrderDynamics":
        """Construct from full project config."""
        return cls(build_liquid_dynamics_config(config, input_dim=input_dim))

    def reset_parameters(self) -> None:
        """Initialize liquid dynamics layers."""
        for layer in [self.candidate_layer, self.tau_h_layer, self.tau_v_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def validate_input(self, zeta: Tensor) -> Tensor:
        """Validate zeta input."""
        zeta = _ensure_sequence_tensor(zeta, "zeta")

        if zeta.shape[-1] != self.config.input_dim:
            raise ValueError(
                "Liquid dynamics input dimension mismatch. "
                f"Expected {self.config.input_dim}, got {zeta.shape[-1]}."
            )

        return zeta

    def compute_candidate_and_time_constants(
        self,
        zeta: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute h_tilde, tau_h, tau_v.

        tau values are bounded:
            tau = tau_min + (tau_max - tau_min) * sigmoid(linear(zeta))
        """
        zeta_dropped = self.dropout(zeta)

        candidate = self.candidate_activation(self.candidate_layer(zeta_dropped))

        tau_h_raw = torch.sigmoid(self.tau_h_layer(zeta_dropped))
        tau_v_raw = torch.sigmoid(self.tau_v_layer(zeta_dropped))

        tau_h = (
            float(self.config.tau_h_min)
            + (float(self.config.tau_h_max) - float(self.config.tau_h_min)) * tau_h_raw
        )
        tau_v = (
            float(self.config.tau_v_min)
            + (float(self.config.tau_v_max) - float(self.config.tau_v_min)) * tau_v_raw
        )

        return candidate, tau_h, tau_v

    def forward(
        self,
        zeta: Tensor,
        delta_t: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        initial_hidden: Optional[Tensor] = None,
        initial_velocity: Optional[Tensor] = None,
        reset_state: Optional[Tensor] = None,
    ) -> LiquidDynamicsOutput:
        """
        Forward pass.

        Args:
            zeta:
                Fused evidence sequence [B,T,F].
            delta_t:
                Time-step durations [B,T].
            padding_mask:
                Optional [B,T], 1 real row, 0 padded row.
            initial_hidden:
                Optional initial h_0 [B,H].
            initial_velocity:
                Optional initial v_0 [B,H].
            reset_state:
                Optional [B], 1 means reset h0/v0 to zero.

        Returns:
            LiquidDynamicsOutput.
        """
        zeta = self.validate_input(zeta)

        batch_size, time_steps, _input_dim = zeta.shape

        padding_mask = _ensure_padding_mask(zeta, padding_mask)

        dt = _ensure_delta_t(
            reference=zeta,
            delta_t=delta_t,
            use_delta_t=self.config.use_delta_t,
            min_delta_t=self.config.min_delta_t,
            max_delta_t=self.config.max_delta_t,
        )

        candidate, tau_h, tau_v = self.compute_candidate_and_time_constants(zeta)

        h_prev = _initial_state(
            batch_size=batch_size,
            hidden_dim=self.config.hidden_dim,
            reference=zeta,
            provided=initial_hidden,
            name="initial_hidden",
        )
        v_prev = _initial_state(
            batch_size=batch_size,
            hidden_dim=self.config.hidden_dim,
            reference=zeta,
            provided=initial_velocity,
            name="initial_velocity",
        )

        h_prev, v_prev = _apply_reset_state(
            h0=h_prev,
            v0=v_prev,
            reset_state=reset_state,
            reset_h=self.config.reset_h0_at_segment_start,
            reset_v=self.config.reset_v0_at_segment_start,
        )

        hidden_outputs = []
        velocity_outputs = []
        gamma_outputs = []
        beta_outputs = []

        for t in range(time_steps):
            candidate_t = candidate[:, t, :]
            tau_h_t = tau_h[:, t, :]
            tau_v_t = tau_v[:, t, :]

            dt_t = dt[:, t].unsqueeze(-1)

            gamma_t = 1.0 - torch.exp(-dt_t / tau_h_t)
            beta_t = torch.exp(-dt_t / tau_v_t)

            v_new = beta_t * v_prev + (1.0 - beta_t) * (candidate_t - h_prev)
            h_new = h_prev + gamma_t * v_new

            if padding_mask is not None:
                valid_t = padding_mask[:, t].unsqueeze(-1)
                h_current = torch.where(valid_t > 0.5, h_new, h_prev)
                v_current = torch.where(valid_t > 0.5, v_new, v_prev)

                h_out = torch.where(
                    valid_t > 0.5,
                    h_current,
                    torch.zeros_like(h_current),
                )
                v_out = torch.where(
                    valid_t > 0.5,
                    v_current,
                    torch.zeros_like(v_current),
                )
                gamma_out = torch.where(
                    valid_t > 0.5,
                    gamma_t,
                    torch.zeros_like(gamma_t),
                )
                beta_out = torch.where(
                    valid_t > 0.5,
                    beta_t,
                    torch.zeros_like(beta_t),
                )
            else:
                h_current = h_new
                v_current = v_new
                h_out = h_current
                v_out = v_current
                gamma_out = gamma_t
                beta_out = beta_t

            hidden_outputs.append(h_out.unsqueeze(1))
            velocity_outputs.append(v_out.unsqueeze(1))
            gamma_outputs.append(gamma_out.unsqueeze(1))
            beta_outputs.append(beta_out.unsqueeze(1))

            h_prev = h_current
            v_prev = v_current

        hidden_sequence = torch.cat(hidden_outputs, dim=1)
        velocity_sequence = torch.cat(velocity_outputs, dim=1)
        gamma_sequence = torch.cat(gamma_outputs, dim=1)
        beta_sequence = torch.cat(beta_outputs, dim=1)

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1)
            candidate = candidate * mask
            tau_h = tau_h * mask
            tau_v = tau_v * mask

        return LiquidDynamicsOutput(
            hidden_sequence=hidden_sequence,
            velocity_sequence=velocity_sequence,
            final_hidden=h_prev,
            final_velocity=v_prev,
            candidate_sequence=candidate,
            gamma_sequence=gamma_sequence,
            beta_sequence=beta_sequence,
            tau_h_sequence=tau_h,
            tau_v_sequence=tau_v,
            padding_mask=padding_mask,
            delta_t=dt,
            config=self.config.to_dict(),
        )

    @torch.no_grad()
    def dynamics_statistics(self, output: LiquidDynamicsOutput) -> Dict[str, Any]:
        """Return JSON-safe dynamics diagnostics."""
        stats: Dict[str, Any] = {}

        tensors = {
            "hidden_sequence": output.hidden_sequence,
            "velocity_sequence": output.velocity_sequence,
            "candidate_sequence": output.candidate_sequence,
            "gamma_sequence": output.gamma_sequence,
            "beta_sequence": output.beta_sequence,
            "tau_h_sequence": output.tau_h_sequence,
            "tau_v_sequence": output.tau_v_sequence,
        }

        for name, tensor in tensors.items():
            values = tensor.detach()

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
            f"input_dim={self.config.input_dim}, "
            f"hidden_dim={self.config.hidden_dim}, "
            f"tau_h=[{self.config.tau_h_min}, {self.config.tau_h_max}], "
            f"tau_v=[{self.config.tau_v_min}, {self.config.tau_v_max}], "
            f"use_delta_t={self.config.use_delta_t}"
        )

    def module_summary(self) -> Dict[str, Any]:
        """JSON-safe module summary for Step-11 inspection."""
        return {
            "module": "LiquidSecondOrderDynamics",
            "config": self.config.to_dict(),
            "candidate_formula": "h_tilde_t = tanh(W_c zeta_t + b_c)",
            "gamma_formula": "gamma_t = 1 - exp(-delta_t / tau_h_t)",
            "beta_formula": "beta_t = exp(-delta_t / tau_v_t)",
            "velocity_update": "v_t = beta_t * v_{t-1} + (1 - beta_t) * (h_tilde_t - h_{t-1})",
            "hidden_update": "h_t = h_{t-1} + gamma_t * v_t",
            "causal": True,
            "uses_future_rows": False,
            "padding_rows_update_state": False,
            "ablation_note": (
                "no_liquid_dynamics is applied in temporal_blocks.py by replacing "
                "this module with GRU/simple first-order temporal block."
            ),
        }


__all__ = [
    "LiquidDynamicsConfig",
    "LiquidDynamicsOutput",
    "LiquidSecondOrderDynamics",
    "build_liquid_dynamics_config",
]