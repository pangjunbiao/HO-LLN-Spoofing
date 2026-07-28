"""
Temporal blocks for the proposed AV-GPS spoofing detector.

Step 11 module:
- liquid_second_order: full proposed second-order liquid dynamics
- gru: ablation replacement for no_liquid_dynamics
- simple_first_order: optional first-order temporal fallback

Important:
The full model uses:
    temporal_block = "liquid_second_order"

The official no_liquid_dynamics ablation uses:
    temporal_block = "gru"

All temporal blocks expose the same interface:
    hidden_sequence:  [B,T,H]
    velocity_sequence:[B,T,H]

For non-liquid blocks, velocity_sequence is returned as zeros to avoid giving
the no-liquid ablation a second-order velocity state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from src.models.liquid_dynamics import (
    LiquidSecondOrderDynamics,
    LiquidDynamicsOutput,
    build_liquid_dynamics_config,
)


@dataclass
class TemporalBlockConfig:
    """Configuration for temporal block wrapper."""

    temporal_block: str = "liquid_second_order"

    input_dim: int = 64
    hidden_dim: int = 64
    velocity_dim: int = 64

    dropout: float = 0.10

    gru_num_layers: int = 1
    gru_bidirectional: bool = False

    simple_tau_min: float = 0.05
    simple_tau_max: float = 10.0

    use_delta_t: bool = True
    max_delta_t: float = 5.0

    reset_state_at_segment_start: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalBlockOutput:
    """Unified temporal block output."""

    hidden_sequence: Tensor
    velocity_sequence: Tensor

    final_hidden: Tensor
    final_velocity: Tensor

    temporal_block: str

    padding_mask: Optional[Tensor]
    delta_t: Optional[Tensor]

    auxiliary: Dict[str, Tensor]
    config: Dict[str, Any]

    def output_tuple(self) -> Tuple[Tensor, Tensor]:
        return self.hidden_sequence, self.velocity_sequence

    def tensor_dict(self) -> Dict[str, Tensor]:
        out = {
            "hidden_sequence": self.hidden_sequence,
            "velocity_sequence": self.velocity_sequence,
            "final_hidden": self.final_hidden,
            "final_velocity": self.final_velocity,
        }
        out.update(self.auxiliary)
        return out


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


def build_temporal_block_config(
    config: Optional[Mapping[str, Any]] = None,
    input_dim: Optional[int] = None,
    temporal_block_override: Optional[str] = None,
) -> TemporalBlockConfig:
    """Build TemporalBlockConfig from full project config."""
    if config is None:
        inferred_input_dim = 64 if input_dim is None else int(input_dim)
        return TemporalBlockConfig(input_dim=inferred_input_dim)

    inferred_input_dim = input_dim
    if inferred_input_dim is None:
        inferred_input_dim = int(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.fusion_dim",
                64,
            )
        )

    temporal_block = str(
        temporal_block_override
        or _get_by_path(config, "model.proposed.temporal_block", "liquid_second_order")
    )

    hidden_dim = int(
        _get_by_path(config, "model.proposed.liquid_second_order.hidden_dim", 64)
    )
    velocity_dim = int(
        _get_by_path(
            config,
            "model.proposed.liquid_second_order.velocity_dim",
            hidden_dim,
        )
    )

    return TemporalBlockConfig(
        temporal_block=temporal_block,
        input_dim=int(inferred_input_dim),
        hidden_dim=hidden_dim,
        velocity_dim=velocity_dim,
        dropout=float(_get_by_path(config, "model.proposed.dropout", 0.10)),
        gru_num_layers=int(
            _get_by_path(config, "model.proposed.gru.num_layers", 1)
        ),
        gru_bidirectional=bool(
            _get_by_path(config, "model.proposed.gru.bidirectional", False)
        ),
        simple_tau_min=float(
            _get_by_path(config, "model.proposed.simple_first_order.tau_min", 0.05)
        ),
        simple_tau_max=float(
            _get_by_path(config, "model.proposed.simple_first_order.tau_max", 10.0)
        ),
        use_delta_t=bool(
            _get_by_path(config, "model.proposed.liquid_second_order.use_delta_t", True)
        ),
        max_delta_t=float(
            _get_by_path(config, "preprocessing.evidence.max_delta_seconds", 5.0)
        ),
        reset_state_at_segment_start=True,
    )


def _ensure_sequence_tensor(x: Tensor, name: str) -> Tensor:
    """Ensure tensor is [B,T,D]."""
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


def _ensure_delta_t(
    reference: Tensor,
    delta_t: Optional[Tensor],
    max_delta_t: float,
) -> Tensor:
    """Normalize delta_t to [B,T]."""
    batch_size, time_steps = reference.shape[0], reference.shape[1]

    if delta_t is None:
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
            raise ValueError(f"delta_t must have shape [B,T], got {tuple(delta_t.shape)}.")

        expected = (batch_size, time_steps)
        got = tuple(delta_t.shape)

        if got != expected:
            raise ValueError(f"delta_t shape mismatch. Expected {expected}, got {got}.")

        dt = delta_t.to(device=reference.device, dtype=reference.dtype)

    dt = torch.nan_to_num(dt, nan=0.0, posinf=max_delta_t, neginf=0.0)
    dt = torch.clamp(dt, min=0.0, max=float(max_delta_t))

    return dt


def _initial_hidden(
    batch_size: int,
    hidden_dim: int,
    reference: Tensor,
    initial_hidden: Optional[Tensor],
) -> Tensor:
    """Create or validate initial hidden state [B,H]."""
    if initial_hidden is None:
        return torch.zeros(
            batch_size,
            hidden_dim,
            dtype=reference.dtype,
            device=reference.device,
        )

    if not torch.is_tensor(initial_hidden):
        raise TypeError("initial_hidden must be a torch.Tensor or None.")

    if initial_hidden.ndim != 2:
        raise ValueError(
            f"initial_hidden must have shape [B,H], got {tuple(initial_hidden.shape)}."
        )

    expected = (batch_size, hidden_dim)
    got = tuple(initial_hidden.shape)

    if got != expected:
        raise ValueError(f"initial_hidden shape mismatch. Expected {expected}, got {got}.")

    return initial_hidden.to(device=reference.device, dtype=reference.dtype)


def _apply_reset_state_to_hidden(
    hidden: Tensor,
    reset_state: Optional[Tensor],
) -> Tensor:
    """
    Apply reset flag to initial hidden.

    reset_state convention:
    - 1 means reset to zero
    - 0 means keep provided state
    """
    if reset_state is None:
        return hidden

    if not torch.is_tensor(reset_state):
        raise TypeError("reset_state must be a torch.Tensor or None.")

    if reset_state.ndim == 0:
        reset_state = reset_state.reshape(1)

    if reset_state.ndim > 1:
        reset_state = reset_state.reshape(reset_state.shape[0])

    if reset_state.shape[0] != hidden.shape[0]:
        raise ValueError(
            f"reset_state length mismatch. Expected {hidden.shape[0]}, got {reset_state.shape[0]}."
        )

    mask = (reset_state.to(device=hidden.device, dtype=hidden.dtype) >= 0.5).unsqueeze(-1)
    return torch.where(mask, torch.zeros_like(hidden), hidden)


def _mask_sequence(x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
    """Zero padded sequence positions."""
    if padding_mask is None:
        return x

    return x * padding_mask.unsqueeze(-1)


def _gather_last_valid(
    sequence: Tensor,
    padding_mask: Optional[Tensor],
) -> Tensor:
    """Gather last valid hidden state for each sequence."""
    if padding_mask is None:
        return sequence[:, -1, :]

    lengths = padding_mask.sum(dim=1).long()
    lengths = torch.clamp(lengths, min=1)
    indices = lengths - 1

    batch_index = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[batch_index, indices, :]


class LiquidTemporalBlock(nn.Module):
    """Wrapper around full second-order liquid dynamics."""

    def __init__(
        self,
        project_config: Optional[Mapping[str, Any]] = None,
        config: Optional[TemporalBlockConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config or build_temporal_block_config(project_config)

        liquid_config = (
            build_liquid_dynamics_config(
                project_config,
                input_dim=self.config.input_dim,
            )
            if project_config is not None
            else None
        )

        if liquid_config is not None:
            liquid_config.hidden_dim = self.config.hidden_dim
            liquid_config.velocity_dim = self.config.velocity_dim

        self.liquid = LiquidSecondOrderDynamics(liquid_config)

    def forward(
        self,
        zeta: Tensor,
        delta_t: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        initial_hidden: Optional[Tensor] = None,
        initial_velocity: Optional[Tensor] = None,
        reset_state: Optional[Tensor] = None,
    ) -> TemporalBlockOutput:
        zeta = _ensure_sequence_tensor(zeta, "zeta")
        padding_mask = _ensure_padding_mask(zeta, padding_mask)

        liquid_output: LiquidDynamicsOutput = self.liquid(
            zeta=zeta,
            delta_t=delta_t,
            padding_mask=padding_mask,
            initial_hidden=initial_hidden,
            initial_velocity=initial_velocity,
            reset_state=reset_state,
        )

        return TemporalBlockOutput(
            hidden_sequence=liquid_output.hidden_sequence,
            velocity_sequence=liquid_output.velocity_sequence,
            final_hidden=liquid_output.final_hidden,
            final_velocity=liquid_output.final_velocity,
            temporal_block="liquid_second_order",
            padding_mask=padding_mask,
            delta_t=liquid_output.delta_t,
            auxiliary={
                "candidate_sequence": liquid_output.candidate_sequence,
                "gamma_sequence": liquid_output.gamma_sequence,
                "beta_sequence": liquid_output.beta_sequence,
                "tau_h_sequence": liquid_output.tau_h_sequence,
                "tau_v_sequence": liquid_output.tau_v_sequence,
            },
            config=self.config.to_dict(),
        )

    def module_summary(self) -> Dict[str, Any]:
        return {
            "module": "LiquidTemporalBlock",
            "temporal_block": "liquid_second_order",
            "config": self.config.to_dict(),
            "uses_second_order_velocity": True,
            "ablation_role": "full_model_temporal_block",
            "causal": True,
            "uses_future_rows": False,
        }


class GRUTemporalBlock(nn.Module):
    """
    GRU replacement for no_liquid_dynamics ablation.

    It intentionally returns zero velocity_sequence so the ablation does not
    receive the liquid model's second-order velocity state.
    """

    def __init__(self, config: TemporalBlockConfig) -> None:
        super().__init__()

        self.config = config

        if self.config.gru_bidirectional:
            raise ValueError(
                "GRUTemporalBlock must be causal. bidirectional=True is not allowed."
            )

        self.input_dropout = nn.Dropout(p=float(self.config.dropout))

        self.gru = nn.GRU(
            input_size=int(self.config.input_dim),
            hidden_size=int(self.config.hidden_dim),
            num_layers=int(self.config.gru_num_layers),
            dropout=float(self.config.dropout) if self.config.gru_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(
        self,
        zeta: Tensor,
        delta_t: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        initial_hidden: Optional[Tensor] = None,
        initial_velocity: Optional[Tensor] = None,
        reset_state: Optional[Tensor] = None,
    ) -> TemporalBlockOutput:
        zeta = _ensure_sequence_tensor(zeta, "zeta")

        if zeta.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"GRUTemporalBlock input_dim mismatch. Expected {self.config.input_dim}, got {zeta.shape[-1]}."
            )

        padding_mask = _ensure_padding_mask(zeta, padding_mask)
        dt = _ensure_delta_t(zeta, delta_t, max_delta_t=self.config.max_delta_t)

        zeta = _mask_sequence(zeta, padding_mask)
        zeta = self.input_dropout(zeta)

        batch_size = zeta.shape[0]

        h0_single = _initial_hidden(
            batch_size=batch_size,
            hidden_dim=self.config.hidden_dim,
            reference=zeta,
            initial_hidden=initial_hidden,
        )
        h0_single = _apply_reset_state_to_hidden(h0_single, reset_state)

        h0 = h0_single.unsqueeze(0).repeat(self.config.gru_num_layers, 1, 1)

        hidden_sequence, h_n = self.gru(zeta, h0)

        hidden_sequence = _mask_sequence(hidden_sequence, padding_mask)
        velocity_sequence = torch.zeros_like(hidden_sequence)

        final_hidden = _gather_last_valid(hidden_sequence, padding_mask)
        final_velocity = torch.zeros_like(final_hidden)

        return TemporalBlockOutput(
            hidden_sequence=hidden_sequence,
            velocity_sequence=velocity_sequence,
            final_hidden=final_hidden,
            final_velocity=final_velocity,
            temporal_block="gru",
            padding_mask=padding_mask,
            delta_t=dt,
            auxiliary={
                "gru_final_layer_hidden": h_n[-1],
            },
            config=self.config.to_dict(),
        )

    def module_summary(self) -> Dict[str, Any]:
        return {
            "module": "GRUTemporalBlock",
            "temporal_block": "gru",
            "config": self.config.to_dict(),
            "uses_second_order_velocity": False,
            "velocity_sequence_is_zero": True,
            "ablation_role": "no_liquid_dynamics",
            "causal": True,
            "bidirectional": False,
            "uses_future_rows": False,
        }


class SimpleFirstOrderTemporalBlock(nn.Module):
    """
    Optional first-order temporal fallback.

    This is not the official no_liquid ablation unless configured.
    It is useful for diagnostics and professor H0/H1 comparisons if needed.
    """

    def __init__(self, config: TemporalBlockConfig) -> None:
        super().__init__()

        self.config = config

        self.candidate_layer = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.tau_layer = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.dropout = nn.Dropout(p=float(self.config.dropout))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in [self.candidate_layer, self.tau_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        zeta: Tensor,
        delta_t: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        initial_hidden: Optional[Tensor] = None,
        initial_velocity: Optional[Tensor] = None,
        reset_state: Optional[Tensor] = None,
    ) -> TemporalBlockOutput:
        zeta = _ensure_sequence_tensor(zeta, "zeta")

        if zeta.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"SimpleFirstOrderTemporalBlock input_dim mismatch. Expected {self.config.input_dim}, got {zeta.shape[-1]}."
            )

        padding_mask = _ensure_padding_mask(zeta, padding_mask)
        dt = _ensure_delta_t(zeta, delta_t, max_delta_t=self.config.max_delta_t)

        batch_size, time_steps, _ = zeta.shape

        h_prev = _initial_hidden(
            batch_size=batch_size,
            hidden_dim=self.config.hidden_dim,
            reference=zeta,
            initial_hidden=initial_hidden,
        )
        h_prev = _apply_reset_state_to_hidden(h_prev, reset_state)

        zeta_dropped = self.dropout(zeta)

        candidate = torch.tanh(self.candidate_layer(zeta_dropped))

        tau_raw = torch.sigmoid(self.tau_layer(zeta_dropped))
        tau = (
            float(self.config.simple_tau_min)
            + (float(self.config.simple_tau_max) - float(self.config.simple_tau_min)) * tau_raw
        )

        outputs = []
        alpha_outputs = []

        for t in range(time_steps):
            dt_t = dt[:, t].unsqueeze(-1)
            alpha_t = 1.0 - torch.exp(-dt_t / tau[:, t, :])

            h_new = (1.0 - alpha_t) * h_prev + alpha_t * candidate[:, t, :]

            if padding_mask is not None:
                valid_t = padding_mask[:, t].unsqueeze(-1)
                h_current = torch.where(valid_t > 0.5, h_new, h_prev)
                h_out = torch.where(valid_t > 0.5, h_current, torch.zeros_like(h_current))
                alpha_out = torch.where(valid_t > 0.5, alpha_t, torch.zeros_like(alpha_t))
            else:
                h_current = h_new
                h_out = h_current
                alpha_out = alpha_t

            outputs.append(h_out.unsqueeze(1))
            alpha_outputs.append(alpha_out.unsqueeze(1))
            h_prev = h_current

        hidden_sequence = torch.cat(outputs, dim=1)
        alpha_sequence = torch.cat(alpha_outputs, dim=1)
        velocity_sequence = torch.zeros_like(hidden_sequence)

        final_hidden = _gather_last_valid(hidden_sequence, padding_mask)
        final_velocity = torch.zeros_like(final_hidden)

        return TemporalBlockOutput(
            hidden_sequence=hidden_sequence,
            velocity_sequence=velocity_sequence,
            final_hidden=final_hidden,
            final_velocity=final_velocity,
            temporal_block="simple_first_order",
            padding_mask=padding_mask,
            delta_t=dt,
            auxiliary={
                "candidate_sequence": _mask_sequence(candidate, padding_mask),
                "tau_sequence": _mask_sequence(tau, padding_mask),
                "alpha_sequence": alpha_sequence,
            },
            config=self.config.to_dict(),
        )

    def module_summary(self) -> Dict[str, Any]:
        return {
            "module": "SimpleFirstOrderTemporalBlock",
            "temporal_block": "simple_first_order",
            "config": self.config.to_dict(),
            "uses_second_order_velocity": False,
            "velocity_sequence_is_zero": True,
            "causal": True,
            "uses_future_rows": False,
        }

class IdentityTemporalBlock(nn.Module):
    """
    Strict no-temporal-dynamics block for no_liquid_dynamics ablation.

    It removes the liquid second-order dynamics and does not replace them
    with another recurrent model. The fused state zeta is passed through as
    hidden_sequence, and velocity_sequence is zero.

    This is stricter than GRU replacement.
    """

    def __init__(self, config: TemporalBlockConfig) -> None:
        super().__init__()
        self.config = config

        if int(self.config.input_dim) != int(self.config.hidden_dim):
            raise ValueError(
                "IdentityTemporalBlock requires input_dim == hidden_dim. "
                f"Got input_dim={self.config.input_dim}, hidden_dim={self.config.hidden_dim}. "
                "For the current model this should be 64 == 64."
            )

    def forward(
        self,
        zeta: Tensor,
        delta_t: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        initial_hidden: Optional[Tensor] = None,
        initial_velocity: Optional[Tensor] = None,
        reset_state: Optional[Tensor] = None,
    ) -> TemporalBlockOutput:
        zeta = _ensure_sequence_tensor(zeta, "zeta")

        if zeta.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"IdentityTemporalBlock input_dim mismatch. "
                f"Expected {self.config.input_dim}, got {zeta.shape[-1]}."
            )

        padding_mask = _ensure_padding_mask(zeta, padding_mask)
        dt = _ensure_delta_t(zeta, delta_t, max_delta_t=self.config.max_delta_t)

        hidden_sequence = _mask_sequence(zeta, padding_mask)
        velocity_sequence = torch.zeros_like(hidden_sequence)

        final_hidden = _gather_last_valid(hidden_sequence, padding_mask)
        final_velocity = torch.zeros_like(final_hidden)

        return TemporalBlockOutput(
            hidden_sequence=hidden_sequence,
            velocity_sequence=velocity_sequence,
            final_hidden=final_hidden,
            final_velocity=final_velocity,
            temporal_block="identity",
            padding_mask=padding_mask,
            delta_t=dt,
            auxiliary={},
            config=self.config.to_dict(),
        )

    def module_summary(self) -> Dict[str, Any]:
        return {
            "module": "IdentityTemporalBlock",
            "temporal_block": "identity",
            "config": self.config.to_dict(),
            "uses_second_order_velocity": False,
            "velocity_sequence_is_zero": True,
            "uses_recurrence": False,
            "ablation_role": "strict_no_liquid_dynamics",
            "causal": True,
            "uses_future_rows": False,
        }


def create_temporal_block(
    config: TemporalBlockConfig,
    project_config: Optional[Mapping[str, Any]] = None,
) -> nn.Module:
    """
    Create temporal block from config.

    Supported:
    - liquid_second_order
    - liquid
    - gru
    - simple_first_order
    - first_order
    """
    block_name = str(config.temporal_block).lower().strip()

    if block_name in {"liquid_second_order", "liquid"}:
        return LiquidTemporalBlock(project_config=project_config, config=config)

    if block_name == "gru":
        return GRUTemporalBlock(config=config)

    if block_name in {"simple_first_order", "first_order"}:
        return SimpleFirstOrderTemporalBlock(config=config)

    if block_name in {"identity", "no_temporal", "no_dynamics", "none"}:
        return IdentityTemporalBlock(config=config)

    raise ValueError(
        f"Unknown temporal_block='{config.temporal_block}'. "
        "Supported: liquid_second_order, gru, simple_first_order, identity."
    )


def create_temporal_block_from_project_config(
    project_config: Mapping[str, Any],
    input_dim: Optional[int] = None,
    temporal_block_override: Optional[str] = None,
) -> nn.Module:
    """Create temporal block from full project config."""
    cfg = build_temporal_block_config(
        config=project_config,
        input_dim=input_dim,
        temporal_block_override=temporal_block_override,
    )
    return create_temporal_block(config=cfg, project_config=project_config)


@torch.no_grad()
def temporal_output_statistics(output: TemporalBlockOutput) -> Dict[str, Any]:
    """Return JSON-safe temporal output diagnostics."""
    stats: Dict[str, Any] = {}

    tensors = {
        "hidden_sequence": output.hidden_sequence,
        "velocity_sequence": output.velocity_sequence,
    }
    tensors.update(output.auxiliary)

    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor):
            continue

        values = tensor.detach()

        if values.ndim >= 3 and output.padding_mask is not None:
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


__all__ = [
    "TemporalBlockConfig",
    "TemporalBlockOutput",
    "build_temporal_block_config",
    "LiquidTemporalBlock",
    "GRUTemporalBlock",
    "SimpleFirstOrderTemporalBlock",
    "IdentityTemporalBlock",
    "create_temporal_block",
    "create_temporal_block_from_project_config",
    "temporal_output_statistics",
]