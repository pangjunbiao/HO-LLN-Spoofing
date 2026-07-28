"""
Kirchhoff-inspired exchange module.

Step 11 module 2:
This module implements the symmetric conductance exchange among the three
evidence states:

- instantaneous residual state s_t^I
- residual evolution state s_t^E
- persistence state s_t^P

Methodology:

C_ij,t = sigmoid(w_C^T [ |s_i - s_j|, s_i * s_j ] + b_C)

bar{s}_i = s_i + 1/2 * sum_{j != i} C_ij,t * (s_j - s_i)

Important:
- Conductance is symmetric because the same function uses |difference| and product.
- Exchange is local at each time step.
- No future information is used.
- Ablation no_kirchhoff_exchange is handled by enabled=False.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class KirchhoffExchangeConfig:
    """Configuration for KirchhoffExchange."""

    state_dim: int = 32
    conductance_hidden_dim: int = 32
    dropout: float = 0.10

    enabled: bool = True
    symmetric_conductance: bool = True
    exchange_scale: float = 0.5

    clamp_conductance_min: float = 0.0
    clamp_conductance_max: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KirchhoffExchangeOutput:
    """Output container for Kirchhoff exchange."""

    instantaneous_state: Tensor
    evolution_state: Tensor
    persistence_state: Tensor

    conductance_ie: Tensor
    conductance_ip: Tensor
    conductance_ep: Tensor

    original_instantaneous_state: Tensor
    original_evolution_state: Tensor
    original_persistence_state: Tensor

    padding_mask: Optional[Tensor]
    config: Dict[str, Any]

    def exchanged_tuple(self) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.instantaneous_state,
            self.evolution_state,
            self.persistence_state,
        )

    def conductance_dict(self) -> Dict[str, Tensor]:
        return {
            "conductance_ie": self.conductance_ie,
            "conductance_ip": self.conductance_ip,
            "conductance_ep": self.conductance_ep,
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


def build_kirchhoff_exchange_config(
    config: Optional[Mapping[str, Any]] = None,
    state_dim: Optional[int] = None,
) -> KirchhoffExchangeConfig:
    """Build KirchhoffExchangeConfig from full project config."""
    if config is None:
        return KirchhoffExchangeConfig(
            state_dim=32 if state_dim is None else int(state_dim)
        )

    inferred_state_dim = state_dim
    if inferred_state_dim is None:
        inferred_state_dim = int(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.instantaneous_branch_dim",
                32,
            )
        )

    return KirchhoffExchangeConfig(
        state_dim=int(inferred_state_dim),
        conductance_hidden_dim=int(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.conductance_hidden_dim",
                32,
            )
        ),
        dropout=float(_get_by_path(config, "model.proposed.dropout", 0.10)),
        enabled=bool(
            _get_by_path(config, "model.proposed.use_kirchhoff_exchange", True)
        ),
        symmetric_conductance=bool(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.symmetric_conductance",
                True,
            )
        ),
        exchange_scale=0.5,
        clamp_conductance_min=0.0,
        clamp_conductance_max=1.0,
    )


def _ensure_state_tensor(x: Tensor, name: str) -> Tensor:
    """Validate state tensor shape."""
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

    if padding_mask.shape[0] != reference.shape[0] or padding_mask.shape[1] != reference.shape[1]:
        raise ValueError(
            "padding_mask shape mismatch. "
            f"Expected {(reference.shape[0], reference.shape[1])}, got {tuple(padding_mask.shape)}."
        )

    return padding_mask.to(device=reference.device, dtype=reference.dtype)


def _apply_padding_mask(x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
    """Zero padded positions."""
    if padding_mask is None:
        return x

    return x * padding_mask.unsqueeze(-1)


class PairwiseConductance(nn.Module):
    """
    Symmetric pairwise conductance function.

    Input pair:
        s_i, s_j with shape [B,T,D]

    Features:
        [abs(s_i - s_j), s_i * s_j]

    Output:
        C_ij with shape [B,T,1]
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(2 * self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize conductance network."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state_i: Tensor, state_j: Tensor) -> Tensor:
        """Compute symmetric conductance."""
        if state_i.shape != state_j.shape:
            raise ValueError(
                "PairwiseConductance input shape mismatch: "
                f"{tuple(state_i.shape)} vs {tuple(state_j.shape)}."
            )

        pair_features = torch.cat(
            [
                torch.abs(state_i - state_j),
                state_i * state_j,
            ],
            dim=-1,
        )

        return self.net(pair_features)


class KirchhoffExchange(nn.Module):
    """
    Kirchhoff-inspired conductance exchange among three evidence states.

    If enabled=False, this module becomes identity pass-through and conductances
    are returned as zeros.
    """

    def __init__(self, config: Optional[KirchhoffExchangeConfig] = None) -> None:
        super().__init__()

        self.config = config or KirchhoffExchangeConfig()

        if self.config.state_dim <= 0:
            raise ValueError("state_dim must be positive.")

        if not self.config.symmetric_conductance:
            raise ValueError(
                "This implementation intentionally supports symmetric conductance only. "
                "Set symmetric_conductance=true."
            )

        self.conductance = PairwiseConductance(
            state_dim=self.config.state_dim,
            hidden_dim=self.config.conductance_hidden_dim,
            dropout=self.config.dropout,
        )

    @classmethod
    def from_project_config(
        cls,
        config: Mapping[str, Any],
        state_dim: Optional[int] = None,
    ) -> "KirchhoffExchange":
        """Construct from full project config."""
        return cls(build_kirchhoff_exchange_config(config, state_dim=state_dim))

    def validate_states(
        self,
        instantaneous_state: Tensor,
        evolution_state: Tensor,
        persistence_state: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Validate and normalize branch state tensors."""
        s_i = _ensure_state_tensor(instantaneous_state, "instantaneous_state")
        s_e = _ensure_state_tensor(evolution_state, "evolution_state")
        s_p = _ensure_state_tensor(persistence_state, "persistence_state")

        if s_i.shape != s_e.shape or s_i.shape != s_p.shape:
            raise ValueError(
                "KirchhoffExchange requires all branch states to have the same shape. "
                f"Got I={tuple(s_i.shape)}, E={tuple(s_e.shape)}, P={tuple(s_p.shape)}."
            )

        if s_i.shape[-1] != self.config.state_dim:
            raise ValueError(
                "KirchhoffExchange state_dim mismatch. "
                f"Expected {self.config.state_dim}, got {s_i.shape[-1]}."
            )

        return s_i, s_e, s_p

    def _zero_conductance_like(self, state: Tensor) -> Tensor:
        """Create zero conductance tensor [B,T,1]."""
        return torch.zeros(
            state.shape[0],
            state.shape[1],
            1,
            dtype=state.dtype,
            device=state.device,
        )

    def compute_conductances(
        self,
        s_i: Tensor,
        s_e: Tensor,
        s_p: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute pairwise conductances."""
        c_ie = self.conductance(s_i, s_e)
        c_ip = self.conductance(s_i, s_p)
        c_ep = self.conductance(s_e, s_p)

        c_ie = torch.clamp(
            c_ie,
            min=float(self.config.clamp_conductance_min),
            max=float(self.config.clamp_conductance_max),
        )
        c_ip = torch.clamp(
            c_ip,
            min=float(self.config.clamp_conductance_min),
            max=float(self.config.clamp_conductance_max),
        )
        c_ep = torch.clamp(
            c_ep,
            min=float(self.config.clamp_conductance_min),
            max=float(self.config.clamp_conductance_max),
        )

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1)
            c_ie = c_ie * mask
            c_ip = c_ip * mask
            c_ep = c_ep * mask

        return {
            "conductance_ie": c_ie,
            "conductance_ip": c_ip,
            "conductance_ep": c_ep,
        }

    def exchange_states(
        self,
        s_i: Tensor,
        s_e: Tensor,
        s_p: Tensor,
        c_ie: Tensor,
        c_ip: Tensor,
        c_ep: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply Kirchhoff exchange.

        bar{s}_I = s_I + 0.5 * [C_IE(s_E - s_I) + C_IP(s_P - s_I)]
        bar{s}_E = s_E + 0.5 * [C_IE(s_I - s_E) + C_EP(s_P - s_E)]
        bar{s}_P = s_P + 0.5 * [C_IP(s_I - s_P) + C_EP(s_E - s_P)]
        """
        scale = float(self.config.exchange_scale)

        s_i_bar = s_i + scale * (
            c_ie * (s_e - s_i)
            + c_ip * (s_p - s_i)
        )

        s_e_bar = s_e + scale * (
            c_ie * (s_i - s_e)
            + c_ep * (s_p - s_e)
        )

        s_p_bar = s_p + scale * (
            c_ip * (s_i - s_p)
            + c_ep * (s_e - s_p)
        )

        return s_i_bar, s_e_bar, s_p_bar

    def forward(
        self,
        instantaneous_state: Tensor,
        evolution_state: Tensor,
        persistence_state: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> KirchhoffExchangeOutput:
        """
        Forward pass.

        Args:
            instantaneous_state: [B,T,D]
            evolution_state: [B,T,D]
            persistence_state: [B,T,D]
            padding_mask: optional [B,T], 1 real row, 0 padded row

        Returns:
            KirchhoffExchangeOutput.
        """
        s_i, s_e, s_p = self.validate_states(
            instantaneous_state=instantaneous_state,
            evolution_state=evolution_state,
            persistence_state=persistence_state,
        )

        padding_mask = _ensure_padding_mask(s_i, padding_mask)

        original_i = s_i
        original_e = s_e
        original_p = s_p

        if not self.config.enabled:
            zero_c = self._zero_conductance_like(s_i)

            return KirchhoffExchangeOutput(
                instantaneous_state=_apply_padding_mask(s_i, padding_mask),
                evolution_state=_apply_padding_mask(s_e, padding_mask),
                persistence_state=_apply_padding_mask(s_p, padding_mask),
                conductance_ie=zero_c,
                conductance_ip=zero_c.clone(),
                conductance_ep=zero_c.clone(),
                original_instantaneous_state=original_i,
                original_evolution_state=original_e,
                original_persistence_state=original_p,
                padding_mask=padding_mask,
                config=self.config.to_dict(),
            )

        conductances = self.compute_conductances(
            s_i=s_i,
            s_e=s_e,
            s_p=s_p,
            padding_mask=padding_mask,
        )

        s_i_bar, s_e_bar, s_p_bar = self.exchange_states(
            s_i=s_i,
            s_e=s_e,
            s_p=s_p,
            c_ie=conductances["conductance_ie"],
            c_ip=conductances["conductance_ip"],
            c_ep=conductances["conductance_ep"],
        )

        s_i_bar = _apply_padding_mask(s_i_bar, padding_mask)
        s_e_bar = _apply_padding_mask(s_e_bar, padding_mask)
        s_p_bar = _apply_padding_mask(s_p_bar, padding_mask)

        return KirchhoffExchangeOutput(
            instantaneous_state=s_i_bar,
            evolution_state=s_e_bar,
            persistence_state=s_p_bar,
            conductance_ie=conductances["conductance_ie"],
            conductance_ip=conductances["conductance_ip"],
            conductance_ep=conductances["conductance_ep"],
            original_instantaneous_state=original_i,
            original_evolution_state=original_e,
            original_persistence_state=original_p,
            padding_mask=padding_mask,
            config=self.config.to_dict(),
        )

    @torch.no_grad()
    def conductance_statistics(self, output: KirchhoffExchangeOutput) -> Dict[str, Any]:
        """Return JSON-safe conductance diagnostics."""
        stats: Dict[str, Any] = {}

        for name, tensor in output.conductance_dict().items():
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
            f"state_dim={self.config.state_dim}, "
            f"conductance_hidden_dim={self.config.conductance_hidden_dim}, "
            f"enabled={self.config.enabled}, "
            f"exchange_scale={self.config.exchange_scale}"
        )

    def module_summary(self) -> Dict[str, Any]:
        """JSON-safe module summary for Step-11 inspection."""
        return {
            "module": "KirchhoffExchange",
            "config": self.config.to_dict(),
            "enabled": bool(self.config.enabled),
            "symmetric_conductance": bool(self.config.symmetric_conductance),
            "exchange_formula": (
                "s_i_bar = s_i + 0.5 * sum_j C_ij * (s_j - s_i)"
            ),
            "ablation_support": {
                "use_kirchhoff_exchange_false_identity_pass_through": True,
            },
        }


__all__ = [
    "KirchhoffExchangeConfig",
    "KirchhoffExchangeOutput",
    "PairwiseConductance",
    "KirchhoffExchange",
    "build_kirchhoff_exchange_config",
]