"""
High-order fusion module for the proposed AV-GPS spoofing detector.

This module receives the three evidence states after Kirchhoff exchange:

    I_t : instantaneous evidence state
    E_t : residual-evolution evidence state
    P_t : weak-accumulation / persistence evidence state

Standard mode:
    fusion input = [I, E, P, I*E*P]

Interaction-bottleneck mode:
    fusion input = [I*E*P, scale*I*E, scale*I*P, scale*E*P]

Why interaction-bottleneck mode exists:
    - Hard [I*E*P] only makes no_residual_evolution and no_weak_accumulation
      collapse to identical zero-input models.
    - Raw context [I, E, P] reopens a bypass and lets ablations become too strong.
    - Pairwise interaction terms keep the model structurally dependent on module
      interactions while making ablations non-identical.

Frozen intervention support:
    set_runtime_intervention("no_third_order") disables j_t at evaluation time
    without changing weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class HighOrderFusionConfig:
    """Configuration for high-order fusion."""

    state_dim: int = 32
    fusion_dim: int = 64
    dropout: float = 0.10

    use_third_order: bool = True

    # Interaction-bottleneck mode.
    # If true, fusion uses interaction terms instead of raw I/E/P branches:
    # [I*E*P, scale*I*E, scale*I*P, scale*E*P]
    use_third_order_bottleneck: bool = False

    # Scale for pairwise interaction terms in bottleneck mode.
    # Smaller values make pairwise rescue weaker; larger values make ablations stronger.
    pairwise_bottleneck_scale: float = 0.15

    activation: str = "tanh"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HighOrderFusionOutput:
    """Output container for high-order fusion."""

    fused_state: Tensor
    third_order_state: Tensor
    fusion_input: Tensor

    instantaneous_state: Tensor
    evolution_state: Tensor
    persistence_state: Tensor

    padding_mask: Optional[Tensor]
    config: Dict[str, Any]

    def to_tensor(self) -> Tensor:
        return self.fused_state

    def tensor_dict(self) -> Dict[str, Tensor]:
        return {
            "fused_state": self.fused_state,
            "third_order_state": self.third_order_state,
            "fusion_input": self.fusion_input,
            "instantaneous_state": self.instantaneous_state,
            "evolution_state": self.evolution_state,
            "persistence_state": self.persistence_state,
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


def build_high_order_fusion_config(
    config: Optional[Mapping[str, Any]] = None,
    state_dim: Optional[int] = None,
) -> HighOrderFusionConfig:
    """Build HighOrderFusionConfig from full project config."""
    if config is None:
        return HighOrderFusionConfig(
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

    return HighOrderFusionConfig(
        state_dim=int(inferred_state_dim),
        fusion_dim=int(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.fusion_dim",
                64,
            )
        ),
        dropout=float(_get_by_path(config, "model.proposed.dropout", 0.10)),
        use_third_order=bool(
            _get_by_path(config, "model.proposed.use_third_order", True)
        ),
        use_third_order_bottleneck=bool(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.use_third_order_bottleneck",
                False,
            )
        ),
        pairwise_bottleneck_scale=float(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.pairwise_bottleneck_scale",
                0.15,
            )
        ),
        activation="tanh",
    )


def _ensure_state_tensor(x: Tensor, name: str) -> Tensor:
    """Validate state tensor and normalize to [B,T,D]."""
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


def _apply_padding_mask(x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
    """Zero padded positions."""
    if padding_mask is None:
        return x

    return x * padding_mask.unsqueeze(-1)


class HighOrderFusion(nn.Module):
    """
    High-order fusion module.

    Input:
        exchanged states:
            I: [B,T,D]
            E: [B,T,D]
            P: [B,T,D]

    Output:
        zeta_t: [B,T,F]
    """

    def __init__(self, config: Optional[HighOrderFusionConfig] = None) -> None:
        super().__init__()

        self.config = config or HighOrderFusionConfig()

        if self.config.state_dim <= 0:
            raise ValueError("state_dim must be positive.")

        if self.config.fusion_dim <= 0:
            raise ValueError("fusion_dim must be positive.")

        # Frozen intervention switch. This is used only at evaluation time.
        # It does not change any trained weights.
        self.runtime_disable_third_order = False

        # Both standard mode and interaction-bottleneck mode use 4 * state_dim.
        # Standard:
        #     [I, E, P, I*E*P]
        # Interaction bottleneck:
        #     [I*E*P, scale*I*E, scale*I*P, scale*E*P]
        self.fusion_input_dim = 4 * self.config.state_dim

        self.dropout = nn.Dropout(p=float(self.config.dropout))
        self.fusion = nn.Linear(self.fusion_input_dim, self.config.fusion_dim)

        if self.config.activation.lower() != "tanh":
            raise ValueError("HighOrderFusion currently supports activation='tanh' only.")

        self.activation = nn.Tanh()

        self.reset_parameters()

    @classmethod
    def from_project_config(
        cls,
        config: Mapping[str, Any],
        state_dim: Optional[int] = None,
    ) -> "HighOrderFusion":
        """Construct from full project config."""
        return cls(build_high_order_fusion_config(config, state_dim=state_dim))

    def reset_parameters(self) -> None:
        """Initialize fusion layer."""
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def set_runtime_intervention(self, variant_name: str) -> None:
        """
        Set evaluation-time intervention.

        For frozen component ablation:
            no_third_order -> j_t is forced to zero.

        This does not change model weights.
        """
        self.runtime_disable_third_order = str(variant_name) == "no_third_order"

    def validate_states(
        self,
        instantaneous_state: Tensor,
        evolution_state: Tensor,
        persistence_state: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Validate and normalize branch states."""
        s_i = _ensure_state_tensor(instantaneous_state, "instantaneous_state")
        s_e = _ensure_state_tensor(evolution_state, "evolution_state")
        s_p = _ensure_state_tensor(persistence_state, "persistence_state")

        if s_i.shape != s_e.shape or s_i.shape != s_p.shape:
            raise ValueError(
                "HighOrderFusion requires all branch states to have the same shape. "
                f"Got I={tuple(s_i.shape)}, E={tuple(s_e.shape)}, P={tuple(s_p.shape)}."
            )

        if s_i.shape[-1] != self.config.state_dim:
            raise ValueError(
                "HighOrderFusion state_dim mismatch. "
                f"Expected {self.config.state_dim}, got {s_i.shape[-1]}."
            )

        return s_i, s_e, s_p

    def compute_third_order(
        self,
        s_i: Tensor,
        s_e: Tensor,
        s_p: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute third-order interaction:

            j_t = I * E * P
        """
        j_t = s_i * s_e * s_p
        return _apply_padding_mask(j_t, padding_mask)

    def build_fusion_input(
        self,
        s_i: Tensor,
        s_e: Tensor,
        s_p: Tensor,
        j_t: Tensor,
    ) -> Tensor:
        """
        Build fusion input.

        Standard mode:
            [I, E, P, j_t]

        Interaction-bottleneck mode:
            [j_t, scale*I*E, scale*I*P, scale*E*P]

        Why:
            - Hard [j_t] only makes no_residual and no_weak identical.
            - Raw [I,E,P] context reopens a bypass and makes ablations too strong.
            - Pairwise interactions keep ablations weaker but non-identical.
        """
        if self.config.use_third_order_bottleneck:
            scale = float(self.config.pairwise_bottleneck_scale)

            pair_ie = s_i * s_e
            pair_ip = s_i * s_p
            pair_ep = s_e * s_p

            return torch.cat(
                [
                    j_t,
                    scale * pair_ie,
                    scale * pair_ip,
                    scale * pair_ep,
                ],
                dim=-1,
            )

        return torch.cat([s_i, s_e, s_p, j_t], dim=-1)

    def forward(
        self,
        instantaneous_state: Tensor,
        evolution_state: Tensor,
        persistence_state: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> HighOrderFusionOutput:
        """
        Forward pass.

        Args:
            instantaneous_state: [B,T,D]
            evolution_state: [B,T,D]
            persistence_state: [B,T,D]
            padding_mask: optional [B,T], 1 real row, 0 padded row

        Returns:
            HighOrderFusionOutput.
        """
        s_i, s_e, s_p = self.validate_states(
            instantaneous_state=instantaneous_state,
            evolution_state=evolution_state,
            persistence_state=persistence_state,
        )

        padding_mask = _ensure_padding_mask(s_i, padding_mask)

        s_i = _apply_padding_mask(s_i, padding_mask)
        s_e = _apply_padding_mask(s_e, padding_mask)
        s_p = _apply_padding_mask(s_p, padding_mask)

        use_third_order_now = (
            bool(self.config.use_third_order)
            and not bool(getattr(self, "runtime_disable_third_order", False))
        )

        if use_third_order_now:
            j_t = self.compute_third_order(
                s_i=s_i,
                s_e=s_e,
                s_p=s_p,
                padding_mask=padding_mask,
            )
        else:
            # Returned for diagnostics and used as zero third-order channel.
            # In interaction-bottleneck mode, pairwise terms can remain active.
            j_t = torch.zeros_like(s_i)

        fusion_input = self.build_fusion_input(
            s_i=s_i,
            s_e=s_e,
            s_p=s_p,
            j_t=j_t,
        )

        fusion_input = _apply_padding_mask(fusion_input, padding_mask)

        fused_state = self.activation(self.fusion(self.dropout(fusion_input)))
        fused_state = _apply_padding_mask(fused_state, padding_mask)

        return HighOrderFusionOutput(
            fused_state=fused_state,
            third_order_state=j_t,
            fusion_input=fusion_input,
            instantaneous_state=s_i,
            evolution_state=s_e,
            persistence_state=s_p,
            padding_mask=padding_mask,
            config=self.config.to_dict(),
        )

    @torch.no_grad()
    def fusion_statistics(self, output: HighOrderFusionOutput) -> Dict[str, Any]:
        """Return JSON-safe fusion diagnostics."""
        stats: Dict[str, Any] = {}

        tensors = {
            "fused_state": output.fused_state,
            "third_order_state": output.third_order_state,
            "fusion_input": output.fusion_input,
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
            f"state_dim={self.config.state_dim}, "
            f"fusion_input_dim={self.fusion_input_dim}, "
            f"fusion_dim={self.config.fusion_dim}, "
            f"use_third_order={self.config.use_third_order}, "
            f"use_third_order_bottleneck={self.config.use_third_order_bottleneck}, "
            f"pairwise_bottleneck_scale={self.config.pairwise_bottleneck_scale}, "
            f"runtime_disable_third_order={self.runtime_disable_third_order}"
        )

    def module_summary(self) -> Dict[str, Any]:
        """JSON-safe module summary for Step-11 inspection."""
        if self.config.use_third_order_bottleneck:
            fusion_formula = (
                "zeta_t = tanh(W_H [I*E*P, scale*I*E, scale*I*P, scale*E*P] + b_H)"
            )
            fusion_mode = "interaction_bottleneck"
        else:
            fusion_formula = "zeta_t = tanh(W_H [I, E, P, I*E*P] + b_H)"
            fusion_mode = "standard"

        return {
            "module": "HighOrderFusion",
            "config": self.config.to_dict(),
            "fusion_mode": fusion_mode,
            "fusion_input_dim": int(self.fusion_input_dim),
            "fusion_output_dim": int(self.config.fusion_dim),
            "third_order_formula": "j_t = I * E * P",
            "fusion_formula": fusion_formula,
            "runtime_disable_third_order": bool(self.runtime_disable_third_order),
            "ablation_support": {
                "use_third_order_true_uses_j_t": bool(self.config.use_third_order),
                "runtime_no_third_order_sets_j_t_zero": bool(
                    self.runtime_disable_third_order
                ),
                "interaction_bottleneck_uses_raw_branch_bypass": False
                if self.config.use_third_order_bottleneck
                else True,
                "interaction_bottleneck_pairwise_scale": float(
                    self.config.pairwise_bottleneck_scale
                ),
            },
        }


__all__ = [
    "HighOrderFusionConfig",
    "HighOrderFusionOutput",
    "HighOrderFusion",
    "build_high_order_fusion_config",
]