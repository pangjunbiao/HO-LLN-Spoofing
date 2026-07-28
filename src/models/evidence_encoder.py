"""
Evidence encoder for the proposed AV-GPS spoofing detector.

Step 11 module 1:
Evidence encoder maps the causal xi_t evidence vector into three evidence states:

1. Instantaneous residual branch:
   [eta_east, eta_north, q_t]

2. Residual evolution branch:
   [eta_dot_east, eta_dot_north, eta_ddot_east, eta_ddot_north]

3. Persistence / weak accumulation branch:
   [accum_log, nu]

This follows the methodology:

s_t^I = sigmoid(W_I [eta_t, q_t] + b_I)
s_t^E = sigmoid(W_E [dot_eta_t, ddot_eta_t] + b_E)
s_t^P = sigmoid(W_P [accum_log_t, nu_t] + b_P)

Important:
- The input must be the 9 scaled xi columns from Step 8/9 only.
- Raw shortcut features must never enter this module.
- Ablation flags are handled here:
    use_residual_evolution = False -> evolution branch evidence is zeroed
    use_weak_accumulation = False -> accum_log evidence is zeroed, nu remains
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


DEFAULT_XI_FEATURE_COLUMNS = [
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


@dataclass
class EvidenceEncoderConfig:
    """Configuration for EvidenceEncoder."""

    input_dim: int = 9

    instantaneous_input_dim: int = 3
    evolution_input_dim: int = 4
    persistence_input_dim: int = 2

    instantaneous_branch_dim: int = 32
    evolution_branch_dim: int = 32
    persistence_branch_dim: int = 32

    dropout: float = 0.10

    use_residual_evolution: bool = True
    use_weak_accumulation: bool = True

    require_exact_input_dim: bool = True

    feature_columns: Tuple[str, ...] = tuple(DEFAULT_XI_FEATURE_COLUMNS)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceEncoderOutput:
    """Output container for evidence encoder."""

    instantaneous_state: Tensor
    evolution_state: Tensor
    persistence_state: Tensor

    instantaneous_input: Tensor
    evolution_input: Tensor
    persistence_input: Tensor

    padding_mask: Optional[Tensor]

    config: Dict[str, Any]

    def state_dict_like(self) -> Dict[str, Tensor]:
        return {
            "instantaneous_state": self.instantaneous_state,
            "evolution_state": self.evolution_state,
            "persistence_state": self.persistence_state,
            "instantaneous_input": self.instantaneous_input,
            "evolution_input": self.evolution_input,
            "persistence_input": self.persistence_input,
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


def build_evidence_encoder_config(config: Optional[Mapping[str, Any]] = None) -> EvidenceEncoderConfig:
    """
    Build EvidenceEncoderConfig from project config.

    Supports the existing model.yaml structure:
    model.proposed.kirchhoff_high_order.instantaneous_branch_dim
    model.proposed.kirchhoff_high_order.evolution_branch_dim
    model.proposed.kirchhoff_high_order.persistence_branch_dim
    model.proposed.dropout
    model.proposed.use_residual_evolution
    model.proposed.use_weak_accumulation
    model.input.recommended_model_input_columns
    """
    if config is None:
        return EvidenceEncoderConfig()

    feature_columns = _get_by_path(
        config,
        "model.input.recommended_model_input_columns",
        DEFAULT_XI_FEATURE_COLUMNS,
    )

    return EvidenceEncoderConfig(
        input_dim=int(_get_by_path(config, "model.input.input_dim", 9)),
        instantaneous_input_dim=3,
        evolution_input_dim=4,
        persistence_input_dim=2,
        instantaneous_branch_dim=int(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.instantaneous_branch_dim",
                32,
            )
        ),
        evolution_branch_dim=int(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.evolution_branch_dim",
                32,
            )
        ),
        persistence_branch_dim=int(
            _get_by_path(
                config,
                "model.proposed.kirchhoff_high_order.persistence_branch_dim",
                32,
            )
        ),
        dropout=float(_get_by_path(config, "model.proposed.dropout", 0.10)),
        use_residual_evolution=bool(
            _get_by_path(config, "model.proposed.use_residual_evolution", True)
        ),
        use_weak_accumulation=bool(
            _get_by_path(config, "model.proposed.use_weak_accumulation", True)
        ),
        require_exact_input_dim=True,
        feature_columns=tuple(str(col) for col in feature_columns),
    )


def _ensure_3d_sequence_tensor(x: Tensor) -> Tensor:
    """
    Ensure tensor shape is [batch, time, feature].

    If x is [batch, feature], it is treated as a one-step sequence.
    """
    if not torch.is_tensor(x):
        raise TypeError("EvidenceEncoder input must be a torch.Tensor.")

    if x.ndim == 2:
        return x.unsqueeze(1)

    if x.ndim == 3:
        return x

    raise ValueError(
        f"EvidenceEncoder expected input with shape [B,F] or [B,T,F], got {tuple(x.shape)}."
    )


def _ensure_padding_mask(
    x: Tensor,
    padding_mask: Optional[Tensor],
) -> Optional[Tensor]:
    """
    Normalize padding mask to float tensor with shape [B,T].

    Padding mask convention:
    - 1.0 means real row
    - 0.0 means padded row
    """
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

    if padding_mask.shape[0] != x.shape[0] or padding_mask.shape[1] != x.shape[1]:
        raise ValueError(
            "padding_mask shape mismatch. "
            f"Expected {(x.shape[0], x.shape[1])}, got {tuple(padding_mask.shape)}."
        )

    return padding_mask.to(device=x.device, dtype=x.dtype)


def _apply_padding_mask_to_state(state: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
    """Zero padded positions in a branch state."""
    if padding_mask is None:
        return state

    return state * padding_mask.unsqueeze(-1)


class EvidenceEncoder(nn.Module):
    """
    Evidence encoder for xi_t.

    Input feature order must be:

    0 xi_eta_east_scaled
    1 xi_eta_north_scaled
    2 xi_eta_dot_east_scaled
    3 xi_eta_dot_north_scaled
    4 xi_eta_ddot_east_scaled
    5 xi_eta_ddot_north_scaled
    6 xi_q_scaled
    7 xi_accum_log_scaled
    8 xi_nu
    """

    eta_east_index = 0
    eta_north_index = 1
    eta_dot_east_index = 2
    eta_dot_north_index = 3
    eta_ddot_east_index = 4
    eta_ddot_north_index = 5
    q_index = 6
    accum_log_index = 7
    nu_index = 8

    def __init__(self, config: Optional[EvidenceEncoderConfig] = None) -> None:
        super().__init__()

        self.config = config or EvidenceEncoderConfig()

        if self.config.input_dim != 9:
            raise ValueError(
                "EvidenceEncoder currently expects the locked 9-dimensional xi input. "
                f"Got input_dim={self.config.input_dim}."
            )

        self.input_dropout = nn.Dropout(p=float(self.config.dropout))

        self.instantaneous_encoder = nn.Linear(
            self.config.instantaneous_input_dim,
            self.config.instantaneous_branch_dim,
        )
        self.evolution_encoder = nn.Linear(
            self.config.evolution_input_dim,
            self.config.evolution_branch_dim,
        )
        self.persistence_encoder = nn.Linear(
            self.config.persistence_input_dim,
            self.config.persistence_branch_dim,
        )

        self.activation = nn.Sigmoid()

        self.reset_parameters()

    @classmethod
    def from_project_config(cls, config: Mapping[str, Any]) -> "EvidenceEncoder":
        """Construct from full project config."""
        return cls(build_evidence_encoder_config(config))

    @property
    def branch_dims(self) -> Dict[str, int]:
        """Return output dimensions for all evidence branches."""
        return {
            "instantaneous": int(self.config.instantaneous_branch_dim),
            "evolution": int(self.config.evolution_branch_dim),
            "persistence": int(self.config.persistence_branch_dim),
        }

    @property
    def state_dim(self) -> int:
        """
        Shared state dimension when all branch dims match.

        Kirchhoff exchange expects equal branch dimensions.
        """
        dims = set(self.branch_dims.values())

        if len(dims) != 1:
            raise ValueError(
                "Kirchhoff exchange requires equal branch dimensions, got "
                f"{self.branch_dims}."
            )

        return int(next(iter(dims)))

    def reset_parameters(self) -> None:
        """Initialize branch encoders."""
        for layer in [
            self.instantaneous_encoder,
            self.evolution_encoder,
            self.persistence_encoder,
        ]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def validate_input_dim(self, x: Tensor) -> None:
        """Validate feature dimension."""
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                "EvidenceEncoder input dimension mismatch. "
                f"Expected {self.config.input_dim}, got {x.shape[-1]}. "
                "Make sure Step 9 passes only the 9 scaled xi feature columns."
            )

    def split_xi_features(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Split xi_t into branch inputs.

        Returns tensors with shapes:
        - instantaneous: [B,T,3]
        - evolution: [B,T,4]
        - persistence: [B,T,2]
        """
        self.validate_input_dim(x)

        eta_east = x[..., self.eta_east_index : self.eta_east_index + 1]
        eta_north = x[..., self.eta_north_index : self.eta_north_index + 1]
        q = x[..., self.q_index : self.q_index + 1]

        eta_dot_east = x[..., self.eta_dot_east_index : self.eta_dot_east_index + 1]
        eta_dot_north = x[..., self.eta_dot_north_index : self.eta_dot_north_index + 1]
        eta_ddot_east = x[..., self.eta_ddot_east_index : self.eta_ddot_east_index + 1]
        eta_ddot_north = x[..., self.eta_ddot_north_index : self.eta_ddot_north_index + 1]

        accum_log = x[..., self.accum_log_index : self.accum_log_index + 1]
        nu = x[..., self.nu_index : self.nu_index + 1]

        instantaneous = torch.cat([eta_east, eta_north, q], dim=-1)
        evolution = torch.cat(
            [
                eta_dot_east,
                eta_dot_north,
                eta_ddot_east,
                eta_ddot_north,
            ],
            dim=-1,
        )
        persistence = torch.cat([accum_log, nu], dim=-1)

        return {
            "instantaneous": instantaneous,
            "evolution": evolution,
            "persistence": persistence,
        }

    def apply_ablation_masks(self, branch_inputs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply strict ablation switches before branch encoding.

        no_residual_evolution:
            zeroes the full evolution input branch.

        no_weak_accumulation:
            zeroes the full persistence / weak-accumulation input branch.

        Important:
            This pre-encoding mask is not enough by itself because
            Linear(0) + sigmoid gives 0.5. Therefore the same disabled
            branches are also zeroed after encoding in apply_ablation_state_masks().
        """
        instantaneous = branch_inputs["instantaneous"]
        evolution = branch_inputs["evolution"]
        persistence = branch_inputs["persistence"]

        if not self.config.use_residual_evolution:
            evolution = torch.zeros_like(evolution)

        if not self.config.use_weak_accumulation:
            persistence = torch.zeros_like(persistence)

        return {
            "instantaneous": instantaneous,
            "evolution": evolution,
            "persistence": persistence,
        }

    def encode_branch_inputs(self, branch_inputs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        """Encode branch inputs into sigmoid evidence states."""
        instantaneous_input = self.input_dropout(branch_inputs["instantaneous"])
        evolution_input = self.input_dropout(branch_inputs["evolution"])
        persistence_input = self.input_dropout(branch_inputs["persistence"])

        instantaneous_state = self.activation(self.instantaneous_encoder(instantaneous_input))
        evolution_state = self.activation(self.evolution_encoder(evolution_input))
        persistence_state = self.activation(self.persistence_encoder(persistence_input))

        return {
            "instantaneous_state": instantaneous_state,
            "evolution_state": evolution_state,
            "persistence_state": persistence_state,
        }

    def apply_ablation_state_masks(self, encoded: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply true component-removal masks after encoding.

        This is required because zero input through Linear + Sigmoid produces
        approximately 0.5 evidence, not zero evidence.
        """
        instantaneous_state = encoded["instantaneous_state"]
        evolution_state = encoded["evolution_state"]
        persistence_state = encoded["persistence_state"]

        if not self.config.use_residual_evolution:
            evolution_state = torch.zeros_like(evolution_state)

        if not self.config.use_weak_accumulation:
            persistence_state = torch.zeros_like(persistence_state)

        return {
            "instantaneous_state": instantaneous_state,
            "evolution_state": evolution_state,
            "persistence_state": persistence_state,
        }

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> EvidenceEncoderOutput:
        """
        Forward pass.

        Args:
            x:
                Tensor with shape [B,T,9] or [B,9].
            padding_mask:
                Optional tensor [B,T], 1 for real rows and 0 for padded rows.

        Returns:
            EvidenceEncoderOutput with branch states and branch inputs.
        """
        x = _ensure_3d_sequence_tensor(x)
        padding_mask = _ensure_padding_mask(x, padding_mask)

        self.validate_input_dim(x)

        raw_branch_inputs = self.split_xi_features(x)
        branch_inputs = self.apply_ablation_masks(raw_branch_inputs)

        encoded = self.encode_branch_inputs(branch_inputs)
        encoded = self.apply_ablation_state_masks(encoded)

        instantaneous_state = _apply_padding_mask_to_state(
            encoded["instantaneous_state"],
            padding_mask,
        )
        evolution_state = _apply_padding_mask_to_state(
            encoded["evolution_state"],
            padding_mask,
        )
        persistence_state = _apply_padding_mask_to_state(
            encoded["persistence_state"],
            padding_mask,
        )

        return EvidenceEncoderOutput(
            instantaneous_state=instantaneous_state,
            evolution_state=evolution_state,
            persistence_state=persistence_state,
            instantaneous_input=branch_inputs["instantaneous"],
            evolution_input=branch_inputs["evolution"],
            persistence_input=branch_inputs["persistence"],
            padding_mask=padding_mask,
            config=self.config.to_dict(),
        )

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.config.input_dim}, "
            f"branch_dims={self.branch_dims}, "
            f"use_residual_evolution={self.config.use_residual_evolution}, "
            f"use_weak_accumulation={self.config.use_weak_accumulation}"
        )

    def module_summary(self) -> Dict[str, Any]:
        """JSON-safe module summary for Step-11 inspection."""
        return {
            "module": "EvidenceEncoder",
            "config": self.config.to_dict(),
            "branch_dims": self.branch_dims,
            "shared_state_dim": self.state_dim,
            "input_feature_columns": list(self.config.feature_columns),
            "ablation_support": {
                "use_residual_evolution": self.config.use_residual_evolution,
                "use_weak_accumulation": self.config.use_weak_accumulation,
            },
            "raw_shortcut_columns_used": False,
        }


__all__ = [
    "DEFAULT_XI_FEATURE_COLUMNS",
    "EvidenceEncoderConfig",
    "EvidenceEncoderOutput",
    "EvidenceEncoder",
    "build_evidence_encoder_config",
]