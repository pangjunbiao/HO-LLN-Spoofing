"""
Full proposed model for causal GNSS spoofing detection.

Step 11 architecture:

    1. EvidenceEncoder
    2. KirchhoffExchange
    3. HighOrderFusion with optional third-order / interaction bottleneck
    4. Temporal block:
       - full: liquid_second_order
       - no_liquid_dynamics intervention: frozen identity temporal bypass
    5. OutputHead

Forward path:
    xi_t
        -> evidence encoder
        -> Kirchhoff exchange
        -> optional frozen runtime intervention
        -> strict residual pipeline state construction
        -> third-order / high-order fusion
        -> temporal dynamics or frozen no-liquid bypass
        -> output head
        -> p_hat_t

Important:
    - This model expects only the 9 Step-8/Step-9 scaled xi features.
    - It does not apply thresholding.
    - It does not apply N_p alarm rules.
    - Threshold and alarm confirmation are handled by evaluation code.
    - Runtime interventions are for frozen ablation evaluation only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from src.models.evidence_encoder import (
    DEFAULT_XI_FEATURE_COLUMNS,
    EvidenceEncoder,
    EvidenceEncoderOutput,
    build_evidence_encoder_config,
)
from src.models.kirchhoff_exchange import (
    KirchhoffExchange,
    KirchhoffExchangeOutput,
    build_kirchhoff_exchange_config,
)
from src.models.high_order_fusion import (
    HighOrderFusion,
    HighOrderFusionOutput,
    build_high_order_fusion_config,
)
from src.models.temporal_blocks import (
    TemporalBlockOutput,
    build_temporal_block_config,
    create_temporal_block,
    temporal_output_statistics,
)
from src.models.output_head import (
    OutputHead,
    OutputHeadResult,
    build_output_head_config,
)


@dataclass
class ProposedModelConfig:
    """Top-level proposed model configuration summary."""

    model_name: str = "KirchhoffLiquidSpoofDetector"
    model_type: str = "proposed"

    input_dim: int = 9
    feature_columns: Tuple[str, ...] = tuple(DEFAULT_XI_FEATURE_COLUMNS)

    hidden_dim: int = 64
    branch_state_dim: int = 32
    fusion_dim: int = 64
    dropout: float = 0.10

    use_residual_evolution: bool = True
    use_weak_accumulation: bool = True
    use_kirchhoff_exchange: bool = True
    use_third_order: bool = True
    temporal_block: str = "liquid_second_order"

    output_head_input: str = "hidden_and_velocity"

    causal_history_only: bool = True
    raw_shortcut_columns_used: bool = False
    threshold_applied_inside_model: bool = False
    alarm_rule_applied_inside_model: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProposedModelOutput:
    """Forward output from ProposedSpoofingModel."""

    logits: Tensor
    probabilities: Tensor

    hidden_sequence: Tensor
    velocity_sequence: Tensor

    padding_mask: Optional[Tensor]
    loss_mask: Optional[Tensor]
    labels: Optional[Tensor]
    delta_t: Optional[Tensor]

    evidence_output: EvidenceEncoderOutput
    exchange_output: KirchhoffExchangeOutput
    fusion_output: HighOrderFusionOutput
    temporal_output: TemporalBlockOutput
    head_output: OutputHeadResult

    model_config: Dict[str, Any]

    def prediction_tensor(self) -> Tensor:
        """Return p_hat_t with shape [B,T]."""
        return self.probabilities

    def logits_tensor(self) -> Tensor:
        """Return logits with shape [B,T]."""
        return self.logits

    def shape_summary(self) -> Dict[str, Any]:
        """JSON-safe shape summary."""
        return {
            "logits": list(self.logits.shape),
            "probabilities": list(self.probabilities.shape),
            "hidden_sequence": list(self.hidden_sequence.shape),
            "velocity_sequence": list(self.velocity_sequence.shape),
            "padding_mask": None if self.padding_mask is None else list(self.padding_mask.shape),
            "loss_mask": None if self.loss_mask is None else list(self.loss_mask.shape),
            "labels": None if self.labels is None else list(self.labels.shape),
            "delta_t": None if self.delta_t is None else list(self.delta_t.shape),
        }

    def compact_dict(self) -> Dict[str, Any]:
        """JSON-safe compact output summary."""
        return {
            "shape_summary": self.shape_summary(),
            "model_config": self.model_config,
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


def build_proposed_model_config(config: Optional[Mapping[str, Any]] = None) -> ProposedModelConfig:
    """Build top-level ProposedModelConfig from full project config."""
    if config is None:
        return ProposedModelConfig()

    feature_columns = tuple(
        str(col)
        for col in _get_by_path(
            config,
            "model.input.recommended_model_input_columns",
            DEFAULT_XI_FEATURE_COLUMNS,
        )
    )

    hidden_dim = int(_get_by_path(config, "model.proposed.liquid_second_order.hidden_dim", 64))
    branch_dim = int(
        _get_by_path(config, "model.proposed.kirchhoff_high_order.instantaneous_branch_dim", 32)
    )
    fusion_dim = int(
        _get_by_path(config, "model.proposed.kirchhoff_high_order.fusion_dim", 64)
    )

    return ProposedModelConfig(
        model_name=str(_get_by_path(config, "model.name", "KirchhoffLiquidSpoofDetector")),
        model_type=str(_get_by_path(config, "model.proposed.model_type", "proposed")),
        input_dim=int(_get_by_path(config, "model.input.input_dim", 9)),
        feature_columns=feature_columns,
        hidden_dim=hidden_dim,
        branch_state_dim=branch_dim,
        fusion_dim=fusion_dim,
        dropout=float(_get_by_path(config, "model.proposed.dropout", 0.10)),
        use_residual_evolution=bool(
            _get_by_path(config, "model.proposed.use_residual_evolution", True)
        ),
        use_weak_accumulation=bool(
            _get_by_path(config, "model.proposed.use_weak_accumulation", True)
        ),
        use_kirchhoff_exchange=bool(
            _get_by_path(config, "model.proposed.use_kirchhoff_exchange", True)
        ),
        use_third_order=bool(
            _get_by_path(config, "model.proposed.use_third_order", True)
        ),
        temporal_block=str(
            _get_by_path(config, "model.proposed.temporal_block", "liquid_second_order")
        ),
        output_head_input=str(
            _get_by_path(config, "model.proposed.output_head.input", "hidden_and_velocity")
        ),
        causal_history_only=bool(
            _get_by_path(config, "model.input.rules.causal_history_only", True)
        ),
        raw_shortcut_columns_used=False,
        threshold_applied_inside_model=False,
        alarm_rule_applied_inside_model=False,
    )


def _ensure_sequence_tensor(x: Tensor, name: str = "x") -> Tensor:
    """Ensure tensor shape [B,T,F]."""
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if x.ndim == 2:
        return x.unsqueeze(1)

    if x.ndim == 3:
        return x

    raise ValueError(f"{name} must have shape [B,F] or [B,T,F], got {tuple(x.shape)}.")


def _ensure_mask(
    reference: Tensor,
    mask: Optional[Tensor],
    name: str,
) -> Optional[Tensor]:
    """Normalize mask to [B,T]."""
    if mask is None:
        return None

    if not torch.is_tensor(mask):
        raise TypeError(f"{name} must be a torch.Tensor or None.")

    if mask.ndim == 1:
        mask = mask.unsqueeze(1)

    if mask.ndim != 2:
        raise ValueError(f"{name} must have shape [B,T], got {tuple(mask.shape)}.")

    expected = (reference.shape[0], reference.shape[1])
    got = tuple(mask.shape)

    if got != expected:
        raise ValueError(f"{name} shape mismatch. Expected {expected}, got {got}.")

    return mask.to(device=reference.device, dtype=reference.dtype)


def _ensure_delta_t(reference: Tensor, delta_t: Optional[Tensor]) -> Optional[Tensor]:
    """Normalize delta_t to [B,T] if provided."""
    if delta_t is None:
        return None

    if not torch.is_tensor(delta_t):
        raise TypeError("delta_t must be a torch.Tensor or None.")

    if delta_t.ndim == 1:
        delta_t = delta_t.unsqueeze(1)

    if delta_t.ndim == 3 and delta_t.shape[-1] == 1:
        delta_t = delta_t.squeeze(-1)

    if delta_t.ndim != 2:
        raise ValueError(f"delta_t must have shape [B,T], got {tuple(delta_t.shape)}.")

    expected = (reference.shape[0], reference.shape[1])
    got = tuple(delta_t.shape)

    if got != expected:
        raise ValueError(f"delta_t shape mismatch. Expected {expected}, got {got}.")

    return delta_t.to(device=reference.device, dtype=reference.dtype)


def _ensure_labels(reference: Tensor, labels: Optional[Tensor]) -> Optional[Tensor]:
    """Normalize labels to [B,T] long if provided."""
    if labels is None:
        return None

    if not torch.is_tensor(labels):
        raise TypeError("labels must be a torch.Tensor or None.")

    if labels.ndim == 1:
        labels = labels.unsqueeze(1)

    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B,T], got {tuple(labels.shape)}.")

    expected = (reference.shape[0], reference.shape[1])
    got = tuple(labels.shape)

    if got != expected:
        raise ValueError(f"labels shape mismatch. Expected {expected}, got {got}.")

    return labels.to(device=reference.device, dtype=torch.long)


class ProposedSpoofingModel(nn.Module):
    """
    Full proposed Kirchhoff-liquid spoofing detector.

    Expected input:
        x: [B,T,9]

    Output:
        p_hat_t: [B,T]
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()

        self.project_config = config
        self.model_config = build_proposed_model_config(config)

        self.validate_feature_contract()

        # 1. Evidence encoder
        evidence_config = build_evidence_encoder_config(config)
        self.evidence_encoder = EvidenceEncoder(evidence_config)

        state_dim = self.evidence_encoder.state_dim

        # 2. Kirchhoff exchange
        exchange_config = build_kirchhoff_exchange_config(config, state_dim=state_dim)
        self.kirchhoff_exchange = KirchhoffExchange(exchange_config)

        # 3-4. Third-order feature and high-order fusion
        fusion_config = build_high_order_fusion_config(config, state_dim=state_dim)
        self.high_order_fusion = HighOrderFusion(fusion_config)

        # 5. Temporal block: liquid or ablation replacement
        temporal_config = build_temporal_block_config(
            config=config,
            input_dim=fusion_config.fusion_dim,
            temporal_block_override=self.model_config.temporal_block,
        )
        self.temporal_block = create_temporal_block(
            config=temporal_config,
            project_config=config,
        )

        # 6. Output head
        output_head_config = build_output_head_config(
            config=config,
            hidden_dim=temporal_config.hidden_dim,
            velocity_dim=temporal_config.velocity_dim,
        )
        self.output_head = OutputHead(output_head_config)

        # Frozen evaluation-time intervention.
        # This is "full" during normal training and Step 13.
        # Frozen Step-16 intervention changes this without changing weights.
        self.runtime_intervention_variant = "full"

    @property
    def input_dim(self) -> int:
        return int(self.model_config.input_dim)

    @property
    def feature_columns(self) -> Tuple[str, ...]:
        return tuple(self.model_config.feature_columns)

    @property
    def temporal_block_name(self) -> str:
        return str(self.model_config.temporal_block)

    def set_runtime_intervention(self, variant_name: str) -> None:
        """
        Set frozen evaluation-time intervention.

        This does not change model weights.
        It only disables one component during forward evaluation.
        """
        allowed = {
            "full",
            "no_residual_evolution",
            "no_weak_accumulation",
            "no_kirchhoff_exchange",
            "no_third_order",
            "no_liquid_dynamics",
        }

        variant_name = str(variant_name)
        if variant_name not in allowed:
            raise ValueError(f"Unknown runtime intervention variant: {variant_name}")

        self.runtime_intervention_variant = variant_name

        if hasattr(self.high_order_fusion, "set_runtime_intervention"):
            self.high_order_fusion.set_runtime_intervention(variant_name)

    def _zero_exchange_state_pair(
        self,
        exchange_output,
        state_name: str,
        original_name: str,
    ):
        """
        Zero both exchanged and original branch states.

        This is required because residual bottleneck uses:
            residual = exchanged_state - original_state

        If only exchanged_state is zeroed, the residual becomes -original_state,
        which reintroduces the disabled branch as a negative signal.
        """
        state = getattr(exchange_output, state_name)
        setattr(exchange_output, state_name, torch.zeros_like(state))

        if hasattr(exchange_output, original_name):
            original_state = getattr(exchange_output, original_name)
            setattr(exchange_output, original_name, torch.zeros_like(original_state))

        return exchange_output

    def apply_runtime_intervention_masks(self, exchange_output):
        """
        Apply frozen ablation intervention after Kirchhoff exchange.

        Used only for frozen component-intervention ablation.
        """
        variant = str(getattr(self, "runtime_intervention_variant", "full"))

        if variant == "full":
            return exchange_output

        if variant == "no_kirchhoff_exchange":
            # Make exchange identity. Because build_strict_pipeline_fusion_states()
            # subtracts original from exchanged state, this yields zero residual.
            exchange_output.instantaneous_state = exchange_output.original_instantaneous_state
            exchange_output.evolution_state = exchange_output.original_evolution_state
            exchange_output.persistence_state = exchange_output.original_persistence_state
            return exchange_output

        if variant == "no_residual_evolution":
            return self._zero_exchange_state_pair(
                exchange_output,
                state_name="evolution_state",
                original_name="original_evolution_state",
            )

        if variant == "no_weak_accumulation":
            return self._zero_exchange_state_pair(
                exchange_output,
                state_name="persistence_state",
                original_name="original_persistence_state",
            )

        # no_third_order is handled inside HighOrderFusion.
        # no_liquid_dynamics is handled at the temporal block stage.
        return exchange_output

    def apply_strict_post_exchange_ablation_masks(self, exchange_output):
        """
        Enforce strict component-removal ablations after Kirchhoff exchange.

        This is used for config-controlled retrained ablations.
        Full model is unchanged.
        """
        if not self.model_config.use_residual_evolution:
            exchange_output.evolution_state = torch.zeros_like(
                exchange_output.evolution_state
            )

        if not self.model_config.use_weak_accumulation:
            exchange_output.persistence_state = torch.zeros_like(
                exchange_output.persistence_state
            )

        return exchange_output

    def build_strict_pipeline_fusion_states(self, exchange_output):
        """
        Build states for strict interaction-bottleneck fusion.

        Standard model:
            fusion receives exchanged states.

        Strict pipeline model:
            fusion receives exchange residuals:
                delta_I = I_after_exchange - I_before_exchange
                delta_E = E_after_exchange - E_before_exchange
                delta_P = P_after_exchange - P_before_exchange

        Important:
            Do NOT add raw original-state context here.
            Raw context reopens bypass paths and can make ablations too strong.
        """
        use_exchange_residual_bottleneck = bool(
            _get_by_path(
                self.project_config or {},
                "model.proposed.kirchhoff_high_order.use_exchange_residual_bottleneck",
                False,
            )
        )

        if not use_exchange_residual_bottleneck:
            return (
                exchange_output.instantaneous_state,
                exchange_output.evolution_state,
                exchange_output.persistence_state,
            )

        instantaneous_state = (
            exchange_output.instantaneous_state
            - exchange_output.original_instantaneous_state
        )
        evolution_state = (
            exchange_output.evolution_state
            - exchange_output.original_evolution_state
        )
        persistence_state = (
            exchange_output.persistence_state
            - exchange_output.original_persistence_state
        )

        return instantaneous_state, evolution_state, persistence_state

    def _last_valid_state(
        self,
        sequence: Tensor,
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        """Return last valid time step from a sequence."""
        if padding_mask is None:
            return sequence[:, -1, :]

        lengths = padding_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(sequence.shape[0], device=sequence.device)
        return sequence[batch_index, lengths, :]

    def build_runtime_no_liquid_temporal_output(
        self,
        fused_state: Tensor,
        padding_mask: Optional[Tensor],
        delta_t: Optional[Tensor],
    ) -> TemporalBlockOutput:
        """
        Frozen no_liquid_dynamics intervention.

        Bypasses the liquid temporal block and passes fused_state directly
        to the output head with zero velocity.
        """
        hidden_sequence = fused_state
        velocity_sequence = torch.zeros_like(hidden_sequence)

        if padding_mask is not None:
            hidden_sequence = hidden_sequence * padding_mask.unsqueeze(-1)
            velocity_sequence = velocity_sequence * padding_mask.unsqueeze(-1)

        final_hidden = self._last_valid_state(hidden_sequence, padding_mask)
        final_velocity = torch.zeros_like(final_hidden)

        return TemporalBlockOutput(
            hidden_sequence=hidden_sequence,
            velocity_sequence=velocity_sequence,
            final_hidden=final_hidden,
            final_velocity=final_velocity,
            temporal_block="frozen_no_liquid_identity",
            padding_mask=padding_mask,
            delta_t=delta_t,
            auxiliary={"runtime_intervention": "no_liquid_dynamics"},
            config={"runtime_intervention": "no_liquid_dynamics"},
        )

    def ablation_summary(self) -> Dict[str, Any]:
        """Return JSON-safe ablation/module-activation summary."""
        disabled_components = []

        if not self.model_config.use_residual_evolution:
            disabled_components.append("residual_evolution")

        if not self.model_config.use_weak_accumulation:
            disabled_components.append("weak_accumulation")

        if not self.model_config.use_kirchhoff_exchange:
            disabled_components.append("kirchhoff_exchange")

        if not self.model_config.use_third_order:
            disabled_components.append("third_order")

        if str(self.model_config.temporal_block) != "liquid_second_order":
            disabled_components.append("liquid_second_order_dynamics")

        runtime_intervention = str(
            getattr(self, "runtime_intervention_variant", "full")
        )

        return {
            "is_full_model": len(disabled_components) == 0,
            "disabled_components": disabled_components,
            "runtime_intervention_variant": runtime_intervention,
            "runtime_intervention_active": runtime_intervention != "full",
            "use_residual_evolution": bool(self.model_config.use_residual_evolution),
            "use_weak_accumulation": bool(self.model_config.use_weak_accumulation),
            "use_kirchhoff_exchange": bool(self.model_config.use_kirchhoff_exchange),
            "use_third_order": bool(self.model_config.use_third_order),
            "temporal_block": str(self.model_config.temporal_block),
            "threshold_inside_model": False,
            "alarm_rule_inside_model": False,
        }

    def validate_feature_contract(self) -> None:
        """
        Enforce locked Step-9 feature contract.

        The model must only consume the 9 scaled xi columns.
        """
        expected = tuple(DEFAULT_XI_FEATURE_COLUMNS)
        actual = tuple(self.model_config.feature_columns)

        if self.model_config.input_dim != 9:
            raise ValueError(
                f"Proposed model requires input_dim=9, got {self.model_config.input_dim}."
            )

        if actual != expected:
            raise ValueError(
                "Proposed model feature column contract violation.\n"
                f"Expected: {list(expected)}\n"
                f"Got:      {list(actual)}\n"
                "Step 11 onward must use only the official 9 scaled xi columns."
            )

    def forward(
        self,
        x: Union[Tensor, Mapping[str, Any]],
        delta_t: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        reset_state: Optional[Tensor] = None,
        initial_hidden: Optional[Tensor] = None,
        initial_velocity: Optional[Tensor] = None,
        return_intermediates: bool = True,
    ) -> ProposedModelOutput:
        """
        Forward pass.

        Args:
            x:
                Either tensor [B,T,9] or a Step-9 batch dictionary containing:
                - x
                - y
                - loss_mask
                - padding_mask
                - delta_t
                - reset_state
            delta_t:
                Optional [B,T].
            padding_mask:
                Optional [B,T], 1 real row, 0 padded row.
            loss_mask:
                Optional [B,T], usually xi_nu/loss mask.
            labels:
                Optional [B,T].
            reset_state:
                Optional [B], 1 means reset hidden state.
            initial_hidden:
                Optional [B,H].
            initial_velocity:
                Optional [B,H].
            return_intermediates:
                Kept for API compatibility. Intermediates are returned in dataclass.
        """
        if isinstance(x, Mapping):
            batch = x
            x_tensor = batch["x"]

            if delta_t is None:
                delta_t = batch.get("delta_t")

            if padding_mask is None:
                padding_mask = batch.get("padding_mask")

            if loss_mask is None:
                loss_mask = batch.get("loss_mask")

            if labels is None:
                labels = batch.get("y")

            if reset_state is None:
                reset_state = batch.get("reset_state")
        else:
            x_tensor = x

        x_tensor = _ensure_sequence_tensor(x_tensor, name="x")

        if x_tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input feature dimension mismatch. Expected {self.input_dim}, got {x_tensor.shape[-1]}."
            )

        padding_mask = _ensure_mask(x_tensor, padding_mask, name="padding_mask")
        loss_mask = _ensure_mask(x_tensor, loss_mask, name="loss_mask")
        delta_t = _ensure_delta_t(x_tensor, delta_t)
        labels = _ensure_labels(x_tensor, labels)

        # 1. Evidence encoder
        evidence_output = self.evidence_encoder(
            x=x_tensor,
            padding_mask=padding_mask,
        )

        # 2. Kirchhoff exchange
        exchange_output = self.kirchhoff_exchange(
            instantaneous_state=evidence_output.instantaneous_state,
            evolution_state=evidence_output.evolution_state,
            persistence_state=evidence_output.persistence_state,
            padding_mask=padding_mask,
        )

        # Config-controlled strict retrained-ablation enforcement.
        exchange_output = self.apply_strict_post_exchange_ablation_masks(
            exchange_output
        )

        # Frozen evaluation-time intervention.
        exchange_output = self.apply_runtime_intervention_masks(exchange_output)

        # Strict pipeline bottleneck:
        # optionally use Kirchhoff exchange residuals instead of pass-through states.
        fusion_instantaneous_state, fusion_evolution_state, fusion_persistence_state = (
            self.build_strict_pipeline_fusion_states(exchange_output)
        )

        # 3-4. Third-order and high-order fusion
        fusion_output = self.high_order_fusion(
            instantaneous_state=fusion_instantaneous_state,
            evolution_state=fusion_evolution_state,
            persistence_state=fusion_persistence_state,
            padding_mask=padding_mask,
        )

        # 5. Temporal block or frozen no-liquid intervention
        if str(getattr(self, "runtime_intervention_variant", "full")) == "no_liquid_dynamics":
            temporal_output = self.build_runtime_no_liquid_temporal_output(
                fused_state=fusion_output.fused_state,
                padding_mask=padding_mask,
                delta_t=delta_t,
            )
        else:
            temporal_output = self.temporal_block(
                zeta=fusion_output.fused_state,
                delta_t=delta_t,
                padding_mask=padding_mask,
                initial_hidden=initial_hidden,
                initial_velocity=initial_velocity,
                reset_state=reset_state,
            )

        # 6. Output head
        head_output = self.output_head(
            hidden_sequence=temporal_output.hidden_sequence,
            velocity_sequence=temporal_output.velocity_sequence,
            padding_mask=padding_mask,
        )

        return ProposedModelOutput(
            logits=head_output.logits,
            probabilities=head_output.probabilities,
            hidden_sequence=temporal_output.hidden_sequence,
            velocity_sequence=temporal_output.velocity_sequence,
            padding_mask=padding_mask,
            loss_mask=loss_mask,
            labels=labels,
            delta_t=delta_t,
            evidence_output=evidence_output,
            exchange_output=exchange_output,
            fusion_output=fusion_output,
            temporal_output=temporal_output,
            head_output=head_output,
            model_config=self.model_config.to_dict(),
        )

    @torch.no_grad()
    def predict_proba(
        self,
        batch_or_x: Union[Tensor, Mapping[str, Any]],
        **kwargs: Any,
    ) -> Tensor:
        """Return probabilities p_hat_t with shape [B,T]."""
        self.eval()
        output = self.forward(batch_or_x, **kwargs)
        return output.probabilities

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "non_trainable_parameters": int(total - trainable),
        }

    def module_summary(self) -> Dict[str, Any]:
        """JSON-safe model/module summary."""
        summary = {
            "model": "ProposedSpoofingModel",
            "model_config": self.model_config.to_dict(),
            "parameter_counts": self.count_parameters(),
            "input_contract": {
                "input_dim": self.input_dim,
                "feature_columns": list(self.feature_columns),
                "uses_only_scaled_xi": list(self.feature_columns) == list(DEFAULT_XI_FEATURE_COLUMNS),
                "raw_shortcut_columns_used": False,
            },
            "ablation_summary": self.ablation_summary(),
            "runtime_intervention_variant": str(
                getattr(self, "runtime_intervention_variant", "full")
            ),
            "modules": {
                "evidence_encoder": self.evidence_encoder.module_summary(),
                "kirchhoff_exchange": self.kirchhoff_exchange.module_summary(),
                "high_order_fusion": self.high_order_fusion.module_summary(),
                "temporal_block": (
                    self.temporal_block.module_summary()
                    if hasattr(self.temporal_block, "module_summary")
                    else {"module": str(type(self.temporal_block))}
                ),
                "output_head": self.output_head.module_summary(),
            },
            "evaluation_note": {
                "threshold_inside_model": False,
                "alarm_rule_inside_model": False,
                "step10_used_for_threshold_and_alarm": True,
                "synthetic_step10_theta_not_final_model_threshold": True,
                "runtime_intervention_for_frozen_ablation_only": True,
            },
        }

        return summary

    @torch.no_grad()
    def forward_diagnostics(self, output: ProposedModelOutput) -> Dict[str, Any]:
        """Collect JSON-safe forward diagnostics."""
        diagnostics: Dict[str, Any] = {
            "shape_summary": output.shape_summary(),
            "runtime_intervention_variant": str(
                getattr(self, "runtime_intervention_variant", "full")
            ),
            "probability_summary": self.output_head.output_statistics(output.head_output).get(
                "probabilities",
                {},
            ),
        }

        if hasattr(self.kirchhoff_exchange, "conductance_statistics"):
            diagnostics["conductance_statistics"] = self.kirchhoff_exchange.conductance_statistics(
                output.exchange_output
            )

        if hasattr(self.high_order_fusion, "fusion_statistics"):
            diagnostics["fusion_statistics"] = self.high_order_fusion.fusion_statistics(
                output.fusion_output
            )

        diagnostics["temporal_statistics"] = temporal_output_statistics(output.temporal_output)
        diagnostics["head_statistics"] = self.output_head.output_statistics(output.head_output)

        return diagnostics

    def extra_repr(self) -> str:
        disabled = self.ablation_summary().get("disabled_components", [])
        runtime_intervention = str(
            getattr(self, "runtime_intervention_variant", "full")
        )

        return (
            f"input_dim={self.input_dim}, "
            f"temporal_block={self.temporal_block_name}, "
            f"use_residual_evolution={self.model_config.use_residual_evolution}, "
            f"use_weak_accumulation={self.model_config.use_weak_accumulation}, "
            f"use_kirchhoff_exchange={self.model_config.use_kirchhoff_exchange}, "
            f"use_third_order={self.model_config.use_third_order}, "
            f"runtime_intervention_variant={runtime_intervention}, "
            f"disabled_components={disabled}"
        )


def create_proposed_model(config: Optional[Mapping[str, Any]] = None) -> ProposedSpoofingModel:
    """Create proposed spoofing model."""
    return ProposedSpoofingModel(config=config)


__all__ = [
    "ProposedModelConfig",
    "ProposedModelOutput",
    "ProposedSpoofingModel",
    "build_proposed_model_config",
    "create_proposed_model",
]