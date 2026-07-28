"""
Model factory for Step 11 proposed model and ablations.

Purpose:
- build the full proposed model,
- build official ablation variants,
- build professor high-order comparison variants,
- run Step-11 model sanity checks,
- save model architecture summaries for inspection.

Important:
This factory does not train models.
It only instantiates model architectures.

Ablations are trained later from scratch using the same Step-9 xi inputs and
same Step-10 metric/threshold framework.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from src.models.proposed_model import ProposedSpoofingModel, create_proposed_model
from src.models.evidence_encoder import DEFAULT_XI_FEATURE_COLUMNS
from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import ensure_dir, save_json


OFFICIAL_ABLATION_NAMES = [
    "no_residual_evolution",
    "no_weak_accumulation",
    "no_kirchhoff_exchange",
    "no_third_order",
    "no_liquid_dynamics",
]

# Step-16 official controlled ablation order.
# This order is locked for paper tables and reproducibility.
STEP16_LOCKED_ABLATION_VARIANTS = [
    "full",
    "no_residual_evolution",
    "no_weak_accumulation",
    "no_kirchhoff_exchange",
    "no_third_order",
    "no_liquid_dynamics",
]


# Hard full-model flags.
# This protects Step 16 from accidentally training a "full" model after a user
# edited model.proposed flags during earlier experiments.
DEFAULT_FULL_MODEL_FLAGS: Dict[str, Any] = {
    "model_type": "proposed",
    "use_residual_evolution": True,
    "use_weak_accumulation": True,
    "use_kirchhoff_exchange": True,
    "use_third_order": True,
    "temporal_block": "liquid_second_order",
}


ABLATION_COMPONENT_REMOVAL_DESCRIPTIONS: Dict[str, str] = {
    "full": "No component removed; complete proposed architecture.",
    "no_residual_evolution": "Residual-evolution branch disabled.",
    "no_weak_accumulation": "Weak-accumulation / persistence branch disabled.",
    "no_kirchhoff_exchange": "Kirchhoff exchange disabled.",
    "no_third_order": "Third-order interaction disabled.",
    "no_liquid_dynamics": "Liquid second-order temporal dynamics removed; fused state passed through with zero velocity.",
}

OFFICIAL_HIGH_ORDER_COMPARISON_NAMES = [
    "H0_no_feature_no_model_high_order",
    "H1_feature_high_order_only",
    "H2_model_high_order_only",
    "H3_full_feature_model_high_order",
]

# Step-17B Kirchhoff / model high-order structure comparison.
#
# These variants keep the full xi feature set fixed and change only
# model-side high-order structure.
#
# K3 is architecturally identical to the official Proposed model.
# In the Step-17B runner, K3 should reuse results/models/proposed_best.pt
# instead of being retrained, so K3 matches the main Proposed result.
KIRCHHOFF_STRUCTURE_COMPARISON_NAMES = [
    "K0_full_features_simple_model",
    "K1_full_features_kirchhoff_only",
    "K2_full_features_kirchhoff_third_order",
    "K3_official_proposed",
]
DEFAULT_ABLATION_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "no_residual_evolution": {
        "use_residual_evolution": False,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": True,
        "use_third_order": True,
        "temporal_block": "liquid_second_order",
    },
    "no_weak_accumulation": {
        "use_residual_evolution": True,
        "use_weak_accumulation": False,
        "use_kirchhoff_exchange": True,
        "use_third_order": True,
        "temporal_block": "liquid_second_order",
    },
    "no_kirchhoff_exchange": {
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": False,
        "use_third_order": True,
        "temporal_block": "liquid_second_order",
    },
    "no_third_order": {
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": True,
        "use_third_order": False,
        "temporal_block": "liquid_second_order",
    },
    "no_liquid_dynamics": {
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": True,
        "use_third_order": True,
        "temporal_block": "identity",
    },
}

DEFAULT_HIGH_ORDER_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # H0:
    # Low-order features only + low-order/non-liquid model.
    # Feature high-order is removed by the input-mask wrapper below.
    "H0_no_feature_no_model_high_order": {
        "use_feature_high_order": False,
        "feature_mask_mode": "low_order_only",
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": False,
        "use_third_order": False,
        "temporal_block": "gru",
        "kirchhoff_high_order": {
            "use_exchange_residual_bottleneck": False,
            "use_third_order_bottleneck": False,
        },
    },

    # H1:
    # High-order engineered xi features are available,
    # but model-side high-order dynamics are removed.
    "H1_feature_high_order_only": {
        "use_feature_high_order": True,
        "feature_mask_mode": "all",
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": False,
        "use_third_order": False,
        "temporal_block": "gru",
        "kirchhoff_high_order": {
            "use_exchange_residual_bottleneck": False,
            "use_third_order_bottleneck": False,
        },
    },

    # H2:
    # Low-order features only, but full model-side high-order mechanism enabled.
    "H2_model_high_order_only": {
        "use_feature_high_order": False,
        "feature_mask_mode": "low_order_only",
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": True,
        "use_third_order": True,
        "temporal_block": "liquid_second_order",
        "kirchhoff_high_order": {
            "use_exchange_residual_bottleneck": True,
            "use_third_order_bottleneck": True,
        },
    },

    # H3:
    # Full proposed setting: high-order features + model high-order.
    "H3_full_feature_model_high_order": {
        "use_feature_high_order": True,
        "feature_mask_mode": "all",
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": True,
        "use_third_order": True,
        "temporal_block": "liquid_second_order",
        "kirchhoff_high_order": {
            "use_exchange_residual_bottleneck": True,
            "use_third_order_bottleneck": True,
        },
    },

    # ------------------------------------------------------------------
    # Step-17B: Kirchhoff / model high-order structure comparison.
    #
    # All K variants use the full 9 xi feature set.
    # Only the model-side high-order structure changes.
    # ------------------------------------------------------------------

    # K0:
    # Full evidence features + simple temporal model.
    # No Kirchhoff exchange, no third-order fusion, no liquid dynamics.
    "K0_full_features_simple_model": {
        "use_feature_high_order": True,
        "feature_mask_mode": "all",
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": False,
        "use_third_order": False,
        "temporal_block": "gru",
        "kirchhoff_high_order": {
            "use_exchange_residual_bottleneck": False,
            "use_third_order_bottleneck": False,
        },
    },

    # K1:
    # Full evidence features + Kirchhoff exchange only.
    # This tests evidence coupling without explicit third-order fusion
    # and without liquid dynamics.
    "K1_full_features_kirchhoff_only": {
        "use_feature_high_order": True,
        "feature_mask_mode": "all",
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": True,
        "use_third_order": False,
        "temporal_block": "gru",
        "kirchhoff_high_order": {
            "use_exchange_residual_bottleneck": False,
            "use_third_order_bottleneck": False,
        },
    },

    # K2:
    # Full evidence features + Kirchhoff exchange + third-order fusion.
    # No liquid dynamics. This tests model-side high-order interaction
    # before the liquid temporal block.
    "K2_full_features_kirchhoff_third_order": {
        "use_feature_high_order": True,
        "feature_mask_mode": "all",
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": True,
        "use_third_order": True,
        "temporal_block": "gru",
        "kirchhoff_high_order": {
            "use_exchange_residual_bottleneck": True,
            "use_third_order_bottleneck": True,
        },
    },

    # K3:
    # Official Proposed architecture.
    # In Step-17B this should reuse results/models/proposed_best.pt,
    # not train a new checkpoint.
    "K3_official_proposed": {
        "use_feature_high_order": True,
        "feature_mask_mode": "all",
        "use_residual_evolution": True,
        "use_weak_accumulation": True,
        "use_kirchhoff_exchange": True,
        "use_third_order": True,
        "temporal_block": "liquid_second_order",
        "kirchhoff_high_order": {
            "use_exchange_residual_bottleneck": True,
            "use_third_order_bottleneck": True,
        },
    },
}

# Step-17 feature masking.
#
# Official 9 xi columns:
#   0 xi_eta_east_scaled          residual position / instantaneous evidence
#   1 xi_eta_north_scaled         residual position / instantaneous evidence
#   2 xi_eta_dot_east_scaled      residual velocity / evolution evidence
#   3 xi_eta_dot_north_scaled     residual velocity / evolution evidence
#   4 xi_eta_ddot_east_scaled     residual acceleration / second-order evolution evidence
#   5 xi_eta_ddot_north_scaled    residual acceleration / second-order evolution evidence
#   6 xi_q_scaled                 Mahalanobis residual energy
#   7 xi_accum_log_scaled         weak evidence accumulation
#   8 xi_nu                       validity indicator, always preserved unless explicitly requested
#
# Step 17A uses group-level feature interventions through the complete trained model:
#   no_eta
#   no_eta_dot
#   no_eta_ddot
#   no_q
#   no_accum_log
#
# Step 17 old H0/H2 uses:
#   low_order_only
LOW_ORDER_FEATURE_INDICES = [0, 1, 8]
FEATURE_HIGH_ORDER_INDICES = [2, 3, 4, 5, 6, 7]

FEATURE_GROUP_INDICES: Dict[str, List[int]] = {
    "eta": [0, 1],
    "eta_dot": [2, 3],
    "eta_ddot": [4, 5],
    "q": [6],
    "accum_log": [7],
    "nu": [8],
    "feature_high_order": [2, 3, 4, 5, 6, 7],
    "low_order": [0, 1, 8],
}

FEATURE_MASK_MODE_ALIASES: Dict[str, List[str]] = {
    "all": [
        "all",
        "none",
        "full",
        "full_features",
        "use_all_features",
        "no_mask",
    ],
    "feature_high_order": [
        "low_order_only",
        "no_feature_high_order",
        "disable_feature_high_order",
        "mask_feature_high_order",
        "without_feature_high_order",
    ],
    "eta": [
        "no_eta",
        "mask_eta",
        "without_eta",
        "disable_eta",
        "no_residual_position",
        "mask_residual_position",
    ],
    "eta_dot": [
        "no_eta_dot",
        "mask_eta_dot",
        "without_eta_dot",
        "disable_eta_dot",
        "no_residual_velocity",
        "mask_residual_velocity",
    ],
    "eta_ddot": [
        "no_eta_ddot",
        "mask_eta_ddot",
        "without_eta_ddot",
        "disable_eta_ddot",
        "no_residual_acceleration",
        "mask_residual_acceleration",
    ],
    "q": [
        "no_q",
        "mask_q",
        "without_q",
        "disable_q",
        "no_residual_energy",
        "mask_residual_energy",
        "no_mahalanobis_energy",
    ],
    "accum_log": [
        "no_accum_log",
        "mask_accum_log",
        "without_accum_log",
        "disable_accum_log",
        "no_accumulation",
        "mask_accumulation",
        "no_weak_evidence_accumulation",
    ],
}


def _canonical_feature_mask_mode(feature_mask_mode: str) -> str:
    """Map feature mask aliases to a canonical feature group name."""
    mode = str(feature_mask_mode).lower().strip()

    for canonical, aliases in FEATURE_MASK_MODE_ALIASES.items():
        if mode in aliases:
            return canonical

    if mode in FEATURE_GROUP_INDICES:
        return mode

    return mode


def resolve_disabled_feature_indices(
    feature_mask_mode: str,
    input_dim: int,
) -> Tuple[List[int], List[str]]:
    """
    Resolve feature-mask mode into disabled feature indices and group names.

    Supports:
    - all / none
    - low_order_only
    - no_eta
    - no_eta_dot
    - no_eta_ddot
    - no_q
    - no_accum_log

    Also supports comma-separated combinations such as:
        no_eta_dot,no_q
    """
    raw_mode = str(feature_mask_mode).lower().strip()

    if raw_mode in {"", "all", "none", "full", "full_features", "use_all_features", "no_mask"}:
        return [], []

    # Allow comma-separated combinations for future sensitivity checks.
    raw_parts = [
        part.strip()
        for part in raw_mode.replace("+", ",").replace("|", ",").split(",")
        if part.strip()
    ]

    disabled_indices: List[int] = []
    disabled_groups: List[str] = []

    for part in raw_parts:
        canonical = _canonical_feature_mask_mode(part)

        if canonical == "all":
            continue

        if canonical == "feature_high_order":
            indices = FEATURE_HIGH_ORDER_INDICES
        elif canonical in FEATURE_GROUP_INDICES:
            indices = FEATURE_GROUP_INDICES[canonical]
        else:
            raise ValueError(
                f"Unknown feature_mask_mode='{feature_mask_mode}'. "
                "Allowed modes include: all, low_order_only, no_eta, no_eta_dot, "
                "no_eta_ddot, no_q, no_accum_log."
            )

        disabled_groups.append(canonical)

        for index in indices:
            if 0 <= int(index) < int(input_dim) and int(index) not in disabled_indices:
                disabled_indices.append(int(index))

    disabled_indices = sorted(disabled_indices)

    return disabled_indices, disabled_groups


class FeatureHighOrderInputMaskWrapper(nn.Module):
    """
    Wrapper that removes selected xi feature groups while keeping the same
    input dimensionality.

    This is used for:
    - old Step 17 H0/H2 feature high-order masking,
    - new Step 17A feature-group interventions.

    The wrapper does not change the model architecture.
    It only zeros selected input columns at runtime.
    """

    def __init__(
        self,
        base_model: ProposedSpoofingModel,
        feature_mask_mode: str = "low_order_only",
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.feature_mask_mode = str(feature_mask_mode)

        input_dim = int(base_model.input_dim)
        mask = torch.ones(input_dim, dtype=torch.float32)

        disabled_indices, disabled_groups = resolve_disabled_feature_indices(
            feature_mask_mode=self.feature_mask_mode,
            input_dim=input_dim,
        )

        for index in disabled_indices:
            mask[index] = 0.0

        self.disabled_feature_indices = list(disabled_indices)
        self.disabled_feature_groups = list(disabled_groups)
        self.disabled_feature_names = [
            self.feature_columns[index]
            for index in disabled_indices
            if index < len(self.feature_columns)
        ]

        # Non-persistent buffer:
        # The mask is determined by config/runtime intervention, not learned.
        # It should not create checkpoint compatibility problems.
        self.register_buffer(
            "feature_input_mask",
            mask.view(1, 1, input_dim),
            persistent=False,
        )

    @property
    def input_dim(self) -> int:
        return int(self.base_model.input_dim)

    @property
    def feature_columns(self) -> Tuple[str, ...]:
        return tuple(self.base_model.feature_columns)

    @property
    def model_config(self):
        return self.base_model.model_config

    @property
    def temporal_block_name(self) -> str:
        return str(self.base_model.temporal_block_name)

    def _mask_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            return x

        if not self.disabled_feature_indices:
            return x

        mask = self.feature_input_mask.to(device=x.device, dtype=x.dtype)

        if x.ndim == 2:
            return x * mask.view(1, -1)

        if x.ndim == 3:
            return x * mask

        return x

    def forward(self, x, *args, **kwargs):
        if isinstance(x, Mapping):
            batch = dict(x)
            batch["x"] = self._mask_tensor(batch["x"])
            return self.base_model(batch, *args, **kwargs)

        return self.base_model(self._mask_tensor(x), *args, **kwargs)

    @torch.no_grad()
    def predict_proba(self, batch_or_x, **kwargs):
        self.eval()
        output = self.forward(batch_or_x, **kwargs)
        return output.probabilities

    def set_runtime_intervention(self, variant_name: str) -> None:
        if hasattr(self.base_model, "set_runtime_intervention"):
            self.base_model.set_runtime_intervention(variant_name)

    def count_parameters(self) -> Dict[str, int]:
        return self.base_model.count_parameters()

    def module_summary(self) -> Dict[str, Any]:
        summary = self.base_model.module_summary()

        summary["feature_input_mask"] = {
            "feature_mask_mode": self.feature_mask_mode,
            "feature_groups_available": dict(FEATURE_GROUP_INDICES),
            "feature_groups_disabled": list(self.disabled_feature_groups),
            "feature_indices_disabled": list(self.disabled_feature_indices),
            "feature_names_disabled": list(self.disabled_feature_names),
            "feature_high_order_enabled": len(
                set(FEATURE_HIGH_ORDER_INDICES).intersection(
                    set(self.disabled_feature_indices)
                )
            ) == 0,
            "same_input_dim_preserved": True,
            "xi_nu_preserved": 8 not in self.disabled_feature_indices,
            "purpose": "Step 17A feature-group intervention and Step 17 feature high-order comparison.",
        }

        # Backward-compatible key used by previous Step 17 diagnostics.
        summary["feature_high_order_input_mask"] = summary["feature_input_mask"]

        return summary

    @torch.no_grad()
    def forward_diagnostics(self, output) -> Dict[str, Any]:
        diagnostics = self.base_model.forward_diagnostics(output)

        diagnostics["feature_input_mask"] = {
            "feature_mask_mode": self.feature_mask_mode,
            "feature_groups_disabled": list(self.disabled_feature_groups),
            "feature_indices_disabled": list(self.disabled_feature_indices),
            "feature_names_disabled": list(self.disabled_feature_names),
        }

        # Backward-compatible key used by previous Step 17 diagnostics.
        diagnostics["feature_high_order_input_mask"] = diagnostics["feature_input_mask"]

        return diagnostics

    def extra_repr(self) -> str:
        return (
            f"feature_mask_mode={self.feature_mask_mode}, "
            f"disabled_feature_groups={self.disabled_feature_groups}, "
            f"disabled_feature_indices={self.disabled_feature_indices}, "
            f"disabled_feature_names={self.disabled_feature_names}"
        )


@dataclass
class ModelBuildInfo:
    """Metadata for one constructed model."""

    model_name: str
    variant_name: str
    variant_group: str
    model_type: str

    use_residual_evolution: bool
    use_weak_accumulation: bool
    use_kirchhoff_exchange: bool
    use_third_order: bool
    temporal_block: str

    input_dim: int
    feature_columns: List[str]

    total_parameters: int
    trainable_parameters: int

    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step11ModelFactoryReport:
    """Step-11 architecture sanity-check report."""

    full_model: Dict[str, Any]
    ablation_models: Dict[str, Dict[str, Any]]
    high_order_comparison_models: Dict[str, Dict[str, Any]]
    forward_sanity_checks: Dict[str, Dict[str, Any]]
    fairness_rules: Dict[str, Any]
    saved_outputs: Dict[str, str]
    final_step11_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively update dictionary."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)

    return base


def _get_proposed_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return mutable model.proposed dict."""
    if "model" not in config or not isinstance(config["model"], dict):
        config["model"] = {}

    if "proposed" not in config["model"] or not isinstance(config["model"]["proposed"], dict):
        config["model"]["proposed"] = {}

    return config["model"]["proposed"]


def _normalize_variant_name(variant_name: Optional[str]) -> str:
    """Normalize variant name."""
    if variant_name is None or str(variant_name).strip() == "":
        return "full"
    return str(variant_name).strip()


def get_step16_locked_ablation_variants() -> List[str]:
    """Return the locked official Step-16 ablation order."""
    return list(STEP16_LOCKED_ABLATION_VARIANTS)


def describe_ablation_variant(variant_name: str) -> str:
    """Return human-readable description of the component removed."""
    variant_name = _normalize_variant_name(variant_name)
    return ABLATION_COMPONENT_REMOVAL_DESCRIPTIONS.get(
        variant_name,
        "Custom or unknown variant.",
    )



def get_step11_summary_path(config: Mapping[str, Any]) -> Path:
    """Resolve Step-11 model factory summary path."""
    value = get_by_path(
        config,
        "paths.step11_model_factory_summary_json",
        "results/tables/step11_model_factory_summary.json",
    )
    return resolve_project_path(config, value)


def get_step11_architecture_dir(config: Mapping[str, Any]) -> Path:
    """Resolve Step-11 architecture summary directory."""
    value = get_by_path(
        config,
        "paths.step11_architecture_dir",
        "results/tables/model_architecture",
    )
    path = resolve_project_path(config, value)
    ensure_dir(path)
    return path


def get_step11_variant_summary_path(config: Mapping[str, Any], variant_name: str) -> Path:
    """Resolve per-variant architecture summary path."""
    safe_name = str(variant_name).replace("/", "_").replace(" ", "_")
    return get_step11_architecture_dir(config) / f"{safe_name}_architecture_summary.json"


def get_available_model_variants(config: Optional[Mapping[str, Any]] = None) -> Dict[str, List[str]]:
    """Return available model variants."""
    ablation_names = list(OFFICIAL_ABLATION_NAMES)

    # Keep old H0/H1/H2/H3 variants and add new K0/K1/K2/K3 variants.
    high_order_names = list(OFFICIAL_HIGH_ORDER_COMPARISON_NAMES)
    for name in KIRCHHOFF_STRUCTURE_COMPARISON_NAMES:
        if name not in high_order_names:
            high_order_names.append(name)

    kirchhoff_structure_names = list(KIRCHHOFF_STRUCTURE_COMPARISON_NAMES)

    if config is not None:
        configured_ablations = get_by_path(config, "model.ablations", {})
        if isinstance(configured_ablations, Mapping):
            for name in configured_ablations.keys():
                name = str(name)
                if name not in ablation_names:
                    ablation_names.append(name)

        # New preferred path for old/current H variants.
        configured_high_order = get_by_path(config, "model.high_order_comparison", {})
        if isinstance(configured_high_order, Mapping):
            for name in configured_high_order.keys():
                name = str(name)
                if name not in high_order_names:
                    high_order_names.append(name)

        # New preferred path for Step-17B K variants.
        configured_kirchhoff_structure = get_by_path(
            config,
            "model.kirchhoff_structure_comparison",
            {},
        )
        if isinstance(configured_kirchhoff_structure, Mapping):
            for name in configured_kirchhoff_structure.keys():
                name = str(name)
                if name not in high_order_names:
                    high_order_names.append(name)
                if name not in kirchhoff_structure_names:
                    kirchhoff_structure_names.append(name)

        # Backward-compatible old path.
        configured_professor_high_order = get_by_path(
            config,
            "model.professor_high_order_comparison",
            {},
        )
        if isinstance(configured_professor_high_order, Mapping):
            for name in configured_professor_high_order.keys():
                name = str(name)
                if name not in high_order_names:
                    high_order_names.append(name)

    return {
        "full": ["full"],
        "official_ablations": ablation_names,
        "high_order_comparison": high_order_names,
        "professor_high_order_comparison": high_order_names,
        "kirchhoff_structure_comparison": kirchhoff_structure_names,
    }

def _get_configured_ablation_override(
    config: Mapping[str, Any],
    variant_name: str,
) -> Dict[str, Any]:
    """Get ablation override from config or defaults."""
    configured = get_by_path(config, f"model.ablations.{variant_name}", None)

    if isinstance(configured, Mapping):
        return dict(configured)

    if variant_name in DEFAULT_ABLATION_OVERRIDES:
        return dict(DEFAULT_ABLATION_OVERRIDES[variant_name])

    raise KeyError(f"Unknown ablation variant: {variant_name}")


def _get_configured_high_order_override(
    config: Mapping[str, Any],
    variant_name: str,
) -> Dict[str, Any]:
    """Get high-order / Kirchhoff-structure override from config or defaults."""

    # Old/current Step-17 H variants.
    configured = get_by_path(
        config,
        f"model.high_order_comparison.{variant_name}",
        None,
    )

    if isinstance(configured, Mapping):
        return dict(configured)

    # New Step-17B K variants.
    configured_kirchhoff_structure = get_by_path(
        config,
        f"model.kirchhoff_structure_comparison.{variant_name}",
        None,
    )

    if isinstance(configured_kirchhoff_structure, Mapping):
        return dict(configured_kirchhoff_structure)

    # Backward-compatible old path.
    configured_legacy = get_by_path(
        config,
        f"model.professor_high_order_comparison.{variant_name}",
        None,
    )

    if isinstance(configured_legacy, Mapping):
        return dict(configured_legacy)

    if variant_name in DEFAULT_HIGH_ORDER_OVERRIDES:
        return dict(DEFAULT_HIGH_ORDER_OVERRIDES[variant_name])

    raise KeyError(f"Unknown high-order / Kirchhoff-structure variant: {variant_name}")

def infer_variant_group(config: Mapping[str, Any], variant_name: str) -> str:
    """Infer variant group."""
    variant_name = _normalize_variant_name(variant_name)

    if variant_name == "full":
        return "full"

    configured_ablations = get_by_path(config, "model.ablations", {})
    if (
        variant_name in DEFAULT_ABLATION_OVERRIDES
        or (isinstance(configured_ablations, Mapping) and variant_name in configured_ablations)
    ):
        return "official_ablation"

    configured_high_order = get_by_path(config, "model.high_order_comparison", {})
    configured_kirchhoff_structure = get_by_path(
        config,
        "model.kirchhoff_structure_comparison",
        {},
    )
    configured_professor_high_order = get_by_path(
        config,
        "model.professor_high_order_comparison",
        {},
    )

    if (
        variant_name in DEFAULT_HIGH_ORDER_OVERRIDES
        or variant_name in KIRCHHOFF_STRUCTURE_COMPARISON_NAMES
        or (isinstance(configured_high_order, Mapping) and variant_name in configured_high_order)
        or (
            isinstance(configured_kirchhoff_structure, Mapping)
            and variant_name in configured_kirchhoff_structure
        )
        or (
            isinstance(configured_professor_high_order, Mapping)
            and variant_name in configured_professor_high_order
        )
    ):
        return "high_order_comparison"

    return "custom"

def validate_step16_ablation_contract(
    config: Mapping[str, Any],
    variants: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Validate the locked Step-16 ablation contract.

    This does not train anything. It checks that each official Step-16 variant
    changes only the intended config flags.
    """
    if variants is None:
        variants = STEP16_LOCKED_ABLATION_VARIANTS

    expected_flags: Dict[str, Dict[str, Any]] = {
        "full": dict(DEFAULT_FULL_MODEL_FLAGS),
        **{name: dict(DEFAULT_ABLATION_OVERRIDES[name]) for name in OFFICIAL_ABLATION_NAMES},
    }

    checked: Dict[str, Any] = {}
    passed = True

    for name in variants:
        name = _normalize_variant_name(name)

        if name not in expected_flags:
            passed = False
            checked[name] = {
                "status": "FAILED",
                "reason": "Variant is not part of the locked Step-16 ablation set.",
            }
            continue

        expected = expected_flags[name]

        if name == "full":
            actual = dict(DEFAULT_FULL_MODEL_FLAGS)
        else:
            actual = _get_configured_ablation_override(config, name)

        mismatches: Dict[str, Dict[str, Any]] = {}

        for key, expected_value in expected.items():
            actual_value = actual.get(key, None)
            if actual_value != expected_value:
                mismatches[key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                }

        status = "PASSED" if not mismatches else "FAILED"
        if mismatches:
            passed = False

        checked[name] = {
            "status": status,
            "description": describe_ablation_variant(name),
            "expected_flags": expected,
            "actual_flags": actual,
            "mismatches": mismatches,
        }

    return {
        "passed": bool(passed),
        "locked_variants": list(STEP16_LOCKED_ABLATION_VARIANTS),
        "checked_variants": checked,
        "rules": {
            "full_model_forced_to_all_components_enabled": True,
            "ablations_are_config_controlled": True,
            "only_locked_step16_variants_allowed": True,
            "threshold_not_inside_model": True,
            "alarm_rule_not_inside_model": True,
        },
    }

def apply_model_variant_overrides(
    config: Mapping[str, Any],
    variant_name: Optional[str] = "full",
    explicit_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Return a deep-copied config with variant overrides applied.

    Overrides target model.proposed flags:
    - use_residual_evolution
    - use_weak_accumulation
    - use_kirchhoff_exchange
    - use_third_order
    - temporal_block
    """
    variant_name = _normalize_variant_name(variant_name)
    new_config = copy.deepcopy(dict(config))

    proposed = _get_proposed_dict(new_config)

    if variant_name == "full":
        # For official Step-16 ablations, "full" must always mean all proposed
        # components are enabled. This protects against accidental YAML edits.
        override: Dict[str, Any] = dict(DEFAULT_FULL_MODEL_FLAGS)
    else:
        group = infer_variant_group(config, variant_name)

        if group == "official_ablation":
            override = _get_configured_ablation_override(config, variant_name)
        elif group == "high_order_comparison":
            override = _get_configured_high_order_override(config, variant_name)
        elif explicit_overrides is not None:
            override = dict(explicit_overrides)
        else:
            raise KeyError(
                f"Unknown model variant '{variant_name}'. "
                "Provide explicit_overrides or add it to model.ablations/model.professor_high_order_comparison."
            )

    if explicit_overrides is not None:
        override = _deep_update(dict(override), explicit_overrides)

    _deep_update(proposed, override)

    # Ensure output head is stable across full/ablations unless explicitly changed.
    if "output_head" not in proposed or not isinstance(proposed["output_head"], dict):
        proposed["output_head"] = {}

    proposed["output_head"].setdefault("input", "hidden_and_velocity")
    proposed["output_head"].setdefault("activation", "sigmoid")

    return new_config


def validate_model_feature_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the locked 9-column xi input contract."""
    feature_columns = list(
        get_by_path(
            config,
            "model.input.recommended_model_input_columns",
            DEFAULT_XI_FEATURE_COLUMNS,
        )
    )

    input_dim = int(get_by_path(config, "model.input.input_dim", 9))

    expected = list(DEFAULT_XI_FEATURE_COLUMNS)
    passed = feature_columns == expected and input_dim == 9

    return {
        "passed": bool(passed),
        "input_dim": input_dim,
        "feature_columns": feature_columns,
        "expected_feature_columns": expected,
        "uses_only_scaled_xi": feature_columns == expected,
        "raw_shortcut_columns_used": False,
    }


def create_model_build_info(
    model: ProposedSpoofingModel,
    variant_name: str,
    variant_group: str,
) -> ModelBuildInfo:
    """Create ModelBuildInfo from constructed model."""
    counts = model.count_parameters()
    cfg = model.model_config

    # Optional metadata for Step-16 reports.
    # Kept outside the dataclass for backward compatibility unless printed through
    # model.module_summary().

    return ModelBuildInfo(
        model_name=cfg.model_name,
        variant_name=variant_name,
        variant_group=variant_group,
        model_type=cfg.model_type,
        use_residual_evolution=bool(cfg.use_residual_evolution),
        use_weak_accumulation=bool(cfg.use_weak_accumulation),
        use_kirchhoff_exchange=bool(cfg.use_kirchhoff_exchange),
        use_third_order=bool(cfg.use_third_order),
        temporal_block=str(cfg.temporal_block),
        input_dim=int(cfg.input_dim),
        feature_columns=list(cfg.feature_columns),
        total_parameters=int(counts["total_parameters"]),
        trainable_parameters=int(counts["trainable_parameters"]),
        status="PASSED",
    )

def maybe_apply_feature_high_order_input_mask(
    model: ProposedSpoofingModel,
    variant_config: Mapping[str, Any],
) -> nn.Module:
    """
    Apply Step-17 feature input masking when requested.

    This keeps the same 9-dimensional input contract but zeros selected
    feature columns for feature-group interventions.

    Supported modes:
    - all / none
    - low_order_only
    - no_eta
    - no_eta_dot
    - no_eta_ddot
    - no_q
    - no_accum_log
    """
    use_feature_high_order = bool(
        get_by_path(
            variant_config,
            "model.proposed.use_feature_high_order",
            True,
        )
    )

    feature_mask_mode = str(
        get_by_path(
            variant_config,
            "model.proposed.feature_mask_mode",
            "all",
        )
    ).lower().strip()

    # Backward compatibility:
    # If a variant disables feature high-order but forgot to specify a mask,
    # use low_order_only.
    if not use_feature_high_order and feature_mask_mode in {
        "all",
        "none",
        "full",
        "full_features",
        "use_all_features",
        "no_mask",
        "",
    }:
        feature_mask_mode = "low_order_only"

    # No mask needed.
    if use_feature_high_order and feature_mask_mode in {
        "all",
        "none",
        "full",
        "full_features",
        "use_all_features",
        "no_mask",
        "",
    }:
        return model

    return FeatureHighOrderInputMaskWrapper(
        base_model=model,
        feature_mask_mode=feature_mask_mode,
    )
def build_model(
    config: Mapping[str, Any],
    variant_name: Optional[str] = "full",
    explicit_overrides: Optional[Mapping[str, Any]] = None,
    device: Optional[torch.device | str] = None,
) -> Tuple[ProposedSpoofingModel, ModelBuildInfo, Dict[str, Any]]:
    """
    Build one model variant.

    Returns:
        model, build_info, variant_config
    """
    variant_name = _normalize_variant_name(variant_name)
    variant_group = infer_variant_group(config, variant_name)

    variant_config = apply_model_variant_overrides(
        config=config,
        variant_name=variant_name,
        explicit_overrides=explicit_overrides,
    )

    feature_contract = validate_model_feature_contract(variant_config)
    if not feature_contract["passed"]:
        raise ValueError(f"Model feature contract failed: {feature_contract}")

    model = create_proposed_model(variant_config)

    model = maybe_apply_feature_high_order_input_mask(
        model=model,
        variant_config=variant_config,
    )

    if device is not None:
        model = model.to(device)

    build_info = create_model_build_info(
        model=model,
        variant_name=variant_name,
        variant_group=variant_group,
    )

    return model, build_info, variant_config


def build_full_model(
    config: Mapping[str, Any],
    device: Optional[torch.device | str] = None,
) -> Tuple[ProposedSpoofingModel, ModelBuildInfo, Dict[str, Any]]:
    """Build the full proposed model."""
    return build_model(config=config, variant_name="full", device=device)


def build_official_ablation_models(
    config: Mapping[str, Any],
    device: Optional[torch.device | str] = None,
) -> Dict[str, Tuple[ProposedSpoofingModel, ModelBuildInfo, Dict[str, Any]]]:
    """Build all official ablation models."""
    available = get_available_model_variants(config)
    names = available["official_ablations"]

    return {
        name: build_model(config=config, variant_name=name, device=device)
        for name in names
    }


def build_high_order_comparison_models(
    config: Mapping[str, Any],
    device: Optional[torch.device | str] = None,
) -> Dict[str, Tuple[ProposedSpoofingModel, ModelBuildInfo, Dict[str, Any]]]:
    """Build professor high-order comparison models."""
    available = get_available_model_variants(config)
    names = available["high_order_comparison"]

    return {
        name: build_model(config=config, variant_name=name, device=device)
        for name in names
    }


def _make_synthetic_step11_batch(
    batch_size: int,
    time_steps: int,
    input_dim: int,
    device: torch.device | str,
) -> Dict[str, torch.Tensor]:
    """Create synthetic Step-9-like batch for architecture sanity check."""
    x = torch.randn(batch_size, time_steps, input_dim, dtype=torch.float32, device=device)

    # xi_nu column must be a validity-like feature.
    if input_dim >= 9:
        x[..., 8] = 1.0

    padding_mask = torch.ones(batch_size, time_steps, dtype=torch.float32, device=device)

    if time_steps >= 4:
        padding_mask[0, -1] = 0.0

    loss_mask = padding_mask.clone()
    delta_t = torch.ones(batch_size, time_steps, dtype=torch.float32, device=device)
    y = torch.zeros(batch_size, time_steps, dtype=torch.long, device=device)

    if time_steps >= 4:
        y[:, time_steps // 2 :] = 1

    reset_state = torch.ones(batch_size, dtype=torch.float32, device=device)

    return {
        "x": x,
        "y": y,
        "loss_mask": loss_mask,
        "padding_mask": padding_mask,
        "delta_t": delta_t,
        "reset_state": reset_state,
    }


@torch.no_grad()
def run_single_model_forward_sanity_check(
    model: ProposedSpoofingModel,
    device: torch.device | str = "cpu",
    batch_size: int = 2,
    time_steps: int = 16,
) -> Dict[str, Any]:
    """Run one forward sanity check."""
    model.eval()
    model.to(device)

    batch = _make_synthetic_step11_batch(
        batch_size=batch_size,
        time_steps=time_steps,
        input_dim=model.input_dim,
        device=device,
    )

    output = model(batch)

    probabilities = output.probabilities.detach()
    logits = output.logits.detach()

    expected_shape = [batch_size, time_steps]
    got_shape = list(probabilities.shape)

    finite_probs = bool(torch.isfinite(probabilities).all().item())
    finite_logits = bool(torch.isfinite(logits).all().item())
    probability_range_ok = bool(
        (probabilities.min().item() >= 0.0)
        and (probabilities.max().item() <= 1.0)
    )

    status = (
        "PASSED"
        if got_shape == expected_shape and finite_probs and finite_logits and probability_range_ok
        else "FAILED"
    )

    diagnostics = model.forward_diagnostics(output)

    return {
        "status": status,
        "expected_probability_shape": expected_shape,
        "actual_probability_shape": got_shape,
        "finite_probabilities": finite_probs,
        "finite_logits": finite_logits,
        "probability_min": float(probabilities.min().item()),
        "probability_max": float(probabilities.max().item()),
        "probability_mean": float(probabilities.mean().item()),
        "probability_range_ok": probability_range_ok,
        "output_shape_summary": output.shape_summary(),
        "diagnostics": diagnostics,
    }


def save_variant_architecture_summary(
    config: Mapping[str, Any],
    variant_name: str,
    model: ProposedSpoofingModel,
    build_info: ModelBuildInfo,
    forward_check: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save per-variant architecture summary."""
    path = get_step11_variant_summary_path(config, variant_name)

    payload = {
        "variant_name": variant_name,
        "build_info": build_info.to_dict(),
        "module_summary": model.module_summary(),
        "forward_sanity_check": dict(forward_check) if forward_check is not None else None,
    }

    save_json(payload, path, indent=2)
    return path


def run_model_factory_sanity_check(
    config: Mapping[str, Any],
    device: Optional[torch.device | str] = None,
    include_ablations: bool = True,
    include_high_order_comparison: bool = True,
    save_outputs: bool = True,
) -> Step11ModelFactoryReport:
    """
    Run Step-11 architecture sanity check.

    This builds:
    - full model,
    - official ablations,
    - professor high-order variants,

    then runs a synthetic forward pass for each model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    saved_outputs: Dict[str, str] = {}

    # Full model
    full_model, full_info, _full_config = build_full_model(config=config, device=device)
    full_forward = run_single_model_forward_sanity_check(
        model=full_model,
        device=device,
    )

    full_payload = {
        "build_info": full_info.to_dict(),
        "module_summary": full_model.module_summary(),
    }

    if save_outputs:
        full_path = save_variant_architecture_summary(
            config=config,
            variant_name="full",
            model=full_model,
            build_info=full_info,
            forward_check=full_forward,
        )
        saved_outputs["full_architecture_summary"] = str(full_path)

    forward_checks: Dict[str, Dict[str, Any]] = {
        "full": full_forward,
    }

    # Official ablations
    ablation_payloads: Dict[str, Dict[str, Any]] = {}

    if include_ablations:
        ablation_models = build_official_ablation_models(config=config, device=device)

        for name, (model, info, _variant_config) in ablation_models.items():
            check = run_single_model_forward_sanity_check(
                model=model,
                device=device,
            )
            forward_checks[name] = check

            ablation_payloads[name] = {
                "build_info": info.to_dict(),
                "module_summary": model.module_summary(),
            }

            if save_outputs:
                path = save_variant_architecture_summary(
                    config=config,
                    variant_name=name,
                    model=model,
                    build_info=info,
                    forward_check=check,
                )
                saved_outputs[f"{name}_architecture_summary"] = str(path)

    # High-order comparison
    high_order_payloads: Dict[str, Dict[str, Any]] = {}

    if include_high_order_comparison:
        high_order_models = build_high_order_comparison_models(config=config, device=device)

        for name, (model, info, _variant_config) in high_order_models.items():
            check = run_single_model_forward_sanity_check(
                model=model,
                device=device,
            )
            forward_checks[name] = check

            high_order_payloads[name] = {
                "build_info": info.to_dict(),
                "module_summary": model.module_summary(),
            }

            if save_outputs:
                path = save_variant_architecture_summary(
                    config=config,
                    variant_name=name,
                    model=model,
                    build_info=info,
                    forward_check=check,
                )
                saved_outputs[f"{name}_architecture_summary"] = str(path)

    all_checks_passed = all(
        item.get("status") == "PASSED"
        for item in forward_checks.values()
    )

    fairness_rules = {
        "all_variants_use_same_input_dim": True,
        "all_variants_use_same_9_scaled_xi_columns": True,
        "raw_shortcut_columns_used": False,
        "ablations_change_only_configured_modules": True,
        "threshold_not_inside_model": True,
        "alarm_rule_not_inside_model": True,
        "step10_selects_theta_np_later_from_validation_predictions": True,
        "synthetic_step10_theta_0_55_not_used_as_final_threshold": True,
        "ablation_models_built_for_training_from_scratch_later": True,
    }

    final_status = "PASSED" if all_checks_passed else "FAILED"

    report = Step11ModelFactoryReport(
        full_model=full_payload,
        ablation_models=ablation_payloads,
        high_order_comparison_models=high_order_payloads,
        forward_sanity_checks=forward_checks,
        fairness_rules=fairness_rules,
        saved_outputs=saved_outputs,
        final_step11_status=final_status,
    )

    if save_outputs:
        summary_path = get_step11_summary_path(config)
        save_json(report.to_dict(), summary_path, indent=2)
        saved_outputs["step11_model_factory_summary"] = str(summary_path)

        # Re-save with summary path included.
        report.saved_outputs = saved_outputs
        save_json(report.to_dict(), summary_path, indent=2)

    print_step11_model_factory_report(report)

    if final_status != "PASSED":
        raise RuntimeError(f"Step 11 model factory sanity check failed: {final_status}")

    return report


def print_model_build_info(info: ModelBuildInfo) -> None:
    """Print one model build summary."""
    print(
        f"{info.variant_name:32s} | "
        f"group={info.variant_group:22s} | "
        f"params={info.trainable_parameters:9d} | "
        f"res_evo={str(info.use_residual_evolution):5s} | "
        f"weak_acc={str(info.use_weak_accumulation):5s} | "
        f"kirchhoff={str(info.use_kirchhoff_exchange):5s} | "
        f"third={str(info.use_third_order):5s} | "
        f"temporal={info.temporal_block:20s} | "
        f"status={info.status}"
    )


def print_step11_model_factory_report(report: Step11ModelFactoryReport) -> None:
    """Print Step-11 model factory report."""
    print("=" * 120)
    print("STEP 11 MODEL FACTORY / PROPOSED MODEL SANITY REPORT")
    print("=" * 120)

    full_info = ModelBuildInfo(**report.full_model["build_info"])
    print_model_build_info(full_info)

    for payload in report.ablation_models.values():
        print_model_build_info(ModelBuildInfo(**payload["build_info"]))

    for payload in report.high_order_comparison_models.values():
        print_model_build_info(ModelBuildInfo(**payload["build_info"]))

    print("-" * 120)
    print("FORWARD SANITY CHECKS")
    print("-" * 120)

    for name, check in report.forward_sanity_checks.items():
        print(
            f"{name:32s} | "
            f"status={check.get('status'):8s} | "
            f"prob_shape={check.get('actual_probability_shape')} | "
            f"prob_min={check.get('probability_min'):.6f} | "
            f"prob_max={check.get('probability_max'):.6f} | "
            f"finite={check.get('finite_probabilities')}"
        )

    print("-" * 120)
    print(f"Fairness rules: {report.fairness_rules}")
    print(f"Saved outputs : {report.saved_outputs}")
    print(f"Final Step 11 status: {report.final_step11_status}")
    print("=" * 120)


def load_model_for_training(
    config: Mapping[str, Any],
    variant_name: str = "full",
    device: Optional[torch.device | str] = None,
) -> ProposedSpoofingModel:
    """
    Convenience helper for future Step 12 training.

    It returns only the model, not build metadata.
    """
    model, _info, _variant_config = build_model(
        config=config,
        variant_name=variant_name,
        device=device,
    )
    return model


__all__ = [
    "OFFICIAL_ABLATION_NAMES",
    "OFFICIAL_HIGH_ORDER_COMPARISON_NAMES",
    "ModelBuildInfo",
    "Step11ModelFactoryReport",
    "get_available_model_variants",
    "apply_model_variant_overrides",
    "validate_model_feature_contract",
    "build_model",
    "build_full_model",
    "build_official_ablation_models",
    "build_high_order_comparison_models",
    "run_single_model_forward_sanity_check",
    "run_model_factory_sanity_check",
    "print_step11_model_factory_report",
    "load_model_for_training",
    "STEP16_LOCKED_ABLATION_VARIANTS",
    "DEFAULT_FULL_MODEL_FLAGS",
    "ABLATION_COMPONENT_REMOVAL_DESCRIPTIONS",
    "get_step16_locked_ablation_variants",
    "describe_ablation_variant",
    "validate_step16_ablation_contract",
    "LOW_ORDER_FEATURE_INDICES",
    "FEATURE_HIGH_ORDER_INDICES",
    "KIRCHHOFF_STRUCTURE_COMPARISON_NAMES",
    "FEATURE_GROUP_INDICES",
    "FEATURE_MASK_MODE_ALIASES",
    "resolve_disabled_feature_indices",
    "FeatureHighOrderInputMaskWrapper",
]