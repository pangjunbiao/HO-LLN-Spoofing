"""
Locked hyperparameter candidates for the GPS-IDS classifier-suite branch.

The GPS-IDS paper states that combinations of hyperparameters were fine-tuned,
but does not publish one executable search configuration. These explicit,
finite candidate lists therefore define the protocol-controlled
reimplementation.

Selection rule
--------------
For each classifier:
1. fit every candidate on Dataset-1 training rows only;
2. rank candidates on Dataset-1 validation valid rows by:
   higher AUPRC, higher AUROC, lower log loss, then lower candidate order;
3. refit the selected candidate on Dataset-1 training rows only;
4. select theta/persistence from its validation probabilities using the
   authoritative unified operating-point selector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from external_baselines.gps_ids_reproduction.models import (
    MODEL_ORDER,
    validate_model_key,
)


SEARCH_PROFILES: Tuple[str, ...] = ("standard", "smoke")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    model_key: str
    imputer_strategy: str
    scaler_name: str
    model_parameters: Dict[str, Any]
    complexity_order: int

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["model_parameters"] = _json_safe_parameters(
            self.model_parameters
        )
        return payload


def _json_safe_parameters(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in parameters.items():
        output[str(key)] = list(value) if isinstance(value, tuple) else value
    return output


def _candidate(
    model_key: str,
    index: int,
    *,
    imputer: str,
    scaler: str,
    parameters: Mapping[str, Any],
) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=f"{model_key}_c{index:02d}",
        model_key=model_key,
        imputer_strategy=imputer,
        scaler_name=scaler,
        model_parameters=dict(parameters),
        complexity_order=int(index),
    )


def _standard_candidates(model_key: str) -> List[CandidateSpec]:
    if model_key == "random_forest":
        definitions = [
            ("median", "none", {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
            }),
            ("median", "none", {
                "n_estimators": 500,
                "max_depth": 20,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
            }),
            ("median", "none", {
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
                "bootstrap": True,
            }),
            ("most_frequent", "none", {
                "n_estimators": 500,
                "max_depth": 20,
                "min_samples_leaf": 3,
                "max_features": 0.75,
                "bootstrap": True,
            }),
        ]
    elif model_key == "xgboost":
        definitions = [
            ("median", "none", {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
            }),
            ("median", "none", {
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
            }),
            ("median", "none", {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 2.0,
                "reg_lambda": 2.0,
            }),
            ("most_frequent", "none", {
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 2.0,
                "reg_lambda": 2.0,
            }),
        ]
    elif model_key == "svc":
        definitions = [
            ("median", "standard", {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "tol": 1.0e-3,
                "max_iter": 20000,
                "shrinking": True,
                "cache_size": 2048.0,
            }),
            ("median", "standard", {
                "C": 10.0,
                "kernel": "rbf",
                "gamma": "scale",
                "tol": 1.0e-3,
                "max_iter": 20000,
                "shrinking": True,
                "cache_size": 2048.0,
            }),
            ("most_frequent", "robust", {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "tol": 1.0e-3,
                "max_iter": 20000,
                "shrinking": True,
                "cache_size": 2048.0,
            }),
        ]
    elif model_key == "mlp":
        common = {
            "activation": "relu",
            "solver": "adam",
            "batch_size": 256,
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.10,
            "n_iter_no_change": 20,
            "tol": 1.0e-4,
        }
        definitions = [
            ("median", "standard", {
                **common,
                "hidden_layer_sizes": (64, 32),
                "alpha": 1.0e-4,
                "learning_rate_init": 1.0e-3,
            }),
            ("median", "standard", {
                **common,
                "hidden_layer_sizes": (128, 64),
                "alpha": 1.0e-4,
                "learning_rate_init": 1.0e-3,
            }),
            ("median", "standard", {
                **common,
                "hidden_layer_sizes": (128, 64, 32),
                "alpha": 1.0e-3,
                "learning_rate_init": 3.0e-4,
            }),
            ("most_frequent", "robust", {
                **common,
                "hidden_layer_sizes": (128, 64),
                "alpha": 1.0e-3,
                "learning_rate_init": 3.0e-4,
            }),
        ]
    elif model_key == "adaboost":
        definitions = [
            ("median", "none", {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "algorithm": "SAMME",
            }),
            ("median", "none", {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "algorithm": "SAMME",
            }),
            ("median", "none", {
                "n_estimators": 300,
                "learning_rate": 0.20,
                "algorithm": "SAMME",
            }),
            ("most_frequent", "none", {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "algorithm": "SAMME",
            }),
        ]
    elif model_key == "gradient_boosting":
        definitions = [
            ("median", "none", {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 2,
                "min_samples_leaf": 1,
                "subsample": 1.0,
            }),
            ("median", "none", {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "min_samples_leaf": 1,
                "subsample": 1.0,
            }),
            ("median", "none", {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 2,
                "min_samples_leaf": 3,
                "subsample": 0.9,
            }),
            ("most_frequent", "none", {
                "n_estimators": 200,
                "learning_rate": 0.10,
                "max_depth": 2,
                "min_samples_leaf": 3,
                "subsample": 0.9,
            }),
        ]
    elif model_key == "decision_tree":
        definitions = [
            ("median", "none", {
                "criterion": "gini",
                "max_depth": 10,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
            }),
            ("median", "none", {
                "criterion": "gini",
                "max_depth": 20,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
            }),
            ("median", "none", {
                "criterion": "entropy",
                "max_depth": 20,
                "min_samples_leaf": 3,
                "min_samples_split": 2,
            }),
            ("most_frequent", "none", {
                "criterion": "gini",
                "max_depth": None,
                "min_samples_leaf": 5,
                "min_samples_split": 10,
            }),
        ]
    else:
        raise AssertionError(model_key)

    return [
        _candidate(
            model_key,
            index,
            imputer=imputer,
            scaler=scaler,
            parameters=parameters,
        )
        for index, (imputer, scaler, parameters) in enumerate(
            definitions,
            start=1,
        )
    ]


def _smoke_candidates(model_key: str) -> List[CandidateSpec]:
    smoke_parameters: Dict[str, Dict[str, Any]] = {
        "random_forest": {
            "n_estimators": 20,
            "max_depth": 6,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        "xgboost": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.10,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
        },
        "svc": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "tol": 1.0e-3,
            "max_iter": 2000,
            "shrinking": True,
            "cache_size": 256.0,
        },
        "mlp": {
            "hidden_layer_sizes": (16,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1.0e-4,
            "batch_size": 16,
            "learning_rate_init": 1.0e-3,
            "max_iter": 80,
            "early_stopping": True,
            "validation_fraction": 0.20,
            "n_iter_no_change": 10,
            "tol": 1.0e-4,
        },
        "adaboost": {
            "n_estimators": 20,
            "learning_rate": 0.10,
            "algorithm": "SAMME",
        },
        "gradient_boosting": {
            "n_estimators": 20,
            "learning_rate": 0.10,
            "max_depth": 2,
            "min_samples_leaf": 1,
            "subsample": 1.0,
        },
        "decision_tree": {
            "criterion": "gini",
            "max_depth": 6,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
        },
    }

    scaler = "standard" if model_key in {"svc", "mlp"} else "none"
    return [
        _candidate(
            model_key,
            1,
            imputer="median",
            scaler=scaler,
            parameters=smoke_parameters[model_key],
        )
    ]


def get_candidate_specs(
    model_key: str,
    search_profile: str = "standard",
) -> List[CandidateSpec]:
    model_key = validate_model_key(model_key)
    profile = str(search_profile).strip().lower()
    if profile not in SEARCH_PROFILES:
        raise KeyError(
            f"Unsupported search profile {search_profile!r}; "
            f"expected one of {list(SEARCH_PROFILES)}."
        )
    candidates = (
        _standard_candidates(model_key)
        if profile == "standard"
        else _smoke_candidates(model_key)
    )
    if not candidates:
        raise AssertionError(f"No candidates configured for {model_key}.")
    return candidates


def validate_complete_suite() -> None:
    for model_key in MODEL_ORDER:
        for profile in SEARCH_PROFILES:
            candidates = get_candidate_specs(model_key, profile)
            identifiers = [item.candidate_id for item in candidates]
            if len(identifiers) != len(set(identifiers)):
                raise AssertionError(
                    f"Duplicate candidate IDs for {model_key}/{profile}."
                )


validate_complete_suite()


__all__ = [
    "CandidateSpec",
    "SEARCH_PROFILES",
    "get_candidate_specs",
    "validate_complete_suite",
]
