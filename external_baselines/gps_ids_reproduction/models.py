"""
Model factory for the protocol-controlled GPS-IDS classifier suite.

Supported classifiers
---------------------
- Random Forest
- XGBoost
- SVC
- MLP
- AdaBoost
- Gradient Boosting
- Decision Tree

Every estimator is placed behind the same train-fitted preprocessing pipeline:
SimpleImputer -> optional scaler -> classifier.

The feature matrix is never modified per model. Only preprocessing and model
hyperparameters differ, and those choices are selected on Dataset-1 validation.
"""

from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import sklearn
from joblib import dump, load
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_ORDER: Tuple[str, ...] = (
    "random_forest",
    "xgboost",
    "svc",
    "mlp",
    "adaboost",
    "gradient_boosting",
    "decision_tree",
)

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "random_forest": "GPS-IDS–Random Forest",
    "xgboost": "GPS-IDS–XGBoost",
    "svc": "GPS-IDS–SVC",
    "mlp": "GPS-IDS–MLP",
    "adaboost": "GPS-IDS–AdaBoost",
    "gradient_boosting": "GPS-IDS–Gradient Boosting",
    "decision_tree": "GPS-IDS–Decision Tree",
}

MODEL_REPORTING_ROLES: Dict[str, str] = {
    "random_forest": "supplementary_reproduction_completeness",
    "xgboost": "supplementary_reproduction_completeness",
    "svc": "supplementary_reproduction_completeness",
    "mlp": "primary_published_method_baseline",
    "adaboost": "supplementary_reproduction_completeness",
    "gradient_boosting": "supplementary_reproduction_completeness",
    "decision_tree": "supplementary_reproduction_completeness",
}


@dataclass(frozen=True)
class PipelineBuildSpec:
    model_key: str
    model_display_name: str
    reporting_role: str
    imputer_strategy: str
    scaler_name: str
    model_parameters: Dict[str, Any]
    active_seed: int
    n_jobs: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_display_name": self.model_display_name,
            "reporting_role": self.reporting_role,
            "imputer_strategy": self.imputer_strategy,
            "scaler_name": self.scaler_name,
            "model_parameters": _json_safe_parameters(
                self.model_parameters
            ),
            "active_seed": int(self.active_seed),
            "n_jobs": int(self.n_jobs),
        }


def _json_safe_parameters(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, tuple):
            output[str(key)] = list(value)
        elif isinstance(value, np.generic):
            output[str(key)] = value.item()
        else:
            output[str(key)] = value
    return output


def validate_model_key(model_key: str) -> str:
    normalized = str(model_key).strip().lower()
    if normalized not in MODEL_ORDER:
        raise KeyError(
            f"Unsupported GPS-IDS classifier {model_key!r}. "
            f"Supported classifiers: {list(MODEL_ORDER)}"
        )
    return normalized


def dependency_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "python": sys_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": importlib.import_module("joblib").__version__,
        "xgboost": None,
    }
    try:
        xgboost = importlib.import_module("xgboost")
        versions["xgboost"] = str(xgboost.__version__)
    except Exception:
        versions["xgboost"] = None
    return versions


def sys_version() -> str:
    import sys

    return sys.version.split()[0]


def require_xgboost() -> Any:
    try:
        module = importlib.import_module("xgboost")
    except Exception as exc:
        raise ImportError(
            "XGBoost is required for the complete seven-classifier GPS-IDS "
            "suite. Install it in the active environment before running Step 5."
        ) from exc
    return module


def _build_classifier(
    model_key: str,
    parameters: Mapping[str, Any],
    *,
    seed: int,
    n_jobs: int,
) -> Any:
    model_key = validate_model_key(model_key)
    params = dict(parameters)

    if model_key == "random_forest":
        return RandomForestClassifier(
            random_state=int(seed),
            n_jobs=int(n_jobs),
            **params,
        )

    if model_key == "xgboost":
        xgboost = require_xgboost()
        return xgboost.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=int(seed),
            n_jobs=int(n_jobs),
            verbosity=0,
            **params,
        )

    if model_key == "svc":
        return SVC(
            probability=True,
            random_state=int(seed),
            cache_size=float(params.pop("cache_size", 2048.0)),
            **params,
        )

    if model_key == "mlp":
        return MLPClassifier(
            random_state=int(seed),
            **params,
        )

    if model_key == "adaboost":
        # scikit-learn removed the ``algorithm`` constructor argument in
        # newer releases. Keep one source file compatible with both APIs.
        supported = inspect.signature(
            AdaBoostClassifier
        ).parameters
        if "algorithm" not in supported:
            params.pop("algorithm", None)
        return AdaBoostClassifier(
            random_state=int(seed),
            **params,
        )

    if model_key == "gradient_boosting":
        return GradientBoostingClassifier(
            random_state=int(seed),
            **params,
        )

    if model_key == "decision_tree":
        return DecisionTreeClassifier(
            random_state=int(seed),
            **params,
        )

    raise AssertionError(f"Unreachable model key: {model_key}")


def _build_scaler(scaler_name: str) -> Any:
    normalized = str(scaler_name).strip().lower()
    if normalized == "none":
        return "passthrough"
    if normalized == "standard":
        return StandardScaler()
    if normalized == "robust":
        return RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
        )
    raise KeyError(
        f"Unsupported scaler {scaler_name!r}; expected none, standard, or robust."
    )


def build_pipeline(
    *,
    model_key: str,
    imputer_strategy: str,
    scaler_name: str,
    model_parameters: Mapping[str, Any],
    seed: int,
    n_jobs: int,
) -> Tuple[Pipeline, PipelineBuildSpec]:
    """
    Build one deterministic candidate pipeline.

    The imputer and scaler are unfitted here. Calling ``fit`` on Dataset-1
    training rows is the only operation allowed to estimate their parameters.
    """
    model_key = validate_model_key(model_key)
    strategy = str(imputer_strategy).strip().lower()
    if strategy not in {"median", "most_frequent"}:
        raise KeyError(
            f"Unsupported imputer strategy {imputer_strategy!r}; "
            "expected median or most_frequent."
        )

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=strategy,
                    add_indicator=False,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", _build_scaler(scaler_name)),
            (
                "model",
                _build_classifier(
                    model_key=model_key,
                    parameters=model_parameters,
                    seed=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )

    spec = PipelineBuildSpec(
        model_key=model_key,
        model_display_name=MODEL_DISPLAY_NAMES[model_key],
        reporting_role=MODEL_REPORTING_ROLES[model_key],
        imputer_strategy=strategy,
        scaler_name=str(scaler_name).strip().lower(),
        model_parameters=dict(model_parameters),
        active_seed=int(seed),
        n_jobs=int(n_jobs),
    )
    return pipeline, spec


def save_pipeline(pipeline: Pipeline, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, path, compress=3)
    return path


def load_pipeline(path: Path) -> Pipeline:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    pipeline = load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"Expected sklearn Pipeline in {path}, got {type(pipeline).__name__}."
        )
    return pipeline


def _tree_node_count(estimator: Any) -> int:
    tree = getattr(estimator, "tree_", None)
    return int(getattr(tree, "node_count", 0)) if tree is not None else 0


def count_learned_parameters(pipeline: Pipeline) -> int:
    """
    Return a transparent learned-state count for provenance.

    For neural/SVM models this counts learned numeric coefficients. For tree
    ensembles it counts learned tree nodes. It is a reproducibility descriptor,
    not a claim of identical statistical degrees of freedom across model types.
    """
    model = pipeline.named_steps["model"]

    if isinstance(model, MLPClassifier):
        total = 0
        for array in getattr(model, "coefs_", []):
            total += int(np.asarray(array).size)
        for array in getattr(model, "intercepts_", []):
            total += int(np.asarray(array).size)
        return int(total)

    if isinstance(model, SVC):
        total = 0
        for name in (
            "support_vectors_",
            "dual_coef_",
            "intercept_",
            "probA_",
            "probB_",
        ):
            value = getattr(model, name, None)
            if value is not None:
                total += int(np.asarray(value).size)
        return int(total)

    if isinstance(model, RandomForestClassifier):
        return int(
            sum(_tree_node_count(tree) for tree in model.estimators_)
        )

    if isinstance(model, GradientBoostingClassifier):
        estimators = np.asarray(model.estimators_, dtype=object).reshape(-1)
        return int(
            sum(_tree_node_count(tree) for tree in estimators)
        )

    if isinstance(model, AdaBoostClassifier):
        return int(
            sum(_tree_node_count(tree) for tree in model.estimators_)
        )

    if isinstance(model, DecisionTreeClassifier):
        return _tree_node_count(model)

    # XGBoost sklearn wrapper.
    if model.__class__.__module__.startswith("xgboost"):
        try:
            frame = model.get_booster().trees_to_dataframe()
            return int(len(frame))
        except Exception:
            try:
                return int(model.get_booster().num_boosted_rounds())
            except Exception:
                return 0

    return 0


def extract_preprocessing_state(
    pipeline: Pipeline,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Extract train-fitted imputer/scaler state for an auditable manifest."""
    feature_names = [str(name) for name in feature_names]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    statistics = np.asarray(imputer.statistics_, dtype=float).reshape(-1)
    if statistics.size != len(feature_names):
        raise ValueError(
            "Imputer statistics length does not match the feature contract."
        )

    payload: Dict[str, Any] = {
        "feature_names": feature_names,
        "imputer": {
            "class": type(imputer).__name__,
            "strategy": str(imputer.strategy),
            "statistics": {
                feature: (
                    float(value) if math.isfinite(float(value)) else None
                )
                for feature, value in zip(feature_names, statistics)
            },
            "add_indicator": bool(imputer.add_indicator),
            "keep_empty_features": bool(imputer.keep_empty_features),
        },
        "scaler": {
            "class": (
                "passthrough"
                if scaler == "passthrough"
                else type(scaler).__name__
            )
        },
    }

    if scaler != "passthrough":
        for attribute in (
            "mean_",
            "scale_",
            "center_",
            "var_",
        ):
            value = getattr(scaler, attribute, None)
            if value is None:
                continue
            array = np.asarray(value, dtype=float).reshape(-1)
            payload["scaler"][attribute] = {
                feature: (
                    float(item) if math.isfinite(float(item)) else None
                )
                for feature, item in zip(feature_names, array)
            }

    return payload


__all__ = [
    "MODEL_DISPLAY_NAMES",
    "MODEL_ORDER",
    "MODEL_REPORTING_ROLES",
    "PipelineBuildSpec",
    "build_pipeline",
    "count_learned_parameters",
    "dependency_versions",
    "extract_preprocessing_state",
    "load_pipeline",
    "require_xgboost",
    "save_pipeline",
    "validate_model_key",
]
