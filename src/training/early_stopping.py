"""
Early stopping utilities for Step 12 training.

Purpose:
- monitor validation metric,
- track best epoch,
- stop after patience epochs without improvement,
- provide consistent protocol for full model and ablations.

This module does not save model weights by itself.
The trainer will save best checkpoints when EarlyStopping reports improvement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import math


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    enabled: bool = True

    monitor: str = "val_loss"
    mode: str = "min"  # min or max

    patience: int = 20
    min_delta: float = 0.0
    warmup_epochs: int = 0

    restore_best: bool = True
    stop_on_nan: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EarlyStoppingStepResult:
    """Result of one early-stopping update."""

    epoch: int
    monitor: str
    current_value: float

    best_value: Optional[float]
    best_epoch: Optional[int]

    improved: bool
    bad_epochs: int

    should_stop: bool
    stop_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EarlyStoppingState:
    """Serializable early-stopping state."""

    best_value: Optional[float] = None
    best_epoch: Optional[int] = None

    bad_epochs: int = 0
    stopped_epoch: Optional[int] = None
    should_stop: bool = False
    stop_reason: Optional[str] = None

    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_by_path(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Small local config helper."""
    current: Any = config

    for key in path.split("."):
        if not isinstance(current, Mapping):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def infer_early_stopping_mode(monitor: str) -> str:
    """Infer monitor direction from metric name."""
    name = str(monitor).lower()

    minimize_keywords = [
        "loss",
        "error",
        "fpr",
        "far",
        "delay",
        "runtime",
    ]

    maximize_keywords = [
        "accuracy",
        "auprc",
        "auroc",
        "f1",
        "precision",
        "recall",
        "adr",
        "attack_detection_rate",
    ]

    if any(item in name for item in minimize_keywords):
        return "min"

    if any(item in name for item in maximize_keywords):
        return "max"

    return "min"


def build_early_stopping_config(config: Optional[Mapping[str, Any]] = None) -> EarlyStoppingConfig:
    """Build EarlyStoppingConfig from project config."""
    if config is None:
        return EarlyStoppingConfig()

    monitor = str(_get_by_path(config, "training.early_stopping.monitor", "val_loss"))
    mode = str(_get_by_path(config, "training.early_stopping.mode", "auto"))

    if mode == "auto":
        mode = infer_early_stopping_mode(monitor)

    return EarlyStoppingConfig(
        enabled=bool(_get_by_path(config, "training.early_stopping.enabled", True)),
        monitor=monitor,
        mode=mode,
        patience=int(_get_by_path(config, "training.early_stopping.patience", 20)),
        min_delta=float(_get_by_path(config, "training.early_stopping.min_delta", 0.0)),
        warmup_epochs=int(_get_by_path(config, "training.early_stopping.warmup_epochs", 0)),
        restore_best=bool(_get_by_path(config, "training.early_stopping.restore_best", True)),
        stop_on_nan=bool(_get_by_path(config, "training.early_stopping.stop_on_nan", True)),
    )


def _is_finite(value: float) -> bool:
    """Check finite scalar."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


class EarlyStopping:
    """
    Early stopping tracker.

    Usage:
        early = EarlyStopping(config)
        result = early.step(epoch, metrics)
        if result.improved:
            save checkpoint
        if result.should_stop:
            break
    """

    def __init__(self, config: Optional[EarlyStoppingConfig] = None) -> None:
        self.config = config or EarlyStoppingConfig()

        self.monitor = str(self.config.monitor)
        self.mode = str(self.config.mode).lower().strip()

        if self.mode not in {"min", "max"}:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'.")

        if self.config.patience < 0:
            raise ValueError("EarlyStopping patience must be non-negative.")

        if self.config.warmup_epochs < 0:
            raise ValueError("EarlyStopping warmup_epochs must be non-negative.")

        self.state = EarlyStoppingState()

    @classmethod
    def from_project_config(cls, config: Mapping[str, Any]) -> "EarlyStopping":
        """Construct from project config."""
        return cls(build_early_stopping_config(config))

    @property
    def best_value(self) -> Optional[float]:
        return self.state.best_value

    @property
    def best_epoch(self) -> Optional[int]:
        return self.state.best_epoch

    @property
    def should_stop(self) -> bool:
        return self.state.should_stop

    def _metric_value_from_metrics(self, metrics: Mapping[str, Any]) -> float:
        """Extract monitored metric from metrics dict."""
        if self.monitor not in metrics:
            available = sorted(str(key) for key in metrics.keys())
            raise KeyError(
                f"EarlyStopping monitor '{self.monitor}' not found in metrics. "
                f"Available keys: {available}"
            )

        value = metrics[self.monitor]

        if hasattr(value, "item"):
            value = value.item()

        return float(value)

    def _is_improvement(self, current_value: float) -> bool:
        """Check whether current value improves best."""
        if self.state.best_value is None:
            return True

        best = float(self.state.best_value)
        delta = float(self.config.min_delta)

        if self.mode == "min":
            return current_value < (best - delta)

        return current_value > (best + delta)

    def step(
        self,
        epoch: int,
        metrics: Mapping[str, Any],
    ) -> EarlyStoppingStepResult:
        """
        Update early stopping with current epoch metrics.

        Args:
            epoch:
                1-based epoch index recommended.
            metrics:
                Dict containing monitor key.

        Returns:
            EarlyStoppingStepResult.
        """
        current_value = self._metric_value_from_metrics(metrics)

        improved = False
        stop_reason: Optional[str] = None

        if not _is_finite(current_value):
            if self.config.stop_on_nan:
                self.state.should_stop = True
                self.state.stopped_epoch = int(epoch)
                self.state.stop_reason = f"Non-finite monitored value: {current_value}"

                result = EarlyStoppingStepResult(
                    epoch=int(epoch),
                    monitor=self.monitor,
                    current_value=float(current_value),
                    best_value=self.state.best_value,
                    best_epoch=self.state.best_epoch,
                    improved=False,
                    bad_epochs=int(self.state.bad_epochs),
                    should_stop=True,
                    stop_reason=self.state.stop_reason,
                )
                self.state.history.append(result.to_dict())
                return result

            current_value = float("inf") if self.mode == "min" else float("-inf")

        if self._is_improvement(current_value):
            improved = True
            self.state.best_value = float(current_value)
            self.state.best_epoch = int(epoch)
            self.state.bad_epochs = 0
        else:
            self.state.bad_epochs += 1

        in_warmup = int(epoch) <= int(self.config.warmup_epochs)

        if not self.config.enabled:
            should_stop = False
        elif in_warmup:
            should_stop = False
        else:
            should_stop = self.state.bad_epochs > int(self.config.patience)

        if should_stop:
            self.state.should_stop = True
            self.state.stopped_epoch = int(epoch)
            stop_reason = (
                f"No improvement in '{self.monitor}' for "
                f"{self.state.bad_epochs} epochs; patience={self.config.patience}."
            )
            self.state.stop_reason = stop_reason

        result = EarlyStoppingStepResult(
            epoch=int(epoch),
            monitor=self.monitor,
            current_value=float(current_value),
            best_value=self.state.best_value,
            best_epoch=self.state.best_epoch,
            improved=bool(improved),
            bad_epochs=int(self.state.bad_epochs),
            should_stop=bool(self.state.should_stop),
            stop_reason=self.state.stop_reason,
        )

        self.state.history.append(result.to_dict())
        return result

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable state."""
        return {
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore state."""
        state = state_dict.get("state", state_dict)

        self.state = EarlyStoppingState(
            best_value=state.get("best_value"),
            best_epoch=state.get("best_epoch"),
            bad_epochs=int(state.get("bad_epochs", 0)),
            stopped_epoch=state.get("stopped_epoch"),
            should_stop=bool(state.get("should_stop", False)),
            stop_reason=state.get("stop_reason"),
            history=list(state.get("history", [])),
        )

    def reset(self) -> None:
        """Reset early stopping state."""
        self.state = EarlyStoppingState()

    def best_summary(self) -> Dict[str, Any]:
        """Return best metric summary."""
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_value": self.state.best_value,
            "best_epoch": self.state.best_epoch,
            "bad_epochs": self.state.bad_epochs,
            "should_stop": self.state.should_stop,
            "stopped_epoch": self.state.stopped_epoch,
            "stop_reason": self.state.stop_reason,
            "enabled": self.config.enabled,
            "patience": self.config.patience,
            "min_delta": self.config.min_delta,
            "warmup_epochs": self.config.warmup_epochs,
            "restore_best": self.config.restore_best,
        }

    def __repr__(self) -> str:
        return (
            "EarlyStopping("
            f"monitor={self.monitor!r}, mode={self.mode!r}, "
            f"best_value={self.state.best_value}, best_epoch={self.state.best_epoch}, "
            f"bad_epochs={self.state.bad_epochs}, should_stop={self.state.should_stop}"
            ")"
        )


__all__ = [
    "EarlyStoppingConfig",
    "EarlyStoppingStepResult",
    "EarlyStoppingState",
    "infer_early_stopping_mode",
    "build_early_stopping_config",
    "EarlyStopping",
]