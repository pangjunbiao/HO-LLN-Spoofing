"""
Learning-rate scheduler utilities for Step 12 training.

Purpose:
- consistent scheduler protocol for full model and ablations,
- optional warmup,
- support common PyTorch schedulers,
- support ReduceLROnPlateau using validation metric,
- JSON-safe scheduler summaries.

Supported scheduler names:
- none
- reduce_on_plateau
- cosine
- step
- exponential

The trainer will call SchedulerController.step(...) once per epoch after
validation metrics are available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import math

import torch


@dataclass
class SchedulerConfig:
    """Configuration for learning-rate scheduler."""

    name: str = "reduce_on_plateau"
    enabled: bool = True

    monitor: str = "val_loss"
    mode: str = "min"

    warmup_epochs: int = 0
    warmup_start_factor: float = 0.1

    # ReduceLROnPlateau
    factor: float = 0.5
    patience: int = 8
    threshold: float = 1.0e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 1.0e-6

    # CosineAnnealingLR
    t_max: int = 50
    eta_min: float = 1.0e-6

    # StepLR
    step_size: int = 20
    gamma: float = 0.5

    # ExponentialLR
    exponential_gamma: float = 0.98

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerStepResult:
    """Result of one scheduler step."""

    epoch: int
    scheduler_name: str

    stepped: bool
    warmup_active: bool

    monitor: str
    metric_value: Optional[float]

    learning_rates_before: list[float]
    learning_rates_after: list[float]

    status: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerBuildSummary:
    """JSON-safe scheduler build summary."""

    scheduler_name: str
    enabled: bool
    monitor: str
    mode: str
    warmup_epochs: int
    base_learning_rates: list[float]
    status: str

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


def infer_scheduler_mode(monitor: str) -> str:
    """Infer scheduler direction from monitor name."""
    name = str(monitor).lower()

    minimize_keywords = ["loss", "error", "fpr", "far", "delay", "runtime"]
    maximize_keywords = ["auprc", "auroc", "f1", "precision", "recall", "adr", "attack_detection_rate"]

    if any(keyword in name for keyword in minimize_keywords):
        return "min"

    if any(keyword in name for keyword in maximize_keywords):
        return "max"

    return "min"


def build_scheduler_config(config: Optional[Mapping[str, Any]] = None) -> SchedulerConfig:
    """Build SchedulerConfig from full project config."""
    if config is None:
        return SchedulerConfig()

    monitor = str(_get_by_path(config, "training.scheduler.monitor", "val_loss"))
    mode = str(_get_by_path(config, "training.scheduler.mode", "auto"))

    if mode == "auto":
        mode = infer_scheduler_mode(monitor)

    return SchedulerConfig(
        name=str(_get_by_path(config, "training.scheduler.name", "reduce_on_plateau")),
        enabled=bool(_get_by_path(config, "training.scheduler.enabled", True)),
        monitor=monitor,
        mode=mode,
        warmup_epochs=int(_get_by_path(config, "training.scheduler.warmup_epochs", 0)),
        warmup_start_factor=float(
            _get_by_path(config, "training.scheduler.warmup_start_factor", 0.1)
        ),
        factor=float(_get_by_path(config, "training.scheduler.factor", 0.5)),
        patience=int(_get_by_path(config, "training.scheduler.patience", 8)),
        threshold=float(_get_by_path(config, "training.scheduler.threshold", 1.0e-4)),
        threshold_mode=str(_get_by_path(config, "training.scheduler.threshold_mode", "rel")),
        cooldown=int(_get_by_path(config, "training.scheduler.cooldown", 0)),
        min_lr=float(_get_by_path(config, "training.scheduler.min_lr", 1.0e-6)),
        t_max=int(_get_by_path(config, "training.scheduler.t_max", 50)),
        eta_min=float(_get_by_path(config, "training.scheduler.eta_min", 1.0e-6)),
        step_size=int(_get_by_path(config, "training.scheduler.step_size", 20)),
        gamma=float(_get_by_path(config, "training.scheduler.gamma", 0.5)),
        exponential_gamma=float(
            _get_by_path(config, "training.scheduler.exponential_gamma", 0.98)
        ),
    )


def get_current_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    """Return current learning rates."""
    return [float(group.get("lr", 0.0)) for group in optimizer.param_groups]


def set_current_lrs(optimizer: torch.optim.Optimizer, learning_rates: Sequence[float]) -> None:
    """Set current learning rates."""
    if len(learning_rates) != len(optimizer.param_groups):
        raise ValueError(
            f"learning_rates length mismatch. Expected {len(optimizer.param_groups)}, "
            f"got {len(learning_rates)}."
        )

    for group, lr in zip(optimizer.param_groups, learning_rates):
        group["lr"] = float(lr)


def _validate_scheduler_config(config: SchedulerConfig) -> None:
    """Validate scheduler config."""
    name = str(config.name).lower().strip()

    if name not in {"none", "reduce_on_plateau", "plateau", "cosine", "step", "exponential"}:
        raise ValueError(
            f"Unknown scheduler name='{config.name}'. "
            "Supported: none, reduce_on_plateau, cosine, step, exponential."
        )

    if config.mode not in {"min", "max"}:
        raise ValueError("scheduler.mode must be 'min', 'max', or 'auto' before construction.")

    if config.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")

    if not (0.0 < config.warmup_start_factor <= 1.0):
        raise ValueError("warmup_start_factor must be in (0, 1].")

    if config.factor <= 0.0 or config.factor >= 1.0:
        raise ValueError("ReduceLROnPlateau factor must be in (0, 1).")

    if config.patience < 0:
        raise ValueError("scheduler patience must be non-negative.")

    if config.min_lr < 0.0 or config.eta_min < 0.0:
        raise ValueError("minimum learning rates must be non-negative.")

    if config.t_max <= 0:
        raise ValueError("t_max must be positive.")

    if config.step_size <= 0:
        raise ValueError("step_size must be positive.")

    if config.gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    if config.exponential_gamma <= 0.0:
        raise ValueError("exponential_gamma must be positive.")


class SchedulerController:
    """
    Unified scheduler wrapper.

    The trainer calls:
        scheduler.step(epoch=epoch, metrics=val_metrics)

    This class handles:
    - optional warmup,
    - ReduceLROnPlateau metric stepping,
    - regular epoch schedulers.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: Optional[SchedulerConfig] = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config or SchedulerConfig()

        _validate_scheduler_config(self.config)

        self.name = str(self.config.name).lower().strip()
        self.enabled = bool(self.config.enabled) and self.name != "none"

        self.base_lrs = get_current_lrs(self.optimizer)
        self.last_epoch = 0

        self.scheduler: Optional[Any] = None

        if self.enabled:
            self.scheduler = self._create_scheduler()

    @classmethod
    def from_project_config(
        cls,
        optimizer: torch.optim.Optimizer,
        config: Mapping[str, Any],
    ) -> "SchedulerController":
        """Construct scheduler controller from project config."""
        return cls(
            optimizer=optimizer,
            config=build_scheduler_config(config),
        )

    def _create_scheduler(self) -> Optional[Any]:
        """Create underlying PyTorch scheduler."""
        if self.name in {"reduce_on_plateau", "plateau"}:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=str(self.config.mode),
                factor=float(self.config.factor),
                patience=int(self.config.patience),
                threshold=float(self.config.threshold),
                threshold_mode=str(self.config.threshold_mode),
                cooldown=int(self.config.cooldown),
                min_lr=float(self.config.min_lr),
            )

        if self.name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(self.config.t_max),
                eta_min=float(self.config.eta_min),
            )

        if self.name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(self.config.step_size),
                gamma=float(self.config.gamma),
            )

        if self.name == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=float(self.config.exponential_gamma),
            )

        return None

    def _warmup_active(self, epoch: int) -> bool:
        """Return whether warmup is active."""
        return self.enabled and int(epoch) <= int(self.config.warmup_epochs) and self.config.warmup_epochs > 0

    def _apply_warmup(self, epoch: int) -> None:
        """
        Apply linear warmup.

        At epoch 1:
            lr = base_lr * warmup_start_factor

        At epoch warmup_epochs:
            lr approaches base_lr
        """
        warmup_epochs = int(self.config.warmup_epochs)

        if warmup_epochs <= 0:
            return

        if warmup_epochs == 1:
            factor = 1.0
        else:
            progress = (int(epoch) - 1) / max(warmup_epochs - 1, 1)
            factor = float(self.config.warmup_start_factor) + (
                1.0 - float(self.config.warmup_start_factor)
            ) * progress

        factor = max(float(self.config.warmup_start_factor), min(1.0, factor))
        set_current_lrs(self.optimizer, [base_lr * factor for base_lr in self.base_lrs])

    def _metric_from_metrics(self, metrics: Optional[Mapping[str, Any]]) -> Optional[float]:
        """Extract monitored metric."""
        if metrics is None:
            return None

        if self.config.monitor not in metrics:
            return None

        value = metrics[self.config.monitor]

        if hasattr(value, "item"):
            value = value.item()

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    def step(
        self,
        epoch: int,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> SchedulerStepResult:
        """
        Step scheduler once per epoch.

        Args:
            epoch:
                1-based epoch index.
            metrics:
                Validation metric dict. Required for ReduceLROnPlateau.

        Returns:
            SchedulerStepResult.
        """
        epoch = int(epoch)
        self.last_epoch = epoch

        lrs_before = get_current_lrs(self.optimizer)

        if not self.enabled:
            return SchedulerStepResult(
                epoch=epoch,
                scheduler_name=self.name,
                stepped=False,
                warmup_active=False,
                monitor=str(self.config.monitor),
                metric_value=None,
                learning_rates_before=lrs_before,
                learning_rates_after=get_current_lrs(self.optimizer),
                status="PASSED",
                message="Scheduler disabled.",
            )

        if self._warmup_active(epoch):
            self._apply_warmup(epoch)

            return SchedulerStepResult(
                epoch=epoch,
                scheduler_name=self.name,
                stepped=True,
                warmup_active=True,
                monitor=str(self.config.monitor),
                metric_value=self._metric_from_metrics(metrics),
                learning_rates_before=lrs_before,
                learning_rates_after=get_current_lrs(self.optimizer),
                status="PASSED",
                message="Warmup step applied.",
            )

        metric_value = self._metric_from_metrics(metrics)

        if self.name in {"reduce_on_plateau", "plateau"}:
            if metric_value is None:
                return SchedulerStepResult(
                    epoch=epoch,
                    scheduler_name=self.name,
                    stepped=False,
                    warmup_active=False,
                    monitor=str(self.config.monitor),
                    metric_value=None,
                    learning_rates_before=lrs_before,
                    learning_rates_after=get_current_lrs(self.optimizer),
                    status="WARNING",
                    message=(
                        f"Metric '{self.config.monitor}' missing or non-finite; "
                        "ReduceLROnPlateau step skipped."
                    ),
                )

            assert self.scheduler is not None
            self.scheduler.step(metric_value)

            return SchedulerStepResult(
                epoch=epoch,
                scheduler_name=self.name,
                stepped=True,
                warmup_active=False,
                monitor=str(self.config.monitor),
                metric_value=float(metric_value),
                learning_rates_before=lrs_before,
                learning_rates_after=get_current_lrs(self.optimizer),
                status="PASSED",
                message="ReduceLROnPlateau step applied.",
            )

        assert self.scheduler is not None
        self.scheduler.step()

        return SchedulerStepResult(
            epoch=epoch,
            scheduler_name=self.name,
            stepped=True,
            warmup_active=False,
            monitor=str(self.config.monitor),
            metric_value=metric_value,
            learning_rates_before=lrs_before,
            learning_rates_after=get_current_lrs(self.optimizer),
            status="PASSED",
            message=f"{self.name} scheduler step applied.",
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable scheduler state."""
        return {
            "config": self.config.to_dict(),
            "name": self.name,
            "enabled": self.enabled,
            "base_lrs": list(self.base_lrs),
            "last_epoch": int(self.last_epoch),
            "scheduler_state": None if self.scheduler is None else self.scheduler.state_dict(),
            "optimizer_lrs": get_current_lrs(self.optimizer),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load scheduler state."""
        self.last_epoch = int(state_dict.get("last_epoch", 0))

        base_lrs = state_dict.get("base_lrs")
        if base_lrs is not None:
            self.base_lrs = [float(value) for value in base_lrs]

        scheduler_state = state_dict.get("scheduler_state")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        optimizer_lrs = state_dict.get("optimizer_lrs")
        if optimizer_lrs is not None:
            set_current_lrs(self.optimizer, [float(value) for value in optimizer_lrs])

    def build_summary(self) -> SchedulerBuildSummary:
        """Return JSON-safe scheduler build summary."""
        return SchedulerBuildSummary(
            scheduler_name=str(self.name),
            enabled=bool(self.enabled),
            monitor=str(self.config.monitor),
            mode=str(self.config.mode),
            warmup_epochs=int(self.config.warmup_epochs),
            base_learning_rates=list(self.base_lrs),
            status="PASSED",
        )

    def summary(self) -> Dict[str, Any]:
        """Return JSON-safe scheduler summary."""
        return {
            "build_summary": self.build_summary().to_dict(),
            "current_learning_rates": get_current_lrs(self.optimizer),
            "last_epoch": int(self.last_epoch),
            "underlying_scheduler": None if self.scheduler is None else self.scheduler.__class__.__name__,
        }


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Optional[SchedulerConfig] = None,
) -> Tuple[SchedulerController, SchedulerBuildSummary]:
    """Create scheduler controller."""
    controller = SchedulerController(
        optimizer=optimizer,
        config=config or SchedulerConfig(),
    )
    return controller, controller.build_summary()


def create_scheduler_from_project_config(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> Tuple[SchedulerController, SchedulerBuildSummary]:
    """Create scheduler controller from project config."""
    return create_scheduler(
        optimizer=optimizer,
        config=build_scheduler_config(config),
    )


__all__ = [
    "SchedulerConfig",
    "SchedulerStepResult",
    "SchedulerBuildSummary",
    "infer_scheduler_mode",
    "build_scheduler_config",
    "get_current_lrs",
    "set_current_lrs",
    "SchedulerController",
    "create_scheduler",
    "create_scheduler_from_project_config",
]