"""
MLP-xi baseline for causal GNSS spoofing detection.

Step 15 purpose:
- Train a standard feed-forward MLP baseline on the same reconstructed xi_t features.
- Use Dataset-1 train only for fitting.
- Use Dataset-1 validation only for early stopping and threshold selection later.
- Evaluate Dataset-1 test, Dataset-2 external, and Dataset-3 online later.
- Do not use raw shortcut columns.

Important:
- This is a fair time-step baseline, not an artificially weakened baseline.
- It sees the same xi_t feature vector as the proposed model.
- It does not receive recurrent/causal hidden state.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.xgboost_baseline import (
    BaselineSplitData,
    compute_scale_pos_weight,
    get_baseline_feature_columns,
    load_baseline_split_data,
    make_prediction_bundle_from_probabilities,
)
from src.evaluation.evaluate_dataset1 import EvaluationPredictionBundle
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir
from src.utils.seed import set_global_seed


@dataclass
class MLPBaselineConfig:
    """Configuration for MLP-xi baseline."""

    model_name: str = "MLP-xi"

    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.10
    activation: str = "relu"
    use_layer_norm: bool = False

    batch_size: int = 512
    max_epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0

    patience: int = 12
    min_delta: float = 1.0e-5

    use_train_class_weight: bool = True
    num_workers: int = 0

    output_model_path: str = "results/models/mlp_xi.pt"
    output_history_csv: str = "results/tables/mlp_xi_training_history.csv"
    output_summary_json: str = "results/tables/mlp_xi_training_summary.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MLPTrainingEpochResult:
    """One MLP training epoch result."""

    epoch: int
    train_loss: float
    val_loss: float
    val_auprc: Optional[float]
    val_accuracy: float
    learning_rate: float
    runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MLPBaselineArtifact:
    """Trained MLP baseline artifact."""

    model_name: str
    model: nn.Module
    config: MLPBaselineConfig
    model_path: Path
    feature_columns: List[str]

    best_epoch: int
    best_val_loss: float
    train_summary: Dict[str, Any]
    val_summary: Dict[str, Any]
    history: List[MLPTrainingEpochResult]
    fit_runtime_seconds: float
    active_seed: int
    device: str

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "config": self.config.to_dict(),
            "feature_columns": list(self.feature_columns),
            "best_epoch": int(self.best_epoch),
            "best_val_loss": float(self.best_val_loss),
            "train_summary": self.train_summary,
            "val_summary": self.val_summary,
            "history": [item.to_dict() for item in self.history],
            "fit_runtime_seconds": float(self.fit_runtime_seconds),
            "active_seed": int(self.active_seed),
            "device": str(self.device),
        }


class MLPXiClassifier(nn.Module):
    """Simple feed-forward MLP for xi_t time-step classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.10,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        activation = str(activation).lower().strip()

        layers: List[nn.Module] = []
        previous_dim = int(input_dim)

        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)

            layers.append(nn.Linear(previous_dim, hidden_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())

            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))

            previous_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(previous_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits with shape [B]."""
        features = self.backbone(x)
        logits = self.output(features).squeeze(-1)
        return logits


def _project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve project-relative path."""
    return resolve_project_path(config, str(value))


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to finite float or None."""
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def _json_safe(value: Any) -> Any:
    """Make payload JSON-safe."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def save_json_safe(payload: Mapping[str, Any], output_path: Path | str) -> Path:
    """Save JSON safely."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(dict(payload)), file, indent=2)

    return output_path


def build_mlp_baseline_config(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> MLPBaselineConfig:
    """Build MLP baseline config."""
    base = "baselines.mlp_xi"

    return MLPBaselineConfig(
        model_name=str(get_by_path(config, f"{base}.model_name", "MLP-xi")),
        hidden_dims=list(get_by_path(config, f"{base}.hidden_dims", [128, 64])),
        dropout=float(get_by_path(config, f"{base}.dropout", 0.10)),
        activation=str(get_by_path(config, f"{base}.activation", "relu")),
        use_layer_norm=bool(get_by_path(config, f"{base}.use_layer_norm", False)),
        batch_size=int(get_by_path(config, f"{base}.batch_size", 512)),
        max_epochs=int(get_by_path(config, f"{base}.max_epochs", 80)),
        learning_rate=float(get_by_path(config, f"{base}.learning_rate", 1.0e-3)),
        weight_decay=float(get_by_path(config, f"{base}.weight_decay", 1.0e-4)),
        gradient_clip_norm=float(get_by_path(config, f"{base}.gradient_clip_norm", 1.0)),
        patience=int(get_by_path(config, f"{base}.patience", 12)),
        min_delta=float(get_by_path(config, f"{base}.min_delta", 1.0e-5)),
        use_train_class_weight=bool(get_by_path(config, f"{base}.use_train_class_weight", True)),
        num_workers=int(get_by_path(config, f"{base}.num_workers", 0)),
        output_model_path=str(get_by_path(config, f"{base}.output_model_path", "results/models/mlp_xi.pt")),
        output_history_csv=str(
            get_by_path(config, f"{base}.output_history_csv", "results/tables/mlp_xi_training_history.csv")
        ),
        output_summary_json=str(
            get_by_path(config, f"{base}.output_summary_json", "results/tables/mlp_xi_training_summary.json")
        ),
    )


def make_mlp_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    active_seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader for flattened MLP baseline."""
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(int(active_seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        generator=generator,
        drop_last=False,
    )


def compute_binary_auprc(y_true: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    """Compute AUPRC if sklearn is available and both classes exist."""
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return None

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)

    if len(np.unique(y_true)) < 2:
        return None

    return float(average_precision_score(y_true, probabilities))


def compute_accuracy(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> float:
    """Compute simple 0.5 accuracy for training monitoring only."""
    y_true = np.asarray(y_true).astype(int)
    predictions = (np.asarray(probabilities) >= float(threshold)).astype(int)

    if len(y_true) == 0:
        return 0.0

    return float((predictions == y_true).mean())


def _make_pos_weight(train_y: np.ndarray, cfg: MLPBaselineConfig, device: torch.device) -> Optional[Tensor]:
    """Create BCE positive-class weight."""
    if not cfg.use_train_class_weight:
        return None

    scale_pos_weight = compute_scale_pos_weight(train_y)
    return torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    """Train one epoch."""
    model.train()

    losses: List[float] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()

        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))

        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        return float("nan")

    return float(np.mean(losses))


@torch.no_grad()
def _evaluate_mlp_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate one epoch."""
    model.eval()

    losses: List[float] = []
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        probs = torch.sigmoid(logits)

        losses.append(float(loss.detach().cpu().item()))
        probabilities.append(probs.detach().cpu().numpy().reshape(-1))
        labels.append(y_batch.detach().cpu().numpy().reshape(-1))

    if probabilities:
        p = np.concatenate(probabilities).astype(np.float32)
        y = np.concatenate(labels).astype(np.int64)
    else:
        p = np.asarray([], dtype=np.float32)
        y = np.asarray([], dtype=np.int64)

    if not losses:
        return float("nan"), p, y

    return float(np.mean(losses)), p, y


def save_mlp_checkpoint(
    model: nn.Module,
    model_path: Path | str,
    cfg: MLPBaselineConfig,
    feature_columns: Sequence[str],
    epoch: int,
    val_loss: float,
    active_seed: int,
) -> Path:
    """Save MLP baseline checkpoint."""
    model_path = Path(model_path)
    ensure_dir(model_path.parent)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "feature_columns": list(feature_columns),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "active_seed": int(active_seed),
            "model_name": cfg.model_name,
        },
        model_path,
    )

    return model_path


def train_mlp_baseline(
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
) -> MLPBaselineArtifact:
    """Train MLP-xi baseline on Dataset-1 train valid rows."""
    start_time = time.perf_counter()

    set_global_seed(int(active_seed))

    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    cfg = build_mlp_baseline_config(config=config, active_seed=active_seed)
    feature_columns = get_baseline_feature_columns(config)

    train_data = load_baseline_split_data(
        config=config,
        split_name="train",
        feature_columns=feature_columns,
    )
    val_data = load_baseline_split_data(
        config=config,
        split_name="val",
        feature_columns=feature_columns,
    )

    model = MLPXiClassifier(
        input_dim=len(feature_columns),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        activation=cfg.activation,
        use_layer_norm=cfg.use_layer_norm,
    ).to(device)

    train_loader = make_mlp_dataloader(
        x=train_data.x_valid,
        y=train_data.y_valid,
        batch_size=cfg.batch_size,
        shuffle=True,
        active_seed=active_seed,
        num_workers=cfg.num_workers,
    )

    val_loader = make_mlp_dataloader(
        x=val_data.x_valid,
        y=val_data.y_valid,
        batch_size=cfg.batch_size,
        shuffle=False,
        active_seed=active_seed,
        num_workers=cfg.num_workers,
    )

    pos_weight = _make_pos_weight(train_data.y_valid, cfg, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    model_path = _project_path(config, cfg.output_model_path)
    history_csv = _project_path(config, cfg.output_history_csv)
    summary_json = _project_path(config, cfg.output_summary_json)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[MLPTrainingEpochResult] = []

    print("=" * 100)
    print("STEP 15 BASELINE TRAINING: MLP-xi")
    print("=" * 100)
    print(f"Device              : {device}")
    print(f"Train rows/valid    : {len(train_data.y_all)} / {len(train_data.y_valid)}")
    print(f"Val rows/valid      : {len(val_data.y_all)} / {len(val_data.y_valid)}")
    print(f"Feature dim         : {len(feature_columns)}")
    print(f"Hidden dims         : {cfg.hidden_dims}")
    print(f"Dropout             : {cfg.dropout}")
    print(f"Max epochs/patience : {cfg.max_epochs} / {cfg.patience}")
    print("Uses same xi_t features only. No raw shortcut columns.")
    print("=" * 100)

    for epoch in range(1, int(cfg.max_epochs) + 1):
        epoch_start = time.perf_counter()

        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=cfg.gradient_clip_norm,
        )

        val_loss, val_probs, val_labels = _evaluate_mlp_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_auprc = compute_binary_auprc(val_labels, val_probs)
        val_accuracy = compute_accuracy(val_labels, val_probs, threshold=0.5)

        epoch_result = MLPTrainingEpochResult(
            epoch=int(epoch),
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            val_auprc=_safe_float(val_auprc),
            val_accuracy=float(val_accuracy),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            runtime_seconds=float(time.perf_counter() - epoch_start),
        )
        history.append(epoch_result)

        improved = val_loss < (best_val_loss - float(cfg.min_delta))

        if improved:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            epochs_without_improvement = 0

            save_mlp_checkpoint(
                model=model,
                model_path=model_path,
                cfg=cfg,
                feature_columns=feature_columns,
                epoch=epoch,
                val_loss=val_loss,
                active_seed=active_seed,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_auprc={val_auprc} | "
            f"val_acc={val_accuracy:.6f} | "
            f"best_epoch={best_epoch}"
        )

        if epochs_without_improvement >= int(cfg.patience):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if not model_path.exists():
        save_mlp_checkpoint(
            model=model,
            model_path=model_path,
            cfg=cfg,
            feature_columns=feature_columns,
            epoch=int(history[-1].epoch if history else 0),
            val_loss=float(history[-1].val_loss if history else float("nan")),
            active_seed=active_seed,
        )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ensure_dir(history_csv.parent)
    pd.DataFrame([item.to_dict() for item in history]).to_csv(history_csv, index=False)

    fit_runtime = float(time.perf_counter() - start_time)

    artifact = MLPBaselineArtifact(
        model_name=cfg.model_name,
        model=model,
        config=cfg,
        model_path=model_path,
        feature_columns=feature_columns,
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val_loss),
        train_summary=train_data.summary(),
        val_summary=val_data.summary(),
        history=history,
        fit_runtime_seconds=fit_runtime,
        active_seed=int(active_seed),
        device=str(device),
    )

    save_json_safe(artifact.summary(), summary_json)

    print("MLP-xi training completed.")
    print(f"Model path          : {model_path}")
    print(f"History CSV         : {history_csv}")
    print(f"Summary JSON        : {summary_json}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best val loss       : {best_val_loss}")
    print(f"Runtime seconds     : {fit_runtime:.3f}")
    print("=" * 100)

    return artifact


def load_mlp_baseline(
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
) -> MLPBaselineArtifact:
    """Load saved MLP baseline checkpoint."""
    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    cfg = build_mlp_baseline_config(config=config, active_seed=active_seed)
    feature_columns = get_baseline_feature_columns(config)
    model_path = _project_path(config, cfg.output_model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Saved MLP baseline not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    feature_columns = list(checkpoint.get("feature_columns", feature_columns))

    model = MLPXiClassifier(
        input_dim=len(feature_columns),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        activation=cfg.activation,
        use_layer_norm=cfg.use_layer_norm,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return MLPBaselineArtifact(
        model_name=cfg.model_name,
        model=model,
        config=cfg,
        model_path=model_path,
        feature_columns=feature_columns,
        best_epoch=int(checkpoint.get("epoch", -1)),
        best_val_loss=float(checkpoint.get("val_loss", float("nan"))),
        train_summary={},
        val_summary={},
        history=[],
        fit_runtime_seconds=0.0,
        active_seed=int(active_seed),
        device=str(device),
    )


@torch.no_grad()
def collect_mlp_predictions(
    config: Mapping[str, Any],
    artifact: MLPBaselineArtifact,
    split_name: str,
    device: Optional[torch.device] = None,
) -> EvaluationPredictionBundle:
    """Collect MLP predictions for one split."""
    if device is None:
        device = next(artifact.model.parameters()).device

    split_data = load_baseline_split_data(
        config=config,
        split_name=split_name,
        feature_columns=artifact.feature_columns,
    )

    model = artifact.model.to(device)
    model.eval()

    x = torch.tensor(split_data.x_all, dtype=torch.float32, device=device)

    logits_list: List[np.ndarray] = []
    probabilities_list: List[np.ndarray] = []

    batch_size = int(artifact.config.batch_size)

    for start in range(0, len(split_data.x_all), batch_size):
        end = min(start + batch_size, len(split_data.x_all))
        logits = model(x[start:end])
        probabilities = torch.sigmoid(logits)

        logits_list.append(logits.detach().cpu().numpy().reshape(-1))
        probabilities_list.append(probabilities.detach().cpu().numpy().reshape(-1))

    if probabilities_list:
        probabilities_np = np.concatenate(probabilities_list).astype(np.float32)
        logits_np = np.concatenate(logits_list).astype(np.float32)
    else:
        probabilities_np = np.asarray([], dtype=np.float32)
        logits_np = np.asarray([], dtype=np.float32)

    return make_prediction_bundle_from_probabilities(
        split_data=split_data,
        probabilities=probabilities_np,
        logits=logits_np,
        checkpoint_path=str(artifact.model_path),
        model_name=artifact.model_name,
    )


__all__ = [
    "MLPBaselineConfig",
    "MLPTrainingEpochResult",
    "MLPBaselineArtifact",
    "MLPXiClassifier",
    "build_mlp_baseline_config",
    "make_mlp_dataloader",
    "compute_binary_auprc",
    "compute_accuracy",
    "save_mlp_checkpoint",
    "train_mlp_baseline",
    "load_mlp_baseline",
    "collect_mlp_predictions",
]